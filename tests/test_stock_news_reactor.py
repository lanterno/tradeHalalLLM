"""StockNewsEventReactor — the architectural foundation for the
"fast in, slow out" momentum-entry pipeline (operator memory:
strategy-fast-in-slow-out).

Tests focus on the dedup + classify + threshold logic since the
network layer is identical to the existing Finnhub news collector
and the supervised lifecycle is identical to the crypto reactor —
both covered elsewhere.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from halal_trader.sentiment.stocks_events import (
    GPTHeadlineClassifier,
    HeadlineClassification,
    StockNewsEvent,
    StockNewsEventReactor,
)


class _FakeClassifier:
    """Deterministic classifier — returns fixed scores by headline."""

    def __init__(self, scores: dict[str, float]):
        # Keys are substrings; first match wins. Default: 0.0
        self._scores = scores
        self.calls: list[tuple[str, str]] = []

    async def classify(
        self, *, symbol: str, headline: str, summary: str = ""
    ) -> HeadlineClassification:
        self.calls.append((symbol, headline))
        for needle, score in self._scores.items():
            if needle in headline:
                return HeadlineClassification(
                    score=score, tag="other", rationale="(fake)"
                )
        return HeadlineClassification(score=0.0)


def _reactor(
    classifier,
    *,
    symbols=("AAPL", "MSFT"),
    api_key: str = "test-key",
    score_threshold: float = 0.7,
):
    return StockNewsEventReactor(
        api_key=api_key,
        symbols=list(symbols),
        classifier=classifier,
        score_threshold=score_threshold,
        per_symbol_request_spacing_s=0,
    )


# ── enabled / disabled ──────────────────────────────────────────


def test_reactor_disabled_without_api_key():
    r = _reactor(_FakeClassifier({}), api_key="")
    assert r.enabled is False


def test_reactor_disabled_without_symbols():
    r = _reactor(_FakeClassifier({}), symbols=())
    assert r.enabled is False


def test_reactor_enabled_when_both_present():
    r = _reactor(_FakeClassifier({}))
    assert r.enabled is True


# ── _maybe_emit: dedup + threshold ──────────────────────────────


@pytest.mark.asyncio
async def test_emit_skips_items_without_url():
    r = _reactor(_FakeClassifier({"good": 0.9}))
    out = await r._maybe_emit("AAPL", {"headline": "good news"})
    assert out is None


@pytest.mark.asyncio
async def test_emit_skips_items_without_headline():
    r = _reactor(_FakeClassifier({"good": 0.9}))
    out = await r._maybe_emit("AAPL", {"url": "https://x.com/1"})
    assert out is None


@pytest.mark.asyncio
async def test_emit_dedups_by_symbol_url_pair():
    r = _reactor(_FakeClassifier({"good": 0.9}))
    item = {"url": "https://x.com/1", "headline": "good news"}
    first = await r._maybe_emit("AAPL", item)
    second = await r._maybe_emit("AAPL", item)
    assert first is not None
    assert second is None  # already seen


@pytest.mark.asyncio
async def test_same_url_different_symbol_emits_twice():
    """A shared news URL for two tickers is two events (e.g. a sector
    headline that mentions multiple symbols). Each per-symbol emit is
    independent so dedup is per (symbol, url) pair."""
    r = _reactor(_FakeClassifier({"good": 0.9}))
    item = {"url": "https://x.com/sector", "headline": "good news for tech"}
    aapl = await r._maybe_emit("AAPL", item)
    msft = await r._maybe_emit("MSFT", item)
    assert aapl is not None
    assert msft is not None


@pytest.mark.asyncio
async def test_emit_skips_below_threshold():
    r = _reactor(_FakeClassifier({"meh": 0.4}), score_threshold=0.7)
    out = await r._maybe_emit("AAPL", {"url": "u1", "headline": "meh news"})
    assert out is None


@pytest.mark.asyncio
async def test_emit_at_threshold_passes():
    """Score == threshold should pass (>= test, not >)."""
    r = _reactor(_FakeClassifier({"on edge": 0.7}), score_threshold=0.7)
    out = await r._maybe_emit("AAPL", {"url": "u1", "headline": "on edge"})
    assert isinstance(out, StockNewsEvent)
    assert out.classification.score == 0.7


@pytest.mark.asyncio
async def test_emit_classifier_exception_returns_none():
    """A misbehaving classifier shouldn't crash the reactor."""

    class _BoomClassifier:
        async def classify(self, *, symbol, headline, summary=""):
            raise RuntimeError("boom")

    r = _reactor(_BoomClassifier())
    out = await r._maybe_emit("AAPL", {"url": "u1", "headline": "x"})
    assert out is None


# ── on_event ────────────────────────────────────────────────────


def test_on_event_registers_callback():
    r = _reactor(_FakeClassifier({}))
    cb = AsyncMock()
    r.on_event(cb)
    assert cb in r._callbacks


# ── GPTHeadlineClassifier ───────────────────────────────────────


@pytest.mark.asyncio
async def test_gpt_classifier_clamps_score_to_unit_interval():
    """LLM occasionally returns scores >1 or <0 — clamp defensively."""

    class _LLM:
        async def generate_json(self, prompt, system=None):
            return {"score": 1.7, "tag": "other"}

    c = GPTHeadlineClassifier(_LLM())
    out = await c.classify(symbol="AAPL", headline="x")
    assert out.score == 1.0


@pytest.mark.asyncio
async def test_gpt_classifier_handles_missing_fields():
    class _LLM:
        async def generate_json(self, prompt, system=None):
            return {}

    c = GPTHeadlineClassifier(_LLM())
    out = await c.classify(symbol="AAPL", headline="x")
    assert out.score == 0.0
    assert out.tag == "other"


@pytest.mark.asyncio
async def test_gpt_classifier_handles_llm_exception_gracefully():
    class _LLM:
        async def generate_json(self, prompt, system=None):
            raise RuntimeError("rate limit")

    c = GPTHeadlineClassifier(_LLM())
    out = await c.classify(symbol="AAPL", headline="x")
    assert out.score == 0.0


@pytest.mark.asyncio
async def test_gpt_classifier_truncates_rationale():
    class _LLM:
        async def generate_json(self, prompt, system=None):
            return {"score": 0.8, "tag": "earnings", "rationale": "x" * 500}

    c = GPTHeadlineClassifier(_LLM())
    out = await c.classify(symbol="AAPL", headline="x")
    assert len(out.rationale) <= 200


# ── entry_type column ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_trade_repo_records_entry_type(engine):
    """The Alembic migration + repo signature add `entry_type` so
    reactor-driven trades can be tagged for the LLM lockout (memory:
    strategy-fast-in-slow-out)."""
    from halal_trader.db.repository import Repository

    repo = Repository(engine)
    trade_id = await repo.record_trade(
        symbol="AAPL",
        side="buy",
        quantity=10,
        status="filled",
        filled_quantity=10,
        entry_type="reactor_momentum",
    )
    assert trade_id > 0

    from sqlmodel.ext.asyncio.session import AsyncSession

    from halal_trader.db.models import Trade

    async with AsyncSession(engine) as session:
        row = await session.get(Trade, trade_id)
        assert row is not None
        assert row.entry_type == "reactor_momentum"


@pytest.mark.asyncio
async def test_trade_repo_default_entry_type_is_none(engine):
    """Backward-compat: existing call sites that don't pass entry_type
    get None — strategy/monitor must treat None as "scheduled"."""
    from halal_trader.db.repository import Repository

    repo = Repository(engine)
    trade_id = await repo.record_trade(
        symbol="MSFT", side="buy", quantity=5, status="filled", filled_quantity=5
    )
    from sqlmodel.ext.asyncio.session import AsyncSession

    from halal_trader.db.models import Trade

    async with AsyncSession(engine) as session:
        row = await session.get(Trade, trade_id)
        assert row is not None
        assert row.entry_type is None
