"""Tests for thesis tagging + attribution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from halal_trader.core.llm.base import BaseLLM, CallUsage
from halal_trader.core.thesis import (
    THESIS_TAGS,
    AttributionRow,
    TagVerdict,
    TaggedTradeContext,
    ThesisTagStore,
    attribute_pnl_by_thesis,
    deprecated_thesis_kill_list,
    heuristic_tag,
    llm_tag,
    render_attribution,
)


def _ctx(
    trade_id: str = "t1",
    pnl: float = 0.01,
    *,
    setup_type: str | None = None,
    reasoning: str = "",
    indicators: dict[str, float] | None = None,
    hold_seconds: int = 1800,
    news_blob: str = "",
) -> TaggedTradeContext:
    return TaggedTradeContext(
        trade_id=trade_id,
        symbol="BTCUSDT",
        side="buy",
        entry_price=100.0,
        exit_price=100.0 * (1.0 + pnl),
        exit_reason="take_profit",
        pnl_pct=pnl,
        hold_seconds=hold_seconds,
        setup_type=setup_type,
        indicators=indicators or {},
        reasoning=reasoning,
        news_blob=news_blob,
    )


# ── Heuristic ─────────────────────────────────────────────────────


def test_heuristic_news_blob_overrides_other_signals() -> None:
    ctx = _ctx(reasoning="rsi=85 momentum", news_blob="big 8-K filing")
    assert heuristic_tag(ctx) == "news_react"


def test_heuristic_setup_type_breakout() -> None:
    ctx = _ctx(setup_type="breakout")
    assert heuristic_tag(ctx) == "breakout"


def test_heuristic_setup_type_mean_revert() -> None:
    ctx = _ctx(setup_type="mean_reversion")
    assert heuristic_tag(ctx) == "mean_revert"


def test_heuristic_indicators_trend_follow() -> None:
    ctx = _ctx(indicators={"rsi_14": 70, "macd_histogram": 0.5})
    assert heuristic_tag(ctx) == "trend_follow"


def test_heuristic_indicators_mean_revert() -> None:
    ctx = _ctx(indicators={"rsi_14": 25, "macd_histogram": -0.3, "bb_position": 0.05})
    assert heuristic_tag(ctx) == "mean_revert"


def test_heuristic_short_hold_scalp() -> None:
    ctx = _ctx(hold_seconds=120)
    assert heuristic_tag(ctx) == "scalp"


def test_heuristic_falls_through_to_unknown() -> None:
    ctx = _ctx(hold_seconds=3000, indicators={"rsi_14": 50, "macd_histogram": 0.0})
    assert heuristic_tag(ctx) == "unknown"


# ── LLM tagger ────────────────────────────────────────────────────


class _ScriptedLLM(BaseLLM):
    def __init__(self, response: dict[str, Any] | Exception) -> None:
        super().__init__(model="thesis-stub")
        self._response = response
        self.calls = 0
        self.last_usage = CallUsage(model="thesis-stub")

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        if isinstance(self._response, Exception):
            raise self._response
        return json.dumps(self._response)


@pytest.mark.asyncio
async def test_llm_tag_basic() -> None:
    llm = _ScriptedLLM({"tag": "breakout", "confidence": 0.8, "reason": "20d high"})
    ctx = _ctx()
    v = await llm_tag(llm, ctx)
    assert isinstance(v, TagVerdict)
    assert v.tag == "breakout"
    assert v.confidence == 0.8
    assert "20d high" in v.reason


@pytest.mark.asyncio
async def test_llm_tag_unknown_tag_falls_to_unknown() -> None:
    llm = _ScriptedLLM({"tag": "made_up", "confidence": 0.9, "reason": "x"})
    ctx = _ctx()
    v = await llm_tag(llm, ctx)
    assert v.tag == "unknown"


@pytest.mark.asyncio
async def test_llm_tag_failure_falls_back_to_heuristic() -> None:
    llm = _ScriptedLLM(RuntimeError("api down"))
    ctx = _ctx(setup_type="breakout")
    v = await llm_tag(llm, ctx)
    assert v.tag == "breakout"  # heuristic match
    assert v.confidence < 0.5


# ── Store ─────────────────────────────────────────────────────────


def test_store_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "tags.json"
    s = ThesisTagStore(path=p)
    s.set("trade-1", "breakout", confidence=0.8, reason="x", method="llm")
    assert s.get("trade-1") == "breakout"


def test_store_unknown_tag_coerced_to_unknown(tmp_path: Path) -> None:
    p = tmp_path / "tags.json"
    s = ThesisTagStore(path=p)
    s.set("t", "magic_tag")
    assert s.get("t") == "unknown"


def test_store_all_returns_lookup(tmp_path: Path) -> None:
    p = tmp_path / "tags.json"
    s = ThesisTagStore(path=p)
    s.set("a", "breakout")
    s.set("b", "trend_follow")
    assert s.all() == {"a": "breakout", "b": "trend_follow"}


def test_store_resilient_to_corrupt_file(tmp_path: Path) -> None:
    p = tmp_path / "tags.json"
    p.write_text("{not json")
    s = ThesisTagStore(path=p)
    assert s.get("any") is None
    s.set("trade-1", "scalp")
    assert s.get("trade-1") == "scalp"


def test_store_creates_parent_dir(tmp_path: Path) -> None:
    p = tmp_path / "deep" / "nested" / "tags.json"
    s = ThesisTagStore(path=p)
    s.set("t", "scalp")
    assert p.exists()


# ── Attribution ───────────────────────────────────────────────────


def test_attribute_groups_by_tag_lookup() -> None:
    trades = [
        _ctx(trade_id="t1", pnl=0.01),
        _ctx(trade_id="t2", pnl=-0.005),
        _ctx(trade_id="t3", pnl=0.02),
    ]
    lookup = {"t1": "breakout", "t2": "breakout", "t3": "trend_follow"}
    rows = attribute_pnl_by_thesis(trades, lookup)
    assert rows["breakout"].n_trades == 2
    assert rows["breakout"].wins == 1
    assert rows["breakout"].losses == 1
    assert rows["trend_follow"].n_trades == 1
    assert rows["trend_follow"].wins == 1


def test_attribute_falls_back_to_heuristic() -> None:
    trades = [
        _ctx(trade_id="t1", setup_type="breakout"),
        _ctx(trade_id="t2", setup_type="momentum"),
    ]
    rows = attribute_pnl_by_thesis(trades)  # no lookup
    assert rows["breakout"].n_trades == 1
    assert rows["trend_follow"].n_trades == 1


def test_attribute_excludes_empty_tags() -> None:
    trades = [_ctx(trade_id="t1", setup_type="breakout")]
    rows = attribute_pnl_by_thesis(trades)
    assert "breakout" in rows
    assert "scalp" not in rows  # nothing landed there


def test_attribution_row_metrics() -> None:
    row = AttributionRow(tag="x", n_trades=5, wins=3, losses=2, sum_pnl_pct=0.10)
    assert row.win_rate == 0.6
    assert row.avg_pnl_pct == pytest.approx(0.02)


def test_kill_list_gates_on_min_trades_and_pnl() -> None:
    rows = {
        "breakout": AttributionRow(tag="breakout", n_trades=50, wins=10, losses=40, sum_pnl_pct=-0.5),
        "trend_follow": AttributionRow(tag="trend_follow", n_trades=20, wins=2, losses=18, sum_pnl_pct=-0.3),
        "scalp": AttributionRow(tag="scalp", n_trades=50, wins=30, losses=20, sum_pnl_pct=0.2),
    }
    kills = deprecated_thesis_kill_list(rows, min_trades=30)
    assert kills == ["breakout"]


def test_render_attribution_smoke() -> None:
    rows = [
        AttributionRow(tag="breakout", n_trades=10, wins=6, losses=4, sum_pnl_pct=0.05),
        AttributionRow(tag="scalp", n_trades=20, wins=11, losses=9, sum_pnl_pct=0.02),
    ]
    out = render_attribution(rows)
    assert "breakout" in out
    assert "scalp" in out


def test_thesis_tags_constant_set() -> None:
    assert "unknown" in THESIS_TAGS
    assert len(set(THESIS_TAGS)) == len(THESIS_TAGS)
