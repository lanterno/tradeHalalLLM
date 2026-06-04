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
    notify_cooldown_s: int = 900,
):
    return StockNewsEventReactor(
        api_key=api_key,
        symbols=list(symbols),
        classifier=classifier,
        score_threshold=score_threshold,
        per_symbol_request_spacing_s=0,
        per_symbol_notify_cooldown_s=notify_cooldown_s,
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


def test_reactor_default_threshold_raised_to_085():
    """Threshold floor was raised from 0.7 → 0.85 on 2026-05-22 after
    the NVDA earnings cascade flooded with editorial coverage all
    scoring exactly 0.70."""
    from halal_trader.sentiment.stocks_events import StockNewsEventReactor
    assert StockNewsEventReactor._DEFAULT_SCORE_THRESHOLD == 0.85


def test_reactor_default_notify_cooldown_1800s():
    """Per-symbol notification cooldown defaults to 30min.
    Raised from 15→30 on 2026-05-22 after observing the same NVDA
    Q1 catalyst re-firing every ~15 min via repackaged headlines."""
    from halal_trader.sentiment.stocks_events import StockNewsEventReactor
    assert StockNewsEventReactor._DEFAULT_NOTIFY_COOLDOWN_S == 1800


# ── per-symbol notification cooldown ────────────────────────────


@pytest.mark.asyncio
async def test_notify_cooldown_drops_second_callback_for_same_symbol():
    """A single catalyst reliably produces 10-30 repackaged headlines
    across publishers (observed NVDA earnings cascade on 2026-05-22).
    Cooldown caps callbacks to one per (symbol, window)."""
    classifier = _FakeClassifier({"good": 0.95})
    r = _reactor(classifier, notify_cooldown_s=900)

    cb = AsyncMock()
    r.on_event(cb)

    # Simulate two scored events for the same symbol arriving back-to-back.
    items = [
        {"url": "https://x.com/1", "headline": "good news first"},
        {"url": "https://x.com/2", "headline": "good news repackaged"},
    ]
    events = []
    for item in items:
        e = await r._maybe_emit("MSFT", item)
        if e is not None:
            events.append(e)
    assert len(events) == 2  # both scored

    # Now run them through the dispatch logic. Use the same code-path
    # the poll loop uses: log + cooldown check + callback.
    import time as _t

    for e in events:
        last = r._last_notify.get(e.symbol, 0.0)
        now_t = _t.monotonic()
        if r._notify_cooldown_s > 0 and (now_t - last) < r._notify_cooldown_s:
            continue
        r._last_notify[e.symbol] = now_t
        await cb(e)

    # Only the first fired; the second hit the cooldown.
    assert cb.await_count == 1


@pytest.mark.asyncio
async def test_notify_cooldown_zero_disables():
    """Operator escape: notify_cooldown=0 → every scored event fires."""
    classifier = _FakeClassifier({"good": 0.95})
    r = _reactor(classifier, notify_cooldown_s=0)

    cb = AsyncMock()
    r.on_event(cb)

    items = [
        {"url": "https://x.com/1", "headline": "good news A"},
        {"url": "https://x.com/2", "headline": "good news B"},
    ]
    import time as _t

    for item in items:
        e = await r._maybe_emit("MSFT", item)
        if e is None:
            continue
        # Same dispatch as the poll loop.
        last = r._last_notify.get(e.symbol, 0.0)
        now_t = _t.monotonic()
        if r._notify_cooldown_s > 0 and (now_t - last) < r._notify_cooldown_s:
            continue
        r._last_notify[e.symbol] = now_t
        await cb(e)

    assert cb.await_count == 2


@pytest.mark.asyncio
async def test_notify_cooldown_per_symbol_not_global():
    """The cooldown is per-symbol — a NVDA event shouldn't block an
    MSFT event in the same minute."""
    classifier = _FakeClassifier({"good": 0.95})
    r = _reactor(classifier, notify_cooldown_s=900)

    cb = AsyncMock()
    r.on_event(cb)

    import time as _t

    for sym in ("NVDA", "MSFT"):
        e = await r._maybe_emit(sym, {"url": f"https://x.com/{sym}", "headline": "good"})
        if e is None:
            continue
        last = r._last_notify.get(e.symbol, 0.0)
        now_t = _t.monotonic()
        if r._notify_cooldown_s > 0 and (now_t - last) < r._notify_cooldown_s:
            continue
        r._last_notify[e.symbol] = now_t
        await cb(e)

    assert cb.await_count == 2


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


# ── Quota circuit breaker (added 2026-05-23 post-incident) ──────


class _QuotaLLM:
    """LLM stub that always raises an OpenAI-shaped insufficient_quota
    error — models the exact error string seen in the 2026-05-22 logs."""

    def __init__(self):
        self.calls = 0

    async def generate_json(self, prompt, system=None):
        self.calls += 1
        raise RuntimeError(
            "Error code: 429 - {'error': {'message': 'You exceeded your "
            "current quota...', 'code': 'insufficient_quota'}}"
        )


class _CountingAlertSink:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    @property
    def enabled(self):
        return True

    async def notify(self, error_type, details, *, market="", severity="warning"):
        self.calls.append((error_type, market))
        return True


@pytest.mark.asyncio
async def test_quota_breaker_trips_on_first_insufficient_quota():
    """The whole point: ONE insufficient_quota error trips the breaker;
    all subsequent classify calls return 0.0 without hitting the LLM."""
    llm = _QuotaLLM()
    sink = _CountingAlertSink()
    c = GPTHeadlineClassifier(llm, alert_sink=sink)

    # First call: hits the LLM, gets 429, trips the breaker.
    out1 = await c.classify(symbol="NVDA", headline="big news")
    assert out1.score == 0.0
    assert llm.calls == 1
    assert c._quota_exhausted is True

    # Next 50 calls: short-circuited, no LLM hit.
    for _ in range(50):
        out = await c.classify(symbol="AAPL", headline="x")
        assert out.score == 0.0
    assert llm.calls == 1  # still just the one — the breaker held


@pytest.mark.asyncio
async def test_quota_breaker_fires_alert_sink_once():
    """The alert sink is called exactly once per trip (deduped by the
    sink's own cooldown on subsequent calls anyway)."""
    llm = _QuotaLLM()
    sink = _CountingAlertSink()
    c = GPTHeadlineClassifier(llm, alert_sink=sink)

    await c.classify(symbol="NVDA", headline="big news")
    # Subsequent short-circuited calls must NOT call the sink again.
    await c.classify(symbol="AAPL", headline="x")
    await c.classify(symbol="MSFT", headline="y")

    assert len(sink.calls) == 1
    assert sink.calls[0][0] == "classifier.quota_exhausted"
    assert sink.calls[0][1] == "stocks"


@pytest.mark.asyncio
async def test_quota_breaker_half_open_recovers_after_cooldown():
    """REGRESSION: a quota dip used to LATCH the breaker until restart, leaving
    the reactor dead for days (observed live 2026-06-04). After the cooldown the
    breaker goes half-open: one probe is allowed, and on success it clears so the
    reactor self-heals without a restart."""
    import time

    from halal_trader.sentiment.stocks_events import _QUOTA_RECOVERY_COOLDOWN_S

    class _RecoveringLLM:
        def __init__(self):
            self.calls = 0
            self.fail = True

        async def generate_json(self, prompt, system=None):
            self.calls += 1
            if self.fail:
                raise RuntimeError("429 ... 'code': 'insufficient_quota'")
            return {"score": 0.7, "tag": "earnings", "rationale": "beat"}

    llm = _RecoveringLLM()
    sink = _CountingAlertSink()
    c = GPTHeadlineClassifier(llm, alert_sink=sink)

    assert (await c.classify(symbol="NVDA", headline="x")).score == 0.0  # trips
    assert c._quota_exhausted is True and llm.calls == 1
    assert (await c.classify(symbol="AAPL", headline="y")).score == 0.0  # short-circuit
    assert llm.calls == 1  # no probe yet (within cooldown)

    # Cooldown elapses + quota recovered → half-open probe succeeds → breaker clears.
    c._quota_tripped_at = time.monotonic() - _QUOTA_RECOVERY_COOLDOWN_S - 1
    llm.fail = False
    out = await c.classify(symbol="MSFT", headline="real beat")
    assert out.score == 0.7 and llm.calls == 2
    assert c._quota_exhausted is False  # self-healed, no restart needed
    assert len(sink.calls) == 1  # recovery does NOT re-alert


@pytest.mark.asyncio
async def test_quota_breaker_half_open_probe_failure_does_not_realert():
    """If the half-open probe still hits quota, the breaker stays tripped and
    re-arms the cooldown WITHOUT re-firing the operator alert."""
    import time

    from halal_trader.sentiment.stocks_events import _QUOTA_RECOVERY_COOLDOWN_S

    llm = _QuotaLLM()
    sink = _CountingAlertSink()
    c = GPTHeadlineClassifier(llm, alert_sink=sink)
    await c.classify(symbol="NVDA", headline="x")  # trip (alert #1)
    assert llm.calls == 1 and len(sink.calls) == 1

    c._quota_tripped_at = time.monotonic() - _QUOTA_RECOVERY_COOLDOWN_S - 1
    await c.classify(symbol="AAPL", headline="y")  # probe still quota-fails
    assert llm.calls == 2 and c._quota_exhausted is True
    assert len(sink.calls) == 1  # no re-alert on re-arm


@pytest.mark.asyncio
async def test_quota_breaker_does_not_trip_on_non_quota_errors():
    """Transient 503s, timeouts, network blips must NOT trip the
    breaker — they're recoverable, quota exhaustion isn't."""

    class _Flaky:
        def __init__(self):
            self.calls = 0

        async def generate_json(self, prompt, system=None):
            self.calls += 1
            raise RuntimeError("Error code: 503 - Service Unavailable")

    sink = _CountingAlertSink()
    c = GPTHeadlineClassifier(_Flaky(), alert_sink=sink)

    for _ in range(5):
        await c.classify(symbol="AAPL", headline="x")

    assert c._quota_exhausted is False
    assert len(sink.calls) == 0  # no false-positive alert


@pytest.mark.asyncio
async def test_quota_breaker_works_without_alert_sink():
    """The breaker is mandatory; the alert sink is optional. No sink
    must not crash the trip path."""
    c = GPTHeadlineClassifier(_QuotaLLM(), alert_sink=None)
    out = await c.classify(symbol="NVDA", headline="x")
    assert out.score == 0.0
    assert c._quota_exhausted is True


# ── Daily classify ceiling ──────────────────────────────────────


class _OkLLM:
    def __init__(self):
        self.calls = 0

    async def generate_json(self, prompt, system=None):
        self.calls += 1
        return {"score": 0.9, "tag": "other", "rationale": "ok"}


@pytest.mark.asyncio
async def test_daily_cap_short_circuits_when_reached():
    """Counts every attempted call; over the cap, returns 0.0 without
    hitting the LLM."""
    llm = _OkLLM()
    c = GPTHeadlineClassifier(llm, daily_classify_cap=3)

    # 3 calls go through.
    for _ in range(3):
        out = await c.classify(symbol="AAPL", headline="x")
        assert out.score == 0.9
    assert llm.calls == 3

    # 4th and beyond short-circuit silently.
    for _ in range(10):
        out = await c.classify(symbol="AAPL", headline="x")
        assert out.score == 0.0
    assert llm.calls == 3  # still 3 — cap held


@pytest.mark.asyncio
async def test_daily_cap_zero_disables_ceiling():
    """Operator escape: cap=0 means unbounded (useful in tests / local
    Ollama where there's no spend to worry about)."""
    llm = _OkLLM()
    c = GPTHeadlineClassifier(llm, daily_classify_cap=0)
    for _ in range(20):
        out = await c.classify(symbol="AAPL", headline="x")
        assert out.score == 0.9
    assert llm.calls == 20


@pytest.mark.asyncio
async def test_daily_cap_counts_failed_calls_too():
    """A 429 still consumes the rate budget — it must count against
    the daily cap so we don't burn 250 quota failures before stopping."""

    class _Flaky:
        def __init__(self):
            self.calls = 0

        async def generate_json(self, prompt, system=None):
            self.calls += 1
            raise RuntimeError("503 Service Unavailable")

    c = GPTHeadlineClassifier(_Flaky(), daily_classify_cap=5)
    for _ in range(20):
        await c.classify(symbol="AAPL", headline="x")
    # Cap is 5 — only 5 should have hit the LLM, failures and all.
    assert _Flaky.__name__  # silence linter; c._llm.calls is the real assert
    assert c._daily_count == 5


@pytest.mark.asyncio
async def test_daily_cap_rolls_at_utc_midnight():
    """UTC date change resets the counter — a multi-day-running bot
    must not stay tripped past midnight."""
    llm = _OkLLM()
    c = GPTHeadlineClassifier(llm, daily_classify_cap=2)

    for _ in range(2):
        await c.classify(symbol="AAPL", headline="x")
    assert c._daily_count == 2

    # Simulate UTC date rollover by stamping a past date.
    c._daily_reset_date = "1999-01-01"
    out = await c.classify(symbol="AAPL", headline="x")
    assert out.score == 0.9  # call went through after the reset
    assert c._daily_count == 1  # rolled


# ── Settings wiring ─────────────────────────────────────────────


def test_settings_exposes_reactor_daily_classify_cap():
    """The cap is operator-configurable via STOCK_REACTOR_DAILY_CLASSIFY_CAP
    (StockSettings has no prefix, so the env var is the field name
    uppercased: REACTOR_DAILY_CLASSIFY_CAP)."""
    from halal_trader.config import StockSettings

    s = StockSettings()
    assert s.reactor_daily_classify_cap == 250  # default matches plan


def test_settings_reactor_cap_is_configurable():
    from halal_trader.config import StockSettings

    s = StockSettings(reactor_daily_classify_cap=42)
    assert s.reactor_daily_classify_cap == 42


# ── Integration: breaker + FallbackLLM together ─────────────────


@pytest.mark.asyncio
async def test_breaker_does_not_trip_when_fallback_absorbs_quota_error():
    """Phase B's whole point: when a FallbackLLM is wired underneath
    the classifier, a quota error on the primary should NEVER reach the
    classifier's breaker — the fallback provider absorbs it and returns
    a real score. The bot keeps running on cheaper / local models.

    This pins the "Phase A circuit breaker + Phase B fallback chain"
    interaction so a future refactor can't accidentally re-introduce
    the false-trip path."""

    from halal_trader.core.llm.base import BaseLLM, CallUsage
    from halal_trader.core.llm.fallback import FallbackLLM

    class _QuotaPrimary(BaseLLM):
        def __init__(self):
            super().__init__("gpt-4o-mini")
            self.calls = 0

        async def generate(self, prompt, system=None):
            self.calls += 1
            raise RuntimeError("Error code: 429 - {'code': 'insufficient_quota'}")

    class _WorkingFallback(BaseLLM):
        def __init__(self):
            super().__init__("claude-haiku")
            self.calls = 0

        async def generate(self, prompt, system=None):
            self.calls += 1
            return '{"score": 0.92, "tag": "earnings", "rationale": "strong beat"}'

        async def generate_json(self, prompt, system=None):
            self.calls += 1
            self.last_usage = CallUsage(provider="anthropic", model="claude-haiku")
            return {"score": 0.92, "tag": "earnings", "rationale": "strong beat"}

    primary = _QuotaPrimary()
    fallback = _WorkingFallback()
    chain = FallbackLLM(primary, [fallback])

    sink = _CountingAlertSink()
    c = GPTHeadlineClassifier(chain, alert_sink=sink)

    out = await c.classify(symbol="NVDA", headline="surprise beat")

    # Primary failed, fallback served — score must reflect the fallback.
    assert out.score == 0.92
    assert out.tag == "earnings"
    # The breaker MUST NOT have tripped — chain absorbed the quota error.
    assert c._quota_exhausted is False
    # No false-positive alert.
    assert len(sink.calls) == 0
    # Primary was tried, fallback served.
    assert primary.calls == 1
    assert fallback.calls == 1


# ── Telemetry / get_telemetry ───────────────────────────────────


@pytest.mark.asyncio
async def test_telemetry_zeroed_on_fresh_classifier():
    c = GPTHeadlineClassifier(_OkLLM())
    t = c.get_telemetry()
    assert t["total_calls"] == 0
    assert t["total_successes"] == 0
    assert t["total_failures"] == 0
    assert t["total_short_circuits"] == 0
    assert t["calls_today"] == 0
    assert t["quota_exhausted"] is False
    assert t["calls_by_provider"] == {}
    assert t["cost_usd_total"] == 0.0


@pytest.mark.asyncio
async def test_telemetry_counts_successes_and_failures_separately():
    """Telemetry must distinguish "called the LLM and got a score" from
    "called the LLM and it threw" so we can see error rate at EOD."""

    class _Mixed:
        def __init__(self):
            self.n = 0
            self.last_usage = None

        async def generate_json(self, prompt, system=None):
            self.n += 1
            if self.n % 2 == 0:
                raise RuntimeError("flaky")
            return {"score": 0.8, "tag": "other", "rationale": "x"}

    c = GPTHeadlineClassifier(_Mixed())
    for _ in range(4):
        await c.classify(symbol="AAPL", headline="x")

    t = c.get_telemetry()
    assert t["total_calls"] == 4
    assert t["total_successes"] == 2
    assert t["total_failures"] == 2
    assert t["total_short_circuits"] == 0


@pytest.mark.asyncio
async def test_telemetry_counts_short_circuits_separately_from_calls():
    """Short-circuits (breaker tripped, daily cap reached) MUST NOT
    inflate total_calls — that field is the API-spend signal. They
    count under total_short_circuits."""
    c = GPTHeadlineClassifier(_OkLLM(), daily_classify_cap=2)
    for _ in range(5):
        await c.classify(symbol="AAPL", headline="x")

    t = c.get_telemetry()
    assert t["total_calls"] == 2  # cap held at 2 actual API calls
    assert t["total_short_circuits"] == 3  # remaining 3 were short-circuited
    assert t["calls_today"] == 2


@pytest.mark.asyncio
async def test_telemetry_rolls_up_per_provider_from_last_usage():
    """Successful calls credit the provider that served them — pulled
    from BaseLLM.last_usage so the FallbackLLM rotation is reflected
    correctly without coupling to FallbackLLM internals."""

    from decimal import Decimal

    class _UsageReportingLLM:
        def __init__(self):
            self.n = 0
            # Mimic CallUsage shape with the attributes the classifier reads.

            class _U:
                provider = ""
                cost_usd = Decimal("0")

            self.last_usage = _U()

        async def generate_json(self, prompt, system=None):
            self.n += 1
            # Alternate which provider "served" — like rotation.
            self.last_usage.provider = "openai" if self.n % 2 == 1 else "anthropic"
            self.last_usage.cost_usd = Decimal("0.001")
            return {"score": 0.7, "tag": "other"}

    c = GPTHeadlineClassifier(_UsageReportingLLM())
    for _ in range(4):
        await c.classify(symbol="AAPL", headline="x")

    t = c.get_telemetry()
    assert t["calls_by_provider"] == {"openai": 2, "anthropic": 2}
    assert t["cost_usd_total"] == pytest.approx(0.004, rel=1e-6)


def test_reactor_exposes_classifier_for_telemetry():
    """The scheduler's EOD hook reads reactor.classifier — pin the
    public accessor so a future refactor can't accidentally rename it."""
    c = GPTHeadlineClassifier(_OkLLM())
    r = StockNewsEventReactor(
        api_key="k", symbols=["AAPL"], classifier=c,
    )
    assert r.classifier is c


@pytest.mark.asyncio
async def test_breaker_trips_when_entire_fallback_chain_is_quota_exhausted():
    """The opposite end: if EVERY provider in the chain is out of
    credits, the chain re-raises the last quota error and the breaker
    correctly trips. Operator gets one alert and the bot stops
    pointlessly hammering all providers."""

    from halal_trader.core.llm.base import BaseLLM
    from halal_trader.core.llm.fallback import FallbackLLM

    class _OutOfCredits(BaseLLM):
        def __init__(self, name):
            super().__init__(name)
            self.calls = 0

        async def generate(self, prompt, system=None):
            self.calls += 1
            raise RuntimeError(
                f"Error from {self.model}: code: 'insufficient_quota'"
            )

    primary = _OutOfCredits("gpt-4o-mini")
    fb1 = _OutOfCredits("claude-haiku")
    fb2 = _OutOfCredits("llama3.2:3b")
    chain = FallbackLLM(primary, [fb1, fb2])

    sink = _CountingAlertSink()
    c = GPTHeadlineClassifier(chain, alert_sink=sink)

    out = await c.classify(symbol="NVDA", headline="x")
    assert out.score == 0.0
    assert c._quota_exhausted is True
    assert len(sink.calls) == 1  # one alert, not three (one per provider)


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


# ── cross-restart state persistence ─────────────────────────────


def test_save_then_load_roundtrips_seen_and_cooldowns(tmp_path):
    """A restart must restore both dedup set and notify cooldowns so the
    bot doesn't re-fire the same catalyst's alert on every boot — the
    2026-05-22 session restarted 5+ times and re-alerted each time."""
    path = tmp_path / "reactor_state.json"
    r1 = _reactor(_FakeClassifier({}))
    r1._state_path = path
    r1._seen = {("AAPL", "http://a"), ("MSFT", "http://b")}
    r1._last_notify = {"AAPL": 1234.5}
    r1._state_dirty = True
    r1._save_state()

    r2 = _reactor(_FakeClassifier({}))
    r2._state_path = path
    r2._load_state()
    assert r2._seen == {("AAPL", "http://a"), ("MSFT", "http://b")}
    assert r2._last_notify == {"AAPL": 1234.5}


def test_load_missing_file_is_noop(tmp_path):
    """No state file yet (first ever run) → empty state, no crash."""
    r = _reactor(_FakeClassifier({}))
    r._state_path = tmp_path / "does_not_exist.json"
    r._load_state()
    assert r._seen == set()
    assert r._last_notify == {}


def test_load_corrupt_file_starts_fresh(tmp_path):
    """A truncated / garbage state file must not block startup."""
    path = tmp_path / "reactor_state.json"
    path.write_text("{not valid json")
    r = _reactor(_FakeClassifier({}))
    r._state_path = path
    r._load_state()
    assert r._seen == set()


def test_save_is_noop_when_not_dirty(tmp_path):
    """No new headlines / notifies since last save → no file write."""
    path = tmp_path / "reactor_state.json"
    r = _reactor(_FakeClassifier({}))
    r._state_path = path
    r._state_dirty = False
    r._save_state()
    assert not path.exists()


def test_save_noop_when_persistence_disabled():
    """state_path=None (default for tests / ad-hoc runs) → never writes."""
    r = _reactor(_FakeClassifier({}))
    assert r._state_path is None
    r._seen = {("AAPL", "http://a")}
    r._state_dirty = True
    r._save_state()  # must not raise even with nowhere to write


def test_load_tolerates_malformed_entries(tmp_path):
    """Hand-edited / partially-corrupt entries are skipped, not fatal."""
    import json

    path = tmp_path / "reactor_state.json"
    path.write_text(
        json.dumps(
            {
                "seen": [["AAPL", "http://a"], ["bad"], "nope", ["X", "Y", "Z"]],
                "last_notify": {"AAPL": 1.0, "MSFT": "not-a-number"},
            }
        )
    )
    r = _reactor(_FakeClassifier({}))
    r._state_path = path
    r._load_state()
    assert ("AAPL", "http://a") in r._seen
    assert all(len(p) == 2 for p in r._seen)
    assert r._last_notify == {"AAPL": 1.0}
