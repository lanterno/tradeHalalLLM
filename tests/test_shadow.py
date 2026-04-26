"""Tests for shadow-bot divergence detector."""

from __future__ import annotations

from pathlib import Path

from halal_trader.core.shadow import (
    ShadowAlertConfig,
    ShadowLedger,
    aggregate_plan_diffs,
    diff_plans,
    divergence_metrics,
    render_status,
    shadow_alert_state,
)
from halal_trader.domain.models import (
    CryptoTradeDecision,
    CryptoTradingPlan,
    TradeAction,
)


# ── Ledger ────────────────────────────────────────────────────────


def test_record_and_size() -> None:
    led = ShadowLedger()
    led.record(cycle_id="c1", live_equity=1000, shadow_equity=1000)
    led.record(cycle_id="c2", live_equity=1010, shadow_equity=1005)
    assert led.size == 2


def test_capacity_trim() -> None:
    led = ShadowLedger(capacity=3)
    for i in range(5):
        led.record(cycle_id=f"c{i}", live_equity=1000, shadow_equity=1000)
    assert led.size == 3
    assert [e.cycle_id for e in led.entries] == ["c2", "c3", "c4"]


def test_save_load_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "shadow.json"
    led = ShadowLedger()
    led.record(cycle_id="c1", live_equity=1010, shadow_equity=1000)
    led.record(cycle_id="c2", live_equity=1020, shadow_equity=1010)
    led.save(p)
    back = ShadowLedger.load(p)
    assert back.size == 2
    assert back.entries[0].live_equity == 1010


# ── Metrics ───────────────────────────────────────────────────────


def test_metrics_too_few_samples_returns_none() -> None:
    led = ShadowLedger()
    led.record(cycle_id="c1", live_equity=100, shadow_equity=100)
    led.record(cycle_id="c2", live_equity=101, shadow_equity=100)
    assert divergence_metrics(led.entries) is None


def test_metrics_live_better_when_outperforming() -> None:
    led = ShadowLedger()
    for i in range(40):
        # live grows faster
        led.record(
            cycle_id=f"c{i}", live_equity=100 + i * 0.6, shadow_equity=100 + i * 0.3
        )
    m = divergence_metrics(led.entries)
    assert m is not None
    assert m.direction == "live_better"
    assert m.mean_diff_pct > 0


def test_metrics_live_worse_when_underperforming() -> None:
    led = ShadowLedger()
    for i in range(40):
        led.record(cycle_id=f"c{i}", live_equity=100 - i * 0.5, shadow_equity=100)
    m = divergence_metrics(led.entries)
    assert m is not None
    assert m.direction == "live_worse"
    assert m.mean_diff_pct < 0


def test_metrics_even_when_no_systematic_diff() -> None:
    led = ShadowLedger()
    # Deterministic alternating tiny diffs that average exactly to 0.
    for i in range(60):
        delta = 0.5 if i % 2 == 0 else -0.5
        led.record(cycle_id=f"c{i}", live_equity=100 + delta, shadow_equity=100)
    m = divergence_metrics(led.entries)
    assert m is not None
    assert m.direction == "even"
    assert abs(m.mean_diff_pct) < 1e-6


# ── Alerts ────────────────────────────────────────────────────────


def test_alert_ok_with_few_samples() -> None:
    led = ShadowLedger()
    for i in range(5):
        led.record(cycle_id=f"c{i}", live_equity=100, shadow_equity=100 + i)
    metrics = divergence_metrics(led.entries)
    assert shadow_alert_state(metrics) == "ok"


def test_alert_diverged_on_sustained_underperformance() -> None:
    led = ShadowLedger()
    for i in range(60):
        led.record(cycle_id=f"c{i}", live_equity=100 - i * 0.2, shadow_equity=100)
    metrics = divergence_metrics(led.entries)
    cfg = ShadowAlertConfig(watch_drawdown_pct=0.02, diverge_drawdown_pct=0.05)
    state = shadow_alert_state(metrics, cfg)
    assert state == "diverged"


def test_alert_watch_for_mid_drawdown() -> None:
    led = ShadowLedger()
    for i in range(60):
        led.record(cycle_id=f"c{i}", live_equity=100 - i * 0.05, shadow_equity=100)
    metrics = divergence_metrics(led.entries)
    cfg = ShadowAlertConfig(watch_drawdown_pct=0.02, diverge_drawdown_pct=0.10)
    state = shadow_alert_state(metrics, cfg)
    assert state in ("watch", "diverged")


def test_alert_diverged_on_single_catastrophic_day() -> None:
    led = ShadowLedger()
    # Quiet 30 days
    for i in range(30):
        led.record(cycle_id=f"c{i}", live_equity=100, shadow_equity=100)
    # Then a single big underperformance
    led.record(cycle_id="cbad", live_equity=92, shadow_equity=100)
    metrics = divergence_metrics(led.entries)
    state = shadow_alert_state(metrics)
    assert state == "diverged"


def test_render_status_smoke() -> None:
    led = ShadowLedger()
    for i in range(40):
        led.record(cycle_id=f"c{i}", live_equity=100, shadow_equity=100 - i * 0.1)
    m = divergence_metrics(led.entries)
    state = shadow_alert_state(m)
    s = render_status(m, state)
    assert "shadow status" in s


# ── Plan diffs ────────────────────────────────────────────────────


def _buy(symbol: str = "BTCUSDT") -> CryptoTradeDecision:
    return CryptoTradeDecision(
        action=TradeAction.BUY, symbol=symbol, quantity=1, confidence=0.5, reasoning="x"
    )


def _sell(symbol: str = "BTCUSDT") -> CryptoTradeDecision:
    return CryptoTradeDecision(
        action=TradeAction.SELL, symbol=symbol, quantity=1, confidence=0.5, reasoning="x"
    )


def test_diff_plans_identical() -> None:
    a = CryptoTradingPlan(decisions=[_buy("BTCUSDT"), _sell("ETHUSDT")])
    b = CryptoTradingPlan(decisions=[_buy("BTCUSDT"), _sell("ETHUSDT")])
    d = diff_plans(a, b)
    assert d == {"shared": 2, "only_live": 0, "only_shadow": 0}


def test_diff_plans_disagreement() -> None:
    a = CryptoTradingPlan(decisions=[_buy("BTCUSDT")])
    b = CryptoTradingPlan(decisions=[_buy("ETHUSDT"), _sell("BTCUSDT")])
    d = diff_plans(a, b)
    assert d["shared"] == 0
    assert d["only_live"] == 1
    assert d["only_shadow"] == 2


def test_aggregate_plan_diffs() -> None:
    diffs = [
        {"shared": 2, "only_live": 0, "only_shadow": 0},
        {"shared": 1, "only_live": 1, "only_shadow": 0},
    ]
    out = aggregate_plan_diffs(diffs)
    assert out["n"] == 2
    assert out["frac_disagreed"] == 0.25  # 1 disagree / 4 total
