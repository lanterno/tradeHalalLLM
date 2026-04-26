"""Tests for the adversarial stress harness."""

from __future__ import annotations

import pytest

from halal_trader.crypto.stress import (
    StressScenario,
    blow_off_pump_klines,
    evaluate_scenarios,
    flash_crash_klines,
    gap_down_klines,
    grade,
    illiquid_drift_klines,
    render_report,
    standard_scenarios,
    sustained_downtrend_klines,
)
from halal_trader.domain.models import (
    CryptoTradeDecision,
    CryptoTradingPlan,
    TradeAction,
)


def _plan(*decisions: CryptoTradeDecision, outlook: str = "") -> CryptoTradingPlan:
    return CryptoTradingPlan(decisions=list(decisions), market_outlook=outlook)


def _buy(symbol: str = "BTCUSDT", qty: float = 0.1, conf: float = 0.5) -> CryptoTradeDecision:
    return CryptoTradeDecision(
        action=TradeAction.BUY,
        symbol=symbol,
        quantity=qty,
        confidence=conf,
        reasoning="test",
    )


def _sell(symbol: str = "BTCUSDT", qty: float = 0.1, conf: float = 0.5) -> CryptoTradeDecision:
    return CryptoTradeDecision(
        action=TradeAction.SELL,
        symbol=symbol,
        quantity=qty,
        confidence=conf,
        reasoning="test",
    )


def _hold(symbol: str = "BTCUSDT") -> CryptoTradeDecision:
    return CryptoTradeDecision(
        action=TradeAction.HOLD,
        symbol=symbol,
        quantity=0,
        confidence=0.5,
        reasoning="hold",
    )


# ── Generator sanity ──────────────────────────────────────────────


def test_flash_crash_generates_expected_drop() -> None:
    klines = flash_crash_klines(base_price=100.0, drop_pct=0.15, n_pre=10, n_crash=2, n_post=3)
    assert len(klines) == 15
    # close after the crash should be ~15% below pre-crash close
    pre_close = klines[9].close
    crash_end = klines[11].close
    assert crash_end < pre_close
    drop = (pre_close - crash_end) / pre_close
    assert 0.10 < drop < 0.20


def test_pump_generates_expected_rise() -> None:
    klines = blow_off_pump_klines(base_price=100.0, pump_pct=0.30, n_pre=5, n_pump=4, n_top=3)
    assert len(klines) == 12
    start = klines[0].close
    peak = max(k.close for k in klines)
    assert peak / start >= 1.20


def test_gap_down_inserts_single_gap_bar() -> None:
    klines = gap_down_klines(base_price=100.0, gap_pct=0.08, n_pre=3, n_post=3)
    assert len(klines) == 7
    # bar at index 3 is the gap bar
    assert klines[3].close < klines[2].close * 0.95


def test_illiquid_drift_low_volume() -> None:
    klines = illiquid_drift_klines(n=20)
    assert len(klines) == 20
    avg_vol = sum(k.volume for k in klines) / len(klines)
    assert avg_vol < 50  # base 100 × 0.2


def test_sustained_downtrend_drops_overall() -> None:
    klines = sustained_downtrend_klines(base_price=100.0, drop_pct=0.20, n=40)
    start = klines[0].close
    end = klines[-1].close
    # Random walk so check directionally — drop should generally be present.
    assert end < start * 0.95


def test_generators_deterministic_per_seed() -> None:
    a = flash_crash_klines(seed=42)
    b = flash_crash_klines(seed=42)
    assert [k.close for k in a] == [k.close for k in b]
    c = flash_crash_klines(seed=7)
    assert [k.close for k in a] != [k.close for k in c]


# ── Grader logic ──────────────────────────────────────────────────


def test_flash_crash_grader_passes_on_no_buys() -> None:
    sc = next(s for s in standard_scenarios() if s.name == "flash_crash")
    v = grade(sc, _plan(_hold(), _sell()))
    assert v.severity == 0.0
    assert v.passed


def test_flash_crash_grader_fails_on_high_confidence_buy() -> None:
    sc = next(s for s in standard_scenarios() if s.name == "flash_crash")
    v = grade(sc, _plan(_buy(conf=0.9)))
    assert v.severity == 1.0
    assert not v.passed


def test_blow_off_pump_grader_borderline() -> None:
    sc = next(s for s in standard_scenarios() if s.name == "blow_off_pump")
    v = grade(sc, _plan(_buy(conf=0.4)))
    assert 0.0 < v.severity < 0.5
    assert v.passed


def test_blow_off_pump_grader_fails_on_high_conf() -> None:
    sc = next(s for s in standard_scenarios() if s.name == "blow_off_pump")
    v = grade(sc, _plan(_buy(conf=0.9)))
    assert v.severity >= 0.5
    assert not v.passed


def test_sustained_downtrend_punishes_counter_trend_buy() -> None:
    sc = next(s for s in standard_scenarios() if s.name == "sustained_downtrend")
    v = grade(sc, _plan(_buy(conf=0.85)))
    assert v.severity >= 0.7


def test_unknown_scenario_grades_zero() -> None:
    sc = StressScenario(name="custom_x", description="test", klines=[], expected="?")
    v = grade(sc, _plan(_buy(conf=0.99)))
    assert v.severity == 0.0


# ── End-to-end harness ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evaluate_scenarios_runs_all_standard() -> None:
    async def safe_strategy(_klines):
        # always sane — never buy
        return _plan(_hold())

    verdicts = await evaluate_scenarios(safe_strategy)
    assert len(verdicts) == len(standard_scenarios())
    assert all(v.passed for v in verdicts)
    assert all(v.severity == 0.0 for v in verdicts)


@pytest.mark.asyncio
async def test_evaluate_scenarios_catches_bad_strategy() -> None:
    async def reckless_strategy(_klines):
        # always full-confidence buy
        return _plan(_buy(conf=0.95))

    verdicts = await evaluate_scenarios(reckless_strategy)
    failed = [v for v in verdicts if not v.passed]
    assert len(failed) >= 3, "reckless strategy should fail multiple scenarios"


@pytest.mark.asyncio
async def test_evaluate_scenarios_handles_strategy_error() -> None:
    async def broken_strategy(_klines):
        raise RuntimeError("simulated crash")

    verdicts = await evaluate_scenarios(broken_strategy)
    assert all(v.severity == 1.0 for v in verdicts)
    assert all(any("strategy raised" in n for n in v.notes) for v in verdicts)


def test_render_report_smoke() -> None:
    sc = next(s for s in standard_scenarios() if s.name == "flash_crash")
    v_pass = grade(sc, _plan(_hold()))
    v_fail = grade(sc, _plan(_buy(conf=0.95)))
    out = render_report([v_pass, v_fail])
    assert "flash_crash" in out
    assert "FAIL" in out
    assert "✔" in out
    assert "✘" in out
