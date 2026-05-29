"""BasicRiskEngine — the three independent halt conditions (REARCHITECTURE L7)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from halabot.belief.schema import BeliefState, Regime
from halabot.risk.engine import (
    BasicRiskEngine,
    PortfolioSnapshot,
    RiskConfig,
    correlation_multipliers,
)

CFG = RiskConfig(max_portfolio_heat_pct=0.05, max_drawdown_pct=0.08, daily_loss_limit=0.02)
ENGINE = BasicRiskEngine(CFG)


def _snap(**kw) -> PortfolioSnapshot:
    base = dict(
        equity=100_000.0,
        peak_equity=100_000.0,
        unrealized_pnl=0.0,
        realized_pnl_today=0.0,
        starting_equity_today=100_000.0,
        gross_exposure=0.0,
    )
    base.update(kw)
    return PortfolioSnapshot(**base)  # type: ignore[arg-type]


def test_clean_state_not_halted():
    s = ENGINE.evaluate(_snap())
    assert not s.halted
    assert s.correlation_multiplier("NVDA") == 1.0
    assert s.volatility_multiplier("NVDA") == 1.0


def test_heat_halts():
    s = ENGINE.evaluate(_snap(unrealized_pnl=-6_000.0))  # 6% unrealized loss > 5%
    assert s.halted and "heat" in s.reason


def test_drawdown_halts():
    s = ENGINE.evaluate(_snap(peak_equity=100_000.0, equity=91_000.0))  # 9% dd > 8%
    assert s.halted and "drawdown" in s.reason


def test_daily_realized_loss_halts_even_without_heat():
    # Realized loss day, but flat now (no unrealized heat, equity near peak):
    # the daily-loss floor must still trip (R-10).
    s = ENGINE.evaluate(
        _snap(realized_pnl_today=-2_500.0, unrealized_pnl=0.0, equity=100_000.0)
    )
    assert s.halted and "daily realized loss" in s.reason


def test_gross_exposure_passed_through():
    s = ENGINE.evaluate(_snap(gross_exposure=0.6))
    assert s.gross_exposure == 0.6


# ── belief-aware multipliers ──
def test_volatile_regime_gets_size_haircut():
    beliefs = [
        BeliefState(asset="NVDA", regime=Regime.TRENDING_UP),
        BeliefState(asset="WILD", regime=Regime.VOLATILE),
    ]
    s = ENGINE.evaluate(_snap(), beliefs=beliefs)
    assert s.volatility_multiplier("WILD") == CFG.volatile_size_mult  # haircut
    assert s.volatility_multiplier("NVDA") == 1.0  # trending → full size


def _timed(vals):
    base = datetime(2026, 5, 28, tzinfo=UTC)
    return [(base + timedelta(minutes=i), v) for i, v in enumerate(vals)]


def test_correlation_multipliers_cluster_haircut():
    # Two perfectly co-moving series + one independent → the pair is haircut.
    up = _timed([float(i) for i in range(20)])
    same = _timed([2.0 * i + 1.0 for i in range(20)])  # ρ(up, same) = 1.0
    zig = _timed([(-1.0) ** i for i in range(20)])  # uncorrelated with the trend
    mults = correlation_multipliers({"A": up, "B": same, "C": zig}, threshold=0.7)
    assert mults["A"] < 1.0 and mults["B"] < 1.0  # clustered → haircut
    assert mults["C"] == 1.0  # lone wolf → full size


def test_correlation_aligns_by_timestamp_not_position():
    # B is the same series as A but SHIFTED one bar later in time. Tail-index
    # alignment would (wrongly) see them as correlated; timestamp alignment must
    # only pair the overlapping bars.
    base = datetime(2026, 5, 28, tzinfo=UTC)
    seq = [float(i % 3) for i in range(30)]  # repeating, so a 1-bar shift decorrelates
    a = [(base + timedelta(minutes=i), seq[i]) for i in range(30)]
    b = [(base + timedelta(minutes=i + 1), seq[i]) for i in range(30)]  # shifted +1 bar
    mults = correlation_multipliers({"A": a, "B": b}, threshold=0.95)
    # On shared timestamps the values differ (phase-shifted) → not >0.95 → no haircut.
    assert mults["A"] == 1.0 and mults["B"] == 1.0


def test_vol_targeting_downsizes_high_vol_names():
    # Opt-in (default off): with a per-bar vol target set, a high-vol name (big
    # swings) is haircut toward equal risk; a calm name is left at full size.
    eng = BasicRiskEngine(RiskConfig(target_vol_per_bar=0.01, vol_size_floor=0.3))
    base = datetime(2026, 5, 28, tzinfo=UTC)
    calm = [(base + timedelta(minutes=i), 0.0005 * ((i % 2) * 2 - 1)) for i in range(20)]
    wild = [(base + timedelta(minutes=i), 0.05 * ((i % 2) * 2 - 1)) for i in range(20)]
    s = eng.evaluate(_snap(), returns_by_asset={"CALM": calm, "WILD": wild})
    assert s.volatility_multiplier("WILD") < 1.0  # downsized for its risk
    assert s.volatility_multiplier("CALM") == 1.0  # calm → full size (no upsize)
    assert s.volatility_multiplier("WILD") >= 0.3  # floored


def test_vol_targeting_off_by_default():
    # Default config (target_vol_per_bar=0) leaves every name at full size.
    base = datetime(2026, 5, 28, tzinfo=UTC)
    wild = [(base + timedelta(minutes=i), 0.05 * ((i % 2) * 2 - 1)) for i in range(20)]
    s = ENGINE.evaluate(_snap(), returns_by_asset={"WILD": wild})
    assert s.volatility_multiplier("WILD") == 1.0  # inert unless opted in


def test_correlation_multiplier_applied_in_evaluate():
    up = _timed([float(i) for i in range(20)])
    same = _timed([3.0 * i for i in range(20)])
    s = ENGINE.evaluate(_snap(), returns_by_asset={"A": up, "B": same})
    assert s.correlation_multiplier("A") < 1.0
