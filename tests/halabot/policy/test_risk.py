"""BasicRiskEngine — the three independent halt conditions (REARCHITECTURE L7)."""

from __future__ import annotations

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


def test_correlation_multipliers_cluster_haircut():
    # Two perfectly co-moving series + one independent → the pair is haircut.
    up = [float(i) for i in range(20)]
    same = [2.0 * i + 1.0 for i in range(20)]  # ρ(up, same) = 1.0
    zig = [(-1.0) ** i for i in range(20)]  # uncorrelated with the trend
    mults = correlation_multipliers({"A": up, "B": same, "C": zig}, threshold=0.7)
    assert mults["A"] < 1.0 and mults["B"] < 1.0  # clustered → haircut
    assert mults["C"] == 1.0  # lone wolf → full size


def test_correlation_multiplier_applied_in_evaluate():
    up = [float(i) for i in range(20)]
    same = [3.0 * i for i in range(20)]
    s = ENGINE.evaluate(_snap(), returns_by_asset={"A": up, "B": same})
    assert s.correlation_multiplier("A") < 1.0
