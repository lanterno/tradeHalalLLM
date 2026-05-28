"""BasicRiskEngine — the three independent halt conditions (REARCHITECTURE L7)."""

from __future__ import annotations

from halabot.risk.engine import BasicRiskEngine, PortfolioSnapshot, RiskConfig

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
