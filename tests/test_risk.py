"""Direct tests for the crypto PortfolioRiskEngine."""

from __future__ import annotations

from halal_trader.crypto.risk import PortfolioRiskEngine
from halal_trader.domain.models import Kline


def _kl(close: float, open_time: int) -> Kline:
    return Kline(
        open_time=open_time,
        open=close,
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        volume=1.0,
        close_time=open_time + 60_000,
    )


def _series(start: float, n: int, step: float) -> list[Kline]:
    return [_kl(start + i * step, i * 60_000) for i in range(n)]


def _engine(**overrides) -> PortfolioRiskEngine:
    base = dict(
        base_max_position_pct=0.25,
        max_portfolio_heat_pct=0.05,
        max_drawdown_pct=0.08,
        high_correlation_threshold=0.7,
        correlation_reduction_factor=0.5,
        atr_baseline=0.02,
    )
    base.update(overrides)
    return PortfolioRiskEngine(**base)


def test_clean_portfolio_no_halt():
    eng = _engine()
    state = eng.evaluate(
        klines_by_symbol={"BTC": _series(100, 50, 0.5)},
        indicators_cache={"BTC": {"atr_pct": 0.02}},
        open_positions_value={},
        unrealized_pnl={},
        total_equity=10_000.0,
    )
    assert not state.is_halted
    assert state.drawdown_pct == 0.0


def test_drawdown_halt_after_peak():
    eng = _engine()
    eng.evaluate({}, {}, {}, {}, total_equity=10_000.0)  # peak set
    state = eng.evaluate({}, {}, {}, {}, total_equity=9_100.0)  # 9% down
    assert state.is_halted
    assert "Drawdown" in state.halt_reason


def test_drawdown_does_not_halt_below_threshold():
    eng = _engine(max_drawdown_pct=0.10)
    eng.evaluate({}, {}, {}, {}, total_equity=10_000.0)
    state = eng.evaluate({}, {}, {}, {}, total_equity=9_500.0)  # 5% down
    assert not state.is_halted


def test_heat_halt_when_unrealized_loss_exceeds_pct():
    eng = _engine(max_portfolio_heat_pct=0.05)
    state = eng.evaluate(
        klines_by_symbol={},
        indicators_cache={},
        open_positions_value={"BTC": 5_000.0},
        unrealized_pnl={"BTC": -600.0},  # -6% on $10k equity
        total_equity=10_000.0,
    )
    assert state.is_halted
    assert "Portfolio heat" in state.halt_reason


def test_heat_below_threshold_no_halt():
    eng = _engine(max_portfolio_heat_pct=0.05)
    state = eng.evaluate(
        klines_by_symbol={},
        indicators_cache={},
        open_positions_value={"BTC": 5_000.0},
        unrealized_pnl={"BTC": -200.0},  # -2%
        total_equity=10_000.0,
    )
    assert not state.is_halted


def test_high_correlation_shrinks_size_for_open_pair():
    eng = _engine(high_correlation_threshold=0.7, correlation_reduction_factor=0.5)
    klines = {
        "BTC": _series(100, 50, 1.0),  # rising linearly
        "ETH": _series(200, 50, 2.0),  # also rising linearly → corr ≈ 1
    }
    state = eng.evaluate(
        klines_by_symbol=klines,
        indicators_cache={
            "BTC": {"atr_pct": 0.02},
            "ETH": {"atr_pct": 0.02},
        },
        open_positions_value={"BTC": 1.0},
        unrealized_pnl={},
        total_equity=10_000.0,
    )
    # ETH is highly correlated with the open BTC → size reduced
    assert state.adjusted_position_pcts["ETH"] < 0.25


def test_low_correlation_keeps_full_size():
    """The engine uses |corr|, so we need actual decorrelated returns."""
    import math

    eng = _engine()
    btc = [_kl(100 + i * 0.5, i * 60_000) for i in range(50)]
    # Sine-wave returns are zero-correlated with a linear trend.
    eth_closes = [200 + math.sin(i / 3.0) * 5 for i in range(50)]
    eth = [_kl(c, i * 60_000) for i, c in enumerate(eth_closes)]
    state = eng.evaluate(
        klines_by_symbol={"BTC": btc, "ETH": eth},
        indicators_cache={
            "BTC": {"atr_pct": 0.02},
            "ETH": {"atr_pct": 0.02},
        },
        open_positions_value={"BTC": 1.0},
        unrealized_pnl={},
        total_equity=10_000.0,
    )
    # Low correlation → no shrink; ETH stays at base 0.25.
    assert state.adjusted_position_pcts["ETH"] >= 0.20


def test_volatility_scaling_caps_at_baseline():
    eng = _engine(atr_baseline=0.02)
    state = eng.evaluate(
        klines_by_symbol={"BTC": _series(100, 50, 0.5)},
        indicators_cache={"BTC": {"atr_pct": 0.04}},  # 2x baseline
        open_positions_value={},
        unrealized_pnl={},
        total_equity=10_000.0,
    )
    # ATR > baseline → vol_scale < 1 → final pct < base
    assert state.adjusted_position_pcts["BTC"] < 0.25


def test_format_for_prompt_includes_metrics():
    eng = _engine()
    state = eng.evaluate(
        klines_by_symbol={"BTC": _series(100, 50, 0.0)},
        indicators_cache={"BTC": {"atr_pct": 0.02}},
        open_positions_value={"BTC": 5_000.0},
        unrealized_pnl={"BTC": -100.0},
        total_equity=10_000.0,
    )
    text = eng.format_for_prompt(state)
    assert "Portfolio Heat" in text


def test_get_adjusted_max_position_pct_falls_back_to_base():
    eng = _engine()
    state = eng.evaluate({}, {}, {}, {}, total_equity=1_000.0)
    # Empty indicators → no entry in adjusted_position_pcts.
    assert eng.get_adjusted_max_position_pct("BTC", state) == 0.25
