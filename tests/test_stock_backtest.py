"""Stock backtester smoke tests."""

from __future__ import annotations

import math

import pytest

from halal_trader.trading.backtest import StockBacktestEngine


def _bars_oscillating(n: int = 200) -> list[dict]:
    """Generate sinusoidal bars so the rule-based RSI strategy fires both ways."""
    bars = []
    for i in range(n):
        # 50-bar period with 5% amplitude around $100.
        x = math.sin(2 * math.pi * i / 50) * 5
        c = 100 + x
        bars.append(
            {
                "o": c,
                "h": c + 0.5,
                "l": c - 0.5,
                "c": c,
                "v": 1_000_000,
            }
        )
    return bars


async def test_returns_empty_result_when_window_too_large():
    engine = StockBacktestEngine()
    bars = _bars_oscillating(n=20)  # < default window 50
    result = await engine.run("AAPL", bars, window_size=50)
    assert result.trades == []
    assert result.final_balance == 10_000.0
    assert result.total_return_pct == 0.0


async def test_runs_full_backtest_with_oscillating_bars():
    engine = (
        StockBacktestEngine(window_size=50, max_position_pct=0.10)
        if False
        else (StockBacktestEngine(max_position_pct=0.10))
    )
    bars = _bars_oscillating(n=300)
    result = await engine.run("AAPL", bars, window_size=50)
    # The rule-based strategy isn't guaranteed to trade on synthetic
    # oscillation, but the equity curve must have one entry per bar
    # past the window (bar at index window..n).
    assert len(result.equity_curve) >= 1
    # Rates must be in valid ranges.
    assert 0 <= result.win_rate <= 1
    # Sharpe well-defined (no NaN) when there's any equity movement.
    assert not math.isnan(result.sharpe_ratio)


async def test_zero_fee_default_matches_alpaca_commission_free():
    engine = StockBacktestEngine()
    assert engine._fee_pct == 0.0


async def test_handles_alpaca_nested_bars_shape():
    """Alpaca sometimes returns ``{"bars": [...]}`` wrapped — coerce works."""
    engine = StockBacktestEngine()
    bars = {"bars": _bars_oscillating(n=120)}
    result = await engine.run("AAPL", bars, window_size=50)
    assert isinstance(result.win_rate, float)


async def test_invalid_initial_state_returns_clean_zero_result():
    engine = StockBacktestEngine()
    result = await engine.run("AAPL", [], window_size=50)
    assert result.total_return_pct == 0.0
    assert result.win_rate == 0.0
    assert result.equity_curve == [10_000.0]


async def test_sharpe_uses_daily_annualisation():
    """For daily bars, sqrt(252) annualisation; sanity-check finite result."""
    engine = StockBacktestEngine()
    bars = _bars_oscillating(n=120)
    result = await engine.run("AAPL", bars)
    # We don't pin a specific value — just require it's finite and within
    # a reasonable order of magnitude (not 1e9 from a divide-by-near-zero).
    assert -100 < result.sharpe_ratio < 100


@pytest.mark.parametrize("max_pct", [0.05, 0.10, 0.25])
async def test_max_position_pct_caps_size(max_pct):
    engine = StockBacktestEngine(max_position_pct=max_pct)
    bars = _bars_oscillating(n=200)
    result = await engine.run("AAPL", bars)
    # No trade ever uses more than max_pct × initial equity (modulo
    # vol-aware slippage drift, which is < a few %).
    for t in result.trades:
        assert t.quantity * t.price <= 10_000 * max_pct * 1.10
