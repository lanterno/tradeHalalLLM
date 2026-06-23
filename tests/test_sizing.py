"""Tests for position-sizing primitives + backtest drawdown throttle."""

from __future__ import annotations

import pytest

from halal_trader.core.sizing import drawdown_throttle, half_kelly_fraction

# ── half_kelly_fraction ──────────────────────────────────────────


def test_kelly_positive_edge_positive_fraction():
    # 60% win rate, 2:1 payoff, plenty of samples → positive half-Kelly.
    f = half_kelly_fraction(0.6, 2.0, n=100)
    # Kelly = .6 - .4/2 = .4 ; half = .2
    assert f == pytest.approx(0.2)


def test_kelly_negative_edge_is_zero_never_short():
    # 40% win rate, 1:1 payoff → Kelly = .4 - .6 = -.2 → clamped to 0.
    assert half_kelly_fraction(0.4, 1.0, n=100) == 0.0


def test_kelly_capped():
    # Strong edge (half-Kelly ~0.47) exceeds a low cap → clamped to the cap.
    assert half_kelly_fraction(0.95, 10.0, n=100, cap=0.1) == 0.1


def test_kelly_sample_gated():
    # Same strong edge but too few trades → 0 (don't bet on noise).
    assert half_kelly_fraction(0.6, 2.0, n=5, min_n=20) == 0.0
    assert half_kelly_fraction(0.6, 2.0, n=20, min_n=20) > 0.0  # boundary inclusive


def test_kelly_degenerate_inputs():
    assert half_kelly_fraction(0.6, 0.0, n=100) == 0.0  # non-positive payoff
    assert half_kelly_fraction(1.5, 2.0, n=100) == 0.0  # win_rate out of range


def test_kelly_monotonic_in_win_rate():
    a = half_kelly_fraction(0.55, 1.5, n=100)
    b = half_kelly_fraction(0.65, 1.5, n=100)
    assert b > a


# ── drawdown_throttle ────────────────────────────────────────────


def test_throttle_full_size_at_peak():
    assert drawdown_throttle(0.0, max_drawdown_budget=0.2) == 1.0
    assert drawdown_throttle(-0.01, max_drawdown_budget=0.2) == 1.0  # above peak


def test_throttle_floor_at_budget():
    assert drawdown_throttle(0.2, max_drawdown_budget=0.2, floor=0.1) == 0.1
    assert drawdown_throttle(0.5, max_drawdown_budget=0.2, floor=0.1) == 0.1  # beyond


def test_throttle_linear_between():
    # halfway to the budget → halfway down from 1.0 to 0.
    assert drawdown_throttle(0.1, max_drawdown_budget=0.2) == pytest.approx(0.5)


def test_throttle_disabled_when_budget_nonpositive():
    assert drawdown_throttle(0.3, max_drawdown_budget=0.0) == 1.0


def test_throttle_monotonic_decreasing():
    vals = [drawdown_throttle(d, max_drawdown_budget=0.25) for d in (0.0, 0.05, 0.1, 0.2, 0.25)]
    assert vals == sorted(vals, reverse=True)


# ── backtest integration: throttle shrinks size in drawdown ──────


def test_backtest_drawdown_throttle_reduces_position():
    from halal_trader.crypto.backtest import SimulatedExecutor

    def _sized_qty(budget):
        ex = SimulatedExecutor(initial_balance=10_000, drawdown_throttle_budget=budget)
        # Simulate a 10% drawdown: peak 10k in the curve, balance now 9k.
        ex.equity_curve = [10_000.0]
        ex.balance = 9_000.0
        ok = ex.buy("BTCUSDT", price=100.0, timestamp=0)
        assert ok
        assert ex.position is not None
        return ex.position.quantity

    throttled = _sized_qty(0.20)  # 10% dd vs 20% budget → 0.5x multiplier
    full = _sized_qty(None)  # throttle off
    assert throttled < full
    assert throttled == pytest.approx(full * 0.5, rel=1e-6)
