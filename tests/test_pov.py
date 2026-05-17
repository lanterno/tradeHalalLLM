"""Tests for trading/pov.py — Round-5 Wave 12.C."""

from __future__ import annotations

import pytest

from halal_trader.trading.pov import (
    ChildOrder,
    POVPolicy,
    POVState,
    initialise_pov,
    render_state,
    tick,
)
from halal_trader.trading.twap import Side


def _init(parent_quantity: float = 1000.0) -> POVState:
    return initialise_pov(
        parent_id="P-001", symbol="AAPL", side=Side.BUY, parent_quantity=parent_quantity
    )


# --- Validation -----------------------------------------------------


def test_default_policy():
    p = POVPolicy()
    assert p.participation_rate == 0.10


def test_policy_below_min_rejected():
    with pytest.raises(ValueError):
        POVPolicy(participation_rate=0.0001)


def test_policy_above_max_rejected():
    with pytest.raises(ValueError):
        POVPolicy(participation_rate=0.60)


def test_policy_min_above_max_rejected():
    with pytest.raises(ValueError):
        POVPolicy(min_child_quantity=10, max_child_quantity=5)


def test_state_zero_parent_rejected():
    with pytest.raises(ValueError):
        POVState(
            parent_id="P",
            symbol="A",
            side=Side.BUY,
            parent_quantity=0,
            cumulative_filled=0,
            cumulative_market_volume=0,
        )


def test_state_filled_exceeds_parent_rejected():
    with pytest.raises(ValueError):
        POVState(
            parent_id="P",
            symbol="A",
            side=Side.BUY,
            parent_quantity=100,
            cumulative_filled=200,
            cumulative_market_volume=1000,
        )


def test_state_remaining_basic():
    s = POVState(
        parent_id="P",
        symbol="A",
        side=Side.BUY,
        parent_quantity=100,
        cumulative_filled=30,
        cumulative_market_volume=1000,
    )
    assert s.remaining() == 70


def test_state_is_complete():
    s = POVState(
        parent_id="P",
        symbol="A",
        side=Side.BUY,
        parent_quantity=100,
        cumulative_filled=100,
        cumulative_market_volume=1000,
    )
    assert s.is_complete()


# --- tick -----------------------------------------------------


def test_tick_first_period_at_target_pct():
    """10% of 5000 mkt volume → child = 500."""
    state = _init(parent_quantity=5000)
    child, new_state = tick(state, 5000.0)
    assert child.quantity == 500.0
    assert new_state.cumulative_filled == 500.0
    assert new_state.cumulative_market_volume == 5000.0


def test_tick_zero_volume_zero_child():
    state = _init()
    child, new_state = tick(state, 0.0)
    assert child.quantity == 0.0
    assert new_state.cumulative_market_volume == 0.0


def test_tick_negative_volume_rejected():
    state = _init()
    with pytest.raises(ValueError):
        tick(state, -1.0)


def test_tick_below_min_returns_zero_child():
    """If 10% of vol < min_child_quantity, no child fires this period."""
    state = _init()
    pol = POVPolicy(min_child_quantity=100.0)
    child, new_state = tick(state, 500.0, policy=pol)  # 10% = 50 < 100
    assert child.quantity == 0.0
    # But cumulative volume tracked
    assert new_state.cumulative_market_volume == 500.0


def test_tick_capped_by_remaining_quantity():
    """When deficit > remaining, child is capped at remaining."""
    state = POVState(
        parent_id="P",
        symbol="A",
        side=Side.BUY,
        parent_quantity=100,
        cumulative_filled=80,
        cumulative_market_volume=0,
    )
    child, new_state = tick(state, 1_000_000.0)
    assert child.quantity == 20  # only 20 left
    assert new_state.is_complete()


def test_tick_capped_by_max_child():
    """child cannot exceed max_child_quantity."""
    state = _init(parent_quantity=1_000_000)
    pol = POVPolicy(max_child_quantity=1000.0)
    # 10% of 100k = 10000, but capped at 1000
    child, _ = tick(state, 100000.0, policy=pol)
    assert child.quantity == 1000.0


def test_tick_pace_catches_up_after_low_volume_period():
    """If last period was below min, next period sends accumulated deficit."""
    state = _init(parent_quantity=5000)
    pol = POVPolicy(min_child_quantity=100.0)
    # Period 1: 500 mkt vol → 10% = 50 < min → no child, but cum_vol=500
    _, state = tick(state, 500.0, policy=pol)
    # Period 2: 2000 mkt vol → cum_vol=2500, target = 250 → child=250
    child, state = tick(state, 2000.0, policy=pol)
    assert child.quantity == 250.0


def test_tick_completes_through_multiple_periods():
    state = _init(parent_quantity=1000)
    for _ in range(20):
        child, state = tick(state, 1000.0)
        if state.is_complete():
            break
    assert state.is_complete()


# --- ChildOrder validation -------------------------------------------


def test_child_negative_qty_rejected():
    with pytest.raises(ValueError):
        ChildOrder(parent_id="P", quantity=-1, side=Side.BUY)


# --- Render --------------------------------------------------------


def test_render_includes_progress():
    state = _init(parent_quantity=1000)
    _, state = tick(state, 1000.0)
    out = render_state(state)
    assert "POV" in out
    assert "AAPL" in out


def test_render_no_secret_leak():
    state = _init()
    out = render_state(state)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ---------------------------------------------------------


def test_e2e_pov_walks_to_completion_with_realistic_volume():
    """Buy 10000 shares at 10% participation across 5 periods of varying volume."""
    state = _init(parent_quantity=10000)
    volumes = [50000, 30000, 20000, 40000, 60000]  # 200k total → 20k @ 10% = 20k
    fills = []
    for v in volumes:
        child, state = tick(state, v)
        fills.append(child.quantity)
    # Cumulative target = 10000; should be hit
    assert sum(fills) == 10000
    assert state.is_complete()


def test_replay_consistency():
    state = _init()
    a_child, a_state = tick(state, 1000.0)
    b_child, b_state = tick(state, 1000.0)
    assert a_child == b_child
    assert a_state == b_state
