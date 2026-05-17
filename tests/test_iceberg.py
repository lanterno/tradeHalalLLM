"""Tests for trading/iceberg.py — Round-5 Wave 12.D."""

from __future__ import annotations

from datetime import timedelta

import pytest

from halal_trader.trading.iceberg import (
    IcebergPolicy,
    IcebergState,
    ReplenishStrategy,
    fill_visible,
    initialise_iceberg,
    render_state,
    replenish_time,
)
from halal_trader.trading.twap import Side


def _init(parent_quantity: float = 1000.0, **kwargs) -> IcebergState:
    return initialise_iceberg(
        parent_id="P-001",
        symbol="AAPL",
        side=Side.BUY,
        parent_quantity=parent_quantity,
        **kwargs,
    )


# --- Validation -------------------------------------------------------------


def test_strategy_string_values():
    assert ReplenishStrategy.ON_FILL.value == "on_fill"
    assert ReplenishStrategy.TIME_BASED.value == "time_based"


def test_default_policy():
    p = IcebergPolicy()
    assert p.max_visible_pct == 0.10
    assert p.replenish_strategy is ReplenishStrategy.ON_FILL


def test_policy_zero_max_visible_rejected():
    with pytest.raises(ValueError):
        IcebergPolicy(max_visible_pct=0.0)


def test_policy_above_one_max_visible_rejected():
    with pytest.raises(ValueError):
        IcebergPolicy(max_visible_pct=1.5)


def test_policy_zero_min_tip_rejected():
    with pytest.raises(ValueError):
        IcebergPolicy(min_tip_quantity=0.0)


def test_policy_zero_interval_rejected():
    with pytest.raises(ValueError):
        IcebergPolicy(time_based_interval=timedelta(0))


def test_state_negative_filled_rejected():
    with pytest.raises(ValueError):
        IcebergState(
            parent_id="P",
            symbol="A",
            side=Side.BUY,
            parent_quantity=100,
            visible_quantity=50,
            hidden_quantity=50,
            filled_quantity=-1,
        )


def test_state_invariant_violation_rejected():
    """visible + hidden + filled must equal parent."""
    with pytest.raises(ValueError):
        IcebergState(
            parent_id="P",
            symbol="A",
            side=Side.BUY,
            parent_quantity=100,
            visible_quantity=20,
            hidden_quantity=20,
            filled_quantity=20,  # 60, not 100
        )


# --- Initialisation ---------------------------------------------------------


def test_init_visible_is_max_pct_of_parent():
    state = _init(parent_quantity=1000.0)
    assert state.visible_quantity == pytest.approx(100.0)  # 10%
    assert state.hidden_quantity == pytest.approx(900.0)


def test_init_visible_capped_at_parent():
    """If max_pct * parent > parent, tip = parent."""
    state = _init(parent_quantity=10.0)  # default min_tip=1.0, max_pct=10% = 1.0
    assert state.visible_quantity <= state.parent_quantity


def test_init_visible_at_least_min_tip():
    """Min_tip overrides max_pct when max_pct yields a smaller tip."""
    # parent=10, max_pct=10% → 1.0; min_tip=2.0 → tip should be 2.0
    state = _init(parent_quantity=10.0, policy=IcebergPolicy(min_tip_quantity=2.0))
    assert state.visible_quantity == pytest.approx(2.0)
    assert state.hidden_quantity == pytest.approx(8.0)


def test_init_hidden_balance_correct():
    state = _init(parent_quantity=1000.0)
    assert state.visible_quantity + state.hidden_quantity == pytest.approx(1000.0)


def test_init_zero_quantity_rejected():
    with pytest.raises(ValueError):
        _init(parent_quantity=0.0)


# --- Fills -----------------------------------------------------------------


def test_fill_partial_visible_keeps_hidden():
    state = _init(parent_quantity=1000.0)  # visible=100, hidden=900
    new_state = fill_visible(state, 30)
    assert new_state.filled_quantity == 30
    assert new_state.visible_quantity == 70
    assert new_state.hidden_quantity == 900


def test_fill_full_visible_replenishes_on_fill_strategy():
    state = _init(parent_quantity=1000.0)  # visible=100, hidden=900
    new_state = fill_visible(state, 100)  # exhausts visible
    assert new_state.filled_quantity == 100
    # Replenishment from hidden: another tip should appear
    assert new_state.visible_quantity == pytest.approx(100.0)
    assert new_state.hidden_quantity == pytest.approx(800.0)


def test_fill_full_visible_no_replenish_when_hidden_empty():
    state = _init(parent_quantity=100.0, policy=IcebergPolicy(max_visible_pct=1.0))
    # Visible=100 (=parent), hidden=0
    new_state = fill_visible(state, 100)
    assert new_state.is_complete()
    assert new_state.visible_quantity == 0
    assert new_state.hidden_quantity == 0


def test_fill_above_visible_rejected():
    state = _init(parent_quantity=1000.0)
    with pytest.raises(ValueError):
        fill_visible(state, 200)  # only 100 visible


def test_fill_zero_rejected():
    state = _init()
    with pytest.raises(ValueError):
        fill_visible(state, 0)


def test_fill_time_based_does_not_replenish_on_fill():
    state = _init(
        parent_quantity=1000.0,
        policy=IcebergPolicy(replenish_strategy=ReplenishStrategy.TIME_BASED),
    )
    pol = IcebergPolicy(replenish_strategy=ReplenishStrategy.TIME_BASED)
    new_state = fill_visible(state, 100, policy=pol)
    # Visible exhausted, but TIME_BASED doesn't auto-replenish on fill
    assert new_state.visible_quantity == 0
    assert new_state.hidden_quantity == 900


def test_replenish_time_tops_up_visible():
    state = _init(parent_quantity=1000.0)
    new_state = fill_visible(state, 100, policy=IcebergPolicy(replenish_strategy=ReplenishStrategy.TIME_BASED))
    # Visible=0, hidden=900
    topped = replenish_time(new_state)
    assert topped.visible_quantity == pytest.approx(100.0)
    assert topped.hidden_quantity == pytest.approx(800.0)


def test_replenish_time_no_op_when_hidden_empty():
    state = _init(parent_quantity=100.0, policy=IcebergPolicy(max_visible_pct=1.0))
    topped = replenish_time(state)
    assert topped == state


def test_replenish_time_no_op_when_visible_full():
    state = _init(parent_quantity=1000.0)
    topped = replenish_time(state)
    assert topped == state


# --- E2E -------------------------------------------------------------------


def test_e2e_iceberg_walks_through_full_quantity():
    state = _init(parent_quantity=1000.0)
    while not state.is_complete():
        # Fill the entire visible tip each time
        state = fill_visible(state, state.visible_quantity)
    assert state.filled_quantity == pytest.approx(1000.0)


def test_e2e_partial_fills_eventually_complete():
    state = _init(parent_quantity=1000.0)
    while not state.is_complete():
        # Fill in 25-share increments
        fill = min(25, state.visible_quantity)
        if fill <= 0:
            # Stuck — replenish manually under TIME_BASED policy
            state = replenish_time(state)
            continue
        state = fill_visible(state, fill)
    assert state.filled_quantity == pytest.approx(1000.0)


# --- Render ----------------------------------------------------------------


def test_render_includes_summary():
    state = _init(parent_quantity=1000.0)
    out = render_state(state)
    assert "P-001" in out
    assert "AAPL" in out
    assert "buy" in out
    assert "🧊" in out


def test_render_no_secret_leak():
    state = _init(parent_quantity=1000.0)
    out = render_state(state)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


def test_replay_consistency():
    a = _init()
    b = _init()
    assert a == b
