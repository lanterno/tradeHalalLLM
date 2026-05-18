"""Tests for trading/twap.py — Round-5 Wave 12.A."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from halal_trader.trading.twap import (
    ChildOrder,
    Side,
    TwapInputs,
    TwapPolicy,
    cumulative_quantity,
    filter_due,
    render_schedule,
    slice_twap,
)


def _inputs(**overrides) -> TwapInputs:
    base = {
        "parent_id": "P-001",
        "symbol": "AAPL",
        "side": Side.BUY,
        "parent_quantity": 1000.0,
        "start_time": datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 5, 10, 30, tzinfo=timezone.utc),
        "n_slices": 4,
    }
    base.update(overrides)
    return TwapInputs(**base)


def test_side_string_values():
    assert Side.BUY.value == "buy"
    assert Side.SELL.value == "sell"


def test_default_policy():
    p = TwapPolicy()
    assert p.min_slice_quantity == 0.0
    assert p.max_slices == 1000


def test_policy_negative_min_rejected():
    with pytest.raises(ValueError):
        TwapPolicy(min_slice_quantity=-1.0)


def test_policy_zero_max_rejected():
    with pytest.raises(ValueError):
        TwapPolicy(max_slices=0)


def test_inputs_empty_parent_rejected():
    with pytest.raises(ValueError):
        _inputs(parent_id="")


def test_inputs_empty_symbol_rejected():
    with pytest.raises(ValueError):
        _inputs(symbol=" ")


def test_inputs_negative_qty_rejected():
    with pytest.raises(ValueError):
        _inputs(parent_quantity=-1.0)


def test_inputs_naive_start_rejected():
    with pytest.raises(ValueError):
        _inputs(start_time=datetime(2026, 5, 5, 9, 30))


def test_inputs_end_before_start_rejected():
    with pytest.raises(ValueError):
        _inputs(
            start_time=datetime(2026, 5, 5, 10, 30, tzinfo=timezone.utc),
            end_time=datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
        )


def test_inputs_zero_slices_rejected():
    with pytest.raises(ValueError):
        _inputs(n_slices=0)


def test_child_order_negative_slice_rejected():
    with pytest.raises(ValueError):
        ChildOrder(
            parent_id="P",
            slice_index=-1,
            symbol="A",
            side=Side.BUY,
            quantity=1.0,
            submit_time=datetime(2026, 5, 5, tzinfo=timezone.utc),
        )


def test_child_order_zero_qty_rejected():
    with pytest.raises(ValueError):
        ChildOrder(
            parent_id="P",
            slice_index=0,
            symbol="A",
            side=Side.BUY,
            quantity=0.0,
            submit_time=datetime(2026, 5, 5, tzinfo=timezone.utc),
        )


# --- Slicing ----------------------------------------------------------------


def test_slice_count_matches_input():
    schedule = slice_twap(_inputs(n_slices=4))
    assert len(schedule) == 4


def test_slice_total_quantity_exact():
    schedule = slice_twap(_inputs(parent_quantity=1000.0, n_slices=4))
    assert cumulative_quantity(schedule) == pytest.approx(1000.0)


def test_slice_total_quantity_exact_with_remainder():
    """1000.001 / 3 has a float remainder → first slice carries it."""
    schedule = slice_twap(_inputs(parent_quantity=1000.001, n_slices=3))
    assert cumulative_quantity(schedule) == pytest.approx(1000.001)
    # First slice is at least as large as later slices
    assert schedule[0].quantity >= schedule[1].quantity


def test_slice_equal_intervals():
    schedule = slice_twap(
        _inputs(
            n_slices=5,
            start_time=datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
            end_time=datetime(2026, 5, 5, 10, 30, tzinfo=timezone.utc),
        )
    )
    intervals = [
        (schedule[i + 1].submit_time - schedule[i].submit_time) for i in range(len(schedule) - 1)
    ]
    assert all(intv == intervals[0] for intv in intervals)
    # First slice at start, last at end
    assert schedule[0].submit_time == datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc)
    assert schedule[-1].submit_time == datetime(2026, 5, 5, 10, 30, tzinfo=timezone.utc)


def test_slice_single_slice_at_start():
    schedule = slice_twap(_inputs(n_slices=1))
    assert len(schedule) == 1
    assert schedule[0].submit_time == datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc)
    assert schedule[0].quantity == 1000.0


def test_slice_indices_zero_through_n_minus_one():
    schedule = slice_twap(_inputs(n_slices=4))
    assert [c.slice_index for c in schedule] == [0, 1, 2, 3]


def test_slice_below_min_rejected():
    """Slice size below min_slice_quantity → reject."""
    with pytest.raises(ValueError):
        slice_twap(
            _inputs(parent_quantity=10.0, n_slices=100),
            policy=TwapPolicy(min_slice_quantity=1.0),
        )


def test_slice_count_exceeds_max_rejected():
    with pytest.raises(ValueError):
        slice_twap(_inputs(n_slices=20), policy=TwapPolicy(max_slices=10))


# --- Helpers ---------------------------------------------------------------


def test_cumulative_quantity_empty_zero():
    assert cumulative_quantity([]) == 0


def test_filter_due_returns_due_only():
    schedule = slice_twap(_inputs(n_slices=4))
    now = datetime(2026, 5, 5, 9, 50, tzinfo=timezone.utc)  # halfway
    due = filter_due(schedule, now=now)
    # First two should be due at 9:30 + 9:50; third at 10:10 (not yet)
    assert len(due) == 2


def test_filter_due_naive_now_rejected():
    schedule = slice_twap(_inputs(n_slices=4))
    with pytest.raises(ValueError):
        filter_due(schedule, now=datetime(2026, 5, 5))


def test_filter_due_after_end_returns_all():
    schedule = slice_twap(_inputs(n_slices=4))
    due = filter_due(schedule, now=datetime(2026, 5, 5, 11, 0, tzinfo=timezone.utc))
    assert len(due) == 4


# --- Render -----------------------------------------------------------------


def test_render_schedule_includes_summary():
    schedule = slice_twap(_inputs(n_slices=4))
    out = render_schedule(schedule)
    assert "P-001" in out
    assert "AAPL" in out
    assert "buy" in out
    assert "4 slices" in out


def test_render_empty_schedule():
    out = render_schedule(())
    assert "empty" in out


def test_render_no_secret_leak():
    schedule = slice_twap(_inputs(n_slices=4))
    out = render_schedule(schedule)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E -------------------------------------------------------------------


def test_e2e_buy_1000_shares_over_one_hour_in_4_slices():
    schedule = slice_twap(_inputs(parent_quantity=1000.0, n_slices=4))
    assert len(schedule) == 4
    assert all(c.quantity == 250.0 for c in schedule)
    assert sum(c.quantity for c in schedule) == 1000.0


def test_replay_consistency():
    a = slice_twap(_inputs())
    b = slice_twap(_inputs())
    assert a == b
