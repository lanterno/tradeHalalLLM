"""Tests for trading/vwap.py — Round-5 Wave 12.B."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.trading.twap import Side
from halal_trader.trading.vwap import (
    VolumeBucket,
    VolumeProfile,
    VwapInputs,
    VwapPolicy,
    cumulative_quantity,
    filter_due,
    render_schedule,
    slice_vwap,
    u_shape_profile,
    uniform_profile,
)


def _bucket_times(n: int = 4, start_minute: int = 30) -> list[datetime]:
    base = datetime(2026, 5, 5, 9, start_minute, tzinfo=timezone.utc)
    return [base + timedelta(minutes=15 * i) for i in range(n)]


# --- Validation -------------------------------------------------------------


def test_volume_bucket_naive_time_rejected():
    with pytest.raises(ValueError):
        VolumeBucket(start_time=datetime(2026, 5, 5), fraction=0.5)


def test_volume_bucket_negative_fraction_rejected():
    with pytest.raises(ValueError):
        VolumeBucket(start_time=datetime(2026, 5, 5, tzinfo=timezone.utc), fraction=-0.1)


def test_volume_bucket_above_one_rejected():
    with pytest.raises(ValueError):
        VolumeBucket(start_time=datetime(2026, 5, 5, tzinfo=timezone.utc), fraction=1.1)


def test_profile_empty_rejected():
    with pytest.raises(ValueError):
        VolumeProfile(buckets=())


def test_profile_unsorted_rejected():
    times = _bucket_times(2)
    with pytest.raises(ValueError):
        VolumeProfile(
            buckets=(
                VolumeBucket(times[1], 0.5),
                VolumeBucket(times[0], 0.5),
            )
        )


def test_profile_does_not_sum_to_one_rejected():
    times = _bucket_times(2)
    with pytest.raises(ValueError):
        VolumeProfile(
            buckets=(
                VolumeBucket(times[0], 0.4),
                VolumeBucket(times[1], 0.4),
            )
        )


def test_profile_sums_to_one_passes():
    times = _bucket_times(2)
    p = VolumeProfile(
        buckets=(
            VolumeBucket(times[0], 0.5),
            VolumeBucket(times[1], 0.5),
        )
    )
    assert len(p) == 2


def test_default_policy():
    p = VwapPolicy()
    assert p.skip_zero_fraction is True
    assert p.max_buckets == 1000


def test_policy_negative_min_rejected():
    with pytest.raises(ValueError):
        VwapPolicy(min_slice_quantity=-1.0)


# --- Profile builders ------------------------------------------------------


def test_uniform_profile_equal_fractions():
    p = uniform_profile(_bucket_times(4))
    assert all(b.fraction == pytest.approx(0.25) for b in p.buckets)


def test_uniform_profile_empty_rejected():
    with pytest.raises(ValueError):
        uniform_profile([])


def test_u_shape_heavier_at_open_and_close():
    p = u_shape_profile(_bucket_times(5))
    # Endpoints heavier than middle
    assert p.buckets[0].fraction > p.buckets[2].fraction
    assert p.buckets[-1].fraction > p.buckets[2].fraction
    # Symmetric
    assert p.buckets[0].fraction == pytest.approx(p.buckets[-1].fraction)


def test_u_shape_single_bucket_full_weight():
    p = u_shape_profile([_bucket_times(1)[0]])
    assert p.buckets[0].fraction == pytest.approx(1.0)


def test_u_shape_empty_rejected():
    with pytest.raises(ValueError):
        u_shape_profile([])


# --- Slicing ---------------------------------------------------------------


def _inputs(profile: VolumeProfile) -> VwapInputs:
    return VwapInputs(
        parent_id="P-001",
        symbol="AAPL",
        side=Side.BUY,
        parent_quantity=1000.0,
        profile=profile,
    )


def test_slice_uniform_distributes_evenly():
    profile = uniform_profile(_bucket_times(4))
    schedule = slice_vwap(_inputs(profile))
    assert len(schedule) == 4
    assert cumulative_quantity(schedule) == pytest.approx(1000.0)


def test_slice_u_shape_endpoints_get_more():
    profile = u_shape_profile(_bucket_times(5))
    schedule = slice_vwap(_inputs(profile))
    # First > middle, last > middle
    qtys = [c.quantity for c in schedule]
    assert qtys[0] > qtys[2]
    assert qtys[-1] > qtys[2]


def test_slice_total_quantity_exact():
    profile = uniform_profile(_bucket_times(4))
    schedule = slice_vwap(_inputs(profile))
    assert cumulative_quantity(schedule) == pytest.approx(1000.0)


def test_slice_skip_zero_fraction_default_skips():
    times = _bucket_times(2)
    # Manually-built profile with one zero bucket
    profile = VolumeProfile(
        buckets=(
            VolumeBucket(times[0], 1.0),
            VolumeBucket(times[1], 0.0),
        )
    )
    schedule = slice_vwap(_inputs(profile))
    assert len(schedule) == 1


def test_slice_too_many_buckets_rejected():
    times = _bucket_times(2)
    profile = VolumeProfile(
        buckets=(
            VolumeBucket(times[0], 0.5),
            VolumeBucket(times[1], 0.5),
        )
    )
    with pytest.raises(ValueError):
        slice_vwap(_inputs(profile), policy=VwapPolicy(max_buckets=1))


def test_slice_min_qty_skips_below_threshold():
    """A bucket whose share is below min is skipped + remainder rebalanced."""
    times = _bucket_times(4)
    profile = VolumeProfile(
        buckets=(
            VolumeBucket(times[0], 0.05),
            VolumeBucket(times[1], 0.05),
            VolumeBucket(times[2], 0.45),
            VolumeBucket(times[3], 0.45),
        )
    )
    # min 100 → first two would be 50 each, rejected; remainder pushed into largest.
    schedule = slice_vwap(_inputs(profile), policy=VwapPolicy(min_slice_quantity=100.0))
    assert len(schedule) == 2
    # Total still equals parent
    assert cumulative_quantity(schedule) == pytest.approx(1000.0)


def test_slice_submit_times_match_bucket_times():
    times = _bucket_times(4)
    profile = uniform_profile(times)
    schedule = slice_vwap(_inputs(profile))
    assert [c.submit_time for c in schedule] == times


# --- Helpers ---------------------------------------------------------------


def test_cumulative_quantity_empty_zero():
    assert cumulative_quantity([]) == 0


def test_filter_due_returns_due_only():
    profile = uniform_profile(_bucket_times(4))
    schedule = slice_vwap(_inputs(profile))
    now = datetime(2026, 5, 5, 9, 50, tzinfo=timezone.utc)
    due = filter_due(schedule, now=now)
    assert len(due) == 2  # 9:30 + 9:45 only


def test_filter_due_naive_now_rejected():
    profile = uniform_profile(_bucket_times(4))
    schedule = slice_vwap(_inputs(profile))
    with pytest.raises(ValueError):
        filter_due(schedule, now=datetime(2026, 5, 5))


# --- Render -----------------------------------------------------------------


def test_render_schedule_includes_summary():
    profile = uniform_profile(_bucket_times(4))
    schedule = slice_vwap(_inputs(profile))
    out = render_schedule(schedule)
    assert "P-001" in out
    assert "AAPL" in out
    assert "buy" in out


def test_render_empty():
    assert "empty" in render_schedule(())


def test_render_no_secret_leak():
    profile = uniform_profile(_bucket_times(4))
    schedule = slice_vwap(_inputs(profile))
    out = render_schedule(schedule)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E -------------------------------------------------------------------


def test_e2e_buy_with_u_shape_market_session():
    profile = u_shape_profile(_bucket_times(7))  # full market session
    schedule = slice_vwap(_inputs(profile))
    assert len(schedule) == 7
    # Round-trip total
    assert cumulative_quantity(schedule) == pytest.approx(1000.0)


def test_replay_consistency():
    profile = uniform_profile(_bucket_times(4))
    a = slice_vwap(_inputs(profile))
    b = slice_vwap(_inputs(profile))
    assert a == b
