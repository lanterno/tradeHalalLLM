"""Tests for trading/anti_frontrun.py — Round-5 Wave 12.G."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.trading.anti_frontrun import (
    FrontrunPolicy,
    FrontrunRisk,
    FrontrunSignal,
    Mitigation,
    OrderSignal,
    assess,
    render_assessment,
)


def _times(n: int = 5, period: timedelta = timedelta(seconds=15)) -> tuple[datetime, ...]:
    start = datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc)
    return tuple(start + period * i for i in range(n))


def _signal(**overrides) -> OrderSignal:
    base = {
        "parent_id": "P-001",
        "submission_times": _times(5),
        "parent_quantity": 100.0,
        "recent_volume": 100000.0,
        "venue_is_public_mempool": False,
        "counterparty_repeat_count": 0,
        "same_block_neighbours": 0,
    }
    base.update(overrides)
    return OrderSignal(**base)


# --- Validation ----------------------------------------------------------


def test_risk_string_values():
    assert FrontrunRisk.LOW.value == "low"
    assert FrontrunRisk.MEDIUM.value == "medium"
    assert FrontrunRisk.HIGH.value == "high"
    assert FrontrunRisk.CRITICAL.value == "critical"


def test_signal_string_values():
    assert FrontrunSignal.PREDICTABLE_CADENCE.value == "predictable_cadence"
    assert FrontrunSignal.LARGE_RELATIVE_SIZE.value == "large_relative_size"
    assert FrontrunSignal.PUBLIC_MEMPOOL.value == "public_mempool"
    assert FrontrunSignal.REPEAT_COUNTERPARTY.value == "repeat_counterparty"
    assert FrontrunSignal.SAME_BLOCK_COLLISION.value == "same_block_collision"


def test_mitigation_string_values():
    assert Mitigation.PRIVATE_MEMPOOL.value == "private_mempool"
    assert Mitigation.RANDOMISED_CADENCE.value == "randomised_cadence"
    assert Mitigation.BATCHED_WITH_PEERS.value == "batched_with_peers"
    assert Mitigation.SLICED_SMALLER.value == "sliced_smaller"


def test_default_policy():
    p = FrontrunPolicy()
    assert p.cadence_cv_threshold == 0.10
    assert p.large_size_pct_of_volume == 0.05
    assert p.repeat_counterparty_threshold == 3


def test_policy_zero_cv_rejected():
    with pytest.raises(ValueError):
        FrontrunPolicy(cadence_cv_threshold=0.0)


def test_policy_zero_size_pct_rejected():
    with pytest.raises(ValueError):
        FrontrunPolicy(large_size_pct_of_volume=0.0)


def test_policy_zero_counterparty_rejected():
    with pytest.raises(ValueError):
        FrontrunPolicy(repeat_counterparty_threshold=0)


def test_signal_empty_id_rejected():
    with pytest.raises(ValueError):
        _signal(parent_id="")


def test_signal_naive_time_rejected():
    with pytest.raises(ValueError):
        OrderSignal(
            parent_id="P",
            submission_times=(datetime(2026, 5, 5),),
            parent_quantity=10,
            recent_volume=100,
            venue_is_public_mempool=False,
        )


def test_signal_negative_volume_rejected():
    with pytest.raises(ValueError):
        _signal(recent_volume=-1)


def test_signal_negative_qty_rejected():
    with pytest.raises(ValueError):
        _signal(parent_quantity=-1)


# --- Detection ---------------------------------------------------------


def test_clean_order_low_risk():
    """Random irregular submissions, small size, private venue → LOW risk."""
    times = (
        datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
        datetime(2026, 5, 5, 9, 33, tzinfo=timezone.utc),
        datetime(2026, 5, 5, 9, 41, tzinfo=timezone.utc),
        datetime(2026, 5, 5, 9, 50, tzinfo=timezone.utc),
    )
    a = assess(_signal(submission_times=times))
    assert a.risk is FrontrunRisk.LOW
    assert a.signals == frozenset()


_IRREGULAR_TIMES = (
    datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
    datetime(2026, 5, 5, 9, 33, tzinfo=timezone.utc),
    datetime(2026, 5, 5, 9, 41, tzinfo=timezone.utc),
    datetime(2026, 5, 5, 9, 50, tzinfo=timezone.utc),
)


def test_predictable_cadence_detected():
    """Exactly-periodic submissions trigger predictable-cadence."""
    a = assess(_signal(submission_times=_times(5, timedelta(seconds=10))))
    assert FrontrunSignal.PREDICTABLE_CADENCE in a.signals
    assert Mitigation.RANDOMISED_CADENCE in a.recommended_mitigations


def test_large_size_detected():
    a = assess(
        _signal(submission_times=_IRREGULAR_TIMES, parent_quantity=10000, recent_volume=100000)
    )
    assert FrontrunSignal.LARGE_RELATIVE_SIZE in a.signals
    assert Mitigation.SLICED_SMALLER in a.recommended_mitigations


def test_public_mempool_detected():
    a = assess(_signal(submission_times=_IRREGULAR_TIMES, venue_is_public_mempool=True))
    assert FrontrunSignal.PUBLIC_MEMPOOL in a.signals
    assert Mitigation.PRIVATE_MEMPOOL in a.recommended_mitigations


def test_repeat_counterparty_detected():
    a = assess(_signal(submission_times=_IRREGULAR_TIMES, counterparty_repeat_count=5))
    assert FrontrunSignal.REPEAT_COUNTERPARTY in a.signals
    assert Mitigation.BATCHED_WITH_PEERS in a.recommended_mitigations


def test_same_block_collision_detected():
    a = assess(_signal(submission_times=_IRREGULAR_TIMES, same_block_neighbours=2))
    assert FrontrunSignal.SAME_BLOCK_COLLISION in a.signals
    assert Mitigation.RANDOMISED_CADENCE in a.recommended_mitigations


def test_below_threshold_repeat_not_detected():
    a = assess(_signal(submission_times=_IRREGULAR_TIMES, counterparty_repeat_count=2))
    assert FrontrunSignal.REPEAT_COUNTERPARTY not in a.signals


# --- Risk laddering ---------------------------------------------------


def test_one_signal_medium_risk():
    """Public mempool only — 1 signal → MEDIUM. Use irregular times to avoid cadence flag."""
    irregular = (
        datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
        datetime(2026, 5, 5, 9, 33, tzinfo=timezone.utc),
        datetime(2026, 5, 5, 9, 41, tzinfo=timezone.utc),
        datetime(2026, 5, 5, 9, 50, tzinfo=timezone.utc),
    )
    a = assess(_signal(submission_times=irregular, venue_is_public_mempool=True))
    assert a.risk is FrontrunRisk.MEDIUM


def test_two_signals_high_risk():
    irregular = (
        datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
        datetime(2026, 5, 5, 9, 33, tzinfo=timezone.utc),
        datetime(2026, 5, 5, 9, 41, tzinfo=timezone.utc),
        datetime(2026, 5, 5, 9, 50, tzinfo=timezone.utc),
    )
    a = assess(
        _signal(
            submission_times=irregular,
            venue_is_public_mempool=True,
            parent_quantity=10000,
            recent_volume=100000,
        )
    )
    assert a.risk is FrontrunRisk.HIGH


def test_four_signals_critical():
    times = _times(5, timedelta(seconds=10))  # predictable
    a = assess(
        _signal(
            submission_times=times,
            parent_quantity=10000,
            recent_volume=100000,
            venue_is_public_mempool=True,
            counterparty_repeat_count=5,
        )
    )
    assert a.risk is FrontrunRisk.CRITICAL


# --- Cadence edge ----------------------------------------------------


def test_too_few_submissions_no_cadence_signal():
    """With fewer than 3 timestamps, predictable-cadence cannot be flagged."""
    times = _times(2, timedelta(seconds=10))
    a = assess(_signal(submission_times=times))
    assert FrontrunSignal.PREDICTABLE_CADENCE not in a.signals


# --- Render --------------------------------------------------------


def test_render_includes_risk():
    a = assess(_signal(submission_times=_IRREGULAR_TIMES, venue_is_public_mempool=True))
    out = render_assessment(a)
    assert "risk=medium" in out


def test_render_lists_signals_and_mitigations():
    a = assess(_signal(submission_times=_IRREGULAR_TIMES, venue_is_public_mempool=True))
    out = render_assessment(a)
    assert "public_mempool" in out
    assert "private_mempool" in out


def test_render_no_secret_leak():
    a = assess(_signal())
    out = render_assessment(a)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ------------------------------------------------------


def test_e2e_dex_swap_flagged_critical():
    """A large DEX swap on public mempool with repeat counterparties → CRITICAL."""
    a = assess(
        _signal(
            submission_times=_times(5, timedelta(seconds=15)),  # predictable
            parent_quantity=50000,
            recent_volume=200000,  # 25%
            venue_is_public_mempool=True,
            counterparty_repeat_count=10,
            same_block_neighbours=3,
        )
    )
    assert a.risk is FrontrunRisk.CRITICAL


def test_replay_consistency():
    a = assess(_signal())
    b = assess(_signal())
    assert a == b
