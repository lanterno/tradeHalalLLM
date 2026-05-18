"""Tests for the adapter circuit breaker state machine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.core.circuit_breaker import (
    BreakerPolicy,
    BreakerSnapshot,
    BreakerState,
    CallOutcome,
    is_call_allowed,
    record_outcome,
    render_snapshot,
    tick,
    time_until_retry,
)

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


# --- Enum string-value pins ---------------------------------------------------


def test_breaker_state_string_values():
    assert BreakerState.CLOSED.value == "closed"
    assert BreakerState.OPEN.value == "open"
    assert BreakerState.HALF_OPEN.value == "half_open"


def test_call_outcome_string_values():
    assert CallOutcome.SUCCESS.value == "success"
    assert CallOutcome.FAILURE.value == "failure"


# --- Policy validation --------------------------------------------------------


def test_default_policy_pins():
    p = BreakerPolicy()
    assert p.failure_threshold == 5
    assert p.cooldown_seconds == 60.0
    assert p.half_open_probe_count == 2


def test_zero_failure_threshold_rejected():
    with pytest.raises(ValueError, match="failure_threshold"):
        BreakerPolicy(failure_threshold=0)


def test_negative_cooldown_rejected():
    with pytest.raises(ValueError, match="cooldown_seconds"):
        BreakerPolicy(cooldown_seconds=0)


def test_zero_probe_count_rejected():
    with pytest.raises(ValueError, match="half_open_probe_count"):
        BreakerPolicy(half_open_probe_count=0)


def test_policy_immutable():
    p = BreakerPolicy()
    with pytest.raises(Exception):
        p.failure_threshold = 99  # type: ignore[misc]


# --- BreakerSnapshot validation -----------------------------------------------


def test_default_snapshot_is_closed():
    s = BreakerSnapshot()
    assert s.state is BreakerState.CLOSED
    assert s.consecutive_failures == 0
    assert s.opened_at is None
    assert s.half_open_successes == 0


def test_snapshot_immutable():
    s = BreakerSnapshot()
    with pytest.raises(Exception):
        s.consecutive_failures = 5  # type: ignore[misc]


def test_negative_consecutive_failures_rejected():
    with pytest.raises(ValueError, match="consecutive_failures"):
        BreakerSnapshot(consecutive_failures=-1)


def test_negative_half_open_successes_rejected():
    with pytest.raises(ValueError, match="half_open_successes"):
        BreakerSnapshot(half_open_successes=-1)


def test_open_state_requires_opened_at():
    with pytest.raises(ValueError, match="OPEN state requires opened_at"):
        BreakerSnapshot(state=BreakerState.OPEN)


def test_naive_opened_at_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        BreakerSnapshot(state=BreakerState.OPEN, opened_at=datetime(2026, 1, 1))


# --- is_call_allowed ----------------------------------------------------------


def test_call_allowed_when_closed():
    assert is_call_allowed(BreakerSnapshot()) is True


def test_call_allowed_when_half_open():
    s = BreakerSnapshot(state=BreakerState.HALF_OPEN)
    assert is_call_allowed(s) is True


def test_call_rejected_when_open():
    s = BreakerSnapshot(state=BreakerState.OPEN, opened_at=T0)
    assert is_call_allowed(s) is False


# --- record_outcome: CLOSED state ---------------------------------------------


def test_closed_success_resets_failures():
    s = BreakerSnapshot(consecutive_failures=3)
    new = record_outcome(s, CallOutcome.SUCCESS, now=T0)
    assert new.state is BreakerState.CLOSED
    assert new.consecutive_failures == 0


def test_closed_failure_increments():
    s = BreakerSnapshot()
    new = record_outcome(s, CallOutcome.FAILURE, now=T0)
    assert new.state is BreakerState.CLOSED
    assert new.consecutive_failures == 1


def test_closed_failure_at_threshold_opens():
    """Pin: 5th consecutive failure trips the breaker."""
    s = BreakerSnapshot(consecutive_failures=4)
    new = record_outcome(s, CallOutcome.FAILURE, now=T0)
    assert new.state is BreakerState.OPEN
    assert new.consecutive_failures == 5
    assert new.opened_at == T0
    assert new.half_open_successes == 0


def test_closed_failure_below_threshold_stays_closed():
    """Pin: 4th failure doesn't trip a 5-threshold breaker."""
    s = BreakerSnapshot(consecutive_failures=3)
    new = record_outcome(s, CallOutcome.FAILURE, now=T0)
    assert new.state is BreakerState.CLOSED
    assert new.consecutive_failures == 4


def test_custom_threshold():
    policy = BreakerPolicy(failure_threshold=2)
    s = BreakerSnapshot()
    after_one = record_outcome(s, CallOutcome.FAILURE, now=T0, policy=policy)
    assert after_one.state is BreakerState.CLOSED
    after_two = record_outcome(after_one, CallOutcome.FAILURE, now=T0, policy=policy)
    assert after_two.state is BreakerState.OPEN


# --- record_outcome: OPEN state -----------------------------------------------


def test_open_state_outcome_is_noop():
    """Pin: a stray outcome while OPEN is identity (defensive).

    Caller should have respected is_call_allowed; we don't punish.
    """
    s = BreakerSnapshot(state=BreakerState.OPEN, opened_at=T0, consecutive_failures=5)
    new = record_outcome(s, CallOutcome.FAILURE, now=T0 + timedelta(seconds=10))
    assert new == s
    new = record_outcome(s, CallOutcome.SUCCESS, now=T0 + timedelta(seconds=10))
    assert new == s


# --- record_outcome: HALF_OPEN state ------------------------------------------


def test_half_open_first_success_increments():
    s = BreakerSnapshot(state=BreakerState.HALF_OPEN)
    new = record_outcome(s, CallOutcome.SUCCESS, now=T0)
    assert new.state is BreakerState.HALF_OPEN
    assert new.half_open_successes == 1


def test_half_open_probe_count_succeeds_closes():
    """Pin: 2 consecutive successes close a default-policy breaker."""
    s = BreakerSnapshot(state=BreakerState.HALF_OPEN, half_open_successes=1)
    new = record_outcome(s, CallOutcome.SUCCESS, now=T0)
    assert new.state is BreakerState.CLOSED
    assert new.consecutive_failures == 0
    assert new.opened_at is None
    assert new.half_open_successes == 0


def test_half_open_failure_reopens():
    """Pin: a single failure in HALF_OPEN re-opens with fresh cooldown."""
    s = BreakerSnapshot(state=BreakerState.HALF_OPEN, half_open_successes=1)
    later = T0 + timedelta(minutes=5)
    new = record_outcome(s, CallOutcome.FAILURE, now=later)
    assert new.state is BreakerState.OPEN
    assert new.opened_at == later
    assert new.half_open_successes == 0


def test_custom_probe_count():
    policy = BreakerPolicy(half_open_probe_count=3)
    s = BreakerSnapshot(state=BreakerState.HALF_OPEN, half_open_successes=2)
    new = record_outcome(s, CallOutcome.SUCCESS, now=T0, policy=policy)
    assert new.state is BreakerState.CLOSED


def test_custom_probe_count_below_threshold():
    policy = BreakerPolicy(half_open_probe_count=3)
    s = BreakerSnapshot(state=BreakerState.HALF_OPEN, half_open_successes=1)
    new = record_outcome(s, CallOutcome.SUCCESS, now=T0, policy=policy)
    assert new.state is BreakerState.HALF_OPEN
    assert new.half_open_successes == 2


# --- record_outcome: now validation -------------------------------------------


def test_record_outcome_naive_now_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        record_outcome(BreakerSnapshot(), CallOutcome.FAILURE, now=datetime(2026, 1, 1))


# --- tick: cooldown-driven OPEN → HALF_OPEN -----------------------------------


def test_tick_closed_is_noop():
    s = BreakerSnapshot()
    assert tick(s, now=T0) == s


def test_tick_half_open_is_noop():
    s = BreakerSnapshot(state=BreakerState.HALF_OPEN, half_open_successes=1)
    assert tick(s, now=T0) == s


def test_tick_open_before_cooldown_stays_open():
    s = BreakerSnapshot(state=BreakerState.OPEN, opened_at=T0, consecutive_failures=5)
    new = tick(s, now=T0 + timedelta(seconds=30))
    assert new.state is BreakerState.OPEN
    assert new == s


def test_tick_open_at_cooldown_boundary_inclusive():
    """Pin: now - opened_at >= cooldown triggers the transition (inclusive)."""
    s = BreakerSnapshot(state=BreakerState.OPEN, opened_at=T0, consecutive_failures=5)
    new = tick(s, now=T0 + timedelta(seconds=60))
    assert new.state is BreakerState.HALF_OPEN
    assert new.opened_at is None
    assert new.half_open_successes == 0


def test_tick_open_after_cooldown_transitions():
    s = BreakerSnapshot(state=BreakerState.OPEN, opened_at=T0, consecutive_failures=5)
    new = tick(s, now=T0 + timedelta(minutes=5))
    assert new.state is BreakerState.HALF_OPEN


def test_tick_custom_cooldown():
    policy = BreakerPolicy(cooldown_seconds=10.0)
    s = BreakerSnapshot(state=BreakerState.OPEN, opened_at=T0, consecutive_failures=5)
    new = tick(s, now=T0 + timedelta(seconds=10), policy=policy)
    assert new.state is BreakerState.HALF_OPEN


def test_tick_naive_now_rejected():
    s = BreakerSnapshot(state=BreakerState.OPEN, opened_at=T0, consecutive_failures=5)
    with pytest.raises(ValueError, match="timezone-aware"):
        tick(s, now=datetime(2026, 1, 1))


# --- time_until_retry ---------------------------------------------------------


def test_time_until_retry_closed_is_zero():
    assert time_until_retry(BreakerSnapshot(), now=T0) == timedelta(0)


def test_time_until_retry_half_open_is_zero():
    s = BreakerSnapshot(state=BreakerState.HALF_OPEN)
    assert time_until_retry(s, now=T0) == timedelta(0)


def test_time_until_retry_open_returns_remaining():
    s = BreakerSnapshot(state=BreakerState.OPEN, opened_at=T0, consecutive_failures=5)
    remaining = time_until_retry(s, now=T0 + timedelta(seconds=20))
    assert remaining == timedelta(seconds=40)


def test_time_until_retry_after_cooldown_is_zero():
    s = BreakerSnapshot(state=BreakerState.OPEN, opened_at=T0, consecutive_failures=5)
    assert time_until_retry(s, now=T0 + timedelta(seconds=120)) == timedelta(0)


# --- Render -------------------------------------------------------------------


def test_render_closed_clean():
    out = render_snapshot(BreakerSnapshot(), name="alpaca")
    assert "🟢" in out
    assert "alpaca" in out
    assert "closed" in out


def test_render_closed_with_recent_failures():
    s = BreakerSnapshot(consecutive_failures=2)
    out = render_snapshot(s, name="alpaca")
    assert "2" in out
    assert "failures" in out


def test_render_open_shows_retry_eta():
    s = BreakerSnapshot(state=BreakerState.OPEN, opened_at=T0, consecutive_failures=5)
    out = render_snapshot(s, name="binance", now=T0 + timedelta(seconds=20))
    assert "🔴" in out
    assert "binance" in out
    assert "retry" in out
    assert "40" in out


def test_render_open_at_cooldown_says_ready():
    s = BreakerSnapshot(state=BreakerState.OPEN, opened_at=T0, consecutive_failures=5)
    out = render_snapshot(s, name="binance", now=T0 + timedelta(seconds=70))
    assert "ready" in out


def test_render_half_open_shows_probe_progress():
    s = BreakerSnapshot(state=BreakerState.HALF_OPEN, half_open_successes=1)
    out = render_snapshot(s, name="zoya")
    assert "🟡" in out
    assert "1/2" in out


def test_render_no_secret_leak():
    """Pin: render output never includes adapter call args / responses."""
    s = BreakerSnapshot(state=BreakerState.OPEN, opened_at=T0, consecutive_failures=5)
    # The breaker doesn't carry these fields; no possible leak.
    out = render_snapshot(s, name="alpaca", now=T0 + timedelta(seconds=10))
    forbidden = ["sk_live", "password", "token", "Authorization", "Bearer"]
    for word in forbidden:
        assert word not in out


# --- E2E flows ----------------------------------------------------------------


def test_e2e_full_recovery_cycle():
    """5 failures → OPEN → cooldown → HALF_OPEN → 2 successes → CLOSED."""
    s = BreakerSnapshot()
    # 5 failures trip the breaker
    for i in range(5):
        s = record_outcome(s, CallOutcome.FAILURE, now=T0 + timedelta(seconds=i))
    assert s.state is BreakerState.OPEN
    assert is_call_allowed(s) is False

    # Cooldown elapses
    s = tick(s, now=T0 + timedelta(seconds=70))
    assert s.state is BreakerState.HALF_OPEN
    assert is_call_allowed(s) is True

    # Two successful probes close it
    s = record_outcome(s, CallOutcome.SUCCESS, now=T0 + timedelta(seconds=71))
    assert s.state is BreakerState.HALF_OPEN
    s = record_outcome(s, CallOutcome.SUCCESS, now=T0 + timedelta(seconds=72))
    assert s.state is BreakerState.CLOSED
    assert s.consecutive_failures == 0


def test_e2e_failed_recovery_reopens():
    """5 failures → OPEN → cooldown → HALF_OPEN → 1 failure → OPEN with fresh cooldown."""
    s = BreakerSnapshot()
    for i in range(5):
        s = record_outcome(s, CallOutcome.FAILURE, now=T0 + timedelta(seconds=i))
    s = tick(s, now=T0 + timedelta(seconds=70))
    assert s.state is BreakerState.HALF_OPEN

    # Probe fails — back to OPEN with fresh cooldown
    later = T0 + timedelta(seconds=71)
    s = record_outcome(s, CallOutcome.FAILURE, now=later)
    assert s.state is BreakerState.OPEN
    assert s.opened_at == later

    # 60s after the second open, not the first
    s_early = tick(s, now=later + timedelta(seconds=30))
    assert s_early.state is BreakerState.OPEN
    s_late = tick(s, now=later + timedelta(seconds=60))
    assert s_late.state is BreakerState.HALF_OPEN


def test_e2e_intermittent_failures_dont_trip():
    """Pin: success resets the failure count; intermittent ≠ outage."""
    s = BreakerSnapshot()
    # 3 failures, 1 success, 3 more failures = breaker holds at 3
    for i in range(3):
        s = record_outcome(s, CallOutcome.FAILURE, now=T0 + timedelta(seconds=i))
    assert s.consecutive_failures == 3
    s = record_outcome(s, CallOutcome.SUCCESS, now=T0 + timedelta(seconds=3))
    assert s.consecutive_failures == 0
    for i in range(3):
        s = record_outcome(s, CallOutcome.FAILURE, now=T0 + timedelta(seconds=4 + i))
    assert s.state is BreakerState.CLOSED
    assert s.consecutive_failures == 3


def test_e2e_replay_consistency():
    """Pin: same operations → equal final state."""

    def run() -> BreakerSnapshot:
        s = BreakerSnapshot()
        for i in range(5):
            s = record_outcome(s, CallOutcome.FAILURE, now=T0 + timedelta(seconds=i))
        s = tick(s, now=T0 + timedelta(seconds=80))
        s = record_outcome(s, CallOutcome.SUCCESS, now=T0 + timedelta(seconds=81))
        s = record_outcome(s, CallOutcome.SUCCESS, now=T0 + timedelta(seconds=82))
        return s

    assert run() == run()
