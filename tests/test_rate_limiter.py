"""Tests for the token bucket rate limiter."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.core.rate_limiter import (
    BucketPolicy,
    BucketSnapshot,
    ConsumeOutcome,
    fill_ratio,
    full_bucket,
    refill,
    render_snapshot,
    time_until_available,
    try_consume,
)

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
DEFAULT_POLICY = BucketPolicy(capacity=10.0, refill_rate_per_sec=1.0)


# --- Enum string-value pins ---------------------------------------------------


def test_consume_outcome_string_values():
    assert ConsumeOutcome.ALLOWED.value == "allowed"
    assert ConsumeOutcome.DENIED_INSUFFICIENT.value == "denied_insufficient"
    assert ConsumeOutcome.DENIED_OVERSIZED.value == "denied_oversized"


# --- Policy validation --------------------------------------------------------


def test_policy_basic():
    p = BucketPolicy(capacity=100, refill_rate_per_sec=10)
    assert p.capacity == 100
    assert p.refill_rate_per_sec == 10


def test_zero_capacity_rejected():
    with pytest.raises(ValueError, match="capacity"):
        BucketPolicy(capacity=0, refill_rate_per_sec=1)


def test_negative_capacity_rejected():
    with pytest.raises(ValueError, match="capacity"):
        BucketPolicy(capacity=-1, refill_rate_per_sec=1)


def test_zero_refill_rate_rejected():
    with pytest.raises(ValueError, match="refill_rate_per_sec"):
        BucketPolicy(capacity=10, refill_rate_per_sec=0)


def test_negative_refill_rate_rejected():
    with pytest.raises(ValueError, match="refill_rate_per_sec"):
        BucketPolicy(capacity=10, refill_rate_per_sec=-1)


def test_infinite_capacity_rejected():
    with pytest.raises(ValueError, match="capacity"):
        BucketPolicy(capacity=math.inf, refill_rate_per_sec=1)


def test_nan_refill_rate_rejected():
    with pytest.raises(ValueError, match="refill_rate_per_sec"):
        BucketPolicy(capacity=10, refill_rate_per_sec=math.nan)


def test_policy_immutable():
    p = BucketPolicy(capacity=10, refill_rate_per_sec=1)
    with pytest.raises(Exception):
        p.capacity = 99  # type: ignore[misc]


# --- BucketSnapshot validation ------------------------------------------------


def test_snapshot_basic():
    s = BucketSnapshot(tokens=5.0, last_refill_at=T0)
    assert s.tokens == 5.0


def test_snapshot_immutable():
    s = BucketSnapshot(tokens=5.0, last_refill_at=T0)
    with pytest.raises(Exception):
        s.tokens = 99  # type: ignore[misc]


def test_negative_tokens_rejected():
    with pytest.raises(ValueError, match="tokens"):
        BucketSnapshot(tokens=-0.5, last_refill_at=T0)


def test_infinite_tokens_rejected():
    with pytest.raises(ValueError, match="tokens"):
        BucketSnapshot(tokens=math.inf, last_refill_at=T0)


def test_naive_last_refill_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        BucketSnapshot(tokens=5.0, last_refill_at=datetime(2026, 1, 1))


# --- full_bucket --------------------------------------------------------------


def test_full_bucket_starts_at_capacity():
    s = full_bucket(now=T0, policy=DEFAULT_POLICY)
    assert s.tokens == DEFAULT_POLICY.capacity
    assert s.last_refill_at == T0


def test_full_bucket_naive_now_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        full_bucket(now=datetime(2026, 1, 1), policy=DEFAULT_POLICY)


# --- refill -------------------------------------------------------------------


def test_refill_adds_proportional_tokens():
    s = BucketSnapshot(tokens=2.0, last_refill_at=T0)
    new = refill(s, now=T0 + timedelta(seconds=3), policy=DEFAULT_POLICY)
    assert new.tokens == pytest.approx(5.0)
    assert new.last_refill_at == T0 + timedelta(seconds=3)


def test_refill_caps_at_capacity():
    """Pin: tokens never exceed capacity."""
    s = BucketSnapshot(tokens=8.0, last_refill_at=T0)
    new = refill(s, now=T0 + timedelta(seconds=100), policy=DEFAULT_POLICY)
    assert new.tokens == 10.0  # capacity


def test_refill_zero_elapsed_is_noop_on_tokens():
    s = BucketSnapshot(tokens=3.0, last_refill_at=T0)
    new = refill(s, now=T0, policy=DEFAULT_POLICY)
    assert new.tokens == 3.0


def test_refill_fractional_rate():
    """Pin: float refill rates work for sub-second cadence."""
    policy = BucketPolicy(capacity=10, refill_rate_per_sec=0.5)
    s = BucketSnapshot(tokens=0.0, last_refill_at=T0)
    new = refill(s, now=T0 + timedelta(seconds=3), policy=policy)
    assert new.tokens == pytest.approx(1.5)


def test_refill_naive_now_rejected():
    s = BucketSnapshot(tokens=5.0, last_refill_at=T0)
    with pytest.raises(ValueError, match="timezone-aware"):
        refill(s, now=datetime(2026, 1, 1), policy=DEFAULT_POLICY)


def test_refill_backwards_clock_rejected():
    """Pin: refill rejects now < last_refill_at."""
    s = BucketSnapshot(tokens=5.0, last_refill_at=T0)
    with pytest.raises(ValueError, match="clock went backwards"):
        refill(s, now=T0 - timedelta(seconds=1), policy=DEFAULT_POLICY)


def test_refill_advances_last_refill_at():
    s = BucketSnapshot(tokens=2.0, last_refill_at=T0)
    new = refill(s, now=T0 + timedelta(seconds=5), policy=DEFAULT_POLICY)
    assert new.last_refill_at == T0 + timedelta(seconds=5)


# --- try_consume --------------------------------------------------------------


def test_consume_allowed_when_sufficient():
    s = BucketSnapshot(tokens=5.0, last_refill_at=T0)
    new, outcome = try_consume(s, 1.0, now=T0, policy=DEFAULT_POLICY)
    assert outcome is ConsumeOutcome.ALLOWED
    assert new.tokens == 4.0


def test_consume_denied_when_insufficient():
    """Pin: insufficient tokens → DENIED_INSUFFICIENT, no spend."""
    s = BucketSnapshot(tokens=2.0, last_refill_at=T0)
    new, outcome = try_consume(s, 5.0, now=T0, policy=DEFAULT_POLICY)
    assert outcome is ConsumeOutcome.DENIED_INSUFFICIENT
    # snapshot reflects refill (zero here) but no spend
    assert new.tokens == 2.0


def test_consume_refills_before_spending():
    """Pin: try_consume is atomic refill+spend."""
    s = BucketSnapshot(tokens=0.0, last_refill_at=T0)
    new, outcome = try_consume(s, 3.0, now=T0 + timedelta(seconds=5), policy=DEFAULT_POLICY)
    # 5 seconds of 1/s refill = 5 tokens, then spend 3
    assert outcome is ConsumeOutcome.ALLOWED
    assert new.tokens == pytest.approx(2.0)


def test_consume_oversized_request_denied():
    """Pin: n > capacity → DENIED_OVERSIZED."""
    s = BucketSnapshot(tokens=10.0, last_refill_at=T0)
    new, outcome = try_consume(s, 11.0, now=T0, policy=DEFAULT_POLICY)
    assert outcome is ConsumeOutcome.DENIED_OVERSIZED
    # No spend even at full capacity
    assert new.tokens == 10.0


def test_consume_exactly_at_capacity_allowed():
    """Pin: n == capacity is allowed when bucket is full."""
    s = BucketSnapshot(tokens=10.0, last_refill_at=T0)
    new, outcome = try_consume(s, 10.0, now=T0, policy=DEFAULT_POLICY)
    assert outcome is ConsumeOutcome.ALLOWED
    assert new.tokens == 0.0


def test_consume_zero_n_rejected():
    s = BucketSnapshot(tokens=5.0, last_refill_at=T0)
    with pytest.raises(ValueError, match="n"):
        try_consume(s, 0.0, now=T0, policy=DEFAULT_POLICY)


def test_consume_negative_n_rejected():
    s = BucketSnapshot(tokens=5.0, last_refill_at=T0)
    with pytest.raises(ValueError, match="n"):
        try_consume(s, -1.0, now=T0, policy=DEFAULT_POLICY)


def test_consume_infinite_n_rejected():
    s = BucketSnapshot(tokens=5.0, last_refill_at=T0)
    with pytest.raises(ValueError, match="n"):
        try_consume(s, math.inf, now=T0, policy=DEFAULT_POLICY)


def test_consume_fractional_cost():
    """Pin: float costs supported (cost-weighted calls)."""
    s = BucketSnapshot(tokens=5.0, last_refill_at=T0)
    new, outcome = try_consume(s, 0.5, now=T0, policy=DEFAULT_POLICY)
    assert outcome is ConsumeOutcome.ALLOWED
    assert new.tokens == 4.5


def test_consume_advances_last_refill_at():
    s = BucketSnapshot(tokens=5.0, last_refill_at=T0)
    new, _ = try_consume(s, 1.0, now=T0 + timedelta(seconds=2), policy=DEFAULT_POLICY)
    assert new.last_refill_at == T0 + timedelta(seconds=2)


def test_consume_denied_still_advances_last_refill_at():
    """Pin: even on DENIED_INSUFFICIENT, last_refill_at advances."""
    s = BucketSnapshot(tokens=0.0, last_refill_at=T0)
    new, outcome = try_consume(s, 5.0, now=T0 + timedelta(seconds=2), policy=DEFAULT_POLICY)
    # 2 sec * 1/s = 2 tokens; need 5, denied
    assert outcome is ConsumeOutcome.DENIED_INSUFFICIENT
    assert new.last_refill_at == T0 + timedelta(seconds=2)
    assert new.tokens == pytest.approx(2.0)


# --- time_until_available -----------------------------------------------------


def test_time_until_zero_when_available():
    s = BucketSnapshot(tokens=5.0, last_refill_at=T0)
    assert time_until_available(s, 1.0, now=T0, policy=DEFAULT_POLICY) == timedelta(0)


def test_time_until_proportional_to_deficit():
    s = BucketSnapshot(tokens=2.0, last_refill_at=T0)
    # need 5; have 2; deficit 3 at 1/s = 3 seconds
    assert time_until_available(s, 5.0, now=T0, policy=DEFAULT_POLICY) == timedelta(seconds=3)


def test_time_until_with_partial_refill():
    s = BucketSnapshot(tokens=0.0, last_refill_at=T0)
    # at T0+2: refilled to 2 tokens; need 5; deficit 3 at 1/s = 3 sec
    result = time_until_available(s, 5.0, now=T0 + timedelta(seconds=2), policy=DEFAULT_POLICY)
    assert result == timedelta(seconds=3)


def test_time_until_oversized_returns_max():
    """Pin: n > capacity → never available, returns timedelta.max."""
    s = BucketSnapshot(tokens=10.0, last_refill_at=T0)
    assert time_until_available(s, 11.0, now=T0, policy=DEFAULT_POLICY) == timedelta.max


def test_time_until_zero_n_rejected():
    s = BucketSnapshot(tokens=5.0, last_refill_at=T0)
    with pytest.raises(ValueError, match="n"):
        time_until_available(s, 0.0, now=T0, policy=DEFAULT_POLICY)


def test_time_until_fractional_rate():
    policy = BucketPolicy(capacity=10, refill_rate_per_sec=0.5)
    s = BucketSnapshot(tokens=0.0, last_refill_at=T0)
    # need 1 at 0.5/s = 2 sec
    assert time_until_available(s, 1.0, now=T0, policy=policy) == timedelta(seconds=2)


# --- fill_ratio ---------------------------------------------------------------


def test_fill_ratio_full():
    s = BucketSnapshot(tokens=10.0, last_refill_at=T0)
    assert fill_ratio(s, now=T0, policy=DEFAULT_POLICY) == 1.0


def test_fill_ratio_empty():
    s = BucketSnapshot(tokens=0.0, last_refill_at=T0)
    assert fill_ratio(s, now=T0, policy=DEFAULT_POLICY) == 0.0


def test_fill_ratio_after_refill():
    s = BucketSnapshot(tokens=0.0, last_refill_at=T0)
    # at T0+5: 5 tokens / 10 capacity = 0.5
    assert fill_ratio(s, now=T0 + timedelta(seconds=5), policy=DEFAULT_POLICY) == 0.5


# --- Render -------------------------------------------------------------------


def test_render_full_is_green():
    s = BucketSnapshot(tokens=10.0, last_refill_at=T0)
    out = render_snapshot(s, name="alpaca", now=T0, policy=DEFAULT_POLICY)
    assert "🟢" in out
    assert "alpaca" in out
    assert "10.0/10.0" in out


def test_render_low_is_yellow():
    s = BucketSnapshot(tokens=3.0, last_refill_at=T0)
    out = render_snapshot(s, name="alpaca", now=T0, policy=DEFAULT_POLICY)
    assert "🟡" in out


def test_render_critical_is_red():
    s = BucketSnapshot(tokens=1.0, last_refill_at=T0)
    out = render_snapshot(s, name="alpaca", now=T0, policy=DEFAULT_POLICY)
    assert "🔴" in out


def test_render_includes_refill_rate():
    s = BucketSnapshot(tokens=5.0, last_refill_at=T0)
    out = render_snapshot(s, name="alpaca", now=T0, policy=DEFAULT_POLICY)
    assert "1.00/s" in out


def test_render_no_secret_leak():
    """Pin: render output never includes adapter call details."""
    s = BucketSnapshot(tokens=5.0, last_refill_at=T0)
    out = render_snapshot(s, name="alpaca", now=T0, policy=DEFAULT_POLICY)
    forbidden = ["sk_live", "password", "token=", "Authorization", "Bearer", "secret"]
    for word in forbidden:
        assert word not in out


# --- E2E flows ----------------------------------------------------------------


def test_e2e_steady_drain_and_recover():
    """Spend at rate >= refill: bucket drains; pause: bucket recovers."""
    policy = BucketPolicy(capacity=5, refill_rate_per_sec=1)
    s = full_bucket(now=T0, policy=policy)

    # Burst 5 calls back-to-back at T0
    for i in range(5):
        s, outcome = try_consume(s, 1.0, now=T0, policy=policy)
        assert outcome is ConsumeOutcome.ALLOWED
    assert s.tokens == 0.0

    # 6th immediate call denied
    s, outcome = try_consume(s, 1.0, now=T0, policy=policy)
    assert outcome is ConsumeOutcome.DENIED_INSUFFICIENT

    # 1 second later: 1 token refilled; can spend
    s, outcome = try_consume(s, 1.0, now=T0 + timedelta(seconds=1), policy=policy)
    assert outcome is ConsumeOutcome.ALLOWED
    assert s.tokens == pytest.approx(0.0)

    # Long pause refills to capacity
    s = refill(s, now=T0 + timedelta(seconds=100), policy=policy)
    assert s.tokens == 5.0


def test_e2e_oversized_request_never_succeeds():
    """Pin: n > capacity is permanently impossible regardless of refill."""
    policy = BucketPolicy(capacity=5, refill_rate_per_sec=1000)  # very fast refill
    s = full_bucket(now=T0, policy=policy)
    s, outcome = try_consume(s, 10.0, now=T0 + timedelta(seconds=100), policy=policy)
    assert outcome is ConsumeOutcome.DENIED_OVERSIZED

    # time_until_available returns max
    assert time_until_available(s, 10.0, now=T0, policy=policy) == timedelta.max


def test_e2e_caller_uses_time_until_to_back_off():
    """Operator pattern: deny → consult time_until → wait → retry."""
    policy = BucketPolicy(capacity=10, refill_rate_per_sec=1)
    s = BucketSnapshot(tokens=2.0, last_refill_at=T0)

    # First attempt denied
    new, outcome = try_consume(s, 5.0, now=T0, policy=policy)
    assert outcome is ConsumeOutcome.DENIED_INSUFFICIENT
    wait = time_until_available(new, 5.0, now=T0, policy=policy)
    assert wait == timedelta(seconds=3)

    # Wait then retry succeeds
    later = T0 + timedelta(seconds=3)
    s2, outcome2 = try_consume(new, 5.0, now=later, policy=policy)
    assert outcome2 is ConsumeOutcome.ALLOWED
    assert s2.tokens == pytest.approx(0.0)


def test_e2e_replay_consistency():
    """Pin: same operations → equal final snapshot."""

    def run() -> BucketSnapshot:
        s = full_bucket(now=T0, policy=DEFAULT_POLICY)
        s, _ = try_consume(s, 3.0, now=T0, policy=DEFAULT_POLICY)
        s, _ = try_consume(s, 2.0, now=T0 + timedelta(seconds=1), policy=DEFAULT_POLICY)
        s = refill(s, now=T0 + timedelta(seconds=10), policy=DEFAULT_POLICY)
        return s

    a = run()
    b = run()
    assert a == b
