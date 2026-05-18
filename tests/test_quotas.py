"""Tests for the per-user resource quota engine."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.web.quotas import (
    DEFAULT_TIER_LIMITS,
    QuotaCheckResult,
    QuotaExceededError,
    QuotaState,
    QuotaTracker,
    ResourceKind,
    ResourceUsage,
    Tier,
    TierLimits,
    render_quota_check,
)

_NOW = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)


def _tracker(now: datetime = _NOW) -> QuotaTracker:
    return QuotaTracker(now_fn=lambda: now)


def _usage(
    *,
    user_id: str = "user-1",
    tier: Tier = Tier.PRO,
    window_started_at: datetime = _NOW,
    llm_usd_used: float = 0.0,
    llm_tokens_used: int = 0,
    broker_api_calls_used: int = 0,
    screener_api_calls_used: int = 0,
    cycle_runs_used: int = 0,
) -> ResourceUsage:
    return ResourceUsage(
        user_id=user_id,
        tier=tier,
        window_started_at=window_started_at,
        llm_usd_used=llm_usd_used,
        llm_tokens_used=llm_tokens_used,
        broker_api_calls_used=broker_api_calls_used,
        screener_api_calls_used=screener_api_calls_used,
        cycle_runs_used=cycle_runs_used,
    )


# ---------------------------------------------------------------------------
# Default tier limits sanity
# ---------------------------------------------------------------------------


def test_default_tier_limits_present_for_every_tier() -> None:
    for tier in Tier:
        assert tier in DEFAULT_TIER_LIMITS


def test_default_free_tier_disables_broker_api() -> None:
    assert DEFAULT_TIER_LIMITS[Tier.FREE].broker_api_calls_daily == 0


def test_default_pro_tier_higher_than_free() -> None:
    free = DEFAULT_TIER_LIMITS[Tier.FREE]
    pro = DEFAULT_TIER_LIMITS[Tier.PRO]
    assert pro.llm_usd_daily > free.llm_usd_daily
    assert pro.llm_tokens_daily > free.llm_tokens_daily
    assert pro.cycle_runs_daily > free.cycle_runs_daily


def test_default_enterprise_tier_higher_than_pro() -> None:
    pro = DEFAULT_TIER_LIMITS[Tier.PRO]
    ent = DEFAULT_TIER_LIMITS[Tier.ENTERPRISE]
    assert ent.llm_usd_daily > pro.llm_usd_daily
    assert ent.cycle_runs_daily > pro.cycle_runs_daily


# ---------------------------------------------------------------------------
# TierLimits validation
# ---------------------------------------------------------------------------


def _basic_tier_kwargs() -> dict:
    return dict(
        tier=Tier.PRO,
        llm_usd_daily=10.0,
        llm_tokens_daily=2_000_000,
        broker_api_calls_daily=10_000,
        screener_api_calls_daily=2_000,
        cycle_runs_daily=288,
    )


def test_tier_limits_rejects_negative_usd() -> None:
    kw = _basic_tier_kwargs()
    kw["llm_usd_daily"] = -1.0
    with pytest.raises(ValueError, match="llm_usd_daily"):
        TierLimits(**kw)


def test_tier_limits_rejects_negative_tokens() -> None:
    kw = _basic_tier_kwargs()
    kw["llm_tokens_daily"] = -1
    with pytest.raises(ValueError, match="llm_tokens_daily"):
        TierLimits(**kw)


def test_tier_limits_rejects_negative_broker_calls() -> None:
    kw = _basic_tier_kwargs()
    kw["broker_api_calls_daily"] = -1
    with pytest.raises(ValueError, match="broker_api_calls_daily"):
        TierLimits(**kw)


def test_tier_limits_accepts_zero_for_disable() -> None:
    """Pin: zero is valid (FREE tier disables broker_api_calls)."""

    kw = _basic_tier_kwargs()
    kw["broker_api_calls_daily"] = 0
    limits = TierLimits(**kw)
    assert limits.broker_api_calls_daily == 0


def test_tier_limits_for_resource_returns_correct_value() -> None:
    limits = TierLimits(**_basic_tier_kwargs())
    assert limits.for_resource(ResourceKind.LLM_USD) == 10.0
    assert limits.for_resource(ResourceKind.LLM_TOKENS) == 2_000_000.0
    assert limits.for_resource(ResourceKind.BROKER_API_CALLS) == 10_000.0
    assert limits.for_resource(ResourceKind.SCREENER_API_CALLS) == 2_000.0
    assert limits.for_resource(ResourceKind.CYCLE_RUNS) == 288.0


# ---------------------------------------------------------------------------
# ResourceUsage validation
# ---------------------------------------------------------------------------


def test_usage_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        _usage(user_id="")


def test_usage_rejects_naive_window() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _usage(window_started_at=datetime(2026, 5, 1))


def test_usage_rejects_negative_usd() -> None:
    with pytest.raises(ValueError, match="llm_usd_used"):
        _usage(llm_usd_used=-1.0)


def test_usage_rejects_negative_tokens() -> None:
    with pytest.raises(ValueError, match="llm_tokens_used"):
        _usage(llm_tokens_used=-1)


def test_usage_used_for_returns_field_value() -> None:
    u = _usage(
        llm_usd_used=2.5,
        llm_tokens_used=1000,
        broker_api_calls_used=50,
        screener_api_calls_used=10,
        cycle_runs_used=3,
    )
    assert u.used_for(ResourceKind.LLM_USD) == 2.5
    assert u.used_for(ResourceKind.LLM_TOKENS) == 1000.0
    assert u.used_for(ResourceKind.BROKER_API_CALLS) == 50.0
    assert u.used_for(ResourceKind.SCREENER_API_CALLS) == 10.0
    assert u.used_for(ResourceKind.CYCLE_RUNS) == 3.0


# ---------------------------------------------------------------------------
# QuotaTracker construction
# ---------------------------------------------------------------------------


def test_tracker_uses_default_limits_when_none_passed() -> None:
    t = QuotaTracker()
    assert t.limits_for(Tier.PRO).llm_usd_daily == 10.0


def test_tracker_rejects_partial_limits_map() -> None:
    partial = {Tier.PRO: DEFAULT_TIER_LIMITS[Tier.PRO]}
    with pytest.raises(ValueError, match="missing tier"):
        QuotaTracker(limits_by_tier=partial)


def test_tracker_accepts_custom_limits() -> None:
    custom = {
        tier: TierLimits(
            tier=tier,
            llm_usd_daily=1.0,
            llm_tokens_daily=100,
            broker_api_calls_daily=100,
            screener_api_calls_daily=100,
            cycle_runs_daily=100,
        )
        for tier in Tier
    }
    t = QuotaTracker(limits_by_tier=custom)
    assert t.limits_for(Tier.PRO).llm_usd_daily == 1.0


# ---------------------------------------------------------------------------
# Quota state classification
# ---------------------------------------------------------------------------


def test_check_zero_use_is_ok() -> None:
    t = _tracker()
    u = _usage()
    result = t.check(u, resource=ResourceKind.LLM_USD)
    assert result.state is QuotaState.OK
    assert result.pct_used == 0.0


def test_check_below_warning_threshold_is_ok() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=5.0)  # 50% of $10
    result = t.check(u, resource=ResourceKind.LLM_USD)
    assert result.state is QuotaState.OK
    assert result.pct_used == 50.0


def test_check_at_warning_threshold_is_warning() -> None:
    """Pin: 80% triggers WARNING (boundary inclusive)."""

    t = _tracker()
    u = _usage(llm_usd_used=8.0)  # 80% of $10
    result = t.check(u, resource=ResourceKind.LLM_USD)
    assert result.state is QuotaState.WARNING


def test_check_above_warning_below_exceeded_is_warning() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=9.5)  # 95% of $10
    result = t.check(u, resource=ResourceKind.LLM_USD)
    assert result.state is QuotaState.WARNING
    assert pytest.approx(result.pct_used) == 95.0


def test_check_at_exceeded_threshold_is_exceeded() -> None:
    """Pin: 100% triggers EXCEEDED (boundary inclusive)."""

    t = _tracker()
    u = _usage(llm_usd_used=10.0)
    result = t.check(u, resource=ResourceKind.LLM_USD)
    assert result.state is QuotaState.EXCEEDED


def test_check_above_exceeded_threshold_is_exceeded() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=15.0)
    result = t.check(u, resource=ResourceKind.LLM_USD)
    assert result.state is QuotaState.EXCEEDED
    assert result.pct_used == 150.0


def test_check_zero_limit_with_zero_use_is_ok() -> None:
    """FREE-tier broker_api_calls = 0; user with 0 use lands OK."""

    t = _tracker()
    u = _usage(tier=Tier.FREE)
    result = t.check(u, resource=ResourceKind.BROKER_API_CALLS)
    assert result.state is QuotaState.OK
    assert result.limit == 0.0


def test_check_zero_limit_with_any_use_is_exceeded() -> None:
    """Pin: FREE-tier broker_api_calls = 0; any use is EXCEEDED."""

    t = _tracker()
    # FREE-tier user with 1 broker call (impossible via quota gate, but
    # if the row was hand-edited or a race condition put a usage there,
    # the check must surface EXCEEDED rather than divide-by-zero).
    u = _usage(tier=Tier.FREE, broker_api_calls_used=1)
    result = t.check(u, resource=ResourceKind.BROKER_API_CALLS)
    assert result.state is QuotaState.EXCEEDED


# ---------------------------------------------------------------------------
# check() with hypothetical request
# ---------------------------------------------------------------------------


def test_check_with_requested_simulates_consume() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=7.0)
    result = t.check(u, resource=ResourceKind.LLM_USD, requested=1.5)
    assert result.used == pytest.approx(8.5)
    assert result.state is QuotaState.WARNING


def test_check_does_not_mutate_input_usage() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=5.0)
    t.check(u, resource=ResourceKind.LLM_USD, requested=2.0)
    assert u.llm_usd_used == 5.0  # input unchanged


def test_check_rejects_negative_requested() -> None:
    t = _tracker()
    u = _usage()
    with pytest.raises(ValueError, match="requested"):
        t.check(u, resource=ResourceKind.LLM_USD, requested=-1.0)


def test_check_remaining_clamps_at_zero_when_over() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=12.0)
    result = t.check(u, resource=ResourceKind.LLM_USD)
    assert result.remaining == 0.0


# ---------------------------------------------------------------------------
# consume() lifecycle
# ---------------------------------------------------------------------------


def test_consume_adds_to_existing_usage() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=2.0)
    new = t.consume(u, resource=ResourceKind.LLM_USD, amount=1.5)
    assert new.llm_usd_used == pytest.approx(3.5)
    assert u.llm_usd_used == 2.0  # input unchanged


def test_consume_returns_new_frozen_row() -> None:
    t = _tracker()
    u = _usage()
    new = t.consume(u, resource=ResourceKind.LLM_USD, amount=1.0)
    assert isinstance(new, ResourceUsage)
    assert new is not u
    with pytest.raises(dataclasses.FrozenInstanceError):
        new.llm_usd_used = 99.0  # type: ignore[misc]


def test_consume_raises_on_overage() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=9.5)
    with pytest.raises(QuotaExceededError) as exc:
        t.consume(u, resource=ResourceKind.LLM_USD, amount=1.0)
    assert exc.value.user_id == "user-1"
    assert exc.value.resource is ResourceKind.LLM_USD
    assert exc.value.tier is Tier.PRO


def test_consume_at_exact_limit_succeeds() -> None:
    """Pin: consuming up to but not past the limit succeeds.

    A consume that lands at exactly limit is allowed (the next
    one fails). This is the operator-friendly choice — better to
    let a user spend their last cent than reject the call that
    brought them to exactly their cap.
    """

    t = _tracker()
    u = _usage(llm_usd_used=9.0)
    new = t.consume(u, resource=ResourceKind.LLM_USD, amount=1.0)
    assert new.llm_usd_used == pytest.approx(10.0)
    # next consume of any positive amount fails
    with pytest.raises(QuotaExceededError):
        t.consume(new, resource=ResourceKind.LLM_USD, amount=0.01)


def test_consume_rejects_negative_amount() -> None:
    t = _tracker()
    u = _usage()
    with pytest.raises(ValueError, match="amount"):
        t.consume(u, resource=ResourceKind.LLM_USD, amount=-1.0)


def test_consume_zero_amount_is_noop() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=5.0)
    new = t.consume(u, resource=ResourceKind.LLM_USD, amount=0.0)
    assert new.llm_usd_used == 5.0


def test_consume_token_count_floors_fractional() -> None:
    """Pin: fractional token consumption rounds down via int().

    Half a token doesn't get charged — the user gets the
    benefit-of-the-doubt for partial counts.
    """

    t = _tracker()
    u = _usage(llm_tokens_used=100)
    new = t.consume(u, resource=ResourceKind.LLM_TOKENS, amount=10.7)
    assert new.llm_tokens_used == 110  # 100 + int(10.7)


# ---------------------------------------------------------------------------
# Rolling 24-hour window auto-reset
# ---------------------------------------------------------------------------


def test_consume_resets_window_after_24_hours() -> None:
    """Pin: rolling 24h windows reset on consume past the boundary."""

    yesterday = _NOW - timedelta(hours=25)
    t = _tracker(now=_NOW)
    u = _usage(window_started_at=yesterday, llm_usd_used=9.5)
    new = t.consume(u, resource=ResourceKind.LLM_USD, amount=2.0)
    # window rolled forward; usage starts fresh
    assert new.window_started_at == _NOW
    assert new.llm_usd_used == pytest.approx(2.0)


def test_consume_does_not_reset_window_at_23_hours() -> None:
    """Pin: 23h59m is still inside the window."""

    almost = _NOW - timedelta(hours=23, minutes=59)
    t = _tracker(now=_NOW)
    u = _usage(window_started_at=almost, llm_usd_used=9.5)
    new = t.consume(u, resource=ResourceKind.LLM_USD, amount=0.4)
    assert new.window_started_at == almost
    assert new.llm_usd_used == pytest.approx(9.9)


def test_consume_reset_at_exactly_24_hours_is_inclusive() -> None:
    """Pin: at exactly 24h elapsed, the window resets (boundary inclusive)."""

    exactly = _NOW - timedelta(hours=24)
    t = _tracker(now=_NOW)
    u = _usage(window_started_at=exactly, llm_usd_used=9.5)
    new = t.consume(u, resource=ResourceKind.LLM_USD, amount=1.0)
    assert new.window_started_at == _NOW
    assert new.llm_usd_used == pytest.approx(1.0)


def test_check_resets_window_in_returned_snapshot() -> None:
    """Pin: check() also rolls the window in the returned snapshot."""

    yesterday = _NOW - timedelta(hours=25)
    t = _tracker(now=_NOW)
    u = _usage(window_started_at=yesterday, llm_usd_used=9.5)
    result = t.check(u, resource=ResourceKind.LLM_USD)
    assert result.window_started_at == _NOW
    assert result.used == 0.0
    # but the input row is unchanged
    assert u.llm_usd_used == 9.5


# ---------------------------------------------------------------------------
# remaining() convenience
# ---------------------------------------------------------------------------


def test_remaining_subtracts_used_from_limit() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=3.0)
    assert t.remaining(u, resource=ResourceKind.LLM_USD) == pytest.approx(7.0)


def test_remaining_clamps_at_zero() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=15.0)
    assert t.remaining(u, resource=ResourceKind.LLM_USD) == 0.0


# ---------------------------------------------------------------------------
# QuotaCheckResult shape
# ---------------------------------------------------------------------------


def test_check_result_carries_all_fields() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=8.5)
    result = t.check(u, resource=ResourceKind.LLM_USD)
    assert isinstance(result, QuotaCheckResult)
    assert result.user_id == "user-1"
    assert result.resource is ResourceKind.LLM_USD
    assert result.tier is Tier.PRO
    assert result.used == pytest.approx(8.5)
    assert result.limit == 10.0
    assert result.remaining == pytest.approx(1.5)
    assert result.state is QuotaState.WARNING
    assert result.warnings  # warning state surfaces a warning


def test_check_warnings_empty_when_ok() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=2.0)
    result = t.check(u, resource=ResourceKind.LLM_USD)
    assert result.state is QuotaState.OK
    assert result.warnings == ()


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


def test_render_ok_state() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=2.0)
    result = t.check(u, resource=ResourceKind.LLM_USD)
    text = render_quota_check(result)
    assert "✅" in text
    assert "user-1" in text
    assert "pro" in text
    assert "OK" in text


def test_render_warning_state() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=8.5)
    result = t.check(u, resource=ResourceKind.LLM_USD)
    text = render_quota_check(result)
    assert "⚠️" in text
    assert "WARNING" in text


def test_render_exceeded_state() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=15.0)
    result = t.check(u, resource=ResourceKind.LLM_USD)
    text = render_quota_check(result)
    assert "🚫" in text
    assert "EXCEEDED" in text


def test_render_usd_uses_dollar_format() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=2.5)
    result = t.check(u, resource=ResourceKind.LLM_USD)
    text = render_quota_check(result)
    assert "$2.5" in text
    assert "$10.00" in text


def test_render_token_count_uses_integer_format() -> None:
    t = _tracker()
    u = _usage(llm_tokens_used=500_000)
    result = t.check(u, resource=ResourceKind.LLM_TOKENS)
    text = render_quota_check(result)
    # tokens render as plain integers
    assert "500000" in text
    assert "$" not in text  # no dollar formatting on token count


def test_render_includes_window_start() -> None:
    t = _tracker()
    u = _usage()
    result = t.check(u, resource=ResourceKind.LLM_USD)
    text = render_quota_check(result)
    assert "2026-05-01" in text


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_tier_limits_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_TIER_LIMITS[Tier.PRO].llm_usd_daily = 99.0  # type: ignore[misc]


def test_resource_usage_is_frozen() -> None:
    u = _usage()
    with pytest.raises(dataclasses.FrozenInstanceError):
        u.llm_usd_used = 99.0  # type: ignore[misc]


def test_check_result_is_frozen() -> None:
    t = _tracker()
    result = t.check(_usage(), resource=ResourceKind.LLM_USD)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.state = QuotaState.EXCEEDED  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB serialisation
# ---------------------------------------------------------------------------


def test_tier_string_values_pinned() -> None:
    assert Tier.FREE.value == "free"
    assert Tier.PRO.value == "pro"
    assert Tier.ENTERPRISE.value == "enterprise"


def test_resource_kind_string_values_pinned() -> None:
    assert ResourceKind.LLM_USD.value == "llm_usd"
    assert ResourceKind.LLM_TOKENS.value == "llm_tokens"
    assert ResourceKind.BROKER_API_CALLS.value == "broker_api_calls"
    assert ResourceKind.SCREENER_API_CALLS.value == "screener_api_calls"
    assert ResourceKind.CYCLE_RUNS.value == "cycle_runs"


def test_state_string_values_pinned() -> None:
    assert QuotaState.OK.value == "ok"
    assert QuotaState.WARNING.value == "warning"
    assert QuotaState.EXCEEDED.value == "exceeded"


# ---------------------------------------------------------------------------
# Cross-tier scenarios
# ---------------------------------------------------------------------------


def test_free_user_with_zero_broker_quota_is_blocked_immediately() -> None:
    t = _tracker()
    u = _usage(tier=Tier.FREE)
    with pytest.raises(QuotaExceededError):
        t.consume(u, resource=ResourceKind.BROKER_API_CALLS, amount=1)


def test_enterprise_user_has_much_higher_headroom() -> None:
    t = _tracker()
    free_u = _usage(tier=Tier.FREE)
    ent_u = _usage(tier=Tier.ENTERPRISE)
    free_remaining = t.remaining(free_u, resource=ResourceKind.LLM_USD)
    ent_remaining = t.remaining(ent_u, resource=ResourceKind.LLM_USD)
    assert ent_remaining > free_remaining * 100  # 0.50 vs 100.00


# ---------------------------------------------------------------------------
# End-to-end realistic usage flow
# ---------------------------------------------------------------------------


def test_realistic_user_flow_through_a_day() -> None:
    """Simulate a PRO-tier user spending across the day:
    - 8am: $1 LLM call (10% — OK)
    - 10am: $5 LLM call (60% — OK)
    - 12pm: $2 LLM call (80% — WARNING)
    - 2pm: $2 LLM call rejected (would be 100%)
    - next day: window resets, fresh $10 budget
    """

    t = _tracker(now=_NOW)
    u = _usage(window_started_at=_NOW)

    u = t.consume(u, resource=ResourceKind.LLM_USD, amount=1.0)
    assert t.check(u, resource=ResourceKind.LLM_USD).state is QuotaState.OK

    u = t.consume(u, resource=ResourceKind.LLM_USD, amount=5.0)
    assert t.check(u, resource=ResourceKind.LLM_USD).state is QuotaState.OK

    u = t.consume(u, resource=ResourceKind.LLM_USD, amount=2.0)
    state = t.check(u, resource=ResourceKind.LLM_USD).state
    assert state is QuotaState.WARNING

    # budget remaining: $2.00; next request of $2.01 is rejected
    with pytest.raises(QuotaExceededError):
        t.consume(u, resource=ResourceKind.LLM_USD, amount=2.01)

    # advance to tomorrow; window resets
    tomorrow = _NOW + timedelta(hours=25)
    t2 = _tracker(now=tomorrow)
    u_fresh = t2.consume(u, resource=ResourceKind.LLM_USD, amount=3.0)
    assert u_fresh.llm_usd_used == pytest.approx(3.0)
    assert u_fresh.window_started_at == tomorrow


# ---------------------------------------------------------------------------
# QuotaExceededError carries useful info
# ---------------------------------------------------------------------------


def test_quota_exceeded_error_carries_context() -> None:
    t = _tracker()
    u = _usage(llm_usd_used=9.9)
    try:
        t.consume(u, resource=ResourceKind.LLM_USD, amount=0.5)
    except QuotaExceededError as exc:
        assert exc.user_id == "user-1"
        assert exc.resource is ResourceKind.LLM_USD
        assert exc.tier is Tier.PRO
        assert exc.used == pytest.approx(10.4)
        assert exc.limit == 10.0
        assert "user-1" in str(exc)
        assert "llm_usd" in str(exc)
        assert "pro" in str(exc)
    else:
        pytest.fail("expected QuotaExceededError")
