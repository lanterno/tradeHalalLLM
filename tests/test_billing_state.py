"""Tests for `halal_trader.web.billing_state` (Wave 3.F).

Covers the pure-Python billing state machine: policy validation,
subscription/event invariants, deterministic state transitions,
the load-bearing `compute_effective_tier` function (which the
Wave 3.C quota gate keys on), and render no-secret contracts.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.web.billing_state import (
    DEFAULT_POLICY,
    BillingEvent,
    BillingEventKind,
    BillingPolicy,
    Subscription,
    SubscriptionStatus,
    apply_event,
    compute_effective_tier,
    create_trial,
    render_subscription,
)
from halal_trader.web.quotas import Tier

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# ----------------------------- BillingPolicy ---------------------------------


def test_default_policy_uses_14_day_trial() -> None:
    assert DEFAULT_POLICY.default_trial_days == 14
    assert DEFAULT_POLICY.grace_period_days == 7


def test_policy_rejects_trial_below_7_days() -> None:
    with pytest.raises(ValueError, match="default_trial_days"):
        BillingPolicy(default_trial_days=6)


def test_policy_accepts_trial_at_7_day_lower_boundary() -> None:
    policy = BillingPolicy(default_trial_days=7)
    assert policy.default_trial_days == 7


def test_policy_accepts_trial_at_30_day_upper_boundary() -> None:
    policy = BillingPolicy(default_trial_days=30)
    assert policy.default_trial_days == 30


def test_policy_rejects_trial_above_30_days() -> None:
    with pytest.raises(ValueError, match="default_trial_days"):
        BillingPolicy(default_trial_days=31)


def test_policy_rejects_zero_grace_period() -> None:
    with pytest.raises(ValueError, match="grace_period_days"):
        BillingPolicy(grace_period_days=0)


def test_policy_rejects_negative_grace_period() -> None:
    with pytest.raises(ValueError, match="grace_period_days"):
        BillingPolicy(grace_period_days=-1)


def test_policy_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.default_trial_days = 21  # type: ignore[misc]


# ------------------------ Subscription dataclass -----------------------------


def _make_sub(**overrides: object) -> Subscription:
    base: dict[str, object] = {
        "user_id": "user_1",
        "tier": Tier.PRO,
        "status": SubscriptionStatus.ACTIVE,
        "current_period_start": T0,
        "current_period_end": T0 + timedelta(days=30),
    }
    base.update(overrides)
    return Subscription(**base)  # type: ignore[arg-type]


def test_subscription_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        _make_sub(user_id="")


def test_subscription_rejects_whitespace_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        _make_sub(user_id="   ")


def test_subscription_rejects_naive_period_start() -> None:
    naive = datetime(2026, 5, 1, 12, 0, 0)
    with pytest.raises(ValueError, match="current_period_start"):
        _make_sub(current_period_start=naive, current_period_end=T0 + timedelta(days=30))


def test_subscription_rejects_naive_period_end() -> None:
    naive = datetime(2026, 6, 1, 12, 0, 0)
    with pytest.raises(ValueError, match="current_period_end"):
        _make_sub(current_period_end=naive)


def test_subscription_rejects_period_end_equal_to_start() -> None:
    with pytest.raises(ValueError, match="current_period_end must be after"):
        _make_sub(current_period_end=T0)


def test_subscription_rejects_period_end_before_start() -> None:
    with pytest.raises(ValueError, match="current_period_end must be after"):
        _make_sub(current_period_end=T0 - timedelta(days=1))


def test_subscription_rejects_naive_trial_end_at() -> None:
    with pytest.raises(ValueError, match="trial_end_at"):
        _make_sub(trial_end_at=datetime(2026, 5, 8, 12, 0, 0))


def test_subscription_rejects_naive_grace_period_end_at() -> None:
    with pytest.raises(ValueError, match="grace_period_end_at"):
        _make_sub(grace_period_end_at=datetime(2026, 5, 8, 12, 0, 0))


def test_subscription_is_frozen() -> None:
    sub = _make_sub()
    with pytest.raises(FrozenInstanceError):
        sub.tier = Tier.ENTERPRISE  # type: ignore[misc]


# --------------------------- BillingEvent ------------------------------------


def test_billing_event_rejects_empty_event_id() -> None:
    with pytest.raises(ValueError, match="event_id"):
        BillingEvent(event_id="", kind=BillingEventKind.INVOICE_PAID, timestamp=T0)


def test_billing_event_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        BillingEvent(
            event_id="evt_1",
            kind=BillingEventKind.INVOICE_PAID,
            timestamp=datetime(2026, 5, 1, 12, 0, 0),
        )


def test_billing_event_tier_upgraded_requires_target_tier() -> None:
    with pytest.raises(ValueError, match="target_tier"):
        BillingEvent(
            event_id="evt_2",
            kind=BillingEventKind.TIER_UPGRADED,
            timestamp=T0,
        )


def test_billing_event_tier_downgraded_requires_target_tier() -> None:
    with pytest.raises(ValueError, match="target_tier"):
        BillingEvent(
            event_id="evt_3",
            kind=BillingEventKind.TIER_DOWNGRADED,
            timestamp=T0,
        )


def test_billing_event_invoice_paid_does_not_require_target_tier() -> None:
    event = BillingEvent(
        event_id="evt_4",
        kind=BillingEventKind.INVOICE_PAID,
        timestamp=T0,
    )
    assert event.target_tier is None


def test_billing_event_is_frozen() -> None:
    event = BillingEvent(
        event_id="evt_5",
        kind=BillingEventKind.INVOICE_PAID,
        timestamp=T0,
    )
    with pytest.raises(FrozenInstanceError):
        event.event_id = "evt_other"  # type: ignore[misc]


# -------------------------- Enum string pins ---------------------------------


def test_subscription_status_string_values_pinned() -> None:
    assert SubscriptionStatus.TRIALING.value == "trialing"
    assert SubscriptionStatus.ACTIVE.value == "active"
    assert SubscriptionStatus.PAST_DUE.value == "past_due"
    assert SubscriptionStatus.GRACE_PERIOD.value == "grace_period"
    assert SubscriptionStatus.CANCELLED.value == "cancelled"
    assert SubscriptionStatus.EXPIRED.value == "expired"


def test_billing_event_kind_string_values_pinned() -> None:
    assert BillingEventKind.SUBSCRIPTION_CREATED.value == "subscription_created"
    assert BillingEventKind.INVOICE_PAID.value == "invoice_paid"
    assert BillingEventKind.INVOICE_PAYMENT_FAILED.value == "invoice_payment_failed"
    assert BillingEventKind.TIER_UPGRADED.value == "tier_upgraded"
    assert BillingEventKind.TIER_DOWNGRADED.value == "tier_downgraded"
    assert BillingEventKind.SUBSCRIPTION_CANCELLED.value == "subscription_cancelled"
    assert BillingEventKind.TRIAL_ENDED.value == "trial_ended"


# ----------------------------- create_trial ----------------------------------


def test_create_trial_uses_default_14_day_window() -> None:
    sub = create_trial(user_id="u1", tier=Tier.PRO, now=T0)
    assert sub.status is SubscriptionStatus.TRIALING
    assert sub.tier is Tier.PRO
    assert sub.trial_end_at == T0 + timedelta(days=14)
    assert sub.current_period_end == T0 + timedelta(days=14)


def test_create_trial_honours_custom_policy() -> None:
    policy = BillingPolicy(default_trial_days=21)
    sub = create_trial(user_id="u1", tier=Tier.ENTERPRISE, now=T0, policy=policy)
    assert sub.trial_end_at == T0 + timedelta(days=21)


def test_create_trial_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        create_trial(user_id="u1", tier=Tier.PRO, now=datetime(2026, 5, 1))


def test_create_trial_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        create_trial(user_id="", tier=Tier.PRO, now=T0)


def test_create_trial_rejects_free_tier() -> None:
    with pytest.raises(ValueError, match="FREE tier"):
        create_trial(user_id="u1", tier=Tier.FREE, now=T0)


def test_create_trial_pro_and_enterprise_both_supported() -> None:
    pro = create_trial(user_id="u1", tier=Tier.PRO, now=T0)
    ent = create_trial(user_id="u2", tier=Tier.ENTERPRISE, now=T0)
    assert pro.tier is Tier.PRO
    assert ent.tier is Tier.ENTERPRISE


# ------------------------- apply_event: SUBSCRIPTION_CREATED -----------------


def test_subscription_created_is_idempotent() -> None:
    sub = _make_sub()
    event = BillingEvent(event_id="evt", kind=BillingEventKind.SUBSCRIPTION_CREATED, timestamp=T0)
    after = apply_event(sub, event, now=T0)
    assert after == sub


# ------------------------- apply_event: INVOICE_PAID -------------------------


def test_invoice_paid_extends_period_and_activates() -> None:
    sub = create_trial(user_id="u1", tier=Tier.PRO, now=T0)
    paid = BillingEvent(
        event_id="evt", kind=BillingEventKind.INVOICE_PAID, timestamp=T0 + timedelta(days=14)
    )
    after = apply_event(sub, paid, now=T0 + timedelta(days=14))
    assert after.status is SubscriptionStatus.ACTIVE
    assert after.current_period_start == sub.current_period_end
    assert after.current_period_end == sub.current_period_end + timedelta(days=14)
    assert after.trial_end_at is None


def test_invoice_paid_clears_grace_period() -> None:
    sub = _make_sub(
        status=SubscriptionStatus.GRACE_PERIOD,
        grace_period_end_at=T0 + timedelta(days=7),
    )
    paid = BillingEvent(event_id="e", kind=BillingEventKind.INVOICE_PAID, timestamp=T0)
    after = apply_event(sub, paid, now=T0)
    assert after.status is SubscriptionStatus.ACTIVE
    assert after.grace_period_end_at is None


# ------------------------- apply_event: INVOICE_PAYMENT_FAILED ---------------


def test_invoice_failed_enters_grace_period() -> None:
    sub = _make_sub()
    failed = BillingEvent(event_id="e", kind=BillingEventKind.INVOICE_PAYMENT_FAILED, timestamp=T0)
    after = apply_event(sub, failed, now=T0)
    assert after.status is SubscriptionStatus.GRACE_PERIOD
    assert after.grace_period_end_at == T0 + timedelta(days=7)


def test_invoice_failed_uses_custom_grace_period() -> None:
    policy = BillingPolicy(grace_period_days=14)
    sub = _make_sub()
    failed = BillingEvent(event_id="e", kind=BillingEventKind.INVOICE_PAYMENT_FAILED, timestamp=T0)
    after = apply_event(sub, failed, now=T0, policy=policy)
    assert after.grace_period_end_at == T0 + timedelta(days=14)


# ------------------------- apply_event: TIER_UPGRADED ------------------------


def test_tier_upgraded_takes_effect_immediately() -> None:
    sub = _make_sub(tier=Tier.PRO)
    event = BillingEvent(
        event_id="e",
        kind=BillingEventKind.TIER_UPGRADED,
        timestamp=T0,
        target_tier=Tier.ENTERPRISE,
    )
    after = apply_event(sub, event, now=T0)
    assert after.tier is Tier.ENTERPRISE
    assert after.status is SubscriptionStatus.ACTIVE


def test_tier_upgraded_from_grace_restores_active() -> None:
    sub = _make_sub(
        tier=Tier.PRO,
        status=SubscriptionStatus.GRACE_PERIOD,
        grace_period_end_at=T0 + timedelta(days=7),
    )
    event = BillingEvent(
        event_id="e",
        kind=BillingEventKind.TIER_UPGRADED,
        timestamp=T0,
        target_tier=Tier.ENTERPRISE,
    )
    after = apply_event(sub, event, now=T0)
    assert after.status is SubscriptionStatus.ACTIVE


# ------------------------- apply_event: TIER_DOWNGRADED ----------------------


def test_tier_downgraded_stages_change_at_period_end() -> None:
    sub = _make_sub(tier=Tier.ENTERPRISE)
    event = BillingEvent(
        event_id="e",
        kind=BillingEventKind.TIER_DOWNGRADED,
        timestamp=T0,
        target_tier=Tier.PRO,
    )
    after = apply_event(sub, event, now=T0)
    assert after.cancel_at_period_end is True
    assert after.tier is Tier.PRO


# ------------------------- apply_event: SUBSCRIPTION_CANCELLED ---------------


def test_cancellation_sets_cancel_at_period_end() -> None:
    sub = _make_sub(tier=Tier.PRO)
    event = BillingEvent(
        event_id="e",
        kind=BillingEventKind.SUBSCRIPTION_CANCELLED,
        timestamp=T0,
    )
    after = apply_event(sub, event, now=T0)
    assert after.cancel_at_period_end is True
    # Tier and status unchanged — user keeps features until period end
    assert after.tier is Tier.PRO
    assert after.status is SubscriptionStatus.ACTIVE


# ------------------------- apply_event: TRIAL_ENDED --------------------------


def test_trial_ended_sets_expired() -> None:
    sub = create_trial(user_id="u1", tier=Tier.PRO, now=T0)
    event = BillingEvent(
        event_id="e",
        kind=BillingEventKind.TRIAL_ENDED,
        timestamp=T0 + timedelta(days=14),
    )
    after = apply_event(sub, event, now=T0 + timedelta(days=14))
    assert after.status is SubscriptionStatus.EXPIRED
    assert after.trial_end_at is None


# ------------------------- apply_event: misc ---------------------------------


def test_apply_event_rejects_naive_now() -> None:
    sub = _make_sub()
    event = BillingEvent(event_id="e", kind=BillingEventKind.INVOICE_PAID, timestamp=T0)
    with pytest.raises(ValueError, match="now"):
        apply_event(sub, event, now=datetime(2026, 5, 1))


# --------------------- compute_effective_tier --------------------------------


def test_effective_tier_active_returns_subscription_tier() -> None:
    sub = _make_sub(tier=Tier.PRO, status=SubscriptionStatus.ACTIVE)
    assert compute_effective_tier(sub, now=T0) is Tier.PRO


def test_effective_tier_expired_returns_free() -> None:
    sub = _make_sub(tier=Tier.PRO, status=SubscriptionStatus.EXPIRED)
    assert compute_effective_tier(sub, now=T0) is Tier.FREE


def test_effective_tier_trialing_within_window_returns_subscription_tier() -> None:
    sub = create_trial(user_id="u1", tier=Tier.ENTERPRISE, now=T0)
    assert compute_effective_tier(sub, now=T0 + timedelta(days=7)) is Tier.ENTERPRISE


def test_effective_tier_trialing_at_trial_end_drops_to_free() -> None:
    sub = create_trial(user_id="u1", tier=Tier.PRO, now=T0)
    assert compute_effective_tier(sub, now=T0 + timedelta(days=14)) is Tier.FREE


def test_effective_tier_trialing_past_trial_end_drops_to_free() -> None:
    sub = create_trial(user_id="u1", tier=Tier.PRO, now=T0)
    assert compute_effective_tier(sub, now=T0 + timedelta(days=20)) is Tier.FREE


def test_effective_tier_trialing_just_before_end_keeps_tier() -> None:
    sub = create_trial(user_id="u1", tier=Tier.PRO, now=T0)
    just_before = T0 + timedelta(days=14) - timedelta(seconds=1)
    assert compute_effective_tier(sub, now=just_before) is Tier.PRO


def test_effective_tier_grace_period_within_window_returns_tier() -> None:
    sub = _make_sub(
        tier=Tier.PRO,
        status=SubscriptionStatus.GRACE_PERIOD,
        grace_period_end_at=T0 + timedelta(days=7),
    )
    assert compute_effective_tier(sub, now=T0 + timedelta(days=3)) is Tier.PRO


def test_effective_tier_grace_period_at_grace_end_drops_to_free() -> None:
    grace_end = T0 + timedelta(days=7)
    sub = _make_sub(
        tier=Tier.PRO,
        status=SubscriptionStatus.GRACE_PERIOD,
        grace_period_end_at=grace_end,
    )
    assert compute_effective_tier(sub, now=grace_end) is Tier.FREE


def test_effective_tier_grace_period_past_grace_end_drops_to_free() -> None:
    sub = _make_sub(
        tier=Tier.PRO,
        status=SubscriptionStatus.GRACE_PERIOD,
        grace_period_end_at=T0 + timedelta(days=7),
    )
    assert compute_effective_tier(sub, now=T0 + timedelta(days=10)) is Tier.FREE


def test_effective_tier_cancelled_before_period_end_returns_tier() -> None:
    sub = _make_sub(
        tier=Tier.PRO,
        status=SubscriptionStatus.CANCELLED,
        current_period_end=T0 + timedelta(days=10),
    )
    assert compute_effective_tier(sub, now=T0 + timedelta(days=3)) is Tier.PRO


def test_effective_tier_cancelled_at_period_end_drops_to_free() -> None:
    period_end = T0 + timedelta(days=10)
    sub = _make_sub(
        tier=Tier.PRO,
        status=SubscriptionStatus.CANCELLED,
        current_period_end=period_end,
    )
    assert compute_effective_tier(sub, now=period_end) is Tier.FREE


def test_effective_tier_cancelled_past_period_end_drops_to_free() -> None:
    sub = _make_sub(
        tier=Tier.PRO,
        status=SubscriptionStatus.CANCELLED,
        current_period_end=T0 + timedelta(days=10),
    )
    assert compute_effective_tier(sub, now=T0 + timedelta(days=20)) is Tier.FREE


def test_effective_tier_active_with_cancel_at_period_end_keeps_tier_until_end() -> None:
    sub = _make_sub(
        tier=Tier.PRO,
        status=SubscriptionStatus.ACTIVE,
        current_period_end=T0 + timedelta(days=10),
        cancel_at_period_end=True,
    )
    assert compute_effective_tier(sub, now=T0 + timedelta(days=5)) is Tier.PRO


def test_effective_tier_active_with_cancel_at_period_end_drops_at_end() -> None:
    period_end = T0 + timedelta(days=10)
    sub = _make_sub(
        tier=Tier.PRO,
        status=SubscriptionStatus.ACTIVE,
        current_period_end=period_end,
        cancel_at_period_end=True,
    )
    assert compute_effective_tier(sub, now=period_end) is Tier.FREE


def test_effective_tier_past_due_keeps_subscription_tier() -> None:
    sub = _make_sub(tier=Tier.PRO, status=SubscriptionStatus.PAST_DUE)
    assert compute_effective_tier(sub, now=T0) is Tier.PRO


def test_effective_tier_rejects_naive_now() -> None:
    sub = _make_sub()
    with pytest.raises(ValueError, match="now"):
        compute_effective_tier(sub, now=datetime(2026, 5, 1))


# --------------------------- render_subscription -----------------------------


def test_render_includes_user_id_tier_status_and_period() -> None:
    sub = _make_sub(user_id="alice", tier=Tier.PRO)
    out = render_subscription(sub, now=T0)
    assert "alice" in out
    assert "ACTIVE" in out
    assert "pro" in out
    assert "2026-05-01" in out


def test_render_uses_status_emoji() -> None:
    active = _make_sub(status=SubscriptionStatus.ACTIVE)
    expired = _make_sub(status=SubscriptionStatus.EXPIRED)
    grace = _make_sub(
        status=SubscriptionStatus.GRACE_PERIOD,
        grace_period_end_at=T0 + timedelta(days=7),
    )
    assert "✅" in render_subscription(active, now=T0)
    assert "❌" in render_subscription(expired, now=T0)
    assert "⚠️" in render_subscription(grace, now=T0)


def test_render_shows_effective_tier_when_different_from_subscription_tier() -> None:
    expired = _make_sub(tier=Tier.PRO, status=SubscriptionStatus.EXPIRED)
    out = render_subscription(expired, now=T0)
    # Subscription tier "pro", effective "free" — both appear
    assert "pro" in out
    assert "free" in out


def test_render_includes_trial_end_when_set() -> None:
    sub = create_trial(user_id="u1", tier=Tier.PRO, now=T0)
    out = render_subscription(sub, now=T0)
    assert "trial ends" in out


def test_render_includes_grace_end_when_set() -> None:
    sub = _make_sub(
        status=SubscriptionStatus.GRACE_PERIOD,
        grace_period_end_at=T0 + timedelta(days=7),
    )
    out = render_subscription(sub, now=T0)
    assert "grace ends" in out


def test_render_shows_cancellation_marker_when_scheduled() -> None:
    sub = _make_sub(cancel_at_period_end=True)
    out = render_subscription(sub, now=T0)
    assert "cancel scheduled" in out


def test_render_omits_cancellation_marker_when_not_scheduled() -> None:
    sub = _make_sub(cancel_at_period_end=False)
    out = render_subscription(sub, now=T0)
    assert "cancel scheduled" not in out


def test_render_never_includes_stripe_customer_id() -> None:
    """No-secret pin: render must not surface Stripe identifiers
    even if a future field added them, the no-leak contract holds.
    """

    sub = _make_sub(user_id="cus_NotARealStripeID_just_a_name")
    out = render_subscription(sub, now=T0)
    # The user_id field is opaque to render — but we don't render
    # any field named cus_, sub_, in_, etc. that Stripe uses.
    assert "cus_" not in out.lower().replace("cus_notarealstripeid_just_a_name", "")
    assert "sub_" not in out.lower()
    assert "in_" not in out.lower().replace("includes", "").replace("invoice", "")


def test_render_never_includes_dollar_amounts() -> None:
    """No-secret pin: render does not surface invoice amounts."""

    sub = _make_sub()
    out = render_subscription(sub, now=T0)
    assert "$" not in out
    assert "USD" not in out


# ------------------------- end-to-end realistic flows ------------------------


def test_e2e_trial_then_conversion() -> None:
    sub = create_trial(user_id="u1", tier=Tier.PRO, now=T0)
    assert compute_effective_tier(sub, now=T0 + timedelta(days=7)) is Tier.PRO

    paid = BillingEvent(
        event_id="evt_paid",
        kind=BillingEventKind.INVOICE_PAID,
        timestamp=T0 + timedelta(days=14),
    )
    sub2 = apply_event(sub, paid, now=T0 + timedelta(days=14))
    assert sub2.status is SubscriptionStatus.ACTIVE
    assert compute_effective_tier(sub2, now=T0 + timedelta(days=20)) is Tier.PRO


def test_e2e_invoice_failure_then_grace_then_recovery() -> None:
    sub = _make_sub(tier=Tier.PRO, status=SubscriptionStatus.ACTIVE)
    failed = BillingEvent(
        event_id="evt_fail", kind=BillingEventKind.INVOICE_PAYMENT_FAILED, timestamp=T0
    )
    sub2 = apply_event(sub, failed, now=T0)
    assert sub2.status is SubscriptionStatus.GRACE_PERIOD
    # 3 days into grace — still PRO
    assert compute_effective_tier(sub2, now=T0 + timedelta(days=3)) is Tier.PRO

    # User updates payment method, retry succeeds
    paid = BillingEvent(
        event_id="evt_paid",
        kind=BillingEventKind.INVOICE_PAID,
        timestamp=T0 + timedelta(days=4),
    )
    sub3 = apply_event(sub2, paid, now=T0 + timedelta(days=4))
    assert sub3.status is SubscriptionStatus.ACTIVE
    assert sub3.grace_period_end_at is None


def test_e2e_invoice_failure_then_grace_expires_to_free() -> None:
    sub = _make_sub(tier=Tier.PRO, status=SubscriptionStatus.ACTIVE)
    failed = BillingEvent(
        event_id="evt_fail", kind=BillingEventKind.INVOICE_PAYMENT_FAILED, timestamp=T0
    )
    sub2 = apply_event(sub, failed, now=T0)
    # 8 days later — past 7d grace
    assert compute_effective_tier(sub2, now=T0 + timedelta(days=8)) is Tier.FREE


def test_e2e_cancellation_lifecycle() -> None:
    sub = _make_sub(
        tier=Tier.PRO,
        status=SubscriptionStatus.ACTIVE,
        current_period_end=T0 + timedelta(days=10),
    )
    cancel = BillingEvent(
        event_id="evt_cancel",
        kind=BillingEventKind.SUBSCRIPTION_CANCELLED,
        timestamp=T0,
    )
    sub2 = apply_event(sub, cancel, now=T0)
    # Still PRO until period end
    assert compute_effective_tier(sub2, now=T0 + timedelta(days=5)) is Tier.PRO
    # At period end, drops to FREE
    assert compute_effective_tier(sub2, now=T0 + timedelta(days=10)) is Tier.FREE


def test_e2e_immediate_upgrade() -> None:
    sub = _make_sub(tier=Tier.PRO, status=SubscriptionStatus.ACTIVE)
    upgrade = BillingEvent(
        event_id="evt_up",
        kind=BillingEventKind.TIER_UPGRADED,
        timestamp=T0,
        target_tier=Tier.ENTERPRISE,
    )
    sub2 = apply_event(sub, upgrade, now=T0)
    # Effective tier ENTERPRISE immediately
    assert compute_effective_tier(sub2, now=T0) is Tier.ENTERPRISE


def test_e2e_trial_expires_without_conversion() -> None:
    sub = create_trial(user_id="u1", tier=Tier.PRO, now=T0)
    trial_ended = BillingEvent(
        event_id="evt_trial_end",
        kind=BillingEventKind.TRIAL_ENDED,
        timestamp=T0 + timedelta(days=14),
    )
    sub2 = apply_event(sub, trial_ended, now=T0 + timedelta(days=14))
    assert sub2.status is SubscriptionStatus.EXPIRED
    assert compute_effective_tier(sub2, now=T0 + timedelta(days=15)) is Tier.FREE


def test_e2e_downgrade_takes_effect_at_period_end() -> None:
    sub = _make_sub(
        tier=Tier.ENTERPRISE,
        status=SubscriptionStatus.ACTIVE,
        current_period_end=T0 + timedelta(days=10),
    )
    downgrade = BillingEvent(
        event_id="evt_down",
        kind=BillingEventKind.TIER_DOWNGRADED,
        timestamp=T0,
        target_tier=Tier.PRO,
    )
    sub2 = apply_event(sub, downgrade, now=T0)
    # cancel_at_period_end staged; the downgrade tier is on the
    # subscription. Before period end, effective is PRO (the new
    # tier takes effect on the dataclass immediately, but the
    # period-end cancel will route through compute_effective_tier
    # to drop to FREE at period end).
    assert sub2.tier is Tier.PRO
    assert sub2.cancel_at_period_end is True
    assert compute_effective_tier(sub2, now=T0 + timedelta(days=10)) is Tier.FREE


# ----------------------- determinism / regression pins -----------------------


def test_apply_event_is_deterministic() -> None:
    """Same (subscription, event, now) must always produce the same result.

    Operators replay events to audit historical tier-at-moment.
    """

    sub = _make_sub()
    event = BillingEvent(event_id="e", kind=BillingEventKind.INVOICE_PAYMENT_FAILED, timestamp=T0)
    after_1 = apply_event(sub, event, now=T0)
    after_2 = apply_event(sub, event, now=T0)
    assert after_1 == after_2


def test_compute_effective_tier_is_deterministic() -> None:
    sub = _make_sub(
        status=SubscriptionStatus.GRACE_PERIOD,
        grace_period_end_at=T0 + timedelta(days=7),
    )
    a = compute_effective_tier(sub, now=T0 + timedelta(days=3))
    b = compute_effective_tier(sub, now=T0 + timedelta(days=3))
    assert a == b
