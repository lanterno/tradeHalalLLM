"""Billing tier-state engine.

The roadmap pins Wave 3.F: "Stripe billing + tier upgrades. Tier
system (free / pro / enterprise) with per-tier knobs. Wire into
`core/llm/budget.py`." This module is the **pure-Python state
machine** that tracks the user's subscription tier through
trial / active / grace-period / cancelled lifecycles. The Stripe
SDK integration (webhook receiver for invoice.paid /
invoice.payment_failed events; subscription.created / .updated /
.deleted) is operator-side; this module ships the deterministic
state-transition primitives the webhook handler composes with.

Picked a focused state machine over a "hand-roll subscription
logic per route" approach because (a) the state transitions
need to be deterministic + regression-pinnable (the same event
applied to the same state always produces the same next state),
(b) operators tracing a billing dispute need to replay the
event history against the engine to confirm the user's tier was
correctly assigned at every moment, (c) the Wave 3.C quota
system (LLM budget gate) keys on the *effective* tier — which
isn't the same as the Stripe-stated tier when the user is in a
grace period or cancellation. This module computes the effective
tier deterministically.

Pinned semantics:
- **Trial period 7-30 days.** Below 7d is operationally awful
  (user doesn't have time to evaluate); above 30d is revenue
  leakage (tier should convert by then).
- **Grace period after invoice failure: 7 days before downgrade
  to FREE.** Stripe's default grace period — gives users time
  to update payment methods. After 7d, effective tier drops to
  FREE; the operator's manual reactivation flow can restore.
- **Cancellation downgrades at period end, not immediately.**
  The user paid for the period; they keep PRO/ENTERPRISE
  features until the period naturally ends. Pinned via test.
- **Upgrades take effect immediately.** A user upgrading from
  PRO to ENTERPRISE gets ENTERPRISE features in the same
  request — pro-rated billing is the Stripe SDK's concern.
- **Render output never includes invoice amounts / Stripe
  customer IDs.** Mirrors no-secret patterns of Wave 3.B vault +
  Wave 8.D OTLP + Wave 12.G co-pilot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

# Re-use the Wave 3.C tier enum so the billing engine and the
# quota engine speak the same language.
from halal_trader.web.quotas import Tier


class SubscriptionStatus(str, Enum):
    """Subscription lifecycle status.

    Pinned string values for JSON / DB stability. The dashboard's
    billing-event audit log keys on these literals.
    """

    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    GRACE_PERIOD = "grace_period"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class BillingEventKind(str, Enum):
    """Standard Stripe-aligned billing event kinds.

    Pinned values; the Stripe webhook handler maps Stripe event
    types to these literals before calling `apply_event`.
    """

    SUBSCRIPTION_CREATED = "subscription_created"
    INVOICE_PAID = "invoice_paid"
    INVOICE_PAYMENT_FAILED = "invoice_payment_failed"
    TIER_UPGRADED = "tier_upgraded"
    TIER_DOWNGRADED = "tier_downgraded"
    SUBSCRIPTION_CANCELLED = "subscription_cancelled"
    TRIAL_ENDED = "trial_ended"


_MIN_TRIAL_DAYS = 7
_MAX_TRIAL_DAYS = 30
_GRACE_PERIOD_DAYS = 7


@dataclass(frozen=True)
class BillingPolicy:
    """Operator-tunable billing policy."""

    default_trial_days: int = 14
    grace_period_days: int = _GRACE_PERIOD_DAYS

    def __post_init__(self) -> None:
        if not _MIN_TRIAL_DAYS <= self.default_trial_days <= _MAX_TRIAL_DAYS:
            raise ValueError(
                f"default_trial_days {self.default_trial_days} out of "
                f"range [{_MIN_TRIAL_DAYS}, {_MAX_TRIAL_DAYS}]"
            )
        if self.grace_period_days <= 0:
            raise ValueError("grace_period_days must be positive")


DEFAULT_POLICY = BillingPolicy()


@dataclass(frozen=True)
class Subscription:
    """One user's subscription state.

    `current_period_end` is the natural end of the paid period;
    `trial_end_at` is set during TRIALING and None otherwise;
    `grace_period_end_at` is set during GRACE_PERIOD and None
    otherwise.
    """

    user_id: str
    tier: Tier
    status: SubscriptionStatus
    current_period_start: datetime
    current_period_end: datetime
    trial_end_at: datetime | None = None
    grace_period_end_at: datetime | None = None
    cancel_at_period_end: bool = False

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if self.current_period_start.tzinfo is None:
            raise ValueError("current_period_start must be timezone-aware")
        if self.current_period_end.tzinfo is None:
            raise ValueError("current_period_end must be timezone-aware")
        if self.current_period_end <= self.current_period_start:
            raise ValueError("current_period_end must be after current_period_start")
        if self.trial_end_at is not None and self.trial_end_at.tzinfo is None:
            raise ValueError("trial_end_at must be timezone-aware when set")
        if self.grace_period_end_at is not None and self.grace_period_end_at.tzinfo is None:
            raise ValueError("grace_period_end_at must be timezone-aware when set")


@dataclass(frozen=True)
class BillingEvent:
    """One billing event from the Stripe webhook (or operator action)."""

    event_id: str
    kind: BillingEventKind
    timestamp: datetime
    target_tier: Tier | None = None  # for TIER_UPGRADED / TIER_DOWNGRADED

    def __post_init__(self) -> None:
        if not self.event_id or not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        # TIER_UPGRADED / TIER_DOWNGRADED require target_tier
        if (
            self.kind
            in (
                BillingEventKind.TIER_UPGRADED,
                BillingEventKind.TIER_DOWNGRADED,
            )
            and self.target_tier is None
        ):
            raise ValueError(f"{self.kind.value} event requires target_tier")


def create_trial(
    *,
    user_id: str,
    tier: Tier,
    now: datetime,
    policy: BillingPolicy = DEFAULT_POLICY,
) -> Subscription:
    """Create a fresh trial subscription.

    Trial expires after `policy.default_trial_days`; the period
    end and trial-end coincide. The operator's webhook handler
    issues a `TRIAL_ENDED` event when the trial expires (or
    `INVOICE_PAID` if the user converted before).
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be non-empty")
    if tier is Tier.FREE:
        raise ValueError("FREE tier doesn't trial; use create_active_free instead")

    trial_end = now + timedelta(days=policy.default_trial_days)
    return Subscription(
        user_id=user_id,
        tier=tier,
        status=SubscriptionStatus.TRIALING,
        current_period_start=now,
        current_period_end=trial_end,
        trial_end_at=trial_end,
    )


def apply_event(
    subscription: Subscription,
    event: BillingEvent,
    *,
    now: datetime,
    policy: BillingPolicy = DEFAULT_POLICY,
) -> Subscription:
    """Apply a billing event to the subscription, returning the new state.

    Pure: deterministic for a given (subscription, event, now).
    Operators replay the event history against the engine to
    audit the tier-at-moment for any past timestamp.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    if event.kind is BillingEventKind.SUBSCRIPTION_CREATED:
        # Idempotent — just return the subscription as-is. Stripe
        # might re-deliver the event after a network hiccup.
        return subscription

    if event.kind is BillingEventKind.INVOICE_PAID:
        return _apply_invoice_paid(subscription, event, now=now)

    if event.kind is BillingEventKind.INVOICE_PAYMENT_FAILED:
        return _apply_invoice_failed(subscription, event, now=now, policy=policy)

    if event.kind is BillingEventKind.TIER_UPGRADED:
        # Pinned: upgrades take effect immediately
        assert event.target_tier is not None
        return _replace(subscription, tier=event.target_tier, status=SubscriptionStatus.ACTIVE)

    if event.kind is BillingEventKind.TIER_DOWNGRADED:
        # Downgrades take effect at period end via cancel_at_period_end
        assert event.target_tier is not None
        # Stage the downgrade — actual tier change at period end
        return _replace(
            subscription,
            cancel_at_period_end=True,
            tier=event.target_tier,
        )

    if event.kind is BillingEventKind.SUBSCRIPTION_CANCELLED:
        return _replace(
            subscription,
            cancel_at_period_end=True,
        )

    if event.kind is BillingEventKind.TRIAL_ENDED:
        return _apply_trial_ended(subscription, event, now=now)

    raise ValueError(f"unrecognised event kind: {event.kind!r}")


def _apply_invoice_paid(sub: Subscription, event: BillingEvent, *, now: datetime) -> Subscription:
    """A successful payment: extend period, restore ACTIVE status."""

    period_length = sub.current_period_end - sub.current_period_start
    new_start = sub.current_period_end
    new_end = new_start + period_length
    return _replace(
        sub,
        status=SubscriptionStatus.ACTIVE,
        current_period_start=new_start,
        current_period_end=new_end,
        trial_end_at=None,  # exit trial if was trialing
        grace_period_end_at=None,
    )


def _apply_invoice_failed(
    sub: Subscription,
    event: BillingEvent,
    *,
    now: datetime,
    policy: BillingPolicy,
) -> Subscription:
    """A failed payment: enter grace period (7d default to retry)."""

    grace_end = now + timedelta(days=policy.grace_period_days)
    return _replace(
        sub,
        status=SubscriptionStatus.GRACE_PERIOD,
        grace_period_end_at=grace_end,
    )


def _apply_trial_ended(sub: Subscription, event: BillingEvent, *, now: datetime) -> Subscription:
    """Trial ended without conversion: subscription expires."""

    return _replace(
        sub,
        status=SubscriptionStatus.EXPIRED,
        trial_end_at=None,
    )


def _replace(sub: Subscription, **changes: object) -> Subscription:
    """Convenience: construct a new Subscription with overrides.

    The frozen-dataclass pattern requires explicit construction;
    this helper saves boilerplate at every transition.
    """

    fields = {
        "user_id": sub.user_id,
        "tier": sub.tier,
        "status": sub.status,
        "current_period_start": sub.current_period_start,
        "current_period_end": sub.current_period_end,
        "trial_end_at": sub.trial_end_at,
        "grace_period_end_at": sub.grace_period_end_at,
        "cancel_at_period_end": sub.cancel_at_period_end,
    }
    fields.update(changes)
    return Subscription(**fields)  # type: ignore[arg-type]


def compute_effective_tier(
    subscription: Subscription,
    *,
    now: datetime,
) -> Tier:
    """Return the user's effective tier at `now`.

    Effective tier is what the Wave 3.C quota system gates on:
    - TRIALING → the trialed tier (until trial_end_at)
    - ACTIVE → the subscription's tier
    - GRACE_PERIOD → the subscription's tier (if within grace)
    - GRACE_PERIOD past grace_period_end_at → FREE
    - CANCELLED before period end → the subscription's tier
    - CANCELLED after period end → FREE
    - EXPIRED → FREE
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    if subscription.status is SubscriptionStatus.EXPIRED:
        return Tier.FREE

    if subscription.status is SubscriptionStatus.CANCELLED:
        if now >= subscription.current_period_end:
            return Tier.FREE
        return subscription.tier

    if subscription.status is SubscriptionStatus.GRACE_PERIOD:
        grace_end = subscription.grace_period_end_at
        if grace_end is not None and now >= grace_end:
            return Tier.FREE
        return subscription.tier

    if subscription.status is SubscriptionStatus.TRIALING:
        trial_end = subscription.trial_end_at
        if trial_end is not None and now >= trial_end:
            return Tier.FREE
        return subscription.tier

    if subscription.status is SubscriptionStatus.PAST_DUE:
        # PAST_DUE is a transient bookkeeping status before the
        # webhook progresses to GRACE_PERIOD. Treat as effective
        # tier still active during the brief window.
        return subscription.tier

    if subscription.status is SubscriptionStatus.ACTIVE:
        if subscription.cancel_at_period_end and now >= subscription.current_period_end:
            return Tier.FREE
        return subscription.tier

    # Defensive default
    return Tier.FREE


_STATUS_EMOJI: dict[SubscriptionStatus, str] = {
    SubscriptionStatus.TRIALING: "🆓",
    SubscriptionStatus.ACTIVE: "✅",
    SubscriptionStatus.PAST_DUE: "⏰",
    SubscriptionStatus.GRACE_PERIOD: "⚠️",
    SubscriptionStatus.CANCELLED: "🛑",
    SubscriptionStatus.EXPIRED: "❌",
}


def render_subscription(
    subscription: Subscription,
    *,
    now: datetime,
) -> str:
    """Format a subscription for ops display.

    Pinned no-secret-leak: never includes Stripe customer IDs,
    invoice amounts, payment-method details. Shows user_id +
    tier + status + period dates + effective tier. Mirrors
    no-secret patterns of Wave 3.B vault + Wave 12.G co-pilot.
    """

    emoji = _STATUS_EMOJI[subscription.status]
    effective = compute_effective_tier(subscription, now=now)
    lines = [
        f"{emoji} {subscription.user_id} — {subscription.status.value.upper()}",
        f"  tier: {subscription.tier.value} (effective: {effective.value})",
        f"  period: {subscription.current_period_start.date().isoformat()} → "
        f"{subscription.current_period_end.date().isoformat()}",
    ]
    if subscription.trial_end_at is not None:
        lines.append(f"  trial ends: {subscription.trial_end_at.date().isoformat()}")
    if subscription.grace_period_end_at is not None:
        lines.append(f"  grace ends: {subscription.grace_period_end_at.date().isoformat()}")
    if subscription.cancel_at_period_end:
        lines.append("  cancel scheduled at period end")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "BillingEvent",
    "BillingEventKind",
    "BillingPolicy",
    "Subscription",
    "SubscriptionStatus",
    "apply_event",
    "compute_effective_tier",
    "create_trial",
    "render_subscription",
]
