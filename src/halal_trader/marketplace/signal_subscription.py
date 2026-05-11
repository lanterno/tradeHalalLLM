"""Signal subscription with Wakalah fee structure — Round-5 Wave 21.B.

Where Wave 17.C ships the strategy gallery (subscriptions to *strategies*
with periodic Wakalah fees), this module ships **signal subscriptions** —
per-author follow-relationships where the subscriber pays per period and
the author receives a fee that is *structurally* a Wakalah service fee
(flat per period, not performance-based).

Pipeline:
1. Subscriber opens a Subscription against a signal-author.
2. The clock-driven `accrue_due_fees(subscription, until)` computes
   how many billing periods have elapsed and produces a ledger of
   billing events.
3. Each billing event splits gross → author + platform, both flat.
4. Subscription can be cancelled — pinned cooldown before re-subscribe.

This module is structurally distinct from `community/strategy_gallery.py`:
- per-author follow (not per-strategy)
- finer-grained period accrual (signals can be daily)
- cooldown enforcement on re-subscribe to prevent fee-arbitrage.

Pinned semantics:

- **Closed-set BillingPeriod** — DAILY / WEEKLY / MONTHLY.
- **Closed-set SubscriptionStatus FSM** — ACTIVE → CANCELLED (terminal).
- **Closed-set FeeTier ladder** — STARTER / PRO / INSTITUTIONAL.
- **Fee tiers** map to a `(amount_per_period_usd, max_signals_per_period)`
  pair; operator can override the table.
- **Cooldown after cancel** = 7 days by default; same (subscriber,
  author) pair cannot re-subscribe within the window. Prevents
  cancel-just-before-billing fee arbitrage.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date, timedelta
from enum import Enum


class BillingPeriod(str, Enum):
    """Closed-set billing period ladder."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


_PERIOD_DAYS: dict[BillingPeriod, int] = {
    BillingPeriod.DAILY: 1,
    BillingPeriod.WEEKLY: 7,
    BillingPeriod.MONTHLY: 30,
}


class FeeTier(str, Enum):
    """Closed-set fee tier ladder."""

    STARTER = "starter"
    PRO = "pro"
    INSTITUTIONAL = "institutional"


@dataclass(frozen=True)
class TierPricing:
    """Per-tier pricing entry."""

    period_fee_usd: float
    max_signals_per_period: int

    def __post_init__(self) -> None:
        if self.period_fee_usd < 0:
            raise ValueError("period_fee_usd must be non-negative")
        if self.period_fee_usd > 5000:
            raise ValueError("period_fee_usd > $5000 is suspicious")
        if self.max_signals_per_period <= 0:
            raise ValueError("max_signals_per_period must be positive")


_DEFAULT_PRICING: dict[FeeTier, TierPricing] = {
    FeeTier.STARTER: TierPricing(period_fee_usd=10.0, max_signals_per_period=5),
    FeeTier.PRO: TierPricing(period_fee_usd=50.0, max_signals_per_period=30),
    FeeTier.INSTITUTIONAL: TierPricing(period_fee_usd=500.0, max_signals_per_period=300),
}


def default_pricing() -> dict[FeeTier, TierPricing]:
    """Return a fresh copy of the default pricing table."""
    return dict(_DEFAULT_PRICING)


class SubscriptionStatus(str, Enum):
    """Closed-set FSM ladder."""

    ACTIVE = "active"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Subscription:
    """One subscriber follow of one signal-author."""

    subscription_id: str
    subscriber_id: str
    author_id: str
    tier: FeeTier
    billing_period: BillingPeriod
    started_on: date
    platform_fee_pct: float = 0.20
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    cancelled_on: date | None = None

    def __post_init__(self) -> None:
        if not self.subscription_id or not self.subscription_id.strip():
            raise ValueError("subscription_id must be non-empty")
        if not self.subscriber_id or not self.subscriber_id.strip():
            raise ValueError("subscriber_id must be non-empty")
        if not self.author_id or not self.author_id.strip():
            raise ValueError("author_id must be non-empty")
        if self.subscriber_id == self.author_id:
            raise ValueError("cannot subscribe to self")
        if not 0.0 <= self.platform_fee_pct <= 0.30:
            raise ValueError("platform_fee_pct must be in [0, 0.30]")
        if self.cancelled_on is not None and self.cancelled_on < self.started_on:
            raise ValueError("cancelled_on must be ≥ started_on")
        if self.status is SubscriptionStatus.CANCELLED and self.cancelled_on is None:
            raise ValueError("CANCELLED requires cancelled_on")
        if self.status is SubscriptionStatus.ACTIVE and self.cancelled_on is not None:
            raise ValueError("ACTIVE must not have cancelled_on set")


@dataclass(frozen=True)
class BillingEvent:
    """One accrued period of billing."""

    subscription_id: str
    period_index: int
    """0-based; period_index=0 is the inception period."""
    period_end_on: date
    gross_usd: float
    platform_share_usd: float
    author_share_usd: float


def _period_end(
    started_on: date,
    billing_period: BillingPeriod,
    *,
    period_index: int,
) -> date:
    """End date of the `period_index`-th period (0-based, inclusive of
    one full period from started_on)."""
    days = _PERIOD_DAYS[billing_period]
    return started_on + timedelta(days=days * (period_index + 1))


def accrue_due_fees(
    subscription: Subscription,
    *,
    until: date,
    pricing: dict[FeeTier, TierPricing] | None = None,
) -> tuple[BillingEvent, ...]:
    """Compute the billing events that should fire between `started_on`
    and `until` (inclusive on `until`).

    Pinned:
    - Cancellation truncates accrual at `cancelled_on`.
    - Each fired period covers `[started_on + i*days, started_on + (i+1)*days]`.
    - First period (period_index=0) fires at `started_on + days`.
    """
    table = pricing if pricing is not None else _DEFAULT_PRICING
    if subscription.tier not in table:
        raise ValueError(f"no pricing entry for tier {subscription.tier.value}")
    pricing_entry = table[subscription.tier]
    end_horizon = until
    if subscription.cancelled_on is not None and subscription.cancelled_on < end_horizon:
        end_horizon = subscription.cancelled_on
    events: list[BillingEvent] = []
    i = 0
    while True:
        pe = _period_end(subscription.started_on, subscription.billing_period, period_index=i)
        if pe > end_horizon:
            break
        gross = pricing_entry.period_fee_usd
        platform = gross * subscription.platform_fee_pct
        author = gross - platform
        events.append(
            BillingEvent(
                subscription_id=subscription.subscription_id,
                period_index=i,
                period_end_on=pe,
                gross_usd=gross,
                platform_share_usd=platform,
                author_share_usd=author,
            )
        )
        i += 1
    return tuple(events)


def cancel_subscription(subscription: Subscription, *, on: date) -> Subscription:
    """Cancel an active subscription."""
    if subscription.status is not SubscriptionStatus.ACTIVE:
        raise ValueError("only ACTIVE subscriptions can be cancelled")
    if on < subscription.started_on:
        raise ValueError("cancellation cannot precede started_on")
    return replace(
        subscription,
        status=SubscriptionStatus.CANCELLED,
        cancelled_on=on,
    )


def can_resubscribe(
    subscriber_id: str,
    author_id: str,
    prior: Iterable[Subscription],
    *,
    on: date,
    cooldown_days: int = 7,
) -> tuple[bool, str]:
    """Can this (subscriber, author) pair re-subscribe on `on`?

    Pinned: if the most recent prior subscription was cancelled within
    `cooldown_days` of `on`, return False. Otherwise True.
    """
    if cooldown_days <= 0:
        raise ValueError("cooldown_days must be positive")
    matching = [s for s in prior if s.subscriber_id == subscriber_id and s.author_id == author_id]
    if not matching:
        return True, "no prior subscription"
    cancelled = [s for s in matching if s.status is SubscriptionStatus.CANCELLED]
    if not cancelled:
        # There's an active subscription already.
        return False, "existing active subscription"
    most_recent = max(cancelled, key=lambda s: s.cancelled_on or s.started_on)
    cooldown_until = (most_recent.cancelled_on or most_recent.started_on) + timedelta(
        days=cooldown_days
    )
    if on < cooldown_until:
        return False, f"cooldown active until {cooldown_until.isoformat()}"
    return True, "ok"


@dataclass(frozen=True)
class FeeRollup:
    """Aggregate of accrued billing events."""

    subscription_id: str
    n_periods: int
    total_gross_usd: float
    total_platform_usd: float
    total_author_usd: float


def rollup_events(events: Iterable[BillingEvent]) -> FeeRollup | None:
    """Roll up a sequence of billing events. Returns None if empty."""
    evs = tuple(events)
    if not evs:
        return None
    sub_ids = {e.subscription_id for e in evs}
    if len(sub_ids) != 1:
        raise ValueError("rollup_events expects events from a single subscription")
    return FeeRollup(
        subscription_id=evs[0].subscription_id,
        n_periods=len(evs),
        total_gross_usd=sum(e.gross_usd for e in evs),
        total_platform_usd=sum(e.platform_share_usd for e in evs),
        total_author_usd=sum(e.author_share_usd for e in evs),
    )


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


def render_subscription(subscription: Subscription) -> str:
    status_emoji = "✅" if subscription.status is SubscriptionStatus.ACTIVE else "🚫"
    return (
        f"{status_emoji} {subscription.subscription_id} "
        f"[{subscription.tier.value}/{subscription.billing_period.value}]: "
        f"{_mask(subscription.subscriber_id)} → "
        f"{_mask(subscription.author_id)} "
        f"(platform {subscription.platform_fee_pct * 100:.0f}%)"
    )


def render_rollup(rollup: FeeRollup) -> str:
    return (
        f"💰 {rollup.subscription_id}: {rollup.n_periods} periods, "
        f"gross=${rollup.total_gross_usd:.2f}, "
        f"platform=${rollup.total_platform_usd:.2f}, "
        f"author=${rollup.total_author_usd:.2f}"
    )
