"""Strategy marketplace listing + subscription contract.

The roadmap pins Wave 7.E: "Operators can publish their strategy
as a template; other operators can subscribe (with revenue share).
Defer the legal / regulatory work to a future round; just spec
the interface here." This module is the **pure-Python contract +
state machine** that the marketplace route consumes — a focused
spec of the listing shape, the subscription lifecycle, and the
revenue-share math, deferred from the legal / payments side which
the operator wires in once jurisdictional review clears.

Picked a focused contract over a "build the marketplace UI first"
approach because (a) the marketplace boundary is where money
flows from subscribers to listing authors; the listing shape +
subscription lifecycle + revenue-share math need to be regression-
pinned before any UI / payment integration consumes them, otherwise
the legal review can't sign off on a moving target, (b) the
listing validation rules (no PII in description; pricing in
allowed range; license terms from a closed enum) are decision
rules best expressed as deterministic functions — composes with
Wave 10.A gallery (gallery is discovery; marketplace is commerce),
(c) the revenue-share math (90% to author / 10% to platform default)
is a single operator-tunable parameter that the operator's
finance + legal review must approve once; encoding it here means
the actual payouts read from the same source of truth.

Pinned semantics:
- **Listing must pass validation gate before publishing.**
  Description must be PII-free, pricing must be in the [$1, $999]
  monthly band, license_terms from the closed enum, halal cert
  level required. Mirrors Wave 10.A gallery's publication gate.
- **Subscription lifecycle: TRIAL → ACTIVE → CANCELLED.** A
  subscription enters TRIAL on subscribe; converts to ACTIVE
  after the trial window (default 7 days); CANCELLED is
  terminal. Pause/resume is via a separate flag; cancellation
  is one-way.
- **Revenue share defaults to 90% author / 10% platform.** The
  Stripe Connect split. Operator-tunable via `MarketplacePolicy`;
  validation enforces (author + platform) sums to 1.0.
- **Pricing is monthly USD.** Single-tier (no annual discount yet);
  the simpler model lets us launch the marketplace without
  per-tier pricing complexity.
- **Render output never includes Stripe customer / invoice IDs,
  card data, payout account numbers.** Mirrors no-secret patterns
  of upstream waves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # email
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),  # IP
    re.compile(r"\+?\d{1,3}[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}"),  # phone
)


_PRICE_FLOOR_USD = 1.0
_PRICE_CEILING_USD = 999.0
_MIN_TRIAL_DAYS = 1
_MAX_TRIAL_DAYS = 30
_DEFAULT_TRIAL_DAYS = 7
_DEFAULT_AUTHOR_SHARE = 0.90


class LicenseTerm(str, Enum):
    """Closed-set license terms operators can publish under.

    Pinned string values for JSON / DB stability. Adding a license
    term is a code review change.
    """

    PERSONAL_USE = "personal_use"
    COMMERCIAL_USE = "commercial_use"
    NON_COMMERCIAL_USE = "non_commercial_use"
    RESEARCH_ONLY = "research_only"


class ListingStatus(str, Enum):
    """Listing lifecycle status."""

    DRAFT = "draft"
    PUBLISHED = "published"
    UNLISTED = "unlisted"
    TAKEN_DOWN = "taken_down"


class SubscriptionStatus(str, Enum):
    """Marketplace subscription status (distinct from Wave 3.F billing).

    Pinned values; the marketplace's per-listing subscription is
    separate from the user's overall HALAL-trader tier.
    """

    TRIAL = "trial"
    ACTIVE = "active"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class HalalCertLevel(str, Enum):
    """Listing-level halal certification declared by the author.

    Pinned values; enforced at listing-validation time so a buyer
    knows the strictness label before subscribing.
    """

    BASIC = "basic"
    MODERATE = "moderate"
    STRICT = "strict"
    SCHOLAR_REVIEWED = "scholar_reviewed"


class ListingViolationError(Exception):
    """Raised when a listing fails validation."""

    def __init__(self, listing_id: str, reason: str) -> None:
        super().__init__(f"listing {listing_id!r}: {reason}")
        self.listing_id = listing_id
        self.reason = reason


@dataclass(frozen=True)
class MarketplacePolicy:
    """Operator-tunable marketplace policy."""

    default_trial_days: int = _DEFAULT_TRIAL_DAYS
    author_share: float = _DEFAULT_AUTHOR_SHARE
    price_floor_usd: float = _PRICE_FLOOR_USD
    price_ceiling_usd: float = _PRICE_CEILING_USD

    def __post_init__(self) -> None:
        if not _MIN_TRIAL_DAYS <= self.default_trial_days <= _MAX_TRIAL_DAYS:
            raise ValueError(
                f"default_trial_days {self.default_trial_days} out of "
                f"[{_MIN_TRIAL_DAYS}, {_MAX_TRIAL_DAYS}]"
            )
        if not 0.0 < self.author_share < 1.0:
            raise ValueError(f"author_share {self.author_share} must be in (0, 1)")
        if self.price_floor_usd <= 0:
            raise ValueError("price_floor_usd must be positive")
        if self.price_ceiling_usd <= self.price_floor_usd:
            raise ValueError("price_ceiling_usd must exceed price_floor_usd")

    @property
    def platform_share(self) -> float:
        """Implicit: 1 - author_share."""

        return 1.0 - self.author_share


DEFAULT_POLICY = MarketplacePolicy()


@dataclass(frozen=True)
class MarketplaceListing:
    """One marketplace listing.

    `author_anonymous_handle` is the same anon-token shape as Wave
    10.A gallery uses (operator's user_id is NOT carried in the
    public listing). `monthly_price_usd` is the subscriber's
    monthly cost in USD.
    """

    listing_id: str
    author_anonymous_handle: str
    name: str
    description: str
    strategy_kind: str
    halal_cert_level: HalalCertLevel
    license_term: LicenseTerm
    monthly_price_usd: float
    status: ListingStatus
    published_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.listing_id or not self.listing_id.strip():
            raise ValueError("listing_id must be non-empty")
        if not self.author_anonymous_handle or not self.author_anonymous_handle.strip():
            raise ValueError("author_anonymous_handle must be non-empty")
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.description or not self.description.strip():
            raise ValueError("description must be non-empty")
        if not self.strategy_kind or not self.strategy_kind.strip():
            raise ValueError("strategy_kind must be non-empty")
        if self.monthly_price_usd <= 0:
            raise ValueError("monthly_price_usd must be positive")
        if self.published_at is not None and self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware when set")


def validate_listing(
    listing: MarketplaceListing,
    *,
    policy: MarketplacePolicy = DEFAULT_POLICY,
) -> None:
    """Run the publication gate. Raises ListingViolationError on failure.

    Operators call this before flipping a DRAFT listing to PUBLISHED.
    """

    # Price band
    if listing.monthly_price_usd < policy.price_floor_usd:
        raise ListingViolationError(
            listing.listing_id,
            f"monthly_price ${listing.monthly_price_usd:.2f} below "
            f"floor ${policy.price_floor_usd:.2f}",
        )
    if listing.monthly_price_usd > policy.price_ceiling_usd:
        raise ListingViolationError(
            listing.listing_id,
            f"monthly_price ${listing.monthly_price_usd:.2f} above "
            f"ceiling ${policy.price_ceiling_usd:.2f}",
        )

    # PII denylist on description + name
    for field_name, value in (("name", listing.name), ("description", listing.description)):
        for pattern in _PII_PATTERNS:
            if pattern.search(value):
                raise ListingViolationError(
                    listing.listing_id,
                    f"PII pattern detected in {field_name}",
                )

    # Status must be DRAFT to be eligible for publishing
    if listing.status not in (ListingStatus.DRAFT, ListingStatus.UNLISTED):
        raise ListingViolationError(
            listing.listing_id,
            f"only DRAFT or UNLISTED listings can be validated for "
            f"publishing; got {listing.status.value}",
        )


def publish_listing(
    listing: MarketplaceListing,
    *,
    now: datetime,
    policy: MarketplacePolicy = DEFAULT_POLICY,
) -> MarketplaceListing:
    """Publish a listing after validation."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    validate_listing(listing, policy=policy)
    return MarketplaceListing(
        listing_id=listing.listing_id,
        author_anonymous_handle=listing.author_anonymous_handle,
        name=listing.name,
        description=listing.description,
        strategy_kind=listing.strategy_kind,
        halal_cert_level=listing.halal_cert_level,
        license_term=listing.license_term,
        monthly_price_usd=listing.monthly_price_usd,
        status=ListingStatus.PUBLISHED,
        published_at=now,
    )


def take_down_listing(
    listing: MarketplaceListing,
) -> MarketplaceListing:
    """Move a listing to TAKEN_DOWN (terminal). Active subscriptions
    keep running but no new subscriptions allowed."""

    return MarketplaceListing(
        listing_id=listing.listing_id,
        author_anonymous_handle=listing.author_anonymous_handle,
        name=listing.name,
        description=listing.description,
        strategy_kind=listing.strategy_kind,
        halal_cert_level=listing.halal_cert_level,
        license_term=listing.license_term,
        monthly_price_usd=listing.monthly_price_usd,
        status=ListingStatus.TAKEN_DOWN,
        published_at=listing.published_at,
    )


@dataclass(frozen=True)
class Subscription:
    """One subscriber's subscription to a listing.

    Operators create with `start_subscription`; lifecycle managed
    through `convert_to_active` (after trial), `pause`, `resume`,
    `cancel`. The dataclass is frozen; ops return new instances.
    """

    subscription_id: str
    listing_id: str
    subscriber_anonymous_handle: str
    status: SubscriptionStatus
    started_at: datetime
    trial_end_at: datetime
    cancelled_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.subscription_id or not self.subscription_id.strip():
            raise ValueError("subscription_id must be non-empty")
        if not self.listing_id or not self.listing_id.strip():
            raise ValueError("listing_id must be non-empty")
        if not self.subscriber_anonymous_handle or not self.subscriber_anonymous_handle.strip():
            raise ValueError("subscriber_anonymous_handle must be non-empty")
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        if self.trial_end_at.tzinfo is None:
            raise ValueError("trial_end_at must be timezone-aware")
        if self.trial_end_at < self.started_at:
            raise ValueError("trial_end_at must be >= started_at")
        if self.cancelled_at is not None and self.cancelled_at.tzinfo is None:
            raise ValueError("cancelled_at must be timezone-aware when set")


def start_subscription(
    *,
    subscription_id: str,
    listing: MarketplaceListing,
    subscriber_anonymous_handle: str,
    now: datetime,
    policy: MarketplacePolicy = DEFAULT_POLICY,
) -> Subscription:
    """Start a new TRIAL subscription against a PUBLISHED listing."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if listing.status is not ListingStatus.PUBLISHED:
        raise ValueError(f"cannot subscribe to listing in {listing.status.value} status")
    trial_end = now + timedelta(days=policy.default_trial_days)
    return Subscription(
        subscription_id=subscription_id,
        listing_id=listing.listing_id,
        subscriber_anonymous_handle=subscriber_anonymous_handle,
        status=SubscriptionStatus.TRIAL,
        started_at=now,
        trial_end_at=trial_end,
    )


def convert_to_active(subscription: Subscription, *, now: datetime) -> Subscription:
    """Convert a TRIAL subscription to ACTIVE after the trial window."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if subscription.status is not SubscriptionStatus.TRIAL:
        raise ValueError(f"can only convert TRIAL subscriptions; got {subscription.status.value}")
    if now < subscription.trial_end_at:
        raise ValueError("cannot convert before trial_end_at")
    return Subscription(
        subscription_id=subscription.subscription_id,
        listing_id=subscription.listing_id,
        subscriber_anonymous_handle=subscription.subscriber_anonymous_handle,
        status=SubscriptionStatus.ACTIVE,
        started_at=subscription.started_at,
        trial_end_at=subscription.trial_end_at,
    )


def pause_subscription(subscription: Subscription) -> Subscription:
    """Pause an ACTIVE subscription. Cannot pause TRIAL or CANCELLED."""

    if subscription.status is not SubscriptionStatus.ACTIVE:
        raise ValueError(f"can only pause ACTIVE; got {subscription.status.value}")
    return Subscription(
        subscription_id=subscription.subscription_id,
        listing_id=subscription.listing_id,
        subscriber_anonymous_handle=subscription.subscriber_anonymous_handle,
        status=SubscriptionStatus.PAUSED,
        started_at=subscription.started_at,
        trial_end_at=subscription.trial_end_at,
        cancelled_at=subscription.cancelled_at,
    )


def resume_subscription(subscription: Subscription) -> Subscription:
    """Resume a PAUSED subscription."""

    if subscription.status is not SubscriptionStatus.PAUSED:
        raise ValueError(f"can only resume PAUSED; got {subscription.status.value}")
    return Subscription(
        subscription_id=subscription.subscription_id,
        listing_id=subscription.listing_id,
        subscriber_anonymous_handle=subscription.subscriber_anonymous_handle,
        status=SubscriptionStatus.ACTIVE,
        started_at=subscription.started_at,
        trial_end_at=subscription.trial_end_at,
        cancelled_at=subscription.cancelled_at,
    )


def cancel_subscription(subscription: Subscription, *, now: datetime) -> Subscription:
    """Cancel a subscription (terminal). Can cancel from TRIAL / ACTIVE / PAUSED."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if subscription.status is SubscriptionStatus.CANCELLED:
        raise ValueError("already cancelled")
    return Subscription(
        subscription_id=subscription.subscription_id,
        listing_id=subscription.listing_id,
        subscriber_anonymous_handle=subscription.subscriber_anonymous_handle,
        status=SubscriptionStatus.CANCELLED,
        started_at=subscription.started_at,
        trial_end_at=subscription.trial_end_at,
        cancelled_at=now,
    )


@dataclass(frozen=True)
class RevenueSplit:
    """One billing-cycle's revenue split between author + platform."""

    listing_id: str
    cycle_revenue_usd: float
    author_amount_usd: float
    platform_amount_usd: float

    def __post_init__(self) -> None:
        if self.cycle_revenue_usd < 0:
            raise ValueError("cycle_revenue_usd must be non-negative")
        if self.author_amount_usd < 0 or self.platform_amount_usd < 0:
            raise ValueError("amounts must be non-negative")
        # Sum equality: allow 0.01 rounding tolerance for the cents
        delta = abs((self.author_amount_usd + self.platform_amount_usd) - self.cycle_revenue_usd)
        if delta > 0.01:
            raise ValueError(
                f"author + platform ({self.author_amount_usd + self.platform_amount_usd:.2f}) "
                f"!= revenue ({self.cycle_revenue_usd:.2f})"
            )


def compute_split(
    *,
    listing: MarketplaceListing,
    cycle_revenue_usd: float,
    policy: MarketplacePolicy = DEFAULT_POLICY,
) -> RevenueSplit:
    """Compute author + platform revenue split for one billing cycle."""

    if cycle_revenue_usd < 0:
        raise ValueError("cycle_revenue_usd must be non-negative")
    author_amount = round(cycle_revenue_usd * policy.author_share, 2)
    platform_amount = round(cycle_revenue_usd - author_amount, 2)
    return RevenueSplit(
        listing_id=listing.listing_id,
        cycle_revenue_usd=cycle_revenue_usd,
        author_amount_usd=author_amount,
        platform_amount_usd=platform_amount,
    )


_LISTING_STATUS_EMOJI: dict[ListingStatus, str] = {
    ListingStatus.DRAFT: "📝",
    ListingStatus.PUBLISHED: "✅",
    ListingStatus.UNLISTED: "👁️",
    ListingStatus.TAKEN_DOWN: "🚫",
}


_SUBSCRIPTION_STATUS_EMOJI: dict[SubscriptionStatus, str] = {
    SubscriptionStatus.TRIAL: "🆓",
    SubscriptionStatus.ACTIVE: "✅",
    SubscriptionStatus.PAUSED: "⏸️",
    SubscriptionStatus.CANCELLED: "🛑",
}


def render_listing(listing: MarketplaceListing) -> str:
    """Format a listing for ops display.

    No-secret-leak: never includes Stripe customer / invoice IDs,
    card data, payout account numbers — the dataclass simply
    doesn't carry those fields.
    """

    emoji = _LISTING_STATUS_EMOJI[listing.status]
    lines = [
        f"{emoji} {listing.name} ({listing.listing_id})",
        f"  by: {listing.author_anonymous_handle}",
        f"  kind: {listing.strategy_kind}",
        f"  halal: {listing.halal_cert_level.value}",
        f"  license: {listing.license_term.value}",
        f"  price: ${listing.monthly_price_usd:.2f}/mo",
        f"  status: {listing.status.value}",
    ]
    if listing.published_at is not None:
        lines.append(f"  published: {listing.published_at.date().isoformat()}")
    return "\n".join(lines)


def render_subscription(subscription: Subscription) -> str:
    """Format a subscription for ops display."""

    emoji = _SUBSCRIPTION_STATUS_EMOJI[subscription.status]
    lines = [
        f"{emoji} subscription {subscription.subscription_id}",
        f"  listing: {subscription.listing_id}",
        f"  subscriber: {subscription.subscriber_anonymous_handle}",
        f"  status: {subscription.status.value}",
        f"  trial ends: {subscription.trial_end_at.date().isoformat()}",
    ]
    if subscription.cancelled_at is not None:
        lines.append(f"  cancelled: {subscription.cancelled_at.date().isoformat()}")
    return "\n".join(lines)


def render_split(split: RevenueSplit) -> str:
    """Format a revenue split for ops display."""

    return (
        f"💰 split for {split.listing_id}: "
        f"revenue ${split.cycle_revenue_usd:.2f} → "
        f"author ${split.author_amount_usd:.2f} / "
        f"platform ${split.platform_amount_usd:.2f}"
    )


__all__ = [
    "DEFAULT_POLICY",
    "HalalCertLevel",
    "LicenseTerm",
    "ListingStatus",
    "ListingViolationError",
    "MarketplaceListing",
    "MarketplacePolicy",
    "RevenueSplit",
    "Subscription",
    "SubscriptionStatus",
    "cancel_subscription",
    "compute_split",
    "convert_to_active",
    "pause_subscription",
    "publish_listing",
    "render_listing",
    "render_split",
    "render_subscription",
    "resume_subscription",
    "start_subscription",
    "take_down_listing",
    "validate_listing",
]
