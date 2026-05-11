"""Strategy gallery — Round-5 Wave 17.C.

Where Wave 17.B publishes individual trade ideas, this module
publishes *full strategies* (parameter sets that drive an ongoing
allocation logic). Subscribers pay a periodic Wakalah fee for the
right to mirror the strategy; the platform takes a cut as a flat
operator fee. Performance carries are forbidden — same structural
pin as the idea marketplace.

This module is the **strategy registry + subscription billing +
performance ledger**:

1. Authors publish a `StrategyListing` with parameters, risk band,
   visibility (FREE / PAID), and a fixed periodic Wakalah fee.
2. Subscribers pay a flat fee per billing period (monthly / quarterly /
   annual). The platform takes `platform_fee_pct` of the gross fee
   as a flat operator service charge (Wakalah-on-Wakalah is fine
   because both legs are flat services).
3. Each strategy carries a periodic performance ledger; subscribers
   can review historical returns before subscribing.

Pinned semantics:

- **Closed-set Visibility**: FREE / PAID.
- **Closed-set BillingPeriod**: MONTHLY / QUARTERLY / ANNUAL.
- **Closed-set ListingStatus**: DRAFT / PUBLISHED / DEPRECATED /
  ARCHIVED.
- **Performance carry forbidden** — same as 17.B.
- **Platform fee is a flat percentage in [0, 0.30]** of the gross
  Wakalah fee. Treated as the platform's service fee, not as a
  performance haircut.
- **Performance ledger is append-only** with hash-chaining for
  tamper-evidence (mirrors the committee transcript log).
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — author/subscriber IDs masked.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any


class Visibility(str, Enum):
    """Closed-set visibility ladder."""

    FREE = "free"
    PAID = "paid"


class BillingPeriod(str, Enum):
    """Closed-set billing-period ladder."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


_BILLING_DAYS: dict[BillingPeriod, int] = {
    BillingPeriod.MONTHLY: 30,
    BillingPeriod.QUARTERLY: 90,
    BillingPeriod.ANNUAL: 365,
}


class ListingStatus(str, Enum):
    """Closed-set listing-status ladder."""

    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    """Author no longer maintains; existing subscribers grandfathered."""
    ARCHIVED = "archived"
    """Removed entirely; no new or grandfathered subscribers."""


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class StrategyListing:
    """A published strategy listing."""

    listing_id: str
    author_id: str
    name: str
    description: str
    visibility: Visibility
    risk_band: RiskBand
    billing_period: BillingPeriod
    wakalah_fee_per_period_usd: float
    """Flat fee per billing period."""
    platform_fee_pct: float
    """Platform's slice of each fee. In [0, 0.30]."""
    parameters: dict[str, Any] = field(default_factory=dict)
    """Strategy parameters — opaque to this layer."""
    published_at: datetime | None = None
    status: ListingStatus = ListingStatus.DRAFT

    def __post_init__(self) -> None:
        if not self.listing_id or not self.listing_id.strip():
            raise ValueError("listing_id must be non-empty")
        if not self.author_id or not self.author_id.strip():
            raise ValueError("author_id must be non-empty")
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if len(self.name) > 100:
            raise ValueError("name must be ≤ 100 chars")
        if not self.description.strip():
            raise ValueError("description must be non-empty")
        if len(self.description) > 1000:
            raise ValueError("description must be ≤ 1000 chars")
        if self.wakalah_fee_per_period_usd < 0:
            raise ValueError("wakalah_fee_per_period_usd must be non-negative")
        if self.wakalah_fee_per_period_usd > 1000:
            raise ValueError(
                "wakalah_fee_per_period_usd > $1000 is suspicious; tighten before publishing"
            )
        if self.visibility is Visibility.FREE and self.wakalah_fee_per_period_usd > 0:
            raise ValueError("FREE listing must have zero fee")
        if not 0.0 <= self.platform_fee_pct <= 0.30:
            raise ValueError("platform_fee_pct must be in [0, 0.30]")
        if self.published_at is None and self.status is not ListingStatus.DRAFT:
            raise ValueError("published_at required when status != DRAFT")

    def author_take_per_period(self) -> float:
        return self.wakalah_fee_per_period_usd * (1 - self.platform_fee_pct)

    def platform_take_per_period(self) -> float:
        return self.wakalah_fee_per_period_usd * self.platform_fee_pct


@dataclass(frozen=True)
class GallerySubscription:
    """A subscriber's strategy subscription."""

    subscription_id: str
    listing_id: str
    subscriber_id: str
    started_at: datetime
    next_billing_at: datetime
    cancelled_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.subscription_id or not self.subscription_id.strip():
            raise ValueError("subscription_id must be non-empty")
        if not self.listing_id or not self.listing_id.strip():
            raise ValueError("listing_id must be non-empty")
        if not self.subscriber_id or not self.subscriber_id.strip():
            raise ValueError("subscriber_id must be non-empty")
        if self.next_billing_at <= self.started_at:
            raise ValueError("next_billing_at must be after started_at")
        if self.cancelled_at is not None and self.cancelled_at < self.started_at:
            raise ValueError("cancelled_at cannot precede started_at")

    def is_active(self, now: datetime) -> bool:
        if self.cancelled_at is None:
            return True
        return now < self.cancelled_at


def subscribe(
    listing: StrategyListing,
    *,
    subscription_id: str,
    subscriber_id: str,
    started_at: datetime,
) -> GallerySubscription:
    """Open a subscription. Author cannot subscribe to own strategy."""
    if listing.author_id == subscriber_id:
        raise ValueError("author cannot subscribe to own strategy")
    if listing.status not in (ListingStatus.PUBLISHED, ListingStatus.DEPRECATED):
        raise ValueError(f"cannot subscribe to listing in {listing.status.value} state")
    if listing.status is ListingStatus.DEPRECATED:
        raise ValueError("DEPRECATED listings only grandfather existing subscribers")
    days = _BILLING_DAYS[listing.billing_period]
    return GallerySubscription(
        subscription_id=subscription_id,
        listing_id=listing.listing_id,
        subscriber_id=subscriber_id,
        started_at=started_at,
        next_billing_at=started_at + timedelta(days=days),
    )


def cancel(subscription: GallerySubscription, *, cancelled_at: datetime) -> GallerySubscription:
    """Cancel a subscription — does not refund the current period."""
    if subscription.cancelled_at is not None:
        raise ValueError("subscription already cancelled")
    return replace(subscription, cancelled_at=cancelled_at)


@dataclass(frozen=True)
class FeeSplit:
    """Output of `compute_fee_split` for one billing event."""

    listing_id: str
    subscription_id: str
    gross_fee_usd: float
    platform_take_usd: float
    author_take_usd: float


def compute_fee_split(
    listing: StrategyListing,
    subscription: GallerySubscription,
) -> FeeSplit:
    """Compute the fee split for one billing event."""
    if subscription.listing_id != listing.listing_id:
        raise ValueError("subscription/listing id mismatch")
    gross = listing.wakalah_fee_per_period_usd
    platform = gross * listing.platform_fee_pct
    author = gross - platform
    return FeeSplit(
        listing_id=listing.listing_id,
        subscription_id=subscription.subscription_id,
        gross_fee_usd=gross,
        platform_take_usd=platform,
        author_take_usd=author,
    )


@dataclass(frozen=True)
class PerformanceEntry:
    """One append-only row in a strategy's performance ledger."""

    listing_id: str
    period_end: date
    return_pct: float
    drawdown_pct: float
    """Realised drawdown over the period — non-negative."""
    benchmark_return_pct: float
    n_subscribers: int
    prev_hash: str

    def __post_init__(self) -> None:
        if not self.listing_id or not self.listing_id.strip():
            raise ValueError("listing_id must be non-empty")
        if not -1.0 <= self.return_pct <= 5.0:
            raise ValueError("return_pct outside reasonable bounds")
        if self.drawdown_pct < 0:
            raise ValueError("drawdown_pct must be non-negative")
        if not -1.0 <= self.benchmark_return_pct <= 5.0:
            raise ValueError("benchmark_return_pct outside reasonable bounds")
        if self.n_subscribers < 0:
            raise ValueError("n_subscribers must be non-negative")

    def payload_for_hash(self) -> dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "period_end": self.period_end.isoformat(),
            "return_pct": self.return_pct,
            "drawdown_pct": self.drawdown_pct,
            "benchmark_return_pct": self.benchmark_return_pct,
            "n_subscribers": self.n_subscribers,
            "prev_hash": self.prev_hash,
        }

    def entry_hash(self) -> str:
        j = json.dumps(self.payload_for_hash(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(j.encode()).hexdigest()


def append_performance(
    ledger: tuple[PerformanceEntry, ...],
    new_entry: PerformanceEntry,
) -> tuple[PerformanceEntry, ...]:
    """Append a performance entry to the ledger. Pinned: chain
    integrity is enforced (prev_hash must equal the previous entry's
    hash); period_end must strictly increase."""
    if ledger:
        last = ledger[-1]
        if new_entry.prev_hash != last.entry_hash():
            raise ValueError("prev_hash does not match the latest entry's hash")
        if new_entry.period_end <= last.period_end:
            raise ValueError("period_end must strictly increase")
        if new_entry.listing_id != last.listing_id:
            raise ValueError("listing_id mismatch")
    else:
        if new_entry.prev_hash != "":
            raise ValueError("first entry's prev_hash must be empty")
    return (*ledger, new_entry)


def verify_ledger(ledger: Iterable[PerformanceEntry]) -> bool:
    """True iff the ledger's hash chain is intact."""
    prev = ""
    last_period: date | None = None
    listing_id: str | None = None
    for e in ledger:
        if e.prev_hash != prev:
            return False
        if last_period is not None and e.period_end <= last_period:
            return False
        if listing_id is not None and e.listing_id != listing_id:
            return False
        prev = e.entry_hash()
        last_period = e.period_end
        listing_id = e.listing_id
    return True


def transition_listing(
    listing: StrategyListing,
    *,
    new_status: ListingStatus,
    at: datetime,
) -> StrategyListing:
    """Transition a listing to a new status. Pinned legal moves:

    DRAFT → PUBLISHED
    PUBLISHED → DEPRECATED
    PUBLISHED → ARCHIVED
    DEPRECATED → ARCHIVED
    ARCHIVED is terminal.
    """
    legal: dict[ListingStatus, set[ListingStatus]] = {
        ListingStatus.DRAFT: {ListingStatus.PUBLISHED, ListingStatus.ARCHIVED},
        ListingStatus.PUBLISHED: {
            ListingStatus.DEPRECATED,
            ListingStatus.ARCHIVED,
        },
        ListingStatus.DEPRECATED: {ListingStatus.ARCHIVED},
        ListingStatus.ARCHIVED: set(),
    }
    if new_status not in legal[listing.status]:
        raise ValueError(f"illegal transition {listing.status.value} → {new_status.value}")
    new_pub = listing.published_at
    if new_status is ListingStatus.PUBLISHED and new_pub is None:
        new_pub = at
    return replace(listing, status=new_status, published_at=new_pub)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


def render_listing(listing: StrategyListing) -> str:
    visibility_emoji = "🆓" if listing.visibility is Visibility.FREE else "💸"
    return (
        f"{visibility_emoji} Strategy {listing.listing_id} "
        f"({listing.status.value}): {listing.name} "
        f"[risk={listing.risk_band.value}]\n"
        f"  • Author: {_mask(listing.author_id)} | "
        f"${listing.wakalah_fee_per_period_usd:.2f}/{listing.billing_period.value} "
        f"(platform {listing.platform_fee_pct * 100:.0f}%, author keeps "
        f"${listing.author_take_per_period():.2f})"
    )


def render_fee_split(split: FeeSplit) -> str:
    return (
        f"💰 Billed {split.subscription_id}: gross=${split.gross_fee_usd:.2f}, "
        f"platform=${split.platform_take_usd:.2f}, author=${split.author_take_usd:.2f}"
    )


def render_ledger(ledger: Iterable[PerformanceEntry]) -> str:
    rows = tuple(ledger)
    if not rows:
        return "📊 No performance entries."
    lines = [f"📊 Performance ledger ({len(rows)} entries):"]
    for e in rows:
        lines.append(
            f"  • {e.period_end.isoformat()}: "
            f"ret={e.return_pct * 100:+.2f}% "
            f"vs bench={e.benchmark_return_pct * 100:+.2f}%, "
            f"DD={e.drawdown_pct * 100:.2f}%, subs={e.n_subscribers}"
        )
    return "\n".join(lines)
