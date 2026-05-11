"""Halal trade idea marketplace — Round-5 Wave 17.B.

Users publish trade ideas (entry, target, stop, rationale, risk band);
other users subscribe + auto-trade. The author earns a Wakalah service
fee per follower (NOT a performance carry — that would tilt toward
gharar/riba).

This module is the **idea + subscription + fee + attribution layer**.
The execution side (auto-trade routing) lives elsewhere; this layer
records what was published, who subscribed, and how the per-idea P&L
attributes back to the author.

Pinned semantics:

- **Wakalah fee = flat fixed amount per subscription**, charged once
  on subscription. NOT % of P&L. The closed-set FeeStructure ladder
  (FLAT_PER_SUBSCRIPTION / FLAT_PER_FOLLOW_DAY) makes this enforceable
  at the type level.
- **No performance fee.** The `assert_no_performance_carry` predicate
  rejects any structure that takes a fraction of profit.
- **Idea status ladder** — DRAFT / PUBLISHED / TRIGGERED / CLOSED /
  REVOKED. Publication is one-way: a PUBLISHED idea cannot return to
  DRAFT (immutable timestamp pin).
- **Risk band must be on the idea**, not negotiated per follower —
  prevents the author from quietly increasing risk after others have
  subscribed.
- **Halal-screen pin**: `assert_idea_compliant` rejects ideas that
  reference haram-screened tickers (caller supplies the screen
  predicate).
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — author/follower IDs masked.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from typing import Any


class IdeaStatus(str, Enum):
    """Closed-set idea lifecycle ladder."""

    DRAFT = "draft"
    PUBLISHED = "published"
    TRIGGERED = "triggered"
    CLOSED = "closed"
    REVOKED = "revoked"


class FeeStructure(str, Enum):
    """Closed-set Wakalah fee structures.

    Both options are *flat* amounts — never a fraction of P&L. The
    enum's closedness is the structural pin against riba creep.
    """

    FLAT_PER_SUBSCRIPTION = "flat_per_subscription"
    FLAT_PER_FOLLOW_DAY = "flat_per_follow_day"


class RiskBand(str, Enum):
    """Closed-set risk classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IdeaSide(str, Enum):
    """Closed-set directional ladder. SKIP/HOLD are not publishable
    — you can't sell a non-action."""

    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class TradeIdea:
    """One published trade idea."""

    idea_id: str
    author_id: str
    ticker: str
    side: IdeaSide
    entry_price: float
    target_price: float
    stop_price: float
    risk_band: RiskBand
    rationale_summary: str
    """Short author-provided rationale; longer reasoning is stored
    elsewhere if needed."""
    published_at: datetime
    horizon_days: int
    fee_structure: FeeStructure
    fee_amount_usd: float
    status: IdeaStatus = IdeaStatus.PUBLISHED
    closed_at: datetime | None = None
    realised_return_pct: float | None = None

    def __post_init__(self) -> None:
        if not self.idea_id or not self.idea_id.strip():
            raise ValueError("idea_id must be non-empty")
        if not self.author_id or not self.author_id.strip():
            raise ValueError("author_id must be non-empty")
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if self.target_price <= 0:
            raise ValueError("target_price must be positive")
        if self.stop_price <= 0:
            raise ValueError("stop_price must be positive")
        # LONG: target > entry > stop; SHORT: target < entry < stop.
        if self.side is IdeaSide.LONG:
            if not self.stop_price < self.entry_price < self.target_price:
                raise ValueError("LONG ideas require stop < entry < target")
        else:
            if not self.target_price < self.entry_price < self.stop_price:
                raise ValueError("SHORT ideas require target < entry < stop")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if not self.rationale_summary.strip():
            raise ValueError("rationale_summary must be non-empty")
        if len(self.rationale_summary) > 500:
            raise ValueError("rationale_summary must be ≤ 500 chars")
        if self.fee_amount_usd < 0:
            raise ValueError("fee_amount_usd must be non-negative")
        if self.fee_amount_usd > 100:
            raise ValueError(
                "fee_amount_usd > $100 is suspicious for a per-subscription "
                "Wakalah; tighten before publishing"
            )
        if self.closed_at is not None and self.closed_at <= self.published_at:
            raise ValueError("closed_at must be after published_at")
        if self.realised_return_pct is not None and not -1.0 <= self.realised_return_pct <= 5.0:
            raise ValueError("realised_return_pct outside reasonable bounds")

    def reward_to_risk(self) -> float:
        """Pinned ratio = (target - entry) / (entry - stop) for LONG;
        (entry - target) / (stop - entry) for SHORT."""
        if self.side is IdeaSide.LONG:
            return (self.target_price - self.entry_price) / (self.entry_price - self.stop_price)
        return (self.entry_price - self.target_price) / (self.stop_price - self.entry_price)


@dataclass(frozen=True)
class Subscription:
    """One follower's subscription to an idea."""

    subscription_id: str
    idea_id: str
    follower_id: str
    subscribed_at: datetime
    fee_paid_usd: float
    unsubscribed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.subscription_id or not self.subscription_id.strip():
            raise ValueError("subscription_id must be non-empty")
        if not self.idea_id or not self.idea_id.strip():
            raise ValueError("idea_id must be non-empty")
        if not self.follower_id or not self.follower_id.strip():
            raise ValueError("follower_id must be non-empty")
        if self.fee_paid_usd < 0:
            raise ValueError("fee_paid_usd must be non-negative")
        if self.unsubscribed_at is not None and self.unsubscribed_at <= self.subscribed_at:
            raise ValueError("unsubscribed_at must be after subscribed_at")

    def is_active(self, now: datetime | None = None) -> bool:
        if self.unsubscribed_at is None:
            return True
        if now is None:
            return False
        return now < self.unsubscribed_at


def assert_no_performance_carry(idea: TradeIdea) -> None:
    """Pin: the closed-set FeeStructure already excludes performance
    carries; this helper exists so callers can fail-loud if a future
    extension tries to sneak one in."""
    if idea.fee_structure not in (
        FeeStructure.FLAT_PER_SUBSCRIPTION,
        FeeStructure.FLAT_PER_FOLLOW_DAY,
    ):
        raise ValueError(
            f"fee_structure {idea.fee_structure.value} is not a flat "
            "Wakalah fee — performance carries are forbidden under "
            "halal marketplace rules"
        )


def assert_idea_compliant(
    idea: TradeIdea,
    *,
    is_ticker_halal: Callable[[str], bool],
) -> None:
    """Reject ideas referencing haram-screened tickers."""
    if not is_ticker_halal(idea.ticker):
        raise ValueError(f"ticker {idea.ticker} is not halal-compliant — idea cannot publish")


def subscribe(
    idea: TradeIdea,
    *,
    subscription_id: str,
    follower_id: str,
    subscribed_at: datetime,
) -> Subscription:
    """Create a Subscription, charging the configured Wakalah fee."""
    if idea.author_id == follower_id:
        raise ValueError("author cannot subscribe to their own idea")
    if idea.status not in (IdeaStatus.PUBLISHED, IdeaStatus.TRIGGERED):
        raise ValueError(f"cannot subscribe to idea in {idea.status.value} state")
    fee_paid = idea.fee_amount_usd
    return Subscription(
        subscription_id=subscription_id,
        idea_id=idea.idea_id,
        follower_id=follower_id,
        subscribed_at=subscribed_at,
        fee_paid_usd=fee_paid,
    )


def unsubscribe(subscription: Subscription, *, unsubscribed_at: datetime) -> Subscription:
    """Mark the subscription as cancelled; returns a new frozen object."""
    if subscription.unsubscribed_at is not None:
        raise ValueError("subscription already unsubscribed")
    return replace(subscription, unsubscribed_at=unsubscribed_at)


def transition_idea(
    idea: TradeIdea,
    *,
    new_status: IdeaStatus,
    at: datetime,
    realised_return_pct: float | None = None,
) -> TradeIdea:
    """Transition an idea to a new status. Pinned legal moves:

    PUBLISHED → TRIGGERED (entry hit)
    PUBLISHED → REVOKED (author pulls before any follower triggered)
    TRIGGERED → CLOSED (target/stop/horizon)
    PUBLISHED → CLOSED (horizon expired without trigger)

    Reverse / out-of-band transitions raise.
    """
    legal: dict[IdeaStatus, set[IdeaStatus]] = {
        IdeaStatus.DRAFT: {IdeaStatus.PUBLISHED, IdeaStatus.REVOKED},
        IdeaStatus.PUBLISHED: {
            IdeaStatus.TRIGGERED,
            IdeaStatus.CLOSED,
            IdeaStatus.REVOKED,
        },
        IdeaStatus.TRIGGERED: {IdeaStatus.CLOSED},
        IdeaStatus.CLOSED: set(),
        IdeaStatus.REVOKED: set(),
    }
    if new_status not in legal[idea.status]:
        raise ValueError(f"illegal transition {idea.status.value} → {new_status.value}")
    if new_status is IdeaStatus.CLOSED and realised_return_pct is None:
        raise ValueError("CLOSED requires realised_return_pct")
    return replace(
        idea,
        status=new_status,
        closed_at=at if new_status is IdeaStatus.CLOSED else idea.closed_at,
        realised_return_pct=(
            realised_return_pct if new_status is IdeaStatus.CLOSED else idea.realised_return_pct
        ),
    )


@dataclass(frozen=True)
class AuthorAttribution:
    """Per-author roll-up of fees + idea performance."""

    author_id: str
    n_ideas_published: int
    n_subscriptions: int
    total_wakalah_fees_usd: float
    closed_ideas_avg_return_pct: float
    """Mean of realised_return_pct across closed ideas. Zero when
    no closed ideas yet."""
    win_rate: float
    """Fraction of CLOSED ideas with positive realised_return_pct."""


def author_attribution(
    ideas: Iterable[TradeIdea],
    subscriptions: Iterable[Subscription],
) -> tuple[AuthorAttribution, ...]:
    """Compute per-author roll-ups across an idea + subscription set."""
    by_author: dict[str, dict[str, Any]] = {}
    for i in ideas:
        rec = by_author.setdefault(
            i.author_id,
            {
                "ideas": [],
                "fees": 0.0,
                "n_subs": 0,
            },
        )
        rec["ideas"].append(i)
    sub_by_idea: dict[str, list[Subscription]] = {}
    for s in subscriptions:
        sub_by_idea.setdefault(s.idea_id, []).append(s)
    for i_list in (rec["ideas"] for rec in by_author.values()):
        pass  # no-op; we walk again below for clarity.
    for author, rec in by_author.items():
        for idea in rec["ideas"]:
            subs = sub_by_idea.get(idea.idea_id, [])
            rec["fees"] += sum(s.fee_paid_usd for s in subs)
            rec["n_subs"] += len(subs)
    out: list[AuthorAttribution] = []
    for author, rec in by_author.items():
        ideas_l: list[TradeIdea] = rec["ideas"]
        closed = [
            i
            for i in ideas_l
            if i.status is IdeaStatus.CLOSED and i.realised_return_pct is not None
        ]
        if closed:
            avg_ret = sum(i.realised_return_pct or 0.0 for i in closed) / len(closed)
            wins = sum(1 for i in closed if (i.realised_return_pct or 0.0) > 0)
            win_rate = wins / len(closed)
        else:
            avg_ret = 0.0
            win_rate = 0.0
        out.append(
            AuthorAttribution(
                author_id=author,
                n_ideas_published=len(ideas_l),
                n_subscriptions=int(rec["n_subs"]),
                total_wakalah_fees_usd=float(rec["fees"]),
                closed_ideas_avg_return_pct=avg_ret,
                win_rate=win_rate,
            )
        )
    out.sort(key=lambda a: a.author_id)
    return tuple(out)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


def render_idea(idea: TradeIdea) -> str:
    """Operator-readable summary."""
    status_emoji = {
        IdeaStatus.DRAFT: "✏️",
        IdeaStatus.PUBLISHED: "📢",
        IdeaStatus.TRIGGERED: "🎯",
        IdeaStatus.CLOSED: "🔒",
        IdeaStatus.REVOKED: "🚫",
    }[idea.status]
    return (
        f"{status_emoji} Idea {idea.idea_id} ({idea.status.value}): "
        f"{idea.side.value} {idea.ticker} entry={idea.entry_price:.2f} "
        f"target={idea.target_price:.2f} stop={idea.stop_price:.2f} "
        f"R:R={idea.reward_to_risk():.2f} "
        f"risk={idea.risk_band.value}\n"
        f"  • Author: {_mask(idea.author_id)} | fee=${idea.fee_amount_usd:.2f} "
        f"({idea.fee_structure.value})"
    )


def render_attribution(attr: AuthorAttribution) -> str:
    return (
        f"🏷️ Author {_mask(attr.author_id)}: "
        f"{attr.n_ideas_published} ideas / {attr.n_subscriptions} subs / "
        f"${attr.total_wakalah_fees_usd:.2f} fees / "
        f"avg-return {attr.closed_ideas_avg_return_pct * 100:+.2f}% / "
        f"win {attr.win_rate * 100:.2f}%"
    )
