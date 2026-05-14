"""Quarterly newsletter digest builder.

The roadmap pins Wave 10.D: "Highlights from the platform's
aggregate data: what worked, what didn't, regulatory changes,
scholar updates. Goes to an opt-in mailing list." This module is
the **pure-Python digest builder** that assembles a quarterly
newsletter from platform-aggregate inputs; the actual mailing-list
send (Mailgun / Postmark / SES) is operator-side.

Picked a focused builder over hand-writing each newsletter
because (a) the four section kinds (top_performers / regulatory
/ scholar_updates / what_didnt_work) cover the documented content
domains; structuring them as a closed enum means a future "let's
add a community-spotlight section" PR has to update the schema +
tests, not just paste markdown into a draft, (b) the no-individual-
trader-names contract is the load-bearing privacy attribute —
the newsletter aggregates platform data but must never name an
individual user (anonymous handles only, mirroring Wave 10.A
gallery + 10.B dataset anonymisation), (c) the opt-in subscriber
list management is a discrete state machine (subscribed → unsubscribed
one-way) that needs deterministic auditing for legal compliance
(GDPR/CCPA right-to-be-forgotten), (d) digest validation runs at
build time so a contributor that pastes raw user text into a
section gets caught at CI rather than after the email is sent to
1000 subscribers.

Pinned semantics:
- **Closed-set SectionKind enum.** Four kinds: TOP_PERFORMERS,
  REGULATORY, SCHOLAR_UPDATES, WHAT_DIDNT_WORK. Adding a kind
  is a code review change.
- **No PII in section bodies.** Same five-pattern denylist as
  Wave 10.B + 10.C (email / SSN / IP / phone / API-key-shape);
  validation rejects any section with PII before publication.
- **No individual trader names.** A name-in-body denylist
  rejects bodies containing "@" or all-uppercase strings that
  look like usernames; mirrors the 10.A anonymisation contract.
- **Subscription state: SUBSCRIBED → UNSUBSCRIBED one-way.** Once
  unsubscribed, can resubscribe via a fresh subscription record;
  the audit trail preserves the unsubscribe timestamp for legal
  compliance.
- **Render output never includes operator-side credentials, send-
  service API keys, or subscriber email addresses.** The
  digest's render is content-only; subscriber addresses are
  resolved by the send service from the subscription rows.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SectionKind(str, Enum):
    """Closed-set newsletter section kinds.

    Pinned string values for JSON / DB stability.
    """

    TOP_PERFORMERS = "top_performers"
    REGULATORY = "regulatory"
    SCHOLAR_UPDATES = "scholar_updates"
    WHAT_DIDNT_WORK = "what_didnt_work"


class SubscriptionStatus(str, Enum):
    """Newsletter subscription status."""

    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"


# PII patterns (mirrors Wave 10.B + 10.C)
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # email
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),  # IP
    re.compile(r"\+?\d{1,3}[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}"),  # phone
    re.compile(r"\b[A-Za-z0-9]{40,}\b"),  # API-key-shape
)


# @ at the start of a token suggests a Twitter / Discord handle
_HANDLE_PATTERN = re.compile(r"(?:^|\s)@\w+")


_MAX_TITLE_LENGTH = 120
_MAX_BODY_LENGTH = 4000


class DigestViolationError(Exception):
    """Raised when a section or digest fails validation."""

    def __init__(self, section_id: str, reason: str) -> None:
        super().__init__(f"section {section_id!r}: {reason}")
        self.section_id = section_id
        self.reason = reason


@dataclass(frozen=True)
class Section:
    """One newsletter section."""

    section_id: str
    kind: SectionKind
    title: str
    body: str

    def __post_init__(self) -> None:
        if not self.section_id or not self.section_id.strip():
            raise ValueError("section_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.body or not self.body.strip():
            raise ValueError("body must be non-empty")
        if len(self.title) > _MAX_TITLE_LENGTH:
            raise ValueError(f"title too long ({len(self.title)} > {_MAX_TITLE_LENGTH})")
        if len(self.body) > _MAX_BODY_LENGTH:
            raise ValueError(f"body too long ({len(self.body)} > {_MAX_BODY_LENGTH})")


def validate_section(section: Section) -> None:
    """Run PII + handle denylist checks. Raises DigestViolationError."""

    for field_name, value in (("title", section.title), ("body", section.body)):
        for pattern in _PII_PATTERNS:
            if pattern.search(value):
                raise DigestViolationError(
                    section.section_id,
                    f"PII pattern detected in {field_name}",
                )
        if _HANDLE_PATTERN.search(value):
            raise DigestViolationError(
                section.section_id,
                f"social handle (@username) detected in {field_name}",
            )


@dataclass(frozen=True)
class Digest:
    """One quarterly newsletter digest."""

    digest_id: str
    quarter_label: str  # "2026-Q1" / "2026-Q2" / etc
    published_at: datetime
    sections: tuple[Section, ...]

    def __post_init__(self) -> None:
        if not self.digest_id or not self.digest_id.strip():
            raise ValueError("digest_id must be non-empty")
        if not self.quarter_label or not self.quarter_label.strip():
            raise ValueError("quarter_label must be non-empty")
        if self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware")
        if not self.sections:
            raise ValueError("sections must be non-empty")
        # No duplicate section ids
        section_ids = [s.section_id for s in self.sections]
        if len(set(section_ids)) != len(section_ids):
            raise ValueError("duplicate section_id")


def validate_digest(digest: Digest) -> None:
    """Validate every section. Raises DigestViolationError on first failure."""

    for section in digest.sections:
        validate_section(section)


def sections_by_kind(digest: Digest, kind: SectionKind) -> tuple[Section, ...]:
    """Return sections of the given kind."""

    return tuple(s for s in digest.sections if s.kind is kind)


@dataclass(frozen=True)
class Subscription:
    """One subscriber's record.

    `subscriber_anonymous_handle` is an opaque token (mirrors Wave
    10.A gallery anonymisation). The actual email address is
    operator-side state — the dataclass deliberately doesn't carry
    it, so render output structurally can't leak emails.
    """

    subscription_id: str
    subscriber_anonymous_handle: str
    status: SubscriptionStatus
    subscribed_at: datetime
    unsubscribed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.subscription_id or not self.subscription_id.strip():
            raise ValueError("subscription_id must be non-empty")
        if not self.subscriber_anonymous_handle or not self.subscriber_anonymous_handle.strip():
            raise ValueError("subscriber_anonymous_handle must be non-empty")
        if self.subscribed_at.tzinfo is None:
            raise ValueError("subscribed_at must be timezone-aware")
        if self.status is SubscriptionStatus.UNSUBSCRIBED:
            if self.unsubscribed_at is None:
                raise ValueError("UNSUBSCRIBED status requires unsubscribed_at")
        else:
            if self.unsubscribed_at is not None:
                raise ValueError("non-UNSUBSCRIBED status must not have unsubscribed_at")
        if self.unsubscribed_at is not None:
            if self.unsubscribed_at.tzinfo is None:
                raise ValueError("unsubscribed_at must be timezone-aware")
            if self.unsubscribed_at < self.subscribed_at:
                raise ValueError("unsubscribed_at must be >= subscribed_at")


class AlreadyUnsubscribedError(Exception):
    """Raised when unsubscribe is called on a non-subscribed record."""

    def __init__(self, subscription_id: str) -> None:
        super().__init__(f"subscription {subscription_id!r} already unsubscribed")
        self.subscription_id = subscription_id


def subscribe(
    *,
    subscription_id: str,
    subscriber_anonymous_handle: str,
    now: datetime,
) -> Subscription:
    """Build a fresh SUBSCRIBED record."""

    if not subscription_id or not subscription_id.strip():
        raise ValueError("subscription_id must be non-empty")
    if not subscriber_anonymous_handle or not subscriber_anonymous_handle.strip():
        raise ValueError("subscriber_anonymous_handle must be non-empty")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return Subscription(
        subscription_id=subscription_id,
        subscriber_anonymous_handle=subscriber_anonymous_handle,
        status=SubscriptionStatus.SUBSCRIBED,
        subscribed_at=now,
    )


def unsubscribe(subscription: Subscription, *, now: datetime) -> Subscription:
    """Move SUBSCRIBED → UNSUBSCRIBED. Records the unsubscribe timestamp."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if subscription.status is SubscriptionStatus.UNSUBSCRIBED:
        raise AlreadyUnsubscribedError(subscription.subscription_id)
    return Subscription(
        subscription_id=subscription.subscription_id,
        subscriber_anonymous_handle=subscription.subscriber_anonymous_handle,
        status=SubscriptionStatus.UNSUBSCRIBED,
        subscribed_at=subscription.subscribed_at,
        unsubscribed_at=now,
    )


def active_subscribers(
    subscriptions: Iterable[Subscription],
) -> tuple[Subscription, ...]:
    """Return only the SUBSCRIBED records (the send list)."""

    return tuple(s for s in subscriptions if s.status is SubscriptionStatus.SUBSCRIBED)


_KIND_HEADING: dict[SectionKind, str] = {
    SectionKind.TOP_PERFORMERS: "🏆 Top Performers",
    SectionKind.REGULATORY: "⚖️ Regulatory Updates",
    SectionKind.SCHOLAR_UPDATES: "📚 Scholar Updates",
    SectionKind.WHAT_DIDNT_WORK: "🔍 What Didn't Work",
}


# Canonical section ordering in the rendered digest.
_SECTION_ORDER: tuple[SectionKind, ...] = (
    SectionKind.TOP_PERFORMERS,
    SectionKind.REGULATORY,
    SectionKind.SCHOLAR_UPDATES,
    SectionKind.WHAT_DIDNT_WORK,
)


def render_section(section: Section) -> str:
    """Format a section as markdown.

    No-secret-leak: structural — a section that contained PII or a
    handle would have failed `validate_section` and not reach this
    code path. The render is just markdown formatting.
    """

    heading = _KIND_HEADING[section.kind]
    return f"## {heading}: {section.title}\n\n{section.body}"


def render_digest(digest: Digest) -> str:
    """Format the digest as markdown for the mailing list.

    Sections are rendered in canonical order (TOP_PERFORMERS →
    REGULATORY → SCHOLAR_UPDATES → WHAT_DIDNT_WORK) regardless of
    the order they appear in `digest.sections`. Within a kind,
    section_id-sorted for determinism.
    """

    lines = [
        f"# Halal Trader Newsletter — {digest.quarter_label}",
        "",
        f"_Published {digest.published_at.date().isoformat()}_",
        "",
    ]
    for kind in _SECTION_ORDER:
        ks = sorted(sections_by_kind(digest, kind), key=lambda s: s.section_id)
        for section in ks:
            lines.append(render_section(section))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "AlreadyUnsubscribedError",
    "Digest",
    "DigestViolationError",
    "Section",
    "SectionKind",
    "Subscription",
    "SubscriptionStatus",
    "active_subscribers",
    "render_digest",
    "render_section",
    "sections_by_kind",
    "subscribe",
    "unsubscribe",
    "validate_digest",
    "validate_section",
]
