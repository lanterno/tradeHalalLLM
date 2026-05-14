"""Community moderation policy + escalation engine.

The roadmap pins Wave 10.C: "Active community space for halal
traders. Operators share strategies, discuss compliance edge cases,
request features. Moderated by the core team." This module is the
**pure-Python content classifier + review state engine** that the
Discord / Slack bot integrations consume to triage messages
before a human moderator sees them.

Picked a focused classifier + state engine over plugging into
Discord auto-mod / Slack DLP because (a) "financial advice"
detection is the load-bearing safety attribute for this community
— the bot is paper-trading software, not investment advice; a
member casually saying "you should buy AAPL" creates regulatory
risk for the project unless flagged + replaced with a disclaimer,
(b) PII detection (member accidentally pastes their broker API
key thinking it's a public string) needs to run client-side
before the message reaches the channel, (c) the moderation state
machine (pending → auto_approved / flagged / escalated / removed)
gives moderators an audit trail for "why did this message get
removed?" without correlating across Discord audit logs + Slack
admin panels.

Pinned semantics:
- **Closed-set ContentClassification enum.** Five classes:
  CLEAN, SPAM, HARASSMENT, FINANCIAL_ADVICE, PII_LEAK. Adding a
  class is a code review change; the moderator dashboard
  groupings can't drift.
- **PII detection blocks the message immediately.** Any of email
  / SSN / IP / phone / API-key-shape patterns triggers
  PII_LEAK; the message is held with auto_removed=True so the
  channel never sees the leak.
- **Financial advice classification flags but doesn't auto-remove.**
  A member's "you should buy X" comment is FLAGGED and queued
  for human moderator; the bot replies with a disclaimer rather
  than silently deleting (transparent moderation builds trust).
- **Spam threshold: 3+ identical messages in 60s = SPAM.**
  Operator-tunable via policy.
- **Render output never includes raw message text in the audit
  row — only the classification + flagged_phrases.** Operators
  reviewing the moderation log don't need to re-read every
  message; they need to see the pattern that triggered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class ContentClassification(str, Enum):
    """Closed-set classification labels.

    Pinned string values for JSON / DB stability.
    """

    CLEAN = "clean"
    SPAM = "spam"
    HARASSMENT = "harassment"
    FINANCIAL_ADVICE = "financial_advice"
    PII_LEAK = "pii_leak"


class ReviewStatus(str, Enum):
    """Moderation review state machine."""

    PENDING = "pending"
    AUTO_APPROVED = "auto_approved"
    FLAGGED = "flagged"
    ESCALATED = "escalated"
    REMOVED = "removed"


_TERMINAL_STATUSES: frozenset[ReviewStatus] = frozenset(
    {ReviewStatus.AUTO_APPROVED, ReviewStatus.REMOVED}
)


# PII patterns (mirrors Wave 10.B + Wave 7.E)
_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "email"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "ssn"),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "ip"),
    (re.compile(r"\+?\d{1,3}[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}"), "phone"),
    # API-key-shaped (40+ alphanumeric, common for Binance/Alpaca/etc)
    (re.compile(r"\b[A-Za-z0-9]{40,}\b"), "api_key"),
)


# Financial-advice detection patterns (case-insensitive)
_FINANCIAL_ADVICE_PHRASES: tuple[str, ...] = (
    "you should buy",
    "you should sell",
    "you must buy",
    "you must sell",
    "guaranteed profit",
    "guaranteed return",
    "risk free",
    "risk-free profit",
    "definitely buy",
    "definitely sell",
    "i recommend buying",
    "i recommend selling",
)


# Harassment patterns (case-insensitive). Conservative — focuses on
# personal attacks, not profanity (community can self-moderate
# profanity via Discord's built-in word filter).
_HARASSMENT_PATTERNS: tuple[str, ...] = (
    "you are an idiot",
    "you're an idiot",
    "you're stupid",
    "shut up loser",
    "kill yourself",
    "go die",
)


_DEFAULT_SPAM_THRESHOLD = 3
_DEFAULT_SPAM_WINDOW = timedelta(seconds=60)


@dataclass(frozen=True)
class ModerationPolicy:
    """Operator-tunable moderation thresholds."""

    spam_threshold: int = _DEFAULT_SPAM_THRESHOLD
    spam_window: timedelta = _DEFAULT_SPAM_WINDOW
    detect_pii: bool = True

    def __post_init__(self) -> None:
        if self.spam_threshold < 2:
            raise ValueError("spam_threshold must be >= 2")
        if self.spam_window <= timedelta(0):
            raise ValueError("spam_window must be positive")


DEFAULT_POLICY = ModerationPolicy()


@dataclass(frozen=True)
class ClassificationResult:
    """Per-message classification.

    `flagged_phrases` lists the specific patterns that triggered
    the classification — operators see them in the moderation
    log without having to re-read the original message.
    """

    classification: ContentClassification
    flagged_phrases: tuple[str, ...]
    severity_score: float  # 0.0 = clean, 1.0 = max severity

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity_score <= 1.0:
            raise ValueError(f"severity_score {self.severity_score} must be in [0, 1]")


def classify(
    text: str,
    *,
    recent_identical_count: int = 0,
    policy: ModerationPolicy = DEFAULT_POLICY,
) -> ClassificationResult:
    """Classify message text.

    Priority order: PII_LEAK > HARASSMENT > FINANCIAL_ADVICE > SPAM
    > CLEAN. PII is the load-bearing block — once detected, no
    further classification needed (the leak is the maximally
    destructive outcome regardless of other content).

    `recent_identical_count` is the count of identical messages from
    the same user in the last `spam_window`; the SPAM gate fires when
    this >= spam_threshold (the current message itself is included
    in this count by the caller).
    """

    if not text or not text.strip():
        return ClassificationResult(
            classification=ContentClassification.CLEAN,
            flagged_phrases=(),
            severity_score=0.0,
        )

    # PII has highest priority — short-circuits everything
    if policy.detect_pii:
        pii_matches: list[str] = []
        for pattern, label in _PII_PATTERNS:
            if pattern.search(text):
                pii_matches.append(label)
        if pii_matches:
            return ClassificationResult(
                classification=ContentClassification.PII_LEAK,
                flagged_phrases=tuple(sorted(set(pii_matches))),
                severity_score=1.0,
            )

    text_lower = text.lower()

    # Harassment
    harassment_hits = [phrase for phrase in _HARASSMENT_PATTERNS if phrase in text_lower]
    if harassment_hits:
        return ClassificationResult(
            classification=ContentClassification.HARASSMENT,
            flagged_phrases=tuple(harassment_hits),
            severity_score=0.9,
        )

    # Financial advice
    fa_hits = [phrase for phrase in _FINANCIAL_ADVICE_PHRASES if phrase in text_lower]
    if fa_hits:
        return ClassificationResult(
            classification=ContentClassification.FINANCIAL_ADVICE,
            flagged_phrases=tuple(fa_hits),
            severity_score=0.6,
        )

    # Spam (last priority — only fires if nothing else matched)
    if recent_identical_count >= policy.spam_threshold:
        return ClassificationResult(
            classification=ContentClassification.SPAM,
            flagged_phrases=(f"identical_message_count={recent_identical_count}",),
            severity_score=0.4,
        )

    return ClassificationResult(
        classification=ContentClassification.CLEAN,
        flagged_phrases=(),
        severity_score=0.0,
    )


@dataclass(frozen=True)
class MessageReview:
    """One message's moderation state.

    Operations (`auto_approve`, `flag`, `escalate`, `remove`) return
    new state. The audit trail is the immutable history of status
    transitions.
    """

    message_id: str
    classification: ContentClassification
    status: ReviewStatus
    decided_at: datetime
    moderator: str = ""  # Empty for auto-decisions

    def __post_init__(self) -> None:
        if not self.message_id or not self.message_id.strip():
            raise ValueError("message_id must be non-empty")
        if self.decided_at.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")
        # Human-decided statuses (ESCALATED → REMOVED, manual approve)
        # require a moderator name; auto-decisions don't.
        if self.status in (ReviewStatus.ESCALATED, ReviewStatus.REMOVED):
            if not self.moderator or not self.moderator.strip():
                raise ValueError(f"{self.status.value} status requires moderator name")


class ReviewTransitionError(Exception):
    """Raised when a status transition violates the state machine."""

    def __init__(self, current: ReviewStatus, attempted: ReviewStatus) -> None:
        super().__init__(f"cannot transition from {current.value} to {attempted.value}")
        self.current = current
        self.attempted = attempted


def initial_review(
    *,
    message_id: str,
    classification: ContentClassification,
    now: datetime,
) -> MessageReview:
    """Build the initial PENDING review."""

    if not message_id or not message_id.strip():
        raise ValueError("message_id must be non-empty")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return MessageReview(
        message_id=message_id,
        classification=classification,
        status=ReviewStatus.PENDING,
        decided_at=now,
    )


def auto_decide(review: MessageReview, *, now: datetime) -> MessageReview:
    """Auto-decide based on classification.

    PII_LEAK → REMOVED (auto-removed; the message never sees the
    channel). HARASSMENT → ESCALATED (human review required).
    FINANCIAL_ADVICE → FLAGGED (visible but tagged; bot replies
    with disclaimer). SPAM → REMOVED. CLEAN → AUTO_APPROVED.

    Pinned: PII auto-removal is the load-bearing safety pin —
    moderators don't need to be online for an API key leak to be
    blocked.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if review.status is not ReviewStatus.PENDING:
        raise ReviewTransitionError(review.status, ReviewStatus.AUTO_APPROVED)

    classification = review.classification
    if classification is ContentClassification.PII_LEAK:
        return MessageReview(
            message_id=review.message_id,
            classification=classification,
            status=ReviewStatus.REMOVED,
            decided_at=now,
            moderator="auto",
        )
    if classification is ContentClassification.HARASSMENT:
        return MessageReview(
            message_id=review.message_id,
            classification=classification,
            status=ReviewStatus.ESCALATED,
            decided_at=now,
            moderator="auto",
        )
    if classification is ContentClassification.FINANCIAL_ADVICE:
        return MessageReview(
            message_id=review.message_id,
            classification=classification,
            status=ReviewStatus.FLAGGED,
            decided_at=now,
        )
    if classification is ContentClassification.SPAM:
        return MessageReview(
            message_id=review.message_id,
            classification=classification,
            status=ReviewStatus.REMOVED,
            decided_at=now,
            moderator="auto",
        )
    # CLEAN
    return MessageReview(
        message_id=review.message_id,
        classification=classification,
        status=ReviewStatus.AUTO_APPROVED,
        decided_at=now,
    )


def moderator_remove(review: MessageReview, *, moderator: str, now: datetime) -> MessageReview:
    """Human moderator removes a flagged or escalated message."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not moderator or not moderator.strip():
        raise ValueError("moderator must be non-empty")
    if review.status not in (ReviewStatus.FLAGGED, ReviewStatus.ESCALATED):
        raise ReviewTransitionError(review.status, ReviewStatus.REMOVED)
    return MessageReview(
        message_id=review.message_id,
        classification=review.classification,
        status=ReviewStatus.REMOVED,
        decided_at=now,
        moderator=moderator,
    )


def moderator_approve(review: MessageReview, *, moderator: str, now: datetime) -> MessageReview:
    """Human moderator approves a flagged or escalated message
    (false positive, e.g. a member quoting someone else's bad advice
    in context)."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not moderator or not moderator.strip():
        raise ValueError("moderator must be non-empty")
    if review.status not in (ReviewStatus.FLAGGED, ReviewStatus.ESCALATED):
        raise ReviewTransitionError(review.status, ReviewStatus.AUTO_APPROVED)
    return MessageReview(
        message_id=review.message_id,
        classification=review.classification,
        status=ReviewStatus.AUTO_APPROVED,
        decided_at=now,
        moderator="",  # Manual approval surfaces same as auto-approve
    )


def is_visible_to_channel(review: MessageReview) -> bool:
    """True if the message should appear in the public channel.

    AUTO_APPROVED, FLAGGED → visible (FLAGGED with disclaimer).
    PENDING → not yet visible (waiting for auto-decide).
    ESCALATED, REMOVED → hidden.
    """

    return review.status in (
        ReviewStatus.AUTO_APPROVED,
        ReviewStatus.FLAGGED,
    )


_CLASSIFICATION_EMOJI: dict[ContentClassification, str] = {
    ContentClassification.CLEAN: "✅",
    ContentClassification.SPAM: "📧",
    ContentClassification.HARASSMENT: "⚠️",
    ContentClassification.FINANCIAL_ADVICE: "💼",
    ContentClassification.PII_LEAK: "🔓",
}


_STATUS_EMOJI: dict[ReviewStatus, str] = {
    ReviewStatus.PENDING: "⏳",
    ReviewStatus.AUTO_APPROVED: "✅",
    ReviewStatus.FLAGGED: "🚩",
    ReviewStatus.ESCALATED: "📣",
    ReviewStatus.REMOVED: "🗑️",
}


def render_classification(result: ClassificationResult) -> str:
    """Format a classification result for moderator display.

    No-secret-leak: never includes raw message text. Shows class
    + flagged_phrases + severity score.
    """

    emoji = _CLASSIFICATION_EMOJI[result.classification]
    phrases = ", ".join(result.flagged_phrases) if result.flagged_phrases else "—"
    return (
        f"{emoji} {result.classification.value} "
        f"(severity {result.severity_score:.2f})\n"
        f"  flagged: {phrases}"
    )


def render_review(review: MessageReview) -> str:
    """Format a review row for moderator display."""

    cls_emoji = _CLASSIFICATION_EMOJI[review.classification]
    st_emoji = _STATUS_EMOJI[review.status]
    moderator_str = f" by {review.moderator}" if review.moderator else ""
    return (
        f"{st_emoji}{cls_emoji} msg {review.message_id} — "
        f"{review.classification.value} → {review.status.value}{moderator_str}"
    )


__all__ = [
    "DEFAULT_POLICY",
    "ClassificationResult",
    "ContentClassification",
    "MessageReview",
    "ModerationPolicy",
    "ReviewStatus",
    "ReviewTransitionError",
    "auto_decide",
    "classify",
    "initial_review",
    "is_visible_to_channel",
    "moderator_approve",
    "moderator_remove",
    "render_classification",
    "render_review",
]
