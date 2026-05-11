"""Daily halal market commentary feed — Round-5 Wave 17.D.

Append-only commentary feed. Operator (or LLM) drops daily updates
on movers, sectors, sukuk yield curve, gold price, etc. Community
members can author, but every post passes a halal moderation pass
before publishing.

Pinned semantics:

- **Closed-set Topic ladder** — MOVERS / SECTORS / SUKUK / GOLD /
  MACRO / OPS / EDUCATION. Open-tag commentary is forbidden.
- **Closed-set Source** — OPERATOR / LLM / COMMUNITY. Each routes
  through different moderation logic.
- **Append-only**. No edit / delete; superseding posts use
  `supersedes_id`.
- **Halal moderation gate**. `moderate_commentary` runs the lexicon-
  based check (delegated to `community/chat_moderation.py` once that
  ships; for now this layer accepts a plug-in predicate so 17.F can
  wire in cleanly).
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — author IDs masked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum


class Topic(str, Enum):
    """Closed-set topic ladder."""

    MOVERS = "movers"
    SECTORS = "sectors"
    SUKUK = "sukuk"
    GOLD = "gold"
    MACRO = "macro"
    OPS = "ops"
    EDUCATION = "education"


class Source(str, Enum):
    """Closed-set source ladder."""

    OPERATOR = "operator"
    LLM = "llm"
    COMMUNITY = "community"


class Severity(str, Enum):
    """Closed-set commentary severity ladder."""

    INFO = "info"
    NOTABLE = "notable"
    URGENT = "urgent"


@dataclass(frozen=True)
class CommentaryPost:
    """One post in the feed."""

    post_id: str
    feed_date: date
    topic: Topic
    source: Source
    author_id: str
    headline: str
    body: str
    severity: Severity = Severity.INFO
    tickers: tuple[str, ...] = ()
    posted_at: datetime | None = None
    supersedes_id: str = ""

    def __post_init__(self) -> None:
        if not self.post_id or not self.post_id.strip():
            raise ValueError("post_id must be non-empty")
        if not self.author_id or not self.author_id.strip():
            raise ValueError("author_id must be non-empty")
        if not self.headline.strip():
            raise ValueError("headline must be non-empty")
        if len(self.headline) > 200:
            raise ValueError("headline must be ≤ 200 chars")
        if not self.body.strip():
            raise ValueError("body must be non-empty")
        if len(self.body) > 5000:
            raise ValueError("body must be ≤ 5000 chars")
        for t in self.tickers:
            if not t or not t.strip():
                raise ValueError("ticker entries must be non-empty")
            if len(t) > 16:
                raise ValueError("ticker entries must be ≤ 16 chars")
        if self.posted_at is None:
            # Default to start-of-day in UTC for the feed_date so all
            # entries get a stable timestamp without forcing the caller.
            object.__setattr__(
                self,
                "posted_at",
                datetime.combine(self.feed_date, datetime.min.time()),
            )


class ModerationOutcome(str, Enum):
    """Closed-set moderation outcome."""

    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


# Default lexicon of "loud trading talk" the platform doesn't host.
# Each maps to (outcome, reason). The lexicon is deliberately *narrow*
# — wave 17.F ships a richer classifier that can replace this.
_DEFAULT_HARAM_LEXICON: dict[str, tuple[ModerationOutcome, str]] = {
    "lottery": (ModerationOutcome.BLOCK, "gambling-style language"),
    "casino": (ModerationOutcome.BLOCK, "gambling-style language"),
    "all-in yolo": (ModerationOutcome.BLOCK, "gambling-style language"),
    "leveraged margin": (ModerationOutcome.WARN, "leveraged-margin reference"),
    "interest income": (ModerationOutcome.WARN, "potential riba reference"),
    "guaranteed return": (ModerationOutcome.BLOCK, "guaranteed-return claim — riba"),
    "moonshot": (ModerationOutcome.WARN, "speculative-style language"),
}


@dataclass(frozen=True)
class ModerationResult:
    """Output of `moderate_commentary`."""

    outcome: ModerationOutcome
    reasons: tuple[str, ...]


def moderate_commentary(
    post: CommentaryPost,
    *,
    lexicon: dict[str, tuple[ModerationOutcome, str]] | None = None,
) -> ModerationResult:
    """Run the lexicon-based halal moderation pass.

    Pinned: scans `headline` + `body` (case-insensitive) for haram-
    coded phrases; collects every matching reason; the worst outcome
    wins (BLOCK > WARN > PASS).
    """
    table = lexicon if lexicon is not None else _DEFAULT_HARAM_LEXICON
    text = f"{post.headline} {post.body}".lower()
    reasons: list[str] = []
    worst = ModerationOutcome.PASS
    for phrase, (outcome, reason) in table.items():
        if phrase in text:
            reasons.append(f"matched '{phrase}' — {reason}")
            if outcome is ModerationOutcome.BLOCK:
                worst = ModerationOutcome.BLOCK
            elif outcome is ModerationOutcome.WARN and worst is not ModerationOutcome.BLOCK:
                worst = ModerationOutcome.WARN
    return ModerationResult(outcome=worst, reasons=tuple(reasons))


@dataclass(frozen=True)
class CommentaryFeed:
    """An append-only feed of moderated posts."""

    posts: tuple[CommentaryPost, ...] = ()

    def __post_init__(self) -> None:
        ids: set[str] = set()
        prev_at: datetime | None = None
        for p in self.posts:
            if p.post_id in ids:
                raise ValueError(f"duplicate post_id {p.post_id}")
            ids.add(p.post_id)
            if prev_at is not None and p.posted_at is not None:
                if p.posted_at < prev_at:
                    raise ValueError("posts must be ordered by posted_at")
            prev_at = p.posted_at


def append_post(
    feed: CommentaryFeed,
    post: CommentaryPost,
    *,
    moderator: Callable[[CommentaryPost], ModerationResult] | None = None,
) -> tuple[CommentaryFeed, ModerationResult]:
    """Append a post to the feed, gated by moderation.

    BLOCKed posts raise; WARN posts append with a flag preserved in the
    moderation result (caller can decide to warn user).
    """
    if any(p.post_id == post.post_id for p in feed.posts):
        raise ValueError(f"duplicate post_id {post.post_id}")
    if post.supersedes_id:
        prior = next((p for p in feed.posts if p.post_id == post.supersedes_id), None)
        if prior is None:
            raise ValueError(f"supersedes_id {post.supersedes_id} does not exist")
        if any(p.supersedes_id == post.supersedes_id for p in feed.posts):
            raise ValueError(f"post {post.supersedes_id} already superseded")
    moderate = moderator if moderator is not None else moderate_commentary
    result = moderate(post)
    if result.outcome is ModerationOutcome.BLOCK:
        raise ValueError(f"post BLOCKED by moderation: {'; '.join(result.reasons)}")
    new_posts = (*feed.posts, post)
    return CommentaryFeed(posts=new_posts), result


def filter_feed(
    feed: CommentaryFeed,
    *,
    topic: Topic | None = None,
    source: Source | None = None,
    severity: Severity | None = None,
    ticker: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> tuple[CommentaryPost, ...]:
    """Filter posts by any combination of fields."""
    out: list[CommentaryPost] = []
    for p in feed.posts:
        if topic is not None and p.topic is not topic:
            continue
        if source is not None and p.source is not source:
            continue
        if severity is not None and p.severity is not severity:
            continue
        if ticker is not None and ticker not in p.tickers:
            continue
        if date_from is not None and p.feed_date < date_from:
            continue
        if date_to is not None and p.feed_date > date_to:
            continue
        out.append(p)
    return tuple(out)


def supersede(
    feed: CommentaryFeed,
    *,
    new_post: CommentaryPost,
    moderator: Callable[[CommentaryPost], ModerationResult] | None = None,
) -> tuple[CommentaryFeed, ModerationResult]:
    """Append a post that supersedes a prior one."""
    if not new_post.supersedes_id:
        raise ValueError("supersede requires non-empty supersedes_id")
    return append_post(feed, new_post, moderator=moderator)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_SEVERITY_EMOJI: dict[Severity, str] = {
    Severity.INFO: "📝",
    Severity.NOTABLE: "📌",
    Severity.URGENT: "🚨",
}


def render_post(post: CommentaryPost, *, body_chars: int = 200) -> str:
    body = post.body if len(post.body) <= body_chars else post.body[:body_chars] + "…"
    tickers = f" [{', '.join(post.tickers)}]" if post.tickers else ""
    super_str = f" supersedes={post.supersedes_id}" if post.supersedes_id else ""
    head = (
        f"{_SEVERITY_EMOJI[post.severity]} [{post.post_id}] "
        f"{post.feed_date.isoformat()} "
        f"[{post.topic.value}/{post.source.value}] "
        f"{post.headline}{tickers}{super_str}"
    )
    return f"{head}\n  Author: {_mask(post.author_id)}\n  {body}"


def render_feed(feed: CommentaryFeed, *, top_n: int = 20) -> str:
    rows = feed.posts[-top_n:] if top_n > 0 else feed.posts
    if not rows:
        return "📰 Feed is empty."
    lines = [f"📰 Feed: {len(feed.posts)} post(s) (showing latest {len(rows)}):"]
    for p in rows:
        lines.append(render_post(p, body_chars=120))
    return "\n".join(lines)
