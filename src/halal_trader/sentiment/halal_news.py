"""Halal-aware news aggregator — Round-5 Wave 11.G.

Filters incoming news items by halal-relevance + tags ESG/halal
alignment so the cycle's prompt context surfaces stories that matter
to the screener (sector reclassifications, debt-issuance announcements,
revenue-mix shifts) and skips noise (promotional content, generic
market chatter).

Pinned semantics:

- **Closed-set HalalRelevance ladder** (NOT_RELEVANT / GENERAL /
  HALAL_RELEVANT / SCHEDULED_RECOMPUTE / EXCEPTION_REVIEW).
- **Closed-set Topic ladder** — tags news by topic for downstream
  routing.
- **Confidence score in [0, 1]** — the filter is ranking-based not
  hard-classifying.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class HalalRelevance(str, Enum):
    """Closed-set halal-relevance ladder."""

    NOT_RELEVANT = "not_relevant"
    GENERAL = "general"
    HALAL_RELEVANT = "halal_relevant"
    SCHEDULED_RECOMPUTE = "scheduled_recompute"
    EXCEPTION_REVIEW = "exception_review"


class Topic(str, Enum):
    """Closed-set news topics."""

    DEBT_ISSUANCE = "debt_issuance"
    DEBT_RETIREMENT = "debt_retirement"
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    EARNINGS = "earnings"
    DIVIDEND = "dividend"
    SECTOR_RECLASS = "sector_reclass"
    EXEC_CHANGE = "exec_change"
    REGULATORY = "regulatory"
    PRODUCT_LAUNCH = "product_launch"
    SCHOLAR_OPINION = "scholar_opinion"
    GENERIC = "generic"


# Lexicon mapping topic → keyword set
_TOPIC_KEYWORDS: dict[Topic, frozenset[str]] = {
    Topic.DEBT_ISSUANCE: frozenset(
        {"debt issuance", "bond offering", "issued bonds", "raised debt"}
    ),
    Topic.DEBT_RETIREMENT: frozenset(
        {"debt retirement", "retired debt", "paid off debt", "redeemed bonds"}
    ),
    Topic.ACQUISITION: frozenset({"acquisition", "acquired", "buyout", "merger"}),
    Topic.DIVESTITURE: frozenset({"divestiture", "divested", "sold subsidiary", "spinoff"}),
    Topic.EARNINGS: frozenset({"earnings", "quarterly results", "revenue", "eps", "guidance"}),
    Topic.DIVIDEND: frozenset({"dividend", "payout", "ex-dividend"}),
    Topic.SECTOR_RECLASS: frozenset({"reclassified", "sector change", "industry reclassification"}),
    Topic.EXEC_CHANGE: frozenset({"ceo resign", "ceo step down", "appointed ceo", "new cfo"}),
    Topic.REGULATORY: frozenset({"sec investigation", "fine", "settlement", "lawsuit"}),
    Topic.PRODUCT_LAUNCH: frozenset({"launched", "unveiled", "new product"}),
    Topic.SCHOLAR_OPINION: frozenset({"fatwa", "scholar verdict", "shariah ruling", "aaoifi"}),
}

_HARAM_KEYWORDS: frozenset[str] = frozenset(
    {
        "alcohol",
        "wine",
        "beer",
        "spirits",
        "casino",
        "gambling",
        "tobacco",
        "cigarette",
        "pork",
        "lottery",
        "interest rate",
        "ribah",
    }
)

_HALAL_KEYWORDS: frozenset[str] = frozenset(
    {
        "halal",
        "shariah",
        "islamic finance",
        "sukuk",
        "zakat",
        "purification",
        "aaoifi",
    }
)


@dataclass(frozen=True)
class NewsItem:
    """One news item received from a feed."""

    item_id: str
    headline: str
    body: str
    source: str
    published_at: datetime
    tickers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.item_id or not self.item_id.strip():
            raise ValueError("item_id must be non-empty")
        if not self.headline or not self.headline.strip():
            raise ValueError("headline must be non-empty")
        if not self.source or not self.source.strip():
            raise ValueError("source must be non-empty")
        if self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware")


@dataclass(frozen=True)
class TaggedNews:
    """A news item after halal-relevance scoring."""

    item: NewsItem
    relevance: HalalRelevance
    topics: frozenset[Topic]
    confidence: float
    haram_keywords_hit: frozenset[str]
    halal_keywords_hit: frozenset[str]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


def _detect_topics(haystack: str) -> frozenset[Topic]:
    text = haystack.lower()
    out: set[Topic] = set()
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            out.add(topic)
    if not out:
        out.add(Topic.GENERIC)
    return frozenset(out)


def _hits(text: str, keywords: frozenset[str]) -> frozenset[str]:
    lowered = text.lower()
    return frozenset(kw for kw in keywords if kw in lowered)


def tag_news(item: NewsItem) -> TaggedNews:
    """Tag a news item with halal-relevance + topics + keyword hits."""
    combined = f"{item.headline} {item.body}"
    topics = _detect_topics(combined)
    haram = _hits(combined, _HARAM_KEYWORDS)
    halal = _hits(combined, _HALAL_KEYWORDS)

    confidence = 0.0
    relevance: HalalRelevance

    # Topic-driven heuristics
    high_relevance_topics = {
        Topic.DEBT_ISSUANCE,
        Topic.DEBT_RETIREMENT,
        Topic.ACQUISITION,
        Topic.DIVESTITURE,
        Topic.SECTOR_RECLASS,
        Topic.SCHOLAR_OPINION,
    }

    if halal:
        relevance = HalalRelevance.HALAL_RELEVANT
        confidence = min(1.0, 0.4 + 0.2 * len(halal))
    elif haram:
        relevance = HalalRelevance.EXCEPTION_REVIEW
        confidence = min(1.0, 0.5 + 0.15 * len(haram))
    elif topics & high_relevance_topics:
        relevance = HalalRelevance.SCHEDULED_RECOMPUTE
        confidence = min(1.0, 0.6 + 0.10 * len(topics & high_relevance_topics))
    elif topics - {Topic.GENERIC}:
        relevance = HalalRelevance.GENERAL
        confidence = 0.40
    else:
        relevance = HalalRelevance.NOT_RELEVANT
        confidence = 0.10

    return TaggedNews(
        item=item,
        relevance=relevance,
        topics=topics,
        confidence=confidence,
        haram_keywords_hit=haram,
        halal_keywords_hit=halal,
    )


def filter_relevant(
    items: Iterable[NewsItem],
    *,
    min_relevance: HalalRelevance = HalalRelevance.HALAL_RELEVANT,
) -> tuple[TaggedNews, ...]:
    """Filter news to those at or above ``min_relevance`` priority."""
    order = {
        HalalRelevance.NOT_RELEVANT: 0,
        HalalRelevance.GENERAL: 1,
        HalalRelevance.SCHEDULED_RECOMPUTE: 2,
        HalalRelevance.HALAL_RELEVANT: 3,
        HalalRelevance.EXCEPTION_REVIEW: 4,
    }
    threshold = order[min_relevance]
    tagged = [tag_news(i) for i in items]
    return tuple(t for t in tagged if order[t.relevance] >= threshold)


def render_tagged(tagged: TaggedNews) -> str:
    emoji = {
        HalalRelevance.NOT_RELEVANT: "⚪",
        HalalRelevance.GENERAL: "🔵",
        HalalRelevance.SCHEDULED_RECOMPUTE: "🟡",
        HalalRelevance.HALAL_RELEVANT: "🟢",
        HalalRelevance.EXCEPTION_REVIEW: "🔴",
    }[tagged.relevance]
    topics = ",".join(sorted(t.value for t in tagged.topics))[:80]
    return (
        f"{emoji} {tagged.item.item_id}: {tagged.relevance.value} "
        f"conf={tagged.confidence:.2f} topics={topics}"
    )
