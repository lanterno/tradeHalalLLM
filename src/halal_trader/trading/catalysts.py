"""Stock catalyst feed — news, earnings, insider transactions for the LLM prompt.

The stock cycle today only sees price + indicators. Real day-trading
edge on stocks lives in *catalysts*:

* Breaking headlines (8-K filings, analyst actions, FDA decisions)
* Pending earnings (volatility skew + post-earnings drift)
* Insider Form 4 transactions (cluster buys / sells often precede moves)

This module is the **abstraction**: the ``Catalyst`` record, the
source protocol, the ``StockCatalystFeed`` that merges sources, and the
prompt formatter. The sources the bot wires live beside it:
``fred_catalysts`` (macro release calendar), ``edgar_catalysts`` (8-K
filings) and ``fed_speak_adapter`` (Fed-speak drift).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Catalyst:
    """A single time-stamped catalyst event for one symbol."""

    symbol: str
    kind: str  # "news" | "earnings" | "insider_buy" | "insider_sell" | "analyst"
    title: str
    timestamp: datetime
    sentiment: str = "neutral"  # "positive" | "negative" | "neutral"
    source: str = ""
    url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class CatalystSource(Protocol):
    """Anything that can produce a list of recent catalysts on demand."""

    async def fetch(self, symbols: Sequence[str]) -> list[Catalyst]: ...


class StockCatalystFeed:
    """Aggregates one or more :class:`CatalystSource`s into a single feed.

    Each source can fail independently — a flaky news API shouldn't take
    out the earnings calendar. Failures are logged, not raised, so the
    cycle can continue on whatever signal *is* available.
    """

    def __init__(self, sources: Sequence[CatalystSource] | None = None) -> None:
        self._sources: list[CatalystSource] = list(sources or [])

    async def fetch_all(self, symbols: Sequence[str]) -> list[Catalyst]:
        """Pull from every source, swallow per-source errors, return combined list."""
        if not self._sources or not symbols:
            return []
        out: list[Catalyst] = []
        for source in self._sources:
            try:
                out.extend(await source.fetch(symbols))
            except Exception as e:  # noqa: BLE001 — never let a source crash the cycle
                logger.debug("Catalyst source %s failed: %s", type(source).__name__, e)
        out.sort(key=lambda c: c.timestamp, reverse=True)
        return out


_KIND_GLYPH = {
    "news": "📰",
    "earnings": "📊",
    "insider_buy": "▲",
    "insider_sell": "▼",
    "analyst": "🎯",
}
_SENTIMENT_GLYPH = {"positive": "+", "negative": "-", "neutral": "·"}


def format_catalysts_for_prompt(
    catalysts: Sequence[Catalyst],
    *,
    symbols: Sequence[str] | None = None,
    limit: int = 8,
    max_age_hours: int = 24,
) -> str:
    """Render the most recent ``limit`` catalysts as a compact bullet list.

    ``symbols`` (optional) restricts to events whose ``symbol`` is in the
    set — useful to avoid burning tokens on the full halal universe when
    we only care about today's actionable names. ``max_age_hours`` drops
    stale events so a long-running bot doesn't anchor on yesterday's
    news after a quiet morning.
    """
    if not catalysts:
        return ""

    cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
    sym_set = {s.upper() for s in symbols} if symbols else None

    fresh: list[Catalyst] = []
    for c in catalysts:
        ts = c.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if ts < cutoff:
            continue
        if sym_set and c.symbol.upper() not in sym_set:
            continue
        fresh.append(c)

    if not fresh:
        return ""

    chosen = fresh[:limit]
    lines: list[str] = []
    for c in chosen:
        glyph = _KIND_GLYPH.get(c.kind, "·")
        sentiment = _SENTIMENT_GLYPH.get(c.sentiment, "·")
        meta_parts = [c.kind.upper(), sentiment]
        if c.source:
            meta_parts.append(f"({c.source})")
        meta = " ".join(meta_parts)
        lines.append(f"  - {glyph} [{c.symbol}] {c.title} — {meta}")
    return "\n".join(lines)
