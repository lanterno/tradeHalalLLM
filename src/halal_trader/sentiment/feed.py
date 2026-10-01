"""Render recent ``NewsEvent`` items into the strategy prompt's news block.

The stock cycle's ``FetchStockNewsStage`` pulls headlines per cycle from
the configured collector and passes them through
:func:`format_news_for_prompt`, which keeps the section compact: at most
``limit`` items, optionally restricted to the symbols being traded.
"""

from __future__ import annotations

from typing import Sequence

from halal_trader.sentiment.events import NewsEvent

_SENTIMENT_GLYPH = {"positive": "▲", "negative": "▼", "neutral": "·"}


def format_news_for_prompt(
    events: Sequence[NewsEvent], *, limit: int = 6, pair_filter: Sequence[str] | None = None
) -> str:
    """Render up to ``limit`` events as a compact bullet list for the LLM.

    ``pair_filter`` (optional) restricts to events whose ``affected_pairs``
    overlaps the filter — useful when the universe is small and we want
    to avoid burning tokens on irrelevant headlines.

    Empty result when there are no matching events; the prompt template
    should omit the section rather than show "Recent News: —".
    """
    if not events:
        return ""

    if pair_filter:
        pf = {p.upper() for p in pair_filter}
        events = [
            e
            for e in events
            if not e.affected_pairs or pf.intersection(p.upper() for p in e.affected_pairs)
        ]
        if not events:
            return ""

    # Newest ``limit``, whatever order the caller passed: this used to take the
    # LAST N assuming oldest-first input, but the stock collector sorts
    # newest-first, so every cycle showed the model the 6 STALEST headlines.
    # Rendered oldest-to-newest. ISO-8601 strings sort chronologically.
    chosen = sorted(events, key=lambda e: e.published_at or "")[-limit:]
    lines: list[str] = []
    for ev in chosen:
        glyph = _SENTIMENT_GLYPH.get(ev.sentiment.lower(), "·")
        importance = ev.importance.upper() if ev.importance != "normal" else ""
        pairs = f" [{','.join(p.upper() for p in ev.affected_pairs)}]" if ev.affected_pairs else ""
        head = f"{glyph} {ev.title}".strip()
        meta = " ".join(filter(None, [importance, pairs.strip(), f"({ev.source})"]))
        lines.append(f"  - {head} — {meta}".rstrip())
    return "\n".join(lines)
