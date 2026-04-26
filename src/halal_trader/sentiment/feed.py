"""Bounded recent-news feed for splicing into the LLM prompt.

The :class:`NewsEventReactor` already fires callbacks on every event,
but the strategy prompt only runs at cycle cadence (≥ 30s) and has no
direct subscription. This module gives the cycle a *snapshot view* of
the last ``capacity`` events so the LLM can reason about freshly-broken
news without us re-polling CryptoPanic per cycle.

Why a separate module:

* The reactor's job is "react in real time" (trigger emergency
  mini-cycles, send Telegram alerts). The feed's job is "summarise for
  the next normal cycle." Both consume the same ``NewsEvent`` stream
  but have different retention rules.
* The buffer is intentionally small (default 10) and time-windowed so
  the prompt section stays compact and stale events drop out — we don't
  want the LLM still anchoring on a 6-hour-old headline.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Sequence

from halal_trader.sentiment.events import NewsEvent


@dataclass
class _Entry:
    event: NewsEvent
    received_at: float  # monotonic seconds


class RecentNewsFeed:
    """Append-only bounded buffer of recent ``NewsEvent`` items.

    Thread-safety: not required — the reactor runs as a single asyncio
    task that pushes here, and the cycle consumes via :meth:`snapshot`
    on its own task. asyncio is cooperative so concurrent push/snapshot
    can't interleave mid-deque-mutation.
    """

    def __init__(self, *, capacity: int = 10, max_age_seconds: int = 1800) -> None:
        self._buf: deque[_Entry] = deque(maxlen=capacity)
        self._max_age = max_age_seconds

    def push(self, event: NewsEvent) -> None:
        self._buf.append(_Entry(event=event, received_at=time.monotonic()))

    def snapshot(self) -> list[NewsEvent]:
        """Return events newer than ``max_age_seconds``, oldest first.

        Stale entries are pruned lazily on read so we don't need a
        background sweep — the bounded deque caps memory either way.
        """
        cutoff = time.monotonic() - self._max_age
        return [e.event for e in self._buf if e.received_at >= cutoff]

    def clear(self) -> None:
        self._buf.clear()


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

    chosen = list(events)[-limit:]
    lines: list[str] = []
    for ev in chosen:
        glyph = _SENTIMENT_GLYPH.get(ev.sentiment.lower(), "·")
        importance = ev.importance.upper() if ev.importance != "normal" else ""
        pairs = f" [{','.join(p.upper() for p in ev.affected_pairs)}]" if ev.affected_pairs else ""
        head = f"{glyph} {ev.title}".strip()
        meta = " ".join(filter(None, [importance, pairs.strip(), f"({ev.source})"]))
        lines.append(f"  - {head} — {meta}".rstrip())
    return "\n".join(lines)
