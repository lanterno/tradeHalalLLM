"""The ``NewsEvent`` record the stocks news collectors emit.

:mod:`sentiment.stocks_news` (Finnhub, with Yahoo as the fallback)
returns these, and :func:`sentiment.feed.format_news_for_prompt` renders
them into the strategy prompt's news block.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NewsEvent:
    """One news headline, with its polarity and the symbols it concerns."""

    title: str
    source: str
    url: str
    published_at: str
    sentiment: str
    symbols: list[str] = field(default_factory=list)
    importance: str = "normal"  # "normal", "hot", "breaking"
