"""Buying after bad-news days: the news reversal, tested on ten years of the lexicon.
(pre-registered 2026-10-10)

In July-October 2026 every reading of tech headlines (GLM's, an expert's, the
lexicon's) had a strongly negative 5-day IC: the names with the worst news
days then beat SPY by about 2% over 5 days, net of cost. That was spotted in
the data it would be judged on, so it proves nothing. This tests the same
idea where it was not seen: the free keyword lexicon over 2016-2024.

Pre-registered (written before any result was seen):

* **signal:** the lexicon polarity (positive minus negative weight,
  sentiment/headline_polarity.py) of a company's own headlines (an article
  naming at most ``MAX_SYMBOLS`` companies), averaged per (symbol, New York
  day), timed at the day's last headline, as llm_eval reads the LLM;
* **trade:** long-only, the most negative decile, entered at its first
  tradable price (events/study.py: timing, costs, abnormal to SPY);
* **primary:** every covered company; secondary: Technology only;
* **rule:** trained on 2016-2021, the bottom decile's 5-day mean net abnormal
  return must be positive with t >= 2; then 2022-2024 must agree in sign with
  t >= 2; 2025-2026 (which holds the window where the idea was spotted) stays
  untouched.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, date, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.study import Observation
from halal_trader.market_hours import MARKET_TZ

MAX_SYMBOLS = 3
HORIZONS = (1, 5, 20)


async def news_days(
    engine: AsyncEngine, start: int, end: int, symbols: frozenset[str] | None = None
) -> list[Observation]:
    """One observation per (symbol, New York day) with company news in [start, end]."""
    from halal_trader.sentiment.headline_polarity import score_headline

    lo = datetime(start, 1, 1, tzinfo=UTC)
    hi = datetime(end + 1, 1, 1, tzinfo=UTC)
    days: dict[tuple[str, date], list[tuple[datetime, float]]] = defaultdict(list)
    async with engine.connect() as conn:
        rows = await conn.stream(
            text(
                "SELECT symbol, published_at, payload->>'headline' AS h FROM events "
                "WHERE kind = 'news' AND published_at >= :lo AND published_at < :hi "
                "AND symbol IS NOT NULL "
                "AND jsonb_array_length(coalesce(payload->'symbols', '[]'::jsonb)) BETWEEN 1 AND :m"
            ),
            {"lo": lo, "hi": hi, "m": MAX_SYMBOLS},
        )
        async for r in rows:
            if symbols is not None and r.symbol not in symbols:
                continue
            pos, neg = score_headline(r.h or "")
            days[(r.symbol, r.published_at.astimezone(MARKET_TZ).date())].append(
                (r.published_at, pos - neg)
            )
    return [
        Observation(symbol, max(t for t, _ in items), sum(v for _, v in items) / len(items))
        for (symbol, _), items in days.items()
    ]
