"""How complete and how trustworthy the event store is (Phase A exit report).

Three questions, each answered from the stored rows alone:

* **Coverage:** events per year, source and kind, and how many of each
  year's screened companies have any event of that kind.
* **Timing:** an earnings release reaches us twice, as Benzinga's
  "EPS ... Estimate" headline and as SEC's 8-K item 2.02. The gap between
  the two for the same company is a check on both clocks; a large or
  negative gap flags a timestamp problem rather than news.
* **Duplication:** the same headline text for the same symbol on the same
  day, which inflates any count-based statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


@dataclass(frozen=True, slots=True)
class Coverage:
    year: int
    source: str
    kind: str
    events: int
    companies: int  # that year's universe members with at least one event
    universe: int  # distinct symbols the screen covered that year


@dataclass(frozen=True, slots=True)
class Timing:
    pairs: int  # releases seen both as a Benzinga result headline and an 8-K 2.02
    median_minutes: float | None  # headline minus 8-K acceptance
    within_hour: float | None  # share of pairs within +-60 minutes
    headline_first: float | None  # share where the headline came first


async def coverage(engine: AsyncEngine) -> list[Coverage]:
    async with engine.connect() as conn:
        universe = {
            int(r.y): int(r.n)
            for r in await conn.execute(
                text(
                    "SELECT extract(year FROM as_of) AS y, count(DISTINCT symbol) AS n "
                    "FROM halal_screen_results GROUP BY 1"
                )
            )
        }
        # Companies counted only if the screen covered them that year, so the
        # share reads "of that year's universe, how many have this event kind".
        rows = await conn.execute(
            text(
                "WITH members AS (SELECT DISTINCT extract(year FROM as_of) AS y, symbol "
                "FROM halal_screen_results) "
                "SELECT extract(year FROM e.published_at) AS y, e.source, e.kind, "
                "count(*) AS n, count(DISTINCT m.symbol) AS c FROM events e "
                "LEFT JOIN members m ON m.symbol = e.symbol "
                "AND m.y = extract(year FROM e.published_at) "
                "GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"
            )
        )
        return [
            Coverage(int(r.y), r.source, r.kind, int(r.n), int(r.c), universe.get(int(r.y), 0))
            for r in rows
        ]


async def timing(engine: AsyncEngine, extractor: str | None = None) -> Timing:
    """Earnings releases seen both as a headline and as an 8-K item 2.02.

    ``extractor`` defaults to the current one (earnings_parse.EXTRACTOR): a
    literal here went stale at v3 and matched no fact at all.
    """
    from halal_trader.events.earnings_parse import EXTRACTOR

    extractor = extractor or EXTRACTOR
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    """
                    WITH pairs AS (
                        SELECT DISTINCT ON (k.id)
                            extract(epoch FROM (n.published_at - k.published_at)) / 60 AS gap
                        FROM events k
                        JOIN events n ON n.symbol = k.symbol AND n.kind = 'news'
                          AND n.published_at BETWEEN k.published_at - interval '1 day'
                                                 AND k.published_at + interval '1 day'
                        JOIN event_facts f ON f.event_id = n.id AND f.extractor = :x
                          AND f.kind = 'result'
                        WHERE k.kind = '8-k' AND k.payload->'items' ? '2.02'
                        ORDER BY k.id, abs(extract(epoch FROM (n.published_at - k.published_at)))
                    )
                    SELECT count(*) AS n,
                        percentile_cont(0.5) WITHIN GROUP (ORDER BY gap) AS med,
                        avg((abs(gap) <= 60)::int) AS hour,
                        avg((gap < 0)::int) AS first
                    FROM pairs
                    """
                ),
                {"x": extractor},
            )
        ).one()
    return Timing(
        pairs=int(row.n),
        median_minutes=float(row.med) if row.med is not None else None,
        within_hour=float(row.hour) if row.hour is not None else None,
        headline_first=float(row.first) if row.first is not None else None,
    )


async def duplicate_share(engine: AsyncEngine) -> float | None:
    """Share of news events repeating an earlier headline for the same symbol and day."""
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT count(*) AS n, count(DISTINCT (symbol, published_at::date, "
                    "payload->>'headline')) AS d FROM events WHERE kind = 'news'"
                )
            )
        ).one()
    return 1 - row.d / row.n if row.n else None
