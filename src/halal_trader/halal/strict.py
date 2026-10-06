"""The strict in-house screen's verdicts, as the day-trader's halal gate reads them.

``halal_screen_results`` holds every run of the in-house screen
(compliance/runner.py): the stricter of AAOIFI and S&P Shariah on every
axis, an index board's exclusion a veto (operator decision 2026-10-02).
The core portfolio has read it since it started; the day-trader and the
reactor used a curated 20-name list instead, seven of which the strict
screen fails. This module is how they read the screen too.

Rules, all failing CLOSED:

* only the **newest** screen counts (the newest ``as_of`` in the table), and
  only while it is **fresh** (``MAX_SCREEN_AGE``, as for the core): a stale
  or missing screen makes nothing halal;
* a symbol absent from the newest screen is not halal;
* a symbol is halal only if **every** row the newest screen has for it says
  so. The table may carry more than one row per symbol and day (a
  ``method`` column is joining its key): whether those rows are a re-run or
  parallel methods, a single non-halal row is enough to refuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

# Same freshness as the core's order boundary (portfolio/core_executor.py):
# the screen runs weekly, so 10 days allows one missed run.
MAX_SCREEN_AGE = timedelta(days=10)


@dataclass(frozen=True, slots=True)
class Verdict:
    halal: bool
    reason: str
    screen_as_of: date | None


async def newest_screen(engine: AsyncEngine) -> date | None:
    async with engine.connect() as conn:
        value = (await conn.execute(text("SELECT max(as_of) FROM halal_screen_results"))).scalar()
    return value if isinstance(value, date) else None


async def verdict(engine: AsyncEngine, symbol: str, *, today: date) -> Verdict:
    """Does the newest fresh strict screen hold ``symbol`` halal?"""
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    """
                    WITH newest AS (SELECT max(as_of) AS as_of FROM halal_screen_results)
                    SELECT newest.as_of,
                           count(r.symbol) AS n,
                           coalesce(bool_and(r.verdict = 'halal'), false) AS halal
                    FROM newest
                    LEFT JOIN halal_screen_results r
                           ON r.as_of = newest.as_of AND r.symbol = :s
                    GROUP BY newest.as_of
                    """
                ),
                {"s": symbol},
            )
        ).one()
    as_of: date | None = row.as_of
    if as_of is None:
        return Verdict(False, "no strict screen on record", None)
    if today - as_of > MAX_SCREEN_AGE:
        return Verdict(False, f"the strict screen of {as_of} is stale", as_of)
    if not row.n:
        return Verdict(False, f"{symbol} is not in the strict screen of {as_of}", as_of)
    if not row.halal:
        return Verdict(False, f"the strict screen of {as_of} does not pass {symbol}", as_of)
    return Verdict(True, f"halal in the strict screen of {as_of}", as_of)


async def halal_universe(
    engine: AsyncEngine, *, today: date, limit: int
) -> tuple[date | None, list[str]]:
    """The ``limit`` largest names (by market cap) the newest fresh screen passes.

    Market cap is the liquidity proxy: the largest names trade the most.
    Returns ``(screen date, symbols)``; no symbols when the screen is stale
    or missing.
    """
    as_of = await newest_screen(engine)
    if as_of is None or today - as_of > MAX_SCREEN_AGE:
        return as_of, []
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                """
                SELECT symbol, max((metrics->>'market_cap')::float) AS cap
                FROM halal_screen_results
                WHERE as_of = :a
                GROUP BY symbol
                HAVING bool_and(verdict = 'halal')
                ORDER BY cap DESC NULLS LAST, symbol
                LIMIT :n
                """
            ),
            {"a": as_of, "n": limit},
        )
        return as_of, [r.symbol for r in rows]
