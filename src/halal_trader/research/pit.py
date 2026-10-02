"""Point-in-time eligibility for backtests: liquid then, and halal then.

At each month start the eligible set is the liquidity universe of that
date (data/universe.py) intersected with the newest halal screen on or
before it (compliance/history.py). Neither knows anything after the date.

What remains biased is measured, not assumed away: a company that has
since delisted cannot be mapped to its SEC filings, so it is never
eligible. ``unmapped`` reports, per year, the share of the universe that
was in that position.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data.universe import month_starts, universe_at

UNMAPPED = "not an SEC registrant (or ticker not mapped)"


@dataclass(frozen=True, slots=True)
class PitSchedule:
    eligible_from: dict[date, set[str]]
    universe: set[str]  # every name ever in the universe, for loading prices
    unmapped: dict[int, float]  # year -> mean share of the universe with no SEC mapping


async def _screens(engine: AsyncEngine) -> dict[date, tuple[set[str], set[str]]]:
    """as_of -> (halal names, unmapped names)."""
    out: dict[date, tuple[set[str], set[str]]] = {}
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT as_of, symbol, verdict, sic_description FROM halal_screen_results")
        )
        for r in rows:
            halal, unmapped = out.setdefault(r.as_of, (set(), set()))
            if r.verdict == "halal":
                halal.add(r.symbol)
            if r.sic_description == UNMAPPED:
                unmapped.add(r.symbol)
    return out


async def pit_schedule(engine: AsyncEngine, *, start: date, end: date, top_n: int) -> PitSchedule:
    screens = await _screens(engine)
    dates = sorted(screens)
    eligible_from: dict[date, set[str]] = {}
    universe: set[str] = set()
    shares: dict[int, list[float]] = {}
    for month in month_starts(start, end):
        members = set(await universe_at(engine, month, top_n=top_n))
        universe |= members
        known = [d for d in dates if d <= month]
        if not known or not members:
            eligible_from[month] = set()
            continue
        halal, unmapped = screens[known[-1]]
        eligible_from[month] = members & halal
        shares.setdefault(month.year, []).append(len(members & unmapped) / len(members))
    return PitSchedule(
        eligible_from=eligible_from,
        universe=universe,
        unmapped={y: sum(v) / len(v) for y, v in shares.items()},
    )
