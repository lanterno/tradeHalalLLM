"""Point-in-time eligibility for backtests: liquid then, and halal then.

At each month start the eligible set is the liquidity universe of that
date (data/universe.py) intersected with the newest halal screen on or
before it (compliance/history.py). Neither knows anything after the date.

What remains biased is measured, not assumed away: a company that has
since delisted cannot be mapped to its SEC filings, so it is never
eligible. ``unmapped`` reports, per year, the share of the universe's
companies in that position (ETFs and funds, which are never companies,
are left out of both sides; compliance/delisted.py recovers what it can).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.runner import UNMAPPED
from halal_trader.data.universe import month_starts, universe_at
from halal_trader.halal.strict import all_screens


@dataclass(frozen=True, slots=True)
class Firm:
    """What the screen knew about an eligible name on the day it screened it."""

    cik: int | None
    cap: float | None  # screen-date price x shares outstanding (spot, not the 36-month average)
    screened: date
    sic: str | None = None  # its SEC industry as that screen recorded it


@dataclass(frozen=True, slots=True)
class PitSchedule:
    eligible_from: dict[date, set[str]]
    universe: set[str]  # every name ever in the universe, for loading prices
    unmapped: dict[int, float]  # year -> mean share of the universe's companies with no CIK
    firms_from: dict[date, dict[str, Firm]] = field(default_factory=dict)  # eligible names only


@dataclass(slots=True)
class _Screen:
    halal: dict[str, Firm] = field(default_factory=dict)
    unmapped: set[str] = field(default_factory=set)


async def _screens(engine: AsyncEngine) -> dict[date, _Screen]:
    out: dict[date, _Screen] = {}
    for as_of, rows in (await all_screens(engine)).items():
        screen = out[as_of] = _Screen()
        for r in rows:
            if r.verdict == "halal":
                cap = r.price * r.shares if r.price is not None and r.shares is not None else None
                screen.halal[r.symbol] = Firm(r.cik, cap, as_of, r.sic_description)
            if r.sic_description == UNMAPPED:
                screen.unmapped.add(r.symbol)
    return out


async def pit_schedule(engine: AsyncEngine, *, start: date, end: date, top_n: int) -> PitSchedule:
    from halal_trader.compliance.delisted import fund_symbols

    screens = await _screens(engine)
    funds = await fund_symbols(engine)
    dates = sorted(screens)
    eligible_from: dict[date, set[str]] = {}
    firms_from: dict[date, dict[str, Firm]] = {}
    universe: set[str] = set()
    shares: dict[int, list[float]] = {}
    for month in month_starts(start, end):
        members = set(await universe_at(engine, month, top_n=top_n))
        universe |= members
        known = [d for d in dates if d <= month]
        if not known or not members:
            eligible_from[month] = set()
            continue
        screen = screens[known[-1]]
        eligible_from[month] = members & screen.halal.keys()
        firms_from[month] = {s: screen.halal[s] for s in eligible_from[month]}
        companies = members - funds
        if companies:
            share = len(companies & screen.unmapped) / len(companies)
            shares.setdefault(month.year, []).append(share)
    return PitSchedule(
        eligible_from=eligible_from,
        universe=universe,
        unmapped={y: sum(v) / len(v) for y, v in shares.items()},
        firms_from=firms_from,
    )
