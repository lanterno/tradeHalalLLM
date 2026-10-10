"""Seeding the test database: minute bars, done units, daily bars and ledger rows."""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data import minutes
from halal_trader.data.minutes import BarArrays
from halal_trader.events.history import mark_units
from halal_trader.market_hours import is_trading_day
from tests.halabot.playbooks._synth import Market


async def seed_bars(engine: AsyncEngine, symbol: str, bars: BarArrays) -> None:
    rows = [
        {
            "s": symbol,
            "t": datetime.fromtimestamp(int(bars.ts[i]), UTC),
            "o": float(bars.o[i]),
            "h": float(bars.h[i]),
            "l": float(bars.l[i]),
            "c": float(bars.c[i]),
            "v": float(bars.v[i]),
            "vw": None if math.isnan(float(bars.vw[i])) else float(bars.vw[i]),
        }
        for i in range(len(bars))
    ]
    if not rows:
        return
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO minute_bars (symbol, ts, open, high, low, close, volume, vwap) "
                "VALUES (:s, :t, :o, :h, :l, :c, :v, :vw) ON CONFLICT DO NOTHING"
            ),
            rows,
        )


async def mark_done(engine: AsyncEngine, units: Iterable[tuple[str, date]]) -> None:
    await mark_units(engine, minutes.TASK, {minutes.unit(s, d): 0 for s, d in units})


async def seed_daily(
    engine: AsyncEngine,
    symbol: str,
    closes: dict[date, float],
    *,
    adjusted: dict[date, float] | None = None,
) -> None:
    """Raw daily bars (and 'all' bars when ``adjusted`` is given) at the given closes."""
    rows = [{"s": symbol, "d": d, "a": "raw", "p": c} for d, c in closes.items()] + [
        {"s": symbol, "d": d, "a": "all", "p": c} for d, c in (adjusted or {}).items()
    ]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES (:s, :d, :a, :p, :p, :p, :p, 1000, now())"
            ),
            rows,
        )


async def seed_calendar(engine: AsyncEngine, start: date, end: date) -> None:
    """SPY raw daily bars on every market_hours session in [start, end]."""
    days = {}
    d = start
    while d <= end:
        if is_trading_day(d):
            days[d] = 200.0
        d += timedelta(days=1)
    await seed_daily(engine, "SPY", days)


async def seed_market(engine: AsyncEngine, market: Market) -> None:
    """Every path's bars, SPY's, and their units marked done."""
    units: set[tuple[str, date]] = set()
    for p in market.paths.values():
        for s, bars in zip(p.sessions, p.bars):
            if (p.symbol, s.day) not in units:
                await seed_bars(engine, p.symbol, bars)
                units.add((p.symbol, s.day))
    for d, bars in market.spy.days.items():
        await seed_bars(engine, "SPY", bars)
        units.add(("SPY", d))
    await mark_done(engine, units)
