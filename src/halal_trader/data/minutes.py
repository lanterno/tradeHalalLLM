"""Minute bars: the one writer and reader of ``minute_bars``.

Raw SIP bars of the regular session, stored per (symbol, session) unit. A
bar's ``ts`` is the **start** of its minute, and only minutes with trades
have a bar, so a gap is a quiet minute or a halt, never a price.

A unit fetched once its session is over (the close plus the SIP embargo) is
marked done in ``backfill_progress`` (task ``minute``, unit ``SYM:YYYY-MM-DD``),
empty units included: a name not yet listed, delisted or halted all day is
not asked for again, and a session fetched before its close is.

Fetching groups units by session and asks for many symbols per request (a
page holds 10,000 bars across them), so a backfill costs about one request
per 25 symbol-sessions rather than one each.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.market_hours import MARKET_OPEN, MARKET_TZ, effective_close_time

logger = logging.getLogger(__name__)

TASK = "minute"
_INSERT_CHUNK = 5_000
_SETTLED = timedelta(minutes=20)  # the SIP embargo (16 min) and a margin


@dataclass(frozen=True, slots=True)
class MinuteBar:
    ts: datetime  # the minute's start, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None


def session_bounds(day: date) -> tuple[datetime, datetime]:
    """[open, close) of ``day``'s regular session, early closes included."""
    return (
        datetime.combine(day, MARKET_OPEN, MARKET_TZ),
        datetime.combine(day, effective_close_time(day), MARKET_TZ),
    )


def unit(symbol: str, day: date) -> str:
    return f"{symbol}:{day.isoformat()}"


def settled(day: date, now: datetime) -> bool:
    """Whether ``day``'s session is over long enough for its bars to be final."""
    return now >= session_bounds(day)[1] + _SETTLED


def _row(symbol: str, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "s": symbol,
        "t": datetime.fromisoformat(str(raw["t"]).replace("Z", "+00:00")),
        "o": float(raw["o"]),
        "h": float(raw["h"]),
        "l": float(raw["l"]),
        "c": float(raw["c"]),
        "v": float(raw["v"]),
        "vw": float(raw["vw"]) if raw.get("vw") is not None else None,
    }


async def store(engine: AsyncEngine, bars: dict[str, list[dict[str, Any]]]) -> int:
    """Insert Alpaca's raw bars (``{symbol: [{t,o,h,l,c,v,vw}, ...]}``); keeps existing rows."""
    rows = [_row(symbol, b) for symbol, raws in bars.items() for b in raws]
    for i in range(0, len(rows), _INSERT_CHUNK):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO minute_bars (symbol, ts, open, high, low, close, volume, vwap) "
                    "VALUES (:s, :t, :o, :h, :l, :c, :v, :vw) ON CONFLICT DO NOTHING"
                ),
                rows[i : i + _INSERT_CHUNK],
            )
    return len(rows)


async def done_units(engine: AsyncEngine) -> set[str]:
    from halal_trader.events.history import _done

    return await _done(engine, TASK)


async def backfill(
    engine: AsyncEngine,
    market: Any,
    units: Iterable[tuple[str, date]],
    *,
    now: datetime | None = None,
    log_every: int = 50,
) -> int:
    """Fetch every (symbol, session) unit not yet done; returns the bars stored.

    Units are grouped by session, newest first, and each session's symbols
    are fetched together (``market.minute_bars_many``).
    """
    from halal_trader.events.history import mark_units

    done = await done_units(engine)
    by_day: dict[date, set[str]] = defaultdict(set)
    for symbol, day in units:
        if unit(symbol, day) not in done:
            by_day[day].add(symbol)
    stored = 0
    for k, day in enumerate(sorted(by_day, reverse=True), 1):
        lo, hi = session_bounds(day)
        symbols = sorted(by_day[day])
        bars = await market.minute_bars_many(symbols, start=lo, end=hi)
        stored += await store(engine, bars)
        if settled(day, now or datetime.now(UTC)):
            await mark_units(engine, TASK, {unit(s, day): len(bars.get(s, [])) for s in symbols})
        if k % log_every == 0:
            logger.info("minute backfill: %d/%d sessions, %d bars stored", k, len(by_day), stored)
    return stored


async def read(engine: AsyncEngine, symbol: str, day: date) -> list[MinuteBar]:
    """``symbol``'s stored bars in ``day``'s regular session, oldest first."""
    return (await read_sessions(engine, symbol, [day])).get(day, [])


async def read_sessions(
    engine: AsyncEngine, symbol: str, days: Iterable[date]
) -> dict[date, list[MinuteBar]]:
    """``symbol``'s stored regular-session bars for each of ``days``, by session."""
    wanted = sorted(set(days))
    out: dict[date, list[MinuteBar]] = {d: [] for d in wanted}
    bounds = {d: session_bounds(d) for d in wanted}
    runs: list[list[date]] = []  # sessions a week or less apart share one range scan
    for d in wanted:
        if runs and (d - runs[-1][-1]).days <= 7:
            runs[-1].append(d)
        else:
            runs.append([d])
    async with engine.connect() as conn:
        for run in runs:
            rows = await conn.execute(
                text(
                    "SELECT ts, open, high, low, close, volume, vwap FROM minute_bars "
                    "WHERE symbol = :s AND ts >= :lo AND ts < :hi ORDER BY ts"
                ),
                {"s": symbol, "lo": bounds[run[0]][0], "hi": bounds[run[-1]][1]},
            )
            for r in rows:
                day = r.ts.astimezone(MARKET_TZ).date()
                b = bounds.get(day)
                if b is not None and b[0] <= r.ts < b[1]:
                    out[day].append(
                        MinuteBar(
                            r.ts,
                            float(r.open),
                            float(r.high),
                            float(r.low),
                            float(r.close),
                            float(r.volume),
                            float(r.vwap) if r.vwap is not None else None,
                        )
                    )
    return out
