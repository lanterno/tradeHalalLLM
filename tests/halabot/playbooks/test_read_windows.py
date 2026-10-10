"""data/minutes.read_windows: many (symbol, session) units as arrays, one query per batch."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data import minutes
from halal_trader.data.minutes import BarArrays
from halal_trader.market_hours import MARKET_TZ

MON, TUE, WED, THU, FRI = (date(2016, 3, d) for d in (7, 8, 9, 10, 11))
HALF = date(2016, 11, 25)  # closes at 13:00


def epoch(day: date, hh: int, mm: int) -> int:
    return int(datetime.combine(day, time(hh, mm), MARKET_TZ).timestamp())


async def _raw_bar(engine: AsyncEngine, symbol: str, day: date, hh: int, mm: int, vwap) -> None:  # type: ignore[no-untyped-def]
    at = datetime.combine(day, time(hh, mm), MARKET_TZ).astimezone(UTC)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO minute_bars (symbol, ts, open, high, low, close, volume, vwap) "
                "VALUES (:s, :t, 10, 11, 9, 10.5, 100, :vw)"
            ),
            {"s": symbol, "t": at, "vw": vwap},
        )


async def test_read_windows_returns_regular_session_arrays_per_unit(engine: AsyncEngine) -> None:
    for hh, mm, vw in ((8, 0, 10.0), (9, 30, 10.2), (9, 31, None), (15, 59, 10.4), (16, 0, 10.5)):
        await _raw_bar(engine, "AAA", MON, hh, mm, vw)
    for hh, mm in ((12, 59), (13, 0), (13, 30)):  # HALF closes at 13:00
        await _raw_bar(engine, "AAA", HALF, hh, mm, 10.0)
    await _raw_bar(engine, "AAA", TUE, 10, 0, 10.0)  # not asked for

    got = await minutes.read_windows(
        engine, [("AAA", MON), ("AAA", HALF), ("BBB", MON), ("AAA", MON)]
    )
    assert sorted(got) == [("AAA", MON), ("AAA", HALF), ("BBB", MON)]
    mon = got[("AAA", MON)]
    assert list(mon.ts) == [epoch(MON, 9, 30), epoch(MON, 9, 31), epoch(MON, 15, 59)]
    assert mon.ts.dtype == np.int64 and mon.o.dtype == np.float64
    assert list(mon.o) == [10.0] * 3 and list(mon.h) == [11.0] * 3 and list(mon.l) == [9.0] * 3
    assert list(mon.c) == [10.5] * 3 and list(mon.v) == [100.0] * 3
    assert mon.vw[0] == 10.2 and math.isnan(mon.vw[1]) and mon.vw[2] == 10.4
    assert list(got[("AAA", HALF)].ts) == [epoch(HALF, 12, 59)]
    assert len(got[("BBB", MON)]) == 0


async def test_read_windows_batches_large_requests(engine: AsyncEngine, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(minutes, "_WINDOW_BATCH", 2)
    days = [MON, TUE, WED, THU, FRI]
    for d in days:
        await _raw_bar(engine, "AAA", d, 10, 0, 10.0)
    got = await minutes.read_windows(engine, [("AAA", d) for d in days])
    assert [len(got[("AAA", d)]) for d in days] == [1] * 5


def test_bar_arrays_empty() -> None:
    e = BarArrays.empty()
    assert len(e) == 0 and e.ts.dtype == np.int64
