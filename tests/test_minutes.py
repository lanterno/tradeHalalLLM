"""Minute bars (data/minutes.py): the batched backfill, its resume units, and session reads."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data import minutes
from halal_trader.data.alpaca_market import AlpacaMarketData
from halal_trader.market_hours import MARKET_TZ

MON = date(2026, 10, 5)
TUE = date(2026, 10, 6)
HALF_DAY = date(2026, 11, 27)  # the day after Thanksgiving closes at 13:00


def _raw(day: date, hh: int, mm: int, price: float = 10.0) -> dict[str, Any]:
    at = datetime.combine(day, time(hh, mm), MARKET_TZ).astimezone(UTC)
    return {
        "t": at.isoformat().replace("+00:00", "Z"),
        "o": price,
        "h": price + 0.1,
        "l": price - 0.1,
        "c": price,
        "v": 100.0,
        "vw": price,
        "n": 3,
    }


class FakeMarket:
    def __init__(self, bars: dict[tuple[str, date], list[dict[str, Any]]]) -> None:
        self._bars = bars
        self.calls: list[tuple[tuple[str, ...], date]] = []

    async def minute_bars_many(self, symbols, *, start, end):  # type: ignore[no-untyped-def]
        day = start.astimezone(MARKET_TZ).date()
        self.calls.append((tuple(symbols), day))
        return {s: self._bars[(s, day)] for s in symbols if (s, day) in self._bars}


async def _units(engine: AsyncEngine) -> dict[str, int]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT unit, items FROM backfill_progress WHERE task = 'minute'")
        )
        return {r.unit: r.items for r in rows}


async def test_backfill_fetches_each_session_once_for_all_its_symbols(engine: AsyncEngine) -> None:
    market = FakeMarket(
        {
            ("AAPL", MON): [_raw(MON, 9, 30), _raw(MON, 9, 31)],
            ("MSFT", MON): [_raw(MON, 9, 30)],
            ("AAPL", TUE): [_raw(TUE, 15, 59)],
        }
    )
    later = datetime(2026, 10, 7, tzinfo=UTC)
    units = [("AAPL", MON), ("MSFT", MON), ("NEWCO", MON), ("AAPL", TUE)]

    assert await minutes.backfill(engine, market, units, now=later) == 4
    assert market.calls == [(("AAPL",), TUE), (("AAPL", "MSFT", "NEWCO"), MON)]  # newest first
    # every unit is done, the empty one too, so none is asked for again
    assert await _units(engine) == {
        "AAPL:2026-10-05": 2,
        "MSFT:2026-10-05": 1,
        "NEWCO:2026-10-05": 0,
        "AAPL:2026-10-06": 1,
    }
    assert await minutes.backfill(engine, market, units, now=later) == 0
    assert len(market.calls) == 2


async def test_a_session_fetched_before_it_settled_is_fetched_again(engine: AsyncEngine) -> None:
    market = FakeMarket({("AAPL", MON): [_raw(MON, 9, 30)]})
    during = datetime.combine(MON, time(15, 0), MARKET_TZ)

    await minutes.backfill(engine, market, [("AAPL", MON)], now=during)
    assert await _units(engine) == {}  # stored, not done
    await minutes.backfill(engine, market, [("AAPL", MON)], now=during + timedelta(days=1))
    assert len(market.calls) == 2
    assert await _units(engine) == {"AAPL:2026-10-05": 1}


async def test_reads_keep_the_regular_session_only(engine: AsyncEngine) -> None:
    await minutes.store(
        engine,
        {
            "AAPL": [
                _raw(MON, 9, 29),  # pre-market
                _raw(MON, 9, 30, 11.0),
                _raw(MON, 15, 59, 12.0),
                _raw(MON, 16, 0),  # the post-close bar Alpaca's inclusive end returns
                _raw(HALF_DAY, 12, 59, 13.0),
                _raw(HALF_DAY, 13, 0),  # after the early close
            ]
        },
    )
    got = await minutes.read_sessions(engine, "AAPL", [MON, TUE, HALF_DAY])
    assert [b.close for b in got[MON]] == [11.0, 12.0]
    assert got[TUE] == []
    assert [b.close for b in got[HALF_DAY]] == [13.0]
    assert got[MON][0].ts == datetime.combine(MON, time(9, 30), MARKET_TZ)
    assert await minutes.read(engine, "AAPL", HALF_DAY) == got[HALF_DAY]


async def test_the_client_asks_for_many_symbols_and_follows_pages() -> None:
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        assert params["timeframe"] == "1Min" and params["feed"] == "sip"
        if "page_token" not in params:
            page = {"AAPL": [_raw(MON, 9, 30)], "MSFT": [_raw(MON, 9, 30)]}
            return httpx.Response(200, json={"bars": page, "next_page_token": "p2"})
        return httpx.Response(200, json={"bars": {"MSFT": [_raw(MON, 9, 31)]}})

    client = AlpacaMarketData(
        "k", "s", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), min_interval_s=0
    )
    lo, hi = minutes.session_bounds(MON)
    try:
        got = await client.minute_bars_many(["msft", "aapl"], start=lo, end=hi)
        one = await client.minute_bars("AAPL", start=lo, end=hi)
    finally:
        await client.aclose()

    assert calls[0]["symbols"] == "AAPL,MSFT"
    assert {s: len(v) for s, v in got.items()} == {"AAPL": 1, "MSFT": 2}
    assert len(one) == 1
