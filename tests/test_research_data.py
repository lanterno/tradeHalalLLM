"""Research data store: Alpaca market-data client, liquidity universe, bar upserts."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data.alpaca_market import AlpacaMarketData, Asset, DailyBar, parse_bar
from halal_trader.data.store import (
    BENCHMARKS,
    liquid_universe,
    store_bars,
    sync_assets,
    update_bars,
)

# A real SIP bar as Alpaca returns it (AAPL, 2026-09-30).
REAL_BAR = {
    "c": 333.02,
    "h": 339.5,
    "l": 330.14,
    "n": 701605,
    "o": 330.8,
    "t": "2026-09-30T04:00:00Z",
    "v": 50149031,
    "vw": 335.1,
}


def test_parse_a_real_bar() -> None:
    b = parse_bar("AAPL", REAL_BAR)
    assert (b.day, b.close, b.volume, b.trades) == (date(2026, 9, 30), 333.02, 50149031.0, 701605)


def test_a_bar_missing_fields_fails_loudly() -> None:
    with pytest.raises(KeyError):
        parse_bar("AAPL", {"t": "2026-09-30T04:00:00Z", "c": 1.0})


async def test_client_follows_pages_and_backs_off_on_429() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        assert params["feed"] == "sip"
        assert request.headers["APCA-API-KEY-ID"] == "k"
        if len(calls) == 1:
            return httpx.Response(429)
        if "page_token" not in params:
            return httpx.Response(200, json={"bars": {"AAPL": [REAL_BAR]}, "next_page_token": "p2"})
        nxt = dict(REAL_BAR, t="2026-10-01T04:00:00Z")
        return httpx.Response(200, json={"bars": {"AAPL": [nxt]}, "next_page_token": None})

    async def no_sleep(_s: float) -> None: ...

    import halal_trader.core.http as mod

    client = AlpacaMarketData(
        "k", "s", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), min_interval_s=0
    )
    orig = mod.asyncio.sleep
    mod.asyncio.sleep = no_sleep  # type: ignore[assignment]
    try:
        bars = await client.daily_bars(["aapl"], start=date(2026, 9, 1))
    finally:
        mod.asyncio.sleep = orig  # type: ignore[assignment]
        await client.aclose()

    assert [b.day for b in bars] == [date(2026, 9, 30), date(2026, 10, 1)]
    assert len(calls) == 3  # 429, page 1, page 2


class FakeMarket:
    def __init__(self, assets: list[Asset], bars: dict[str, list[DailyBar]]) -> None:
        self._assets = assets
        self._bars = bars
        self.requests: list[tuple[tuple[str, ...], date, str]] = []

    async def assets(self) -> list[Asset]:
        return self._assets

    async def daily_bars(self, symbols, *, start, end=None, adjustment="raw"):  # type: ignore[no-untyped-def]
        self.requests.append((tuple(symbols), start, adjustment))
        return [b for s in symbols for b in self._bars.get(s, []) if b.day >= start]


def _bar(sym: str, day: date, close: float, volume: float) -> DailyBar:
    return DailyBar(sym, day, close, close, close, close, volume, None, None)


def _asset(sym: str, exchange: str = "NASDAQ", tradable: bool = True) -> Asset:
    return Asset(sym, sym, exchange, tradable, True, "active")


async def test_universe_ranks_by_dollar_volume_and_drops_penny_and_odd_symbols(
    engine: AsyncEngine,
) -> None:
    days = [date(2026, 9, d) for d in range(1, 26)]
    bars = {
        "BIG": [_bar("BIG", d, 100.0, 1_000_000) for d in days],  # $100M/day
        "MID": [_bar("MID", d, 50.0, 100_000) for d in days],  # $5M/day
        "PENNY": [_bar("PENNY", d, 1.0, 900_000_000) for d in days],  # below $5
        "NEW": [_bar("NEW", d, 100.0, 9_000_000) for d in days[:5]],  # too few sessions
    }
    assets = [_asset(s) for s in bars] + [_asset("BRK.B", "NYSE"), _asset("ETF", "ARCA")]
    fake = FakeMarket(assets, bars)
    await sync_assets(engine, fake)  # type: ignore[arg-type]

    members = await liquid_universe(engine, fake, top_n=10, as_of=date(2026, 9, 30))  # type: ignore[arg-type]

    assert [m.symbol for m in members] == ["BIG", "MID"]
    asked = set(fake.requests[0][0])
    assert "BRK.B" not in asked and "ETF" not in asked  # share classes and ARCA ETFs skipped


async def test_bars_upsert_and_update_resumes_from_the_last_session(engine: AsyncEngine) -> None:
    early = [_bar("AAPL", date(2026, 9, d), 300.0, 1.0) for d in (28, 29)]
    later = early + [_bar("AAPL", date(2026, 9, 30), 333.0, 1.0)]
    fake = FakeMarket([], {"AAPL": later})
    await store_bars(engine, early, "raw")

    await update_bars(engine, fake, ["AAPL"], since=date(2016, 1, 1), adjustments=("raw",))  # type: ignore[arg-type]

    assert fake.requests == [(("AAPL",), date(2026, 9, 28), "raw")]  # last day - 1, not 2016
    async with engine.connect() as conn:
        n = (await conn.execute(text("SELECT count(*) FROM daily_bars"))).scalar()
    assert n == 3  # upserted, no duplicates


def test_benchmarks_include_the_halal_etfs() -> None:
    assert {"SPUS", "HLAL"} <= set(BENCHMARKS)
