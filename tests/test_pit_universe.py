"""The point-in-time liquidity universe (data/universe.py)."""

from __future__ import annotations

from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data.alpaca_market import Asset
from halal_trader.data.universe import month_starts, stock_symbols, universe_at


def _asset(symbol: str, exchange: str = "NYSE", status: str = "active") -> Asset:
    return Asset(symbol, "", exchange, True, False, status)


def test_only_plain_exchange_tickers_are_stocks() -> None:
    assets = [
        _asset("AAPL", "NASDAQ"),
        _asset("SIVB", "NASDAQ", "inactive"),  # delisted: kept
        _asset("BRK.B"),
        _asset("ALZH_DELISTED", "NASDAQ", "inactive"),
        _asset("PBTDF", "OTC"),
        _asset("SPY", "ARCA"),
    ]
    assert stock_symbols(assets) == ["AAPL", "SIVB"]


def test_month_starts_cross_a_year() -> None:
    assert month_starts(date(2025, 11, 20), date(2026, 2, 1)) == [
        date(2025, 11, 1),
        date(2025, 12, 1),
        date(2026, 1, 1),
        date(2026, 2, 1),
    ]


async def _months(
    engine: AsyncEngine, symbol: str, months: list[date], close: float, vol: float
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO monthly_bars (symbol, month, close, volume, vwap) "
                "VALUES (:s, :m, :c, :v, :c)"
            ),
            [{"s": symbol, "m": m, "c": close, "v": vol} for m in months],
        )


async def test_the_universe_knows_only_the_trailing_twelve_months(engine: AsyncEngine) -> None:
    year = month_starts(date(2025, 3, 1), date(2026, 2, 1))  # the 12 months before 2026-03
    await _months(engine, "BIG", year, 100.0, 1e6)  # 1e8 a month
    await _months(engine, "MID", year, 50.0, 1e6)  # 5e7
    await _months(engine, "DEAD", year[:8], 80.0, 1e6)  # delisted after 8 months: still counts
    await _months(engine, "PENNY", year, 2.0, 1e9)  # huge volume, under $5
    await _months(engine, "NEW", year[-3:], 100.0, 1e9)  # only 3 months of history
    await _months(engine, "LATER", [date(2026, 3, 1)], 100.0, 1e12)  # the as-of month: unknown

    assert await universe_at(engine, date(2026, 3, 15), top_n=10) == ["BIG", "DEAD", "MID"]
    assert await universe_at(engine, date(2026, 3, 15), top_n=1) == ["BIG"]


async def test_pit_schedule_joins_that_months_universe_with_the_last_screen_before_it(
    engine: AsyncEngine,
) -> None:
    from halal_trader.research.pit import UNMAPPED, pit_schedule

    year = month_starts(date(2025, 1, 1), date(2026, 3, 1))
    for sym in ("A", "B", "GONE"):
        await _months(engine, sym, year, 100.0, 1e6)
    async with engine.begin() as conn:
        rows = [
            # 2025-12-31: A halal, B not, GONE unmapped.
            ("2025-12-31", "A", "halal", ""),
            ("2025-12-31", "B", "not_halal", ""),
            ("2025-12-31", "GONE", "doubtful", UNMAPPED),
            # 2026-03-31: B becomes halal -- after the months asked for below.
            ("2026-03-31", "A", "halal", ""),
            ("2026-03-31", "B", "halal", ""),
        ]
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES (:a, :s, :d, :v, "
                "'[]', '{}', 'test', now())"
            ),
            [{"a": date.fromisoformat(a), "s": s, "v": v, "d": d} for a, s, v, d in rows],
        )

    sched = await pit_schedule(engine, start=date(2025, 12, 1), end=date(2026, 3, 1), top_n=10)

    assert sched.eligible_from[date(2025, 12, 1)] == set()  # no screen yet
    assert sched.eligible_from[date(2026, 1, 1)] == {"A"}
    assert sched.eligible_from[date(2026, 3, 1)] == {"A"}  # the March 31 screen is in the future
    assert sched.universe == {"A", "B", "GONE"}
    assert sched.unmapped == {2026: 1 / 3}
