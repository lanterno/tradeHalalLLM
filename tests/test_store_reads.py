"""The stored-bars reads every consumer shares (data/store.py)."""

from __future__ import annotations

from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data.store import last_closes, stored_symbols


async def _bar(engine: AsyncEngine, symbol: str, day: str, close: float, adj: str = "raw") -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES (:s, :d, :a, :c, :c, :c, :c, 1, now())"
            ),
            {"s": symbol, "d": date.fromisoformat(day), "a": adj, "c": close},
        )


async def test_last_closes_are_raw_positive_and_as_of_a_day(engine: AsyncEngine) -> None:
    await _bar(engine, "AAPL", "2026-10-01", 250.0)
    await _bar(engine, "AAPL", "2026-10-02", 255.0)
    await _bar(engine, "AAPL", "2026-10-02", 1.0, adj="all")  # adjusted: never a close here
    await _bar(engine, "BAD", "2026-10-01", 9.0)
    await _bar(engine, "BAD", "2026-10-02", 0.0)  # a zero close is bad data

    assert await last_closes(engine, ["AAPL", "BAD", "NONE"]) == {
        "AAPL": (date(2026, 10, 2), 255.0),
        "BAD": (date(2026, 10, 1), 9.0),
    }
    assert await last_closes(engine, ["AAPL"], on_or_before=date(2026, 10, 1)) == {
        "AAPL": (date(2026, 10, 1), 250.0)
    }
    assert await last_closes(engine, []) == {}


async def test_stored_symbols_by_adjustment(engine: AsyncEngine) -> None:
    await _bar(engine, "MSFT", "2026-10-01", 500.0)
    await _bar(engine, "AAPL", "2026-10-01", 250.0, adj="all")

    assert await stored_symbols(engine) == ["AAPL", "MSFT"]
    assert await stored_symbols(engine, "all") == ["AAPL"]
