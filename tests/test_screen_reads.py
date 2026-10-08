"""Every consumer reads the strict screen through halal/strict.py."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.halal import strict
from halal_trader.portfolio.core_executor import is_halal_now

TODAY = date(2026, 10, 6)


async def _screen(engine: AsyncEngine, as_of: date, rows: dict[str, str]) -> None:
    async with engine.begin() as conn:
        for symbol, verdict in rows.items():
            await conn.execute(
                text(
                    "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, "
                    "verdict, reasons, metrics, method, screened_at) VALUES (:a, :s, 7, "
                    "'Services-prepackaged software', :v, '[]'::jsonb, CAST(:m AS JSONB), "
                    "'test', :t)"
                ),
                {
                    "a": as_of,
                    "s": symbol,
                    "v": verdict,
                    "m": json.dumps({"price": 10.0, "shares_outstanding": 2e9, "market_cap": 2e10}),
                    "t": datetime.now(UTC),
                },
            )


async def test_newest_screen_on_or_before_a_day(engine: AsyncEngine) -> None:
    await _screen(engine, TODAY - timedelta(days=8), {"AAPL": "halal"})
    await _screen(engine, TODAY - timedelta(days=1), {"AAPL": "halal"})

    assert await strict.newest_screen(engine) == TODAY - timedelta(days=1)
    assert await strict.newest_screen(engine, on_or_before=TODAY - timedelta(days=2)) == (
        TODAY - timedelta(days=8)
    )
    assert await strict.newest_screen(engine, on_or_before=TODAY - timedelta(days=30)) is None


async def test_screen_rows_filter_by_verdict_and_symbol(engine: AsyncEngine) -> None:
    await _screen(engine, TODAY, {"AAPL": "halal", "NVDA": "not_halal", "TSM": "doubtful"})

    every = await strict.screen_rows(engine, TODAY)
    assert [r.symbol for r in every] == ["AAPL", "NVDA", "TSM"]
    assert [r.symbol for r in await strict.screen_rows(engine, TODAY, halal_only=True)] == ["AAPL"]
    (tsm,) = await strict.screen_rows(engine, TODAY, symbols=["TSM", "ZZZZ"])
    assert (tsm.verdict, tsm.cik, tsm.price, tsm.shares, tsm.market_cap) == (
        "doubtful",
        7,
        10.0,
        2e9,
        2e10,
    )


async def test_both_strategies_apply_one_order_boundary_rule(engine: AsyncEngine) -> None:
    await _screen(engine, TODAY - timedelta(days=1), {"AAPL": "halal", "NVDA": "not_halal"})

    for symbol, halal in (("AAPL", True), ("NVDA", False), ("ABSENT", False)):
        v = await strict.verdict(engine, symbol, today=TODAY)
        assert await is_halal_now(engine, symbol, TODAY) == (halal, v.screen_as_of)
        assert v.halal is halal
    # Stale: neither passes anything.
    late = TODAY + timedelta(days=20)
    assert await is_halal_now(engine, "AAPL", late) == (False, TODAY - timedelta(days=1))
