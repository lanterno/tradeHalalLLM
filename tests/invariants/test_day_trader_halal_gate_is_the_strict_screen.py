"""INVARIANT: the day-trader's and the reactor's halal gate is the strict in-house screen.

It used a curated 20-name list, seven of which the strict screen fails. The
gate (HalalScreener.is_halal, asked by TradeExecutor._check_halal before
every BUY) now reads the newest strict screen at the moment of the order and
fails CLOSED on a stale or missing screen, a symbol the screen does not
cover, or any non-halal row.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.db.repository import Repository
from halal_trader.halal import strict
from halal_trader.halal.cache import HalalScreener

TODAY = date(2026, 10, 6)


async def _screen(engine: AsyncEngine, as_of: date, rows: dict[str, tuple[str, float]]) -> None:
    async with engine.begin() as conn:
        for symbol, (verdict, cap) in rows.items():
            await conn.execute(
                text(
                    "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, "
                    "verdict, reasons, metrics, method, screened_at) VALUES (:a, :s, NULL, 'x', "
                    ":v, '[]'::jsonb, CAST(:m AS JSONB), 'test', :t)"
                ),
                {
                    "a": as_of,
                    "s": symbol,
                    "v": verdict,
                    "m": json.dumps({"market_cap": cap}),
                    "t": datetime.now(UTC),
                },
            )


@pytest.fixture
async def screened(engine: AsyncEngine) -> AsyncEngine:
    await _screen(
        engine,
        TODAY - timedelta(days=12),
        {"NVDA": ("halal", 5e12), "AAPL": ("halal", 4e12)},  # an older screen
    )
    await _screen(
        engine,
        TODAY - timedelta(days=1),
        {
            "AAPL": ("halal", 4e12),
            "MSFT": ("halal", 3.9e12),
            "NVDA": ("not_halal", 5e12),
            "TSM": ("doubtful", 1e12),
            "CSCO": ("halal", 2e11),
        },
    )
    return engine


async def test_the_newest_fresh_screen_decides(screened: AsyncEngine) -> None:
    async def halal(symbol: str) -> bool:
        return (await strict.verdict(screened, symbol, today=TODAY)).halal

    assert await halal("AAPL")
    assert not await halal("NVDA")  # halal in the older screen; the newest fails it
    assert not await halal("TSM")  # doubtful is not halal
    assert not await halal("ZZZZ")  # not covered by the screen


async def test_a_stale_or_missing_screen_makes_nothing_halal(
    engine: AsyncEngine, screened: AsyncEngine
) -> None:
    late = TODAY + timedelta(days=30)
    v = await strict.verdict(screened, "AAPL", today=late)
    assert not v.halal and "stale" in v.reason

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM halal_screen_results"))
    assert not (await strict.verdict(engine, "AAPL", today=TODAY)).halal


async def test_the_universe_is_the_largest_halal_names(screened: AsyncEngine) -> None:
    as_of, names = await strict.halal_universe(screened, today=TODAY, limit=2)

    assert as_of == TODAY - timedelta(days=1)
    assert names == ["AAPL", "MSFT"]
    assert (await strict.halal_universe(screened, today=TODAY + timedelta(days=30), limit=2))[
        1
    ] == []


async def test_the_screener_gates_on_the_strict_screen_not_its_cache(
    screened: AsyncEngine,
) -> None:
    repo = Repository(screened)
    await repo.cache_halal_status("NVDA", "halal", "the old curated list")
    screener = HalalScreener(repo, engine=screened, today=lambda: TODAY)

    # Before any refresh the cache still says NVDA is halal; the gate does not care.
    assert not await screener.is_halal("NVDA")
    assert await screener.is_halal("CSCO")  # strict-halal, wherever it ranks

    await screener.ensure_cache(force=True)

    assert sorted(await screener.get_halal_symbols()) == ["AAPL", "CSCO", "MSFT"]
    assert await repo.get_halal_status("NVDA") is None  # dropped from the universe


async def test_a_stale_screen_empties_the_universe(screened: AsyncEngine) -> None:
    repo = Repository(screened)
    screener = HalalScreener(repo, engine=screened, today=lambda: TODAY)
    await screener.ensure_cache(force=True)
    assert await screener.get_halal_symbols()

    later = HalalScreener(repo, engine=screened, today=lambda: TODAY + timedelta(days=30))
    await later.ensure_cache(force=True)

    assert await later.get_halal_symbols() == []
    assert not await later.is_halal("AAPL")


async def test_the_order_boundary_refuses_a_name_the_strict_screen_fails(
    screened: AsyncEngine,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from halal_trader.trading.executor import TradeExecutor

    screener = HalalScreener(Repository(screened), engine=screened, today=lambda: TODAY)
    executor = TradeExecutor(
        MagicMock(),
        MagicMock(),
        max_position_pct=0.2,
        max_simultaneous_positions=5,
        screener=screener,
    )
    executor._broker = MagicMock(place_order=AsyncMock())

    assert await executor._check_halal("NVDA") is not None
    assert await executor._check_halal("AAPL") is None


async def test_the_recommendation_universe_is_the_strict_screen(screened: AsyncEngine) -> None:
    from unittest.mock import MagicMock

    from halal_trader.recommendation.engine import DailyRecommendationEngine

    eng = DailyRecommendationEngine(
        broker=MagicMock(), repo=MagicMock(), settings=MagicMock(), llm=MagicMock(), engine=screened
    )

    universe = await eng._resolve_universe()

    assert "NVDA" not in universe
    assert universe[:2] == ["AAPL", "MSFT"]
