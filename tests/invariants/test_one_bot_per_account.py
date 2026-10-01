"""INVARIANT: one stock bot per database, and `start --once` never flattens the book."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from halal_trader.trading.scheduler import TradingBot


def _bot(engine: AsyncEngine) -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    bot._engine = engine
    return bot


async def test_a_second_bot_on_the_same_database_is_refused(database_url: str) -> None:
    # Two separate engines = two processes.
    first_engine = create_async_engine(database_url)
    second_engine = create_async_engine(database_url)
    try:
        first, second = _bot(first_engine), _bot(second_engine)
        await first._acquire_trading_lock()

        with pytest.raises(RuntimeError, match="trading lock"):
            await second._acquire_trading_lock()

        await first._release_trading_lock()  # first bot stops
        await second._acquire_trading_lock()  # now the second may run
        await second._release_trading_lock()
    finally:
        await first_engine.dispose()
        await second_engine.dispose()


async def test_run_once_runs_one_cycle_and_never_the_end_of_day_flatten() -> None:
    bot = TradingBot.__new__(TradingBot)
    bot._acquire_lock = MagicMock()  # type: ignore[method-assign]
    bot.initialize = AsyncMock()  # type: ignore[method-assign]
    bot._acquire_trading_lock = AsyncMock()  # type: ignore[method-assign]
    bot.pre_market = AsyncMock()  # type: ignore[method-assign]
    bot.end_of_day = AsyncMock()  # type: ignore[method-assign]
    bot.shutdown = AsyncMock()  # type: ignore[method-assign]
    cycle = MagicMock()
    cycle.run_cycle = AsyncMock()
    bot._get_cycle_service = MagicMock(return_value=cycle)  # type: ignore[method-assign]

    await bot.run_once()

    cycle.run_cycle.assert_awaited_once()
    bot.end_of_day.assert_not_awaited()  # used to close every position
    bot._acquire_trading_lock.assert_awaited_once()
    bot.shutdown.assert_awaited_once()
