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


async def _lock_pid(bot: TradingBot) -> int:
    from sqlalchemy import text

    return int((await bot._trading_lock_conn.execute(text("SELECT pg_backend_pid()"))).scalar())


async def test_the_lock_session_holds_no_transaction_open(database_url: str) -> None:
    # A lock taken inside a never-committed transaction pinned the database's
    # xmin horizon for the bot's whole life, so vacuum reclaimed nothing.
    from sqlalchemy import text

    engine, observer = create_async_engine(database_url), create_async_engine(database_url)
    bot = _bot(engine)
    try:
        await bot._acquire_trading_lock()
        await bot._holds_trading_lock()  # a second statement on the session
        pid = await _lock_pid(bot)
        async with observer.connect() as conn:
            row = (
                await conn.execute(
                    text("SELECT state, xact_start FROM pg_stat_activity WHERE pid = :p"),
                    {"p": pid},
                )
            ).one()
        assert row.state == "idle"
        assert row.xact_start is None
    finally:
        await bot._release_trading_lock()
        await engine.dispose()
        await observer.dispose()


async def test_a_lost_lock_is_taken_back(database_url: str) -> None:
    from sqlalchemy import text

    engine, admin = create_async_engine(database_url), create_async_engine(database_url)
    bot = _bot(engine)
    try:
        await bot._acquire_trading_lock()
        assert await bot._holds_trading_lock()
        pid = await _lock_pid(bot)
        async with admin.connect() as conn:  # a Postgres restart, in miniature
            await conn.execute(text("SELECT pg_terminate_backend(:p)"), {"p": pid})
        assert not await bot._holds_trading_lock()

        await bot._ensure_trading_lock()

        assert await bot._holds_trading_lock()
        assert await _lock_pid(bot) != pid
    finally:
        await bot._release_trading_lock()
        await engine.dispose()
        await admin.dispose()


async def test_a_lost_lock_another_bot_took_stops_this_bot(database_url: str) -> None:
    from sqlalchemy import text

    engines = [create_async_engine(database_url) for _ in range(3)]
    first, second = _bot(engines[0]), _bot(engines[1])
    try:
        await first._acquire_trading_lock()
        pid = await _lock_pid(first)
        async with engines[2].connect() as conn:
            await conn.execute(text("SELECT pg_terminate_backend(:p)"), {"p": pid})
        await second._acquire_trading_lock()  # the gap a second bot could use

        with pytest.raises(RuntimeError, match="trading lock lost"):
            await first._ensure_trading_lock()
    finally:
        await second._release_trading_lock()
        for e in engines:
            await e.dispose()
