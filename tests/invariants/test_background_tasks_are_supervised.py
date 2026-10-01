"""INVARIANT: the SL/TP monitor and the news reactor are restarted if they die.

They used to be bare create_task()s: an escaped exception ended the monitor
silently, and the bot kept trading with no stop-loss enforcement.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from halal_trader.trading.scheduler import TradingBot


def _bot() -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    bot._running = True
    bot._alerts = SimpleNamespace(notify=AsyncMock())
    return bot


async def test_a_crashed_component_is_alerted_and_restarted() -> None:
    bot = _bot()
    runs = 0

    async def component() -> None:
        nonlocal runs
        runs += 1
        if runs == 1:
            raise RuntimeError("boom")
        bot._running = False  # second run: shut down cleanly

    await bot._supervise("position monitor", component, backoff_s=0)

    assert runs == 2
    kind, message = bot._alerts.notify.await_args.args
    assert kind == "supervisor.position_monitor"
    assert "RuntimeError: boom" in message


async def test_a_component_that_returns_while_running_is_restarted() -> None:
    bot = _bot()
    runs = 0

    async def component() -> None:
        nonlocal runs
        runs += 1
        if runs == 2:
            bot._running = False

    await bot._supervise("news reactor", component, backoff_s=0)

    assert runs == 2
    assert "returned unexpectedly" in bot._alerts.notify.await_args.args[1]


async def test_shutdown_cancellation_passes_through() -> None:
    bot = _bot()
    started = asyncio.Event()

    async def component() -> None:
        started.set()
        await asyncio.sleep(3600)

    task = asyncio.create_task(bot._supervise("position monitor", component, backoff_s=0))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    bot._alerts.notify.assert_not_awaited()  # a clean shutdown is not an incident
