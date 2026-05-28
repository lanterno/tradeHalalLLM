"""Root task supervisor (REARCHITECTURE Appendix F).

Runs the engine's long-lived loops (heartbeat, perception sources, and — when
live — the monitor/reconcile/risk loops) as supervised tasks: a loop that exits
or crashes is restarted after a backoff, one loop's failure never takes down the
others (INV-1), and crashes are logged with their type (INV-4). Cancellation
(``shutdown``) is clean and never treated as a crash.

``spawn`` takes a *factory* (0-arg callable returning a fresh awaitable) so each
restart re-creates the coroutine.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from halabot.platform.bus import EventBus
from halabot.platform.clock import Clock
from halabot.platform.events import EventType, new_event

logger = logging.getLogger(__name__)

Factory = Callable[[], Awaitable[None]]
Sleep = Callable[[float], Awaitable[None]]


class Supervisor:
    """Restart-on-error manager for named long-lived loops."""

    def __init__(self, *, restart_backoff_s: float = 5.0, sleep: Sleep = asyncio.sleep) -> None:
        self._backoff = restart_backoff_s
        self._sleep = sleep
        self._tasks: list[asyncio.Task[None]] = []

    def spawn(self, name: str, factory: Factory, *, restart: bool = True) -> None:
        self._tasks.append(asyncio.create_task(self._supervise(name, factory, restart), name=name))

    async def _supervise(self, name: str, factory: Factory, restart: bool) -> None:
        while True:
            try:
                await factory()
                logger.warning("task %s exited unexpectedly", name)
            except asyncio.CancelledError:
                raise  # shutdown — not a crash
            except Exception as exc:  # noqa: BLE001 — isolate + self-heal (INV-1/INV-4)
                logger.error(
                    "task %s crashed: %r — restarting after %.0fs", name, exc, self._backoff
                )
            if not restart:
                return
            await self._sleep(self._backoff)

    async def shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()


async def heartbeat_loop(
    bus: EventBus, clock: Clock, interval_s: float, *, sleep: Sleep = asyncio.sleep
) -> None:
    """Emit a ``system.heartbeat`` every ``interval_s`` — drives time-decay so
    conviction fades on the passage of time even with no new data (R-08)."""
    while True:
        await sleep(interval_s)
        await bus.publish(new_event(clock, EventType.SYSTEM_HEARTBEAT, source="heartbeat"))
