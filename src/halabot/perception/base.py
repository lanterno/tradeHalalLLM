"""Source contract + restart-on-error supervisor (REARCHITECTURE L1).

A :class:`Source` is a long-lived coroutine that emits ``observation.*`` events.
:class:`SourceSupervisor` runs each source as a task and restarts it (after a
backoff) if it exits or crashes — so a flaky feed self-heals and one source's
failure never takes down the others (INV-1/INV-2). Crashes are logged with the
exception type (INV-4).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from halabot.platform.events import Event

logger = logging.getLogger(__name__)

Emit = Callable[[Event], Awaitable[None]]
Sleep = Callable[[float], Awaitable[None]]


class Source(Protocol):
    """A long-lived feed adapter. ``run`` loops until cancelled, emitting events."""

    name: str

    async def run(self, emit: Emit) -> None: ...


class SourceSupervisor:
    """Runs sources as supervised tasks with restart-on-error.

    ``sleep`` is injectable so tests run without real delays. Cancellation
    (``stop``) propagates cleanly and is never treated as a crash.
    """

    def __init__(self, *, restart_backoff_s: float = 5.0, sleep: Sleep = asyncio.sleep) -> None:
        self._backoff = restart_backoff_s
        self._sleep = sleep
        self._tasks: list[asyncio.Task[None]] = []

    def start(self, sources: list[Source], emit: Emit) -> None:
        for source in sources:
            self._tasks.append(
                asyncio.create_task(self._supervise(source, emit), name=f"source:{source.name}")
            )

    async def _supervise(self, source: Source, emit: Emit) -> None:
        while True:
            try:
                await source.run(emit)
                logger.warning(
                    "source %s exited unexpectedly — restarting after %.0fs",
                    source.name,
                    self._backoff,
                )
            except asyncio.CancelledError:
                raise  # shutdown — not a crash
            except Exception as exc:  # noqa: BLE001 — isolate + self-heal (INV-1/INV-2)
                logger.error(
                    "source %s crashed: %r — restarting after %.0fs",
                    source.name,
                    exc,
                    self._backoff,
                )
            await self._sleep(self._backoff)

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
