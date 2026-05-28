"""Source contract + restart-on-error supervisor (REARCHITECTURE L1).

A :class:`Source` is a long-lived coroutine that emits ``observation.*`` events.
:class:`SourceSupervisor` runs each source as a task and restarts it (after a
backoff) if it exits or crashes — so a flaky feed self-heals and one source's
failure never takes down the others (INV-1/INV-2). Crashes are logged with the
exception type (INV-4).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from halabot.platform.events import Event
from halabot.platform.supervisor import Supervisor

Emit = Callable[[Event], Awaitable[None]]
Sleep = Callable[[float], Awaitable[None]]


class Source(Protocol):
    """A long-lived feed adapter. ``run`` loops until cancelled, emitting events."""

    name: str

    async def run(self, emit: Emit) -> None: ...


class SourceSupervisor:
    """Runs sources as restart-on-error tasks (delegates to :class:`Supervisor`).

    Kept as a thin source-specialized facade over the general supervisor so the
    restart/backoff/isolation logic lives in one place (INV-1/INV-2)."""

    def __init__(self, *, restart_backoff_s: float = 5.0, sleep: Sleep = asyncio.sleep) -> None:
        self._sup = Supervisor(restart_backoff_s=restart_backoff_s, sleep=sleep)

    def start(self, sources: list[Source], emit: Emit) -> None:
        for source in sources:

            def factory(s: Source = source) -> Awaitable[None]:
                return s.run(emit)

            self._sup.spawn(f"source:{source.name}", factory)

    async def stop(self) -> None:
        await self._sup.shutdown()
