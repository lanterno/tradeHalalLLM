"""Structured-concurrency supervisor for the bot's background tasks.

Wave C swaps the ad-hoc ``asyncio.create_task(...)`` per subsystem
for one rooted ``anyio`` task group with explicit restart policies.
A crash in one supervised task either:

* ``RestartPolicy.CRASH_BOT`` — propagates the exception so the
  whole bot shuts down. The default for safety-critical loops
  (monitor, ws_manager) where silent failure is unacceptable.
* ``RestartPolicy.RESTART`` — logs the error, sleeps with
  exponential backoff, and restarts the coroutine. Suitable for
  best-effort sources (sentiment_manager, news_reactor) where a
  flaky upstream shouldn't kill the bot.
* ``RestartPolicy.IGNORE`` — log-and-forget. Reserved for tasks
  whose work is incidental (the optional reddit fetcher).

Use::

    async with TaskSupervisor() as sup:
        sup.start("monitor", monitor.run, policy=RestartPolicy.CRASH_BOT)
        sup.start("news", news_reactor.poll, policy=RestartPolicy.RESTART)
        # ... main bot loop runs in this scope ...
        await main_loop()

When the ``async with`` exits, every supervised task is cancelled
cleanly via the task group's normal teardown.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import TYPE_CHECKING

import anyio

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

logger = logging.getLogger(__name__)


class RestartPolicy(enum.Enum):
    CRASH_BOT = "crash_bot"
    RESTART = "restart"
    IGNORE = "ignore"


_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 60.0


class TaskSupervisor:
    """Async-context manager that owns a root ``anyio`` task group."""

    def __init__(self) -> None:
        self._tg: "TaskGroup | None" = None

    async def __aenter__(self) -> "TaskSupervisor":
        self._tg = anyio.create_task_group()
        await self._tg.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._tg is None:
            return
        try:
            await self._tg.__aexit__(exc_type, exc, tb)
        finally:
            self._tg = None

    def cancel(self) -> None:
        """Cancel every supervised task so the scope exits promptly."""
        if self._tg is not None:
            self._tg.cancel_scope.cancel()

    def start(
        self,
        name: str,
        func: Callable[[], Awaitable[None]],
        *,
        policy: RestartPolicy = RestartPolicy.CRASH_BOT,
    ) -> None:
        """Spawn one supervised task into the rooted group."""
        if self._tg is None:
            raise RuntimeError("TaskSupervisor is not active — use 'async with'")
        self._tg.start_soon(self._run_one, name, func, policy)

    async def _run_one(
        self,
        name: str,
        func: Callable[[], Awaitable[None]],
        policy: RestartPolicy,
    ) -> None:
        """Wrap one supervised task with its restart policy."""
        if policy is RestartPolicy.CRASH_BOT:
            # Let the task group catch any exception — anyio cancels
            # all sibling tasks and re-raises at the context exit.
            await func()
            return

        backoff = _INITIAL_BACKOFF_S
        while True:
            try:
                await func()
                logger.info("supervised task %r exited cleanly", name)
                return
            except (anyio.get_cancelled_exc_class(),):
                raise
            except Exception as exc:  # noqa: BLE001
                if policy is RestartPolicy.IGNORE:
                    logger.warning("ignoring failure in %r: %s", name, exc)
                    return
                logger.warning(
                    "supervised task %r crashed (will restart in %.1fs): %s",
                    name,
                    backoff,
                    exc,
                )
                await anyio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_S)
