"""WebSocket reconnect + heartbeat helpers.

The dashboard's ``/ws/prices`` endpoint and the Binance WS manager both
need the same recovery story when the connection drops:

* Reconnect with **exponential backoff** so a flapping endpoint doesn't
  hammer us with reconnects every second.
* **Heartbeat** so half-dead connections (TCP keepalive intact, app
  layer hung) get noticed and recycled inside one loop interval rather
  than 15 minutes of silent stalling.

Both are pure-async generators / contexts that the call site composes
around its actual receive-loop. We don't bake either into a specific
WS library — kept abstract so the pattern works against any
async-iterating connection.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass
class BackoffPolicy:
    """Exponential backoff with cap + reset window.

    ``reset_after_seconds`` resets the attempt counter after a
    successful run that lasted at least that long, so a connection that
    survives an hour doesn't keep paying for last week's flap.
    """

    base_seconds: float = 1.0
    max_seconds: float = 30.0
    factor: float = 2.0
    reset_after_seconds: float = 60.0


class _BackoffState:
    def __init__(self, policy: BackoffPolicy) -> None:
        self._policy = policy
        self._attempt = 0

    def next_delay(self) -> float:
        delay = min(
            self._policy.max_seconds,
            self._policy.base_seconds * (self._policy.factor**self._attempt),
        )
        self._attempt += 1
        return delay

    def reset(self) -> None:
        self._attempt = 0


async def reconnect_loop(
    *,
    name: str,
    connect_and_run: Callable[[], Awaitable[None]],
    is_running: Callable[[], bool],
    policy: BackoffPolicy | None = None,
) -> None:
    """Repeatedly run ``connect_and_run`` until ``is_running()`` is False.

    On exception, sleep with exponential backoff and try again. On any
    successful run that lasted longer than ``policy.reset_after_seconds``
    the backoff counter resets — the connection proved stable, so the
    *next* drop should be retried immediately.
    """
    policy = policy or BackoffPolicy()
    state = _BackoffState(policy)
    loop = asyncio.get_event_loop()

    while is_running():
        started = loop.time()
        try:
            await connect_and_run()
            elapsed = loop.time() - started
            if elapsed >= policy.reset_after_seconds:
                state.reset()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            elapsed = loop.time() - started
            if elapsed >= policy.reset_after_seconds:
                state.reset()
            delay = state.next_delay()
            logger.warning(
                "%s WS connection error after %.1fs: %s — reconnecting in %.1fs",
                name,
                elapsed,
                e,
                delay,
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise


async def heartbeat_guard(
    *,
    last_activity: Callable[[], float],
    interval_seconds: float,
    timeout_seconds: float,
    on_stall: Callable[[], Awaitable[None]] | None = None,
    is_running: Callable[[], bool] | None = None,
) -> None:
    """Background task that fires ``on_stall`` when no activity has occurred.

    ``last_activity`` returns the monotonic timestamp of the last
    received frame. We sleep ``interval_seconds`` then check whether
    ``timeout_seconds`` have elapsed since that timestamp. If so the
    connection is considered stalled and ``on_stall`` is invoked once.
    Caller is responsible for actually closing/reconnecting in the
    callback.
    """

    def _always() -> bool:
        return True

    running: Callable[[], bool] = _always if is_running is None else is_running

    loop = asyncio.get_event_loop()
    while running():
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            return
        if not running():
            return
        idle = loop.time() - last_activity()
        if idle > timeout_seconds:
            logger.warning(
                "WS heartbeat timeout: %.1fs since last frame (limit %.1fs) — stalling",
                idle,
                timeout_seconds,
            )
            if on_stall is not None:
                try:
                    await on_stall()
                except Exception as e:
                    logger.warning("heartbeat on_stall callback raised: %s", e)
            # Reset the clock so we don't keep firing on each tick — the
            # caller's reconnect should bump last_activity once it finishes.
            return
