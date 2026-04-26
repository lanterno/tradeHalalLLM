"""WS reconnect-loop + heartbeat tests."""

from __future__ import annotations

import asyncio

from halal_trader.core.ws_resilience import (
    BackoffPolicy,
    _BackoffState,
    heartbeat_guard,
    reconnect_loop,
)

# ── BackoffPolicy / state ─────────────────────────────────────


def test_backoff_grows_exponentially_then_caps():
    state = _BackoffState(BackoffPolicy(base_seconds=1, max_seconds=8, factor=2))
    delays = [state.next_delay() for _ in range(6)]
    assert delays[:4] == [1, 2, 4, 8]
    # After cap, stays at max.
    assert delays[4] == 8
    assert delays[5] == 8


def test_backoff_reset_returns_to_base():
    state = _BackoffState(BackoffPolicy(base_seconds=1, max_seconds=8, factor=2))
    for _ in range(3):
        state.next_delay()
    state.reset()
    assert state.next_delay() == 1


# ── reconnect_loop ────────────────────────────────────────────


async def test_reconnect_loop_exits_when_not_running():
    """is_running returning False after one pass should exit cleanly."""
    runs = {"n": 0}
    flag = {"keep": True}

    async def _connect():
        runs["n"] += 1
        flag["keep"] = False  # tell loop to stop after one pass

    await reconnect_loop(name="t", connect_and_run=_connect, is_running=lambda: flag["keep"])
    assert runs["n"] == 1


async def test_reconnect_loop_retries_on_exception():
    """Exception in connect should trigger backoff sleep, then retry."""
    attempts = {"n": 0}

    async def _connect():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("connection reset")
        # On the third attempt, exit cleanly.

    # Tiny backoff so the test runs in milliseconds.
    policy = BackoffPolicy(base_seconds=0.001, max_seconds=0.01, factor=2)
    await reconnect_loop(
        name="t",
        connect_and_run=_connect,
        is_running=lambda: attempts["n"] < 3,
        policy=policy,
    )
    assert attempts["n"] == 3


async def test_reconnect_loop_propagates_cancelled():
    """CancelledError must propagate so the caller can shut down cleanly."""

    async def _connect():
        await asyncio.sleep(10)

    task = asyncio.create_task(
        reconnect_loop(
            name="t",
            connect_and_run=_connect,
            is_running=lambda: True,
        )
    )
    await asyncio.sleep(0.01)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── heartbeat_guard ───────────────────────────────────────────


async def test_heartbeat_fires_on_stall():
    loop = asyncio.get_event_loop()
    last = loop.time() - 100  # ancient activity
    fired = {"n": 0}

    async def _on_stall():
        fired["n"] += 1

    await heartbeat_guard(
        last_activity=lambda: last,
        interval_seconds=0.01,
        timeout_seconds=0.05,
        on_stall=_on_stall,
    )
    assert fired["n"] == 1


async def test_heartbeat_does_not_fire_when_active():
    loop = asyncio.get_event_loop()

    fired = {"n": 0}

    async def _on_stall():
        fired["n"] += 1

    # Each tick keeps activity fresh; loop exits after a short window.
    iterations = {"k": 0}

    def _is_running() -> bool:
        iterations["k"] += 1
        return iterations["k"] < 4

    await heartbeat_guard(
        last_activity=lambda: loop.time(),
        interval_seconds=0.01,
        timeout_seconds=0.05,
        on_stall=_on_stall,
        is_running=_is_running,
    )
    assert fired["n"] == 0
