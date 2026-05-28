"""Supervisor — restart-on-error, cancellation, and the heartbeat loop."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from halabot.platform.clock import FakeClock
from halabot.platform.event_log import InMemoryEventLog
from halabot.platform.events import Event, EventType
from halabot.platform.supervisor import Supervisor, heartbeat_loop


async def _noop_sleep(_s: float) -> None:
    # Yield control without real time so backoff loops stay fast in tests.
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_crashing_task_is_restarted():
    calls = {"n": 0}

    async def flaky() -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        # Third run blocks forever (a healthy long-lived loop).
        await asyncio.Event().wait()

    sup = Supervisor(restart_backoff_s=0.0, sleep=_noop_sleep)
    sup.spawn("flaky", flaky)
    for _ in range(50):
        await asyncio.sleep(0)
        if calls["n"] >= 3:
            break
    assert calls["n"] == 3
    await sup.shutdown()


@pytest.mark.asyncio
async def test_no_restart_when_disabled():
    calls = {"n": 0}

    async def once() -> None:
        calls["n"] += 1

    sup = Supervisor(restart_backoff_s=0.0, sleep=_noop_sleep)
    sup.spawn("once", once, restart=False)
    for _ in range(20):
        await asyncio.sleep(0)
    assert calls["n"] == 1
    await sup.shutdown()


@pytest.mark.asyncio
async def test_shutdown_cancels_running_tasks():
    started = asyncio.Event()

    async def forever() -> None:
        started.set()
        await asyncio.Event().wait()

    sup = Supervisor(restart_backoff_s=0.0, sleep=_noop_sleep)
    sup.spawn("forever", forever)
    await started.wait()
    await sup.shutdown()  # must return (cancels cleanly, not treated as a crash)


@pytest.mark.asyncio
async def test_heartbeat_loop_emits_heartbeats():
    from halabot.platform.bus import InProcessEventBus

    bus = InProcessEventBus(InMemoryEventLog())
    seen: list[Event] = []
    bus.subscribe({EventType.SYSTEM_HEARTBEAT}, lambda e: _record(seen, e))
    clock = FakeClock(datetime(2026, 5, 28, tzinfo=UTC))

    task = asyncio.create_task(heartbeat_loop(bus, clock, 0.0, sleep=_noop_sleep))
    for _ in range(20):
        await asyncio.sleep(0)
        if len(seen) >= 2:
            break
    task.cancel()
    assert len(seen) >= 2
    assert all(e.type == EventType.SYSTEM_HEARTBEAT for e in seen)


async def _record(sink: list, e: Event) -> None:
    sink.append(e)
