"""Tests for the structured-concurrency TaskSupervisor."""

from __future__ import annotations

import asyncio

import pytest

from halal_trader.core.supervisor import RestartPolicy, TaskSupervisor


async def test_explicit_cancel_stops_supervised_tasks() -> None:
    """sup.cancel() stops every supervised task so the scope exits."""
    cancelled = asyncio.Event()

    async def long_running() -> None:
        try:
            await asyncio.sleep(60)
        except BaseException:
            cancelled.set()
            raise

    async with TaskSupervisor() as sup:
        sup.start("long", long_running, policy=RestartPolicy.RESTART)
        await asyncio.sleep(0.01)
        sup.cancel()
    assert cancelled.is_set()


async def test_crash_bot_propagates_exception() -> None:
    """A CRASH_BOT task raising bubbles out of the scope."""

    async def boom() -> None:
        raise RuntimeError("boom")

    with pytest.raises(BaseExceptionGroup) as exc_info:
        async with TaskSupervisor() as sup:
            sup.start("boom", boom, policy=RestartPolicy.CRASH_BOT)
            await asyncio.sleep(0.01)

    flat = [e for e in exc_info.value.exceptions if isinstance(e, RuntimeError)]
    assert flat and "boom" in str(flat[0])


async def test_restart_policy_runs_after_failure() -> None:
    """RESTART policy retries with backoff (truncated for the test)."""
    runs = 0
    succeeded = asyncio.Event()

    async def flaky() -> None:
        nonlocal runs
        runs += 1
        if runs < 3:
            raise RuntimeError(f"attempt {runs} fails")
        succeeded.set()

    # We patch the backoff to ~0 so the test runs in subseconds.
    import halal_trader.core.supervisor as sup_mod

    orig_initial = sup_mod._INITIAL_BACKOFF_S
    sup_mod._INITIAL_BACKOFF_S = 0.001
    try:
        async with TaskSupervisor() as sup:
            sup.start("flaky", flaky, policy=RestartPolicy.RESTART)
            await asyncio.wait_for(succeeded.wait(), timeout=2.0)
    finally:
        sup_mod._INITIAL_BACKOFF_S = orig_initial
    assert runs >= 3


async def test_ignore_policy_swallows_failure_and_exits() -> None:
    """IGNORE — log it, exit the task, don't crash the bot."""
    started = asyncio.Event()

    async def short_lived() -> None:
        started.set()
        raise RuntimeError("ignored")

    async with TaskSupervisor() as sup:
        sup.start("ignored", short_lived, policy=RestartPolicy.IGNORE)
        await asyncio.wait_for(started.wait(), timeout=1.0)
        await asyncio.sleep(0.05)
