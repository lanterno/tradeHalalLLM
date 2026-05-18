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


async def test_crash_bot_task_cancels_other_supervised_tasks() -> None:
    """The bot's run() registers the monitor (CRASH_BOT), ws (CRASH_BOT),
    news/sentiment (RESTART), reconcile (RESTART), and the cycle loop.
    When the monitor crashes, every sibling — including the *cycle loop
    itself* — must be cancelled and the exception propagates.

    This is the Wave-C acceptance bar: a monitor crash never leaves the
    cycle loop running half-blind.
    """
    sibling_cancelled = asyncio.Event()

    async def monitor_crashes() -> None:
        # Simulate the monitor's SL/TP loop dying on a DB error.
        await asyncio.sleep(0.01)
        raise RuntimeError("monitor: DB connection lost")

    async def cycle_loop_like() -> None:
        # Stand-in for the bot's main cycle loop — would normally loop
        # forever. Cancellation should land here when the monitor dies.
        try:
            await asyncio.sleep(60)
        except BaseException:
            sibling_cancelled.set()
            raise

    with pytest.raises(BaseExceptionGroup) as exc_info:
        async with TaskSupervisor() as sup:
            sup.start("monitor", monitor_crashes, policy=RestartPolicy.CRASH_BOT)
            sup.start("cycle_loop", cycle_loop_like, policy=RestartPolicy.CRASH_BOT)
            await asyncio.sleep(0.5)

    flat = [e for e in exc_info.value.exceptions if isinstance(e, RuntimeError)]
    assert flat and "monitor: DB connection lost" in str(flat[0])
    assert sibling_cancelled.is_set(), "sibling cycle loop was NOT cancelled by monitor crash"


async def test_restart_policy_does_not_kill_bot_on_one_failure() -> None:
    """The bot wires news_reactor + sentiment_manager with RESTART so
    one upstream flake never kills the cycle. Confirm a single
    RESTART-policy failure leaves the supervisor running its siblings."""
    cycle_kept_running = False

    async def flaky_external_api() -> None:
        raise RuntimeError("CryptoPanic 503 — transient")

    async def faux_cycle() -> None:
        nonlocal cycle_kept_running
        await asyncio.sleep(0.2)
        cycle_kept_running = True

    import halal_trader.core.supervisor as sup_mod

    orig_initial = sup_mod._INITIAL_BACKOFF_S
    sup_mod._INITIAL_BACKOFF_S = 0.001
    try:
        async with TaskSupervisor() as sup:
            sup.start("news_reactor", flaky_external_api, policy=RestartPolicy.RESTART)
            sup.start("faux_cycle", faux_cycle, policy=RestartPolicy.CRASH_BOT)
            await asyncio.wait_for(asyncio.sleep(0.3), timeout=1.0)
            sup.cancel()
    finally:
        sup_mod._INITIAL_BACKOFF_S = orig_initial

    assert cycle_kept_running, "RESTART-policy failure killed the CRASH_BOT cycle task"
