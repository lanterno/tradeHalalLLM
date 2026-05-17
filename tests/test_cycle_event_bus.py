"""Tests for :class:`BaseCycleService`'s event-bus publishing.

`test_observability.py` covers the cycle_id ContextVar threading;
`test_halt.py` covers the kill-switch gate. This file pins the
explicit `_publish_event` calls — what the dashboard's `/ws/cycle`
WebSocket actually sees from each cycle.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.cycle import BaseCycleService


class _StubCycle(BaseCycleService):
    """Bare cycle whose hooks all return cleanly."""

    def __init__(self, *, should_halt: bool = False, raise_on_impl: bool = False):
        super().__init__()
        # NB: don't shadow the inherited `_should_halt` method name —
        # store under a distinct attribute and read it from the override.
        self._halt_flag = should_halt
        self._raise_on_impl = raise_on_impl
        self.impl_calls = 0

    async def _pre_cycle_checks(self) -> bool:
        return True

    async def _should_halt(self) -> bool:
        return self._halt_flag

    async def _run_cycle_impl(self) -> None:
        self.impl_calls += 1
        if self._raise_on_impl:
            raise RuntimeError("boom")


class _SkippingCycle(_StubCycle):
    async def _pre_cycle_checks(self) -> bool:
        return False


# ── happy path ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_cycle_publishes_start_then_complete():
    bus = MagicMock()
    bus.publish = AsyncMock()
    cycle = _StubCycle()
    cycle._bus = bus

    await cycle.run_cycle()

    topics = [call.args[0] for call in bus.publish.await_args_list]
    assert topics == ["cycle.start", "cycle.complete"]


@pytest.mark.asyncio
async def test_complete_event_carries_elapsed_ms():
    bus = MagicMock()
    bus.publish = AsyncMock()
    cycle = _StubCycle()
    cycle._bus = bus

    await cycle.run_cycle()

    complete_payload = bus.publish.await_args_list[1].args[1]
    assert "elapsed_ms" in complete_payload
    assert complete_payload["elapsed_ms"] >= 0


# ── halt path ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_halt_skips_impl_and_no_complete_event():
    """When `_should_halt` is True, `_run_cycle_impl` mustn't run AND
    the cycle.complete event must not fire (only cycle.start)."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    cycle = _StubCycle(should_halt=True)
    cycle._bus = bus

    await cycle.run_cycle()

    assert cycle.impl_calls == 0
    topics = [call.args[0] for call in bus.publish.await_args_list]
    assert topics == ["cycle.start"]  # no cycle.complete


# ── pre-cycle skip ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_pre_cycle_check_false_skips_impl_no_complete_event():
    bus = MagicMock()
    bus.publish = AsyncMock()
    cycle = _SkippingCycle()
    cycle._bus = bus

    await cycle.run_cycle()

    assert cycle.impl_calls == 0
    topics = [call.args[0] for call in bus.publish.await_args_list]
    assert topics == ["cycle.start"]


# ── failure path ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_failure_publishes_failed_event_with_error_repr():
    """An exception in `_run_cycle_impl` must publish cycle.failed
    (not cycle.complete) with the error's repr in the payload."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    cycle = _StubCycle(raise_on_impl=True)
    cycle._bus = bus

    await cycle.run_cycle()  # exception swallowed by template

    topics = [call.args[0] for call in bus.publish.await_args_list]
    assert "cycle.failed" in topics
    assert "cycle.complete" not in topics
    failed_payload = bus.publish.await_args_list[-1].args[1]
    assert "boom" in failed_payload["error"]


@pytest.mark.asyncio
async def test_failure_invokes_alert_sink():
    """When an `AlertSink` is wired, a failure also fires `notify`
    so the operator sees a Telegram alert."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    alerts = MagicMock()
    alerts.notify = AsyncMock()
    cycle = _StubCycle(raise_on_impl=True)
    cycle._bus = bus
    cycle._alerts = alerts

    await cycle.run_cycle()

    alerts.notify.assert_awaited_once()
    args = alerts.notify.await_args.args
    assert args[0] == "cycle.failed"
    assert "boom" in args[1]


# ── no-bus tolerant ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_cycle_works_without_a_bus():
    """`_publish_event` is a no-op when no bus is wired — the cycle
    must still complete cleanly (used in pure-bot mode without the
    dashboard)."""
    cycle = _StubCycle()
    # cycle._bus stays None
    await cycle.run_cycle()  # must not raise
    assert cycle.impl_calls == 1


@pytest.mark.asyncio
async def test_publish_event_swallows_bus_failure():
    """A misbehaving bus must not crash the cycle — events are
    best-effort observability, not load-bearing."""
    bus = MagicMock()
    bus.publish = AsyncMock(side_effect=RuntimeError("bus down"))
    cycle = _StubCycle()
    cycle._bus = bus

    await cycle.run_cycle()  # must not raise
    assert cycle.impl_calls == 1
