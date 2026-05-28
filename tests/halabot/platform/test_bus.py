"""InProcessEventBus — two-tier durability, handler isolation, pub/sub."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halabot.platform.bus import DurableAppendError, InProcessEventBus
from halabot.platform.clock import FakeClock
from halabot.platform.event_log import InMemoryEventLog
from halabot.platform.events import Event, EventType, new_event

CLOCK = FakeClock(datetime(2026, 5, 28, 12, 0, tzinfo=UTC))


class _FailingLog(InMemoryEventLog):
    """EventLog whose append always fails — simulates a DB outage."""

    async def append(self, event: Event) -> None:
        raise RuntimeError("db down")


# Minimal well-formed payloads so the bus's ingress validation (fail-closed)
# doesn't drop these mechanics-focused fixtures.
_PAYLOADS: dict[EventType, dict] = {
    EventType.OBSERVATION_BAR: {"o": 1.0, "h": 1.0, "low": 1.0, "c": 1.0},
    EventType.OBSERVATION_NEWS: {"headline": "x", "url": "http://x"},
    EventType.OBSERVATION_PRICE: {"price": 1.0},
    EventType.COMPLIANCE_VERDICT: {"status": "halal"},
}


def _ev(t: EventType, asset: str | None = None):
    return new_event(CLOCK, t, source="test", asset=asset, payload=_PAYLOADS.get(t, {}))


@pytest.mark.asyncio
async def test_publish_durable_appends_then_dispatches():
    log = InMemoryEventLog()
    bus = InProcessEventBus(log)
    seen: list[Event] = []
    bus.subscribe({EventType.OBSERVATION_BAR}, lambda e: _record(seen, e))

    await bus.publish(_ev(EventType.OBSERVATION_BAR, "NVDA"))
    assert len(log) == 1          # durably appended
    assert len(seen) == 1         # and dispatched


@pytest.mark.asyncio
async def test_subscriber_only_gets_subscribed_types():
    bus = InProcessEventBus(InMemoryEventLog())
    bars: list[Event] = []
    bus.subscribe({EventType.OBSERVATION_BAR}, lambda e: _record(bars, e))

    await bus.publish(_ev(EventType.OBSERVATION_BAR, "NVDA"))
    await bus.publish(_ev(EventType.OBSERVATION_NEWS, "NVDA"))
    assert [e.type for e in bars] == [EventType.OBSERVATION_BAR]


@pytest.mark.asyncio
async def test_durable_append_failure_blocks_dispatch_and_raises():
    """DB down + a DURABLE event → not dispatched, caller is told (no silent loss)."""
    bus = InProcessEventBus(_FailingLog())
    seen: list[Event] = []
    bus.subscribe({EventType.OBSERVATION_BAR}, lambda e: _record(seen, e))

    with pytest.raises(DurableAppendError):
        await bus.publish(_ev(EventType.OBSERVATION_BAR, "NVDA"))
    assert seen == []             # NOT dispatched — new work refused on DB outage


@pytest.mark.asyncio
async def test_control_event_dispatches_even_when_append_fails():
    """DB down + a CONTROL event (exit/halt/heartbeat) → STILL dispatched.

    This is the core resilience property: risk-reducing actions never block on
    the DB (Appendix E, fix R DB-down deadlock)."""
    bus = InProcessEventBus(_FailingLog())
    exits: list[Event] = []
    bus.subscribe({EventType.BELIEF_INVALIDATED}, lambda e: _record(exits, e))

    await bus.publish(_ev(EventType.BELIEF_INVALIDATED, "NVDA"))  # must NOT raise
    assert len(exits) == 1        # exit signal got through despite the dead DB


@pytest.mark.asyncio
async def test_handler_exception_is_isolated():
    """One handler raising must not stop the others or break the bus (INV-1)."""
    bus = InProcessEventBus(InMemoryEventLog())
    good: list[Event] = []

    async def boom(_e: Event) -> None:
        raise ValueError("handler bug")

    bus.subscribe({EventType.OBSERVATION_BAR}, boom)
    bus.subscribe({EventType.OBSERVATION_BAR}, lambda e: _record(good, e))

    await bus.publish(_ev(EventType.OBSERVATION_BAR, "NVDA"))  # must not raise
    assert len(good) == 1         # the healthy handler still ran


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    bus = InProcessEventBus(InMemoryEventLog())
    seen: list[Event] = []
    sub = bus.subscribe({EventType.OBSERVATION_BAR}, lambda e: _record(seen, e))

    await bus.publish(_ev(EventType.OBSERVATION_BAR, "NVDA"))
    sub.unsubscribe()
    await bus.publish(_ev(EventType.OBSERVATION_BAR, "NVDA"))
    assert len(seen) == 1         # only the pre-unsubscribe event


@pytest.mark.asyncio
async def test_handler_can_unsubscribe_during_dispatch():
    """A handler mutating subscriptions mid-dispatch must not corrupt iteration."""
    bus = InProcessEventBus(InMemoryEventLog())
    seen: list[Event] = []

    sub: list = []

    async def self_detach(e: Event) -> None:
        seen.append(e)
        sub[0].unsubscribe()

    sub.append(bus.subscribe({EventType.OBSERVATION_BAR}, self_detach))
    await bus.publish(_ev(EventType.OBSERVATION_BAR, "NVDA"))
    await bus.publish(_ev(EventType.OBSERVATION_BAR, "NVDA"))
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_replay_delegates_to_log():
    log = InMemoryEventLog()
    bus = InProcessEventBus(log)
    await bus.publish(_ev(EventType.OBSERVATION_BAR, "NVDA"))
    out = [e async for e in bus.replay(types={EventType.OBSERVATION_BAR})]
    assert len(out) == 1


async def _record(sink: list, e: Event) -> None:
    sink.append(e)
