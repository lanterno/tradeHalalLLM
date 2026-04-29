"""Tests for the in-process EventBus."""

from __future__ import annotations

import asyncio

from halal_trader.core.event_bus import EventBus


async def test_publish_and_subscribe_roundtrip() -> None:
    bus = EventBus()
    received: list[str] = []

    async def consumer() -> None:
        async for event in bus.subscribe("cycle.*"):
            received.append(event.topic)
            if len(received) == 2:
                break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)  # let the consumer install its sub
    await bus.publish("cycle.start")
    await bus.publish("cycle.complete")
    await asyncio.wait_for(task, timeout=1.0)
    assert received == ["cycle.start", "cycle.complete"]


async def test_pattern_filter_drops_non_matching_topics() -> None:
    bus = EventBus()
    received: list[str] = []

    async def consumer() -> None:
        async for event in bus.subscribe("llm.*"):
            received.append(event.topic)
            if len(received) == 1:
                break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    await bus.publish("cycle.start")  # ignored
    await bus.publish("llm.call.start")  # matches
    await asyncio.wait_for(task, timeout=1.0)
    assert received == ["llm.call.start"]


async def test_unsubscribe_via_aclose() -> None:
    bus = EventBus()
    sub = bus.subscribe()

    async def consumer() -> None:
        async for _ in sub:
            await sub.aclose()  # cleans up the subscriber slot
            return

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    await bus.publish("any.event")
    await asyncio.wait_for(task, timeout=1.0)
    assert bus.subscriber_count == 0


async def test_slow_consumer_doesnt_block_publisher() -> None:
    """A subscriber with a small queue must drop oldest, not block."""
    bus = EventBus(default_queue_size=2)

    async def slow_consumer() -> None:
        async for _ in bus.subscribe():
            await asyncio.sleep(10)  # never drains

    task = asyncio.create_task(slow_consumer())
    await asyncio.sleep(0)
    # Publish 100 events; with queue_size=2 most are dropped but
    # publish itself never blocks beyond the bounded loop.
    for i in range(100):
        await bus.publish(f"e.{i}")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_subscriber_unsubscribes_on_iterator_close() -> None:
    bus = EventBus()
    sub = bus.subscribe()

    async def consumer() -> None:
        async for _ in sub:
            await sub.aclose()
            return

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    await bus.publish("ping")
    await asyncio.wait_for(task, timeout=1.0)
    assert bus.subscriber_count == 0
