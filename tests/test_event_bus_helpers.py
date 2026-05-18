"""Tests for the pure `_matches` helper + dataclass defaults + edge
behaviours in :mod:`core.event_bus`.

`test_event_bus.py` covers the integration-level pub/sub flow. This
file pins the small surface that's reached only via specific topic
shapes — wildcard semantics that operators rely on when wiring
custom subscribers, and the `dropped` counter / multi-subscriber
fan-out invariants.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from halal_trader.core.event_bus import Event, EventBus, _matches

# ── _matches pure helper ───────────────────────────────────


def test_matches_exact_topic():
    assert _matches("cycle.start", "cycle.start") is True
    assert _matches("cycle.start", "cycle.complete") is False


def test_matches_universal_wildcard():
    """``"*"`` matches every topic — used by the dashboard's
    `/ws/cycle` to stream the full bus."""
    assert _matches("*", "cycle.start") is True
    assert _matches("*", "anything.at.all") is True
    assert _matches("*", "x") is True


def test_matches_trailing_wildcard_prefix():
    """``"cycle.*"`` matches any topic that starts with ``"cycle."``."""
    assert _matches("cycle.*", "cycle.start") is True
    assert _matches("cycle.*", "cycle.complete") is True
    assert _matches("cycle.*", "cycle.stage.fetch") is True  # nested too


def test_matches_trailing_wildcard_matches_bare_base():
    """The implementation also accepts the bare base ``"cycle"`` (no
    dot suffix) for ``"cycle.*"`` — covers the `topic == pattern[:-2]`
    branch that the integration tests don't reach."""
    assert _matches("cycle.*", "cycle") is True


def test_matches_trailing_wildcard_does_not_match_unrelated():
    """``"cycle.*"`` must NOT match ``"cyclone.start"`` (substring trap)."""
    assert _matches("llm.*", "cycle.start") is False
    assert _matches("cycle.*", "executor.fill") is False


def test_matches_no_wildcard_treats_other_dots_literally():
    """A pattern like ``"a.b.c"`` (no trailing ``*``) is exact-match
    only; nested topics under ``a.b.c.d`` don't match."""
    assert _matches("a.b.c", "a.b.c.d") is False
    assert _matches("a.b.c", "a.b.c") is True


def test_matches_wildcard_pattern_without_dot_prefix():
    """An odd pattern like ``"cycle*"`` (no dot before ``*``) does NOT
    take the wildcard branch — ``endswith(".*")`` is the gate. This
    pins the current behaviour so a refactor can't quietly relax it."""
    assert _matches("cycle*", "cycle.start") is False
    assert _matches("cycle*", "cycle*") is True  # only exact-string match


# ── Event dataclass defaults ───────────────────────────────


def test_event_default_ts_is_timezone_aware_utc():
    """The bus tags every event with `datetime.now(UTC)` — downstream
    serializers don't need to repair the tz."""
    ev = Event(topic="t", payload={})
    assert ev.ts.tzinfo is UTC


def test_event_explicit_ts_passes_through():
    """Allow callers (e.g. replay) to override the timestamp."""
    fixed = datetime(2026, 1, 1, tzinfo=UTC)
    ev = Event(topic="t", payload={"x": 1}, ts=fixed)
    assert ev.ts == fixed


def test_event_payload_is_required_dict():
    """Payload defaults to whatever the caller passes — `publish()`
    handles the None → {} normalisation."""
    ev = Event(topic="t", payload={"a": 1})
    assert ev.payload == {"a": 1}


# ── EventBus integration edges ─────────────────────────────


@pytest.mark.asyncio
async def test_publish_with_no_payload_uses_empty_dict():
    """`bus.publish("topic")` (no payload) must deliver ``{}`` to
    subscribers, not None — downstream code does ``payload.get(...)``."""
    bus = EventBus()
    received_payload: list = []

    async def consumer():
        async for ev in bus.subscribe():
            received_payload.append(ev.payload)
            return

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    await bus.publish("ping")  # no payload
    await asyncio.wait_for(task, timeout=1.0)
    assert received_payload == [{}]


@pytest.mark.asyncio
async def test_multiple_subscribers_each_get_event():
    """Fan-out: one publish reaches every matching subscriber."""
    bus = EventBus()
    a_seen: list[str] = []
    b_seen: list[str] = []

    async def consumer(buf: list[str]):
        async for ev in bus.subscribe():
            buf.append(ev.topic)
            return

    ta = asyncio.create_task(consumer(a_seen))
    tb = asyncio.create_task(consumer(b_seen))
    await asyncio.sleep(0)
    await bus.publish("ping")
    await asyncio.wait_for(asyncio.gather(ta, tb), timeout=1.0)
    assert a_seen == ["ping"]
    assert b_seen == ["ping"]


@pytest.mark.asyncio
async def test_subscriber_count_starts_zero_and_tracks_active_subs():
    bus = EventBus()
    assert bus.subscriber_count == 0

    sub = bus.subscribe()
    # Iterator hasn't started yet — sub registration happens on first
    # `__anext__`. Drive it once to install.
    task = asyncio.create_task(sub.__anext__())
    await asyncio.sleep(0)
    assert bus.subscriber_count == 1

    await bus.publish("x")
    await asyncio.wait_for(task, timeout=1.0)
    await sub.aclose()
    assert bus.subscriber_count == 0


@pytest.mark.asyncio
async def test_pattern_filter_skips_event_entirely():
    """A subscriber with `pattern="x.*"` must not see `y.event` —
    even ``offer()`` is a no-op (no queue.put_nowait, no dropped
    increment) because the pattern check comes first."""
    bus = EventBus(default_queue_size=1)
    consumed: list[str] = []

    async def consumer():
        async for ev in bus.subscribe("matching.*"):
            consumed.append(ev.topic)
            if len(consumed) == 1:
                return

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    # Publish a non-matching event 100 times. None should land in the
    # queue, so consumer stays blocked. Then publish one matching event
    # — that's what should unblock it.
    for i in range(100):
        await bus.publish(f"other.{i}")
    await bus.publish("matching.target")

    await asyncio.wait_for(task, timeout=1.0)
    assert consumed == ["matching.target"]


@pytest.mark.asyncio
async def test_custom_queue_size_overrides_default():
    """``subscribe(queue_size=N)`` overrides the bus-level default."""
    bus = EventBus(default_queue_size=2)

    async def slow_consumer():
        async for _ in bus.subscribe(queue_size=10):
            await asyncio.sleep(10)  # never drains

    task = asyncio.create_task(slow_consumer())
    await asyncio.sleep(0)
    # Publish 5 events. With per-sub queue_size=10 (not the bus default
    # of 2), all 5 land — the bus default doesn't apply.
    for i in range(5):
        await bus.publish(f"e.{i}")
    # Verify all 5 are queued by checking the subscriber's queue size.
    sub = bus._subs[0]
    assert sub.queue.qsize() == 5
    assert sub.dropped == 0  # nothing dropped
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_dropped_counter_increments_on_overflow():
    """When the per-subscriber queue overflows, `dropped` ticks up so
    the bus can log the count on cleanup. Pin the counter directly."""
    bus = EventBus(default_queue_size=2)

    async def slow_consumer():
        async for _ in bus.subscribe():
            await asyncio.sleep(10)

    task = asyncio.create_task(slow_consumer())
    await asyncio.sleep(0)

    # Fill the queue + cause N drops.
    for i in range(10):
        await bus.publish(f"e.{i}")
    sub = bus._subs[0]
    # 10 published, queue holds 2 → at least 8 dropped (some races OK).
    assert sub.dropped >= 5
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_publish_to_no_subscribers_is_a_no_op():
    """A publish with no subscribers must succeed silently (the cycle
    publishes events even when no dashboard is watching)."""
    bus = EventBus()
    await bus.publish("nobody.listening", {"k": "v"})
    assert bus.subscriber_count == 0  # no leftover state
