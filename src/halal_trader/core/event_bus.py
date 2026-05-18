"""In-process async pub/sub event bus.

Used by the cycle / monitor / executor to publish structured events,
and by the dashboard's ``/ws/cycle`` WebSocket to stream them out.

Design notes
------------
* Topic strings use dot-separated namespaces (``cycle.start``,
  ``llm.call.complete``, ``executor.fill``). Subscribers can use
  exact matches or a single trailing wildcard
  (``"cycle.*"`` matches ``"cycle.start"`` and ``"cycle.stage.fetch"``).
* Each subscriber has its own bounded queue. If the consumer falls
  behind, we **drop the oldest events** (not block the publisher).
  Trading throughput must never wait for a slow WebSocket client.
* Only one publisher path: ``await bus.publish(topic, payload)``. The
  bus has no synchronous equivalent — it's an async-first primitive
  living next to the async cycle.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """One published event."""

    topic: str
    payload: dict[str, Any]
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


def _matches(pattern: str, topic: str) -> bool:
    """Glob match — single trailing ``*`` is a prefix wildcard."""
    if pattern == topic:
        return True
    if pattern.endswith(".*"):
        return topic.startswith(pattern[:-1]) or topic == pattern[:-2]
    if pattern == "*":
        return True
    return False


class _Subscriber:
    """One queue + the topic glob it cares about."""

    def __init__(self, pattern: str, max_queue: int) -> None:
        self.pattern = pattern
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue)
        self.dropped = 0

    def offer(self, event: Event) -> None:
        if not _matches(self.pattern, event.topic):
            return
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop oldest — slow consumers must never block the cycle.
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
            self.dropped += 1


class EventBus:
    """Async fan-out bus with bounded per-subscriber queues."""

    def __init__(self, *, default_queue_size: int = 256) -> None:
        self._subs: list[_Subscriber] = []
        self._default_qsize = default_queue_size
        self._lock = asyncio.Lock()

    async def publish(self, topic: str, payload: dict[str, Any] | None = None) -> None:
        from halal_trader.core.metrics import event_published

        event = Event(topic=topic, payload=payload or {})
        # Snapshot under the lock; ``offer`` is non-blocking so the
        # outer publish never awaits user-supplied work.
        async with self._lock:
            subs = list(self._subs)
        for sub in subs:
            sub.offer(event)
        event_published(topic)

    async def subscribe(
        self, pattern: str = "*", *, queue_size: int | None = None
    ) -> AsyncIterator[Event]:
        """Yield matching events forever; cancel the iteration to unsubscribe."""
        sub = _Subscriber(pattern, queue_size or self._default_qsize)
        async with self._lock:
            self._subs.append(sub)
        try:
            while True:
                yield await sub.queue.get()
        finally:
            async with self._lock:
                if sub in self._subs:
                    self._subs.remove(sub)
            if sub.dropped:
                logger.debug("event-bus subscriber %r dropped %d events", sub.pattern, sub.dropped)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)
