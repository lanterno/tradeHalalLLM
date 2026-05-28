"""The event bus — durable-append-then-dispatch, two-tier.

Subscribers register for a set of :class:`EventType`s; :meth:`publish` appends
the event to the durable log and fans it out to matching handlers.

**Two-tier durability (Appendix E, fix R DB-down):**

* **DURABLE** events (observations, belief versions, outcomes) must be appended
  before dispatch. If the append fails (DB down), the event is *not* dispatched
  and the failure propagates as :class:`DurableAppendError` — new work is
  refused, but nothing is silently lost.
* **CONTROL** events (exits, halts, the heartbeat) dispatch **best-effort even
  if the append fails** — a risk-reducing exit must never block on the DB. The
  append is still attempted (and logged on failure) for later reconciliation.

Handler isolation (INV-1): one handler raising never aborts the others or the
bus; the error is logged with its type (INV-4).

Ordering: events dispatch in publish order. Per-asset ordering is the caller's
responsibility (the belief worker serializes per-asset writes — Appendix F);
the bus does not reorder.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from halabot.platform.event_log import EventLog
from halabot.platform.events import Event, EventType

logger = logging.getLogger(__name__)

Handler = Callable[[Event], Awaitable[None]]


class DurableAppendError(RuntimeError):
    """A DURABLE event could not be persisted, so it was not dispatched."""


@dataclass
class Subscription:
    """A live registration; call :meth:`unsubscribe` to detach."""

    types: frozenset[EventType]
    handler: Handler
    _bus: "InProcessEventBus" = field(repr=False)
    active: bool = True

    def unsubscribe(self) -> None:
        if self.active:
            self._bus._remove(self)
            self.active = False


class EventBus(Protocol):
    """Structural interface that subscribers depend on.

    A future Redis/NATS-backed bus conforms to this without inheritance (the
    ports-and-adapters idiom used across this codebase), so swapping the
    transport never touches subscriber code.
    """

    async def publish(self, event: Event) -> None: ...
    def subscribe(self, types: set[EventType], handler: Handler) -> Subscription: ...
    def replay(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        types: set[EventType] | None = None,
        asset: str | None = None,
    ) -> AsyncIterator[Event]: ...


class InProcessEventBus:
    """Single-node async fan-out over a durable :class:`EventLog` (conforms to :class:`EventBus`).

    The default for a home bot. The ``EventLog`` swap (in-memory → Postgres)
    is invisible to subscribers, and the whole bus is swappable for a
    distributed broker behind the same surface if the engine ever goes
    multi-process.
    """

    def __init__(self, log: EventLog) -> None:
        self._log = log
        self._subs: list[Subscription] = []

    # ── pub/sub ──
    def subscribe(self, types: set[EventType], handler: Handler) -> Subscription:
        sub = Subscription(types=frozenset(types), handler=handler, _bus=self)
        self._subs.append(sub)
        return sub

    def _remove(self, sub: Subscription) -> None:
        try:
            self._subs.remove(sub)
        except ValueError:
            pass

    async def publish(self, event: Event) -> None:
        if event.is_control:
            # Best-effort durability: attempt the append, but dispatch
            # regardless so exits/halts/heartbeat flow during a DB outage.
            try:
                await self._log.append(event)
            except Exception as exc:  # noqa: BLE001 — never block a control event on the DB
                logger.error(
                    "event_log append failed for CONTROL event %s (%s) — "
                    "dispatching anyway, will reconcile later: %r",
                    event.type,
                    event.id,
                    exc,
                )
            await self._dispatch(event)
            return

        # DURABLE: the append must succeed before dispatch. On failure the
        # event is not dispatched and the caller learns about it.
        try:
            await self._log.append(event)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "event_log append failed for DURABLE event %s (%s) — NOT dispatched: %r",
                event.type,
                event.id,
                exc,
            )
            raise DurableAppendError(str(event.type)) from exc
        await self._dispatch(event)

    async def _dispatch(self, event: Event) -> None:
        # Snapshot the subscriber list so a handler that (un)subscribes during
        # dispatch doesn't mutate the list we're iterating.
        for sub in list(self._subs):
            if not sub.active or event.type not in sub.types:
                continue
            try:
                await sub.handler(event)
            except Exception as exc:  # noqa: BLE001 — handler isolation (INV-1)
                logger.error(
                    "subscriber %r failed handling %s (%s): %r",
                    getattr(sub.handler, "__qualname__", sub.handler),
                    event.type,
                    event.id,
                    exc,
                )

    # ── replay ──
    def replay(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        types: set[EventType] | None = None,
        asset: str | None = None,
    ) -> AsyncIterator[Event]:
        return self._log.replay(since=since, until=until, types=types, asset=asset)
