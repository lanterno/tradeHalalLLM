"""Durable event log — the replay source (INV-5).

The log is append-only. :meth:`EventLog.append` persists an event;
:meth:`EventLog.replay` streams events back in event-time (``ts``) order,
optionally filtered by time and type, so the belief store can warm itself on
restart and the learning loop can train off history.

This module ships the Protocol + an in-memory implementation (used by tests
and by the bus before the Postgres-backed log is wired in a later step). The
Postgres implementation lands with its Alembic migration; it satisfies the
same Protocol so the bus is agnostic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol

from halabot.platform.events import Event, EventType


class EventLog(Protocol):
    """Durable, append-only, replayable event store."""

    async def append(self, event: Event) -> None: ...

    def replay(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        types: set[EventType] | None = None,
        asset: str | None = None,
    ) -> AsyncIterator[Event]: ...


class InMemoryEventLog:
    """List-backed :class:`EventLog` for tests and pre-Postgres bring-up.

    Not durable across process restarts — that's the Postgres impl's job —
    but it satisfies the full contract (append + ordered, filtered replay) so
    the bus and belief bootstrap can be developed and tested against it.
    """

    def __init__(self) -> None:
        self._events: list[Event] = []

    async def append(self, event: Event) -> None:
        self._events.append(event)

    async def replay(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        types: set[EventType] | None = None,
        asset: str | None = None,
    ) -> AsyncIterator[Event]:
        # Replay is in event-time order regardless of append order, so an
        # out-of-order ingest still replays deterministically (INV-5).
        for event in sorted(self._events, key=lambda e: e.ts):
            if since is not None and event.ts < since:
                continue
            if until is not None and event.ts > until:
                continue
            if types is not None and event.type not in types:
                continue
            if asset is not None and event.asset != asset:
                continue
            yield event

    def __len__(self) -> int:
        return len(self._events)
