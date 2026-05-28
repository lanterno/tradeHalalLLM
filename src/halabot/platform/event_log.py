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
from datetime import UTC, datetime
from typing import Any, Protocol

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.platform.db import event_log as _event_log_table
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


class PgEventLog:
    """Postgres-backed :class:`EventLog` over the shared async engine.

    Stores into ``hb_event_log`` (own metadata, no Alembic-chain impact —
    see ``platform/db.py``). Conforms to the same Protocol as
    :class:`InMemoryEventLog`, so the bus is agnostic to which is wired.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def append(self, event: Event) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                sa.insert(_event_log_table).values(
                    id=event.id,
                    type=str(event.type),
                    asset=event.asset,
                    ts=event.ts,
                    ingested_at=datetime.now(UTC),
                    source=event.source,
                    payload=event.payload,
                    causation_id=event.causation_id,
                    correlation_id=event.correlation_id,
                    schema_version=event.schema_version,
                )
            )

    async def replay(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        types: set[EventType] | None = None,
        asset: str | None = None,
    ) -> AsyncIterator[Event]:
        t = _event_log_table
        stmt = sa.select(t).order_by(t.c.ts.asc())  # event-time order (INV-5)
        if since is not None:
            stmt = stmt.where(t.c.ts >= since)
        if until is not None:
            stmt = stmt.where(t.c.ts <= until)
        if types is not None:
            stmt = stmt.where(t.c.type.in_([str(x) for x in types]))
        if asset is not None:
            stmt = stmt.where(t.c.asset == asset)

        async with self._engine.connect() as conn:
            result = await conn.stream(stmt)
            async for row in result:
                yield _row_to_event(row)


def _row_to_event(row: sa.Row[Any]) -> Event:
    m = row._mapping
    return Event(
        id=m["id"],
        type=EventType(m["type"]),
        ts=m["ts"],
        source=m["source"],
        asset=m["asset"],
        payload=dict(m["payload"]),
        causation_id=m["causation_id"],
        correlation_id=m["correlation_id"],
        schema_version=m["schema_version"],
    )
