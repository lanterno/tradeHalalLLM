"""Persisted perception dedup (REARCHITECTURE §5, INV-2 idempotency).

A :class:`PollingSource` suppresses re-emitting a seen item via a key
(``asset:url`` for news). In memory that set resets on restart, so a restart
re-emits the last day of headlines — and because each re-fetch gets a fresh
``Event.id``, ``merge``'s event_id dedup wouldn't catch them, double-counting the
evidence. Persisting the seen keys closes that across restarts.

``PgDedupStore`` keys server-time so callers need no clock; ``load`` filters by a
retention window (bounding the working set) and ``prune`` expires old rows.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql.elements import TextClause

from halabot.platform.db import perception_seen


class DedupStore(Protocol):
    async def load(self, namespace: str) -> set[str]: ...
    async def add(self, namespace: str, keys: Iterable[str]) -> None: ...


class InMemoryDedupStore:
    """Process-local dedup (tests / no-DB). Does NOT survive a restart."""

    def __init__(self) -> None:
        self._seen: dict[str, set[str]] = {}

    async def load(self, namespace: str) -> set[str]:
        return set(self._seen.get(namespace, set()))

    async def add(self, namespace: str, keys: Iterable[str]) -> None:
        self._seen.setdefault(namespace, set()).update(keys)


class PgDedupStore:
    """Postgres-backed dedup over ``hb_perception_seen`` (shared engine)."""

    def __init__(self, engine: AsyncEngine, *, retention_days: float = 3.0) -> None:
        self._engine = engine
        self._retention_s = retention_days * 86400.0

    def _cutoff(self) -> TextClause:
        # now() - make_interval(secs => :s): server-side cutoff, float bind (no
        # timedelta-as-interval encoding ambiguity over the raw driver).
        return text("now() - make_interval(secs => :s)").bindparams(s=self._retention_s)

    async def load(self, namespace: str) -> set[str]:
        """Keys seen within the retention window (older ones are pruned first)."""
        await self.prune()
        async with self._engine.connect() as conn:
            rows = await conn.execute(
                select(perception_seen.c.key).where(
                    perception_seen.c.namespace == namespace,
                    perception_seen.c.seen_at >= self._cutoff(),
                )
            )
            return {r[0] for r in rows}

    async def add(self, namespace: str, keys: Iterable[str]) -> None:
        rows = [{"namespace": namespace, "key": k, "seen_at": func.now()} for k in set(keys)]
        if not rows:
            return
        # ON CONFLICT refreshes seen_at so a still-recurring item stays in-window.
        stmt = pg_insert(perception_seen).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_hb_perception_seen",
            set_={"seen_at": func.now()},
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def prune(self) -> int:
        """Delete rows older than the retention window. Returns rows removed."""
        async with self._engine.begin() as conn:
            result = await conn.execute(
                delete(perception_seen).where(perception_seen.c.seen_at < self._cutoff())
            )
            return result.rowcount or 0
