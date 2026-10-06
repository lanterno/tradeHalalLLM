"""Belief persistence — versioned (INV-5).

Each :meth:`BeliefStore.put` writes a NEW version of an asset's belief and
returns the new version number; old versions are retained so the system can
reconstruct history and link positions to the exact belief that opened them
(INV-8). The Postgres-backed implementation lands with the batched Alembic
migration (alongside ``event_log``); this in-memory impl satisfies the full
contract for tests and pre-persistence bring-up.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from typing import Protocol

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.belief.schema import BeliefState
from halabot.belief.serde import belief_from_row, belief_to_values
from halabot.platform.db import belief_state as _belief_table


class BeliefStore(Protocol):
    async def get(self, asset: str) -> BeliefState | None:
        """Latest version for ``asset``, or None if never persisted."""
        ...

    async def get_version(self, asset: str, version: int) -> BeliefState | None: ...

    async def put(self, belief: BeliefState) -> int:
        """Persist a new version of ``belief``; return the new version number."""
        ...

    async def all_active(self) -> list[BeliefState]:
        """Latest version of every known asset."""
        ...


class InMemoryBeliefStore:
    """Dict-of-version-lists store for tests and pre-Postgres bring-up.

    ``put`` deep-copies and stamps an incremented version, so the returned/
    stored belief is decoupled from the caller's mutable instance (a later
    in-place mutation of the caller's object can't corrupt a persisted version).
    """

    def __init__(self) -> None:
        self._versions: dict[str, list[BeliefState]] = {}

    async def get(self, asset: str) -> BeliefState | None:
        history = self._versions.get(asset)
        return deepcopy(history[-1]) if history else None

    async def get_version(self, asset: str, version: int) -> BeliefState | None:
        for b in self._versions.get(asset, []):
            if b.version == version:
                return deepcopy(b)
        return None

    async def put(self, belief: BeliefState) -> int:
        history = self._versions.setdefault(belief.asset, [])
        new_version = (history[-1].version + 1) if history else 1
        stored = replace(deepcopy(belief), version=new_version)
        history.append(stored)
        return new_version

    async def all_active(self) -> list[BeliefState]:
        return [deepcopy(h[-1]) for h in self._versions.values() if h]


class PgBeliefStore:
    """Postgres-backed versioned :class:`BeliefStore` over the shared engine.

    Each :meth:`put` reads the asset's current max version and inserts max+1.
    The per-asset single-writer model (one belief worker per asset, REARCHITECTURE
    Appendix F) means there is never a concurrent put for the same asset, so the
    read-then-insert needs no extra locking.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def get(self, asset: str) -> BeliefState | None:
        t = _belief_table
        stmt = sa.select(t).where(t.c.asset == asset).order_by(t.c.version.desc()).limit(1)
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).first()
        return belief_from_row(dict(row._mapping)) if row is not None else None

    async def get_version(self, asset: str, version: int) -> BeliefState | None:
        t = _belief_table
        stmt = sa.select(t).where(t.c.asset == asset, t.c.version == version)
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).first()
        return belief_from_row(dict(row._mapping)) if row is not None else None

    async def put(self, belief: BeliefState) -> int:
        """A new version when the belief changed; otherwise the latest version,
        re-stamped.

        Until 2026-10 every update wrote a version, changed or not: about
        60,000 rows a day across 21 assets, nearly all identical to the one
        before (2.4 GB in a fortnight). A version now marks a change of
        content; an unchanged belief only moves the latest row's
        ``last_updated``/``updated_at`` forward, so freshness reads the same
        and every version a position links to (INV-8) is still a distinct
        state.
        """
        t = _belief_table
        values = belief_to_values(belief)
        async with self._engine.begin() as conn:
            latest = (
                await conn.execute(
                    sa.select(t)
                    .where(t.c.asset == belief.asset)
                    .order_by(t.c.version.desc())
                    .limit(1)
                )
            ).first()
            now = datetime.now(UTC)
            if latest is not None and _same_content(dict(latest._mapping), values):
                await conn.execute(
                    sa.update(t)
                    .where(t.c.id == latest.id)
                    .values(last_updated=values["last_updated"], updated_at=now)
                )
                return int(latest.version)
            new_version = (latest.version if latest is not None else 0) + 1
            values["version"] = new_version
            values["updated_at"] = now
            await conn.execute(sa.insert(t).values(**values))
        return new_version

    async def all_active(self) -> list[BeliefState]:
        t = _belief_table
        # Latest version per asset via DISTINCT ON (asset) ordered by version desc.
        stmt = sa.select(t).order_by(t.c.asset, t.c.version.desc()).distinct(t.c.asset)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return [belief_from_row(dict(r._mapping)) for r in rows]


# Timestamps say when, not what: two beliefs differing only in these are one state.
_NOT_CONTENT = frozenset({"id", "version", "updated_at", "last_updated"})


def _same_content(row: dict[str, object], values: dict[str, object]) -> bool:
    """Does the stored row hold the same belief as ``values``? Both sides are put
    through the row round-trip, so JSONB and Python forms compare alike."""
    stored = belief_to_values(belief_from_row(row))
    return all(stored[k] == values[k] for k in values if k not in _NOT_CONTENT)
