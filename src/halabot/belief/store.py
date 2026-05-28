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
from typing import Protocol

from halabot.belief.schema import BeliefState


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
