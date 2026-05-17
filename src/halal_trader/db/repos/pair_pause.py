"""Per-pair operator pause repository.

Wave D extraction. The pair-pause table is the operator's runtime kill
switch for individual symbols — listed pairs skip the cycle's entry
path. The matching ``PairPauseRepo`` Protocol lives in ``protocols.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import PairPause


class PairPauseRepoImpl:
    """Concrete implementation of :class:`PairPauseRepo`."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def pause_pair(
        self, pair: str, *, set_by: str | None = None, reason: str | None = None
    ) -> None:
        """Insert (or update) a pause row for ``pair``."""
        async with AsyncSession(self._engine) as session:
            row = await session.get(PairPause, pair.upper())
            if row is None:
                row = PairPause(pair=pair.upper(), set_by=set_by, reason=reason)
            else:
                row.set_by = set_by
                row.reason = reason
                row.set_at = datetime.now(UTC)
            session.add(row)
            await session.commit()

    async def resume_pair(self, pair: str) -> bool:
        """Delete the pause row. Returns True if a row was actually removed."""
        async with AsyncSession(self._engine) as session:
            row = await session.get(PairPause, pair.upper())
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def get_paused_pairs(self) -> set[str]:
        """The set of currently paused pair symbols (uppercased)."""
        async with AsyncSession(self._engine) as session:
            results = await session.exec(select(PairPause))
            return {r.pair for r in results.all()}

    async def list_pair_pauses(self) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            results = await session.exec(select(PairPause))
            return [r.model_dump() for r in results.all()]
