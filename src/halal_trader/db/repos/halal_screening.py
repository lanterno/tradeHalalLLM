"""Halal screening audit trail repository.

Wave D extraction. One row per screening decision; each ``Trade`` /
``CryptoTrade`` carries the row id via ``halal_screening_id`` so every
fill is provably linked to the compliance call that gated it. The
matching ``HalalScreeningRepo`` Protocol lives in ``protocols.py``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import HalalScreening


class HalalScreeningRepoImpl:
    """Concrete implementation of :class:`HalalScreeningRepo`."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def record_halal_screening(
        self,
        *,
        symbol: str,
        asset_class: str,
        source: str,
        decision: str,
        criteria: dict[str, Any] | None = None,
        cache_hit: bool = False,
    ) -> int:
        """Persist a screening decision and return its row id."""
        row = HalalScreening(
            symbol=symbol,
            asset_class=asset_class,
            source=source,
            decision=decision,
            criteria=criteria,
            cache_hit=cache_hit,
        )
        async with AsyncSession(self._engine) as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            assert row.id is not None
            return row.id

    async def get_halal_screening(self, screening_id: int) -> dict[str, Any] | None:
        async with AsyncSession(self._engine) as session:
            row = await session.get(HalalScreening, screening_id)
            if row is None:
                return None
            return row.model_dump()
