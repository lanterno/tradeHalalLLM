"""Daily "stock of the day" recommendation repository (advisory).

Each row is one LLM-picked most-promising halal stock. The latest row is
the active pick; regenerating appends a row so the history is preserved.
Advisory only — nothing here is ever wired into the execution path. The
matching ``DailyRecommendationRepo`` Protocol lives in ``protocols.py``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import DailyRecommendation


class DailyRecommendationRepoImpl:
    """Concrete implementation of :class:`DailyRecommendationRepo`."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def save_recommendation(self, rec: dict[str, Any]) -> int:
        row = DailyRecommendation(**rec)
        async with AsyncSession(self._engine) as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            assert row.id is not None
            return row.id

    async def get_latest_recommendation(self) -> dict[str, Any] | None:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(DailyRecommendation)
                .order_by(col(DailyRecommendation.id).desc())
                .limit(1)
            )
            results = await session.exec(statement)
            row = results.first()
            return row.model_dump() if row is not None else None

    async def get_recent_recommendations(self, limit: int = 30) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(DailyRecommendation)
                .order_by(col(DailyRecommendation.id).desc())
                .limit(limit)
            )
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]
