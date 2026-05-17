"""Strategy-adjustment repository — operator + LLM-driven knob changes.

Wave D extraction. Records every change to tunable strategy parameters
(``max_position_pct``, ``stop_loss_pct``, …) with the reasoning the
self-improvement loop produced. The cycle reads the latest values on
boot to apply persisted overrides. Matching ``StrategyAdjustmentRepo``
Protocol in ``protocols.py``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import StrategyAdjustment


class StrategyAdjustmentRepoImpl:
    """Concrete implementation of :class:`StrategyAdjustmentRepo`."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def record_strategy_adjustment(
        self,
        parameter: str,
        old_value: float | None,
        new_value: float,
        reasoning: str | None = None,
    ) -> int:
        adj = StrategyAdjustment(
            parameter=parameter,
            old_value=old_value,
            new_value=new_value,
            reasoning=reasoning,
        )
        async with AsyncSession(self._engine) as session:
            session.add(adj)
            await session.commit()
            await session.refresh(adj)
            assert adj.id is not None
            return adj.id

    async def get_latest_strategy_adjustments(self) -> dict[str, float]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(StrategyAdjustment)
                .order_by(col(StrategyAdjustment.timestamp).desc())
                .limit(100)
            )
            results = await session.exec(statement)
            latest: dict[str, float] = {}
            for row in results.all():
                if row.parameter not in latest:
                    latest[row.parameter] = row.new_value
            return latest

    async def get_recent_adjustments(self, limit: int = 20) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(StrategyAdjustment)
                .order_by(col(StrategyAdjustment.timestamp).desc())
                .limit(limit)
            )
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]
