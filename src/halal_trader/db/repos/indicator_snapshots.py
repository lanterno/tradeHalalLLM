"""Indicator-snapshot repository — features captured at trade entry.

Wave D extraction. Each buy snapshots the indicator vector that drove
the decision; the position monitor labels the row with realized return
on close. The retraining loop reads ``get_labeled_snapshots`` to refit
the anomaly detector / signal classifier. Matching protocol in
``protocols.py``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import IndicatorSnapshot


class IndicatorSnapshotRepoImpl:
    """Concrete implementation of :class:`IndicatorSnapshotRepo`."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def record_indicator_snapshot(
        self,
        *,
        trade_id: int,
        pair: str,
        indicators: dict[str, float],
    ) -> int:
        snap = IndicatorSnapshot(
            trade_id=trade_id,
            pair=pair,
            rsi_14=indicators.get("rsi_14"),
            macd_histogram=indicators.get("macd_histogram"),
            volume_ratio=indicators.get("volume_ratio"),
            atr_14=indicators.get("atr_14"),
            bb_position=indicators.get("bb_position"),
            price_change_5m=indicators.get("price_change_5m"),
            ema_9=indicators.get("ema_9"),
            ema_21=indicators.get("ema_21"),
            vwap=indicators.get("vwap"),
        )
        async with AsyncSession(self._engine) as session:
            session.add(snap)
            await session.commit()
            await session.refresh(snap)
            assert snap.id is not None
            return snap.id

    async def label_indicator_snapshot(
        self, trade_id: int, label: int, return_pct: float
    ) -> None:
        async with AsyncSession(self._engine) as session:
            statement = select(IndicatorSnapshot).where(IndicatorSnapshot.trade_id == trade_id)
            result = await session.exec(statement)
            snap = result.first()
            if snap:
                snap.label = label
                snap.return_pct = return_pct
                session.add(snap)
                await session.commit()

    async def get_labeled_snapshots(self, min_samples: int = 50) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(IndicatorSnapshot)
                .where(col(IndicatorSnapshot.label).is_not(None))
                .order_by(col(IndicatorSnapshot.timestamp).desc())
                .limit(5000)
            )
            results = await session.exec(statement)
            rows = results.all()
            if len(rows) < min_samples:
                return []
            return [r.model_dump() for r in rows]
