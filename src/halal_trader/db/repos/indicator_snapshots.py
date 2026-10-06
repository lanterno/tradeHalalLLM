"""Indicator-snapshot repository — features captured at trade entry.

Each buy snapshots the indicator vector that drove the decision. The
``label``/``return_pct`` columns served the ML retrainer, deleted on
2026-10-01; nothing labels or reads rows back now. Matching protocol in
``protocols.py``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine
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
