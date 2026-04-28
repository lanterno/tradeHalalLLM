"""DB-backed thesis tag store over ``thesis_tags``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel import select

from halal_trader.core.thesis import THESIS_TAGS
from halal_trader.db.models import ThesisTagRow

logger = logging.getLogger(__name__)


@dataclass
class DBThesisTagStore:
    engine: AsyncEngine

    @property
    def _sm(self) -> "async_sessionmaker[Any]":
        return async_sessionmaker(self.engine, expire_on_commit=False)

    async def get(self, trade_id: str) -> str | None:
        async with self._sm() as s:
            row = await s.get(ThesisTagRow, str(trade_id))
            return row.tag if row else None

    async def set(
        self,
        trade_id: str,
        tag: str,
        *,
        confidence: float = 1.0,
        reason: str = "",
        method: str = "heuristic",
    ) -> None:
        if tag not in THESIS_TAGS:
            tag = "unknown"
        async with self._sm() as s:
            existing = await s.get(ThesisTagRow, str(trade_id))
            if existing is None:
                s.add(
                    ThesisTagRow(
                        trade_id=str(trade_id),
                        tag=tag,
                        confidence=confidence,
                        reason=reason or None,
                        method=method,
                        set_at=datetime.now(UTC),
                    )
                )
            else:
                existing.tag = tag
                existing.confidence = confidence
                existing.reason = reason or None
                existing.method = method
                existing.set_at = datetime.now(UTC)
            await s.commit()

    async def all(self) -> dict[str, str]:
        async with self._sm() as s:
            rows = (await s.execute(select(ThesisTagRow))).scalars().all()
            return {r.trade_id: r.tag for r in rows}
