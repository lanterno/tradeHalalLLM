"""Stock halal-screening cache repository.

Wave D extraction. Mirrors :class:`HalalCacheRepoImpl` (crypto) over the
``halal_cache`` table — symbol → compliance verdict from Zoya/AAOIFI.
Matching ``StockHalalCacheRepo`` Protocol in ``protocols.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import HalalCache


class StockHalalCacheRepoImpl:
    """Concrete implementation of :class:`StockHalalCacheRepo`."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def cache_halal_status(
        self, symbol: str, compliance: str, detail: str | None = None
    ) -> None:
        async with AsyncSession(self._engine) as session:
            statement = select(HalalCache).where(HalalCache.symbol == symbol)
            result = await session.exec(statement)
            existing = result.first()
            if existing is not None:
                existing.compliance = compliance
                existing.detail = detail
                existing.updated_at = datetime.now(UTC)
                session.add(existing)
            else:
                session.add(HalalCache(symbol=symbol, compliance=compliance, detail=detail))
            await session.commit()

    async def get_halal_status(self, symbol: str) -> str | None:
        async with AsyncSession(self._engine) as session:
            statement = select(HalalCache).where(HalalCache.symbol == symbol)
            result = await session.exec(statement)
            row = result.first()
            return row.compliance if row else None

    async def get_halal_symbols(self) -> list[str]:
        async with AsyncSession(self._engine) as session:
            statement = select(HalalCache).where(HalalCache.compliance == "halal")
            results = await session.exec(statement)
            return [r.symbol for r in results.all()]

    async def is_cache_fresh(self, max_age_hours: int = 24) -> bool:
        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        async with AsyncSession(self._engine) as session:
            statement = select(HalalCache).where(HalalCache.updated_at > cutoff)
            results = await session.exec(statement)
            return len(results.all()) > 0
