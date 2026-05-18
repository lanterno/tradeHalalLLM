"""Crypto halal-screening cache repository.

Wave D extraction. The cache stores the latest screening verdict per
symbol so the cycle's halal-symbol fetch is a single indexed read.
The cache is refreshed daily by ``halal-trader crypto screen``.
Matching ``HalalCacheRepo`` Protocol in ``protocols.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import CryptoHalalCache


class HalalCacheRepoImpl:
    """Concrete implementation of :class:`HalalCacheRepo` for crypto."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def cache_crypto_halal_status(
        self,
        symbol: str,
        compliance: str,
        category: str | None = None,
        market_cap: float | None = None,
        screening_criteria: dict[str, Any] | None = None,
    ) -> None:
        async with AsyncSession(self._engine) as session:
            statement = select(CryptoHalalCache).where(CryptoHalalCache.symbol == symbol)
            result = await session.exec(statement)
            existing = result.first()
            if existing is not None:
                existing.compliance = compliance
                existing.category = category
                existing.market_cap = market_cap
                existing.screening_criteria = screening_criteria
                existing.updated_at = datetime.now(UTC)
                session.add(existing)
            else:
                session.add(
                    CryptoHalalCache(
                        symbol=symbol,
                        compliance=compliance,
                        category=category,
                        market_cap=market_cap,
                        screening_criteria=screening_criteria,
                    )
                )
            await session.commit()

    async def get_crypto_halal_status(self, symbol: str) -> str | None:
        async with AsyncSession(self._engine) as session:
            statement = select(CryptoHalalCache).where(CryptoHalalCache.symbol == symbol)
            result = await session.exec(statement)
            row = result.first()
            return row.compliance if row else None

    async def get_crypto_halal_symbols(self) -> list[str]:
        async with AsyncSession(self._engine) as session:
            statement = select(CryptoHalalCache).where(CryptoHalalCache.compliance == "halal")
            results = await session.exec(statement)
            return [r.symbol for r in results.all()]

    async def is_crypto_cache_fresh(self, max_age_hours: int = 24) -> bool:
        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        async with AsyncSession(self._engine) as session:
            statement = select(CryptoHalalCache).where(CryptoHalalCache.updated_at > cutoff)
            results = await session.exec(statement)
            return len(results.all()) > 0
