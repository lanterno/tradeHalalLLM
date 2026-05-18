"""Purification ledger repository — Sharia obligations from haram dividends.

Wave D extraction. Records per-trade purification amounts owed and the
running outstanding/paid totals the operator settles. The matching
``PurificationRepo`` Protocol lives in ``protocols.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import PurificationEntry


class PurificationRepoImpl:
    """Concrete implementation of :class:`PurificationRepo`."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def record_purification(
        self,
        *,
        symbol: str,
        dividend_usd: float,
        haram_pct: float,
        purification_usd: float,
        notes: str | None = None,
    ) -> int:
        """Append a purification obligation; return its row id."""
        row = PurificationEntry(
            symbol=symbol.upper(),
            dividend_usd=float(dividend_usd),
            haram_pct=float(haram_pct),
            purification_usd=float(purification_usd),
            notes=notes,
        )
        async with AsyncSession(self._engine) as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            assert row.id is not None
            return row.id

    async def mark_purification_paid(self, entry_id: int, paid_at: datetime | None = None) -> bool:
        """Stamp ``paid_at`` on an entry. Returns ``False`` if the id is unknown."""
        async with AsyncSession(self._engine) as session:
            row = await session.get(PurificationEntry, entry_id)
            if row is None:
                return False
            row.paid_at = paid_at or datetime.now(UTC)
            session.add(row)
            await session.commit()
            return True

    async def get_outstanding_purification(self) -> list[dict[str, Any]]:
        """Unpaid obligations only — what the operator owes today."""
        async with AsyncSession(self._engine) as session:
            statement = (
                select(PurificationEntry)
                .where(col(PurificationEntry.paid_at).is_(None))
                .order_by(col(PurificationEntry.timestamp).desc())
            )
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]

    async def get_purification_totals(self) -> dict[str, float]:
        """Aggregate outstanding + paid totals in USD across all rows."""
        async with AsyncSession(self._engine) as session:
            outstanding = await session.exec(
                select(func.coalesce(func.sum(PurificationEntry.purification_usd), 0.0)).where(
                    col(PurificationEntry.paid_at).is_(None)
                )
            )
            paid = await session.exec(
                select(func.coalesce(func.sum(PurificationEntry.purification_usd), 0.0)).where(
                    col(PurificationEntry.paid_at).is_not(None)
                )
            )
            return {
                "outstanding_usd": float(outstanding.one() or 0.0),
                "paid_usd": float(paid.one() or 0.0),
            }
