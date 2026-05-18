"""Stock daily P&L repository.

Wave D extraction. Mirrors :class:`PnlRepoImpl` (crypto) over the
``daily_pnl`` table — one row per trading day with starting/ending
equity and realized P&L. Matching ``StockPnlRepo`` Protocol in
``protocols.py``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import DailyPnl
from halal_trader.market_hours import today_eastern


class StockPnlRepoImpl:
    """Concrete implementation of :class:`StockPnlRepo`."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def start_day(self, starting_equity: float) -> None:
        today = today_eastern().isoformat()
        async with AsyncSession(self._engine) as session:
            statement = select(DailyPnl).where(DailyPnl.date == today)
            result = await session.exec(statement)
            if result.first() is None:
                session.add(DailyPnl(date=today, starting_equity=starting_equity))
                await session.commit()

    async def end_day(self, ending_equity: float, realized_pnl: float, trades_count: int) -> None:
        today = today_eastern().isoformat()
        async with AsyncSession(self._engine) as session:
            statement = select(DailyPnl).where(DailyPnl.date == today)
            result = await session.exec(statement)
            row = result.first()
            if row is None:
                return
            starting = row.starting_equity
            return_pct = (ending_equity - starting) / starting if starting else 0
            row.ending_equity = ending_equity
            row.realized_pnl = realized_pnl
            row.return_pct = return_pct
            row.trades_count = trades_count
            session.add(row)
            await session.commit()

    async def get_pnl_history(self, limit: int = 30) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = select(DailyPnl).order_by(col(DailyPnl.date).desc()).limit(limit)
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]
