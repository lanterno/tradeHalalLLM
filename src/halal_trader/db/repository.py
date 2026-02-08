"""Data access layer using SQLModel."""

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import DailyPnl, HalalCache, LlmDecision, Trade


class Repository:
    """SQLModel-based repository for trade/PnL/halal operations."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    # ── Trades ──────────────────────────────────────────────────

    async def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_id: str | None = None,
        status: str = "pending",
        llm_reasoning: str | None = None,
    ) -> int:
        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_id=order_id,
            status=status,
            llm_reasoning=llm_reasoning,
        )
        async with AsyncSession(self._engine) as session:
            session.add(trade)
            await session.commit()
            await session.refresh(trade)
            return trade.id  # type: ignore[return-value]

    async def update_trade_status(
        self, trade_id: int, status: str, price: float | None = None
    ) -> None:
        async with AsyncSession(self._engine) as session:
            trade = await session.get(Trade, trade_id)
            if trade is None:
                return
            trade.status = status
            if price is not None:
                trade.price = price
            session.add(trade)
            await session.commit()

    async def get_today_trades(self) -> list[dict[str, Any]]:
        today = date.today().isoformat()
        async with AsyncSession(self._engine) as session:
            statement = (
                select(Trade)
                .where(Trade.timestamp >= datetime.fromisoformat(today))
                .where(Trade.timestamp < datetime.fromisoformat(today) + timedelta(days=1))
                .order_by(Trade.timestamp.desc())  # type: ignore[union-attr]
            )
            results = await session.exec(statement)
            return [trade.model_dump() for trade in results.all()]

    async def get_recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(Trade)
                .order_by(Trade.timestamp.desc())  # type: ignore[union-attr]
                .limit(limit)
            )
            results = await session.exec(statement)
            return [trade.model_dump() for trade in results.all()]

    # ── Daily P&L ───────────────────────────────────────────────

    async def start_day(self, starting_equity: float) -> None:
        today = date.today().isoformat()
        async with AsyncSession(self._engine) as session:
            # Check if row already exists for today
            statement = select(DailyPnl).where(DailyPnl.date == today)
            result = await session.exec(statement)
            existing = result.first()
            if existing is None:
                row = DailyPnl(date=today, starting_equity=starting_equity)
                session.add(row)
                await session.commit()

    async def end_day(self, ending_equity: float, realized_pnl: float, trades_count: int) -> None:
        today = date.today().isoformat()
        async with AsyncSession(self._engine) as session:
            statement = select(DailyPnl).where(DailyPnl.date == today)
            result = await session.exec(statement)
            row = result.first()
            if row is None:
                starting = ending_equity
            else:
                starting = row.starting_equity

            return_pct = (ending_equity - starting) / starting if starting else 0

            if row is not None:
                row.ending_equity = ending_equity
                row.realized_pnl = realized_pnl
                row.return_pct = return_pct
                row.trades_count = trades_count
                session.add(row)
                await session.commit()

    async def get_pnl_history(self, limit: int = 30) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(DailyPnl)
                .order_by(DailyPnl.date.desc())  # type: ignore[union-attr]
                .limit(limit)
            )
            results = await session.exec(statement)
            return [row.model_dump() for row in results.all()]

    # ── Halal Cache ─────────────────────────────────────────────

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
                row = HalalCache(
                    symbol=symbol,
                    compliance=compliance,
                    detail=detail,
                )
                session.add(row)
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
            return [row.symbol for row in results.all()]

    async def is_cache_fresh(self, max_age_hours: int = 24) -> bool:
        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        async with AsyncSession(self._engine) as session:
            statement = select(HalalCache).where(HalalCache.updated_at > cutoff)
            results = await session.exec(statement)
            return len(results.all()) > 0

    # ── LLM Decisions ───────────────────────────────────────────

    async def record_decision(
        self,
        provider: str,
        model: str,
        prompt_summary: str | None = None,
        raw_response: str | None = None,
        parsed_action: dict | None = None,
        symbols: list[str] | None = None,
        execution_ms: int | None = None,
    ) -> int:
        decision = LlmDecision(
            provider=provider,
            model=model,
            prompt_summary=prompt_summary,
            raw_response=raw_response,
            parsed_action=json.dumps(parsed_action) if parsed_action else None,
            symbols=json.dumps(symbols) if symbols else None,
            execution_ms=execution_ms,
        )
        async with AsyncSession(self._engine) as session:
            session.add(decision)
            await session.commit()
            await session.refresh(decision)
            return decision.id  # type: ignore[return-value]
