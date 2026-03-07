"""Data access layer using SQLModel."""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import (
    CryptoDailyPnl,
    CryptoHalalCache,
    CryptoTrade,
    DailyPnl,
    HalalCache,
    LlmDecision,
    StrategyAdjustment,
    Trade,
    TradeJournal,
)
from halal_trader.market_hours import today_eastern, trading_day_end_utc, trading_day_start_utc


class Repository:
    """SQLModel-based repository for trade/PnL/halal operations."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    # ── Stock Trades ───────────────────────────────────────────

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
        today = today_eastern()
        day_start = trading_day_start_utc(today)
        day_end = trading_day_end_utc(today)
        async with AsyncSession(self._engine) as session:
            statement = (
                select(Trade)
                .where(Trade.timestamp >= day_start)
                .where(Trade.timestamp < day_end)
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

    # ── Stock Daily P&L ────────────────────────────────────────

    async def start_day(self, starting_equity: float) -> None:
        today = today_eastern().isoformat()
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
        today = today_eastern().isoformat()
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

    # ── Halal Cache (Stocks) ───────────────────────────────────

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

    # ── LLM Decisions (shared) ──────────────────────────────────

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

    # ── Crypto Trades ──────────────────────────────────────────

    async def record_crypto_trade(
        self,
        pair: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_id: str | None = None,
        exchange: str = "binance",
        status: str = "pending",
        llm_reasoning: str | None = None,
        entry_price: float | None = None,
        stop_loss: float | None = None,
        target_price: float | None = None,
    ) -> int:
        trade = CryptoTrade(
            pair=pair,
            side=side,
            quantity=quantity,
            price=price,
            order_id=order_id,
            exchange=exchange,
            status=status,
            llm_reasoning=llm_reasoning,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
        )
        async with AsyncSession(self._engine) as session:
            session.add(trade)
            await session.commit()
            await session.refresh(trade)
            return trade.id  # type: ignore[return-value]

    async def update_crypto_trade_status(
        self, trade_id: int, status: str, price: float | None = None
    ) -> None:
        async with AsyncSession(self._engine) as session:
            trade = await session.get(CryptoTrade, trade_id)
            if trade is None:
                return
            trade.status = status
            if price is not None:
                trade.price = price
            session.add(trade)
            await session.commit()

    async def close_crypto_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """Mark a buy trade as closed with exit details."""
        async with AsyncSession(self._engine) as session:
            trade = await session.get(CryptoTrade, trade_id)
            if trade is None:
                return
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason
            trade.closed_at = datetime.now(UTC)
            trade.status = "closed"
            session.add(trade)
            await session.commit()

    async def get_today_crypto_trades(self) -> list[dict[str, Any]]:
        today = today_eastern()
        day_start = trading_day_start_utc(today)
        day_end = trading_day_end_utc(today)
        async with AsyncSession(self._engine) as session:
            statement = (
                select(CryptoTrade)
                .where(CryptoTrade.timestamp >= day_start)
                .where(CryptoTrade.timestamp < day_end)
                .order_by(CryptoTrade.timestamp.desc())  # type: ignore[union-attr]
            )
            results = await session.exec(statement)
            return [trade.model_dump() for trade in results.all()]

    async def get_open_crypto_trades(self) -> list[CryptoTrade]:
        """Return buy trades that haven't been closed yet (no exit recorded)."""
        async with AsyncSession(self._engine) as session:
            statement = (
                select(CryptoTrade)
                .where(CryptoTrade.side == "buy")
                .where(CryptoTrade.closed_at.is_(None))  # type: ignore[union-attr]
                .where(CryptoTrade.status != "rejected")
                .order_by(CryptoTrade.timestamp.asc())  # type: ignore[union-attr]
            )
            results = await session.exec(statement)
            return list(results.all())

    async def get_recent_crypto_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(CryptoTrade)
                .order_by(CryptoTrade.timestamp.desc())  # type: ignore[union-attr]
                .limit(limit)
            )
            results = await session.exec(statement)
            return [trade.model_dump() for trade in results.all()]

    # ── Crypto Round-Trip Analytics ──────────────────────────────

    async def get_completed_round_trips(
        self, limit: int = 100, lookback_days: int | None = None
    ) -> list[dict[str, Any]]:
        """Return closed buy trades paired with their exit data.

        Each result contains entry/exit prices, P&L, duration, and exit reason.
        """
        async with AsyncSession(self._engine) as session:
            statement = (
                select(CryptoTrade)
                .where(CryptoTrade.side == "buy")
                .where(CryptoTrade.closed_at.is_not(None))  # type: ignore[union-attr]
                .order_by(CryptoTrade.closed_at.desc())  # type: ignore[union-attr]
            )
            if lookback_days is not None:
                cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
                statement = statement.where(CryptoTrade.closed_at >= cutoff)
            statement = statement.limit(limit)

            results = await session.exec(statement)
            round_trips = []
            for trade in results.all():
                entry = trade.entry_price or trade.price or 0
                exit_p = trade.exit_price or 0
                pnl = (exit_p - entry) * trade.quantity
                pnl_pct = (exit_p - entry) / entry if entry > 0 else 0
                duration_min = 0.0
                if trade.closed_at and trade.timestamp:
                    duration_min = (trade.closed_at - trade.timestamp).total_seconds() / 60

                round_trips.append({
                    "id": trade.id,
                    "pair": trade.pair,
                    "buy_price": entry,
                    "sell_price": exit_p,
                    "quantity": trade.quantity,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "duration_minutes": duration_min,
                    "exit_reason": trade.exit_reason,
                    "opened_at": trade.timestamp,
                    "closed_at": trade.closed_at,
                })
            return round_trips

    # ── Crypto Daily P&L ───────────────────────────────────────

    async def start_crypto_day(self, starting_equity: float) -> None:
        today = today_eastern().isoformat()
        async with AsyncSession(self._engine) as session:
            statement = select(CryptoDailyPnl).where(CryptoDailyPnl.date == today)
            result = await session.exec(statement)
            existing = result.first()
            if existing is None:
                row = CryptoDailyPnl(date=today, starting_equity=starting_equity)
                session.add(row)
                await session.commit()

    async def end_crypto_day(
        self, ending_equity: float, realized_pnl: float, trades_count: int
    ) -> None:
        today = today_eastern().isoformat()
        async with AsyncSession(self._engine) as session:
            statement = select(CryptoDailyPnl).where(CryptoDailyPnl.date == today)
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

    async def get_crypto_pnl_history(self, limit: int = 30) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(CryptoDailyPnl)
                .order_by(CryptoDailyPnl.date.desc())  # type: ignore[union-attr]
                .limit(limit)
            )
            results = await session.exec(statement)
            return [row.model_dump() for row in results.all()]

    # ── Crypto Halal Cache ─────────────────────────────────────

    async def cache_crypto_halal_status(
        self,
        symbol: str,
        compliance: str,
        category: str | None = None,
        market_cap: float | None = None,
        screening_criteria: str | None = None,
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
                row = CryptoHalalCache(
                    symbol=symbol,
                    compliance=compliance,
                    category=category,
                    market_cap=market_cap,
                    screening_criteria=screening_criteria,
                )
                session.add(row)
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
            return [row.symbol for row in results.all()]

    async def is_crypto_cache_fresh(self, max_age_hours: int = 24) -> bool:
        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        async with AsyncSession(self._engine) as session:
            statement = select(CryptoHalalCache).where(CryptoHalalCache.updated_at > cutoff)
            results = await session.exec(statement)
            return len(results.all()) > 0

    # ── Strategy Adjustments ──────────────────────────────────

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
            return adj.id  # type: ignore[return-value]

    async def get_recent_adjustments(self, limit: int = 20) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(StrategyAdjustment)
                .order_by(StrategyAdjustment.timestamp.desc())  # type: ignore[union-attr]
                .limit(limit)
            )
            results = await session.exec(statement)
            return [row.model_dump() for row in results.all()]

    # ── Trade Journal ─────────────────────────────────────────

    async def record_trade_journal(
        self,
        trade_id: int,
        entry_context: str | None = None,
        exit_context: str | None = None,
        review_notes: str | None = None,
    ) -> int:
        entry = TradeJournal(
            trade_id=trade_id,
            entry_context=entry_context,
            exit_context=exit_context,
            review_notes=review_notes,
        )
        async with AsyncSession(self._engine) as session:
            session.add(entry)
            await session.commit()
            await session.refresh(entry)
            return entry.id  # type: ignore[return-value]
