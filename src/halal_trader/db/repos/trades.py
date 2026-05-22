"""Stock trades repository.

Wave D extraction. Owns the ``trades`` table — order intents, fills,
exits — used by the stock executor + monitor + reconcile loop. The
matching ``TradeRepo`` Protocol lives in ``protocols.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import Trade
from halal_trader.market_hours import (
    today_eastern,
    trading_day_end_utc,
    trading_day_start_utc,
)


class TradeRepoImpl:
    """Concrete implementation of :class:`TradeRepo` for stocks."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_id: str | None = None,
        status: str = "pending",
        llm_reasoning: str | None = None,
        submitted_at: datetime | None = None,
        filled_at: datetime | None = None,
        filled_price: float | None = None,
        filled_quantity: float | None = None,
        halal_screening_id: int | None = None,
        stop_loss: float | None = None,
        target_price: float | None = None,
        paper_slippage_pct: float | None = None,
    ) -> int:
        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_id=order_id,
            status=status,
            llm_reasoning=llm_reasoning,
            submitted_at=submitted_at,
            filled_at=filled_at,
            filled_price=filled_price,
            filled_quantity=filled_quantity,
            halal_screening_id=halal_screening_id,
            stop_loss=stop_loss,
            target_price=target_price,
            paper_slippage_pct=paper_slippage_pct,
        )
        async with AsyncSession(self._engine) as session:
            session.add(trade)
            await session.commit()
            await session.refresh(trade)
            assert trade.id is not None
            return trade.id

    async def get_today_trades(self) -> list[dict[str, Any]]:
        today = today_eastern()
        day_start = trading_day_start_utc(today)
        day_end = trading_day_end_utc(today)
        async with AsyncSession(self._engine) as session:
            statement = (
                select(Trade)
                .where(Trade.timestamp >= day_start)
                .where(Trade.timestamp < day_end)
                .order_by(col(Trade.timestamp).desc())
            )
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]

    async def get_recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = select(Trade).order_by(col(Trade.timestamp).desc()).limit(limit)
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]

    async def get_open_trades(self) -> list[Trade]:
        """Stock trades with status != 'closed' and an SL/TP set."""
        async with AsyncSession(self._engine) as session:
            statement = (
                select(Trade).where(col(Trade.closed_at).is_(None)).where(Trade.side == "buy")
            )
            results = await session.exec(statement)
            return list(results.all())

    async def get_recently_closed(self, *, minutes: int = 60) -> list[dict[str, Any]]:
        """Closed stock trades (BUYs whose ``closed_at`` is within the window).

        The LLM uses this to avoid re-buying a symbol it just sold —
        the prompt only shows current positions, not recent exits, so
        without surfacing this the bot can flip out and back into the
        same symbol on consecutive cycles (observed 2026-05-21 13:15
        → 13:30 on AMZN: sold then re-bought 15 min later).
        """
        cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
        async with AsyncSession(self._engine) as session:
            statement = (
                select(Trade)
                .where(Trade.side == "buy")
                .where(col(Trade.closed_at).is_not(None))
                .where(col(Trade.closed_at) >= cutoff)
                .order_by(col(Trade.closed_at).desc())
            )
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]

    async def get_recent_sells(self, *, minutes: int = 60) -> list[dict[str, Any]]:
        """SELL trades created within the window.

        Companion to ``get_recently_closed`` for the executor's
        re-entry cooldown — captures LLM-initiated SELLs whose
        ``close_open_trades_for_symbol`` write may have lagged or
        (for rows that predate that fix) never fired. Using
        ``Trade.timestamp`` rather than a separate close-time column
        because SELLs are terminal events in their own right.
        """
        cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
        async with AsyncSession(self._engine) as session:
            statement = (
                select(Trade)
                .where(Trade.side == "sell")
                .where(Trade.timestamp >= cutoff)
                .order_by(col(Trade.timestamp).desc())
            )
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]

    async def close_trade(self, trade_id: int, exit_price: float, exit_reason: str) -> None:
        """Mark a stock trade as closed with exit price + reason."""
        async with AsyncSession(self._engine) as session:
            trade = await session.get(Trade, trade_id)
            if trade is None:
                return
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason
            trade.closed_at = datetime.now(UTC)
            trade.status = "closed"
            session.add(trade)
            await session.commit()

    async def close_open_trades_for_symbol(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
    ) -> int:
        """Mark all open BUYs for ``symbol`` as closed (mirrors the crypto
        ``close_open_crypto_trades_for_pair`` helper).

        Called from the executor's ``_execute_sell`` so an LLM-initiated
        SELL doesn't just record a SELL row but also stamps ``closed_at``
        on the underlying open BUY(s). Without this, the recent-close
        cooldown query missed LLM exits and same-symbol re-buys leaked
        through (observed 2026-05-21 14:45 ET: QCOM sold 14:30, bought
        back 14:45 — cooldown was blind because the BUY's ``closed_at``
        stayed NULL).
        """
        async with AsyncSession(self._engine) as session:
            statement = (
                select(Trade)
                .where(Trade.side == "buy")
                .where(Trade.symbol == symbol)
                .where(col(Trade.closed_at).is_(None))
                .where(Trade.status != "rejected")
            )
            results = await session.exec(statement)
            trades = list(results.all())
            now = datetime.now(UTC)
            count = 0
            for trade in trades:
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.closed_at = now
                trade.status = "closed"
                session.add(trade)
                count += 1
            await session.commit()
            return count

    async def update_stock_trade_stop_loss(self, trade_id: int, new_stop_loss: float) -> None:
        """Ratchet up the stop_loss on a stock trade (trailing-stop helper)."""
        async with AsyncSession(self._engine) as session:
            trade = await session.get(Trade, trade_id)
            if trade is None:
                return
            trade.stop_loss = new_stop_loss
            session.add(trade)
            await session.commit()

    async def get_completed_stock_round_trips(
        self, limit: int = 100, lookback_days: int | None = None
    ) -> list[dict[str, Any]]:
        """Closed stock round-trips reshaped for cross-asset analytics.

        Returns the same dict shape ``CryptoTradeRepoImpl.get_completed_round_trips``
        does (with ``pair`` set to the symbol) so a single analytics
        module can consume either source.
        """
        async with AsyncSession(self._engine) as session:
            statement = (
                select(Trade)
                .where(Trade.side == "buy")
                .where(col(Trade.closed_at).is_not(None))
                .order_by(col(Trade.closed_at).desc())
            )
            if lookback_days is not None:
                cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
                statement = statement.where(col(Trade.closed_at) >= cutoff)
            statement = statement.limit(limit)

            results = await session.exec(statement)
            round_trips = []
            for trade in results.all():
                entry = trade.filled_price or trade.price or 0
                exit_p = trade.exit_price or 0
                pnl = (exit_p - entry) * trade.quantity
                pnl_pct = (exit_p - entry) / entry if entry > 0 else 0
                duration_min = 0.0
                if trade.closed_at and trade.timestamp:
                    duration_min = (trade.closed_at - trade.timestamp).total_seconds() / 60
                round_trips.append(
                    {
                        "id": trade.id,
                        "pair": trade.symbol,
                        "buy_price": entry,
                        "sell_price": exit_p,
                        "quantity": trade.quantity,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "duration_minutes": duration_min,
                        "exit_reason": trade.exit_reason,
                        "opened_at": trade.timestamp,
                        "closed_at": trade.closed_at,
                    }
                )
            return round_trips
