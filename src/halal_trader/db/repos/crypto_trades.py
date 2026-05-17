"""Crypto trades repository.

Wave D extraction. Owns the ``crypto_trades`` table — the per-fill
ledger driving the crypto cycle, monitor, and reconcile loop. Also
owns :meth:`get_completed_round_trips`, the buy↔sell pairing query
used by analytics. Matching ``CryptoTradeRepo`` Protocol in
``protocols.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import CryptoTrade
from halal_trader.market_hours import (
    today_eastern,
    trading_day_end_utc,
    trading_day_start_utc,
)


class CryptoTradeRepoImpl:
    """Concrete implementation of :class:`CryptoTradeRepo`."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

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
        submitted_at: datetime | None = None,
        filled_at: datetime | None = None,
        filled_price: float | None = None,
        filled_quantity: float | None = None,
        halal_screening_id: int | None = None,
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
            submitted_at=submitted_at,
            filled_at=filled_at,
            filled_price=filled_price,
            filled_quantity=filled_quantity,
            halal_screening_id=halal_screening_id,
        )
        async with AsyncSession(self._engine) as session:
            session.add(trade)
            await session.commit()
            await session.refresh(trade)
            assert trade.id is not None
            return trade.id

    async def update_crypto_trade_stop_loss(
        self, trade_id: int, new_stop_loss: float
    ) -> None:
        async with AsyncSession(self._engine) as session:
            trade = await session.get(CryptoTrade, trade_id)
            if trade is None:
                return
            trade.stop_loss = new_stop_loss
            session.add(trade)
            await session.commit()

    async def close_crypto_trade(
        self, trade_id: int, exit_price: float, exit_reason: str
    ) -> None:
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
                .order_by(col(CryptoTrade.timestamp).desc())
            )
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]

    async def get_open_crypto_trades(self) -> list[CryptoTrade]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(CryptoTrade)
                .where(CryptoTrade.side == "buy")
                .where(col(CryptoTrade.closed_at).is_(None))
                .where(CryptoTrade.status != "rejected")
                .order_by(col(CryptoTrade.timestamp).asc())
            )
            results = await session.exec(statement)
            return list(results.all())

    async def get_open_crypto_trades_for_pair(self, pair: str) -> list[CryptoTrade]:
        """Return all open (unclosed, non-rejected) buy trades for a specific pair."""
        async with AsyncSession(self._engine) as session:
            statement = (
                select(CryptoTrade)
                .where(CryptoTrade.side == "buy")
                .where(CryptoTrade.pair == pair)
                .where(col(CryptoTrade.closed_at).is_(None))
                .where(CryptoTrade.status != "rejected")
            )
            results = await session.exec(statement)
            return list(results.all())

    async def close_open_crypto_trades_for_pair(
        self,
        pair: str,
        exit_price: float,
        exit_reason: str,
        exclude_id: int | None = None,
    ) -> int:
        """Close all open trades for a pair (except exclude_id). Returns count closed."""
        async with AsyncSession(self._engine) as session:
            statement = (
                select(CryptoTrade)
                .where(CryptoTrade.side == "buy")
                .where(CryptoTrade.pair == pair)
                .where(col(CryptoTrade.closed_at).is_(None))
                .where(CryptoTrade.status != "rejected")
            )
            if exclude_id is not None:
                statement = statement.where(CryptoTrade.id != exclude_id)
            results = await session.exec(statement)
            trades = results.all()
            now = datetime.now(UTC)
            count = 0
            for trade in trades:
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.closed_at = now
                trade.status = "closed"
                session.add(trade)
                count += 1
            if count > 0:
                await session.commit()
            return count

    async def get_recent_crypto_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(CryptoTrade).order_by(col(CryptoTrade.timestamp).desc()).limit(limit)
            )
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]

    async def get_completed_round_trips(
        self, limit: int = 100, lookback_days: int | None = None
    ) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(CryptoTrade)
                .where(CryptoTrade.side == "buy")
                .where(col(CryptoTrade.closed_at).is_not(None))
                .order_by(col(CryptoTrade.closed_at).desc())
            )
            if lookback_days is not None:
                cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
                statement = statement.where(col(CryptoTrade.closed_at) >= cutoff)
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
                    duration_min = (
                        trade.closed_at - trade.timestamp
                    ).total_seconds() / 60
                round_trips.append(
                    {
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
                    }
                )
            return round_trips
