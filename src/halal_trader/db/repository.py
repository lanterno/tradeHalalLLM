"""Data access layer using SQLModel."""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import (
    CryptoDailyPnl,
    CryptoHalalCache,
    CryptoTrade,
    DailyPnl,
    HalalCache,
    IndicatorSnapshot,
    LlmDecision,
    StrategyAdjustment,
    Trade,
)
from halal_trader.market_hours import today_eastern, trading_day_end_utc, trading_day_start_utc


class Repository:
    """SQLModel-based repository for trade/PnL/halal operations."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    # ── Generic helpers (private) ─────────────────────────────

    async def _get_today_rows(self, model: type[SQLModel]) -> list[dict[str, Any]]:
        today = today_eastern()
        day_start = trading_day_start_utc(today)
        day_end = trading_day_end_utc(today)
        async with AsyncSession(self._engine) as session:
            statement = (
                select(model)
                .where(model.timestamp >= day_start)  # type: ignore[attr-defined]
                .where(model.timestamp < day_end)  # type: ignore[attr-defined]
                .order_by(model.timestamp.desc())  # type: ignore[attr-defined]
            )
            results = await session.exec(statement)
            return [row.model_dump() for row in results.all()]

    async def _get_recent_rows(self, model: type[SQLModel], limit: int) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(model)
                .order_by(model.timestamp.desc())  # type: ignore[attr-defined]
                .limit(limit)
            )
            results = await session.exec(statement)
            return [row.model_dump() for row in results.all()]

    async def _start_day(self, pnl_model: type[SQLModel], starting_equity: float) -> None:
        today = today_eastern().isoformat()
        async with AsyncSession(self._engine) as session:
            statement = select(pnl_model).where(
                pnl_model.date == today  # type: ignore[attr-defined]
            )
            result = await session.exec(statement)
            if result.first() is None:
                session.add(pnl_model(date=today, starting_equity=starting_equity))
                await session.commit()

    async def _end_day(
        self,
        pnl_model: type[SQLModel],
        ending_equity: float,
        realized_pnl: float,
        trades_count: int,
    ) -> None:
        today = today_eastern().isoformat()
        async with AsyncSession(self._engine) as session:
            statement = select(pnl_model).where(
                pnl_model.date == today  # type: ignore[attr-defined]
            )
            result = await session.exec(statement)
            row = result.first()
            if row is None:
                return
            starting = row.starting_equity  # type: ignore[union-attr]
            return_pct = (ending_equity - starting) / starting if starting else 0
            row.ending_equity = ending_equity  # type: ignore[union-attr]
            row.realized_pnl = realized_pnl  # type: ignore[union-attr]
            row.return_pct = return_pct  # type: ignore[union-attr]
            row.trades_count = trades_count  # type: ignore[union-attr]
            session.add(row)
            await session.commit()

    async def _get_pnl_history(self, pnl_model: type[SQLModel], limit: int) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(pnl_model)
                .order_by(pnl_model.date.desc())  # type: ignore[attr-defined]
                .limit(limit)
            )
            results = await session.exec(statement)
            return [row.model_dump() for row in results.all()]

    async def _get_halal_status(self, cache_model: type[SQLModel], symbol: str) -> str | None:
        async with AsyncSession(self._engine) as session:
            statement = select(cache_model).where(
                cache_model.symbol == symbol  # type: ignore[attr-defined]
            )
            result = await session.exec(statement)
            row = result.first()
            return row.compliance if row else None  # type: ignore[union-attr]

    async def _get_halal_symbols(self, cache_model: type[SQLModel]) -> list[str]:
        async with AsyncSession(self._engine) as session:
            statement = select(cache_model).where(
                cache_model.compliance == "halal"  # type: ignore[attr-defined]
            )
            results = await session.exec(statement)
            return [row.symbol for row in results.all()]  # type: ignore[union-attr]

    async def _is_cache_fresh(self, cache_model: type[SQLModel], max_age_hours: int) -> bool:
        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        async with AsyncSession(self._engine) as session:
            statement = select(cache_model).where(
                cache_model.updated_at > cutoff  # type: ignore[attr-defined]
            )
            results = await session.exec(statement)
            return len(results.all()) > 0

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

    async def get_today_trades(self) -> list[dict[str, Any]]:
        return await self._get_today_rows(Trade)

    async def get_recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._get_recent_rows(Trade, limit)

    # ── Stock Daily P&L ────────────────────────────────────────

    async def start_day(self, starting_equity: float) -> None:
        await self._start_day(DailyPnl, starting_equity)

    async def end_day(self, ending_equity: float, realized_pnl: float, trades_count: int) -> None:
        await self._end_day(DailyPnl, ending_equity, realized_pnl, trades_count)

    async def get_pnl_history(self, limit: int = 30) -> list[dict[str, Any]]:
        return await self._get_pnl_history(DailyPnl, limit)

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
                session.add(HalalCache(symbol=symbol, compliance=compliance, detail=detail))
            await session.commit()

    async def get_halal_status(self, symbol: str) -> str | None:
        return await self._get_halal_status(HalalCache, symbol)

    async def get_halal_symbols(self) -> list[str]:
        return await self._get_halal_symbols(HalalCache)

    async def is_cache_fresh(self, max_age_hours: int = 24) -> bool:
        return await self._is_cache_fresh(HalalCache, max_age_hours)

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
        thinking: str | None = None,
    ) -> int:
        decision = LlmDecision(
            provider=provider,
            model=model,
            prompt_summary=prompt_summary,
            raw_response=raw_response,
            parsed_action=json.dumps(parsed_action) if parsed_action else None,
            symbols=json.dumps(symbols) if symbols else None,
            execution_ms=execution_ms,
            thinking=thinking,
        )
        async with AsyncSession(self._engine) as session:
            session.add(decision)
            await session.commit()
            await session.refresh(decision)
            return decision.id  # type: ignore[return-value]

    async def get_recent_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._get_recent_rows(LlmDecision, limit)

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

    async def update_crypto_trade_stop_loss(self, trade_id: int, new_stop_loss: float) -> None:
        async with AsyncSession(self._engine) as session:
            trade = await session.get(CryptoTrade, trade_id)
            if trade is None:
                return
            trade.stop_loss = new_stop_loss
            session.add(trade)
            await session.commit()

    async def close_crypto_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
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
        return await self._get_today_rows(CryptoTrade)

    async def get_open_crypto_trades(self) -> list[CryptoTrade]:
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

    async def get_open_crypto_trades_for_pair(self, pair: str) -> list[CryptoTrade]:
        """Return all open (unclosed, non-rejected) buy trades for a specific pair."""
        async with AsyncSession(self._engine) as session:
            statement = (
                select(CryptoTrade)
                .where(CryptoTrade.side == "buy")
                .where(CryptoTrade.pair == pair)
                .where(CryptoTrade.closed_at.is_(None))  # type: ignore[union-attr]
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
                .where(CryptoTrade.closed_at.is_(None))  # type: ignore[union-attr]
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
        return await self._get_recent_rows(CryptoTrade, limit)

    async def get_completed_round_trips(
        self, limit: int = 100, lookback_days: int | None = None
    ) -> list[dict[str, Any]]:
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

    # ── Crypto Daily P&L ───────────────────────────────────────

    async def start_crypto_day(self, starting_equity: float) -> None:
        await self._start_day(CryptoDailyPnl, starting_equity)

    async def end_crypto_day(
        self, ending_equity: float, realized_pnl: float, trades_count: int
    ) -> None:
        await self._end_day(CryptoDailyPnl, ending_equity, realized_pnl, trades_count)

    async def get_crypto_pnl_history(self, limit: int = 30) -> list[dict[str, Any]]:
        return await self._get_pnl_history(CryptoDailyPnl, limit)

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
        return await self._get_halal_status(CryptoHalalCache, symbol)

    async def get_crypto_halal_symbols(self) -> list[str]:
        return await self._get_halal_symbols(CryptoHalalCache)

    async def is_crypto_cache_fresh(self, max_age_hours: int = 24) -> bool:
        return await self._is_cache_fresh(CryptoHalalCache, max_age_hours)

    # ── Indicator Snapshots (ML training) ────────────────────

    async def record_indicator_snapshot(
        self,
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
            return snap.id  # type: ignore[return-value]

    async def label_indicator_snapshot(self, trade_id: int, label: int, return_pct: float) -> None:
        async with AsyncSession(self._engine) as session:
            statement = select(IndicatorSnapshot).where(IndicatorSnapshot.trade_id == trade_id)
            result = await session.exec(statement)
            snap = result.first()
            if snap:
                snap.label = label
                snap.return_pct = return_pct
                session.add(snap)
                await session.commit()

    async def get_labeled_snapshots(self, min_samples: int = 50) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(IndicatorSnapshot)
                .where(IndicatorSnapshot.label.is_not(None))  # type: ignore[union-attr]
                .order_by(IndicatorSnapshot.timestamp.desc())  # type: ignore[union-attr]
                .limit(5000)
            )
            results = await session.exec(statement)
            rows = results.all()
            if len(rows) < min_samples:
                return []
            return [r.model_dump() for r in rows]

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

    async def get_latest_strategy_adjustments(self) -> dict[str, float]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(StrategyAdjustment)
                .order_by(StrategyAdjustment.timestamp.desc())  # type: ignore[union-attr]
                .limit(100)
            )
            results = await session.exec(statement)
            latest: dict[str, float] = {}
            for row in results.all():
                if row.parameter not in latest:
                    latest[row.parameter] = row.new_value
            return latest

    async def get_recent_adjustments(self, limit: int = 20) -> list[dict[str, Any]]:
        return await self._get_recent_rows(StrategyAdjustment, limit)
