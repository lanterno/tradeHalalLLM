"""Data access layer using SQLModel."""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import SQLModel, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import (
    CryptoDailyPnl,
    CryptoHalalCache,
    CryptoTrade,
    DailyPnl,
    HalalCache,
    HalalScreening,
    IndicatorSnapshot,
    LlmDecision,
    PairPause,
    PurificationEntry,
    ResearchJob,
    RuntimeConfig,
    StrategyAdjustment,
    Trade,
    WebAction,
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
                .where(model.timestamp >= day_start)
                .where(model.timestamp < day_end)
                .order_by(model.timestamp.desc())
            )
            results = await session.exec(statement)
            return [row.model_dump() for row in results.all()]

    async def _get_recent_rows(self, model: type[SQLModel], limit: int) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = select(model).order_by(model.timestamp.desc()).limit(limit)
            results = await session.exec(statement)
            return [row.model_dump() for row in results.all()]

    async def _start_day(self, pnl_model: type[SQLModel], starting_equity: float) -> None:
        today = today_eastern().isoformat()
        async with AsyncSession(self._engine) as session:
            statement = select(pnl_model).where(pnl_model.date == today)
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
            statement = select(pnl_model).where(pnl_model.date == today)
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

    async def _get_pnl_history(self, pnl_model: type[SQLModel], limit: int) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = select(pnl_model).order_by(pnl_model.date.desc()).limit(limit)
            results = await session.exec(statement)
            return [row.model_dump() for row in results.all()]

    async def _get_halal_status(self, cache_model: type[SQLModel], symbol: str) -> str | None:
        async with AsyncSession(self._engine) as session:
            statement = select(cache_model).where(cache_model.symbol == symbol)
            result = await session.exec(statement)
            row = result.first()
            return row.compliance if row else None

    async def _get_halal_symbols(self, cache_model: type[SQLModel]) -> list[str]:
        async with AsyncSession(self._engine) as session:
            statement = select(cache_model).where(cache_model.compliance == "halal")
            results = await session.exec(statement)
            return [row.symbol for row in results.all()]

    async def _is_cache_fresh(self, cache_model: type[SQLModel], max_age_hours: int) -> bool:
        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        async with AsyncSession(self._engine) as session:
            statement = select(cache_model).where(cache_model.updated_at > cutoff)
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
        submitted_at: datetime | None = None,
        filled_at: datetime | None = None,
        filled_price: float | None = None,
        filled_quantity: float | None = None,
        halal_screening_id: int | None = None,
        stop_loss: float | None = None,
        target_price: float | None = None,
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
        )
        async with AsyncSession(self._engine) as session:
            session.add(trade)
            await session.commit()
            await session.refresh(trade)
            return trade.id

    async def get_today_trades(self) -> list[dict[str, Any]]:
        return await self._get_today_rows(Trade)

    async def get_recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._get_recent_rows(Trade, limit)

    async def get_open_trades(self) -> list[Trade]:
        """Stock trades with status != 'closed' and an SL/TP set.

        Mirrors :meth:`get_open_crypto_trades` so the stock monitor can
        consume the same shape. Filters out fully-closed rows so a long
        history doesn't slow the per-tick scan.
        """
        async with AsyncSession(self._engine) as session:
            statement = select(Trade).where(Trade.closed_at.is_(None)).where(Trade.side == "buy")
            results = await session.exec(statement)
            return list(results.all())

    async def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
    ) -> None:
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

    async def update_stock_trade_stop_loss(self, trade_id: int, new_stop_loss: float) -> None:
        """Ratchet up the stop_loss on a stock trade (trailing-stop helper)."""
        async with AsyncSession(self._engine) as session:
            trade = await session.get(Trade, trade_id)
            if trade is None:
                return
            trade.stop_loss = new_stop_loss
            session.add(trade)
            await session.commit()

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

    # ── Research jobs (backtest queue) ─────────────────────────

    async def enqueue_research_job(
        self, *, kind: str, params: dict, name: str | None = None
    ) -> int:
        row = ResearchJob(kind=kind, name=name, params=json.dumps(params))
        async with AsyncSession(self._engine) as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row.id

    async def update_research_job(
        self,
        job_id: int,
        *,
        status: str,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        from datetime import UTC as _UTC
        from datetime import datetime as _dt

        async with AsyncSession(self._engine) as session:
            row = await session.get(ResearchJob, job_id)
            if row is None:
                return
            row.status = status
            if result is not None:
                row.result = json.dumps(result)
            if error is not None:
                row.error = error
            if status in ("ok", "error"):
                row.finished_at = _dt.now(_UTC)
            session.add(row)
            await session.commit()

    async def get_research_job(self, job_id: int) -> dict[str, Any] | None:
        async with AsyncSession(self._engine) as session:
            row = await session.get(ResearchJob, job_id)
            if row is None:
                return None
            data = row.model_dump()
            for key in ("params", "result"):
                if data.get(key):
                    try:
                        data[key] = json.loads(data[key])
                    except json.JSONDecodeError:
                        pass
            return data

    async def list_research_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = select(ResearchJob).order_by(ResearchJob.id.desc()).limit(limit)
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]

    async def pin_research_job(self, job_id: int, pinned: bool) -> bool:
        async with AsyncSession(self._engine) as session:
            row = await session.get(ResearchJob, job_id)
            if row is None:
                return False
            row.pinned = pinned
            session.add(row)
            await session.commit()
            return True

    # ── Runtime config overlay ─────────────────────────────────

    async def set_runtime_config(self, key: str, value: Any, *, set_by: str | None = None) -> None:
        """Insert/update a runtime overlay value (JSON-encoded for type fidelity)."""
        async with AsyncSession(self._engine) as session:
            row = await session.get(RuntimeConfig, key.upper())
            payload = json.dumps(value)
            if row is None:
                row = RuntimeConfig(key=key.upper(), value=payload, set_by=set_by)
            else:
                from datetime import UTC as _UTC
                from datetime import datetime as _dt

                row.value = payload
                row.set_by = set_by
                row.set_at = _dt.now(_UTC)
            session.add(row)
            await session.commit()

    async def delete_runtime_config(self, key: str) -> bool:
        async with AsyncSession(self._engine) as session:
            row = await session.get(RuntimeConfig, key.upper())
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def list_runtime_config(self) -> dict[str, Any]:
        async with AsyncSession(self._engine) as session:
            results = await session.exec(select(RuntimeConfig))
            out: dict[str, Any] = {}
            for r in results.all():
                try:
                    out[r.key] = json.loads(r.value)
                except json.JSONDecodeError:
                    out[r.key] = r.value
            return out

    # ── Per-pair operator pauses ───────────────────────────────

    async def pause_pair(
        self, pair: str, *, set_by: str | None = None, reason: str | None = None
    ) -> None:
        """Insert (or update) a pause row for ``pair``."""
        async with AsyncSession(self._engine) as session:
            row = await session.get(PairPause, pair.upper())
            if row is None:
                row = PairPause(pair=pair.upper(), set_by=set_by, reason=reason)
            else:
                row.set_by = set_by
                row.reason = reason
                from datetime import UTC as _UTC
                from datetime import datetime as _dt

                row.set_at = _dt.now(_UTC)
            session.add(row)
            await session.commit()

    async def resume_pair(self, pair: str) -> bool:
        """Delete the pause row. Returns True if a row was actually removed."""
        async with AsyncSession(self._engine) as session:
            row = await session.get(PairPause, pair.upper())
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def get_paused_pairs(self) -> set[str]:
        """The set of currently paused pair symbols (uppercased)."""
        async with AsyncSession(self._engine) as session:
            results = await session.exec(select(PairPause))
            return {r.pair for r in results.all()}

    async def list_pair_pauses(self) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            results = await session.exec(select(PairPause))
            return [r.model_dump() for r in results.all()]

    # ── Web mutation audit ─────────────────────────────────────

    async def begin_web_action(
        self, *, actor: str, method: str, path: str, payload: str | None = None
    ) -> int:
        """Insert a 'pending' web_actions row before the handler runs."""
        row = WebAction(actor=actor, method=method, path=path, payload=payload)
        async with AsyncSession(self._engine) as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row.id

    async def complete_web_action(
        self, action_id: int, *, status_code: int, error: str | None = None
    ) -> None:
        """Update a pending row with the final outcome."""
        async with AsyncSession(self._engine) as session:
            row = await session.get(WebAction, action_id)
            if row is None:
                return
            row.status_code = status_code
            row.outcome = "ok" if 200 <= status_code < 400 and error is None else "error"
            row.error = error
            session.add(row)
            await session.commit()

    async def get_recent_web_actions(self, limit: int = 50) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = select(WebAction).order_by(WebAction.id.desc()).limit(limit)
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]

    async def delete_old_web_actions(self, *, older_than: timedelta) -> int:
        """Prune ``web_actions`` rows older than ``older_than``.

        Returns the number of rows deleted. The daily-end scheduler
        hook calls this so a long-running deployment doesn't accumulate
        unbounded mutation-audit rows.
        """
        from sqlalchemy import delete as sa_delete

        cutoff = datetime.now(UTC) - older_than
        async with AsyncSession(self._engine) as session:
            result = await session.exec(sa_delete(WebAction).where(WebAction.timestamp < cutoff))
            await session.commit()
            return int(result.rowcount or 0)

    # ── Purification ledger ────────────────────────────────────

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
                .where(PurificationEntry.paid_at.is_(None))
                .order_by(PurificationEntry.timestamp.desc())
            )
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]

    async def get_purification_totals(self) -> dict[str, float]:
        """Aggregate outstanding + paid totals in USD across all rows."""
        async with AsyncSession(self._engine) as session:
            outstanding = await session.exec(
                select(func.coalesce(func.sum(PurificationEntry.purification_usd), 0.0)).where(
                    PurificationEntry.paid_at.is_(None)
                )
            )
            paid = await session.exec(
                select(func.coalesce(func.sum(PurificationEntry.purification_usd), 0.0)).where(
                    PurificationEntry.paid_at.is_not(None)
                )
            )
            return {
                "outstanding_usd": float(outstanding.one() or 0.0),
                "paid_usd": float(paid.one() or 0.0),
            }

    # ── Halal Screenings (shared audit trail) ──────────────────

    async def record_halal_screening(
        self,
        *,
        symbol: str,
        asset_class: str,
        source: str,
        decision: str,
        criteria: dict | None = None,
        cache_hit: bool = False,
    ) -> int:
        """Persist a screening decision and return its row id.

        Callers should pass the returned id to ``record_trade`` /
        ``record_crypto_trade`` via ``halal_screening_id`` so each trade
        is provably linked to the compliance decision that gated it.
        """
        row = HalalScreening(
            symbol=symbol,
            asset_class=asset_class,
            source=source,
            decision=decision,
            criteria=json.dumps(criteria) if criteria else None,
            cache_hit=cache_hit,
        )
        async with AsyncSession(self._engine) as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row.id

    async def get_halal_screening(self, screening_id: int) -> dict[str, Any] | None:
        async with AsyncSession(self._engine) as session:
            row = await session.get(HalalScreening, screening_id)
            if row is None:
                return None
            data = row.model_dump()
            if data.get("criteria"):
                data["criteria"] = json.loads(data["criteria"])
            return data

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
        prompt_version: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        cost_usd: float | None = None,
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
            prompt_version=prompt_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=cost_usd,
        )
        async with AsyncSession(self._engine) as session:
            session.add(decision)
            await session.commit()
            await session.refresh(decision)
            return decision.id

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
            return trade.id

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
                .where(CryptoTrade.closed_at.is_(None))
                .where(CryptoTrade.status != "rejected")
                .order_by(CryptoTrade.timestamp.asc())
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
                .where(CryptoTrade.closed_at.is_(None))
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
                .where(CryptoTrade.closed_at.is_(None))
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

    async def get_completed_stock_round_trips(
        self, limit: int = 100, lookback_days: int | None = None
    ) -> list[dict[str, Any]]:
        """Stock equivalent of :meth:`get_completed_round_trips`.

        Returns the same dict shape (with ``pair`` set to the symbol, so
        the shared analytics module doesn't need to special-case stocks).
        """
        async with AsyncSession(self._engine) as session:
            statement = (
                select(Trade)
                .where(Trade.side == "buy")
                .where(Trade.closed_at.is_not(None))
                .order_by(Trade.closed_at.desc())
            )
            if lookback_days is not None:
                cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
                statement = statement.where(Trade.closed_at >= cutoff)
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
                        "pair": trade.symbol,  # alias so analytics is symmetric
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

    async def get_completed_round_trips(
        self, limit: int = 100, lookback_days: int | None = None
    ) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(CryptoTrade)
                .where(CryptoTrade.side == "buy")
                .where(CryptoTrade.closed_at.is_not(None))
                .order_by(CryptoTrade.closed_at.desc())
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
            return snap.id

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
                .where(IndicatorSnapshot.label.is_not(None))
                .order_by(IndicatorSnapshot.timestamp.desc())
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
            return adj.id

    async def get_latest_strategy_adjustments(self) -> dict[str, float]:
        async with AsyncSession(self._engine) as session:
            statement = (
                select(StrategyAdjustment).order_by(StrategyAdjustment.timestamp.desc()).limit(100)
            )
            results = await session.exec(statement)
            latest: dict[str, float] = {}
            for row in results.all():
                if row.parameter not in latest:
                    latest[row.parameter] = row.new_value
            return latest

    async def get_recent_adjustments(self, limit: int = 20) -> list[dict[str, Any]]:
        return await self._get_recent_rows(StrategyAdjustment, limit)
