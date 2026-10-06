"""Data access layer using SQLModel.

:class:`Repository` is a flat facade over the per-table repos in
``db/repos/``: every method forwards to the repo that owns its table.
New code should depend on the narrowest protocol it needs
(``TradeRepo``, ``LlmDecisionRepo``, …) and take it from a
:class:`RepoBundle` (``RepoBundle.from_engine(engine)`` or
:attr:`Repository.bundle`).
"""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.db.models import Trade

if TYPE_CHECKING:
    from halal_trader.db.repos import RepoBundle


class Repository:
    """Flat facade — see module docstring; prefer ``RepoBundle``."""

    def __init__(self, engine: AsyncEngine) -> None:
        from halal_trader.db.repos import RepoBundle

        self._engine = engine
        self._bundle = RepoBundle.from_engine(engine)
        self._web_audit = self._bundle.web_audit
        self._purification = self._bundle.purification
        self._halal_screening = self._bundle.halal_screening
        self._daily_recommendations = self._bundle.daily_recommendations
        self._indicator_snapshots = self._bundle.indicator_snapshots
        self._llm_decisions = self._bundle.llm_decisions
        self._trades = self._bundle.trades
        self._strategy_adjustments = self._bundle.strategy_adjustments
        self._stock_halal_cache = self._bundle.stock_halal_cache
        self._stock_pnl = self._bundle.stock_pnl

    @property
    def bundle(self) -> RepoBundle:
        """The per-table repos this facade delegates to (the same instances)."""
        return self._bundle

    # ── Stock Trades (delegated to TradeRepoImpl) ──────────────────

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
        entry_type: str | None = None,
    ) -> int:
        return await self._trades.record_trade(
            symbol,
            side,
            quantity,
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
            entry_type=entry_type,
        )

    async def get_today_trades(self) -> list[dict[str, Any]]:
        return await self._trades.get_today_trades()

    async def get_recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._trades.get_recent_trades(limit)

    async def get_open_trades(self) -> list[Trade]:
        return await self._trades.get_open_trades()

    async def get_recently_closed(self, *, minutes: int = 60) -> list[dict[str, Any]]:
        return await self._trades.get_recently_closed(minutes=minutes)

    async def get_recent_sells(self, *, minutes: int = 60) -> list[dict[str, Any]]:
        return await self._trades.get_recent_sells(minutes=minutes)

    async def close_trade(self, trade_id: int, exit_price: float, exit_reason: str) -> None:
        await self._trades.close_trade(trade_id, exit_price, exit_reason)

    async def close_open_trades_for_symbol(
        self, symbol: str, exit_price: float, exit_reason: str
    ) -> int:
        return await self._trades.close_open_trades_for_symbol(symbol, exit_price, exit_reason)

    async def update_stock_trade_stop_loss(
        self, trade_id: int, new_stop_loss: float, high_water: float | None = None
    ) -> None:
        await self._trades.update_stock_trade_stop_loss(trade_id, new_stop_loss, high_water)

    # ── Stock Daily P&L ────────────────────────────────────────

    async def start_day(self, starting_equity: float) -> float:
        return await self._stock_pnl.start_day(starting_equity)

    async def end_day(self, ending_equity: float, realized_pnl: float, trades_count: int) -> None:
        await self._stock_pnl.end_day(ending_equity, realized_pnl, trades_count)

    async def get_pnl_history(self, limit: int = 30) -> list[dict[str, Any]]:
        return await self._stock_pnl.get_pnl_history(limit)

    # ── Halal Cache (delegated to StockHalalCacheRepoImpl) ─────────

    async def cache_halal_status(
        self, symbol: str, compliance: str, detail: str | None = None
    ) -> None:
        await self._stock_halal_cache.cache_halal_status(symbol, compliance, detail)

    async def get_halal_status(self, symbol: str) -> str | None:
        return await self._stock_halal_cache.get_halal_status(symbol)

    async def get_halal_symbols(self) -> list[str]:
        return await self._stock_halal_cache.get_halal_symbols()

    async def is_cache_fresh(self, max_age_hours: int = 24) -> bool:
        return await self._stock_halal_cache.is_cache_fresh(max_age_hours)

    # ── Daily recommendation (delegated to DailyRecommendationRepoImpl) ──

    async def save_recommendation(self, rec: dict[str, Any]) -> int:
        return await self._daily_recommendations.save_recommendation(rec)

    async def get_latest_recommendation(self) -> dict[str, Any] | None:
        return await self._daily_recommendations.get_latest_recommendation()

    async def get_recent_recommendations(self, limit: int = 30) -> list[dict[str, Any]]:
        return await self._daily_recommendations.get_recent_recommendations(limit)

    async def get_recommendations_to_score(self, limit: int = 500) -> list[dict[str, Any]]:
        return await self._daily_recommendations.get_recommendations_to_score(limit)

    async def update_recommendation_outcome(self, rec_id: int, **fields: Any) -> bool:
        return await self._daily_recommendations.update_recommendation_outcome(rec_id, **fields)

    # ── Web mutation audit ─────────────────────────────────────

    async def begin_web_action(
        self, *, actor: str, method: str, path: str, payload: str | None = None
    ) -> int:
        return await self._web_audit.begin_web_action(
            actor=actor, method=method, path=path, payload=payload
        )

    async def complete_web_action(
        self, action_id: int, *, status_code: int, error: str | None = None
    ) -> None:
        await self._web_audit.complete_web_action(action_id, status_code=status_code, error=error)

    async def get_recent_web_actions(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._web_audit.get_recent_web_actions(limit)

    async def delete_old_web_actions(self, *, older_than: timedelta) -> int:
        return await self._web_audit.delete_old_web_actions(older_than=older_than)

    # ── Purification ledger (delegated to PurificationRepoImpl) ──────

    async def record_purification(
        self,
        *,
        symbol: str,
        dividend_usd: float,
        haram_pct: float,
        purification_usd: float,
        notes: str | None = None,
    ) -> int:
        return await self._purification.record_purification(
            symbol=symbol,
            dividend_usd=dividend_usd,
            haram_pct=haram_pct,
            purification_usd=purification_usd,
            notes=notes,
        )

    async def mark_purification_paid(self, entry_id: int, paid_at: datetime | None = None) -> bool:
        return await self._purification.mark_purification_paid(entry_id, paid_at)

    async def get_outstanding_purification(self) -> list[dict[str, Any]]:
        return await self._purification.get_outstanding_purification()

    async def get_purification_totals(self) -> dict[str, float]:
        return await self._purification.get_purification_totals()

    # ── Halal Screenings (delegated to HalalScreeningRepoImpl) ───────

    async def record_halal_screening(
        self,
        *,
        symbol: str,
        asset_class: str,
        source: str,
        decision: str,
        criteria: dict[str, Any] | None = None,
        cache_hit: bool = False,
    ) -> int:
        return await self._halal_screening.record_halal_screening(
            symbol=symbol,
            asset_class=asset_class,
            source=source,
            decision=decision,
            criteria=criteria,
            cache_hit=cache_hit,
        )

    async def get_halal_screening(self, screening_id: int) -> dict[str, Any] | None:
        return await self._halal_screening.get_halal_screening(screening_id)

    # ── LLM Decisions (delegated to LlmDecisionRepoImpl) ────────────

    async def record_decision(
        self,
        provider: str,
        model: str,
        prompt_summary: str | None = None,
        raw_response: str | None = None,
        parsed_action: dict[str, Any] | None = None,
        symbols: list[str] | None = None,
        execution_ms: int | None = None,
        thinking: str | None = None,
        prompt_version: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        cost_usd: float | None = None,
        tool_transcript: list[dict[str, Any]] | None = None,
    ) -> int:
        return await self._llm_decisions.record_decision(
            provider,
            model,
            prompt_summary=prompt_summary,
            raw_response=raw_response,
            parsed_action=parsed_action,
            symbols=symbols,
            execution_ms=execution_ms,
            thinking=thinking,
            prompt_version=prompt_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=cost_usd,
            tool_transcript=tool_transcript,
        )

    async def get_recent_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._llm_decisions.get_recent_decisions(limit)

    async def get_completed_stock_round_trips(
        self, limit: int = 100, lookback_days: int | None = None
    ) -> list[dict[str, Any]]:
        return await self._trades.get_completed_stock_round_trips(
            limit=limit, lookback_days=lookback_days
        )

    # ── Indicator Snapshots (delegated to IndicatorSnapshotRepoImpl) ─

    async def record_indicator_snapshot(
        self,
        *,
        trade_id: int,
        pair: str,
        indicators: dict[str, float],
    ) -> int:
        return await self._indicator_snapshots.record_indicator_snapshot(
            trade_id=trade_id, pair=pair, indicators=indicators
        )

    # ── Strategy Adjustments (delegated to StrategyAdjustmentRepoImpl) ─

    async def record_strategy_adjustment(
        self,
        parameter: str,
        old_value: float | None,
        new_value: float,
        reasoning: str | None = None,
    ) -> int:
        return await self._strategy_adjustments.record_strategy_adjustment(
            parameter, old_value, new_value, reasoning=reasoning
        )

    async def get_latest_strategy_adjustments(self) -> dict[str, float]:
        return await self._strategy_adjustments.get_latest_strategy_adjustments()

    async def get_recent_adjustments(self, limit: int = 20) -> list[dict[str, Any]]:
        return await self._strategy_adjustments.get_recent_adjustments(limit)
