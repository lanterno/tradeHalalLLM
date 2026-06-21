"""Data access layer using SQLModel.

Wave D split this monolithic class into per-table mini-repos under
``db/repos/``. The :class:`Repository` here is now a thin delegator
that keeps the legacy flat method surface working — every method
forwards to the mini-repo that owns its table. New code should depend
on the narrowest protocol it needs (``TradeRepo``,
``CryptoTradeRepo``, ``LlmDecisionRepo``, …) and pull the impl from
:class:`RepoBundle` via :meth:`Repository.bundle` or
``RepoBundle.from_engine(engine)``.
"""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.db.models import (
    CryptoTrade,
    Trade,
)

if TYPE_CHECKING:
    from halal_trader.db.repos import RepoBundle


class Repository:
    """Legacy facade — see module docstring; prefer ``RepoBundle``."""

    def __init__(self, engine: AsyncEngine) -> None:
        from halal_trader.db.repos.crypto_trades import CryptoTradeRepoImpl
        from halal_trader.db.repos.daily_recommendations import (
            DailyRecommendationRepoImpl,
        )
        from halal_trader.db.repos.halal_cache import HalalCacheRepoImpl
        from halal_trader.db.repos.halal_screening import HalalScreeningRepoImpl
        from halal_trader.db.repos.indicator_snapshots import IndicatorSnapshotRepoImpl
        from halal_trader.db.repos.llm_decisions import LlmDecisionRepoImpl
        from halal_trader.db.repos.pair_pause import PairPauseRepoImpl
        from halal_trader.db.repos.pnl import PnlRepoImpl
        from halal_trader.db.repos.purification import PurificationRepoImpl
        from halal_trader.db.repos.research_jobs import ResearchJobRepoImpl
        from halal_trader.db.repos.runtime_config import RuntimeConfigRepoImpl
        from halal_trader.db.repos.stock_halal_cache import StockHalalCacheRepoImpl
        from halal_trader.db.repos.stock_pnl import StockPnlRepoImpl
        from halal_trader.db.repos.strategy_adjustments import StrategyAdjustmentRepoImpl
        from halal_trader.db.repos.trades import TradeRepoImpl
        from halal_trader.db.repos.web_audit import WebAuditRepoImpl

        self._engine = engine
        # Per-table mini-repos extracted under Wave D of the cleanup
        # roadmap. The legacy ``Repository`` keeps its method surface as
        # thin delegators so call sites migrate incrementally.
        self._web_audit = WebAuditRepoImpl(engine)
        self._runtime_config = RuntimeConfigRepoImpl(engine)
        self._pair_pause = PairPauseRepoImpl(engine)
        self._purification = PurificationRepoImpl(engine)
        self._halal_screening = HalalScreeningRepoImpl(engine)
        self._research_jobs = ResearchJobRepoImpl(engine)
        self._daily_recommendations = DailyRecommendationRepoImpl(engine)
        self._indicator_snapshots = IndicatorSnapshotRepoImpl(engine)
        self._llm_decisions = LlmDecisionRepoImpl(engine)
        self._pnl = PnlRepoImpl(engine)
        self._halal_cache = HalalCacheRepoImpl(engine)
        self._trades = TradeRepoImpl(engine)
        self._crypto_trades = CryptoTradeRepoImpl(engine)
        self._strategy_adjustments = StrategyAdjustmentRepoImpl(engine)
        self._stock_halal_cache = StockHalalCacheRepoImpl(engine)
        self._stock_pnl = StockPnlRepoImpl(engine)

    @property
    def bundle(self) -> "RepoBundle":
        """Expose the mini-repos as a typed ``RepoBundle``.

        Migration aid: code that wants the narrower per-table
        protocols can take ``repo.bundle.crypto_trades`` instead of
        the full ``Repository`` flat surface. The exposed instances
        are the *same* objects this Repository delegates to, so there
        is no double-construction cost.
        """
        from halal_trader.db.repos import RepoBundle

        return RepoBundle(
            trades=self._trades,
            crypto_trades=self._crypto_trades,
            pnl=self._pnl,
            stock_pnl=self._stock_pnl,
            halal_cache=self._halal_cache,
            stock_halal_cache=self._stock_halal_cache,
            halal_screening=self._halal_screening,
            runtime_config=self._runtime_config,
            research_jobs=self._research_jobs,
            daily_recommendations=self._daily_recommendations,
            web_audit=self._web_audit,
            indicator_snapshots=self._indicator_snapshots,
            llm_decisions=self._llm_decisions,
            purification=self._purification,
            pair_pauses=self._pair_pause,
            strategy_adjustments=self._strategy_adjustments,
        )

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
        return await self._trades.close_open_trades_for_symbol(
            symbol, exit_price, exit_reason
        )

    async def update_stock_trade_stop_loss(self, trade_id: int, new_stop_loss: float) -> None:
        await self._trades.update_stock_trade_stop_loss(trade_id, new_stop_loss)

    # ── Stock Daily P&L ────────────────────────────────────────

    async def start_day(self, starting_equity: float) -> None:
        await self._stock_pnl.start_day(starting_equity)

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

    # ── Research jobs (delegated to ResearchJobRepoImpl) ────────────

    async def enqueue_research_job(
        self, *, kind: str, params: dict[str, Any], name: str | None = None
    ) -> int:
        return await self._research_jobs.enqueue_research_job(kind=kind, params=params, name=name)

    async def update_research_job(
        self,
        job_id: int,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        await self._research_jobs.update_research_job(
            job_id, status=status, result=result, error=error
        )

    async def get_research_job(self, job_id: int) -> dict[str, Any] | None:
        return await self._research_jobs.get_research_job(job_id)

    async def list_research_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._research_jobs.list_research_jobs(limit)

    # ── Daily recommendation (delegated to DailyRecommendationRepoImpl) ──

    async def save_recommendation(self, rec: dict[str, Any]) -> int:
        return await self._daily_recommendations.save_recommendation(rec)

    async def get_latest_recommendation(self) -> dict[str, Any] | None:
        return await self._daily_recommendations.get_latest_recommendation()

    async def get_recent_recommendations(
        self, limit: int = 30
    ) -> list[dict[str, Any]]:
        return await self._daily_recommendations.get_recent_recommendations(limit)

    async def get_recommendations_to_score(
        self, limit: int = 500
    ) -> list[dict[str, Any]]:
        return await self._daily_recommendations.get_recommendations_to_score(limit)

    async def update_recommendation_outcome(
        self, rec_id: int, **fields: Any
    ) -> bool:
        return await self._daily_recommendations.update_recommendation_outcome(
            rec_id, **fields
        )

    async def pin_research_job(self, job_id: int, pinned: bool) -> bool:
        return await self._research_jobs.pin_research_job(job_id, pinned)

    # ── Runtime config overlay (delegated to RuntimeConfigRepoImpl) ──

    async def set_runtime_config(self, key: str, value: Any, *, set_by: str | None = None) -> None:
        await self._runtime_config.set_runtime_config(key, value, set_by=set_by)

    async def delete_runtime_config(self, key: str) -> bool:
        return await self._runtime_config.delete_runtime_config(key)

    async def list_runtime_config(self) -> dict[str, Any]:
        return await self._runtime_config.list_runtime_config()

    # ── Per-pair operator pauses (delegated to PairPauseRepoImpl) ────

    async def pause_pair(
        self, pair: str, *, set_by: str | None = None, reason: str | None = None
    ) -> None:
        await self._pair_pause.pause_pair(pair, set_by=set_by, reason=reason)

    async def resume_pair(self, pair: str) -> bool:
        return await self._pair_pause.resume_pair(pair)

    async def get_paused_pairs(self) -> set[str]:
        return await self._pair_pause.get_paused_pairs()

    async def list_pair_pauses(self) -> list[dict[str, Any]]:
        return await self._pair_pause.list_pair_pauses()

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

    # ── Crypto Trades ──────────────────────────────────────────

    # ── Crypto Trades (delegated to CryptoTradeRepoImpl) ────────────

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
        return await self._crypto_trades.record_crypto_trade(
            pair,
            side,
            quantity,
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

    async def update_crypto_trade_stop_loss(self, trade_id: int, new_stop_loss: float) -> None:
        await self._crypto_trades.update_crypto_trade_stop_loss(trade_id, new_stop_loss)

    async def close_crypto_trade(self, trade_id: int, exit_price: float, exit_reason: str) -> None:
        await self._crypto_trades.close_crypto_trade(trade_id, exit_price, exit_reason)

    async def get_today_crypto_trades(self) -> list[dict[str, Any]]:
        return await self._crypto_trades.get_today_crypto_trades()

    async def get_open_crypto_trades(self) -> list[CryptoTrade]:
        return await self._crypto_trades.get_open_crypto_trades()

    async def get_open_crypto_trades_for_pair(self, pair: str) -> list[CryptoTrade]:
        return await self._crypto_trades.get_open_crypto_trades_for_pair(pair)

    async def close_open_crypto_trades_for_pair(
        self,
        pair: str,
        exit_price: float,
        exit_reason: str,
        exclude_id: int | None = None,
    ) -> int:
        return await self._crypto_trades.close_open_crypto_trades_for_pair(
            pair, exit_price, exit_reason, exclude_id=exclude_id
        )

    async def get_recent_crypto_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._crypto_trades.get_recent_crypto_trades(limit)

    async def get_completed_round_trips(
        self, limit: int = 100, lookback_days: int | None = None
    ) -> list[dict[str, Any]]:
        return await self._crypto_trades.get_completed_round_trips(
            limit=limit, lookback_days=lookback_days
        )

    async def get_completed_stock_round_trips(
        self, limit: int = 100, lookback_days: int | None = None
    ) -> list[dict[str, Any]]:
        return await self._trades.get_completed_stock_round_trips(
            limit=limit, lookback_days=lookback_days
        )

    # ── Crypto Daily P&L (delegated to PnlRepoImpl) ────────────────

    async def start_crypto_day(self, starting_equity: float) -> None:
        await self._pnl.start_crypto_day(starting_equity)

    async def end_crypto_day(
        self, ending_equity: float, realized_pnl: float, trades_count: int
    ) -> None:
        await self._pnl.end_crypto_day(
            ending_equity=ending_equity,
            realized_pnl=realized_pnl,
            trades_count=trades_count,
        )

    async def get_crypto_pnl_history(self, limit: int = 30) -> list[dict[str, Any]]:
        return await self._pnl.get_crypto_pnl_history(limit)

    # ── Crypto Halal Cache (delegated to HalalCacheRepoImpl) ───────

    async def cache_crypto_halal_status(
        self,
        symbol: str,
        compliance: str,
        category: str | None = None,
        market_cap: float | None = None,
        screening_criteria: dict[str, Any] | None = None,
    ) -> None:
        await self._halal_cache.cache_crypto_halal_status(
            symbol=symbol,
            compliance=compliance,
            category=category,
            market_cap=market_cap,
            screening_criteria=screening_criteria,
        )

    async def get_crypto_halal_status(self, symbol: str) -> str | None:
        return await self._halal_cache.get_crypto_halal_status(symbol)

    async def get_crypto_halal_symbols(self) -> list[str]:
        return await self._halal_cache.get_crypto_halal_symbols()

    async def is_crypto_cache_fresh(self, max_age_hours: int = 24) -> bool:
        return await self._halal_cache.is_crypto_cache_fresh(max_age_hours)

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

    async def label_indicator_snapshot(self, trade_id: int, label: int, return_pct: float) -> None:
        await self._indicator_snapshots.label_indicator_snapshot(trade_id, label, return_pct)

    async def get_labeled_snapshots(self, min_samples: int = 50) -> list[dict[str, Any]]:
        return await self._indicator_snapshots.get_labeled_snapshots(min_samples)

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
