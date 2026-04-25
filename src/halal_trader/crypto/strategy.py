"""Crypto trading strategy — LLM prompt engineering for 1-minute scalping."""

from __future__ import annotations

import logging
import time
from typing import Any

from halal_trader.core.strategy import BaseStrategy
from halal_trader.crypto.prompts import PromptContext, StrategyParams, build_prompts
from halal_trader.db.repository import Repository
from halal_trader.domain.models import (
    CryptoAccount,
    CryptoTradingPlan,
    Kline,
)
from halal_trader.domain.ports import LLMBackend

logger = logging.getLogger(__name__)


class CryptoTradingStrategy(BaseStrategy):
    """Crypto scalping strategy with LLM circuit breaker."""

    def __init__(
        self,
        llm: LLMBackend,
        repo: Repository,
        *,
        llm_provider_name: str,
        max_position_pct: float,
        daily_loss_limit: float,
        daily_return_target: float,
        max_simultaneous_positions: int,
        llm_failure_threshold: int = 5,
        llm_cooldown_seconds: int = 600,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float = 0.02,
    ) -> None:
        super().__init__(
            llm,
            repo,
            llm_provider_name=llm_provider_name,
            max_position_pct=max_position_pct,
            daily_loss_limit=daily_loss_limit,
            daily_return_target=daily_return_target,
            max_simultaneous_positions=max_simultaneous_positions,
        )
        self._consecutive_llm_failures = 0
        self._llm_cooldown_until: float = 0
        self._llm_failure_threshold = llm_failure_threshold
        self._llm_cooldown_seconds = llm_cooldown_seconds
        self._stop_loss_pct = stop_loss_pct
        self._take_profit_pct = take_profit_pct

    def _on_llm_success(self) -> None:
        self._consecutive_llm_failures = 0

    def _on_llm_failure(self, error: Exception, elapsed_ms: int, _prefix: str) -> None:
        self._consecutive_llm_failures += 1
        logger.error(
            "Crypto LLM analysis failed after %dms (%d consecutive): %s",
            elapsed_ms,
            self._consecutive_llm_failures,
            error,
        )
        if self._consecutive_llm_failures >= self._llm_failure_threshold:
            self._llm_cooldown_until = time.monotonic() + self._llm_cooldown_seconds
            logger.warning(
                "LLM failed %d times consecutively — entering %ds cooldown",
                self._consecutive_llm_failures,
                self._llm_cooldown_seconds,
            )

    async def analyze(
        self,
        account: CryptoAccount,
        positions_text: str,
        halal_pairs: list[str],
        klines_by_symbol: dict[str, list[Kline]],
        orderbooks: dict[str, dict[str, Any]],
        today_pnl: float = 0.0,
        performance_text: str = "",
        sentiment_text: str = "",
        timeframe_text: str = "",
        ml_signals_text: str = "",
        regime_text: str = "",
        active_adjustments: str = "",
        exchange_rules_text: str = "",
        indicators_cache: dict[str, dict] | None = None,
        open_position_count: int = 0,
        risk_text: str = "",
    ) -> CryptoTradingPlan:
        now = time.monotonic()
        if now < self._llm_cooldown_until:
            remaining = int(self._llm_cooldown_until - now)
            logger.warning("LLM in cooldown (%ds remaining) — holding positions", remaining)
            return CryptoTradingPlan(
                market_outlook="LLM cooldown active — holding",
                risk_notes=(
                    f"Cooldown for {remaining}s after "
                    f"{self._consecutive_llm_failures} consecutive failures"
                ),
            )

        ctx = PromptContext(
            account=account,
            positions_text=positions_text,
            halal_pairs=halal_pairs,
            klines_by_symbol=klines_by_symbol,
            orderbooks=orderbooks,
            today_pnl=today_pnl,
            performance_text=performance_text,
            sentiment_text=sentiment_text,
            timeframe_text=timeframe_text,
            ml_signals_text=ml_signals_text,
            regime_text=regime_text,
            active_adjustments=active_adjustments,
            exchange_rules_text=exchange_rules_text,
            indicators_cache=indicators_cache,
            open_position_count=open_position_count,
            risk_text=risk_text,
        )
        params = StrategyParams(
            max_position_pct=self._max_position_pct,
            daily_loss_limit=self._daily_loss_limit,
            daily_return_target=self._daily_return_target,
            max_positions=self._max_simultaneous_positions,
            stop_loss_pct=self._stop_loss_pct,
            take_profit_pct=self._take_profit_pct,
        )
        system, user_prompt = build_prompts(ctx, params)

        return await self._run_llm_analysis(
            system,
            user_prompt,
            prompt_summary=(
                f"Crypto: analyzed {len(halal_pairs)} halal pairs, "
                f"balance=${account.total_balance_usdt:.2f}"
            ),
            validate=lambda raw: CryptoTradingPlan.model_validate(raw),
            make_empty=lambda msg: CryptoTradingPlan(
                market_outlook="Analysis failed — holding positions",
                risk_notes=msg,
            ),
            extract_symbols=lambda p: [d.symbol for d in p.decisions],
            count_actions=lambda p: {
                "buys": len(p.buys),
                "sells": len(p.sells),
                "holds": len(p.holds),
            },
            log_prefix="Crypto",
        )
