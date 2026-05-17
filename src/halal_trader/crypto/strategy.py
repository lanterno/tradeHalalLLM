"""Crypto trading strategy — LLM prompt engineering for 1-minute scalping."""

from __future__ import annotations

import logging
import time
from typing import Any

from halal_trader.core.llm.adversarial import apply_review_to_buys, critique_plan
from halal_trader.core.llm.ensemble import EnsembleVariant, run_ensemble, wrap_existing
from halal_trader.core.strategy import BaseStrategy
from halal_trader.crypto.prompts import (
    PROMPT_VERSION as _CRYPTO_PROMPT_VERSION,
)
from halal_trader.crypto.prompts import (
    PromptContext,
    StrategyParams,
    build_prompts,
)
from halal_trader.db.repos import LlmDecisionRepo
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
        repo: LlmDecisionRepo,
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
        attacker_llm: LLMBackend | None = None,
        adversarial_downsize_at: float = 0.45,
        adversarial_skip_at: float = 0.75,
        ensemble_llms: list[LLMBackend] | None = None,
        ensemble_quorum: int = 2,
        ensemble_skip_at: float | None = None,
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
        # Optional adversarial co-bot — disabled by default. When set,
        # the strategy runs a cheap critique LLM on every produced plan
        # and downsizes/skips buys whose counter-thesis is convincing.
        self._attacker_llm = attacker_llm
        self._adv_downsize_at = adversarial_downsize_at
        self._adv_skip_at = adversarial_skip_at
        self.last_adversarial_review = None  # exposed for the dashboard

        # Optional ensemble fan-out — additional LLMs that vote alongside
        # the primary. Disabled by default. When set, every analyze()
        # call runs the primary + ensemble in parallel and consensus
        # quantities replace the primary's. Adversarial review (if any)
        # then runs on the consensus plan.
        self._ensemble_llms = list(ensemble_llms or [])
        self._ensemble_quorum = ensemble_quorum
        self._ensemble_skip_at = ensemble_skip_at
        self.last_ensemble_verdict = None

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

        # `insufficient_quota` (out of OpenAI/Anthropic credits) is non-
        # transient — every retry fails with the same error and burns
        # API attempts. Trip cooldown immediately + log loudly so the
        # operator can top up. Match by error-message substring because
        # the SDK exception class isn't always surfaced through our
        # wrapper layers.
        err_text = str(error)
        if "insufficient_quota" in err_text or "exceeded your current quota" in err_text:
            self._llm_cooldown_until = time.monotonic() + self._llm_cooldown_seconds
            logger.critical(
                "LLM provider account out of credits — top up to resume trading "
                "(cooldown engaged for %ds)",
                self._llm_cooldown_seconds,
                extra={"event": "llm.insufficient_quota"},
            )
            return

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
        microstructure_text: str = "",
        news_text: str = "",
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
            microstructure_text=microstructure_text,
            news_text=news_text,
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

        plan = await self._run_llm_analysis(
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
            prompt_version=_CRYPTO_PROMPT_VERSION.short,
        )

        if self._ensemble_llms and plan.decisions:
            plan = await self._apply_ensemble(plan, ctx, params)

        if self._attacker_llm is not None and plan.decisions:
            plan = await self._apply_adversarial_review(plan, user_prompt)

        return plan

    async def _apply_ensemble(
        self,
        primary_plan: CryptoTradingPlan,
        ctx: PromptContext,
        params: StrategyParams,
    ) -> CryptoTradingPlan:
        """Fan-out to ensemble LLMs and merge with the primary's plan.

        On any error, returns the primary plan unchanged — the ensemble
        is advisory and must never block trading.
        """

        async def _call_for(llm: LLMBackend):
            system, user_prompt = build_prompts(ctx, params)
            # Use the BaseLLM directly (no audit / no repair) — those
            # belong to the primary path. Ensemble votes are consumed
            # by aggregate_plans, not persisted as separate decisions.
            try:
                raw = await llm.generate_json(user_prompt, system=system)
                return CryptoTradingPlan.model_validate(raw)
            except Exception as exc:  # noqa: BLE001
                logger.debug("ensemble variant %s failed: %s", getattr(llm, "model", "?"), exc)
                raise

        variants = [
            EnsembleVariant(
                name=f"primary:{getattr(self._llm, 'model', 'primary')}",
                call=lambda p=primary_plan: wrap_existing(p),
            )
        ]
        for i, alt in enumerate(self._ensemble_llms):

            async def _alt_call(alt=alt):
                return await _call_for(alt)

            variants.append(
                EnsembleVariant(name=f"alt-{i}:{getattr(alt, 'model', 'alt')}", call=_alt_call)
            )

        try:
            verdict = await run_ensemble(
                variants,
                quorum=self._ensemble_quorum,
                skip_quorum_at=self._ensemble_skip_at,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ensemble run failed: %s — keeping primary plan", exc)
            return primary_plan
        self.last_ensemble_verdict = verdict
        consensus = verdict.consensus_plan
        if not isinstance(consensus, CryptoTradingPlan):
            return primary_plan
        # Apply sizing multiplier from agreement to buys only
        if verdict.sizing_multiplier == 0.0:
            return consensus.model_copy(update={"decisions": []})
        if verdict.sizing_multiplier < 1.0 and consensus.decisions:
            new_decisions = []
            for d in consensus.decisions:
                action = d.action.value if hasattr(d.action, "value") else str(d.action)
                if action.lower() == "buy":
                    new_decisions.append(
                        d.model_copy(update={"quantity": d.quantity * verdict.sizing_multiplier})
                    )
                else:
                    new_decisions.append(d)
            consensus = consensus.model_copy(update={"decisions": new_decisions})
        return consensus

    async def _apply_adversarial_review(
        self, plan: CryptoTradingPlan, user_prompt: str
    ) -> CryptoTradingPlan:
        """Run the co-bot critic and shrink/skip buys when it finds a strong
        counter-thesis. Errors are advisory — never block trading on them.
        """
        try:
            review = await critique_plan(
                self._attacker_llm,  # type: ignore[arg-type]
                decisions=plan.decisions,
                market_outlook=plan.market_outlook,
                context_excerpt=user_prompt,
                downsize_at=self._adv_downsize_at,
                skip_at=self._adv_skip_at,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("adversarial review wrapper failed: %s — skipping", exc)
            return plan
        self.last_adversarial_review = review
        if review.recommendation == "proceed":
            return plan
        new_decisions = apply_review_to_buys(plan.decisions, review)
        return plan.model_copy(
            update={
                "decisions": new_decisions,
                "risk_notes": (
                    (plan.risk_notes + " | " if plan.risk_notes else "")
                    + f"adversarial: {review.recommendation} "
                    f"(severity {review.severity:.2f}) — {review.counter_thesis}"
                ),
            }
        )
