"""Day-trading strategy with prompt engineering for LLM decision-making."""

from __future__ import annotations

import logging
from typing import Any

from halal_trader.core.llm.ensemble import EnsembleVariant, run_ensemble, wrap_existing
from halal_trader.core.llm.prompts import register as _register_prompt
from halal_trader.core.strategy import AgentConfig, BaseStrategy
from halal_trader.db.repos import LlmDecisionRepo
from halal_trader.domain.models import Account, Position, TradingPlan
from halal_trader.domain.ports import LLMBackend

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are an expert intraday stock trader AI. Your job is to analyze market data \
and make precise buy/sell decisions to achieve at least 1% daily portfolio return.

RULES:
1. You ONLY trade stocks from the provided halal-compliant list.
2. You make ONLY intraday trades — all positions must be closeable by market close.
3. You optimize for high-probability short-term momentum trades.
4. Each trade must have a clear reasoning based on the data provided.
5. You manage risk: no single position should exceed {max_position_pct:.0%} of the portfolio.
6. Current daily loss limit is {daily_loss_limit:.0%} — if losses approach this, be conservative.
7. Target daily return: {daily_return_target:.0%}.
8. Maximum simultaneous open positions: {max_positions}.

STRATEGY GUIDELINES:
- Look for stocks with strong pre-market/intraday momentum.
- Consider volume spikes as entry signals.
- Use support/resistance from recent price bars.
- Prefer liquid, large-cap stocks for easier fills.
- Set mental stop-losses for every trade.
- If the market outlook is uncertain, it is OK to HOLD and not trade.

You MUST respond with valid JSON matching this exact schema:
{{
  "decisions": [
    {{
      "action": "buy" | "sell" | "hold",
      "symbol": "TICKER",
      "quantity": <integer>,
      "confidence": <float 0-1>,
      "reasoning": "<brief explanation>",
      "target_price": <float or null>,
      "stop_loss": <float or null>
    }}
  ],
  "market_outlook": "<1-2 sentence market assessment>",
  "risk_notes": "<any risk concerns>"
}}

If there are no good trades, return an empty decisions list with your market outlook.
"""

USER_PROMPT_TEMPLATE = """\
=== PORTFOLIO STATUS ===
Buying Power: ${buying_power:,.2f}
Portfolio Value: ${portfolio_value:,.2f}
Cash: ${cash:,.2f}
Today's P&L: ${today_pnl:+,.2f} ({today_pnl_pct:+.2%})

=== CURRENT POSITIONS ===
{positions_text}

=== HALAL-COMPLIANT STOCK UNIVERSE ===
{halal_symbols}

=== MARKET DATA (Snapshots) ===
{snapshots_text}

=== RECENT PRICE BARS (5-day daily) ===
{bars_text}

=== SENTIMENT ANALYSIS ===
{sentiment_text}

=== PORTFOLIO RISK ===
{risk_text}

=== MARKET REGIME ===
{regime_text}

=== ML SIGNALS ===
{ml_signals_text}

=== MULTI-TIMEFRAME ANALYSIS ===
{timeframe_text}

=== RECENT CATALYSTS (news / earnings / insider) ===
{catalysts_text}

Based on this data, what trades should I make right now? \
Remember: optimize for 1%+ daily return with proper risk management.
"""


# Register the static templates so every LlmDecision row records exactly
# which template version produced it. Editing either bumps the hash.
PROMPT_VERSION = _register_prompt("trading.strategy.system", SYSTEM_PROMPT)
USER_PROMPT_VERSION = _register_prompt("trading.strategy.user", USER_PROMPT_TEMPLATE)


def _format_positions(positions: list[Position]) -> str:
    if not positions:
        return "No open positions."
    lines = []
    for p in positions:
        lines.append(
            f"  {p.symbol}: {p.qty} shares @ "
            f"${p.avg_entry_price:.2f} | "
            f"Current: ${p.current_price:.2f} | "
            f"P&L: ${p.unrealized_pl:+.2f} ({p.unrealized_plpc:+.2%})"
        )
    return "\n".join(lines)


def _format_snapshots(snapshots: dict[str, Any]) -> str:
    if not snapshots:
        return "No snapshot data available."
    lines = []
    if isinstance(snapshots, dict):
        for sym, data in snapshots.items():
            if isinstance(data, dict):
                price = data.get("latest_trade", {}).get("price", "N/A")
                bid = data.get("latest_quote", {}).get("bid_price", "N/A")
                ask = data.get("latest_quote", {}).get("ask_price", "N/A")
                vol = data.get("daily_bar", {}).get("volume", "N/A")
                lines.append(f"  {sym}: Price=${price} Bid=${bid} Ask=${ask} Vol={vol}")
            else:
                lines.append(f"  {sym}: {data}")
    else:
        lines.append(str(snapshots))
    return "\n".join(lines) if lines else str(snapshots)


def _format_bars(bars: dict[str, Any]) -> str:
    if not bars:
        return "No bar data available."
    lines = []
    if isinstance(bars, dict):
        for sym, data in bars.items():
            lines.append(f"  {sym}:")
            if isinstance(data, list):
                for bar in data[-5:]:
                    ts = bar.get("timestamp", "")
                    lines.append(
                        f"    {ts}: O={bar.get('open', 0):.2f} H={bar.get('high', 0):.2f} "
                        f"L={bar.get('low', 0):.2f} "
                        f"C={bar.get('close', 0):.2f} V={bar.get('volume', 0)}"
                    )
            else:
                lines.append(f"    {data}")
    else:
        lines.append(str(bars))
    return "\n".join(lines) if lines else str(bars)


class TradingStrategy(BaseStrategy):
    """Stock trading strategy — LLM-based intraday decisions."""

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
        attacker_llm: LLMBackend | None = None,
        adversarial_downsize_at: float = 0.45,
        adversarial_skip_at: float = 0.75,
        ensemble_llms: list[LLMBackend] | None = None,
        ensemble_quorum: int = 2,
        ensemble_skip_at: float | None = None,
        agentic_enabled: bool = False,
        agentic_max_turns: int = 5,
        agentic_max_seconds: float = 30.0,
        agentic_hub: Any | None = None,
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
        # Optional adversarial co-bot — mirrors the crypto wiring.
        self._attacker_llm = attacker_llm
        self._adv_downsize_at = adversarial_downsize_at
        self._adv_skip_at = adversarial_skip_at
        self.last_adversarial_review = None

        # Optional ensemble fan-out — mirrors crypto. Empty = disabled.
        # When set, every analyze() call runs the primary + ensemble in
        # parallel; consensus quantities replace the primary's. Adversarial
        # review (if any) then runs on the consensus plan.
        self._ensemble_llms = list(ensemble_llms or [])
        self._ensemble_quorum = ensemble_quorum
        self._ensemble_skip_at = ensemble_skip_at
        self.last_ensemble_verdict = None

        # Wave H — stocks-side agentic mode. Two asset-agnostic tools:
        # query_rag (analogous past rationales) and
        # query_regime_memory (analogous past regimes). The crypto-side
        # analyze_pair / compute_var_95 tools have no clean stocks
        # equivalent today and are intentionally omitted.
        self._agentic_enabled = agentic_enabled
        self._agentic_max_turns = agentic_max_turns
        self._agentic_max_seconds = agentic_max_seconds
        self._agentic_hub = agentic_hub
        self.last_agent_transcript: list[dict[str, Any]] | None = None

    async def analyze(
        self,
        account: Account,
        positions: list[Position],
        halal_symbols: list[str],
        snapshots: dict[str, Any],
        bars: dict[str, Any],
        today_pnl: float = 0.0,
        sentiment_text: str = "Sentiment data: not available",
        risk_text: str = "",
        catalysts_text: str = "",
        regime_text: str = "",
        ml_signals_text: str = "",
        timeframe_text: str = "",
    ) -> TradingPlan:
        portfolio_value = account.portfolio_value or account.equity or 100000
        today_pnl_pct = today_pnl / portfolio_value if portfolio_value else 0

        system = SYSTEM_PROMPT.format(
            max_position_pct=self._max_position_pct,
            daily_loss_limit=self._daily_loss_limit,
            daily_return_target=self._daily_return_target,
            max_positions=self._max_simultaneous_positions,
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            buying_power=account.buying_power,
            portfolio_value=portfolio_value,
            cash=account.cash,
            today_pnl=today_pnl,
            today_pnl_pct=today_pnl_pct,
            positions_text=_format_positions(positions),
            halal_symbols=", ".join(halal_symbols),
            snapshots_text=_format_snapshots(snapshots),
            bars_text=_format_bars(bars),
            sentiment_text=sentiment_text,
            risk_text=risk_text or "No portfolio risk data available.",
            regime_text=regime_text or "No regime data available.",
            ml_signals_text=ml_signals_text or "No ML signals available.",
            timeframe_text=timeframe_text or "No multi-timeframe data available.",
            catalysts_text=catalysts_text or "No recent catalysts.",
        )

        from halal_trader.core.llm.tools import (
            QUERY_RAG_TOOL,
            QUERY_REGIME_MEMORY_TOOL,
            SUBMIT_DECISIONS_TOOL,
        )

        agent_cfg: AgentConfig | None = None
        if self._agentic_enabled:
            from halal_trader.trading.agent_tools import build_agent_handlers

            agent_cfg = AgentConfig(
                tools=[
                    QUERY_RAG_TOOL,
                    QUERY_REGIME_MEMORY_TOOL,
                    SUBMIT_DECISIONS_TOOL,
                ],
                handlers=build_agent_handlers(hub=self._agentic_hub),
                terminal_tool="submit_decisions",
                max_turns=self._agentic_max_turns,
                max_seconds=self._agentic_max_seconds,
            )

        plan = await self._run_llm_analysis(
            system,
            user_prompt,
            prompt_summary=(
                f"Analyzed {len(halal_symbols)} halal symbols, "
                f"{len(positions)} positions, buying_power=${account.buying_power}"
            ),
            validate=lambda raw: TradingPlan.model_validate(raw),
            make_empty=lambda msg: TradingPlan(
                market_outlook="Analysis failed — holding positions",
                risk_notes=msg,
            ),
            extract_symbols=lambda p: [d.symbol for d in p.decisions],
            count_actions=lambda p: {
                "buys": len(p.buys),
                "sells": len(p.sells),
                "holds": len(p.holds),
            },
            prompt_version=PROMPT_VERSION.short,
            tool=SUBMIT_DECISIONS_TOOL,
            agent=agent_cfg,
        )

        if self._ensemble_llms and plan.decisions:
            plan = await self._apply_ensemble(plan, system, user_prompt)

        if self._attacker_llm is not None and plan.decisions:
            plan = await self._apply_adversarial_review(plan, user_prompt)

        return plan

    async def _apply_ensemble(
        self, primary_plan: TradingPlan, system: str, user_prompt: str
    ) -> TradingPlan:
        """Fan-out to ensemble LLMs and merge with the primary's plan.

        On any error, returns the primary plan unchanged — the ensemble
        is advisory and must never block trading.
        """

        async def _call_for(llm: LLMBackend) -> TradingPlan:
            try:
                raw = await llm.generate_json(user_prompt, system=system)
                return TradingPlan.model_validate(raw)
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

            async def _alt_call(alt: LLMBackend = alt) -> TradingPlan:
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
            logger.warning("stocks ensemble run failed: %s — keeping primary plan", exc)
            return primary_plan
        self.last_ensemble_verdict = verdict
        consensus = verdict.consensus_plan
        if not isinstance(consensus, TradingPlan):
            return primary_plan
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

    async def _apply_adversarial_review(self, plan: TradingPlan, user_prompt: str) -> TradingPlan:
        """Run the co-bot critic and shrink/skip buys when convincing."""
        from halal_trader.core.llm.adversarial import (
            apply_review_to_buys,
            critique_plan,
        )

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
            logger.warning("stocks adversarial review failed: %s", exc)
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
