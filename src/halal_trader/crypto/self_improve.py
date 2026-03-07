"""LLM self-improvement loop — reviews trades and adjusts strategy parameters."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from halal_trader.db.repository import Repository
from halal_trader.domain.ports import LLMProvider

logger = logging.getLogger(__name__)

_SAFE_BOUNDS = {
    "rsi_buy_threshold": (25.0, 45.0),
    "rsi_sell_threshold": (55.0, 80.0),
    "max_position_pct": (0.10, 0.30),
    "stop_loss_pct": (0.003, 0.020),
    "take_profit_pct": (0.005, 0.030),
    "volatile_sl_multiplier": (1.0, 2.5),
}

_REVIEW_SYSTEM_PROMPT = """\
You are reviewing your own crypto trading decisions. Your goal is to identify patterns \
in losing trades and suggest concrete parameter adjustments to improve future performance.

Analyze the trades below. Each trade includes:
- The indicators at entry time
- The exit price, P&L, and exit reason
- Hold duration

Focus on:
1. What patterns appear in the losing trades? (e.g., trading against the trend, SL too tight)
2. Are there pairs that consistently lose money?
3. Are there indicator thresholds that should be adjusted?
4. What market conditions led to losses?

You MUST respond with valid JSON:
{{
  "observations": ["<pattern 1>", "<pattern 2>", ...],
  "parameter_adjustments": {{
    "rsi_buy_threshold": <float or null>,
    "rsi_sell_threshold": <float or null>,
    "max_position_pct": <float or null>,
    "stop_loss_pct": <float or null>,
    "take_profit_pct": <float or null>,
    "volatile_sl_multiplier": <float or null>
  }},
  "pairs_to_avoid": ["<pair1>", ...],
  "strategy_notes": "<overall strategy recommendation>"
}}

Only suggest adjustments you are confident about. Use null for parameters that don't need changing.
"""


@dataclass
class StrategyAdjustment:
    """A single parameter adjustment with reasoning."""

    parameter: str
    old_value: float | None
    new_value: float
    reasoning: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ReviewResult:
    """Result of a self-review session."""

    observations: list[str] = field(default_factory=list)
    adjustments: list[StrategyAdjustment] = field(default_factory=list)
    pairs_to_avoid: list[str] = field(default_factory=list)
    strategy_notes: str = ""


class TradeSelfReview:
    """Reviews closed trades and suggests strategy adjustments."""

    def __init__(
        self,
        llm: LLMProvider,
        repo: Repository,
        *,
        consecutive_loss_trigger: int = 3,
    ) -> None:
        self._llm = llm
        self._repo = repo
        self._consecutive_loss_trigger = consecutive_loss_trigger
        self._active_adjustments: dict[str, float] = {}
        self._pairs_to_avoid: list[str] = []

    @property
    def active_adjustments(self) -> dict[str, float]:
        return self._active_adjustments.copy()

    @property
    def pairs_to_avoid(self) -> list[str]:
        return self._pairs_to_avoid.copy()

    def format_adjustments_for_prompt(self) -> str:
        """Format active adjustments as text for the trading prompt."""
        lines = []
        if self._active_adjustments:
            for param, value in self._active_adjustments.items():
                lines.append(f"- {param}: {value}")
        if self._pairs_to_avoid:
            lines.append(f"- Avoid these pairs: {', '.join(self._pairs_to_avoid)}")
        return "\n".join(lines) if lines else ""

    async def should_trigger_review(self) -> bool:
        """Check if conditions warrant a review (consecutive losses)."""
        round_trips = await self._repo.get_completed_round_trips(
            limit=self._consecutive_loss_trigger
        )
        if len(round_trips) < self._consecutive_loss_trigger:
            return False

        recent = round_trips[:self._consecutive_loss_trigger]
        return all(rt["pnl"] < 0 for rt in recent)

    async def review(self, lookback_days: int = 1) -> ReviewResult:
        """Run a self-review session on recent trades."""
        round_trips = await self._repo.get_completed_round_trips(
            limit=100, lookback_days=lookback_days
        )

        if not round_trips:
            logger.info("No trades to review")
            return ReviewResult()

        trades_text = self._format_trades_for_review(round_trips)

        prompt = f"""\
=== TRADES TO REVIEW ({len(round_trips)} trades from last {lookback_days} day(s)) ===

{trades_text}

=== SUMMARY ===
Total trades: {len(round_trips)}
Winners: {sum(1 for rt in round_trips if rt['pnl'] > 0)}
Losers: {sum(1 for rt in round_trips if rt['pnl'] <= 0)}
Total P&L: ${sum(rt['pnl'] for rt in round_trips):+,.2f}

Analyze these trades and suggest improvements.
"""

        try:
            raw = await self._llm.generate_json(prompt, system=_REVIEW_SYSTEM_PROMPT)
            result = self._parse_review(raw)

            for adj in result.adjustments:
                await self._repo.record_strategy_adjustment(
                    parameter=adj.parameter,
                    old_value=adj.old_value,
                    new_value=adj.new_value,
                    reasoning=adj.reasoning,
                )

            self._apply_adjustments(result)

            logger.info(
                "Self-review complete: %d observations, %d adjustments, %d pairs to avoid",
                len(result.observations),
                len(result.adjustments),
                len(result.pairs_to_avoid),
            )

            return result

        except Exception as e:
            logger.error("Self-review failed: %s", e)
            return ReviewResult()

    def _format_trades_for_review(self, round_trips: list[dict[str, Any]]) -> str:
        """Format trades with context for the review prompt."""
        lines = []
        for i, rt in enumerate(round_trips, 1):
            pnl_label = "WIN" if rt["pnl"] > 0 else "LOSS"
            dur = rt["duration_minutes"]
            dur_str = f"{dur:.0f}m" if dur < 60 else f"{dur / 60:.1f}h"

            lines.append(
                f"Trade #{i} [{pnl_label}]: {rt['pair']} | "
                f"Entry: ${rt['buy_price']:,.2f} → Exit: ${rt['sell_price']:,.2f} | "
                f"P&L: ${rt['pnl']:+,.2f} ({rt['pnl_pct']:+.2%}) | "
                f"Duration: {dur_str} | Reason: {rt.get('exit_reason', 'unknown')}"
            )

        return "\n".join(lines)

    def _parse_review(self, raw: dict[str, Any]) -> ReviewResult:
        """Parse the LLM review response into a ReviewResult."""
        result = ReviewResult()
        result.observations = raw.get("observations", [])
        result.pairs_to_avoid = raw.get("pairs_to_avoid", [])
        result.strategy_notes = raw.get("strategy_notes", "")

        param_adjustments = raw.get("parameter_adjustments", {})
        for param, value in param_adjustments.items():
            if value is None or param not in _SAFE_BOUNDS:
                continue

            low, high = _SAFE_BOUNDS[param]
            clamped = max(low, min(high, float(value)))

            result.adjustments.append(StrategyAdjustment(
                parameter=param,
                old_value=self._active_adjustments.get(param),
                new_value=clamped,
                reasoning=f"Self-review suggested {param}={value}, clamped to [{low}, {high}]",
            ))

        return result

    def _apply_adjustments(self, result: ReviewResult) -> None:
        """Apply validated adjustments to the active state."""
        for adj in result.adjustments:
            self._active_adjustments[adj.parameter] = adj.new_value
            logger.info(
                "Strategy adjusted: %s = %.4f (was: %s)",
                adj.parameter,
                adj.new_value,
                adj.old_value,
            )

        if result.pairs_to_avoid:
            self._pairs_to_avoid = result.pairs_to_avoid
            logger.info("Pairs to avoid updated: %s", result.pairs_to_avoid)
