"""LLM self-improvement loop — reviews trades and adjusts strategy parameters."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from halal_trader.db.repository import Repository
from halal_trader.domain.ports import LLMBackend

if TYPE_CHECKING:
    from halal_trader.crypto.strategy import CryptoTradingStrategy

logger = logging.getLogger(__name__)

_SAFE_BOUNDS = {
    "rsi_buy_threshold": (25.0, 45.0),
    "rsi_sell_threshold": (55.0, 80.0),
    "max_position_pct": (0.10, 0.30),
    "stop_loss_pct": (0.003, 0.020),
    "take_profit_pct": (0.005, 0.030),
    "volatile_sl_multiplier": (1.0, 2.5),
}

_STRATEGY_PARAM_MAP = {
    "max_position_pct": "_max_position_pct",
    "stop_loss_pct": "_stop_loss_pct",
    "take_profit_pct": "_take_profit_pct",
}

_NOOP_EPSILON = 1e-6

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
        llm: LLMBackend,
        repo: Repository,
        *,
        strategy: CryptoTradingStrategy | None = None,
        consecutive_loss_trigger: int = 3,
        exec_failure_trigger: int = 10,
    ) -> None:
        self._llm = llm
        self._repo = repo
        self._strategy = strategy
        self._consecutive_loss_trigger = consecutive_loss_trigger
        self._exec_failure_trigger = exec_failure_trigger
        self._active_adjustments: dict[str, float] = {}
        self._pairs_to_avoid: list[str] = []
        self._exec_failures: dict[str, list[str]] = {}
        self._last_review_time: float = 0
        self._review_cooldown = 300  # 5 minutes between reviews

    async def load_from_db(self) -> None:
        """Load previously saved adjustments from the database."""
        try:
            saved = await self._repo.get_latest_strategy_adjustments()
            if saved:
                for param, value in saved.items():
                    if param in _SAFE_BOUNDS:
                        self._active_adjustments[param] = value
                if self._active_adjustments:
                    logger.info("Loaded %d strategy adjustments from DB", len(self._active_adjustments))
                    self._apply_to_strategy()
        except Exception as e:
            logger.debug("Failed to load strategy adjustments from DB: %s", e)

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

    def record_execution_failure(self, pair: str, error_type: str) -> None:
        """Track an execution failure for a pair."""
        failures = self._exec_failures.setdefault(pair, [])
        failures.append(error_type)
        if len(failures) > 50:
            self._exec_failures[pair] = failures[-50:]

    def _get_failure_summary(self) -> str:
        """Summarize execution failures for the review prompt."""
        if not self._exec_failures:
            return ""
        lines = ["=== EXECUTION FAILURES ==="]
        for pair, errors in sorted(self._exec_failures.items()):
            from collections import Counter
            counts = Counter(errors)
            summary = ", ".join(f"{err}: {cnt}" for err, cnt in counts.most_common(5))
            lines.append(f"  {pair}: {len(errors)} failures ({summary})")
        return "\n".join(lines)

    async def should_trigger_review(self) -> bool:
        """Check if conditions warrant a review (losses or repeated failures)."""
        import time as _time
        now = _time.monotonic()
        if now - self._last_review_time < self._review_cooldown:
            return False

        total_exec_failures = sum(len(v) for v in self._exec_failures.values())
        if total_exec_failures >= self._exec_failure_trigger:
            return True

        round_trips = await self._repo.get_completed_round_trips(
            limit=self._consecutive_loss_trigger
        )
        if len(round_trips) < self._consecutive_loss_trigger:
            return False

        recent = round_trips[:self._consecutive_loss_trigger]
        return all(rt["pnl"] < 0 for rt in recent)

    async def review(self, lookback_days: int = 1) -> ReviewResult:
        """Run a self-review session on recent trades."""
        import time as _time
        self._last_review_time = _time.monotonic()

        round_trips = await self._repo.get_completed_round_trips(
            limit=100, lookback_days=lookback_days
        )

        failure_summary = self._get_failure_summary()

        if not round_trips and not failure_summary:
            logger.info("No trades to review")
            return ReviewResult()

        if round_trips:
            trades_text = self._format_trades_for_review(round_trips)
        else:
            trades_text = "No completed trades."

        prompt = f"""\
=== TRADES TO REVIEW ({len(round_trips)} trades from last {lookback_days} day(s)) ===

{trades_text}

{failure_summary}

=== SUMMARY ===
Total trades: {len(round_trips)}
Winners: {sum(1 for rt in round_trips if rt['pnl'] > 0)}
Losers: {sum(1 for rt in round_trips if rt['pnl'] <= 0)}
Total P&L: ${sum(rt['pnl'] for rt in round_trips):+,.2f}

Analyze these trades and execution failures, and suggest improvements.
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
            self._exec_failures.clear()

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

            old_value = self._active_adjustments.get(param)
            if old_value is not None and abs(clamped - old_value) < _NOOP_EPSILON:
                continue

            result.adjustments.append(StrategyAdjustment(
                parameter=param,
                old_value=old_value,
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
            existing = set(self._pairs_to_avoid)
            existing.update(result.pairs_to_avoid)
            self._pairs_to_avoid = list(existing)
            logger.info("Pairs to avoid updated: %s", self._pairs_to_avoid)

        self._apply_to_strategy()

    def _apply_to_strategy(self) -> None:
        """Push active adjustments directly into the strategy instance."""
        if not self._strategy:
            return
        for param, value in self._active_adjustments.items():
            attr = _STRATEGY_PARAM_MAP.get(param)
            if attr and hasattr(self._strategy, attr):
                setattr(self._strategy, attr, value)
                logger.debug("Applied %s = %.4f to strategy", param, value)
