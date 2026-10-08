"""Stocks-side LLM self-improvement loop.

The bot reviews its own closed trades through an LLM and converts
observations into bounded parameter overrides: cooldown, exec-failure
tracking, prompt assembly, then parse/clamp/apply onto the live strategy.

The knob menu is small because the stocks :class:`TradingStrategy`
doesn't expose global SL/TP fallbacks (the LLM emits SL/TP per decision;
there's no ``stop_loss_pct`` instance attribute to override). It has 2
entries:

* ``max_position_pct`` — max share of portfolio in a single position.
* ``daily_loss_limit`` — daily P&L floor before the cycle halts.

A future expansion (e.g. when stocks gets a self-tuning RSI gate)
would just add the new knob name + bounds to ``_STOCK_SAFE_BOUNDS``
and to the JSON schema inside ``_STOCK_SYSTEM_PROMPT``.
"""

from __future__ import annotations

import logging
import time as _time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from halal_trader.core import events
from halal_trader.db.repos import StrategyAdjustmentRepo, TradeRepo
from halal_trader.domain.ports import LLMBackend

if TYPE_CHECKING:
    from halal_trader.trading.strategy import TradingStrategy

logger = logging.getLogger(__name__)

# Clamped adjustment within ε of the current value is dropped on
# parse so the bot doesn't log "no-op" changes every review.
_NOOP_EPSILON = 1e-6


_STOCK_SAFE_BOUNDS: dict[str, tuple[float, float]] = {
    # Stocks strategy default is 0.20; allow ±0.10 of room.
    "max_position_pct": (0.05, 0.30),
    # Stocks strategy default is 0.02 (2%); allow tightening to 0.5%
    # or loosening to 5%. Below 0.5% would trigger nuisance halts;
    # above 5% defeats the daily-loss safeguard.
    "daily_loss_limit": (0.005, 0.05),
}


_STOCK_STRATEGY_PARAM_MAP: dict[str, str] = {
    "max_position_pct": "_max_position_pct",
    "daily_loss_limit": "_daily_loss_limit",
}


_STOCK_SYSTEM_PROMPT = """\
You are reviewing your own stock trading decisions. Your goal is to identify patterns \
in losing trades and suggest concrete parameter adjustments to improve future performance.

Analyze the trades below. Each trade includes:
- The exit price, P&L, and exit reason
- Hold duration

Focus on:
1. What patterns appear in the losing trades? (e.g., trading against the trend, exit timing)
2. Are there symbols that consistently lose money?
3. Is position sizing too aggressive given the win rate?
4. Is the daily loss limit too tight (cutting winners) or too loose (no protection)?

You MUST respond with valid JSON:
{{
  "observations": ["<pattern 1>", "<pattern 2>", ...],
  "parameter_adjustments": {{
    "max_position_pct": <float or null>,
    "daily_loss_limit": <float or null>
  }},
  "symbols_to_avoid": ["<symbol1>", ...],
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
    symbols_to_avoid: list[str] = field(default_factory=list)
    strategy_notes: str = ""


class StockTradeSelfReview:
    """Stocks-side self-review — 2 tunable knobs over Alpaca round-trips.

    Per-decision SL/TP isn't a tunable knob because the
    :class:`TradingStrategy` doesn't carry a global fallback (the
    LLM emits SL/TP per buy decision). If the model wants to
    influence stop-loss behavior, it should be done by adjusting the
    daily-loss-limit floor (which the cycle's risk halt reads) or
    by the LLM emitting tighter stops per decision — not by
    self-tuning a global SL knob that doesn't exist.
    """

    _SAFE_BOUNDS: ClassVar[dict[str, tuple[float, float]]] = _STOCK_SAFE_BOUNDS
    """Knob name → (low, high) clamps. Any knob the LLM suggests outside
    these bounds is clamped to the boundary, not rejected."""

    _STRATEGY_PARAM_MAP: ClassVar[dict[str, str]] = _STOCK_STRATEGY_PARAM_MAP
    """Knob name (matches a ``_SAFE_BOUNDS`` key) → attribute name on the
    live strategy instance. ``_apply_to_strategy`` writes
    ``setattr(strategy, attr, value)`` so the next cycle picks up
    the new value without a restart."""

    _SYSTEM_PROMPT: ClassVar[str] = _STOCK_SYSTEM_PROMPT
    """The JSON schema inside this prompt MUST match ``_SAFE_BOUNDS`` keys —
    out-of-schema knobs the LLM emits land in ``raw["parameter_adjustments"]``
    and get dropped by ``_parse_review`` (the ``param not in _SAFE_BOUNDS``
    guard)."""

    _REVIEW_COOLDOWN_SECONDS: ClassVar[int] = 300
    """Min interval between consecutive reviews."""

    # ── Lifecycle ────────────────────────────────────────────────

    def __init__(
        self,
        llm: LLMBackend,
        *,
        strategy_adjustments: StrategyAdjustmentRepo,
        trades: TradeRepo,
        strategy: TradingStrategy | None = None,
        consecutive_loss_trigger: int = 3,
        exec_failure_trigger: int = 10,
    ) -> None:
        self._llm = llm
        self._trades = trades
        self._strategy_adjustments = strategy_adjustments
        self._strategy = strategy
        self._consecutive_loss_trigger = consecutive_loss_trigger
        self._exec_failure_trigger = exec_failure_trigger
        self._active_adjustments: dict[str, float] = {}
        self._symbols_to_avoid: list[str] = []
        self._exec_failures: dict[str, list[str]] = {}
        self._last_review_time: float = 0

    async def load_from_db(self) -> None:
        """Load previously saved adjustments from the database."""
        try:
            saved = await self._strategy_adjustments.get_latest_strategy_adjustments()
            if saved:
                for param, value in saved.items():
                    if param in self._SAFE_BOUNDS:
                        self._active_adjustments[param] = value
                if self._active_adjustments:
                    logger.info(
                        "Loaded %d strategy adjustments from DB", len(self._active_adjustments)
                    )
                    self._apply_to_strategy()
        except Exception as e:
            logger.debug("Failed to load strategy adjustments from DB: %r", e)

    @property
    def active_adjustments(self) -> dict[str, float]:
        return self._active_adjustments.copy()

    @property
    def symbols_to_avoid(self) -> list[str]:
        return self._symbols_to_avoid.copy()

    def format_adjustments_for_prompt(self) -> str:
        """Format active adjustments as text for the trading prompt."""
        lines = []
        if self._active_adjustments:
            for param, value in self._active_adjustments.items():
                lines.append(f"- {param}: {value}")
        if self._symbols_to_avoid:
            lines.append(f"- Avoid these symbols: {', '.join(self._symbols_to_avoid)}")
        return "\n".join(lines) if lines else ""

    # ── Exec-failure tracking ────────────────────────────────────

    def record_execution_failure(self, symbol: str, error_type: str) -> None:
        """Track an execution failure for a symbol."""
        failures = self._exec_failures.setdefault(symbol, [])
        failures.append(error_type)
        if len(failures) > 50:
            self._exec_failures[symbol] = failures[-50:]

    def _get_failure_summary(self) -> str:
        """Summarize execution failures for the review prompt."""
        if not self._exec_failures:
            return ""
        lines = ["=== EXECUTION FAILURES ==="]
        for symbol, errors in sorted(self._exec_failures.items()):
            counts = Counter(errors)
            summary = ", ".join(f"{err}: {cnt}" for err, cnt in counts.most_common(5))
            lines.append(f"  {symbol}: {len(errors)} failures ({summary})")
        return "\n".join(lines)

    # ── Trigger logic ────────────────────────────────────────────

    async def should_trigger_review(self) -> bool:
        """Check if conditions warrant a review (losses or repeated failures)."""
        now = _time.monotonic()
        if now - self._last_review_time < self._REVIEW_COOLDOWN_SECONDS:
            return False

        total_exec_failures = sum(len(v) for v in self._exec_failures.values())
        if total_exec_failures >= self._exec_failure_trigger:
            return True

        round_trips = await self._fetch_round_trips(
            limit=self._consecutive_loss_trigger, lookback_days=None
        )
        if len(round_trips) < self._consecutive_loss_trigger:
            return False

        recent = round_trips[: self._consecutive_loss_trigger]
        return all(rt["pnl"] < 0 for rt in recent)

    # ── Review orchestration ─────────────────────────────────────

    async def review(self, lookback_days: int = 1) -> ReviewResult:
        """Run a self-review session on recent trades."""
        self._last_review_time = _time.monotonic()

        round_trips = await self._fetch_round_trips(limit=100, lookback_days=lookback_days)
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
Winners: {sum(1 for rt in round_trips if rt["pnl"] > 0)}
Losers: {sum(1 for rt in round_trips if rt["pnl"] <= 0)}
Total P&L: ${sum(rt["pnl"] for rt in round_trips):+,.2f}

Analyze these trades and execution failures, and suggest improvements.
"""

        try:
            raw = await self._llm.generate_json(prompt, system=self._SYSTEM_PROMPT)
            result = self._parse_review(raw)

            for adj in result.adjustments:
                await self._strategy_adjustments.record_strategy_adjustment(
                    parameter=adj.parameter,
                    old_value=adj.old_value,
                    new_value=adj.new_value,
                    reasoning=adj.reasoning,
                )

            # Persist observations so they survive into tomorrow's
            # session prompt. Stored as StrategyAdjustment rows with
            # parameter="self_review_observation" so we avoid a new
            # table / Alembic migration; the strategy reader filters
            # on that exact parameter.
            for obs in result.observations:
                obs_text = (obs or "").strip()
                if not obs_text:
                    continue
                try:
                    await self._strategy_adjustments.record_strategy_adjustment(
                        parameter="self_review_observation",
                        old_value=None,
                        new_value=0.0,
                        reasoning=obs_text[:500],
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to persist observation: %r", exc)

            self._apply_adjustments(result)
            self._exec_failures.clear()

            logger.info(
                "Self-review complete (stock): %d observations, %d adjustments, "
                "%d symbols to avoid",
                len(result.observations),
                len(result.adjustments),
                len(result.symbols_to_avoid),
            )

            return result

        except Exception as e:
            # Mirror the strategy's insufficient_quota handling: this
            # error is non-transient, every retry burns another API
            # call. Push the next-eligible review out by 1 hour
            # (instead of the normal cooldown) so the operator gets
            # one log per hour, not one per 5 minutes.
            err_text = str(e)
            if "insufficient_quota" in err_text or "exceeded your current quota" in err_text:
                self._last_review_time = _time.monotonic() + 3600 - self._REVIEW_COOLDOWN_SECONDS
                logger.critical(
                    "Self-review LLM out of credits — review backed off 1h",
                    extra={"event": events.LLM_INSUFFICIENT_QUOTA},
                )
            else:
                logger.error("Self-review failed: %r", e)
            return ReviewResult()

    # ── Helpers ──────────────────────────────────────────────────

    def _format_trades_for_review(self, round_trips: list[dict[str, Any]]) -> str:
        """Format trades with context for the review prompt.

        Reads ``rt["symbol"]``, as ``get_completed_stock_round_trips`` returns it.
        """
        lines = []
        for i, rt in enumerate(round_trips, 1):
            pnl_label = "WIN" if rt["pnl"] > 0 else "LOSS"
            dur = rt["duration_minutes"]
            dur_str = f"{dur:.0f}m" if dur < 60 else f"{dur / 60:.1f}h"

            lines.append(
                f"Trade #{i} [{pnl_label}]: {rt['symbol']} | "
                f"Entry: ${rt['buy_price']:,.2f} → Exit: ${rt['sell_price']:,.2f} | "
                f"P&L: ${rt['pnl']:+,.2f} ({rt['pnl_pct']:+.2%}) | "
                f"Duration: {dur_str} | Reason: {rt.get('exit_reason', 'unknown')}"
            )

        return "\n".join(lines)

    def _parse_review(self, raw: dict[str, Any]) -> ReviewResult:
        """Parse the LLM review response into a ReviewResult.

        Knobs not in ``_SAFE_BOUNDS`` are silently dropped — this is
        the safety net for a confused LLM that hallucinates extra
        parameter names. Knobs at ``null`` are also dropped.
        """
        result = ReviewResult()
        result.observations = raw.get("observations", [])
        result.symbols_to_avoid = raw.get("symbols_to_avoid", [])
        result.strategy_notes = raw.get("strategy_notes", "")

        param_adjustments = raw.get("parameter_adjustments", {})
        for param, value in param_adjustments.items():
            if value is None or param not in self._SAFE_BOUNDS:
                continue

            low, high = self._SAFE_BOUNDS[param]
            clamped = max(low, min(high, float(value)))

            old_value = self._active_adjustments.get(param)
            if old_value is not None and abs(clamped - old_value) < _NOOP_EPSILON:
                continue

            result.adjustments.append(
                StrategyAdjustment(
                    parameter=param,
                    old_value=old_value,
                    new_value=clamped,
                    reasoning=f"Self-review suggested {param}={value}, clamped to [{low}, {high}]",
                )
            )

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

        if result.symbols_to_avoid:
            existing = set(self._symbols_to_avoid)
            existing.update(result.symbols_to_avoid)
            self._symbols_to_avoid = list(existing)
            logger.info("Symbols to avoid updated: %s", self._symbols_to_avoid)

        self._apply_to_strategy()

    def _apply_to_strategy(self) -> None:
        """Push active adjustments directly into the strategy instance.

        Writes through ``setattr`` against ``_STRATEGY_PARAM_MAP`` —
        the strategy instance must already carry the attribute (the
        check is ``hasattr(strategy, attr)``).
        """
        if not self._strategy:
            return
        for param, value in self._active_adjustments.items():
            attr = self._STRATEGY_PARAM_MAP.get(param)
            if attr and hasattr(self._strategy, attr):
                setattr(self._strategy, attr, value)
                logger.debug("Applied %s = %.4f to strategy", param, value)

    async def _fetch_round_trips(
        self, *, limit: int, lookback_days: int | None
    ) -> list[dict[str, Any]]:
        """Closed round-trips, newest first, in the canonical dict shape
        (``symbol``, ``buy_price``, ``sell_price``, ``pnl``, ``pnl_pct``,
        ``duration_minutes``, ``exit_reason``)."""
        return await self._trades.get_completed_stock_round_trips(
            limit=limit, lookback_days=lookback_days
        )


__all__ = ["StockTradeSelfReview"]
