"""Asset-agnostic core for the LLM self-improvement loop.

Crypto and stocks both review their own closed trades through an LLM
and convert observations into bounded parameter overrides. The
orchestration (cooldown, exec-failure tracking, prompt assembly,
parse/clamp/apply) is identical; only three things vary:

1. **Which repo fetches the round-trip list** — crypto pulls from
   :class:`CryptoTradeRepo.get_completed_round_trips`; stocks pull
   from :class:`TradeRepo.get_completed_stock_round_trips`.
2. **The knob menu** (``_SAFE_BOUNDS``) the LLM is allowed to tune —
   crypto has 6 knobs (RSI thresholds, SL/TP percentages, vol-SL
   multiplier, position size); stocks has 2 (position size + daily
   loss limit) because the strategy doesn't carry global SL/TP
   fallbacks.
3. **The review system prompt's JSON schema and asset label** —
   the LLM needs to be told whether it's reviewing crypto or stock
   decisions and which knobs it can suggest.

The three asset-specific bits are class attributes / one abstract
method; everything else lives on :class:`TradeSelfReviewBase`.
Subclasses are thin (~60 lines each).

Round-trip dict shape is asset-agnostic by construction:
``TradeRepo.get_completed_stock_round_trips`` already reshapes
stocks to the same dict crypto emits (with ``pair`` set to the
symbol), so the prompt formatter and trigger logic work over
either source without a switch.
"""

from __future__ import annotations

import logging
import time as _time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

from halal_trader.db.repos import StrategyAdjustmentRepo
from halal_trader.domain.ports import LLMBackend

logger = logging.getLogger(__name__)


# ── Module-level invariant ────────────────────────────────────────

# Clamped adjustment within ε of the current value is dropped on
# parse so the bot doesn't log "no-op" changes every review.
_NOOP_EPSILON = 1e-6


# ── Result dataclasses (asset-agnostic) ──────────────────────────


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


# ── Base class ────────────────────────────────────────────────────


class TradeSelfReviewBase(ABC):
    """Reviews closed trades and suggests strategy adjustments.

    Asset-class-specific behavior is set by overriding three class
    attributes — ``_ASSET_LABEL`` (used in logs + the prompt),
    ``_SAFE_BOUNDS`` (the knob menu with clamps), and
    ``_STRATEGY_PARAM_MAP`` (knob name → strategy attribute) — and
    one method, :meth:`_fetch_round_trips`, that pulls closed
    round-trips from the asset's trade repo.

    ``_SYSTEM_PROMPT`` is also a class attribute because the JSON
    schema the LLM emits must match ``_SAFE_BOUNDS`` keys exactly
    (the LLM can only emit knobs that were named in the schema).
    Subclasses spell out the prompt inline so the JSON schema and
    the knob menu are visibly co-located.
    """

    # ── Subclass-overridable config ──────────────────────────────

    _ASSET_LABEL: ClassVar[str]
    """e.g. ``"crypto"`` or ``"stock"``. Threads into the system prompt
    and the review-complete log line."""

    _SAFE_BOUNDS: ClassVar[dict[str, tuple[float, float]]]
    """Knob name → (low, high) clamps. Any knob the LLM suggests outside
    these bounds is clamped to the boundary, not rejected (matches
    crypto's pre-refactor behavior)."""

    _STRATEGY_PARAM_MAP: ClassVar[dict[str, str]]
    """Knob name (matches a ``_SAFE_BOUNDS`` key) → attribute name on the
    live strategy instance. ``_apply_to_strategy`` writes
    ``setattr(strategy, attr, value)`` so the next cycle picks up
    the new value without a restart."""

    _SYSTEM_PROMPT: ClassVar[str]
    """Asset-flavored review prompt. The JSON schema inside this prompt
    MUST match ``_SAFE_BOUNDS`` keys — out-of-schema knobs the LLM
    emits land in ``raw["parameter_adjustments"]`` and get dropped
    by ``_parse_review`` (the ``param not in _SAFE_BOUNDS`` guard)."""

    _REVIEW_COOLDOWN_SECONDS: ClassVar[int] = 300
    """Min interval between consecutive reviews. Crypto sets this in
    the original code; lifted to a class var so a backtest harness
    can override it without monkey-patching."""

    @property
    def _review_cooldown(self) -> int:
        """Back-compat alias for the pre-refactor instance attribute —
        the quota-handling test reads ``self._review_cooldown`` directly.
        Reads from the class var so subclass overrides still apply.
        """
        return self._REVIEW_COOLDOWN_SECONDS

    # ── Lifecycle ────────────────────────────────────────────────

    def __init__(
        self,
        llm: LLMBackend,
        *,
        strategy_adjustments: StrategyAdjustmentRepo,
        strategy: Any | None = None,
        consecutive_loss_trigger: int = 3,
        exec_failure_trigger: int = 10,
    ) -> None:
        self._llm = llm
        self._strategy_adjustments = strategy_adjustments
        self._strategy = strategy
        self._consecutive_loss_trigger = consecutive_loss_trigger
        self._exec_failure_trigger = exec_failure_trigger
        self._active_adjustments: dict[str, float] = {}
        self._pairs_to_avoid: list[str] = []
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

    # ── Exec-failure tracking ────────────────────────────────────

    def record_execution_failure(self, pair: str, error_type: str) -> None:
        """Track an execution failure for a pair / symbol."""
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
            counts = Counter(errors)
            summary = ", ".join(f"{err}: {cnt}" for err, cnt in counts.most_common(5))
            lines.append(f"  {pair}: {len(errors)} failures ({summary})")
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
                    logger.debug("Failed to persist observation: %s", exc)

            self._apply_adjustments(result)
            self._exec_failures.clear()

            logger.info(
                "Self-review complete (%s): %d observations, %d adjustments, %d pairs to avoid",
                self._ASSET_LABEL,
                len(result.observations),
                len(result.adjustments),
                len(result.pairs_to_avoid),
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
                    extra={"event": "llm.insufficient_quota"},
                )
            else:
                logger.error("Self-review failed: %s", e)
            return ReviewResult()

    # ── Helpers (asset-agnostic over the canonical round-trip dict) ──

    def _format_trades_for_review(self, round_trips: list[dict[str, Any]]) -> str:
        """Format trades with context for the review prompt.

        Reads ``rt["pair"]`` — the stocks repo
        (``get_completed_stock_round_trips``) reshapes ``symbol`` →
        ``pair`` exactly so this formatter is asset-agnostic.
        """
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
        """Parse the LLM review response into a ReviewResult.

        Knobs not in ``_SAFE_BOUNDS`` are silently dropped — this is
        the safety net for a confused LLM that hallucinates extra
        parameter names. Knobs at ``null`` are also dropped.
        """
        result = ReviewResult()
        result.observations = raw.get("observations", [])
        result.pairs_to_avoid = raw.get("pairs_to_avoid", [])
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

        if result.pairs_to_avoid:
            existing = set(self._pairs_to_avoid)
            existing.update(result.pairs_to_avoid)
            self._pairs_to_avoid = list(existing)
            logger.info("Pairs to avoid updated: %s", self._pairs_to_avoid)

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

    # ── Asset-class hook ─────────────────────────────────────────

    @abstractmethod
    async def _fetch_round_trips(
        self, *, limit: int, lookback_days: int | None
    ) -> list[dict[str, Any]]:
        """Pull closed round-trips from the asset's trade repo.

        Crypto calls :meth:`CryptoTradeRepo.get_completed_round_trips`;
        stocks calls :meth:`TradeRepo.get_completed_stock_round_trips`.
        Both return the same canonical dict shape (``pair``,
        ``buy_price``, ``sell_price``, ``pnl``, ``pnl_pct``,
        ``duration_minutes``, ``exit_reason``).
        """
