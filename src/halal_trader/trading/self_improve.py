"""Stocks-side LLM self-improvement loop.

Thin subclass of :class:`TradeSelfReviewBase` — mirrors
:mod:`halal_trader.crypto.self_improve` but with a smaller knob menu
because the stocks :class:`TradingStrategy` doesn't expose global
SL/TP fallbacks (the LLM emits SL/TP per decision; there's no
``stop_loss_pct`` instance attribute to override).

Knob menu has 2 entries:

* ``max_position_pct`` — max share of portfolio in a single position.
* ``daily_loss_limit`` — daily P&L floor before the cycle halts.

A future expansion (e.g. when stocks gets a self-tuning RSI gate)
would just add the new knob name + bounds to ``_STOCK_SAFE_BOUNDS``
and to the JSON schema inside ``_STOCK_SYSTEM_PROMPT``.

Pulls round-trips from :meth:`TradeRepo.get_completed_stock_round_trips`
which reshapes ``symbol`` → ``pair`` so the asset-agnostic base
formatter works unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from halal_trader.core.self_improve import TradeSelfReviewBase
from halal_trader.db.repos import StrategyAdjustmentRepo, TradeRepo
from halal_trader.domain.ports import LLMBackend

if TYPE_CHECKING:
    from halal_trader.trading.strategy import TradingStrategy


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
  "pairs_to_avoid": ["<symbol1>", ...],
  "strategy_notes": "<overall strategy recommendation>"
}}

Only suggest adjustments you are confident about. Use null for parameters that don't need changing.

Note: ``pairs_to_avoid`` is the field name for backward compatibility with the crypto-side schema; \
list stock tickers here.
"""


class StockTradeSelfReview(TradeSelfReviewBase):
    """Stocks-side self-review — 2 tunable knobs over Alpaca round-trips.

    Per-decision SL/TP isn't a tunable knob because the
    :class:`TradingStrategy` doesn't carry a global fallback (the
    LLM emits SL/TP per buy decision). If the model wants to
    influence stop-loss behavior, it should be done by adjusting the
    daily-loss-limit floor (which the cycle's risk halt reads) or
    by the LLM emitting tighter stops per decision — not by
    self-tuning a global SL knob that doesn't exist.
    """

    _ASSET_LABEL: ClassVar[str] = "stock"
    _SAFE_BOUNDS: ClassVar[dict[str, tuple[float, float]]] = _STOCK_SAFE_BOUNDS
    _STRATEGY_PARAM_MAP: ClassVar[dict[str, str]] = _STOCK_STRATEGY_PARAM_MAP
    _SYSTEM_PROMPT: ClassVar[str] = _STOCK_SYSTEM_PROMPT

    def __init__(
        self,
        llm: LLMBackend,
        *,
        strategy_adjustments: StrategyAdjustmentRepo,
        trades: TradeRepo,
        strategy: "TradingStrategy | None" = None,
        consecutive_loss_trigger: int = 3,
        exec_failure_trigger: int = 10,
    ) -> None:
        super().__init__(
            llm,
            strategy_adjustments=strategy_adjustments,
            strategy=strategy,
            consecutive_loss_trigger=consecutive_loss_trigger,
            exec_failure_trigger=exec_failure_trigger,
        )
        self._trades = trades

    async def _fetch_round_trips(
        self, *, limit: int, lookback_days: int | None
    ) -> list[dict[str, Any]]:
        return await self._trades.get_completed_stock_round_trips(
            limit=limit, lookback_days=lookback_days
        )


__all__ = ["StockTradeSelfReview"]
