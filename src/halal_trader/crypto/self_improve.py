"""Crypto-side LLM self-improvement loop.

Thin subclass of :class:`TradeSelfReviewBase` — the asset-agnostic
orchestration (cooldown, exec-failure tracking, prompt assembly,
parse/clamp/apply) lives in :mod:`halal_trader.core.self_improve`.
This module only carries the three crypto-specific bits:

1. The knob menu (``_CRYPTO_SAFE_BOUNDS``) — 6 parameters: RSI
   thresholds, SL/TP percentages, volatility multiplier, position
   size. Stocks' menu is smaller because its strategy doesn't carry
   global SL/TP fallbacks.
2. The system prompt — flavored "crypto trading decisions" and
   spelling the 6 knobs into the JSON schema.
3. ``_fetch_round_trips`` — pulls from
   :meth:`CryptoTradeRepo.get_completed_round_trips`.

``TradeSelfReview`` is preserved as a public alias of
:class:`CryptoTradeSelfReview` so existing import sites
(``crypto/components.py``, the ``test_self_improve_*`` suite) keep
working unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from halal_trader.core.self_improve import (
    ReviewResult,
    StrategyAdjustment,
    TradeSelfReviewBase,
)
from halal_trader.db.repos import CryptoTradeRepo, StrategyAdjustmentRepo
from halal_trader.domain.ports import LLMBackend

if TYPE_CHECKING:
    from halal_trader.crypto.strategy import CryptoTradingStrategy


_CRYPTO_SAFE_BOUNDS: dict[str, tuple[float, float]] = {
    "rsi_buy_threshold": (25.0, 45.0),
    "rsi_sell_threshold": (55.0, 80.0),
    "max_position_pct": (0.10, 0.30),
    "stop_loss_pct": (0.003, 0.020),
    "take_profit_pct": (0.005, 0.030),
    "volatile_sl_multiplier": (1.0, 2.5),
}


_CRYPTO_STRATEGY_PARAM_MAP: dict[str, str] = {
    "max_position_pct": "_max_position_pct",
    "stop_loss_pct": "_stop_loss_pct",
    "take_profit_pct": "_take_profit_pct",
}


_CRYPTO_SYSTEM_PROMPT = """\
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


class CryptoTradeSelfReview(TradeSelfReviewBase):
    """Crypto-side self-review — 6 tunable knobs over Binance round-trips."""

    _ASSET_LABEL: ClassVar[str] = "crypto"
    _SAFE_BOUNDS: ClassVar[dict[str, tuple[float, float]]] = _CRYPTO_SAFE_BOUNDS
    _STRATEGY_PARAM_MAP: ClassVar[dict[str, str]] = _CRYPTO_STRATEGY_PARAM_MAP
    _SYSTEM_PROMPT: ClassVar[str] = _CRYPTO_SYSTEM_PROMPT

    def __init__(
        self,
        llm: LLMBackend,
        *,
        strategy_adjustments: StrategyAdjustmentRepo,
        crypto_trades: CryptoTradeRepo,
        strategy: "CryptoTradingStrategy | None" = None,
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
        self._crypto_trades = crypto_trades

    async def _fetch_round_trips(
        self, *, limit: int, lookback_days: int | None
    ) -> list[dict[str, Any]]:
        # ``lookback_days=None`` is the trigger-check path (no time
        # filter, just "most recent N"); a value is the periodic-review
        # path. ``CryptoTradeRepo.get_completed_round_trips`` accepts
        # both shapes natively.
        return await self._crypto_trades.get_completed_round_trips(
            limit=limit, lookback_days=lookback_days
        )


# Backward-compat alias — every existing import site
# (``crypto.components``, ``test_self_improve_*``) reads
# ``TradeSelfReview`` from this module. Preserving the name avoids a
# cascading rename across already-passing tests + the working crypto
# wiring.
TradeSelfReview = CryptoTradeSelfReview


__all__ = [
    "CryptoTradeSelfReview",
    "ReviewResult",
    "StrategyAdjustment",
    "TradeSelfReview",
]
