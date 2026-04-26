"""Confidence × edge × volatility position sizer.

The legacy sizing path used a flat ``max_position_pct`` (default 25%)
for every trade regardless of how confident the LLM was, what the
indicator-implied edge looked like, or how volatile the underlying
moved. That under-sizes high-conviction setups in calm markets and
over-sizes low-conviction setups in turbulent ones — both leak edge.

This sizer is deliberately conservative — *fractional Kelly*, capped at
``KELLY_FRACTION`` of the optimal Kelly fraction so a single mis-bet
can't wipe out the bankroll. The output is a position size in *quote
currency* (USD/USDT) that the caller multiplies by entry price to get a
quantity.

Halal constraints (no leverage, no shorts) are honoured by clamping
size to never exceed available equity and refusing negative sizes —
this module never returns a result that would require margin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from halal_trader.domain.money import quantize_usd, to_decimal

logger = logging.getLogger(__name__)


# Fractional Kelly multiplier — never bet more than 25% of the
# theoretically-optimal Kelly fraction. Full Kelly is a return-maximising
# but psychologically and practically untenable target; quarter-Kelly is
# the textbook practitioner default and matches how we currently halt on
# 8% drawdown (full Kelly would routinely produce 40%+ drawdowns).
KELLY_FRACTION = Decimal("0.25")

# Volatility scaling clamps — same shape the existing risk engine uses.
# At 5× baseline ATR we scale to 0.3 (severely under-size); at 0.5×
# baseline we scale to 2.0 (boost a calm-market setup).
VOL_SCALE_MIN = Decimal("0.3")
VOL_SCALE_MAX = Decimal("2.0")


@dataclass(frozen=True)
class SizingInputs:
    """All the per-trade inputs the sizer needs.

    Kept as a frozen dataclass so call sites can build it once and pass
    it to both the sizer and any logging surface without risking
    accidental mutation between read and write.
    """

    equity_usd: Decimal
    confidence: float  # LLM confidence in [0, 1]
    atr_pct: float  # ATR/price as decimal (e.g. 0.018 == 1.8% intraday vol)
    atr_baseline: float  # vol regime anchor (e.g. 0.02 default)
    base_max_position_pct: float  # absolute ceiling, never exceeded
    available_usd: Decimal | None = None  # cap to actual buying power if set


@dataclass(frozen=True)
class SizingResult:
    notional_usd: Decimal
    fraction_used: Decimal  # final fraction of equity allocated
    kelly_fraction: Decimal  # raw quarter-Kelly suggestion before vol scaling
    vol_scale: Decimal  # multiplier applied for volatility regime
    capped_by: str  # 'kelly', 'base_max', 'available', or 'min_dust'


def _clamp(value: Decimal, lo: Decimal, hi: Decimal) -> Decimal:
    return max(lo, min(value, hi))


def _confidence_to_edge(confidence: float) -> Decimal:
    """Map an LLM-reported probability to an expected-edge proxy.

    The Kelly formula needs ``p`` (win probability) and ``b`` (win/loss
    ratio). We don't have a per-trade ``b`` until after the trade closes,
    so we approximate edge as ``2p - 1`` — that's the expected value of
    a +1/-1 payoff at probability ``p``. It's deliberately *too small*
    near p=0.5 (we don't bet on coin flips) and saturates at p=1.

    Confidence in (0, 0.5] yields zero size — we don't take negative-edge
    bets even at quarter-Kelly. Confidence > 1 is clamped (defensive
    against LLM hallucination).
    """
    p = max(0.0, min(1.0, float(confidence)))
    if p <= 0.5:
        return Decimal("0")
    # Route via Decimal-from-str to avoid binary-float drift like
    # 0.7 → Decimal('0.69999…975'). The string form preserves the
    # user-facing value, so 2p-1 lands on a clean rational.
    edge = Decimal(2) * Decimal(str(p)) - Decimal(1)
    return edge


def _vol_scale(atr_pct: float, baseline: float) -> Decimal:
    """Scale by inverse vol relative to the baseline.

    Above-baseline vol shrinks size; below-baseline vol boosts it.
    Returns 1.0 when we have no useful ATR signal so the sizer falls
    back to confidence × Kelly only.
    """
    if atr_pct <= 0 or baseline <= 0:
        return Decimal("1")
    raw = Decimal(str(baseline)) / Decimal(str(atr_pct))
    return _clamp(raw, VOL_SCALE_MIN, VOL_SCALE_MAX)


def size_position(inputs: SizingInputs) -> SizingResult:
    """Return a position-size decision for one trade.

    Calculation order:
      1. Quarter-Kelly fraction from confidence (zero if p ≤ 0.5)
      2. Multiply by the volatility scale (calm boost, turbulent shrink)
      3. Cap at ``base_max_position_pct`` (the legacy hard ceiling)
      4. Convert to USD and cap at ``available_usd`` if provided
      5. If the result is below dust ($1), return zero with reason ``min_dust``
    """
    edge = _confidence_to_edge(inputs.confidence)
    if edge == 0:
        return SizingResult(
            notional_usd=Decimal("0"),
            fraction_used=Decimal("0"),
            kelly_fraction=Decimal("0"),
            vol_scale=Decimal("1"),
            capped_by="kelly",
        )

    # Quarter-Kelly on a +1/-1 payoff approximation.
    kelly = edge * KELLY_FRACTION

    vol_scale = _vol_scale(inputs.atr_pct, inputs.atr_baseline)
    fraction = kelly * vol_scale

    base_cap = Decimal(str(inputs.base_max_position_pct))
    capped_by = "kelly"
    if fraction > base_cap:
        fraction = base_cap
        capped_by = "base_max"

    notional = quantize_usd(to_decimal(inputs.equity_usd) * fraction)

    if inputs.available_usd is not None and notional > inputs.available_usd:
        notional = quantize_usd(inputs.available_usd)
        capped_by = "available"
        if inputs.equity_usd > 0:
            fraction = notional / to_decimal(inputs.equity_usd)

    if notional < Decimal("1.00"):
        return SizingResult(
            notional_usd=Decimal("0"),
            fraction_used=Decimal("0"),
            kelly_fraction=kelly,
            vol_scale=vol_scale,
            capped_by="min_dust",
        )

    return SizingResult(
        notional_usd=notional,
        fraction_used=fraction,
        kelly_fraction=kelly,
        vol_scale=vol_scale,
        capped_by=capped_by,
    )
