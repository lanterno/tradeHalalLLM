"""Volatility-aware slippage model.

The legacy backtester used a flat 5bps slippage in both directions.
That under-estimates real-fill cost during high-vol regimes (where
spreads widen and depth thins) and over-estimates it during calm
periods (where market orders barely move the book). This module
provides a simple but defensible scaling: the effective slippage is
the configured *baseline* multiplied by ``ratio = atr_pct /
atr_baseline``, clamped so an outlier ATR can't 100×-ify the cost.

There is also an optional **size-impact** term scaled by the trade's
notional relative to recent quote-volume — bigger orders eat more of
the book. Both terms are additive and conservative; a future iteration
can fit them to live paper-vs-real fill divergence stats once Phase 4
records those.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bounds on the volatility multiplier so a single ATR spike doesn't
# explode slippage in the backtest. 0.5× lower bound prevents calm
# regimes from making the model unrealistically optimistic.
_VOL_MULT_MIN = 0.5
_VOL_MULT_MAX = 4.0

# Bounds on the size-impact term as a fraction of price. Capped so an
# (impossible-in-practice) order eating 100% of book volume doesn't
# produce a negative fill price in the backtester.
_SIZE_IMPACT_MAX = 0.01  # 100bps absolute ceiling


@dataclass(frozen=True)
class SlippageInputs:
    """All inputs the model needs to price one fill."""

    side: str  # "buy" | "sell"
    notional_usd: float
    atr_pct: float = 0.0
    atr_baseline: float = 0.02
    recent_quote_volume_usd: float = 0.0  # rolling 1m or 5m quote vol
    baseline_slippage_pct: float = 0.0005  # legacy default — 5bps


@dataclass(frozen=True)
class SlippageResult:
    fill_price: float
    slippage_pct: float
    components: dict[str, float]


def _vol_multiplier(atr_pct: float, atr_baseline: float) -> float:
    if atr_pct <= 0 or atr_baseline <= 0:
        return 1.0
    raw = atr_pct / atr_baseline
    return max(_VOL_MULT_MIN, min(_VOL_MULT_MAX, raw))


def _size_impact(notional_usd: float, recent_quote_volume_usd: float) -> float:
    """Linear impact: trade size relative to recent throughput.

    A trade that's 1% of the last minute's volume adds ~1bp of slippage;
    10% adds 10bps; 100% adds 100bps (the cap). The relationship is
    deliberately linear and modest — empirical book-impact studies
    show square-root scaling, but that requires curve-fitting we don't
    have data for yet. Linear over-estimates the impact for big orders,
    which matches our "be conservative in backtests" stance.
    """
    if notional_usd <= 0 or recent_quote_volume_usd <= 0:
        return 0.0
    fraction = notional_usd / recent_quote_volume_usd
    impact = fraction * 0.001  # 0.1% per 100% of recent vol
    return min(_SIZE_IMPACT_MAX, impact)


def estimate_fill(price: float, inputs: SlippageInputs) -> SlippageResult:
    """Apply vol-scaled slippage + size-impact to a market-order fill price.

    Returns both the fill price and a breakdown for logging — that lets
    the backtester surface why a fill was bad (was it the size? the
    vol regime? the baseline?).
    """
    if price <= 0:
        raise ValueError("price must be positive")
    if inputs.side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell'; got {inputs.side!r}")

    vol_mult = _vol_multiplier(inputs.atr_pct, inputs.atr_baseline)
    base_slip = inputs.baseline_slippage_pct * vol_mult
    size_imp = _size_impact(inputs.notional_usd, inputs.recent_quote_volume_usd)
    total_slip = base_slip + size_imp

    if inputs.side == "buy":
        fill_price = price * (1 + total_slip)
    else:
        fill_price = price * (1 - total_slip)

    return SlippageResult(
        fill_price=fill_price,
        slippage_pct=total_slip,
        components={
            "baseline": inputs.baseline_slippage_pct,
            "vol_multiplier": vol_mult,
            "size_impact": size_imp,
        },
    )


def confidence_weighted_quantity(
    base_quantity: float,
    confidence: float,
    *,
    floor: float = 0.5,
    ceiling: float = 1.5,
) -> float:
    """Scale a trade quantity by LLM confidence inside ``[floor, ceiling]``.

    Confidence 0.5 (the "no-edge" midpoint) leaves the size unchanged;
    confidence 1.0 scales up to ``ceiling``; confidence 0.0 scales down
    to ``floor``. Bounded so a hallucinated 0.0 doesn't zero out a trade
    that the cycle has otherwise approved — this is a *modulator*, not a
    gate.

    Mirrors the spirit of the Phase 1 sizer (which keys off ``2p - 1``)
    but stays multiplicative on a pre-decided base quantity, which is
    what the backtester wants when it replays historic plans.
    """
    if base_quantity <= 0:
        return 0.0
    p = max(0.0, min(1.0, float(confidence)))
    # Linear interpolation: p=0 → floor, p=0.5 → 1.0, p=1 → ceiling.
    if p <= 0.5:
        scale = floor + (1.0 - floor) * (p / 0.5)
    else:
        scale = 1.0 + (ceiling - 1.0) * ((p - 0.5) / 0.5)
    return base_quantity * scale
