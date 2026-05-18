"""Sukuk-based duration hedge — Round-5 Wave 13.D.

When the operator's portfolio carries duration risk (long-dated sukuk
holdings, fixed-coupon Ijara), interest-rate-equivalent shocks
translate to capital losses. This module composes a **sukuk-ladder
hedge** that neutralises portfolio modified duration: short or sell
sukuk positions whose dollar-duration offsets the primary book.

Pinned semantics:

- **Modified duration approximation** = Macaulay / (1 + y).
- **Dollar-duration matching** is the headline objective.
- **Closed-set HedgeStance ladder** (NEUTRAL / OFFSETTING / ENHANCING).
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from halal_trader.markets.sukuk_pricing import Sukuk


class HedgeStance(str, Enum):
    """Closed-set hedge stances."""

    NEUTRAL = "neutral"
    OFFSETTING = "offsetting"
    ENHANCING = "enhancing"


def macaulay_duration(sukuk: Sukuk, *, yield_rate: float) -> float:
    """Macaulay duration in years for a sukuk priced at the given yield."""
    if yield_rate < -0.05:
        raise ValueError("yield_rate too negative")
    if not sukuk.cashflows:
        return 0.0
    pv_total = 0.0
    weighted = 0.0
    for cf in sukuk.cashflows:
        df = math.exp(-yield_rate * cf.time_years)
        pv = cf.amount * df
        pv_total += pv
        weighted += pv * cf.time_years
    if pv_total == 0:
        return 0.0
    return weighted / pv_total


def modified_duration(sukuk: Sukuk, *, yield_rate: float) -> float:
    """Modified duration ≈ Macaulay / (1 + y) for compounded yields."""
    mac = macaulay_duration(sukuk, yield_rate=yield_rate)
    return mac / (1.0 + yield_rate) if yield_rate > -1.0 else mac


def dollar_duration(sukuk: Sukuk, *, yield_rate: float, market_value: float) -> float:
    """Dollar duration = modified_duration × market_value."""
    if market_value < 0:
        raise ValueError("market_value must be non-negative")
    return modified_duration(sukuk, yield_rate=yield_rate) * market_value


@dataclass(frozen=True)
class PortfolioDuration:
    """Portfolio-level duration aggregate."""

    total_market_value: float
    total_dollar_duration: float
    average_duration: float

    def __post_init__(self) -> None:
        if self.total_market_value < 0:
            raise ValueError("total_market_value must be non-negative")


def compute_portfolio_duration(holdings: Iterable[tuple[Sukuk, float, float]]) -> PortfolioDuration:
    """Compute portfolio-level dollar duration.

    Each holding tuple = ``(sukuk, yield_rate, market_value)``.
    """
    total_dd = 0.0
    total_mv = 0.0
    for s, y, mv in holdings:
        total_dd += dollar_duration(s, yield_rate=y, market_value=mv)
        total_mv += mv
    avg = total_dd / total_mv if total_mv > 0 else 0.0
    return PortfolioDuration(
        total_market_value=total_mv,
        total_dollar_duration=total_dd,
        average_duration=avg,
    )


@dataclass(frozen=True)
class HedgeRecommendation:
    """Recommendation for a duration hedge."""

    stance: HedgeStance
    target_dollar_duration_offset: float
    hedge_market_value: float

    def __post_init__(self) -> None:
        if self.hedge_market_value < 0:
            raise ValueError("hedge_market_value must be non-negative")


def recommend_hedge(
    portfolio: PortfolioDuration,
    *,
    hedge_sukuk: Sukuk,
    hedge_yield: float,
    target_residual_dollar_duration: float = 0.0,
) -> HedgeRecommendation:
    """Recommend a hedge sukuk MV that brings residual dollar duration to target."""
    hedge_md = modified_duration(hedge_sukuk, yield_rate=hedge_yield)
    if hedge_md <= 0:
        return HedgeRecommendation(
            stance=HedgeStance.NEUTRAL,
            target_dollar_duration_offset=0.0,
            hedge_market_value=0.0,
        )

    offset_needed = target_residual_dollar_duration - portfolio.total_dollar_duration
    hedge_mv = abs(offset_needed) / hedge_md

    if abs(offset_needed) < 1e-9:
        return HedgeRecommendation(
            stance=HedgeStance.NEUTRAL,
            target_dollar_duration_offset=0.0,
            hedge_market_value=0.0,
        )

    stance = (
        HedgeStance.OFFSETTING
        if (offset_needed * portfolio.total_dollar_duration) < 0
        else HedgeStance.ENHANCING
    )
    return HedgeRecommendation(
        stance=stance,
        target_dollar_duration_offset=offset_needed,
        hedge_market_value=hedge_mv,
    )


def render_recommendation(rec: HedgeRecommendation) -> str:
    emoji = {
        HedgeStance.NEUTRAL: "⚖️",
        HedgeStance.OFFSETTING: "🛡️",
        HedgeStance.ENHANCING: "🔥",
    }[rec.stance]
    return (
        f"{emoji} duration hedge: {rec.stance.value} "
        f"target_offset=${rec.target_dollar_duration_offset:+.2f} "
        f"hedge_mv=${rec.hedge_market_value:.2f}"
    )
