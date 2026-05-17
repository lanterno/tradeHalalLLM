"""Liquidity-adjusted risk model — Round-5 Wave 13.F.

Standard VaR assumes positions can be liquidated at the market price.
For thinly-traded names, the actual exit price drifts away as the
operator's order eats into the book. This module ships
**liquidity-adjusted VaR** + the underlying **liquidity score** that
combines bid-ask spread, average daily volume, and depth.

Pinned semantics:

- **Closed-set LiquidityTier ladder** (DEEP / NORMAL / THIN / ILLIQUID).
- **Liquidity score in [0, 1]** — 1.0 = deep, 0.0 = illiquid.
- **Liquidity-adjusted VaR** = base_VaR × (1 + liquidation_cost_pct);
  liquidation_cost_pct grows when score is low.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LiquidityTier(str, Enum):
    """Closed-set liquidity tiers."""

    DEEP = "deep"
    NORMAL = "normal"
    THIN = "thin"
    ILLIQUID = "illiquid"


@dataclass(frozen=True)
class LiquidityInputs:
    """Inputs for liquidity scoring."""

    symbol: str
    bid_ask_spread_bps: float  # basis points
    average_daily_volume: float
    market_depth_at_top: float
    position_size: float

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.bid_ask_spread_bps < 0:
            raise ValueError("bid_ask_spread_bps must be non-negative")
        if self.average_daily_volume < 0:
            raise ValueError("average_daily_volume must be non-negative")
        if self.market_depth_at_top < 0:
            raise ValueError("market_depth_at_top must be non-negative")
        if self.position_size < 0:
            raise ValueError("position_size must be non-negative")


@dataclass(frozen=True)
class LiquidityPolicy:
    """Operator-tunable thresholds."""

    deep_spread_max_bps: float = 5.0
    normal_spread_max_bps: float = 25.0
    thin_spread_max_bps: float = 100.0
    deep_pos_pct_adv: float = 0.005  # 0.5% of ADV
    normal_pos_pct_adv: float = 0.05  # 5% of ADV
    illiquid_threshold_pos_pct_adv: float = 0.20

    def __post_init__(self) -> None:
        if not (
            0.0 < self.deep_spread_max_bps
            < self.normal_spread_max_bps
            < self.thin_spread_max_bps
        ):
            raise ValueError("spread thresholds must increase")
        if not (
            0.0 < self.deep_pos_pct_adv
            < self.normal_pos_pct_adv
            < self.illiquid_threshold_pos_pct_adv
        ):
            raise ValueError("pos pct thresholds must increase")


@dataclass(frozen=True)
class LiquidityAssessment:
    """Result of liquidity scoring."""

    symbol: str
    score: float
    tier: LiquidityTier
    position_pct_of_adv: float
    estimated_liquidation_cost_pct: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")
        if self.position_pct_of_adv < 0:
            raise ValueError("position_pct_of_adv must be non-negative")
        if self.estimated_liquidation_cost_pct < 0:
            raise ValueError("estimated_liquidation_cost_pct must be non-negative")


def assess_liquidity(
    inputs: LiquidityInputs, *, policy: LiquidityPolicy | None = None
) -> LiquidityAssessment:
    """Score liquidity + classify tier + estimate liquidation cost."""
    pol = policy if policy is not None else LiquidityPolicy()

    pos_pct = (
        inputs.position_size / inputs.average_daily_volume
        if inputs.average_daily_volume > 0
        else float("inf")
    )

    # Tier laddering
    if (
        inputs.bid_ask_spread_bps <= pol.deep_spread_max_bps
        and pos_pct <= pol.deep_pos_pct_adv
    ):
        tier = LiquidityTier.DEEP
    elif (
        inputs.bid_ask_spread_bps <= pol.normal_spread_max_bps
        and pos_pct <= pol.normal_pos_pct_adv
    ):
        tier = LiquidityTier.NORMAL
    elif (
        inputs.bid_ask_spread_bps <= pol.thin_spread_max_bps
        and pos_pct <= pol.illiquid_threshold_pos_pct_adv
    ):
        tier = LiquidityTier.THIN
    else:
        tier = LiquidityTier.ILLIQUID

    # Score: simple two-component blend
    spread_score = max(
        0.0, 1.0 - inputs.bid_ask_spread_bps / pol.thin_spread_max_bps
    )
    if pos_pct == float("inf"):
        size_score = 0.0
    else:
        size_score = max(
            0.0, 1.0 - pos_pct / pol.illiquid_threshold_pos_pct_adv
        )
    score = (spread_score + size_score) / 2.0
    score = max(0.0, min(1.0, score))

    # Liquidation cost: half-spread + size-impact
    half_spread_pct = inputs.bid_ask_spread_bps / 2.0 / 10000.0
    if pos_pct == float("inf"):
        size_impact_pct = 0.20  # cap
    else:
        size_impact_pct = min(0.20, pos_pct * 0.10)
    liq_cost = half_spread_pct + size_impact_pct

    return LiquidityAssessment(
        symbol=inputs.symbol,
        score=score,
        tier=tier,
        position_pct_of_adv=min(pos_pct, 1e9),
        estimated_liquidation_cost_pct=liq_cost,
    )


def liquidity_adjusted_var(
    base_var: float, assessment: LiquidityAssessment
) -> float:
    """Inflate base VaR by the estimated liquidation-cost fraction."""
    if base_var < 0:
        raise ValueError("base_var must be non-negative")
    return base_var * (1.0 + assessment.estimated_liquidation_cost_pct)


def render_assessment(a: LiquidityAssessment) -> str:
    emoji = {
        LiquidityTier.DEEP: "🟢",
        LiquidityTier.NORMAL: "🟡",
        LiquidityTier.THIN: "🟠",
        LiquidityTier.ILLIQUID: "🔴",
    }[a.tier]
    return (
        f"{emoji} {a.symbol} liquidity={a.score:.2f} ({a.tier.value}) "
        f"pos%ADV={a.position_pct_of_adv * 100:.2f}% "
        f"liq_cost={a.estimated_liquidation_cost_pct * 100:.3f}%"
    )
