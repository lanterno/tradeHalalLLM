"""Halal robo-advisor engine.

For users who don't want to actively trade: a managed-portfolio
mode with halal asset allocation, target-date glide path, and
threshold-based rebalancing. The roadmap defers full integration
(per-user portfolio account + automatic rebalance scheduler) but
the **allocation + glide-path + rebalance math** is pure-Python —
exactly the isolated-module pattern of every other Round-4
landed wave.

Picked a focused engine over wedging the rules into the existing
strategy because the failure modes are different: a robo
allocation cares about long-horizon glide-paths + cohort risk
profiles, not per-cycle alpha. The active-trading strategy
optimises for short-term P&L; the robo allocation optimises for
the user's target-date retirement income.

Pinned semantics:
- **Halal-only asset classes by construction.** The
  `HalalAssetClass` enum names every category the engine accepts:
  HALAL_EQUITY (Zoya-screened stocks); SUKUK (Wave 1.H — Islamic
  bonds, never conventional bonds); HALAL_COMMODITIES (Wave 1.G —
  allocated-physical gold / silver only); HALAL_REIT (Wave 1.I —
  passing the property-trust screen); CASH (the fully-liquid
  reserve). Conventional bonds, leveraged ETFs, and any Wave-1.G-
  failing commodity vehicle are categorically absent — the engine
  *cannot* allocate to them because the enum doesn't include them.
- **Weights sum to 1.0.** Constructing a `TargetAllocation` with
  weights that sum outside [0.999, 1.001] raises at construction.
  The float-tolerance window prevents 0.50 + 0.30 + 0.20 = 1.0000…
  rounding noise from rejecting clean inputs.
- **No leverage / no shorts.** Every weight ∈ [0, 1] enforced at
  construction.
- **Rebalance threshold prevents churn.** A drift below
  `threshold_pct` (default 5%) for every asset class returns a
  no-op `RebalancePlan` rather than a flurry of tiny trades that
  burn fees + LLM cost without improving the allocation. Pinned
  via test that a 4.99% drift produces no trades; 5.00% triggers
  the first.
- **Glide path is monotonic toward conservative.** As the user
  approaches their target date, equity weight drops and sukuk +
  cash weights rise. Pinned via test that a 30y-out allocation has
  more equity than a 5y-out one and a 1y-out one has more cash
  than a 30y-out one.
- **Render output never includes USD balances or P&L.** The
  user-facing receipt summarises target weights + drift + planned
  trades by asset class share, never the absolute dollar amounts —
  mirrors the no-PII pattern of Wave 11.D privacy + Wave 11.C KYC
  + Wave 3.B vault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RiskProfile(str, Enum):
    """User's risk tolerance bucket.

    Pinned string values for DB / JSON stability. The three
    profiles map deterministically to base allocations the glide
    path then ages toward conservative.
    """

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class HalalAssetClass(str, Enum):
    """Halal-only asset categories the engine can allocate to.

    The set is closed: a future contributor adding a new asset
    class needs to (a) prove halal compliance, (b) extend the
    enum, (c) update the base allocations + glide path. The
    structural friction is intentional — no runtime "just add a
    string" knob that could silently allow a non-halal category.
    """

    HALAL_EQUITY = "halal_equity"
    SUKUK = "sukuk"
    HALAL_COMMODITIES = "halal_commodities"
    HALAL_REIT = "halal_reit"
    CASH = "cash"


# Base allocations at the **far horizon** (≥ 30 years to target).
# These are the maximum-equity-tolerance allocations per profile;
# the glide path interpolates toward the near-horizon allocation
# below as the target date approaches.
_FAR_HORIZON_ALLOCATIONS: dict[RiskProfile, dict[HalalAssetClass, float]] = {
    RiskProfile.CONSERVATIVE: {
        HalalAssetClass.HALAL_EQUITY: 0.40,
        HalalAssetClass.SUKUK: 0.40,
        HalalAssetClass.HALAL_COMMODITIES: 0.10,
        HalalAssetClass.HALAL_REIT: 0.05,
        HalalAssetClass.CASH: 0.05,
    },
    RiskProfile.MODERATE: {
        HalalAssetClass.HALAL_EQUITY: 0.60,
        HalalAssetClass.SUKUK: 0.25,
        HalalAssetClass.HALAL_COMMODITIES: 0.07,
        HalalAssetClass.HALAL_REIT: 0.05,
        HalalAssetClass.CASH: 0.03,
    },
    RiskProfile.AGGRESSIVE: {
        HalalAssetClass.HALAL_EQUITY: 0.80,
        HalalAssetClass.SUKUK: 0.10,
        HalalAssetClass.HALAL_COMMODITIES: 0.05,
        HalalAssetClass.HALAL_REIT: 0.03,
        HalalAssetClass.CASH: 0.02,
    },
}

# Near-horizon allocations (target date is < 1 year away). Cash
# is dominant; equity is minimised to protect against drawdown
# right before the user needs the money.
_NEAR_HORIZON_ALLOCATIONS: dict[RiskProfile, dict[HalalAssetClass, float]] = {
    RiskProfile.CONSERVATIVE: {
        HalalAssetClass.HALAL_EQUITY: 0.10,
        HalalAssetClass.SUKUK: 0.30,
        HalalAssetClass.HALAL_COMMODITIES: 0.05,
        HalalAssetClass.HALAL_REIT: 0.05,
        HalalAssetClass.CASH: 0.50,
    },
    RiskProfile.MODERATE: {
        HalalAssetClass.HALAL_EQUITY: 0.20,
        HalalAssetClass.SUKUK: 0.35,
        HalalAssetClass.HALAL_COMMODITIES: 0.05,
        HalalAssetClass.HALAL_REIT: 0.05,
        HalalAssetClass.CASH: 0.35,
    },
    RiskProfile.AGGRESSIVE: {
        HalalAssetClass.HALAL_EQUITY: 0.30,
        HalalAssetClass.SUKUK: 0.30,
        HalalAssetClass.HALAL_COMMODITIES: 0.05,
        HalalAssetClass.HALAL_REIT: 0.05,
        HalalAssetClass.CASH: 0.30,
    },
}


# Pinned glide-path horizons in years. Allocations between these
# anchor points interpolate linearly per asset class.
_FAR_HORIZON_YEARS = 30
_NEAR_HORIZON_YEARS = 1


@dataclass(frozen=True)
class TargetAllocation:
    """Target weights per asset class.

    Validated to be a complete allocation (every asset class
    present), non-negative, sum-to-1.0 (with a small float
    tolerance window).
    """

    weights: dict[HalalAssetClass, float]

    def __post_init__(self) -> None:
        for cat in HalalAssetClass:
            if cat not in self.weights:
                raise ValueError(f"weights missing asset class {cat.value!r}")
            w = self.weights[cat]
            if not 0.0 <= w <= 1.0:
                raise ValueError(f"weight for {cat.value!r} must be in [0, 1], got {w}")
        total = sum(self.weights.values())
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"weights must sum to 1.0 (within tolerance), got {total:.6f}")

    def weight_for(self, asset: HalalAssetClass) -> float:
        return self.weights[asset]


@dataclass(frozen=True)
class Holding:
    """A user's current weight in one asset class.

    Weight is share-of-portfolio (not USD); the engine deals in
    weights so the no-USD-in-render contract holds at the type
    level.
    """

    asset: HalalAssetClass
    weight: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(
                f"weight for {self.asset.value!r} must be in [0, 1], got {self.weight}"
            )


@dataclass(frozen=True)
class CurrentAllocation:
    """User's current portfolio weights.

    Validated like `TargetAllocation` so a buggy persistence layer
    that drifts the sum away from 1.0 surfaces immediately.
    """

    holdings: tuple[Holding, ...]

    def __post_init__(self) -> None:
        seen: set[HalalAssetClass] = set()
        for h in self.holdings:
            if h.asset in seen:
                raise ValueError(f"duplicate asset class {h.asset.value!r}")
            seen.add(h.asset)
        for cat in HalalAssetClass:
            if cat not in seen:
                raise ValueError(f"holdings missing asset class {cat.value!r}")
        total = sum(h.weight for h in self.holdings)
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"holdings must sum to 1.0 (within tolerance), got {total:.6f}")

    def weight_for(self, asset: HalalAssetClass) -> float:
        for h in self.holdings:
            if h.asset is asset:
                return h.weight
        raise KeyError(asset)


@dataclass(frozen=True)
class RebalanceTrade:
    """One asset's buy / sell instruction in weight terms.

    `delta` is positive for buys (move toward target) and negative
    for sells. The persistence layer converts weight deltas into
    USD trades using the user's portfolio NAV at execution time.
    """

    asset: HalalAssetClass
    current_weight: float
    target_weight: float
    delta: float

    @property
    def is_buy(self) -> bool:
        return self.delta > 0

    @property
    def is_sell(self) -> bool:
        return self.delta < 0


@dataclass(frozen=True)
class RebalancePlan:
    """The full rebalance instruction set.

    `trades` is sorted by asset class enum order for deterministic
    output. `is_noop` is True when no asset's drift exceeded the
    threshold — the persistence layer skips execution to avoid
    churn.
    """

    user_id: str
    trades: tuple[RebalanceTrade, ...]
    threshold_pct: float
    is_noop: bool
    max_drift_pct: float
    warnings: tuple[str, ...] = field(default_factory=tuple)


def compute_target_allocation(
    *,
    profile: RiskProfile,
    years_to_target: float,
) -> TargetAllocation:
    """Compute the target allocation for the user's profile + horizon.

    The allocation interpolates linearly between the far-horizon
    (≥ 30 years) and near-horizon (1 year) anchor points.
    `years_to_target` past 30 clamps to far-horizon; below 1 clamps
    to near-horizon (the user is essentially at their target date).
    """

    if years_to_target < 0:
        raise ValueError("years_to_target must be non-negative")

    far = _FAR_HORIZON_ALLOCATIONS[profile]
    near = _NEAR_HORIZON_ALLOCATIONS[profile]

    if years_to_target >= _FAR_HORIZON_YEARS:
        weights = dict(far)
    elif years_to_target <= _NEAR_HORIZON_YEARS:
        weights = dict(near)
    else:
        # Linear interpolation between the anchors.
        span = _FAR_HORIZON_YEARS - _NEAR_HORIZON_YEARS
        offset = years_to_target - _NEAR_HORIZON_YEARS
        far_weight = offset / span
        near_weight = 1.0 - far_weight
        weights = {}
        for cat in HalalAssetClass:
            weights[cat] = far[cat] * far_weight + near[cat] * near_weight
    # Normalise to compensate for any tiny float drift in the
    # interpolation (the interpolated weights *should* sum to 1.0
    # since both anchors do, but we normalise defensively).
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return TargetAllocation(weights=weights)


def compute_rebalance(
    *,
    user_id: str,
    current: CurrentAllocation,
    target: TargetAllocation,
    threshold_pct: float = 5.0,
) -> RebalancePlan:
    """Build a rebalance plan honouring the drift threshold.

    If every asset class has drifted less than `threshold_pct`
    from its target, returns a no-op plan with empty `trades` and
    `is_noop=True`. Otherwise emits one `RebalanceTrade` per
    asset class with delta = target − current.
    """

    if not user_id or not user_id.strip():
        raise ValueError("user_id must be non-empty")
    if threshold_pct <= 0:
        raise ValueError("threshold_pct must be positive")

    trades: list[RebalanceTrade] = []
    drifts: list[float] = []
    for cat in HalalAssetClass:
        cur = current.weight_for(cat)
        tgt = target.weight_for(cat)
        delta = tgt - cur
        drifts.append(abs(delta) * 100.0)
        trades.append(
            RebalanceTrade(
                asset=cat,
                current_weight=cur,
                target_weight=tgt,
                delta=delta,
            )
        )

    max_drift = max(drifts) if drifts else 0.0
    is_noop = max_drift < threshold_pct

    if is_noop:
        return RebalancePlan(
            user_id=user_id,
            trades=(),
            threshold_pct=threshold_pct,
            is_noop=True,
            max_drift_pct=max_drift,
        )

    return RebalancePlan(
        user_id=user_id,
        trades=tuple(trades),
        threshold_pct=threshold_pct,
        is_noop=False,
        max_drift_pct=max_drift,
    )


_PROFILE_EMOJI: dict[RiskProfile, str] = {
    RiskProfile.CONSERVATIVE: "🛡️",
    RiskProfile.MODERATE: "⚖️",
    RiskProfile.AGGRESSIVE: "🚀",
}


def render_target_allocation(
    *, profile: RiskProfile, years_to_target: float, allocation: TargetAllocation
) -> str:
    """User-facing target-allocation summary."""

    lines = [
        f"{_PROFILE_EMOJI[profile]} Target allocation ({profile.value}, "
        f"{years_to_target:.1f}y horizon)",
    ]
    for cat in HalalAssetClass:
        w = allocation.weight_for(cat)
        lines.append(f"  {cat.value}: {w * 100:.1f}%")
    return "\n".join(lines)


def render_rebalance_plan(plan: RebalancePlan) -> str:
    """User-facing rebalance-plan summary.

    Pinned no-USD contract: the receipt shows weight deltas only,
    never absolute dollar amounts. Operators wanting USD figures
    iterate through `plan.trades` and apply their portfolio NAV
    separately.
    """

    if plan.is_noop:
        return (
            f"♻️ {plan.user_id} — within threshold "
            f"(max drift {plan.max_drift_pct:.2f}% < {plan.threshold_pct:.1f}%); "
            "no rebalance needed"
        )
    lines = [
        f"♻️ {plan.user_id} — rebalance required "
        f"(max drift {plan.max_drift_pct:.2f}% ≥ {plan.threshold_pct:.1f}%)",
    ]
    for t in plan.trades:
        if t.is_buy:
            arrow = "↑"
        elif t.is_sell:
            arrow = "↓"
        else:
            arrow = "·"
        lines.append(
            f"  {arrow} {t.asset.value}: "
            f"{t.current_weight * 100:.1f}% → {t.target_weight * 100:.1f}% "
            f"(Δ {t.delta * 100:+.2f}%)"
        )
    return "\n".join(lines)


__all__ = [
    "CurrentAllocation",
    "HalalAssetClass",
    "Holding",
    "RebalancePlan",
    "RebalanceTrade",
    "RiskProfile",
    "TargetAllocation",
    "compute_rebalance",
    "compute_target_allocation",
    "render_rebalance_plan",
    "render_target_allocation",
]
