"""Halal robo-advisor allocator — Round-5 Wave 21.E.

White-label robo product: given a user's risk profile + horizon +
KYC-passed eligibility flags, pick a halal model portfolio (equity /
sukuk / cash bands) and the rebalance triggers.

This composes with:
- `education.risk_assessment.RiskProfile` for the risk dimension
- `marketplace.etf_basket` for the actual basket implementation

This module is the **profile → model-portfolio mapper + drift trigger
+ Wakalah fee accounting** for the robo. The actual rebalance execution
lives outside this layer.

Pinned semantics:

- **Closed-set TimeHorizon** — SHORT (≤3y) / MEDIUM (3-7y) / LONG (≥7y).
- **Closed-set ModelPortfolio** — DEFENSIVE / BALANCED / GROWTH /
  AGGRESSIVE_GROWTH. Each pins (equity_min, equity_max, sukuk_min,
  sukuk_max, cash_min, cash_max) bands.
- **Profile × horizon → ModelPortfolio mapping** is a closed lookup
  table; operator can override.
- **Robo Wakalah fee** is flat per AUM bracket (NOT performance-based);
  operator-tunable.
- **Drift threshold** triggers rebalance when |actual − target| >
  threshold; default 5pp.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from halal_trader.education.risk_assessment import RiskProfile


class TimeHorizon(str, Enum):
    """Closed-set time-horizon ladder."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class ModelPortfolio(str, Enum):
    """Closed-set model-portfolio ladder."""

    DEFENSIVE = "defensive"
    BALANCED = "balanced"
    GROWTH = "growth"
    AGGRESSIVE_GROWTH = "aggressive_growth"


@dataclass(frozen=True)
class AllocationBand:
    """Equity / sukuk / cash bands for a model portfolio."""

    equity_min: float
    equity_max: float
    sukuk_min: float
    sukuk_max: float
    cash_min: float
    cash_max: float

    def __post_init__(self) -> None:
        for name, lo, hi in (
            ("equity", self.equity_min, self.equity_max),
            ("sukuk", self.sukuk_min, self.sukuk_max),
            ("cash", self.cash_min, self.cash_max),
        ):
            if not 0.0 <= lo <= hi <= 1.0:
                raise ValueError(f"{name} band invalid: 0 ≤ min ≤ max ≤ 1")
        # Feasibility: there must exist (e, s, c) with e+s+c=1 inside all bands.
        if (
            self.equity_min + self.sukuk_min + self.cash_min > 1.0 + 1e-9
            or self.equity_max + self.sukuk_max + self.cash_max < 1.0 - 1e-9
        ):
            raise ValueError("bands do not admit any allocation summing to 1")


_MODEL_ALLOCATIONS: dict[ModelPortfolio, AllocationBand] = {
    ModelPortfolio.DEFENSIVE: AllocationBand(
        equity_min=0.10,
        equity_max=0.30,
        sukuk_min=0.55,
        sukuk_max=0.75,
        cash_min=0.05,
        cash_max=0.20,
    ),
    ModelPortfolio.BALANCED: AllocationBand(
        equity_min=0.40,
        equity_max=0.60,
        sukuk_min=0.30,
        sukuk_max=0.50,
        cash_min=0.05,
        cash_max=0.15,
    ),
    ModelPortfolio.GROWTH: AllocationBand(
        equity_min=0.65,
        equity_max=0.80,
        sukuk_min=0.15,
        sukuk_max=0.30,
        cash_min=0.02,
        cash_max=0.10,
    ),
    ModelPortfolio.AGGRESSIVE_GROWTH: AllocationBand(
        equity_min=0.85,
        equity_max=0.95,
        sukuk_min=0.03,
        sukuk_max=0.13,
        cash_min=0.02,
        cash_max=0.10,
    ),
}


def band_for(model: ModelPortfolio) -> AllocationBand:
    return _MODEL_ALLOCATIONS[model]


# Closed lookup: (RiskProfile, TimeHorizon) → ModelPortfolio.
_PROFILE_HORIZON_MAP: dict[tuple[RiskProfile, TimeHorizon], ModelPortfolio] = {
    (RiskProfile.CONSERVATIVE, TimeHorizon.SHORT): ModelPortfolio.DEFENSIVE,
    (RiskProfile.CONSERVATIVE, TimeHorizon.MEDIUM): ModelPortfolio.DEFENSIVE,
    (RiskProfile.CONSERVATIVE, TimeHorizon.LONG): ModelPortfolio.BALANCED,
    (RiskProfile.BALANCED, TimeHorizon.SHORT): ModelPortfolio.DEFENSIVE,
    (RiskProfile.BALANCED, TimeHorizon.MEDIUM): ModelPortfolio.BALANCED,
    (RiskProfile.BALANCED, TimeHorizon.LONG): ModelPortfolio.GROWTH,
    (RiskProfile.AGGRESSIVE, TimeHorizon.SHORT): ModelPortfolio.BALANCED,
    (RiskProfile.AGGRESSIVE, TimeHorizon.MEDIUM): ModelPortfolio.GROWTH,
    (RiskProfile.AGGRESSIVE, TimeHorizon.LONG): ModelPortfolio.AGGRESSIVE_GROWTH,
}


def map_profile_to_model(
    profile: RiskProfile,
    horizon: TimeHorizon,
    *,
    overrides: dict[tuple[RiskProfile, TimeHorizon], ModelPortfolio] | None = None,
) -> ModelPortfolio:
    """Return the model portfolio for a profile + horizon."""
    if overrides is not None and (profile, horizon) in overrides:
        return overrides[(profile, horizon)]
    return _PROFILE_HORIZON_MAP[(profile, horizon)]


@dataclass(frozen=True)
class FeeBracket:
    """One AUM-bracket fee entry. Flat $ per year over the bracket."""

    aum_min_usd: float
    aum_max_usd: float | None
    """None = open-ended top bracket."""
    annual_fee_pct: float

    def __post_init__(self) -> None:
        if self.aum_min_usd < 0:
            raise ValueError("aum_min_usd must be non-negative")
        if self.aum_max_usd is not None and self.aum_max_usd <= self.aum_min_usd:
            raise ValueError("aum_max_usd must be > aum_min_usd when set")
        if not 0.0 <= self.annual_fee_pct < 0.05:
            # Pin: > 5%/yr reads as performance carry rather than Wakalah.
            raise ValueError("annual_fee_pct must be in [0, 0.05)")


def _default_fee_schedule() -> tuple[FeeBracket, ...]:
    """Tiered Wakalah-style schedule. Conservative defaults."""
    return (
        FeeBracket(aum_min_usd=0, aum_max_usd=50_000.0, annual_fee_pct=0.0050),  # 50bps
        FeeBracket(aum_min_usd=50_000.0, aum_max_usd=500_000.0, annual_fee_pct=0.0035),  # 35bps
        FeeBracket(aum_min_usd=500_000.0, aum_max_usd=None, annual_fee_pct=0.0020),  # 20bps
    )


def annual_fee_for(
    aum_usd: float,
    *,
    schedule: tuple[FeeBracket, ...] | None = None,
) -> float:
    """Compute the blended annual fee on AUM.

    Pinned: the fee schedule is *bracketed*, so the fee is computed
    per bracket and summed — the operator does not get hammered with a
    cliff at bracket boundaries.
    """
    if aum_usd < 0:
        raise ValueError("aum_usd must be non-negative")
    sched = schedule if schedule is not None else _default_fee_schedule()
    if not sched:
        raise ValueError("schedule must be non-empty")
    # Sort by min.
    ordered = sorted(sched, key=lambda b: b.aum_min_usd)
    total_fee = 0.0
    for b in ordered:
        bracket_top = b.aum_max_usd if b.aum_max_usd is not None else aum_usd
        if aum_usd <= b.aum_min_usd:
            continue
        capped = min(aum_usd, bracket_top)
        portion = capped - b.aum_min_usd
        if portion <= 0:
            continue
        total_fee += portion * b.annual_fee_pct
    return total_fee


@dataclass(frozen=True)
class RoboPlan:
    """Output of `build_plan`."""

    plan_id: str
    user_id: str
    risk_profile: RiskProfile
    horizon: TimeHorizon
    model: ModelPortfolio
    band: AllocationBand
    aum_usd: float
    annual_fee_usd: float
    drift_threshold_pct: float


def build_plan(
    *,
    plan_id: str,
    user_id: str,
    profile: RiskProfile,
    horizon: TimeHorizon,
    aum_usd: float,
    drift_threshold_pct: float = 0.05,
    overrides: dict[tuple[RiskProfile, TimeHorizon], ModelPortfolio] | None = None,
    fee_schedule: tuple[FeeBracket, ...] | None = None,
) -> RoboPlan:
    """Map profile + horizon → ModelPortfolio + compute fee + drift trigger."""
    if not plan_id or not plan_id.strip():
        raise ValueError("plan_id must be non-empty")
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be non-empty")
    if aum_usd < 0:
        raise ValueError("aum_usd must be non-negative")
    if not 0.0 < drift_threshold_pct <= 0.30:
        raise ValueError("drift_threshold_pct must be in (0, 0.30]")
    model = map_profile_to_model(profile, horizon, overrides=overrides)
    band = band_for(model)
    fee = annual_fee_for(aum_usd, schedule=fee_schedule)
    return RoboPlan(
        plan_id=plan_id,
        user_id=user_id,
        risk_profile=profile,
        horizon=horizon,
        model=model,
        band=band,
        aum_usd=aum_usd,
        annual_fee_usd=fee,
        drift_threshold_pct=drift_threshold_pct,
    )


def needs_rebalance(
    plan: RoboPlan,
    *,
    actual_equity_pct: float,
    actual_sukuk_pct: float,
    actual_cash_pct: float,
) -> bool:
    """True iff any sleeve has drifted outside its band by more than
    the threshold."""
    for actual, lo, hi in (
        (actual_equity_pct, plan.band.equity_min, plan.band.equity_max),
        (actual_sukuk_pct, plan.band.sukuk_min, plan.band.sukuk_max),
        (actual_cash_pct, plan.band.cash_min, plan.band.cash_max),
    ):
        if actual < 0 or actual > 1:
            raise ValueError("actual percentages must be in [0, 1]")
        if actual < lo - plan.drift_threshold_pct:
            return True
        if actual > hi + plan.drift_threshold_pct:
            return True
    return False


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_MODEL_EMOJI: dict[ModelPortfolio, str] = {
    ModelPortfolio.DEFENSIVE: "🛡️",
    ModelPortfolio.BALANCED: "⚖️",
    ModelPortfolio.GROWTH: "🌱",
    ModelPortfolio.AGGRESSIVE_GROWTH: "🚀",
}


def render_plan(plan: RoboPlan) -> str:
    band = plan.band
    return (
        f"{_MODEL_EMOJI[plan.model]} {plan.plan_id} "
        f"[{plan.model.value}] for {_mask(plan.user_id)}: "
        f"AUM ${plan.aum_usd:,.0f}, "
        f"fee ${plan.annual_fee_usd:,.2f}/yr "
        f"({plan.annual_fee_usd / max(plan.aum_usd, 1) * 10_000:.1f}bps)\n"
        f"  Equity {band.equity_min * 100:.0f}–{band.equity_max * 100:.0f}% / "
        f"Sukuk {band.sukuk_min * 100:.0f}–{band.sukuk_max * 100:.0f}% / "
        f"Cash {band.cash_min * 100:.0f}–{band.cash_max * 100:.0f}%"
    )
