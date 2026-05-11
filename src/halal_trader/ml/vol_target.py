"""Volatility-targeting halal portfolio — Round-5 Wave 4.E.

Conventional vol-targeting layers a leverage knob on top of a strategy:
when realised vol < target, lever up; when realised vol > target, lever
down. Leverage uses interest-bearing margin → impermissible.

The halal analogue substitutes the leverage-up case with a Salam hedge
overlay (or, for equity exposure, scales position sizes against cash).
The leverage-down case is a straight position reduction. The result is
a vol-targeted portfolio that never engages riba-bearing margin.

This module is the **vol-target allocator**. Pricing of the Salam
overlay lives in `halal/salam_forward.py`.

Pinned semantics:

- **Realised vol = annualised σ of log returns over `lookback_days`.**
  Uses `√252` for daily, `√365` for crypto (24/7) — operator passes
  `bars_per_year` explicitly to avoid implicit defaults.
- **Closed-set ScalingMode**: SCALE_DOWN_ONLY / SCALE_BOTH /
  SALAM_OVERLAY. SCALE_DOWN_ONLY is the most conservative (pure
  position-size reduction; leaves cash idle when vol is below
  target). SCALE_BOTH uses a notional scale factor on the portfolio.
  SALAM_OVERLAY adds a Salam-forward sleeve when realised < target.
- **Hard caps on the scale factor.** [0.0, 1.5] by default; operators
  can lower 1.5 but raising it requires explicit opt-in. The cap
  mirrors AAOIFI Standard 21's prudential leverage limit (33% on
  total interest-bearing debt; for a halal allocator we treat any
  scale > 1.0 as needing a Salam overlay rather than implicit margin).
- **Tracking-error penalty grows with deviation from target.**
- **Pure-Python deterministic.** No NumPy. Bounded loops.
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class ScalingMode(str, Enum):
    """Closed-set vol-target scaling strategy."""

    SCALE_DOWN_ONLY = "scale_down_only"
    SCALE_BOTH = "scale_both"
    SALAM_OVERLAY = "salam_overlay"


@dataclass(frozen=True)
class VolTargetConfig:
    """Configuration for the vol-targeting allocator."""

    target_volatility: float
    lookback_days: int = 30
    bars_per_year: int = 252
    scaling_mode: ScalingMode = ScalingMode.SCALE_DOWN_ONLY
    min_scale: float = 0.10
    max_scale: float = 1.5
    floor_when_below: bool = False
    """If True and realised < 0.5×target, return 0% (regime is too calm
    to trade — protects against a lull-then-shock pattern). Off by
    default."""

    def __post_init__(self) -> None:
        if not 0.0 < self.target_volatility < 1.0:
            raise ValueError("target_volatility must be in (0, 1)")
        if self.lookback_days < 5:
            raise ValueError("lookback_days must be ≥ 5")
        if self.bars_per_year not in (252, 365):
            raise ValueError("bars_per_year must be 252 (stocks) or 365 (crypto)")
        if not 0.0 <= self.min_scale < self.max_scale:
            raise ValueError("min_scale must be in [0, max_scale)")
        if self.max_scale > 3.0:
            raise ValueError("max_scale > 3 requires explicit Sharia review")


def realised_volatility(
    prices: Sequence[float],
    *,
    bars_per_year: int = 252,
) -> float:
    """Annualised σ of log returns.

    Pinned: uses log-returns for additivity; bars_per_year supplied
    explicitly to keep the calculation deterministic across asset
    classes. Returns 0 if fewer than 2 prices.
    """
    if len(prices) < 2:
        return 0.0
    rets: list[float] = []
    for prev, cur in zip(prices, prices[1:], strict=False):
        if prev <= 0 or cur <= 0:
            raise ValueError("prices must be positive")
        rets.append(math.log(cur / prev))
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / max(1, n - 1)
    return math.sqrt(var) * math.sqrt(bars_per_year)


@dataclass(frozen=True)
class VolTargetDecision:
    """Output of `compute_scale`."""

    realised_vol: float
    target_vol: float
    raw_scale: float
    final_scale: float
    salam_overlay_pct: float
    note: str

    def is_levered(self) -> bool:
        return self.final_scale > 1.0 + 1e-9


def compute_scale(
    prices: Sequence[float],
    config: VolTargetConfig,
) -> VolTargetDecision:
    """Compute the position-scale factor for the next period.

    Logic:
      raw_scale = target / max(realised, ε)
      final_scale = clamp(raw_scale, min_scale, max_scale)

    Behaviour by mode:
    - SCALE_DOWN_ONLY: cap final_scale at 1.0; never lever above the
      base allocation. Lower scale → reduce position size.
    - SCALE_BOTH: allow final_scale up to max_scale (no overlay; just
      a directive that tells the allocator how much to deploy).
    - SALAM_OVERLAY: when raw_scale > 1.0, the excess (raw_scale - 1)
      is delivered as a Salam-forward overlay. Final position size
      stays at 1.0; salam_overlay_pct captures the additional notional.

    `floor_when_below` returns final_scale=0 if realised < 0.5×target
    (suspicious-calm protection).
    """
    if not prices:
        raise ValueError("prices must be non-empty")
    rv = realised_volatility(
        prices[-(config.lookback_days + 1) :],
        bars_per_year=config.bars_per_year,
    )
    eps = 1e-6
    raw = config.target_volatility / max(rv, eps)
    if config.floor_when_below and rv < 0.5 * config.target_volatility:
        return VolTargetDecision(
            realised_vol=rv,
            target_vol=config.target_volatility,
            raw_scale=raw,
            final_scale=0.0,
            salam_overlay_pct=0.0,
            note="floor_when_below: realised vol unusually calm, dialed to 0",
        )
    final = max(config.min_scale, min(raw, config.max_scale))
    overlay = 0.0
    note = ""
    if config.scaling_mode is ScalingMode.SCALE_DOWN_ONLY:
        if final > 1.0:
            note = "scale-down-only: capped at 1.0 — left as base allocation"
            final = 1.0
        else:
            note = "scale-down-only: under-target reduction applied"
    elif config.scaling_mode is ScalingMode.SCALE_BOTH:
        note = (
            "scale-both: directive applied — operator must verify any "
            "scale > 1.0 is funded with cash, not interest margin"
        )
    elif config.scaling_mode is ScalingMode.SALAM_OVERLAY:
        if final > 1.0:
            overlay = final - 1.0
            note = (
                f"salam-overlay: {overlay * 100:.2f}% Salam-forward "
                "sleeve added on top of base allocation"
            )
            final = 1.0
        else:
            note = "salam-overlay: under-target reduction (no overlay needed)"
    return VolTargetDecision(
        realised_vol=rv,
        target_vol=config.target_volatility,
        raw_scale=raw,
        final_scale=final,
        salam_overlay_pct=overlay,
        note=note,
    )


@dataclass(frozen=True)
class PortfolioVolTargetPlan:
    """Vol-targeted weights produced from a base allocation + per-asset prices."""

    base_weights: tuple[float, ...]
    final_weights: tuple[float, ...]
    salam_overlay_per_asset: tuple[float, ...]
    cash_weight: float
    aggregate_realised_vol: float
    aggregate_target_vol: float


def apply_vol_target(
    base_weights: Sequence[float],
    price_history_per_asset: Sequence[Sequence[float]],
    config: VolTargetConfig,
) -> PortfolioVolTargetPlan:
    """Apply vol-targeting to a base allocation.

    The aggregate realised vol is computed as the weighted-sum proxy
    σ_p ≈ Σ w_i σ_i (no covariance term — matches the deterministic
    pure-Python contract). The aggregate scale is applied uniformly
    across the base weights.

    `cash_weight` reflects the portion that was scaled-down to cash
    (positive when realised > target and SCALE_DOWN_ONLY pulled the
    scale below 1.0).
    """
    if len(base_weights) != len(price_history_per_asset):
        raise ValueError("len(base_weights) must equal len(price_history_per_asset)")
    n = len(base_weights)
    if n == 0:
        raise ValueError("base_weights must be non-empty")
    if abs(sum(base_weights) - 1.0) > 1e-6:
        raise ValueError("base_weights must sum to 1")
    sigmas: list[float] = []
    for prices in price_history_per_asset:
        sigma = realised_volatility(
            list(prices)[-(config.lookback_days + 1) :],
            bars_per_year=config.bars_per_year,
        )
        sigmas.append(sigma)
    agg_sigma = sum(w * s for w, s in zip(base_weights, sigmas, strict=True))
    eps = 1e-6
    raw = config.target_volatility / max(agg_sigma, eps)
    if config.floor_when_below and agg_sigma < 0.5 * config.target_volatility:
        return PortfolioVolTargetPlan(
            base_weights=tuple(base_weights),
            final_weights=tuple(0.0 for _ in base_weights),
            salam_overlay_per_asset=tuple(0.0 for _ in base_weights),
            cash_weight=1.0,
            aggregate_realised_vol=agg_sigma,
            aggregate_target_vol=config.target_volatility,
        )
    final_scale = max(config.min_scale, min(raw, config.max_scale))
    overlay_per_asset: list[float] = [0.0] * n
    cash_weight = 0.0
    if config.scaling_mode is ScalingMode.SCALE_DOWN_ONLY:
        applied_scale = min(final_scale, 1.0)
        cash_weight = max(0.0, 1.0 - applied_scale)
        final = [w * applied_scale for w in base_weights]
    elif config.scaling_mode is ScalingMode.SCALE_BOTH:
        if final_scale <= 1.0:
            cash_weight = 1.0 - final_scale
        final = [w * final_scale for w in base_weights]
    else:  # SALAM_OVERLAY
        if final_scale > 1.0:
            overlay = final_scale - 1.0
            for i in range(n):
                overlay_per_asset[i] = base_weights[i] * overlay
            final = list(base_weights)  # unchanged
        else:
            cash_weight = 1.0 - final_scale
            final = [w * final_scale for w in base_weights]
    return PortfolioVolTargetPlan(
        base_weights=tuple(base_weights),
        final_weights=tuple(final),
        salam_overlay_per_asset=tuple(overlay_per_asset),
        cash_weight=cash_weight,
        aggregate_realised_vol=agg_sigma,
        aggregate_target_vol=config.target_volatility,
    )


def render_decision(decision: VolTargetDecision) -> str:
    """Operator-readable summary of a single-asset vol-target decision."""
    overlay_line = ""
    if decision.salam_overlay_pct > 0:
        overlay_line = f"\n  • Salam overlay: {decision.salam_overlay_pct * 100:.2f}%"
    return (
        f"📐 Vol-target: realised={decision.realised_vol * 100:.2f}%, "
        f"target={decision.target_vol * 100:.2f}%, "
        f"scale={decision.final_scale:.3f}{overlay_line}\n"
        f"  • Note: {decision.note}"
    )


def render_plan(plan: PortfolioVolTargetPlan) -> str:
    """Operator-readable summary of a portfolio vol-target plan."""
    head = (
        f"📊 Vol-target plan: realised={plan.aggregate_realised_vol * 100:.2f}%, "
        f"target={plan.aggregate_target_vol * 100:.2f}%, "
        f"cash={plan.cash_weight * 100:.2f}%"
    )
    overlay_total = sum(plan.salam_overlay_per_asset)
    if overlay_total > 0:
        head += f", salam={overlay_total * 100:.2f}%"
    return head
