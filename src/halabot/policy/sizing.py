"""Per-asset target weight with hysteresis (REARCHITECTURE B.5, fix R-13).

THE single canonical sizing function. Hysteresis (entry band > exit band) is
the first anti-churn mechanism: a position clears a higher bar to open than to
stay open, so conviction noise around the threshold can't flip it in and out.
The config invariant ``0 ≤ exit < entry < 1`` is validated so the
``1 - exit_band`` denominator is never zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from halabot.belief.schema import BeliefState, Direction
from halabot.risk.engine import RiskState


@dataclass(frozen=True)
class PolicyConfig:
    conviction_entry_band: float = 0.60
    conviction_exit_band: float = 0.45
    max_weight_per_asset: float = 0.20
    max_gross_exposure: float = 1.0  # no implicit leverage (INV-10)
    target_rebalance_threshold: float = 0.05  # min weight change to trade (R-14 / anti-churn)
    max_open_positions: int = 0  # cap on concurrent positions; 0 = unlimited
    relstrength_gate: float = 0.5  # veto buys lagging the benchmark by this much; 0 = off

    def __post_init__(self) -> None:
        if not (0.0 <= self.conviction_exit_band < self.conviction_entry_band < 1.0):
            raise ValueError(
                "require 0 <= conviction_exit_band < conviction_entry_band < 1 "
                f"(got exit={self.conviction_exit_band}, entry={self.conviction_entry_band})"
            )


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def target_weight(b: BeliefState, risk: RiskState, *, held: bool, cfg: PolicyConfig) -> float:
    """Target portfolio weight for one asset (0 = no/zero position)."""
    if b.direction != Direction.LONG_BIAS:
        return 0.0
    # HYSTERESIS: a held position only needs to clear the lower exit band to stay;
    # a new entry must clear the higher entry band.
    threshold = cfg.conviction_exit_band if held else cfg.conviction_entry_band
    if b.conviction < threshold:
        return 0.0
    scale = _clamp(
        (b.conviction - cfg.conviction_exit_band) / (1.0 - cfg.conviction_exit_band), 0.0, 1.0
    )
    raw = scale * cfg.max_weight_per_asset
    raw *= risk.correlation_multiplier(b.asset)
    raw *= risk.volatility_multiplier(b.asset)
    return min(raw, cfg.max_weight_per_asset)
