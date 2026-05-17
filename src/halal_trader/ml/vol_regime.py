"""Halal volatility regime detector — Round-5 Wave 13.A.

Vol regime is the load-bearing input to tail-risk hedging (Wave 13.B–E)
+ position-sizing (`crypto/components.py`). The bot already has a
rule-based regime detector for *direction* (`crypto/regime.py`) and a
causal Bayesian net for macro→regime (`ml/causal_regime.py`); this
module adds the vol-specific overlay.

Detection is deliberately rule-based + interpretable: realised vol +
vol-of-vol over rolling windows, classified into a closed ladder of
four regimes.

Pinned semantics:

- **Closed-set VolRegime ladder** (LOW / NORMAL / ELEVATED / CRISIS).
- **Default annualisation factor 252** (US trading days). Tunable.
- **Hysteresis pin**: regime classification uses *previous* regime as
  a tie-breaker to avoid jitter at boundary; the previous regime
  is supplied by the caller. Without it the engine still returns a
  classification but flags `is_borderline=True`.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class VolRegime(str, Enum):
    """Closed-set volatility regime ladder."""

    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"


_REGIME_ORDER = {
    VolRegime.LOW: 0,
    VolRegime.NORMAL: 1,
    VolRegime.ELEVATED: 2,
    VolRegime.CRISIS: 3,
}


@dataclass(frozen=True)
class VolPolicy:
    """Operator-tunable thresholds (annualised vol)."""

    low_threshold: float = 0.10
    normal_threshold: float = 0.20
    elevated_threshold: float = 0.40
    annualisation_factor: int = 252
    hysteresis: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 < self.low_threshold < self.normal_threshold < self.elevated_threshold:
            raise ValueError(
                "thresholds must be strictly increasing positive: low < normal < elevated"
            )
        if self.elevated_threshold >= 5.0:
            raise ValueError("elevated_threshold should be a fraction (e.g. 0.40 = 40%)")
        if self.annualisation_factor <= 0:
            raise ValueError("annualisation_factor must be positive")
        if not 0.0 <= self.hysteresis <= 0.10:
            raise ValueError("hysteresis must be in [0, 0.10]")


@dataclass(frozen=True)
class RegimeAssessment:
    """Result of running vol detection."""

    regime: VolRegime
    annualised_vol: float
    realised_returns_count: int
    is_borderline: bool

    def __post_init__(self) -> None:
        if self.annualised_vol < 0:
            raise ValueError("annualised_vol cannot be negative")
        if self.realised_returns_count < 0:
            raise ValueError("realised_returns_count cannot be negative")


def _stddev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def annualised_vol(returns: Sequence[float], factor: int = 252) -> float:
    """Compute annualised standard-deviation of returns."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    if len(returns) < 2:
        return 0.0
    return _stddev(returns) * math.sqrt(factor)


def _classify_raw(vol: float, policy: VolPolicy) -> VolRegime:
    if vol < policy.low_threshold:
        return VolRegime.LOW
    if vol < policy.normal_threshold:
        return VolRegime.NORMAL
    if vol < policy.elevated_threshold:
        return VolRegime.ELEVATED
    return VolRegime.CRISIS


def _is_borderline(vol: float, policy: VolPolicy) -> bool:
    """True if vol is within `hysteresis` of any threshold."""
    for threshold in (
        policy.low_threshold,
        policy.normal_threshold,
        policy.elevated_threshold,
    ):
        if abs(vol - threshold) <= policy.hysteresis:
            return True
    return False


def detect(
    returns: Sequence[float],
    *,
    previous_regime: VolRegime | None = None,
    policy: VolPolicy | None = None,
) -> RegimeAssessment:
    """Detect the current vol regime from a sequence of period returns."""
    pol = policy if policy is not None else VolPolicy()
    if not returns:
        return RegimeAssessment(
            regime=previous_regime if previous_regime is not None else VolRegime.NORMAL,
            annualised_vol=0.0,
            realised_returns_count=0,
            is_borderline=False,
        )

    vol = annualised_vol(returns, pol.annualisation_factor)
    regime = _classify_raw(vol, pol)
    borderline = _is_borderline(vol, pol)

    # Hysteresis: if borderline AND previous regime is one step adjacent,
    # snap to previous to avoid jitter.
    if borderline and previous_regime is not None:
        if abs(_REGIME_ORDER[previous_regime] - _REGIME_ORDER[regime]) <= 1:
            regime = previous_regime

    return RegimeAssessment(
        regime=regime,
        annualised_vol=vol,
        realised_returns_count=len(returns),
        is_borderline=borderline,
    )


def transitioned(previous: RegimeAssessment | None, current: RegimeAssessment) -> bool:
    """True if the regime moved across this assessment."""
    if previous is None:
        return False
    return previous.regime is not current.regime


def render_assessment(assessment: RegimeAssessment) -> str:
    emoji = {
        VolRegime.LOW: "🟢",
        VolRegime.NORMAL: "🟡",
        VolRegime.ELEVATED: "🟠",
        VolRegime.CRISIS: "🔴",
    }[assessment.regime]
    border = " (borderline)" if assessment.is_borderline else ""
    return (
        f"{emoji} regime={assessment.regime.value} "
        f"annual_vol={assessment.annualised_vol:.4f} "
        f"n={assessment.realised_returns_count}{border}"
    )
