"""Correlation regime modeling — Round-5 Wave 14.D.

Asset correlations are not stable. In normal regimes, diversification
works — equities and bonds are roughly uncorrelated. In crisis
regimes ("everything-down events"), correlations spike toward 1 and
diversification fails. The bot's portfolio risk model needs to track
this regime to size positions correctly.

This module ships the **correlation-regime detector**: given a
recent window of returns, classify the prevailing correlation regime
and surface the average pairwise correlation.

Pinned semantics:

- **Closed-set CorrelationRegime ladder** (DECORRELATED / NORMAL /
  ELEVATED / CRISIS_CORRELATED).
- **Average pairwise correlation** is the headline number; the
  regime classification thresholds operate on it.
- **Hysteresis** — borderline classifications snap to previous
  regime (suppresses flip-flopping at thresholds).
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum


class CorrelationRegime(str, Enum):
    """Closed-set correlation regimes."""

    DECORRELATED = "decorrelated"
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS_CORRELATED = "crisis_correlated"


_REGIME_ORDER = {
    CorrelationRegime.DECORRELATED: 0,
    CorrelationRegime.NORMAL: 1,
    CorrelationRegime.ELEVATED: 2,
    CorrelationRegime.CRISIS_CORRELATED: 3,
}


@dataclass(frozen=True)
class CorrelationPolicy:
    """Operator-tunable thresholds."""

    decorrelated_threshold: float = 0.20
    normal_threshold: float = 0.40
    elevated_threshold: float = 0.70
    hysteresis: float = 0.05

    def __post_init__(self) -> None:
        if (
            not 0.0
            <= self.decorrelated_threshold
            < self.normal_threshold
            < self.elevated_threshold
            < 1.0
        ):
            raise ValueError("thresholds must be increasing in [0, 1)")
        if not 0.0 <= self.hysteresis <= 0.20:
            raise ValueError("hysteresis must be in [0, 0.20]")


def _stddev(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Sample Pearson correlation between two series."""
    if len(a) != len(b):
        raise ValueError("series lengths must match")
    n = len(a)
    if n < 2:
        return 0.0
    sa = _stddev(a)
    sb = _stddev(b)
    if sa == 0 or sb == 0:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (n - 1)
    return max(-1.0, min(1.0, cov / (sa * sb)))


def average_pairwise_correlation(returns: Mapping[str, Sequence[float]]) -> float:
    """Compute the average of all unique pairwise correlations."""
    symbols = sorted(returns.keys())
    if len(symbols) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            total += pearson(returns[symbols[i]], returns[symbols[j]])
            count += 1
    return total / count if count > 0 else 0.0


@dataclass(frozen=True)
class RegimeAssessment:
    """Result of running correlation-regime detection."""

    regime: CorrelationRegime
    average_correlation: float
    n_assets: int
    is_borderline: bool

    def __post_init__(self) -> None:
        if not -1.0 <= self.average_correlation <= 1.0:
            raise ValueError("average_correlation must be in [-1, 1]")
        if self.n_assets < 0:
            raise ValueError("n_assets must be non-negative")


def _classify_raw(corr: float, policy: CorrelationPolicy) -> CorrelationRegime:
    if corr < policy.decorrelated_threshold:
        return CorrelationRegime.DECORRELATED
    if corr < policy.normal_threshold:
        return CorrelationRegime.NORMAL
    if corr < policy.elevated_threshold:
        return CorrelationRegime.ELEVATED
    return CorrelationRegime.CRISIS_CORRELATED


def _is_borderline(corr: float, policy: CorrelationPolicy) -> bool:
    for t in (
        policy.decorrelated_threshold,
        policy.normal_threshold,
        policy.elevated_threshold,
    ):
        if abs(corr - t) <= policy.hysteresis:
            return True
    return False


def detect(
    returns: Mapping[str, Sequence[float]],
    *,
    previous_regime: CorrelationRegime | None = None,
    policy: CorrelationPolicy | None = None,
) -> RegimeAssessment:
    """Detect the current correlation regime."""
    pol = policy if policy is not None else CorrelationPolicy()
    n_assets = len(returns)

    if n_assets < 2:
        return RegimeAssessment(
            regime=previous_regime if previous_regime is not None else CorrelationRegime.NORMAL,
            average_correlation=0.0,
            n_assets=n_assets,
            is_borderline=False,
        )

    corr = average_pairwise_correlation(returns)
    regime = _classify_raw(corr, pol)
    borderline = _is_borderline(corr, pol)

    if borderline and previous_regime is not None:
        if abs(_REGIME_ORDER[previous_regime] - _REGIME_ORDER[regime]) <= 1:
            regime = previous_regime

    return RegimeAssessment(
        regime=regime,
        average_correlation=corr,
        n_assets=n_assets,
        is_borderline=borderline,
    )


def render_assessment(a: RegimeAssessment) -> str:
    emoji = {
        CorrelationRegime.DECORRELATED: "🟢",
        CorrelationRegime.NORMAL: "🟡",
        CorrelationRegime.ELEVATED: "🟠",
        CorrelationRegime.CRISIS_CORRELATED: "🔴",
    }[a.regime]
    border = " (borderline)" if a.is_borderline else ""
    return (
        f"{emoji} regime={a.regime.value} "
        f"avg_corr={a.average_correlation:+.3f} "
        f"n={a.n_assets}{border}"
    )
