"""Tail dependence + copula primitives — Round-5 Wave 14.F.

Pearson correlation captures linear co-movement but says nothing
about *tail* behaviour: two assets can have moderate Pearson
correlation but co-crash hard during extreme moves. **Tail
dependence** is the conditional probability that one asset is in
its extreme tail given the other is — it's what risk managers
care about when sizing tail-hedges.

This module ships:

- **Empirical tail-dependence coefficient** for upper + lower tails.
- **Gaussian copula CDF** for benchmarking (Gaussian copulas have
  zero asymptotic tail dependence — useful as a null hypothesis).
- **Clayton copula tail-dependence formula** for fat-tailed
  alternatives.

Pinned semantics:

- **Closed-set TailDirection ladder** (LOWER / UPPER).
- **Empirical estimator** uses the (1-q) / q quantile threshold;
  default q = 0.05.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class TailDirection(str, Enum):
    """Closed-set tail directions."""

    LOWER = "lower"
    UPPER = "upper"


def _empirical_quantile(values: Sequence[float], q: float) -> float:
    """Empirical q-quantile via simple sort + index."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if not values:
        raise ValueError("values must be non-empty")
    sorted_v = sorted(values)
    idx = int(q * (len(sorted_v) - 1))
    return sorted_v[idx]


def empirical_tail_dependence(
    a: Sequence[float],
    b: Sequence[float],
    *,
    q: float = 0.05,
    direction: TailDirection = TailDirection.LOWER,
) -> float:
    """Empirical tail-dependence coefficient.

    For LOWER: λ_L(q) = P(B ≤ q-quantile_B | A ≤ q-quantile_A).
    For UPPER: λ_U(q) = P(B ≥ (1-q)-quantile_B | A ≥ (1-q)-quantile_A).

    Returns a number in [0, 1]; close to 1 means tail-co-movement.
    """
    if len(a) != len(b):
        raise ValueError("a and b must have same length")
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    if len(a) < 20:
        raise ValueError("at least 20 observations required for tail estimation")

    if direction is TailDirection.LOWER:
        threshold_a = _empirical_quantile(a, q)
        threshold_b = _empirical_quantile(b, q)
        in_a_tail = sum(1 for v in a if v <= threshold_a)
        co_tail = sum(1 for i in range(len(a)) if a[i] <= threshold_a and b[i] <= threshold_b)
    else:
        threshold_a = _empirical_quantile(a, 1.0 - q)
        threshold_b = _empirical_quantile(b, 1.0 - q)
        in_a_tail = sum(1 for v in a if v >= threshold_a)
        co_tail = sum(1 for i in range(len(a)) if a[i] >= threshold_a and b[i] >= threshold_b)

    if in_a_tail == 0:
        return 0.0
    return co_tail / in_a_tail


# --- Gaussian copula -------------------------------------------------


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gaussian_copula_cdf(u: float, v: float, *, rho: float) -> float:
    """Bivariate Gaussian copula CDF — Drezner-Wesolowsky approximation.

    Inputs u, v are uniform-marginal CDF values in (0, 1); rho is the
    correlation. Returns ``C(u, v; rho)``.
    """
    if not 0.0 < u < 1.0 or not 0.0 < v < 1.0:
        raise ValueError("u and v must be in (0, 1)")
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be in (-1, 1)")

    # Inverse standard normal via approximate (Beasley-Springer-Moro)
    from halal_trader.ml.bayesian_var import _normal_quantile

    x = _normal_quantile(u)
    y = _normal_quantile(v)

    # Drezner-Wesolowsky bivariate normal CDF (via numerical integration approx)
    return _bvn(x, y, rho)


def _bvn(x: float, y: float, rho: float) -> float:
    """Bivariate normal CDF P(X ≤ x, Y ≤ y) for standard normals with corr rho."""
    if rho == 0:
        return _norm_cdf(x) * _norm_cdf(y)
    # Owen-style series: simple polynomial expansion for moderate rho.
    # For |rho| < 1 we use a Drezner formula; here we use numerical
    # integration via Simpson's rule for compactness.
    n_steps = 200
    h = rho / n_steps
    total = 0.0
    for i in range(n_steps + 1):
        t = h * i
        if 1 - t * t <= 0:
            continue
        weight = 1 if i in (0, n_steps) else (4 if i % 2 else 2)
        denom = 2 * math.pi * math.sqrt(1 - t * t)
        exponent = -(x * x - 2 * t * x * y + y * y) / (2 * (1 - t * t))
        total += weight * math.exp(exponent) / denom
    integral = total * h / 3
    return _norm_cdf(x) * _norm_cdf(y) + integral


def gaussian_copula_lower_tail(rho: float) -> float:
    """Gaussian copula has λ_L = 0 in the limit. For any q, λ_L(q) → 0 as q → 0."""
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be in (-1, 1)")
    return 0.0


# --- Clayton copula -------------------------------------------------


def clayton_copula_cdf(u: float, v: float, *, theta: float) -> float:
    """Clayton copula CDF: ``(u^-θ + v^-θ - 1)^(-1/θ)``."""
    if not 0.0 < u < 1.0 or not 0.0 < v < 1.0:
        raise ValueError("u and v must be in (0, 1)")
    if theta <= 0:
        raise ValueError("theta must be > 0 (Clayton requires positive dependence)")
    val = (u ** (-theta)) + (v ** (-theta)) - 1.0
    if val <= 0:
        return 0.0
    return val ** (-1.0 / theta)


def clayton_lower_tail(theta: float) -> float:
    """Closed-form lower-tail dependence of Clayton: ``2^(-1/θ)``."""
    if theta <= 0:
        raise ValueError("theta must be > 0")
    return 2.0 ** (-1.0 / theta)


# --- Reporting -----------------------------------------------------


@dataclass(frozen=True)
class TailDependenceReport:
    """Combined tail-dependence summary."""

    lower_tail: float
    upper_tail: float
    n_observations: int
    quantile_threshold: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.lower_tail <= 1.0:
            raise ValueError("lower_tail must be in [0, 1]")
        if not 0.0 <= self.upper_tail <= 1.0:
            raise ValueError("upper_tail must be in [0, 1]")
        if self.n_observations <= 0:
            raise ValueError("n_observations must be positive")
        if not 0.0 < self.quantile_threshold < 1.0:
            raise ValueError("quantile_threshold must be in (0, 1)")


def estimate_tail_dependence(
    a: Sequence[float], b: Sequence[float], *, q: float = 0.05
) -> TailDependenceReport:
    lower = empirical_tail_dependence(a, b, q=q, direction=TailDirection.LOWER)
    upper = empirical_tail_dependence(a, b, q=q, direction=TailDirection.UPPER)
    return TailDependenceReport(
        lower_tail=lower,
        upper_tail=upper,
        n_observations=len(a),
        quantile_threshold=q,
    )


def render_report(r: TailDependenceReport) -> str:
    return (
        f"Tail dependence (q={r.quantile_threshold:.3f}, n={r.n_observations}): "
        f"lower={r.lower_tail:.3f}, upper={r.upper_tail:.3f}"
    )
