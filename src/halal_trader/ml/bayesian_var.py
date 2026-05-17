"""Bayesian VaR with skew + kurtosis (Cornish-Fisher) — Round-5 Wave 14.A.

Standard VaR assumes normal returns. Real return distributions are
typically negatively skewed + fat-tailed; ignoring this systematically
under-estimates downside risk. The **Cornish-Fisher expansion** is
the textbook adjustment: it modifies the standard-normal quantile
using sample skewness + excess kurtosis.

This module ships the pure-Python (math + statistics) Bayesian-style
VaR estimator. Optional numpy import for speed but not required.

Pinned semantics:

- **Cornish-Fisher quantile** for VaR adjustment.
- **Skewness uses Fisher–Pearson definition** (third standardised
  moment), kurtosis is *excess* kurtosis (fourth standardised moment
  − 3, so Gaussian = 0).
- **Empty / single-sample inputs return 0.0**.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _moment_about_mean(xs: Sequence[float], k: int) -> float:
    if len(xs) == 0:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** k for x in xs) / len(xs)


def stddev(xs: Sequence[float]) -> float:
    """Sample standard deviation (Bessel-corrected, n-1)."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def skewness(xs: Sequence[float]) -> float:
    """Fisher–Pearson sample skewness."""
    n = len(xs)
    if n < 3:
        return 0.0
    s = stddev(xs)
    if s == 0:
        return 0.0
    m3 = _moment_about_mean(xs, 3)
    return m3 / (s**3)


def excess_kurtosis(xs: Sequence[float]) -> float:
    """Excess kurtosis (Gaussian = 0)."""
    n = len(xs)
    if n < 4:
        return 0.0
    s = stddev(xs)
    if s == 0:
        return 0.0
    m4 = _moment_about_mean(xs, 4)
    return m4 / (s**4) - 3.0


def _normal_quantile(p: float) -> float:
    """Inverse CDF of standard normal — Beasley-Springer-Moro approximation."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    # Beasley-Springer-Moro algorithm — fast + accurate to ~1e-9
    a = (
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239,
    )
    b = (
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    )
    c = (
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    )
    d = (
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def cornish_fisher_quantile(z: float, *, skew: float, excess_kurt: float) -> float:
    """Apply Cornish-Fisher correction to a normal quantile."""
    return (
        z
        + (z**2 - 1) * skew / 6.0
        + (z**3 - 3 * z) * excess_kurt / 24.0
        - (2 * z**3 - 5 * z) * (skew**2) / 36.0
    )


@dataclass(frozen=True)
class BayesianVarResult:
    """Result of running Bayesian VaR estimation."""

    alpha: float
    var_normal: float
    var_cornish_fisher: float
    sample_skewness: float
    sample_excess_kurtosis: float
    n_samples: int

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.var_normal < 0 or self.var_cornish_fisher < 0:
            raise ValueError("VaR values must be non-negative")
        if self.n_samples < 0:
            raise ValueError("n_samples must be non-negative")


def bayesian_var(returns: Sequence[float], *, alpha: float = 0.05) -> BayesianVarResult:
    """Compute Cornish-Fisher-adjusted VaR + summary statistics."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = len(returns)
    if n < 2:
        return BayesianVarResult(
            alpha=alpha,
            var_normal=0.0,
            var_cornish_fisher=0.0,
            sample_skewness=0.0,
            sample_excess_kurtosis=0.0,
            n_samples=n,
        )

    mu = _mean(returns)
    sigma = stddev(returns)
    skew = skewness(returns)
    kurt = excess_kurtosis(returns)

    z = _normal_quantile(alpha)
    var_n = max(0.0, -(mu + z * sigma))

    z_cf = cornish_fisher_quantile(z, skew=skew, excess_kurt=kurt)
    var_cf = max(0.0, -(mu + z_cf * sigma))

    return BayesianVarResult(
        alpha=alpha,
        var_normal=var_n,
        var_cornish_fisher=var_cf,
        sample_skewness=skew,
        sample_excess_kurtosis=kurt,
        n_samples=n,
    )


def render_result(result: BayesianVarResult) -> str:
    return (
        f"Bayesian VaR α={result.alpha:.3f}: "
        f"normal={result.var_normal:.4f} | "
        f"CF={result.var_cornish_fisher:.4f} "
        f"(skew={result.sample_skewness:+.3f}, "
        f"excess_kurt={result.sample_excess_kurtosis:+.3f}, "
        f"n={result.n_samples})"
    )
