"""Probabilistic & Deflated Sharpe Ratio (López de Prado).

A raw Sharpe ratio from a short or non-normal return series, or one cherry-
picked from many trials, routinely overstates real edge. These estimators
correct for that:

- **PSR** — the probability the *true* Sharpe exceeds a benchmark (default 0),
  given the sample length, skew and kurtosis. A short, fat-tailed track needs
  a much higher observed Sharpe to clear the same PSR.
- **DSR** — PSR against a *deflated* benchmark equal to the expected maximum
  Sharpe under the null across ``n_trials`` independent attempts. This is the
  multiple-testing correction: try 100 strategies and the best one's Sharpe is
  inflated by selection alone.

Pure functions (numpy + stdlib only — no scipy). Used to gate backtests,
walk-forward folds, and the prompt-evolution GA so we don't promote noise.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

_NORM = NormalDist()
_EULER = 0.5772156649015329  # Euler–Mascheroni constant


def _sharpe_and_moments(returns: Any) -> tuple[int, float, float, float] | None:
    """(n, per-period sample Sharpe, skew, full-kurtosis) or None if degenerate."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 3:
        return None
    sd_sample = float(r.std(ddof=1))
    sd_pop = float(r.std())
    if sd_sample == 0.0 or sd_pop == 0.0:
        return None
    sr = float(r.mean()) / sd_sample  # per-period sample Sharpe
    z = (r - r.mean()) / sd_pop
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))  # full kurtosis (3.0 for a normal)
    return n, sr, skew, kurt


def _sharpe_estimator_variance(n: int, sr: float, skew: float, kurt: float) -> float:
    """Variance of the Sharpe estimator (the PSR denominator), per period."""
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    return max(denom / (n - 1), 0.0)


def probabilistic_sharpe_ratio(returns: Any, sr_benchmark: float = 0.0) -> float:
    """Probability the true per-period Sharpe exceeds ``sr_benchmark`` (0..1)."""
    m = _sharpe_and_moments(returns)
    if m is None:
        return 0.0
    n, sr, skew, kurt = m
    denom_var = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    if denom_var <= 0:
        return 0.0
    stat = float((sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom_var))
    return float(_NORM.cdf(stat))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum per-period Sharpe under the null across ``n_trials``.

    The deflation benchmark for DSR. Zero when there's effectively one trial.
    """
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    sd = np.sqrt(sr_variance)
    a = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    b = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * np.e))
    return float(sd * ((1.0 - _EULER) * a + _EULER * b))


def deflated_sharpe_ratio(
    returns: Any, n_trials: int = 1, sr_variance: float | None = None
) -> float:
    """PSR against the expected-max-Sharpe benchmark for ``n_trials`` (0..1).

    ``sr_variance`` is the variance of the Sharpe estimates ACROSS trials; when
    unknown (a single track), it's estimated from the track's own estimator
    variance — a conservative fallback.
    """
    m = _sharpe_and_moments(returns)
    if m is None:
        return 0.0
    if sr_variance is None:
        n, sr, skew, kurt = m
        sr_variance = _sharpe_estimator_variance(n, sr, skew, kurt)
    benchmark = expected_max_sharpe(sr_variance, n_trials)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=benchmark)


def passes_sharpe_gate(
    returns: Any,
    *,
    n_trials: int = 1,
    sr_variance: float | None = None,
    min_prob: float = 0.95,
) -> bool:
    """True if the (deflated, when multi-trial) Sharpe is significant.

    Uses DSR when ``n_trials > 1`` (multiple-testing correction), else PSR.
    """
    if n_trials > 1:
        return deflated_sharpe_ratio(returns, n_trials, sr_variance) >= min_prob
    return probabilistic_sharpe_ratio(returns) >= min_prob
