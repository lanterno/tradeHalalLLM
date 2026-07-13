"""Forecast-evaluation primitives (Phase 0 of the quant roadmap).

The honest "are these predicted highs/lows any good?" toolkit, shared by the
scorecard and the backtests so every level/band family is judged by the same
yardstick. Quantile forecasts are scored with the pinball loss, interval
forecasts with empirical coverage (PICP) and the Winkler interval score
(width plus a miss penalty, so tight-but-wrong bands can't game coverage).
Whether a band's *breach rate* is statistically consistent with its nominal
rate — and whether breaches cluster in time, the classic sign of a vol model
that lags regimes — is tested with the Kupiec proportion-of-failures LR test
and the Christoffersen independence / conditional-coverage LR tests.
``coverage_by_bucket`` splits coverage by arbitrary labels (e.g. vol-regime
buckets) for conditional-coverage reporting.

Pure numpy + stdlib by design: no scipy, no pandas. The only distribution
needed is the chi-square survival function at 1 and 2 degrees of freedom,
which has closed forms (``erfc(sqrt(x/2))`` and ``exp(-x/2)``) — see
``_chi2_sf``. All array inputs accept any 1-d sequence/ndarray of floats and
are normalized with ``np.asarray``; empty or length-mismatched inputs raise
``ValueError`` rather than silently scoring garbage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = Sequence[float] | npt.NDArray[Any]
"""Any 1-d float-coercible input; normalized internally via ``np.asarray``."""

BreachArray = Sequence[bool] | Sequence[int] | Sequence[float] | npt.NDArray[Any]
"""A 1-d breach-indicator series; entries must be boolean or 0/1."""


@dataclass(frozen=True, slots=True)
class LRTestResult:
    """Outcome of a likelihood-ratio test: the LR statistic and its p-value."""

    lr_stat: float
    p_value: float


@dataclass(frozen=True, slots=True)
class BucketCoverage:
    """Per-bucket interval coverage: sample count ``n`` and PICP ``coverage``."""

    n: int
    coverage: float


def _as_float_1d(x: FloatArray, name: str) -> npt.NDArray[np.float64]:
    """Coerce ``x`` to a non-empty 1-d float64 array or raise ``ValueError``."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _as_breach_1d(breaches: BreachArray) -> npt.NDArray[np.int64]:
    """Coerce a bool/0-1 breach series to a non-empty 1-d int64 array of {0, 1}."""
    arr = np.asarray(breaches)
    if arr.ndim != 1:
        raise ValueError(f"breaches must be 1-dimensional, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError("breaches must be non-empty")
    vals = arr.astype(np.float64)
    if not bool(np.isin(vals, (0.0, 1.0)).all()):
        raise ValueError("breaches must contain only booleans or 0/1 values")
    return vals.astype(np.int64)


def _validated_interval(
    y_true: FloatArray, lower: FloatArray, upper: FloatArray
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Coerce and length-check an (observations, lower, upper) interval triple."""
    y = _as_float_1d(y_true, "y_true")
    lo = _as_float_1d(lower, "lower")
    hi = _as_float_1d(upper, "upper")
    if not (y.size == lo.size == hi.size):
        raise ValueError(f"length mismatch: y_true={y.size}, lower={lo.size}, upper={hi.size}")
    return y, lo, hi


def _xlogy(x: float, y: float) -> float:
    """``x * log(y)`` with the MLE convention ``0 * log(0) == 0``.

    This is the exact limit of the binomial log-likelihood terms as an
    empirical rate hits 0 or 1, so the LR tests below stay finite at the
    edges (0 breaches, all breaches) without arbitrary epsilon clamping.
    """
    if x == 0.0:
        return 0.0
    return x * math.log(y)


def _chi2_sf(x: float, dof: int) -> float:
    """Chi-square survival function ``P(X > x)`` for ``dof`` in {1, 2}.

    Closed forms, no scipy: for 1 dof, X = Z² with Z standard normal, so
    ``P(X > x) = 2·P(Z > √x) = erfc(√(x/2))``; for 2 dof, chi-square is
    Exponential(rate=1/2), so ``P(X > x) = exp(-x/2)``. Non-positive ``x``
    returns 1.0.
    """
    if dof not in (1, 2):
        raise ValueError(f"_chi2_sf supports dof 1 or 2 only, got {dof}")
    if x <= 0.0:
        return 1.0
    if dof == 1:
        return math.erfc(math.sqrt(x / 2.0))
    return math.exp(-x / 2.0)


def pinball_loss(y_true: FloatArray, y_pred: FloatArray, quantile: float) -> float:
    """Mean pinball (quantile) loss of ``y_pred`` as the ``quantile``-forecast.

    Per observation, with error ``e = y_true - y_pred``::

        L_q(e) = q·e        if e >= 0   (under-prediction)
               = (q-1)·e    if e <  0   (over-prediction)

    Averaged over all observations. The unique proper score for quantile
    forecasts: it is minimized in expectation by the true ``q``-quantile. At
    ``q = 0.5`` it equals half the mean absolute error.

    Raises ``ValueError`` unless ``0 < quantile < 1``, inputs are non-empty,
    and lengths match.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    y = _as_float_1d(y_true, "y_true")
    yhat = _as_float_1d(y_pred, "y_pred")
    if y.size != yhat.size:
        raise ValueError(f"length mismatch: y_true={y.size}, y_pred={yhat.size}")
    err = y - yhat
    loss = np.where(err >= 0.0, quantile * err, (quantile - 1.0) * err)
    return float(loss.mean())


def interval_coverage(y_true: FloatArray, lower: FloatArray, upper: FloatArray) -> float:
    """PICP: fraction of ``y_true`` falling inside ``[lower, upper]`` (inclusive).

    ``PICP = (1/n) · Σ 1{lower_i <= y_i <= upper_i}``. Boundary values count
    as covered. A calibrated central (1-α) interval should score ≈ 1-α.

    Raises ``ValueError`` on empty or length-mismatched inputs.
    """
    y, lo, hi = _validated_interval(y_true, lower, upper)
    covered = (lo <= y) & (y <= hi)
    return float(covered.mean())


def winkler_score(y_true: FloatArray, lower: FloatArray, upper: FloatArray, alpha: float) -> float:
    """Mean Winkler (interval) score of a central (1-alpha) interval.

    Per observation::

        W = (upper - lower)                          # always pay the width
          + (2/alpha)·(lower - y)   if y < lower     # undershoot penalty
          + (2/alpha)·(y - upper)   if y > upper     # overshoot penalty

    Averaged over all observations; lower is better. A proper score for
    central prediction intervals: it rewards narrow bands but charges misses
    at ``2/alpha`` per unit, so coverage can't be bought with vacuous width
    nor sharpness with systematic misses.

    Raises ``ValueError`` unless ``0 < alpha < 1``, inputs are non-empty, and
    lengths match.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    y, lo, hi = _validated_interval(y_true, lower, upper)
    width = hi - lo
    undershoot = np.maximum(lo - y, 0.0)
    overshoot = np.maximum(y - hi, 0.0)
    score = width + (2.0 / alpha) * (undershoot + overshoot)
    return float(score.mean())


def kupiec_pof(n_breaches: int, n_obs: int, expected_rate: float) -> LRTestResult:
    """Kupiec proportion-of-failures test: is the breach *rate* as advertised?

    With ``n1 = n_breaches``, ``n0 = n_obs - n1``, ``p = expected_rate`` and
    the observed rate ``π̂ = n1/n_obs``::

        LR_pof = -2·ln[ (1-p)^n0 · p^n1 / ((1-π̂)^n0 · π̂^n1) ]  ~  χ²(1)

    under H0 (true breach probability = ``p``). Small p-value ⇒ the observed
    breach count is inconsistent with the nominal rate (either direction:
    too many breaches *or* suspiciously few, i.e. over-wide bands). The
    ``n1 = 0`` and ``n1 = n_obs`` edges are exact via the ``0·log(0) = 0``
    convention (see ``_xlogy``) — no epsilon fudging.

    Raises ``ValueError`` unless ``n_obs > 0``, ``0 <= n_breaches <= n_obs``,
    and ``0 < expected_rate < 1``.
    """
    if n_obs <= 0:
        raise ValueError(f"n_obs must be positive, got {n_obs}")
    if not 0 <= n_breaches <= n_obs:
        raise ValueError(f"n_breaches must be in [0, n_obs], got {n_breaches} of {n_obs}")
    if not 0.0 < expected_rate < 1.0:
        raise ValueError(f"expected_rate must be in (0, 1), got {expected_rate}")
    n1 = n_breaches
    n0 = n_obs - n_breaches
    pi_hat = n1 / n_obs
    ll_null = _xlogy(n0, 1.0 - expected_rate) + _xlogy(n1, expected_rate)
    ll_alt = _xlogy(n0, 1.0 - pi_hat) + _xlogy(n1, pi_hat)
    lr = max(0.0, 2.0 * (ll_alt - ll_null))
    return LRTestResult(lr_stat=lr, p_value=_chi2_sf(lr, 1))


def christoffersen_independence(breaches: BreachArray) -> LRTestResult:
    """Christoffersen independence test: do breaches cluster in time?

    Fits a first-order Markov chain to the 0/1 breach series. With ``n_ij``
    the count of ``i → j`` transitions, ``π01 = n01/(n00+n01)``,
    ``π11 = n11/(n10+n11)`` and the pooled ``π = (n01+n11)/Σn_ij``::

        LR_ind = -2·ln[ (1-π)^(n00+n10) · π^(n01+n11)
                        / ((1-π01)^n00 · π01^n01 · (1-π11)^n10 · π11^n11) ]

    ~ χ²(1) under H0 (breach today is independent of breach yesterday).
    Small p-value ⇒ breaches cluster — the signature of a vol/band model
    that reacts too slowly to regime shifts, even if the overall rate is
    fine.

    Degenerate cases where the test is undefined return ``LRTestResult(0.0,
    1.0)`` by convention: fewer than 2 observations (no transitions), or a
    transition matrix with an empty row (the series never leaves one state —
    e.g. all zeros — so π01 or π11 has no observations to estimate).

    Raises ``ValueError`` on empty input or entries outside {0, 1}.
    """
    b = _as_breach_1d(breaches)
    if b.size < 2:
        return LRTestResult(lr_stat=0.0, p_value=1.0)
    prev, curr = b[:-1], b[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    from_calm = n00 + n01
    from_breach = n10 + n11
    if from_calm == 0 or from_breach == 0:
        return LRTestResult(lr_stat=0.0, p_value=1.0)
    pi01 = n01 / from_calm
    pi11 = n11 / from_breach
    pi = (n01 + n11) / (from_calm + from_breach)
    ll_alt = (
        _xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11)
    )
    ll_null = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    lr = max(0.0, 2.0 * (ll_alt - ll_null))
    return LRTestResult(lr_stat=lr, p_value=_chi2_sf(lr, 1))


def christoffersen_conditional(breaches: BreachArray, expected_rate: float) -> LRTestResult:
    """Christoffersen conditional-coverage test: right rate *and* independent.

    The joint test combining Kupiec POF and the independence test::

        LR_cc = LR_pof + LR_ind  ~  χ²(2)

    under H0 (breaches are i.i.d. Bernoulli(``expected_rate``)). This is the
    single headline number for "is this band series calibrated": it fails on
    a wrong breach rate, on clustered breaches, or both. If the independence
    component is degenerate (see ``christoffersen_independence``) its LR
    contribution is 0 and the test reduces to POF on 2 dof (conservative).

    Raises ``ValueError`` on empty input, entries outside {0, 1}, or
    ``expected_rate`` outside (0, 1).
    """
    b = _as_breach_1d(breaches)
    pof = kupiec_pof(int(b.sum()), int(b.size), expected_rate)
    ind = christoffersen_independence(b)
    lr = pof.lr_stat + ind.lr_stat
    return LRTestResult(lr_stat=lr, p_value=_chi2_sf(lr, 2))


def coverage_by_bucket(
    y_true: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    buckets: Sequence[str],
) -> dict[str, BucketCoverage]:
    """Interval coverage split by bucket label (e.g. vol-regime), for
    conditional-coverage reporting.

    For each distinct label in ``buckets`` (first-seen order preserved),
    computes ``BucketCoverage(n, coverage)`` where ``coverage`` is the PICP
    (inclusive bounds, as in ``interval_coverage``) restricted to that
    bucket's observations. A band series can be calibrated unconditionally
    yet badly miscalibrated per regime — this is the table that exposes it.

    Raises ``ValueError`` on empty inputs or any length mismatch.
    """
    y, lo, hi = _validated_interval(y_true, lower, upper)
    labels = list(buckets)
    if len(labels) != y.size:
        raise ValueError(f"length mismatch: y_true={y.size}, buckets={len(labels)}")
    covered = (lo <= y) & (y <= hi)
    result: dict[str, BucketCoverage] = {}
    for label in dict.fromkeys(labels):
        mask = np.asarray([item == label for item in labels])
        n = int(mask.sum())
        result[label] = BucketCoverage(n=n, coverage=float(covered[mask].mean()))
    return result
