"""Rolling daily-volatility estimators from OHLC bars (Phase 1 of the quant roadmap).

Range-based estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang)
extract the intrabar high/low information that the close-to-close baseline
throws away, buying several-fold statistical efficiency from the same number
of daily bars — which matters on a ~20-symbol × ~500-bar universe where every
observation counts. ``close_to_close`` is kept as the yardstick every range
estimator must beat, and ``ewma_vol`` is the RiskMetrics recency-weighted
baseline for regime-tracking comparisons.

All estimators share one contract: 1-d strictly positive price arrays of
equal length in, an equal-length float64 array out, in DAILY volatility units
— the standard deviation of daily log returns — NOT annualized; the band
layer scales by ``sqrt(h)`` for an h-day horizon directly. Warm-up slots are
NaN: the first ``window - 1`` slots for estimators that only need the current
bar (``parkinson``, ``garman_klass``, ``rogers_satchell``); estimators that
also need a previous close additionally have slot 0 undefined, so
``close_to_close`` and ``yang_zhang`` yield NaN for the first ``window``
slots and ``ewma_vol`` for slot 0. Empty inputs, length mismatches,
non-positive prices, ``window < 2`` and out-of-range ``lam`` raise
``ValueError`` rather than silently producing garbage.

Pure numpy + stdlib by design: no scipy, no pandas. Rolling windows use a
plain O(n·window) loop (see ``_rolling``) — clarity beats stride tricks at
this data scale.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = Sequence[float] | npt.NDArray[Any]
"""Any 1-d float-coercible input; normalized internally via ``np.asarray``."""


def _as_price_1d(x: FloatArray, name: str) -> npt.NDArray[np.float64]:
    """Coerce ``x`` to a non-empty 1-d float64 array of strictly positive prices.

    NaN entries fail the positivity check (``nan > 0`` is False), so bad
    feeds raise here instead of poisoning every log downstream.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.all(arr > 0.0)):
        raise ValueError(f"{name} must contain only positive prices")
    return arr


def _validated_prices(
    named: Sequence[tuple[str, FloatArray]],
) -> list[npt.NDArray[np.float64]]:
    """Coerce each (name, series) pair via ``_as_price_1d`` and length-check."""
    arrays = [_as_price_1d(x, name) for name, x in named]
    if len({arr.size for arr in arrays}) > 1:
        detail = ", ".join(f"{name}={arr.size}" for (name, _), arr in zip(named, arrays))
        raise ValueError(f"length mismatch: {detail}")
    return arrays


def _validate_window(window: int) -> None:
    """Raise ``ValueError`` unless ``window >= 2`` (variances need ddof=1 room)."""
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")


def _rolling(
    values: npt.NDArray[np.float64],
    window: int,
    stat: Callable[[npt.NDArray[np.float64]], float],
) -> npt.NDArray[np.float64]:
    """Apply ``stat`` to each trailing ``window``-slice of ``values``.

    Returns an array of the same length with NaN in the first ``window - 1``
    slots. NaNs *inside* a window (e.g. the slot-0 padding of return series
    that need a previous close) propagate to the output, which is exactly how
    the extra leading NaN of ``close_to_close`` / ``yang_zhang`` arises.

    Simple O(n·window) loop on purpose: the universe is ~20 symbols × ~500
    daily bars, so readability wins over sliding-window stride tricks.
    """
    out = np.full(values.size, np.nan, dtype=np.float64)
    for end in range(window - 1, values.size):
        out[end] = stat(values[end - window + 1 : end + 1])
    return out


def _mean(w: npt.NDArray[np.float64]) -> float:
    """Plain window mean."""
    return float(w.mean())


def _var_ddof1(w: npt.NDArray[np.float64]) -> float:
    """Sample variance about the window mean (ddof=1)."""
    return float(w.var(ddof=1))


def _std_ddof1(w: npt.NDArray[np.float64]) -> float:
    """Sample standard deviation about the window mean (ddof=1)."""
    return float(w.std(ddof=1))


def close_to_close(close: FloatArray, window: int) -> npt.NDArray[np.float64]:
    """Rolling close-to-close volatility: sample std dev of daily log returns.

    ::

        r_t     = ln(C_t / C_{t-1})
        sigma_t = std(r_{t-window+1}, ..., r_t; ddof=1)

    The textbook baseline every range estimator is judged against: unbiased
    under any i.i.d. return process, but it uses one number per bar, so its
    sampling error is the largest of the family. Daily units, not annualized.

    Output has the input's length; the first ``window`` slots are NaN
    (``window - 1`` warm-up slots plus slot 0, which has no previous close).

    Raises ``ValueError`` on empty input, non-positive prices, or
    ``window < 2``.
    """
    c = _as_price_1d(close, "close")
    _validate_window(window)
    r = np.full(c.size, np.nan, dtype=np.float64)
    r[1:] = np.log(c[1:] / c[:-1])
    return _rolling(r, window, _std_ddof1)


def parkinson(high: FloatArray, low: FloatArray, window: int) -> npt.NDArray[np.float64]:
    """Parkinson (1980) range volatility from rolling high/low extremes.

    ::

        sigma_t^2 = (1 / (4·ln 2)) · mean( ln(H/L)^2  over the window )

    Roughly 5× more efficient than close-to-close under driftless GBM with
    continuous monitoring, but blind to opening gaps and biased by drift.
    Daily units. First ``window - 1`` slots are NaN.

    Raises ``ValueError`` on empty inputs, length mismatch, non-positive
    prices, or ``window < 2``.
    """
    hi, lo = _validated_prices([("high", high), ("low", low)])
    _validate_window(window)
    term = np.log(hi / lo) ** 2
    var = _rolling(term, window, _mean) / (4.0 * math.log(2.0))
    result: npt.NDArray[np.float64] = np.sqrt(var)
    return result


def garman_klass(
    open_: FloatArray, high: FloatArray, low: FloatArray, close: FloatArray, window: int
) -> npt.NDArray[np.float64]:
    """Garman-Klass (1980) OHLC volatility.

    ::

        sigma_t^2 = mean( 0.5·ln(H/L)^2 − (2·ln 2 − 1)·ln(C/O)^2 )

    Adds the open→close body to Parkinson's range for ~7.4× theoretical
    efficiency over close-to-close; still assumes no drift and no overnight
    gap. Daily units. First ``window - 1`` slots are NaN.

    Negative window means are clamped to 0 before the square root: the GK
    per-bar term subtracts the body from the range, so degenerate bars whose
    recorded range is small relative to |C − O| (bad ticks, partially
    adjusted bars violating H ≥ max(O, C) ≥ min(O, C) ≥ L) can push the
    estimator negative — clamping yields 0 vol there instead of NaN-poisoning
    the series.

    Raises ``ValueError`` on empty inputs, length mismatch, non-positive
    prices, or ``window < 2``.
    """
    o, hi, lo, c = _validated_prices(
        [("open_", open_), ("high", high), ("low", low), ("close", close)]
    )
    _validate_window(window)
    term = 0.5 * np.log(hi / lo) ** 2 - (2.0 * math.log(2.0) - 1.0) * np.log(c / o) ** 2
    var = np.maximum(_rolling(term, window, _mean), 0.0)  # NaN prefix propagates
    result: npt.NDArray[np.float64] = np.sqrt(var)
    return result


def rogers_satchell(
    open_: FloatArray, high: FloatArray, low: FloatArray, close: FloatArray, window: int
) -> npt.NDArray[np.float64]:
    """Rogers-Satchell (1991) drift-independent OHLC volatility.

    ::

        sigma_t^2 = mean( ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O) )

    Unbiased under nonzero drift (unlike Parkinson/Garman-Klass), which is
    why it is the intraday component inside Yang-Zhang; still ignores
    overnight gaps. Daily units. First ``window - 1`` slots are NaN.

    On consistent bars (H ≥ max(O, C), L ≤ min(O, C)) every per-bar term is
    non-negative, but inconsistent bars can drive the window mean negative,
    so it is clamped to 0 before the square root — same rationale as
    ``garman_klass``.

    Raises ``ValueError`` on empty inputs, length mismatch, non-positive
    prices, or ``window < 2``.
    """
    o, hi, lo, c = _validated_prices(
        [("open_", open_), ("high", high), ("low", low), ("close", close)]
    )
    _validate_window(window)
    term = np.log(hi / c) * np.log(hi / o) + np.log(lo / c) * np.log(lo / o)
    var = np.maximum(_rolling(term, window, _mean), 0.0)  # NaN prefix propagates
    result: npt.NDArray[np.float64] = np.sqrt(var)
    return result


def yang_zhang(
    open_: FloatArray, high: FloatArray, low: FloatArray, close: FloatArray, window: int
) -> npt.NDArray[np.float64]:
    """Yang-Zhang (2000) minimum-variance OHLC volatility.

    The default estimator of the family: it is the only one that handles both
    overnight gaps (which dominate large-cap risk) and drift, by summing an
    overnight variance, a weighted open-to-close variance, and the
    drift-independent Rogers-Satchell term::

        o_t  = ln(O_t / C_{t-1})                    # overnight return
        c_t  = ln(C_t / O_t)                        # open-to-close return
        rs_t = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)    # Rogers-Satchell term

        sigma_t^2 = Var(o; ddof=1) + k·Var(c; ddof=1) + (1 − k)·mean(rs)
        k         = 0.34 / (1.34 + (window + 1) / (window − 1))

    where each ``Var`` is the sample variance about the window mean and ``k``
    minimizes the estimator's variance (Yang & Zhang 2000, eq. 10). The
    Rogers-Satchell window mean is clamped to 0 (see ``rogers_satchell``), so
    with k ∈ (0, 1) the combined variance is non-negative by construction.
    Daily units. The first ``window`` slots are NaN (``window - 1`` warm-up
    slots plus slot 0, whose overnight return needs a previous close).

    Raises ``ValueError`` on empty inputs, length mismatch, non-positive
    prices, or ``window < 2``.
    """
    o, hi, lo, c = _validated_prices(
        [("open_", open_), ("high", high), ("low", low), ("close", close)]
    )
    _validate_window(window)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    overnight = np.full(c.size, np.nan, dtype=np.float64)
    overnight[1:] = np.log(o[1:] / c[:-1])
    open_to_close = np.log(c / o)
    rs_term = np.log(hi / c) * np.log(hi / o) + np.log(lo / c) * np.log(lo / o)

    var_overnight = _rolling(overnight, window, _var_ddof1)
    var_open_to_close = _rolling(open_to_close, window, _var_ddof1)
    var_rs = np.maximum(_rolling(rs_term, window, _mean), 0.0)

    var = var_overnight + k * var_open_to_close + (1.0 - k) * var_rs
    result: npt.NDArray[np.float64] = np.sqrt(var)
    return result


def ewma_vol(close: FloatArray, lam: float = 0.94) -> npt.NDArray[np.float64]:
    """RiskMetrics EWMA volatility on squared daily log returns.

    ::

        r_t   = ln(C_t / C_{t-1})
        var_1 = r_1^2                              # seed: first return squared
        var_t = lam·var_{t-1} + (1 − lam)·r_t^2    # t >= 2

    Returns ``sqrt(var_t)`` per slot — a recency-weighted vol with effective
    memory ≈ 1/(1 − lam) bars (~17 at the RiskMetrics default lam = 0.94), so
    it tracks regime shifts faster than any fixed rolling window. Daily
    units. Slot 0 is NaN (no previous close); no other warm-up — the seed
    makes every later slot defined.

    Raises ``ValueError`` on empty input, non-positive prices, or ``lam``
    outside the open interval (0, 1).
    """
    c = _as_price_1d(close, "close")
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    out = np.full(c.size, np.nan, dtype=np.float64)
    if c.size < 2:
        return out
    r_sq = np.log(c[1:] / c[:-1]) ** 2
    var = float(r_sq[0])
    out[1] = math.sqrt(var)
    for t in range(2, c.size):
        var = lam * var + (1.0 - lam) * float(r_sq[t - 1])
        out[t] = math.sqrt(var)
    return out
