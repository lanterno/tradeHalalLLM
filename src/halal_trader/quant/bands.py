"""Expected high/low price bands from a daily-volatility forecast.

Phase 1 of ``docs/QUANT_PREDICTION_ROADMAP.md``: the deterministic
"how high / how low" core. Three pieces:

* **HAR forecaster** (`fit_har` / `HARModel.forecast`) — Corsi's
  heterogeneous autoregression on *log* daily vol: the h-day-ahead mean
  vol is regressed on the current vol and its 5-day and 22-day trailing
  means. Fit in logs (log-vol is near-Gaussian; raw-vol OLS is
  heteroskedastic), exponentiated with the half-variance bias correction.
* **Band conversion** (`price_bands`, `expected_range`) — a vol forecast
  becomes price bands ``close·exp(±z·σ̂·√h)``. The textbook Gaussian z is
  a *starting point only*: bands on the horizon max/min under-cover for
  three compounding reasons (path extremes vs endpoints — reflection
  principle; fat tails; range-estimator downward bias).
* **Empirical z calibration** (`calibrate_z`) — the one step that absorbs
  all three biases at once: on walk-forward history, convert each
  realized horizon max-high/min-low into its implied z, and pick the
  empirical quantile that delivers the target *two-sided path* coverage.
  Never trust a theoretical z where a measured one is available.

The ATR-multiple band (`atr_band`) is the naive baseline every model must
beat on pinball/Winkler/coverage (see ``quant/eval.py``) before shipping.

Pure numpy by design. Daily vol units throughout (NOT annualized), matching
``quant/volatility.py``; horizons are in trading days.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = Any
"""1-d float sequence/ndarray; normalized internally via ``np.asarray``."""

_HAR_LAGS = (1, 5, 22)
_MIN_HAR_SAMPLES = 60  # regression rows below this → refuse to fit


@dataclass(frozen=True, slots=True)
class HARModel:
    """Fitted HAR(1, 5, 22) direct forecaster of h-day-ahead mean log vol.

    ``coefs`` is ``[intercept, b_daily, b_weekly, b_monthly]`` on log vol;
    ``resid_var`` is the residual variance used for the lognormal
    bias correction ``E[σ] = exp(µ + resid_var/2)``; ``horizon`` is the
    number of forward days the target averaged over; ``n`` is the number
    of regression rows the fit saw.
    """

    coefs: tuple[float, float, float, float]
    resid_var: float
    horizon: int
    n: int

    def forecast(self, vol: FloatArray) -> float:
        """Forecast mean daily vol over the next ``horizon`` days.

        ``vol`` is the same daily-vol series the model was fit on (e.g.
        Yang-Zhang), of which only the trailing 22 finite values are used.
        """
        v = _clean_vol(vol)
        if v.size < _HAR_LAGS[-1]:
            raise ValueError(f"need >= {_HAR_LAGS[-1]} finite vol points to forecast, got {v.size}")
        feats = (
            1.0,
            float(np.log(v[-1])),
            float(np.log(v[-5:].mean())),
            float(np.log(v[-22:].mean())),
        )
        mu = float(np.dot(self.coefs, feats))
        return float(np.exp(mu + self.resid_var / 2.0))


def _clean_vol(vol: FloatArray) -> npt.NDArray[np.float64]:
    """Coerce to 1-d float64, drop NaN warm-up, require strictly positive."""
    arr = np.asarray(vol, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"vol must be 1-dimensional, got ndim={arr.ndim}")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("vol has no finite values")
    if (arr <= 0).any():
        raise ValueError("vol must be strictly positive (log-vol regression)")
    return arr


def fit_har(vol: FloatArray, horizon: int) -> HARModel:
    """Fit a direct HAR(1, 5, 22) model of h-day-ahead mean log vol.

    Regression rows are built at every ``t`` with a full 22-day lookback
    and a full ``horizon`` forward window::

        log(mean σ[t+1 .. t+h]) ~ 1 + log σ[t] + log(mean σ[t-4..t])
                                    + log(mean σ[t-21..t])

    Direct (one model per horizon) rather than iterated — the roadmap's
    choice for 1–5 day horizons. Raises ``ValueError`` with fewer than
    ``60`` regression rows: below that the fit is noise (and the caller
    should fall back to the ATR/EWMA baseline).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    v = _clean_vol(vol)
    lookback = _HAR_LAGS[-1]
    n_rows = v.size - lookback + 1 - horizon
    if n_rows < _MIN_HAR_SAMPLES:
        raise ValueError(
            f"need >= {_MIN_HAR_SAMPLES} HAR regression rows, got {max(n_rows, 0)} "
            f"(vol series too short: {v.size})"
        )
    log_v = np.log(v)
    xs = []
    ys = []
    for i in range(lookback - 1, v.size - horizon):
        xs.append(
            (
                1.0,
                log_v[i],
                float(np.log(v[i - 4 : i + 1].mean())),
                float(np.log(v[i - 21 : i + 1].mean())),
            )
        )
        ys.append(float(np.log(v[i + 1 : i + 1 + horizon].mean())))
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    coefs, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coefs
    # ddof=4: intercept + three slopes.
    resid_var = float(resid.var(ddof=min(4, resid.size - 1))) if resid.size > 1 else 0.0
    c = coefs.astype(float)
    return HARModel(
        coefs=(c[0], c[1], c[2], c[3]),
        resid_var=resid_var,
        horizon=horizon,
        n=int(y.size),
    )


@dataclass(frozen=True, slots=True)
class PriceBands:
    """An expected high/low band for one horizon.

    ``low``/``high`` bound the *path extreme* (horizon min-low/max-high)
    at the calibration's target coverage; ``expected_range`` is the
    point estimate of the h-day high-minus-low in price units.
    """

    horizon: int
    low: float
    high: float
    expected_range: float
    sigma_daily: float
    z: float


def price_bands(close: float, sigma_daily: float, horizon: int, z: float) -> PriceBands:
    """Convert a daily-vol forecast into lognormal price bands.

    ``low/high = close·exp(∓z·σ̂·√h)`` — symmetric in log space. ``z``
    should come from :func:`calibrate_z` (path-extreme coverage), not a
    normal table: textbook z under-covers the horizon max/min. The
    expected h-day range uses the driftless-Brownian identity
    ``E[range] = √(8/π)·σ·√h ≈ 1.596·σ·√h`` (in log terms, then scaled
    by price).
    """
    if close <= 0 or sigma_daily <= 0 or horizon < 1 or z <= 0:
        raise ValueError(
            f"need close>0, sigma>0, horizon>=1, z>0; got {close=}, "
            f"{sigma_daily=}, {horizon=}, {z=}"
        )
    scale = sigma_daily * np.sqrt(horizon)
    return PriceBands(
        horizon=horizon,
        low=float(close * np.exp(-z * scale)),
        high=float(close * np.exp(z * scale)),
        expected_range=float(close * np.sqrt(8.0 / np.pi) * scale),
        sigma_daily=float(sigma_daily),
        z=float(z),
    )


def atr_band(close: float, atr: float, horizon: int, multiple: float = 1.0) -> PriceBands:
    """ATR-multiple band: ``close ± m·ATR·√h`` — the naive baseline.

    Every fitted band model must beat this on pinball/Winkler/coverage on
    disjoint OOS windows before it ships (roadmap validation gate 1).
    ``expected_range`` is ``2·m·ATR·√h`` (the band width itself — ATR is
    an average range, not a quantile, which is exactly why this is the
    baseline and not the product).
    """
    if close <= 0 or atr <= 0 or horizon < 1 or multiple <= 0:
        raise ValueError(
            f"need close>0, atr>0, horizon>=1, multiple>0; got {close=}, "
            f"{atr=}, {horizon=}, {multiple=}"
        )
    half = multiple * atr * np.sqrt(horizon)
    return PriceBands(
        horizon=horizon,
        low=float(max(close - half, 0.0)),
        high=float(close + half),
        expected_range=float(2.0 * half),
        sigma_daily=float(atr / close),
        z=float(multiple),
    )


@dataclass(frozen=True, slots=True)
class CalibratedZ:
    """An empirically calibrated band multiplier.

    ``z`` delivers ``target_coverage`` of *two-sided path containment*
    (both the horizon max-high and min-low inside the band) on the
    calibration sample of ``n`` observations.
    """

    z: float
    target_coverage: float
    n: int


def calibrate_z(
    closes: FloatArray,
    sigmas: FloatArray,
    realized_highs: FloatArray,
    realized_lows: FloatArray,
    horizon: int,
    target_coverage: float = 0.8,
    min_samples: int = 40,
) -> CalibratedZ:
    """Calibrate the band multiplier ``z`` on realized path extremes.

    For each historical observation ``i`` (a forecast made at ``t_i``):
    ``closes[i]`` is the anchor close, ``sigmas[i]`` the daily-vol
    forecast made *at that time* (walk-forward — never a lookahead
    refit), and ``realized_highs/lows[i]`` the max-high/min-low over the
    following ``horizon`` days. Each observation's binding multiplier is::

        z_i = max( ln(high_i/close_i), -ln(low_i/close_i) ) / (σ_i·√h)

    — the smallest z whose band would have contained BOTH extremes. The
    calibrated z is the empirical ``target_coverage`` quantile of the
    ``z_i``. This single step absorbs the reflection-principle gap, fat
    tails, and estimator bias simultaneously; validate the result with
    ``quant.eval`` coverage tests on a *disjoint* window before use.
    """
    if not 0.0 < target_coverage < 1.0:
        raise ValueError(f"target_coverage must be in (0, 1), got {target_coverage}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    c = np.asarray(closes, dtype=np.float64)
    s = np.asarray(sigmas, dtype=np.float64)
    hi = np.asarray(realized_highs, dtype=np.float64)
    lo = np.asarray(realized_lows, dtype=np.float64)
    if not (c.ndim == s.ndim == hi.ndim == lo.ndim == 1):
        raise ValueError("all calibration inputs must be 1-dimensional")
    if not (c.size == s.size == hi.size == lo.size):
        raise ValueError(
            f"length mismatch: closes={c.size}, sigmas={s.size}, highs={hi.size}, lows={lo.size}"
        )
    valid = (
        np.isfinite(c)
        & np.isfinite(s)
        & np.isfinite(hi)
        & np.isfinite(lo)
        & (c > 0)
        & (s > 0)
        & (hi > 0)
        & (lo > 0)
    )
    c, s, hi, lo = c[valid], s[valid], hi[valid], lo[valid]
    if c.size < min_samples:
        raise ValueError(f"need >= {min_samples} valid calibration observations, got {c.size}")
    scale = s * np.sqrt(horizon)
    z_up = np.log(hi / c) / scale
    z_dn = -np.log(lo / c) / scale
    z_binding = np.maximum(z_up, z_dn)
    # A degenerate flat sample can produce z <= 0; floor at a tiny band.
    z = max(float(np.quantile(z_binding, target_coverage)), 1e-6)
    return CalibratedZ(z=z, target_coverage=target_coverage, n=int(c.size))
