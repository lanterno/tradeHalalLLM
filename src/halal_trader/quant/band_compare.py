"""Disjoint-OOS comparison of band sources — the Phase 2 ship-gate.

Roadmap validation gate 1: a fitted band model ships into the engine only
if it beats the naive baseline on coverage + interval score on **disjoint
out-of-sample windows**, same-bars, walk-forward. This module runs that
A/B for three sources on identical observations:

* ``atr`` — ``close ± 1·ATR14·√h`` (the naive baseline),
* ``har_cal`` — HAR-on-Yang-Zhang σ with an empirically calibrated z
  (production's deterministic band). Honest OOS: within each evaluation
  window, z is recalibrated from observations in EARLIER windows only —
  the shipped artifact never scores itself.
* ``garch_fhs`` — the GJR-GARCH filtered-historical-simulation path
  extremes (``quant/garch.py``), jointly-calibrated ``band(coverage)``.

Observations step by ``2·horizon`` so consecutive windows don't overlap
and double-count episodes; rows exist only where EVERY source could
produce a band (same-rows discipline — comparing sources on different
samples is invalid). The first time window is calibration-only and never
scored. Metrics per source per window: two-sided path coverage (target =
``coverage``) and the joint Winkler/interval score normalized by price
(width in % + ``2/α`` per unit of breach) — coverage alone is gameable
by width, Winkler punishes both. Pinball per marginal quantile is NOT
comparable across these sources (har_cal emits a band, not quantiles) and
is deliberately omitted; the docstring says so, so the checklist doesn't.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from halal_trader.quant.bands import fit_har
from halal_trader.quant.garch import garch_fhs_path_extremes
from halal_trader.quant.levels import atr_series
from halal_trader.quant.volatility import FloatArray, yang_zhang

logger = logging.getLogger(__name__)

_YZ_WINDOW = 20
_MIN_HAR_HISTORY = 110
_MIN_Z_OBS = 40  # a window needs this many prior obs to calibrate har z


@dataclass(frozen=True, slots=True)
class _Row:
    date: str
    close: float
    realized_high: float
    realized_low: float
    sigma_har: float
    atr: float
    garch_low: float
    garch_high: float


@dataclass(frozen=True, slots=True)
class SourceScore:
    """One band source's pooled score on the evaluated OOS windows."""

    coverage: float
    coverage_error: float  # |coverage - target|
    winkler: float  # mean joint interval score, % of price
    n: int


def build_rows(
    dates: list[str],
    opens: FloatArray,
    highs: FloatArray,
    lows: FloatArray,
    closes: FloatArray,
    *,
    horizon: int = 5,
    step: int | None = None,
    garch_sims: int = 1000,
    garch_min_returns: int = 250,
    seed: int = 0,
) -> list[_Row]:
    """Walk-forward observation rows for one symbol (same-rows discipline).

    A row exists only where the HAR sigma, the ATR and the GARCH band are
    all computable from data ≤ t — every source is scored on identical
    observations or the comparison is invalid.
    """
    h_arr = np.asarray(highs, dtype=np.float64)
    l_arr = np.asarray(lows, dtype=np.float64)
    c_arr = np.asarray(closes, dtype=np.float64)
    yz = yang_zhang(opens, highs, lows, closes, window=_YZ_WINDOW)
    atr = atr_series(h_arr, l_arr, c_arr)
    n = c_arr.size
    stride = step or 2 * horizon
    rows: list[_Row] = []
    start = max(_MIN_HAR_HISTORY, garch_min_returns + 1) - 1
    for t in range(start, n - horizon, stride):
        vol_prefix = yz[: t + 1]
        try:
            sigma = fit_har(vol_prefix, horizon).forecast(vol_prefix)
        except ValueError:
            continue
        atr_t = float(atr[t])
        if atr_t <= 0:
            continue
        gb = garch_fhs_path_extremes(
            c_arr[: t + 1],
            horizon,
            n_sims=garch_sims,
            seed=seed + t,
        )
        if gb is None:
            continue
        g_lo, g_hi = gb.band(0.8)
        rows.append(
            _Row(
                date=dates[t],
                close=float(c_arr[t]),
                realized_high=float(h_arr[t + 1 : t + 1 + horizon].max()),
                realized_low=float(l_arr[t + 1 : t + 1 + horizon].min()),
                sigma_har=float(sigma),
                atr=atr_t,
                garch_low=g_lo,
                garch_high=g_hi,
            )
        )
    return rows


def _score(bands: list[tuple[float, float]], rows: list[_Row], target: float) -> SourceScore:
    covered = 0
    winkler_sum = 0.0
    alpha = 1.0 - target
    for (lo, hi), r in zip(bands, rows, strict=True):
        inside = r.realized_low >= lo and r.realized_high <= hi
        covered += int(inside)
        width = (hi - lo) / r.close
        undershoot = max(lo - r.realized_low, 0.0) / r.close
        overshoot = max(r.realized_high - hi, 0.0) / r.close
        winkler_sum += (width + (2.0 / alpha) * (undershoot + overshoot)) * 100
    n = len(rows)
    cov = covered / n if n else 0.0
    return SourceScore(
        coverage=round(cov, 4),
        coverage_error=round(abs(cov - target), 4),
        winkler=round(winkler_sum / n, 4) if n else 0.0,
        n=n,
    )


def _har_z_from(rows: list[_Row], horizon: int, target: float) -> float | None:
    """Binding-z quantile from PRIOR-window rows (the walk-forward z)."""
    if len(rows) < _MIN_Z_OBS:
        return None
    scale = np.array([r.sigma_har for r in rows]) * np.sqrt(horizon)
    closes = np.array([r.close for r in rows])
    z_up = np.log(np.array([r.realized_high for r in rows]) / closes) / scale
    z_dn = -np.log(np.array([r.realized_low for r in rows]) / closes) / scale
    return float(np.quantile(np.maximum(z_up, z_dn), target))


def compare_band_sources(
    rows_by_symbol: dict[str, list[_Row]],
    *,
    horizon: int = 5,
    target: float = 0.8,
    n_windows: int = 3,
) -> dict[str, dict[str, SourceScore]]:
    """Score every source per disjoint OOS window (window 0 never scored).

    Rows pool across symbols in date order and split into ``n_windows``
    equal-count windows. Window ``k`` is scored with the har z calibrated
    on windows ``< k`` only. Returns ``{window_label: {source: score}}``
    plus an ``"aggregate"`` entry pooling all evaluated windows.
    """
    pooled = sorted((r for rows in rows_by_symbol.values() for r in rows), key=lambda r: r.date)
    if len(pooled) < n_windows * _MIN_Z_OBS:
        raise ValueError(
            f"need >= {n_windows * _MIN_Z_OBS} pooled rows for {n_windows} "
            f"windows, got {len(pooled)}"
        )
    windows = np.array_split(np.asarray(pooled, dtype=object), n_windows)
    results: dict[str, dict[str, SourceScore]] = {}
    agg_rows: list[_Row] = []
    agg_bands: dict[str, list[tuple[float, float]]] = {"atr": [], "har_cal": [], "garch_fhs": []}
    sqrt_h = float(np.sqrt(horizon))
    for k in range(1, n_windows):
        prior: list[_Row] = [r for w in windows[:k] for r in list(w)]
        z = _har_z_from(prior, horizon, target)
        if z is None:
            continue
        w_rows = [r for r in list(windows[k])]
        bands = {
            "atr": [(r.close - r.atr * sqrt_h, r.close + r.atr * sqrt_h) for r in w_rows],
            "har_cal": [
                (
                    r.close * float(np.exp(-z * r.sigma_har * sqrt_h)),
                    r.close * float(np.exp(z * r.sigma_har * sqrt_h)),
                )
                for r in w_rows
            ],
            "garch_fhs": [(r.garch_low, r.garch_high) for r in w_rows],
        }
        results[f"window_{k}"] = {src: _score(b, w_rows, target) for src, b in bands.items()}
        agg_rows.extend(w_rows)
        for src, b in bands.items():
            agg_bands[src].extend(b)
    if not results:
        raise ValueError("no window had enough prior observations to calibrate z")
    results["aggregate"] = {src: _score(b, agg_rows, target) for src, b in agg_bands.items()}
    return results


def garch_verdict(results: dict[str, dict[str, SourceScore]]) -> str:
    """Pre-registered ship rule for GARCH-FHS vs the deterministic bands.

    ``pass``: aggregate Winkler strictly better than har_cal AND aggregate
    coverage error no worse, AND not worse on Winkler in a majority of
    individual windows. ``fail``: aggregate Winkler worse than the naive
    ATR baseline. Anything else: ``inconclusive``.
    """
    agg = results["aggregate"]
    g, h, a = agg["garch_fhs"], agg["har_cal"], agg["atr"]
    if g.winkler > a.winkler:
        return "fail"
    windows = [v for k, v in results.items() if k != "aggregate"]
    better_windows = sum(1 for w in windows if w["garch_fhs"].winkler <= w["har_cal"].winkler)
    if (
        g.winkler < h.winkler
        and g.coverage_error <= h.coverage_error
        and better_windows * 2 > len(windows)
    ):
        return "pass"
    return "inconclusive"
