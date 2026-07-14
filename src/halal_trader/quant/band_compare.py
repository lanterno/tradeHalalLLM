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
from halal_trader.quant.qgbm import fit_qgbm, predict_bands
from halal_trader.quant.volatility import FloatArray, yang_zhang

logger = logging.getLogger(__name__)

_YZ_WINDOW = 20
_MIN_HAR_HISTORY = 110
_MIN_Z_OBS = 40  # a window needs this many prior obs to calibrate har z


@dataclass(frozen=True, slots=True)
class _Row:
    date: str
    symbol: str
    close: float
    realized_high: float
    realized_low: float
    sigma_har: float
    atr: float
    garch_low: float
    garch_high: float
    # Leakage-safe feature vector computed from data <= t (see build_rows) —
    # the training input for learned band sources (quant/qgbm.py).
    features: tuple[float, ...] = ()


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
    symbol: str = "?",
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
                symbol=symbol,
                close=float(c_arr[t]),
                realized_high=float(h_arr[t + 1 : t + 1 + horizon].max()),
                realized_low=float(l_arr[t + 1 : t + 1 + horizon].min()),
                sigma_har=float(sigma),
                atr=atr_t,
                garch_low=g_lo,
                garch_high=g_hi,
                features=_features_at(t, yz, atr, c_arr),
            )
        )
    return rows


def _features_at(
    t: int,
    yz: np.ndarray,
    atr: np.ndarray,
    closes: np.ndarray,
) -> tuple[float, ...]:
    """Leakage-safe feature vector at ``t`` for learned band sources.

    Deliberately small and vol-centric (the predictable quantity): log
    Yang-Zhang vol at 1/5/22-day aggregation (the HAR trio), relative ATR,
    recent returns and a 20-day range position. Everything uses bars ≤ t.
    """
    c = float(closes[t])
    yz_now = float(yz[t])
    yz5 = float(np.nanmean(yz[max(t - 4, 0) : t + 1]))
    yz22 = float(np.nanmean(yz[max(t - 21, 0) : t + 1]))
    win = closes[max(t - 19, 0) : t + 1]
    rng = float(win.max() - win.min())
    return (
        float(np.log(max(yz_now, 1e-8))),
        float(np.log(max(yz5, 1e-8))),
        float(np.log(max(yz22, 1e-8))),
        float(atr[t] / c),
        float(c / closes[t - 5] - 1.0) if t >= 5 else 0.0,
        float(c / closes[t - 20] - 1.0) if t >= 20 else 0.0,
        float(abs(c / closes[t - 1] - 1.0)) if t >= 1 else 0.0,
        float((c - win.min()) / rng) if rng > 0 else 0.5,
    )


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
    agg: dict[str, tuple[list[tuple[float, float]], list[_Row]]] = {}
    n_scored = 0
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
        # Learned source: refit per window on PRIOR rows only, with a
        # per-symbol embargo (each symbol's latest prior row may share
        # future bars with the window's earliest targets — drop it).
        models = fit_qgbm(_embargo_last_per_symbol(prior))
        if models is not None:
            bands["qgbm"] = predict_bands(models, w_rows)
        n_scored += 1
        results[f"window_{k}"] = {src: _score(b, w_rows, target) for src, b in bands.items()}
        for src, b in bands.items():
            got = agg.setdefault(src, ([], []))
            got[0].extend(b)
            got[1].extend(w_rows)
    if not results:
        raise ValueError("no window had enough prior observations to calibrate z")
    # Aggregate only sources present in EVERY scored window — a source that
    # skipped a window would otherwise be pooled on an easier subset.
    counts: dict[str, int] = {}
    for scores in results.values():
        for src in scores:
            counts[src] = counts.get(src, 0) + 1
    results["aggregate"] = {
        src: _score(agg[src][0], agg[src][1], target)
        for src, count in counts.items()
        if count == n_scored
    }
    return results


def _embargo_last_per_symbol(rows: list[_Row]) -> list[_Row]:
    """Drop each symbol's latest row — its target window may extend past the
    train/test boundary and share future bars with early test targets."""
    latest: dict[str, str] = {}
    for r in rows:
        if r.date > latest.get(r.symbol, ""):
            latest[r.symbol] = r.date
    return [r for r in rows if r.date != latest.get(r.symbol)]


def ship_verdict(
    results: dict[str, dict[str, SourceScore]],
    candidate: str,
    *,
    target: float = 0.8,
) -> str:
    """Pre-registered ship rule for ``candidate`` vs the deterministic bands.

    Pooled over the windows the candidate PARTICIPATED in (a learned source
    may skip early windows for lack of training rows — it is compared to
    har_cal/atr on exactly those shared windows, never on an easier
    subset). ``pass``: pooled Winkler strictly better than har_cal AND
    pooled coverage error no worse, AND not worse on Winkler in a majority
    of shared windows. ``fail``: pooled Winkler worse than the naive ATR
    baseline. Anything else: ``inconclusive``.
    """
    shared = [v for k, v in results.items() if k != "aggregate" and candidate in v]
    if not shared:
        return "inconclusive"

    def pooled(src: str) -> tuple[float, float]:
        n = sum(w[src].n for w in shared)
        wink = sum(w[src].winkler * w[src].n for w in shared) / n
        cov = sum(w[src].coverage * w[src].n for w in shared) / n
        return wink, abs(cov - target)

    c_wink, c_err = pooled(candidate)
    h_wink, h_err = pooled("har_cal")
    a_wink, _ = pooled("atr")
    if c_wink > a_wink:
        return "fail"
    better = sum(1 for w in shared if w[candidate].winkler <= w["har_cal"].winkler)
    if c_wink < h_wink and c_err <= h_err and better * 2 > len(shared):
        return "pass"
    return "inconclusive"


def garch_verdict(results: dict[str, dict[str, SourceScore]]) -> str:
    """Back-compat wrapper: the GARCH-FHS ship rule via :func:`ship_verdict`."""
    return ship_verdict(results, "garch_fhs")
