"""Quantile gradient boosting on path-extreme targets (Phase 2, ``[ml]``).

The second of the roadmap's two extreme-native methods: train quantile
regressors DIRECTLY on the horizon extremes —

* ``y_high = log(max High[t+1..t+h] / Close_t)`` at the upper quantile,
* ``y_low  = log(min  Low[t+1..t+h] / Close_t)`` at the lower quantile —

pooled cross-sectionally over the universe (per-symbol series are far too
thin), on the leakage-safe vol-centric features carried by the comparison
rows. Marginal 0.90/0.10 quantiles approximate an 80 % two-sided band;
jointly they under-cover slightly — the compare harness scores the joint
band directly, so if that costs the model the A/B, it loses fairly.

Uses sklearn's ``HistGradientBoostingRegressor(loss="quantile")`` (already
in the ``[ml]`` extra; the roadmap rejects LightGBM — no cp314 wheels) and
degrades to ``None`` without it. Crossing quantiles are monotonized by
swapping. Like every band source, this ships into the engine ONLY on a
``pass`` from the disjoint-OOS compare-bands gate — the same gate that
already failed GARCH-FHS.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

UPPER_Q = 0.90
LOWER_Q = 0.10
_MIN_TRAIN_ROWS = 200
_sklearn_missing_logged = False


def fit_qgbm(rows: list[Any]) -> tuple[Any, Any] | None:
    """Fit the (upper, lower) quantile models on comparison rows.

    ``rows`` are ``band_compare._Row`` instances (need ``features``,
    ``close``, ``realized_high/low``). Returns ``None`` when sklearn is
    unavailable, the training set is too thin (< 200 rows), or rows carry
    no features — callers simply omit the source.
    """
    global _sklearn_missing_logged
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        if not _sklearn_missing_logged:
            logger.info("sklearn not installed ([ml] extra) — qgbm bands disabled")
            _sklearn_missing_logged = True
        return None
    usable = [r for r in rows if r.features]
    if len(usable) < _MIN_TRAIN_ROWS:
        return None
    x = np.asarray([r.features for r in usable], dtype=np.float64)
    y_high = np.log([r.realized_high / r.close for r in usable])
    y_low = np.log([r.realized_low / r.close for r in usable])
    # Small trees + strong regularization: ~8 features, hundreds of rows —
    # anything deeper memorizes the panel.
    kwargs: dict[str, Any] = {
        "max_iter": 150,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_samples_leaf": 40,
        "random_state": 0,
    }
    hi = HistGradientBoostingRegressor(loss="quantile", quantile=UPPER_Q, **kwargs)
    lo = HistGradientBoostingRegressor(loss="quantile", quantile=LOWER_Q, **kwargs)
    hi.fit(x, y_high)
    lo.fit(x, y_low)
    return hi, lo


def predict_bands(models: tuple[Any, Any], rows: list[Any]) -> list[tuple[float, float]]:
    """Predict (low, high) price bands for comparison rows.

    Crossing predictions (low ≥ high — rare but possible with independent
    quantile models) are monotonized by swapping; a degenerate band is
    floored to ±0.1 % around the close so the interval never inverts.
    """
    hi_m, lo_m = models
    x = np.asarray([r.features for r in rows], dtype=np.float64)
    hi_pred = hi_m.predict(x)
    lo_pred = lo_m.predict(x)
    out: list[tuple[float, float]] = []
    for r, hp, lp in zip(rows, hi_pred, lo_pred, strict=True):
        high = r.close * float(np.exp(hp))
        low = r.close * float(np.exp(lp))
        if low > high:
            low, high = high, low
        if high - low < r.close * 0.002:
            low = r.close * 0.999
            high = r.close * 1.001
        out.append((low, high))
    return out
