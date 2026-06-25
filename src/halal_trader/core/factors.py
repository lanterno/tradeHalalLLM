"""Cross-sectional factor ranking — one core, many consumers.

Ranks a universe by a long-only composite of three classic, halal-compatible
factors:

- **momentum** — trailing return (winners keep winning, on average);
- **low-volatility** — lower realised vol scores higher (the low-vol anomaly);
- **trend-quality** — ADX-style trend strength (a clean trend beats a choppy one).

Each factor is z-scored *across the universe* (cross-sectional), then blended.
The result is a ranked, long-only tilt — no shorts, the bottom of the ranking is
simply "don't buy", not "sell short". Pure (numpy); the same core feeds the
advisory recommendation today and could feed a live top-N tilt later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Default blend weights (momentum, low_vol, trend_quality) — equal-weight.
DEFAULT_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)


@dataclass(frozen=True, slots=True)
class FactorScore:
    """One symbol's cross-sectional factor breakdown (z-scores) + composite."""

    symbol: str
    composite: float
    momentum: float
    low_vol: float
    trend_quality: float


def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if np.isfinite(f) else 0.0


def _first(metrics: dict[str, Any], *keys: str) -> float:
    for k in keys:
        if metrics.get(k) is not None:
            return _num(metrics[k])
    return 0.0


def _low_vol_raw(metrics: dict[str, Any]) -> float:
    """Negated relative volatility (ATR / price) → lower vol scores higher."""
    atr = _first(metrics, "atr")
    price = _first(metrics, "price")
    if price <= 0:
        return 0.0
    return -(atr / price)


def _zscores(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    std = float(arr.std())
    if std == 0.0:
        return [0.0] * len(values)
    mean = float(arr.mean())
    return [float(x) for x in (arr - mean) / std]


def rank_factors(
    metrics: dict[str, dict[str, Any]],
    *,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> list[FactorScore]:
    """Rank a universe descending by the composite factor score.

    ``metrics`` maps symbol → its metric dict (expects ``chg20d``/``chg5d`` for
    momentum, ``atr`` + ``price`` for low-vol, ``adx`` for trend-quality;
    missing values degrade to 0 for that factor). Long-only: the ranking is a
    buy-tilt, never a short list.
    """
    symbols = list(metrics.keys())
    if not symbols:
        return []
    momentum = [_first(metrics[s], "chg20d", "chg5d") for s in symbols]
    low_vol = [_low_vol_raw(metrics[s]) for s in symbols]
    trend = [_first(metrics[s], "adx") for s in symbols]
    zm, zlv, ztq = _zscores(momentum), _zscores(low_vol), _zscores(trend)
    wm, wlv, wtq = weights
    scores = [
        FactorScore(
            symbol=s,
            composite=round(wm * zm[i] + wlv * zlv[i] + wtq * ztq[i], 4),
            momentum=round(zm[i], 4),
            low_vol=round(zlv[i], 4),
            trend_quality=round(ztq[i], 4),
        )
        for i, s in enumerate(symbols)
    ]
    scores.sort(key=lambda f: f.composite, reverse=True)
    return scores
