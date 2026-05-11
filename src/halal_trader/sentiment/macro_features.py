"""Macro feature engineering primitives — Round-5 Wave 11.I.

FRED / IMF / World Bank publish thousands of macro time series. The
operator-side problem is *picking the most predictive ~50 per regime*;
the platform-side problem is shipping a clean primitive layer for:

1. **Storage** — series + observations as immutable dataclasses, keyed
   by `series_id` (e.g. "FRED:UNRATE").
2. **Feature engineering** — z-score, MoM (month-over-month), YoY,
   lagged differences. All pure-Python, deterministic, NaN-safe.
3. **Regime-keyed feature ranker** — given a set of regime labels and
   a forward-return series, rank features by their absolute Spearman
   correlation. Top-K survives.

This module is the **pure-functional primitive layer**. The ingestion
adapter (FRED API key, IMF SDMX, World Bank JSON) lives outside; this
layer accepts already-parsed `MacroSeries` objects.

Pinned semantics:

- **Closed-set Frequency** — DAILY / WEEKLY / MONTHLY / QUARTERLY /
  ANNUAL. Mixed-frequency operations require the caller to resample
  first.
- **z-score uses sample std** (ddof=1), windowed.
- **MoM and YoY** — return *None* when insufficient history, never
  raise. Operators decide how to handle missing values upstream.
- **Spearman correlation is rank-based** — robust to outliers.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum


class Frequency(str, Enum):
    """Closed-set frequency ladder."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


@dataclass(frozen=True)
class Observation:
    """One observation in a macro series."""

    obs_date: date
    value: float

    def __post_init__(self) -> None:
        if math.isnan(self.value):
            raise ValueError("value must not be NaN")
        if math.isinf(self.value):
            raise ValueError("value must be finite")


@dataclass(frozen=True)
class MacroSeries:
    """A named, frequency-pinned macro time series."""

    series_id: str
    name: str
    frequency: Frequency
    observations: tuple[Observation, ...]
    """Sorted ascending by obs_date."""

    def __post_init__(self) -> None:
        if not self.series_id or not self.series_id.strip():
            raise ValueError("series_id must be non-empty")
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.observations:
            return
        # Strict-ascending date order.
        dates = [o.obs_date for o in self.observations]
        if dates != sorted(dates):
            raise ValueError("observations must be sorted by obs_date")
        if len(set(dates)) != len(dates):
            raise ValueError("observations must have unique obs_dates")

    def values(self) -> tuple[float, ...]:
        return tuple(o.value for o in self.observations)

    def dates(self) -> tuple[date, ...]:
        return tuple(o.obs_date for o in self.observations)

    def latest(self) -> Observation | None:
        if not self.observations:
            return None
        return self.observations[-1]


# --- Feature primitives ------------------------------------------------


def z_score(series: MacroSeries, *, window: int = 36) -> tuple[float | None, ...]:
    """Rolling-window z-score per observation.

    Pinned: ddof=1 (sample std). Returns None for the first
    `window-1` entries (not enough history).
    """
    if window < 2:
        raise ValueError("window must be ≥ 2")
    vals = series.values()
    out: list[float | None] = []
    for i in range(len(vals)):
        if i < window - 1:
            out.append(None)
            continue
        win = vals[i - window + 1 : i + 1]
        mean = sum(win) / len(win)
        var = sum((v - mean) ** 2 for v in win) / max(1, len(win) - 1)
        std = math.sqrt(var)
        if std < 1e-12:
            out.append(0.0)
        else:
            out.append((vals[i] - mean) / std)
    return tuple(out)


def month_over_month(
    series: MacroSeries,
) -> tuple[float | None, ...]:
    """Period-over-period delta (current / previous - 1).

    For non-MONTHLY frequencies, this is "previous-period-over". The
    name is preserved for FRED-naming conventions.
    """
    vals = series.values()
    out: list[float | None] = [None]
    for i in range(1, len(vals)):
        prev = vals[i - 1]
        if abs(prev) < 1e-12:
            out.append(None)
        else:
            out.append(vals[i] / prev - 1.0)
    return tuple(out)


def year_over_year(
    series: MacroSeries,
) -> tuple[float | None, ...]:
    """Year-over-year delta, frequency-aware.

    Lookback per frequency:
    - DAILY: 252 (trading days)
    - WEEKLY: 52
    - MONTHLY: 12
    - QUARTERLY: 4
    - ANNUAL: 1
    """
    lookback = {
        Frequency.DAILY: 252,
        Frequency.WEEKLY: 52,
        Frequency.MONTHLY: 12,
        Frequency.QUARTERLY: 4,
        Frequency.ANNUAL: 1,
    }[series.frequency]
    vals = series.values()
    out: list[float | None] = []
    for i in range(len(vals)):
        if i < lookback:
            out.append(None)
            continue
        prev = vals[i - lookback]
        if abs(prev) < 1e-12:
            out.append(None)
        else:
            out.append(vals[i] / prev - 1.0)
    return tuple(out)


def lagged_diff(series: MacroSeries, *, lag: int = 1) -> tuple[float | None, ...]:
    """Difference vs `lag`-period prior (no normalisation)."""
    if lag <= 0:
        raise ValueError("lag must be positive")
    vals = series.values()
    out: list[float | None] = []
    for i in range(len(vals)):
        if i < lag:
            out.append(None)
        else:
            out.append(vals[i] - vals[i - lag])
    return tuple(out)


# --- Spearman correlation ----------------------------------------------


def _ranks(xs: Sequence[float]) -> tuple[float, ...]:
    """Average-ranks for ties (standard Spearman pre-step)."""
    indexed = sorted(enumerate(xs), key=lambda kv: kv[1])
    out: list[float] = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        # Group ties.
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            out[indexed[k][0]] = avg_rank
        i = j + 1
    return tuple(out)


def spearman_correlation(
    xs: Sequence[float | None],
    ys: Sequence[float | None],
) -> float | None:
    """Spearman rank correlation, NaN-safe.

    Drops index pairs where either side is None. Returns None if fewer
    than 3 valid pairs survive.
    """
    if len(xs) != len(ys):
        raise ValueError("length mismatch")
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    rx = _ranks(x_vals)
    ry = _ranks(y_vals)
    n = len(rx)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    den_x = math.sqrt(sum((r - mean_x) ** 2 for r in rx))
    den_y = math.sqrt(sum((r - mean_y) ** 2 for r in ry))
    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0
    return num / (den_x * den_y)


# --- Feature ranker ----------------------------------------------------


@dataclass(frozen=True)
class FeatureRank:
    """One ranked feature."""

    series_id: str
    feature_name: str
    """e.g. 'z_score', 'mom', 'yoy', 'lagged_diff_1'."""
    correlation: float
    n_pairs: int


def rank_features(
    features: dict[str, dict[str, Sequence[float | None]]],
    target: Sequence[float | None],
    *,
    top_k: int = 50,
) -> tuple[FeatureRank, ...]:
    """Rank features by absolute Spearman correlation against target.

    `features` is a dict of `series_id -> dict[feature_name -> values]`.
    Each values sequence must align with `target` by index.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rows: list[FeatureRank] = []
    for series_id, fbag in features.items():
        for fname, vals in fbag.items():
            corr = spearman_correlation(vals, target)
            if corr is None:
                continue
            n = sum(1 for x, y in zip(vals, target, strict=True) if x is not None and y is not None)
            rows.append(
                FeatureRank(
                    series_id=series_id,
                    feature_name=fname,
                    correlation=corr,
                    n_pairs=n,
                )
            )
    rows.sort(key=lambda r: (-abs(r.correlation), r.series_id, r.feature_name))
    return tuple(rows[:top_k])


def render_rank(rank: FeatureRank) -> str:
    return (
        f"📈 {rank.series_id}::{rank.feature_name} corr={rank.correlation:+.3f} (n={rank.n_pairs})"
    )


def render_top(features: Iterable[FeatureRank], *, top_n: int = 10) -> str:
    rows = tuple(features)[:top_n]
    if not rows:
        return "📊 No features ranked."
    lines = [f"📊 Top {len(rows)} features:"]
    for r in rows:
        lines.append(f"  • {render_rank(r)}")
    return "\n".join(lines)
