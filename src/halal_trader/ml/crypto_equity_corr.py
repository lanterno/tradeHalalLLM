"""Crypto-on-chain → equity correlation engine — Round-5 Wave 11.J.

Coinbase volume, Tether mint patterns, and on-chain stablecoin flow
correlate (sometimes leading) with US equity flows. This module is
the **rolling-window correlator + lagged-leadership detector**:

1. Given two aligned time series (e.g. daily Tether mint $ and SPY
   close), compute the rolling Pearson correlation over a window.
2. Compute correlation across a *lag grid* (e.g. -10..+10 days). The
   lag with maximum absolute correlation indicates leadership: a
   negative lag says crypto leads equity; a positive lag says equity
   leads crypto.
3. Surface "leadership shift" events when the dominant lag flips sign
   from one window to the next.

This module is the primitive layer; the live ingestion (Coinbase API,
Etherscan, Glassnode) lives elsewhere.

Pinned semantics:

- **Pearson correlation** (parametric); for rank-based use
  `sentiment.macro_features.spearman_correlation`.
- **Aligned series only**. Caller resamples to a common frequency
  before passing in.
- **NaN-safe**: index pairs with None on either side are dropped.
- **Closed-set Leadership** — CRYPTO_LEADS / EQUITY_LEADS / SYNCHRONOUS.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum


class Leadership(str, Enum):
    """Closed-set leadership ladder."""

    CRYPTO_LEADS = "crypto_leads"
    EQUITY_LEADS = "equity_leads"
    SYNCHRONOUS = "synchronous"
    """|best lag| ≤ 1 → treated as synchronous."""


@dataclass(frozen=True)
class TimeSeriesPoint:
    """One aligned data point."""

    obs_date: date
    crypto_value: float | None
    equity_value: float | None

    def __post_init__(self) -> None:
        for name, v in (
            ("crypto_value", self.crypto_value),
            ("equity_value", self.equity_value),
        ):
            if v is not None:
                if math.isnan(v) or math.isinf(v):
                    raise ValueError(f"{name} must be finite")


def pearson_correlation(
    xs: Sequence[float | None],
    ys: Sequence[float | None],
) -> float | None:
    """Pearson r, NaN-safe. Returns None if fewer than 3 valid pairs.

    Returns 0.0 if either series has zero variance after NaN drop.
    """
    if len(xs) != len(ys):
        raise ValueError("length mismatch")
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    n = len(pairs)
    mean_x = sum(p[0] for p in pairs) / n
    mean_y = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mean_x) * (p[1] - mean_y) for p in pairs)
    den_x = math.sqrt(sum((p[0] - mean_x) ** 2 for p in pairs))
    den_y = math.sqrt(sum((p[1] - mean_y) ** 2 for p in pairs))
    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0
    return num / (den_x * den_y)


def rolling_correlation(
    points: Sequence[TimeSeriesPoint],
    *,
    window: int,
) -> tuple[float | None, ...]:
    """Rolling Pearson correlation over `window` points.

    Output index i is the correlation over points [i - window + 1 .. i].
    First `window - 1` entries are None.
    """
    if window < 3:
        raise ValueError("window must be ≥ 3")
    n = len(points)
    out: list[float | None] = []
    for i in range(n):
        if i < window - 1:
            out.append(None)
            continue
        win = points[i - window + 1 : i + 1]
        xs = [p.crypto_value for p in win]
        ys = [p.equity_value for p in win]
        out.append(pearson_correlation(xs, ys))
    return tuple(out)


def _shift_left(seq: Sequence[float | None], n: int) -> list[float | None]:
    """Shift the sequence left by n (drop first n; pad right with None)."""
    if n <= 0:
        return list(seq)
    return list(seq[n:]) + [None] * n


@dataclass(frozen=True)
class LagCorrelation:
    """One row of the lag-correlation table.

    Pinned convention: at lag=k>0 we align ``xs = crypto[:-k]`` with
    ``ys = equity[k:]``, i.e. crypto[t] is paired with equity[t+k].
    A high correlation at positive lag therefore means *crypto leads
    equity*. At negative lag, equity leads crypto.
    """

    lag: int
    correlation: float
    n_pairs: int


def lag_correlation_grid(
    points: Sequence[TimeSeriesPoint],
    *,
    max_lag: int = 10,
) -> tuple[LagCorrelation, ...]:
    """Compute correlation across a lag grid of [-max_lag, +max_lag].

    For lag k > 0, we test whether equity[t] correlates with crypto[t-k]
    — i.e. crypto k-periods-prior leads equity now.
    For lag k < 0, we test the inverse.
    """
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")
    crypto = [p.crypto_value for p in points]
    equity = [p.equity_value for p in points]
    out: list[LagCorrelation] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            # equity[t] vs crypto[t - lag] → align by trimming.
            xs = crypto[:-lag] if lag < len(crypto) else []
            ys = equity[lag:] if lag < len(equity) else []
        elif lag < 0:
            k = -lag
            xs = crypto[k:] if k < len(crypto) else []
            ys = equity[:-k] if k < len(equity) else []
        else:
            xs = list(crypto)
            ys = list(equity)
        if len(xs) < 3:
            continue
        corr = pearson_correlation(xs, ys)
        if corr is None:
            continue
        n_pairs = sum(1 for x, y in zip(xs, ys, strict=True) if x is not None and y is not None)
        out.append(LagCorrelation(lag=lag, correlation=corr, n_pairs=n_pairs))
    return tuple(out)


@dataclass(frozen=True)
class LeadershipReport:
    """Output of `detect_leadership`."""

    best_lag: int
    best_correlation: float
    leadership: Leadership
    grid: tuple[LagCorrelation, ...]


def detect_leadership(
    points: Sequence[TimeSeriesPoint],
    *,
    max_lag: int = 10,
    synchronous_threshold: int = 1,
) -> LeadershipReport:
    """Find the lag with peak |correlation|; classify leadership."""
    if synchronous_threshold < 0:
        raise ValueError("synchronous_threshold must be ≥ 0")
    grid = lag_correlation_grid(points, max_lag=max_lag)
    if not grid:
        raise ValueError("not enough data to compute lag grid")
    best = max(grid, key=lambda r: abs(r.correlation))
    if abs(best.lag) <= synchronous_threshold:
        leadership = Leadership.SYNCHRONOUS
    elif best.lag > 0:
        # Positive lag: crypto[t] correlates with equity[t+lag] →
        # crypto leads equity.
        leadership = Leadership.CRYPTO_LEADS
    else:
        leadership = Leadership.EQUITY_LEADS
    return LeadershipReport(
        best_lag=best.lag,
        best_correlation=best.correlation,
        leadership=leadership,
        grid=grid,
    )


@dataclass(frozen=True)
class LeadershipShift:
    """Output of `detect_leadership_shifts`."""

    window_index: int
    """Index of the *new* window where the shift was detected."""
    prior_leadership: Leadership
    new_leadership: Leadership


def detect_leadership_shifts(
    points: Sequence[TimeSeriesPoint],
    *,
    window: int,
    max_lag: int = 10,
) -> tuple[LeadershipShift, ...]:
    """Walk overlapping windows and surface leadership-shift events.

    Pinned: a shift fires when the new window's leadership ≠ the prior
    window's leadership. Synchronous windows count as a distinct
    state.
    """
    if window < 2 * max_lag + 1:
        raise ValueError("window must be ≥ 2*max_lag + 1")
    out: list[LeadershipShift] = []
    prior_leadership: Leadership | None = None
    for i in range(window, len(points) + 1):
        win = points[i - window : i]
        try:
            report = detect_leadership(win, max_lag=max_lag)
        except ValueError:
            continue
        if prior_leadership is not None and report.leadership is not prior_leadership:
            out.append(
                LeadershipShift(
                    window_index=i,
                    prior_leadership=prior_leadership,
                    new_leadership=report.leadership,
                )
            )
        prior_leadership = report.leadership
    return tuple(out)


_LEADERSHIP_EMOJI: dict[Leadership, str] = {
    Leadership.CRYPTO_LEADS: "🟠",
    Leadership.EQUITY_LEADS: "🔵",
    Leadership.SYNCHRONOUS: "⚪",
}


def render_report(report: LeadershipReport) -> str:
    return (
        f"{_LEADERSHIP_EMOJI[report.leadership]} {report.leadership.value}: "
        f"best_lag={report.best_lag:+d}, "
        f"corr={report.best_correlation:+.3f} "
        f"(grid={len(report.grid)} points)"
    )


def render_shift(shift: LeadershipShift) -> str:
    return (
        f"🔄 Window {shift.window_index}: "
        f"{shift.prior_leadership.value} → {shift.new_leadership.value}"
    )
