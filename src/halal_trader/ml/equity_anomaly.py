"""Equity-curve anomaly detector — flag when *our own* P&L deviates
from its baseline distribution.

Round-4 wave 4.I: complements the existing market-side anomaly
detector in ``ml/anomaly.py`` (which flags weird *market* state) with
its mirror — flag weird *bot* state. Two failure modes operators care
about:

* **Drawdown anomaly.** A streak of losses that exceeds what the
  recent return distribution should produce. The classic "your edge
  may have eroded" signal — caused by regime change, prompt drift,
  data-source rot, broker outage. The right response is to reduce
  size or halt and investigate.
* **Hot-streak anomaly.** An unusually large positive deviation. Less
  obviously bad, but a meaningful tail event still deserves operator
  awareness — usually means we got lucky on a single high-conviction
  trade, and the operator should avoid extrapolating it into raised
  position sizing.

Both detectors are statistical and parameter-free in spirit (they
don't fit a model; they compute z-scores against a rolling window).
A z-score that crosses ``z_threshold`` (default 3 — the 0.27%
two-tailed tail of a normal) trips the alert. The detector is
strategy- and broker-agnostic — operates on a flat numpy array of
per-trade returns or per-period equity values, ordered oldest →
newest. The caller is responsible for pulling the array out of
Postgres / the in-memory ledger.

Halal alignment: the response to a tripped detector is to *halt*
trading, not to short or hedge. The detector exposes only the
signal; the wiring decides how to act on it (`core/halt.py` is the
intended consumer for the auto-halt path).

Pure-numpy + math; no scipy / sklearn dependency, so the module
imports without the ``[ml]`` extra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EquityAnomalyReport:
    """Outcome of a single check.

    ``z_score`` is the (return − mean) / std value computed on the
    most-recent observation against the lookback window. Positive
    values indicate hot-streak deviation; negative indicate
    drawdown deviation.

    ``severity`` is a stable label for the dashboard / notifier:
    ``"normal"``, ``"warn"``, or ``"alert"``. ``warn`` corresponds to
    ``|z| ≥ z_warn`` (default 2 — the 4.6% two-tailed tail);
    ``alert`` corresponds to ``|z| ≥ z_threshold`` (default 3).

    ``direction`` is ``"drawdown"`` / ``"hot"`` / ``"normal"`` for
    UI rendering. ``recommendation`` is a one-line operator-readable
    summary suitable for a Telegram / Slack message.
    """

    z_score: float
    severity: str
    direction: str
    window_size: int
    window_mean: float
    window_std: float
    last_value: float
    recommendation: str


def _trim_to_finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Strip NaN / inf — closed trades occasionally have a missing
    ``return_pct``; one bad row mustn't poison the rolling stats."""
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _format_recommendation(z: float, direction: str, severity: str) -> str:
    """Produce a single-line operator-readable nudge.

    The wording deliberately doesn't prescribe an exact action — the
    operator's job is to investigate. We just frame the question.
    """
    if severity == "normal":
        return "Equity curve is within normal bounds."
    if direction == "drawdown":
        if severity == "alert":
            return (
                f"Drawdown anomaly (z={z:+.2f}). Edge may have eroded — "
                "consider halting and investigating."
            )
        return f"Drawdown warning (z={z:+.2f}). Watch the next few cycles before sizing up."
    # hot streak
    if severity == "alert":
        return (
            f"Hot-streak anomaly (z={z:+.2f}). Unusually positive tail — "
            "avoid raising position sizes off this run."
        )
    return f"Hot-streak warning (z={z:+.2f}). Streak is above baseline."


def detect_return_anomaly(
    returns: Sequence[float] | np.ndarray,
    *,
    z_warn: float = 2.0,
    z_threshold: float = 3.0,
    min_window: int = 30,
) -> EquityAnomalyReport:
    """Score the *last* return against the prior window.

    ``returns`` is an ordered (oldest first) array of per-trade
    return-percentages. The function compares ``returns[-1]`` to the
    distribution of ``returns[:-1]`` and reports the resulting
    z-score plus severity / direction labels.

    ``min_window`` rejects the check on too-thin a history; below
    that, the detector returns a ``"normal"`` report — we'd rather
    say "I don't know" than fire false alerts on cold-start.
    """
    if z_warn < 0 or z_threshold < 0:
        raise ValueError("z thresholds must be non-negative")
    if z_threshold < z_warn:
        raise ValueError("z_threshold must be >= z_warn")

    arr = _trim_to_finite(returns)
    if arr.size < min_window + 1:
        return EquityAnomalyReport(
            z_score=0.0,
            severity="normal",
            direction="normal",
            window_size=int(arr.size - 1) if arr.size > 0 else 0,
            window_mean=0.0,
            window_std=0.0,
            last_value=float(arr[-1]) if arr.size > 0 else 0.0,
            recommendation=_format_recommendation(0.0, "normal", "normal"),
        )

    last = float(arr[-1])
    window = arr[:-1]
    mean = float(window.mean())
    # ddof=1 sample std — we're estimating from a finite window.
    std = float(window.std(ddof=1))
    if std == 0:
        # Degenerate baseline (every prior trade returned the same
        # value). We report 0 z-score rather than +∞ so the alert
        # path doesn't trip on cold-start synthetic data.
        z = 0.0
    else:
        z = (last - mean) / std

    if abs(z) >= z_threshold:
        severity = "alert"
    elif abs(z) >= z_warn:
        severity = "warn"
    else:
        severity = "normal"
    if severity == "normal":
        direction = "normal"
    elif z < 0:
        direction = "drawdown"
    else:
        direction = "hot"

    return EquityAnomalyReport(
        z_score=z,
        severity=severity,
        direction=direction,
        window_size=int(window.size),
        window_mean=mean,
        window_std=std,
        last_value=last,
        recommendation=_format_recommendation(z, direction, severity),
    )


def detect_drawdown_anomaly(
    equity_curve: Sequence[float] | np.ndarray,
    *,
    z_warn: float = 2.0,
    z_threshold: float = 3.0,
    min_window: int = 30,
) -> EquityAnomalyReport:
    """Drawdown-specific check on a cumulative equity curve.

    ``equity_curve`` is the cumulative equity (starting balance × ∏(1+r)),
    ordered oldest → newest. The detector measures the *current
    drawdown from peak* and z-scores it against the historical
    drawdown distribution computed on the same curve.

    Why a separate function from :func:`detect_return_anomaly`:
    drawdown is a path-dependent statistic (it depends on the
    running maximum), while per-trade returns are not. They expose
    different anomalies: a string of small losses won't trip the
    return detector but can drive a multi-week peak-to-trough
    drawdown that the operator must see.
    """
    if z_warn < 0 or z_threshold < 0:
        raise ValueError("z thresholds must be non-negative")
    if z_threshold < z_warn:
        raise ValueError("z_threshold must be >= z_warn")

    arr = _trim_to_finite(equity_curve)
    if arr.size < min_window + 1 or (arr <= 0).any():
        # Degenerate history (too short, or contains a non-positive
        # equity value which would explode the percent calc).
        return EquityAnomalyReport(
            z_score=0.0,
            severity="normal",
            direction="normal",
            window_size=int(arr.size),
            window_mean=0.0,
            window_std=0.0,
            last_value=float(arr[-1]) if arr.size > 0 else 0.0,
            recommendation=_format_recommendation(0.0, "normal", "normal"),
        )

    running_peak = np.maximum.accumulate(arr)
    drawdown_pct = (arr - running_peak) / running_peak  # ≤ 0
    last_dd = float(drawdown_pct[-1])
    history = drawdown_pct[:-1]
    mean = float(history.mean())
    std = float(history.std(ddof=1)) if history.size > 1 else 0.0

    if std == 0:
        z = 0.0
    else:
        z = (last_dd - mean) / std

    # Drawdown is bounded above by 0, so a "hot" deviation isn't
    # meaningful — only the negative tail matters. Clamp anything
    # above zero to severity normal regardless of magnitude.
    if z >= 0:
        severity = "normal"
        direction = "normal"
    elif abs(z) >= z_threshold:
        severity = "alert"
        direction = "drawdown"
    elif abs(z) >= z_warn:
        severity = "warn"
        direction = "drawdown"
    else:
        severity = "normal"
        direction = "normal"

    return EquityAnomalyReport(
        z_score=z,
        severity=severity,
        direction=direction,
        window_size=int(history.size),
        window_mean=mean,
        window_std=std,
        last_value=last_dd,
        recommendation=_format_recommendation(z, direction, severity),
    )


def equity_curve_from_returns(
    returns: Sequence[float] | np.ndarray, *, starting: float = 1.0
) -> np.ndarray:
    """Convenience: build a cumulative equity curve from a series of
    per-trade returns. ``starting`` defaults to 1.0 so the resulting
    array is unitless / equity-multiple; pass actual starting
    cash if the caller wants USD-denominated values."""
    arr = _trim_to_finite(returns)
    if arr.size == 0:
        return np.array([starting], dtype=float)
    if starting <= 0:
        raise ValueError(f"starting must be positive; got {starting}")
    return starting * np.cumprod(1.0 + arr)


# Sanity: export math.erf so callers can build their own one-tail
# probability if they want to convert a z-score to a p-value without
# pulling in scipy. Kept minimal — equity-anomaly callers virtually
# always want the labelled severity, not a raw probability.
__all__ = [
    "EquityAnomalyReport",
    "detect_drawdown_anomaly",
    "detect_return_anomaly",
    "equity_curve_from_returns",
    "math",
]
