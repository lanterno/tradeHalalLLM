"""Google Trends-style signal extractor — Round-5 Wave 11.B.

Search-volume time series can be a leading indicator of retail
interest in a name. This module ships the **trend-signal extractor**:
given a time-indexed series of normalised search interest, it
identifies surge / fade signals + classifies the regime.

Pinned semantics:

- **Closed-set TrendSignal ladder** (NEUTRAL / EMERGING_INTEREST /
  PEAKING / FADING / DEAD).
- **Surge threshold** = z-score against rolling 30-period mean.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class TrendSignal(str, Enum):
    """Closed-set trend signals."""

    NEUTRAL = "neutral"
    EMERGING_INTEREST = "emerging_interest"
    PEAKING = "peaking"
    FADING = "fading"
    DEAD = "dead"


@dataclass(frozen=True)
class TrendPolicy:
    """Operator-tunable thresholds."""

    surge_z_threshold: float = 2.0
    peak_z_threshold: float = 3.0
    dead_pct_of_peak: float = 0.10
    rolling_window: int = 30

    def __post_init__(self) -> None:
        if not 0.5 <= self.surge_z_threshold < self.peak_z_threshold:
            raise ValueError("surge_z_threshold < peak_z_threshold required")
        if not 0.0 < self.dead_pct_of_peak < 0.5:
            raise ValueError("dead_pct_of_peak must be in (0, 0.5)")
        if self.rolling_window < 5:
            raise ValueError("rolling_window must be >= 5")


@dataclass(frozen=True)
class TrendAssessment:
    """Result of trend-signal extraction."""

    keyword: str
    signal: TrendSignal
    z_score_latest: float
    pct_of_peak: float
    n_observations: int

    def __post_init__(self) -> None:
        if not self.keyword or not self.keyword.strip():
            raise ValueError("keyword must be non-empty")
        if not 0.0 <= self.pct_of_peak <= 5.0:
            raise ValueError("pct_of_peak must be in [0, 5]")
        if self.n_observations < 0:
            raise ValueError("n_observations must be non-negative")


def assess(
    keyword: str,
    series: Sequence[float],
    *,
    policy: TrendPolicy | None = None,
) -> TrendAssessment:
    """Extract a trend signal from a normalised search-interest series."""
    if not keyword or not keyword.strip():
        raise ValueError("keyword must be non-empty")
    pol = policy if policy is not None else TrendPolicy()
    n = len(series)
    if n == 0:
        return TrendAssessment(
            keyword=keyword,
            signal=TrendSignal.NEUTRAL,
            z_score_latest=0.0,
            pct_of_peak=0.0,
            n_observations=0,
        )
    if any(v < 0 for v in series):
        raise ValueError("series values must be non-negative")

    latest = series[-1]
    peak = max(series)
    pct_of_peak = latest / peak if peak > 0 else 0.0

    if n < pol.rolling_window:
        # Too few observations for z-score
        signal = TrendSignal.NEUTRAL
        z_score = 0.0
    else:
        window = list(series[-pol.rolling_window :])
        mean = statistics.mean(window)
        stdev = statistics.pstdev(window)
        z_score = (latest - mean) / stdev if stdev > 0 else 0.0

        if pct_of_peak <= pol.dead_pct_of_peak:
            signal = TrendSignal.DEAD
        elif z_score >= pol.peak_z_threshold:
            signal = TrendSignal.PEAKING
        elif z_score >= pol.surge_z_threshold:
            signal = TrendSignal.EMERGING_INTEREST
        elif stdev > 0 and latest < mean and pct_of_peak < 0.50:
            signal = TrendSignal.FADING
        else:
            signal = TrendSignal.NEUTRAL

    return TrendAssessment(
        keyword=keyword,
        signal=signal,
        z_score_latest=z_score,
        pct_of_peak=pct_of_peak,
        n_observations=n,
    )


def render_assessment(a: TrendAssessment) -> str:
    emoji = {
        TrendSignal.NEUTRAL: "⚪",
        TrendSignal.EMERGING_INTEREST: "📈",
        TrendSignal.PEAKING: "🔥",
        TrendSignal.FADING: "📉",
        TrendSignal.DEAD: "💀",
    }[a.signal]
    return (
        f"{emoji} {a.keyword}: {a.signal.value} "
        f"z={a.z_score_latest:+.2f} pct_of_peak={a.pct_of_peak * 100:.1f}%"
    )
