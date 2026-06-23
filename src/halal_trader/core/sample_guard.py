"""Minimum-sample guard for learned statistics.

A win-rate off 4 trades, a Kelly fraction off 6, a calibration curve off a
dozen predictions, an IC off two weeks — these are noise dressed as signal.
Acting on them is how a small bot fools itself. This is the shared gate every
learned statistic passes through: below ``min_n`` observations the statistic is
"insufficient" and callers fall back to a safe default rather than betting on
noise.

Pure + dependency-free so sizing (Kelly/vol-target), calibration, the IC
harness, and the recommendation scorecard all share one definition of "enough".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")

# Default "enough observations to trust a rate/ratio" threshold. ~20 is the
# common rule-of-thumb floor where a proportion's standard error starts to be
# informative; callers needing more (e.g. Kelly) should pass a higher min_n.
DEFAULT_MIN_SAMPLES = 20


@dataclass(frozen=True, slots=True)
class SampleGate:
    """Whether ``n`` observations clear ``min_n`` to trust a learned stat."""

    n: int
    min_n: int = DEFAULT_MIN_SAMPLES

    @property
    def sufficient(self) -> bool:
        return self.n >= self.min_n

    @property
    def shortfall(self) -> int:
        """How many more observations are needed (0 once sufficient)."""
        return max(0, self.min_n - self.n)


def gate_stat(value: T, n: int, *, min_n: int = DEFAULT_MIN_SAMPLES, fallback: T) -> T:
    """Return ``value`` only when ``n >= min_n``, else ``fallback``.

    The one-liner every learned stat uses: a Kelly fraction falls back to 0
    (no edge bet on thin data), a calibrated confidence falls back to the raw
    confidence, an IC falls back to None. Never let a small-N estimate act.
    """
    return value if n >= min_n else fallback
