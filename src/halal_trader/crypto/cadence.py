"""Adaptive cycle-interval selector.

The legacy crypto loop ran on a fixed 60s tick. That under-samples
during fast moves (we miss exits and entries) and over-samples during
chop (burning LLM tokens on cycles where nothing's changed). The
selector below maps the universe-median ATR (relative to the
configured baseline) to a cycle interval inside operator-set bounds.

Conservative defaults:

    * vol ≥ 1.5× baseline  →  half the configured interval (faster)
    * vol ≤ 0.7× baseline  →  double the configured interval (slower)
    * otherwise the configured interval

Returned values are always clamped to ``[min_interval, max_interval]``
so a runaway scaling factor can never push the bot to a 1-second loop
or a 10-minute loop. Pure function — no I/O, no asyncio. The scheduler
calls it at the top of each loop and uses the result for the next
sleep.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class CadenceDecision:
    """Output of :func:`select_interval`. Carries the *why* for logging."""

    interval_seconds: int
    regime: str  # "fast" | "normal" | "slow"
    median_atr: float
    ratio: float


def select_interval(
    *,
    indicators_cache: dict[str, dict],
    base_interval: int,
    atr_baseline: float,
    min_interval: int = 15,
    max_interval: int = 300,
    fast_threshold: float = 1.5,
    slow_threshold: float = 0.7,
) -> CadenceDecision:
    """Pick the next cycle's sleep interval based on universe-median ATR.

    ``indicators_cache`` is the dict-per-symbol shape the cycle already
    builds. We pull ``atr_pct`` (preferred) or ``atr_14`` per symbol and
    use the median so a single highly-volatile pair doesn't drag the
    entire bot into 5s loops.

    Returns the configured ``base_interval`` (still clamped) when there
    is no usable ATR signal — defaulting to "do what was configured" is
    safer than picking an extreme.
    """
    if base_interval <= 0:
        raise ValueError(f"base_interval must be positive; got {base_interval}")
    base_interval = max(min_interval, min(max_interval, base_interval))

    atrs = [
        ind.get("atr_pct", ind.get("atr_14", 0))
        for ind in (indicators_cache or {}).values()
        if not ind.get("error")
    ]
    atrs = [a for a in atrs if a and a > 0]

    if not atrs or atr_baseline <= 0:
        return CadenceDecision(
            interval_seconds=base_interval, regime="normal", median_atr=0.0, ratio=1.0
        )

    median_atr = float(median(atrs))
    ratio = median_atr / atr_baseline

    if ratio >= fast_threshold:
        regime = "fast"
        interval = base_interval // 2
    elif ratio <= slow_threshold:
        regime = "slow"
        interval = base_interval * 2
    else:
        regime = "normal"
        interval = base_interval

    interval = max(min_interval, min(max_interval, interval))
    return CadenceDecision(
        interval_seconds=interval, regime=regime, median_atr=median_atr, ratio=ratio
    )
