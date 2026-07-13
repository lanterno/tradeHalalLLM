"""Touch-and-hold validation harness for level families, with placebo.

The shared honesty tool of the levels track (roadmap Phase 1): a level
family earns a prompt slot only if its walk-forward hold rate beats a
distance-matched placebo. Definitions (research-derived, adapted to daily
bars — an **approximation**: intraday bars are the honest mode; daily bars
cannot see intra-session reaction order, so outcomes use daily closes):

* **touch** — a bar's range enters the level zone ``level ± eps·ATR``.
* **reject** — after the touch, a daily close moves ``hold·ATR`` beyond the
  level on the approach side (support: close ≥ level + hold·ATR;
  resistance: close ≤ level − hold·ATR) before any close crosses the level.
* **break** — a daily close beyond ``level ∓ eps`` first.
* **undecided** — horizon ends first (excluded from the hold rate).

Only *reachable* levels are tested (within ``3·ATR`` of the day's close):
distant levels trivially "hold" because price never gets there — the
unconditional-hold-rate trap the research warns about. Placebo levels are
drawn per real level on the SAME side at a uniform-random distance in the
same reachability band (seeded — reproducible), so a family only scores
above placebo if the *location* of its levels carries information beyond
"some price nearby". Sampling steps by ``horizon`` so consecutive windows
don't overlap and double-count the same episode.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from halal_trader.quant.levels import atr_series
from halal_trader.quant.volatility import FloatArray

FamilyFn = Callable[[list[str], np.ndarray, np.ndarray, np.ndarray, float], Sequence[float]]
"""(dates, highs, lows, closes — the walk-forward PREFIX — plus current ATR)
-> level prices. The prefix ends at the decision bar; no future data."""

_REACH_ATR = 3.0
_MIN_DIST_ATR = 0.35  # closer than this = already in the zone, untestable


@dataclass(frozen=True, slots=True)
class TouchStats:
    """Pooled touch-and-hold outcomes for one family (or its placebo)."""

    label: str
    n_level_days: int
    touches: int
    rejects: int
    breaks: int
    undecided: int

    @property
    def decided(self) -> int:
        return self.rejects + self.breaks

    @property
    def hold_rate(self) -> float | None:
        return self.rejects / self.decided if self.decided else None


def evaluate_family(
    dates: list[str],
    highs: FloatArray,
    lows: FloatArray,
    closes: FloatArray,
    family: FamilyFn,
    *,
    label: str,
    eps_atr: float = 0.25,
    hold_atr: float = 1.0,
    horizon: int = 5,
    warmup: int = 40,
    placebo_seed: int | None = None,
) -> TouchStats:
    """Walk-forward touch-and-hold stats for one family on one symbol.

    At each sampled day ``t`` the family sees ONLY bars ``[0..t]``; its
    levels are then tested against bars ``[t+1 .. t+horizon]``. With
    ``placebo_seed`` set, each real level is replaced by a same-side,
    distance-band-matched random level — run both and compare; the family
    result means nothing without its placebo twin.
    """
    h = np.asarray(highs, dtype=np.float64)
    lo = np.asarray(lows, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    if not (len(dates) == h.size == lo.size == c.size):
        raise ValueError("dates/highs/lows/closes must be equal length")
    if horizon < 1 or warmup < 1:
        raise ValueError("horizon and warmup must be >= 1")
    atr = atr_series(h, lo, c)
    rng = np.random.default_rng(placebo_seed) if placebo_seed is not None else None

    n_level_days = touches = rejects = breaks = undecided = 0
    for t in range(warmup, c.size - horizon, horizon):
        atr_t = float(atr[t])
        if atr_t <= 0:
            continue
        close_t = float(c[t])
        raw = family(dates[: t + 1], h[: t + 1], lo[: t + 1], c[: t + 1], atr_t)
        eps = eps_atr * atr_t
        for level in raw:
            d = float(level) - close_t
            if abs(d) < _MIN_DIST_ATR * atr_t or abs(d) > _REACH_ATR * atr_t:
                continue
            if rng is not None:
                dist = rng.uniform(_MIN_DIST_ATR, _REACH_ATR) * atr_t
                lvl = close_t + np.sign(d) * dist
            else:
                lvl = float(level)
            n_level_days += 1
            is_res = lvl > close_t
            touch_j: int | None = None
            for j in range(t + 1, t + 1 + horizon):
                if lo[j] <= lvl + eps and h[j] >= lvl - eps:
                    touch_j = j
                    break
            if touch_j is None:
                continue
            touches += 1
            outcome = "undecided"
            for j in range(touch_j, t + 1 + horizon):
                cj = float(c[j])
                if is_res:
                    if cj > lvl + eps:
                        outcome = "break"
                        break
                    if cj <= lvl - hold_atr * atr_t:
                        outcome = "reject"
                        break
                else:
                    if cj < lvl - eps:
                        outcome = "break"
                        break
                    if cj >= lvl + hold_atr * atr_t:
                        outcome = "reject"
                        break
            if outcome == "reject":
                rejects += 1
            elif outcome == "break":
                breaks += 1
            else:
                undecided += 1
    return TouchStats(
        label=label,
        n_level_days=n_level_days,
        touches=touches,
        rejects=rejects,
        breaks=breaks,
        undecided=undecided,
    )


def merge_stats(label: str, parts: Sequence[TouchStats]) -> TouchStats:
    """Pool per-symbol stats into one universe-level TouchStats."""
    return TouchStats(
        label=label,
        n_level_days=sum(p.n_level_days for p in parts),
        touches=sum(p.touches for p in parts),
        rejects=sum(p.rejects for p in parts),
        breaks=sum(p.breaks for p in parts),
        undecided=sum(p.undecided for p in parts),
    )


def placebo_uplift(real: TouchStats, placebo: TouchStats) -> float | None:
    """Hold-rate uplift of the real family over its placebo twin.

    Positive = the family's level locations carry information; ``None``
    when either side has no decided touches. This is the number the
    roadmap's validation gate 3 reads (alongside sample sufficiency).
    """
    if real.hold_rate is None or placebo.hold_rate is None:
        return None
    return real.hold_rate - placebo.hold_rate
