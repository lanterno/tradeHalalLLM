"""Statistics for pre-registered event trials: pure, deterministic, no scipy.

Trades that enter on the same session share that day's market, so they are
not independent draws: a mean's standard error is **clustered by entry
session** (CR1, the sandwich for OLS on a constant with the G/(G-1) small-
sample factor) and its t is read on G-1 degrees of freedom, not N-1.
Holding windows that span several sessions also overlap across entry
dates, which the date clusters do not see; for those (MD3) the trial adds a
**calendar-time** series, one equal-weighted mean per session of the open
trades' daily legs, with a Newey-West (Bartlett, 4 lags) standard error.

Several cells tested at once are corrected by **Holm**'s step-down. Gates
that need an interval for a statistic with no closed-form error (decile
spreads, rank ICs) **resample whole clusters**, seeded so a rerun gives the
same interval. Equivalence ("the minute and daily harnesses agree") is a
**TOST**: the 1-2α clustered interval must lie within ±margin.

p-values come from ``halabot.analysis.significance.student_t_sf_two_sided``
(the regularized incomplete beta), so they agree with the rest of the repo.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np

from halabot.analysis.significance import student_t_sf_two_sided

SEED: Final = 20261010
NW_LAGS: Final = 4
BOOTSTRAP_DRAWS: Final = 2000


@dataclass(frozen=True, slots=True)
class ClusteredMean:
    """A mean with its CR1 cluster-robust standard error (one-sided H1: mean > 0)."""

    n: int
    clusters: int
    mean: float
    se: float
    t: float
    df: int  # clusters - 1
    p_one_sided: float  # P(T_df >= t)
    deff: float  # design effect: V / (s^2 / n)

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        """The two-sided ``level`` interval ``mean ± t_{(1+level)/2, df} · se``."""
        half = t_quantile(0.5 + level / 2.0, self.df) * self.se
        return self.mean - half, self.mean + half


@dataclass(frozen=True, slots=True)
class NWMean:
    """A time-series mean with its Newey-West standard error (one-sided H1: mean > 0)."""

    n: int
    mean: float
    se: float
    t: float
    df: int  # n - 1
    p_one_sided: float


@dataclass(frozen=True, slots=True)
class Leg:
    """One held session of one trade: its return that day, net and abnormal."""

    day: date
    ret: float
    story_id: str


def p_one_sided(t: float, df: float) -> float:
    """P(T_df >= t): half the two-sided p when t >= 0, one minus that half otherwise."""
    sf = student_t_sf_two_sided(t, df)
    return sf / 2.0 if t >= 0 else 1.0 - sf / 2.0


def t_quantile(p: float, df: float) -> float:
    """The ``p``-quantile of Student's t on ``df`` degrees of freedom.

    Bisection on the two-sided tail probability, so it is exactly the inverse
    of :func:`student_t_sf_two_sided` (to about 1e-10).
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"quantile level must be in (0, 1), got {p}")
    if df <= 0:
        raise ValueError(f"degrees of freedom must be positive, got {df}")
    if p == 0.5:
        return 0.0
    target = 2.0 * min(p, 1.0 - p)  # the two-sided tail mass beyond |t|
    lo, hi = 0.0, 1.0
    while student_t_sf_two_sided(hi, df) > target:
        lo, hi = hi, hi * 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_sf_two_sided(mid, df) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * hi:
            break
    t = 0.5 * (lo + hi)
    return t if p > 0.5 else -t


@dataclass(frozen=True, slots=True)
class _Cr1:
    n: int
    clusters: int
    mean: float
    variance: float  # V, the CR1 variance of the mean
    sample_variance: float  # s^2 (ddof=1)


def _cr1(values: Sequence[float], clusters: Sequence[Hashable]) -> _Cr1 | None:
    if len(values) != len(clusters):
        raise ValueError(f"{len(values)} values but {len(clusters)} cluster keys")
    n = len(values)
    if n == 0:
        return None
    xs = [float(v) for v in values]
    if not all(math.isfinite(x) for x in xs):
        raise ValueError("a value is not finite")
    mean = math.fsum(xs) / n
    residuals: dict[Hashable, list[float]] = defaultdict(list)
    for x, g in zip(xs, clusters, strict=True):
        residuals[g].append(x - mean)
    g_count = len(residuals)
    if g_count < 2:
        return None
    sums = [math.fsum(e) for e in residuals.values()]
    variance = g_count / (g_count - 1) * math.fsum(s * s for s in sums) / (n * n)
    sample_variance = math.fsum((x - mean) ** 2 for x in xs) / (n - 1)
    return _Cr1(n, g_count, mean, variance, sample_variance)


def clustered_mean(values: Sequence[float], clusters: Sequence[Hashable]) -> ClusteredMean | None:
    """Mean of ``values`` with its CR1 standard error, clustered by ``clusters``.

    ``r̄ = Σr/N``; ``E_g = Σ_{i∈g}(r_i − r̄)``; ``V = G/(G−1) · Σ_g E_g² / N²``;
    ``t = r̄/√V`` on ``G − 1`` degrees of freedom; ``DEFF = V/(s²/N)``.

    None when it cannot be computed: no values, fewer than two clusters, or a
    zero clustered variance (every cluster's residuals cancel).
    """
    fit = _cr1(values, clusters)
    if fit is None or fit.variance <= 0.0:
        return None
    se = math.sqrt(fit.variance)
    t = fit.mean / se
    df = fit.clusters - 1
    return ClusteredMean(
        n=fit.n,
        clusters=fit.clusters,
        mean=fit.mean,
        se=se,
        t=t,
        df=df,
        p_one_sided=p_one_sided(t, df),
        deff=fit.variance / (fit.sample_variance / fit.n),
    )


def newey_west_mean(series: Sequence[float], *, lags: int = NW_LAGS) -> NWMean | None:
    """Mean of a time series with a Newey-West (Bartlett) standard error.

    ``e_d = x_d − x̄``; ``γ_k = Σ_{d>k} e_d e_{d−k} / T``;
    ``V = [γ_0 + 2 Σ_{k=1..L} (1 − k/(L+1)) γ_k] / T``; ``t = x̄/√V`` on
    ``T − 1`` degrees of freedom.

    None for fewer than two observations or a zero variance.
    """
    if lags < 0:
        raise ValueError(f"lags must be >= 0, got {lags}")
    xs = [float(x) for x in series]
    n = len(xs)
    if n < 2:
        return None
    if not all(math.isfinite(x) for x in xs):
        raise ValueError("a value is not finite")
    mean = math.fsum(xs) / n
    e = [x - mean for x in xs]
    gamma = [math.fsum(e[d] * e[d - k] for d in range(k, n)) / n for k in range(lags + 1)]
    weighted = math.fsum((1.0 - k / (lags + 1)) * gamma[k] for k in range(1, lags + 1))
    variance = (gamma[0] + 2.0 * weighted) / n
    if variance <= 0.0:
        return None
    se = math.sqrt(variance)
    t = mean / se
    return NWMean(n=n, mean=mean, se=se, t=t, df=n - 1, p_one_sided=p_one_sided(t, n - 1))


def calendar_series(
    legs: Mapping[str, Sequence[Leg]], sessions: Sequence[date]
) -> list[tuple[date, float]]:
    """One point per session with at least one open trade: the equal-weighted
    mean of that session's legs, in session order.

    ``legs`` maps a story id to its trade's legs. A leg on a day that is not
    one of ``sessions``, or filed under another story's id, is an error, not
    something to drop.
    """
    known = set(sessions)
    by_day: dict[date, list[float]] = defaultdict(list)
    for story_id, story_legs in legs.items():
        for leg in story_legs:
            if leg.story_id != story_id:
                raise ValueError(f"a leg of {leg.story_id} is filed under {story_id}")
            if leg.day not in known:
                raise ValueError(f"{story_id} has a leg on {leg.day}, which is not a session")
            if not math.isfinite(leg.ret):
                raise ValueError(f"{story_id} has a non-finite leg on {leg.day}")
            by_day[leg.day].append(leg.ret)
    return [
        (day, math.fsum(rets) / len(rets)) for day in sorted(known) if (rets := by_day.get(day))
    ]


def holm(p: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, bool]:
    """Holm's step-down: which hypotheses are rejected at family-wise ``alpha``.

    Sort the p-values ascending (ties by name); reject the k-th (1-based)
    while ``p_(k) <= alpha/(m − k + 1)``; stop at the first that is not.
    """
    for name, value in p.items():
        if not 0.0 <= value <= 1.0:  # NaN fails too
            raise ValueError(f"p-value of {name} is {value}, not in [0, 1]")
    m = len(p)
    out = dict.fromkeys(p, False)
    for k, name in enumerate(sorted(p, key=lambda key: (p[key], key))):
        if p[name] > alpha / (m - k):
            break
        out[name] = True
    return out


def cluster_bootstrap_ci(
    stat: Callable[[Sequence[int]], float],
    clusters: Sequence[Hashable],
    *,
    b: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval of ``stat`` over ``b`` resamples of whole clusters.

    ``clusters[i]`` is observation i's cluster. Each draw takes G clusters
    with replacement (``numpy.random.default_rng(seed).integers(0, G,
    (b, G))``, clusters numbered in order of first appearance) and calls
    ``stat`` with the indices of every observation in them, a cluster drawn
    twice contributing its observations twice. Draws where ``stat`` is not
    finite (undefined on that resample) are left out.
    """
    if b < 1:
        raise ValueError(f"b must be >= 1, got {b}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    members: dict[Hashable, list[int]] = {}
    for i, g in enumerate(clusters):
        members.setdefault(g, []).append(i)
    groups = list(members.values())
    if len(groups) < 2:
        raise ValueError("a cluster bootstrap needs at least two clusters")
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(b, len(groups)))
    values = np.array(
        [stat([i for g in row for i in groups[g]]) for row in draws.tolist()], dtype=float
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("the statistic is undefined on every resample")
    lo, hi = np.quantile(finite, [(1.0 - level) / 2.0, (1.0 + level) / 2.0])
    return float(lo), float(hi)


def tost(
    values: Sequence[float],
    clusters: Sequence[Hashable],
    *,
    margin: float = 0.001,
    alpha: float = 0.05,
) -> bool:
    """Two one-sided tests: is the mean within ±``margin`` at level ``alpha``?

    Equivalent to the ``1 − 2α`` CR1 interval (``mean ± t_{1−α, G−1}·se``)
    lying inside ``[−margin, +margin]``; a bound exactly on the margin
    passes (each one-sided test rejects at ``p <= alpha``). Fewer than two
    clusters show nothing and fail; a zero clustered variance leaves the
    interval at the mean.
    """
    if margin <= 0.0:
        raise ValueError(f"margin must be positive, got {margin}")
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    fit = _cr1(values, clusters)
    if fit is None:
        return False
    half = t_quantile(1.0 - alpha, fit.clusters - 1) * math.sqrt(max(fit.variance, 0.0))
    return fit.mean - half >= -margin and fit.mean + half <= margin
