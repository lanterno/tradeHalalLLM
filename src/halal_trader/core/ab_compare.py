"""Strategy A/B comparison statistics.

Round-4 wave 5.B: backend half of the dashboard's "compare two
strategies side by side" feature. Given two cohorts of closed trades
(typically two prompt-versions, two strategies, or live vs shadow),
compute the headline performance metrics — Sharpe, win rate, max
drawdown, total return — plus a Welch's t-test on the per-trade
return distributions so the operator knows whether the observed
difference is statistically meaningful or just noise.

Why we own this in-house instead of pulling in `pyfolio` /
`empyrical`:

* The whole comparator is ~120 lines of NumPy. Pulling in a heavy
  vendored stack adds dozens of transitive deps and a slow import.
* We need it to live in `core/` (no `[ml]` extra needed). Importing
  scipy is best-effort: if installed, we get a precise two-tailed
  p-value; if not, we degrade to a normal-approximation threshold.
* It's strategy-agnostic — operates on plain return arrays, doesn't
  know about Trade / CryptoTrade rows. Keeps the math testable in
  isolation; the SQL layer can grow/change without touching the math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CohortStats:
    """Headline performance numbers for a single cohort of trades.

    All ratios / percentages are returned as fractions (0.05 = 5%);
    the dashboard formatter is responsible for rendering as %.
    """

    n_trades: int
    win_rate: float  # fraction of trades with return > 0
    mean_return: float  # arithmetic mean per-trade return
    median_return: float
    std_return: float
    sharpe: float  # mean / std (no annualisation; per-trade)
    max_drawdown: float  # most negative trough on compound equity curve
    total_return: float  # compound (1+r1)(1+r2)…  − 1
    profit_factor: float  # sum(wins) / abs(sum(losses)); inf if no losses


@dataclass(frozen=True)
class ABComparison:
    """Result of comparing cohort A vs cohort B.

    ``t_statistic`` is Welch's t (positive ⇒ A > B in mean return).
    ``p_value`` is the two-tailed probability under the null
    "difference is zero". ``significant_at_05`` is the convenience
    boolean (operator-readable).

    ``mean_diff`` = ``a.mean_return - b.mean_return`` is exposed so
    the dashboard can show the *direction* of the win without
    re-deriving it.
    """

    a: CohortStats
    b: CohortStats
    mean_diff: float
    t_statistic: float
    degrees_of_freedom: float
    p_value: float | None  # None when scipy not installed AND df small
    significant_at_05: bool


def _safe_array(returns: Sequence[float] | np.ndarray) -> np.ndarray:
    """Drop NaN / inf / None — closed-trade rows occasionally have
    a missing ``return_pct`` (early-life bugs, manual exits) and
    one bad row shouldn't poison the whole stat."""
    arr = np.asarray(list(returns), dtype=float)
    return arr[np.isfinite(arr)]


def _max_drawdown(returns: np.ndarray) -> float:
    """Most negative trough on the compound equity curve.

    Returns a non-positive number (0.0 for an empty / all-positive
    track record). Same convention pyfolio uses, so the dashboard
    formatter doesn't have to flip signs.
    """
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def _profit_factor(returns: np.ndarray) -> float:
    wins = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / abs(losses))


def cohort_stats(returns: Sequence[float] | np.ndarray) -> CohortStats:
    """Compute headline metrics for one cohort.

    Empty / all-NaN cohort returns a zero-everywhere stats block —
    callers don't have to guard, and the dashboard renders a clean
    "no data" tile.
    """
    arr = _safe_array(returns)
    if arr.size == 0:
        return CohortStats(
            n_trades=0,
            win_rate=0.0,
            mean_return=0.0,
            median_return=0.0,
            std_return=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            total_return=0.0,
            profit_factor=0.0,
        )
    mean = float(arr.mean())
    # ddof=1 sample std — we're estimating population std from a
    # finite cohort. Matches pandas / scipy default.
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sharpe = mean / std if std > 0 else 0.0
    return CohortStats(
        n_trades=int(arr.size),
        win_rate=float((arr > 0).mean()),
        mean_return=mean,
        median_return=float(np.median(arr)),
        std_return=std,
        sharpe=float(sharpe),
        max_drawdown=_max_drawdown(arr),
        total_return=float(np.prod(1.0 + arr) - 1.0),
        profit_factor=_profit_factor(arr),
    )


def _welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-statistic and degrees of freedom for unequal-variance
    two-sample comparison. Returns ``(0.0, 0.0)`` for degenerate
    inputs (one cohort empty or both with zero variance) — caller
    handles by reporting ``p_value = None``.
    """
    if a.size < 2 or b.size < 2:
        return 0.0, 0.0
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    if var_a == 0 and var_b == 0:
        return 0.0, 0.0
    se = math.sqrt(var_a / a.size + var_b / b.size)
    if se == 0:
        return 0.0, 0.0
    t = (a.mean() - b.mean()) / se
    # Welch–Satterthwaite degrees of freedom.
    num = (var_a / a.size + var_b / b.size) ** 2
    denom = (var_a**2) / ((a.size**2) * (a.size - 1)) + (var_b**2) / ((b.size**2) * (b.size - 1))
    df = num / denom if denom > 0 else 0.0
    return float(t), float(df)


def _two_tailed_p(t: float, df: float) -> float | None:
    """Two-tailed p-value for Welch's t.

    Prefers ``scipy.stats.t.sf`` when scipy is installed (precise);
    falls back to the standard-normal approximation when df ≥ 30
    (the textbook "use z when df is large" rule of thumb); returns
    None for small-df cases without scipy so the caller knows the
    answer is unknown rather than wrong.
    """
    if df <= 0:
        return None
    try:
        from scipy.stats import t as scipy_t

        return float(2.0 * scipy_t.sf(abs(t), df))
    except Exception:
        if df < 30:
            return None
        # Standard normal SF for |t|, doubled for two-tailed.
        return float(2.0 * (1.0 - _normal_cdf(abs(t))))


def _normal_cdf(x: float) -> float:
    """Φ(x) using the math.erf identity. Saves us a scipy import in
    the fallback path; accuracy is ample for p-value reporting."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compare(
    a_returns: Sequence[float] | np.ndarray,
    b_returns: Sequence[float] | np.ndarray,
) -> ABComparison:
    """Compare two cohorts of per-trade returns.

    Returns an :class:`ABComparison` carrying both per-cohort headline
    stats and a Welch's t-test on the difference of means.
    """
    a_arr = _safe_array(a_returns)
    b_arr = _safe_array(b_returns)
    a_stats = cohort_stats(a_arr)
    b_stats = cohort_stats(b_arr)
    t, df = _welch_t(a_arr, b_arr)
    p = _two_tailed_p(t, df)
    significant = p is not None and p < 0.05
    return ABComparison(
        a=a_stats,
        b=b_stats,
        mean_diff=a_stats.mean_return - b_stats.mean_return,
        t_statistic=t,
        degrees_of_freedom=df,
        p_value=p,
        significant_at_05=significant,
    )
