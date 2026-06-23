"""Tail-risk metrics: Value-at-Risk and Conditional VaR (Expected Shortfall).

Variance and Sharpe describe the middle of the distribution; they say little
about how bad the bad days get. For a fat-tailed book (crypto especially) the
tail is what blows up an account. VaR is the loss you only exceed ``alpha`` of
the time; CVaR (Expected Shortfall) is the *average* loss in that worst-``alpha``
tail — a coherent risk measure and a better budget than variance.

Pure functions (numpy only). Returns are signed fractions; VaR/CVaR come back
**negative** for a losing tail (e.g. -0.04 = a 4% expected shortfall).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _clean(returns: Any) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return np.asarray(r[np.isfinite(r)], dtype=float)


def value_at_risk(returns: Any, alpha: float = 0.05) -> float:
    """The ``alpha``-quantile return (historical VaR).

    With ``alpha=0.05`` this is the return you only do worse than 5% of the
    time. Negative for a loss. 0.0 on empty input.
    """
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    return float(np.percentile(r, alpha * 100.0))


def conditional_value_at_risk(returns: Any, alpha: float = 0.05) -> float:
    """Expected Shortfall: the mean of the worst ``alpha`` tail of returns.

    The average outcome *given* you're in the bad tail — strictly more
    conservative than VaR (CVaR <= VaR). Negative for a loss; falls back to the
    VaR point when the tail is empty (tiny samples); 0.0 on empty input.
    """
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    var = np.percentile(r, alpha * 100.0)
    tail = r[r <= var]
    if tail.size == 0:
        return float(var)
    return float(tail.mean())
