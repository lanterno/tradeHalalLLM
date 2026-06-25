"""Signal evaluation: Information Coefficient (IC) and ICIR.

The honest "does this signal actually predict?" test. The IC is the Spearman
rank correlation between a signal (a conviction, a factor score, an indicator)
and the forward return it's meant to forecast: +1 means the signal ranks
outcomes perfectly, 0 means no information, negative means it's inverted. ICIR
(mean IC / std IC across periods) is its risk-adjusted, decay-aware cousin —
the keep/kill number for a signal.

Pure (numpy only). Use it to prune dead signals before they cost real money,
and to check whether the model's own conviction carries any edge.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean of their rank span), like scipy."""
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0  # mean of ranks (i+1)..(j+1)
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt(float((xc**2).sum()) * float((yc**2).sum())))
    if denom == 0.0:
        return 0.0
    return float(float((xc * yc).sum()) / denom)


def information_coefficient(signal: Any, outcomes: Any) -> float:
    """Spearman rank correlation between ``signal`` and forward ``outcomes``.

    +1 the signal ranks outcomes perfectly, 0 no information, -1 inverted.
    Returns 0.0 for fewer than 3 aligned finite pairs or a constant input.
    """
    s = np.asarray(signal, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    if s.shape != o.shape or s.size < 3:
        return 0.0
    mask = np.isfinite(s) & np.isfinite(o)
    s, o = s[mask], o[mask]
    if s.size < 3:
        return 0.0
    return _pearson(_rankdata(s), _rankdata(o))


def icir(ic_values: Any) -> float:
    """Information Ratio of an IC series: mean(IC) / std(IC).

    Rewards a signal whose predictive power is *consistent* across periods, not
    one lucky big IC. Returns 0.0 for fewer than 2 values or zero dispersion.
    """
    ics = np.asarray(ic_values, dtype=float)
    ics = ics[np.isfinite(ics)]
    if ics.size < 2:
        return 0.0
    sd = float(ics.std(ddof=1))
    if sd < 1e-12:  # effectively constant (float noise, not real dispersion)
        return 0.0
    return float(ics.mean() / sd)
