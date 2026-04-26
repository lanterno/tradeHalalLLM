"""Historical Value-at-Risk and Expected Shortfall.

Backward-looking heat (current unrealized loss) is what the existing
risk engine guards. The forward-looking question — *if today's open
positions had had yesterday's worst returns, how much would I have
lost?* — needs VaR/ES. We use the historical method (no Gaussian
assumption) because crypto returns are famously fat-tailed.

Both measures are returned as **positive** decimal fractions of equity:
``var=0.012`` means "1.2% of equity at 95% confidence." Callers can
multiply by current equity to get the dollar figure for an alert.

Halal note: long-only, no leverage. There's no "downside" caveat about
shorts — every position can lose at most its entry notional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VarResult:
    """Output of :func:`portfolio_var_es`. All fractions are positive."""

    var: float  # tail loss at the chosen confidence
    expected_shortfall: float  # mean loss conditional on exceeding VaR
    confidence: float  # e.g. 0.95
    sample_size: int


def historical_var(
    returns: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return ``(VaR, ES)`` from a return series.

    Returns are expected to be one-period changes (e.g. daily). The
    function returns the historical loss at the ``(1 - confidence)``
    percentile and the mean of all losses below that cutoff.

    Returns ``(0, 0)`` when there are too few samples for the chosen
    confidence to be meaningful — we don't want to fabricate a "5% VaR"
    out of three data points.
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0.5, 1.0); got {confidence}")
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    # Need at least enough samples that 1 - confidence has 5+ observations
    # below the cutoff — otherwise ES is just one point.
    min_samples = int(20 / (1 - confidence))
    if arr.size < min_samples:
        return 0.0, 0.0

    cutoff_q = (1 - confidence) * 100
    var_threshold = float(np.percentile(arr, cutoff_q))
    var_loss = max(0.0, -var_threshold)

    tail = arr[arr <= var_threshold]
    if tail.size == 0:
        return var_loss, var_loss
    es_loss = max(0.0, -float(np.mean(tail)))
    return var_loss, es_loss


def portfolio_var_es(
    weights: dict[str, float],
    returns_by_symbol: dict[str, Sequence[float]],
    confidence: float = 0.95,
) -> VarResult:
    """Aggregate per-symbol returns into a portfolio series, then VaR/ES.

    ``weights`` are dollar-weighted exposures expressed as fractions of
    equity (e.g. ``{"BTCUSDT": 0.10, "ETHUSDT": 0.05}`` means 10% in BTC,
    5% in ETH, 85% cash). Symbols absent from ``returns_by_symbol`` are
    skipped. Returns aligned to the shortest available series.
    """
    aligned: dict[str, np.ndarray] = {}
    for sym, w in weights.items():
        if w <= 0 or sym not in returns_by_symbol:
            continue
        r = np.asarray(returns_by_symbol[sym], dtype=float)
        r = r[np.isfinite(r)]
        if r.size == 0:
            continue
        aligned[sym] = r

    if not aligned:
        return VarResult(var=0.0, expected_shortfall=0.0, confidence=confidence, sample_size=0)

    min_len = min(arr.size for arr in aligned.values())
    portfolio = np.zeros(min_len, dtype=float)
    for sym, r in aligned.items():
        portfolio += weights[sym] * r[-min_len:]

    var, es = historical_var(portfolio.tolist(), confidence=confidence)
    return VarResult(var=var, expected_shortfall=es, confidence=confidence, sample_size=min_len)


def klines_to_returns(closes: Sequence[float]) -> list[float]:
    """Convenience: turn a close-price series into period-on-period returns."""
    arr = np.asarray(closes, dtype=float)
    if arr.size < 2:
        return []
    out: list[float] = (np.diff(arr) / arr[:-1]).tolist()
    return out
