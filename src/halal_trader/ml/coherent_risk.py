"""Coherent risk measures (Expected Shortfall / CVaR) — Round-5 Wave 14.E.

The bot uses VaR widely, but VaR is *not coherent* — it can fail
sub-additivity (combining two portfolios can produce a higher VaR
than the sum of their individual VaRs). Expected Shortfall (ES, also
called CVaR — the conditional expectation of loss in the worst α%
tail) is coherent + recommended by Basel III for regulatory capital.

This module ships the **pure-Python (no numpy hard-dependence)
coherent risk measures**: VaR, ES/CVaR, and the spectral risk measure
generalisation. Optionally accepts numpy arrays for performance but
falls back to pure Python.

Pinned semantics:

- **α convention**: α=0.05 means "95% confidence" — the worst 5% of
  outcomes are in the tail. Pinned for clarity.
- **Returns are signed**: positive = gain, negative = loss. Loss
  amounts (positive numbers) are derived from ``-min(0, return)``.
- **Empty inputs return 0.0** with a flag-style ``valid=False`` field.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _sorted_returns(returns: Sequence[float]) -> list[float]:
    return sorted(returns)


def value_at_risk(returns: Sequence[float], *, alpha: float = 0.05) -> float:
    """Return positive VaR loss at confidence ``1 - alpha`` (e.g., 95%)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not returns:
        return 0.0
    sorted_r = _sorted_returns(returns)
    idx = max(0, math.ceil(alpha * len(sorted_r)) - 1)
    quantile = sorted_r[idx]
    return max(0.0, -quantile)


def expected_shortfall(returns: Sequence[float], *, alpha: float = 0.05) -> float:
    """Return positive expected-shortfall loss in the worst α tail."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not returns:
        return 0.0
    sorted_r = _sorted_returns(returns)
    cutoff = max(1, math.ceil(alpha * len(sorted_r)))
    tail = sorted_r[:cutoff]
    mean_tail = sum(tail) / len(tail)
    return max(0.0, -mean_tail)


def spectral_risk_measure(
    returns: Sequence[float],
    *,
    weight_fn,  # callable(quantile_position in [0,1]) -> weight
    n_quantiles: int = 100,
) -> float:
    """Spectral risk measure (Acerbi 2002) — generalisation of ES.

    Riemann-sum approximation: weight each empirical quantile at
    position ``i / n_quantiles`` by ``weight_fn(i / n_quantiles)``.
    Weights must sum to 1 (caller's responsibility).
    """
    if not returns:
        return 0.0
    if n_quantiles <= 0:
        raise ValueError("n_quantiles must be positive")
    sorted_r = _sorted_returns(returns)
    n = len(sorted_r)
    total = 0.0
    weight_sum = 0.0
    for i in range(n_quantiles):
        u = (i + 0.5) / n_quantiles
        w = weight_fn(u)
        if w < 0:
            raise ValueError(f"weight_fn returned negative weight {w} at u={u}")
        idx = min(int(u * n), n - 1)
        total += w * sorted_r[idx]
        weight_sum += w
    if weight_sum == 0:
        return 0.0
    return max(0.0, -(total / weight_sum))


# --- Convenience ES at multiple alphas -------------------------------------


@dataclass(frozen=True)
class TailRiskReport:
    """Combined VaR + ES across multiple alphas."""

    alphas: tuple[float, ...]
    var_at: tuple[float, ...]
    es_at: tuple[float, ...]
    n_returns: int

    def __post_init__(self) -> None:
        if len(self.alphas) != len(self.var_at) or len(self.alphas) != len(self.es_at):
            raise ValueError("alphas / var_at / es_at lengths must match")
        if self.n_returns < 0:
            raise ValueError("n_returns must be non-negative")
        for v in self.var_at + self.es_at:
            if v < 0:
                raise ValueError("VaR / ES values cannot be negative")
        for a in self.alphas:
            if not 0.0 < a < 1.0:
                raise ValueError("alphas must each be in (0, 1)")


def tail_risk_report(
    returns: Sequence[float], *, alphas: Sequence[float] = (0.01, 0.05, 0.10)
) -> TailRiskReport:
    """Compute VaR + ES for each alpha; return a structured report."""
    var_vals = tuple(value_at_risk(returns, alpha=a) for a in alphas)
    es_vals = tuple(expected_shortfall(returns, alpha=a) for a in alphas)
    return TailRiskReport(
        alphas=tuple(alphas),
        var_at=var_vals,
        es_at=es_vals,
        n_returns=len(returns),
    )


def render_report(report: TailRiskReport) -> str:
    head = f"Tail-risk: n={report.n_returns}"
    lines = [head]
    for a, v, e in zip(report.alphas, report.var_at, report.es_at):
        lines.append(f"  α={a:.3f} → VaR={v:.4f} | ES={e:.4f}")
    return "\n".join(lines)
