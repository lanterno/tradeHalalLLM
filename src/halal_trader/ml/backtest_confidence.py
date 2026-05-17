"""Confidence-bounded backtest reporting — Round-5 Wave 14.G.

A typical backtest reports a single Sharpe ratio. That hides the
sample-size uncertainty: a 30-trade backtest's Sharpe has a much
wider confidence interval than a 3000-trade one. This module
adds **bootstrap confidence intervals** around the headline metrics
so the reader sees both the point estimate + the uncertainty band.

Pinned semantics:

- **Closed-set Metric ladder** (SHARPE / SORTINO / WIN_RATE /
  PROFIT_FACTOR / MAX_DRAWDOWN).
- **Bootstrap is non-parametric** — resamples returns with
  replacement.
- **Default 1000 bootstrap replicates** at 95% CI.
- **Deterministic when seeded** — replay-friendly.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class Metric(str, Enum):
    """Closed-set backtest metrics."""

    SHARPE = "sharpe"
    SORTINO = "sortino"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"


@dataclass(frozen=True)
class BootstrapPolicy:
    """Operator-tunable bootstrap policy."""

    n_replicates: int = 1000
    confidence_level: float = 0.95
    annualisation_factor: int = 252

    def __post_init__(self) -> None:
        if self.n_replicates < 100:
            raise ValueError("n_replicates must be >= 100")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.annualisation_factor <= 0:
            raise ValueError("annualisation_factor must be positive")


@dataclass(frozen=True)
class MetricEstimate:
    """Point estimate + bootstrap CI for a metric."""

    metric: Metric
    point_estimate: float
    lower_ci: float
    upper_ci: float
    n_samples: int

    def __post_init__(self) -> None:
        if self.lower_ci > self.upper_ci:
            raise ValueError("lower_ci must be <= upper_ci")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")


def sharpe_ratio(returns: Sequence[float], *, annualisation: int = 252) -> float:
    """Annualised Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    mean = statistics.mean(returns)
    sd = statistics.stdev(returns)
    if sd == 0:
        return 0.0
    return (mean / sd) * math.sqrt(annualisation)


def sortino_ratio(returns: Sequence[float], *, annualisation: int = 252) -> float:
    """Annualised Sortino ratio."""
    if len(returns) < 2:
        return 0.0
    mean = statistics.mean(returns)
    downside = [r for r in returns if r < 0]
    if not downside:
        return float("inf") if mean > 0 else 0.0
    downside_dev = math.sqrt(sum(r**2 for r in downside) / len(returns))
    if downside_dev == 0:
        return 0.0
    return (mean / downside_dev) * math.sqrt(annualisation)


def win_rate(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    wins = sum(1 for r in returns if r > 0)
    return wins / len(returns)


def profit_factor(returns: Sequence[float]) -> float:
    gains = sum(r for r in returns if r > 0)
    losses = -sum(r for r in returns if r < 0)
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def max_drawdown(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
    return max_dd


_METRIC_FUNCS = {
    Metric.SHARPE: lambda r, ann: sharpe_ratio(r, annualisation=ann),
    Metric.SORTINO: lambda r, ann: sortino_ratio(r, annualisation=ann),
    Metric.WIN_RATE: lambda r, ann: win_rate(r),
    Metric.PROFIT_FACTOR: lambda r, ann: profit_factor(r),
    Metric.MAX_DRAWDOWN: lambda r, ann: max_drawdown(r),
}


def _bootstrap_metric(
    returns: Sequence[float],
    metric: Metric,
    *,
    policy: BootstrapPolicy,
    seed: int | None,
) -> MetricEstimate:
    """Compute the metric + bootstrap CI."""
    func = _METRIC_FUNCS[metric]
    point = func(returns, policy.annualisation_factor)

    rng = random.Random(seed)
    n = len(returns)
    replicates: list[float] = []
    for _ in range(policy.n_replicates):
        sample = [returns[rng.randrange(n)] for _ in range(n)]
        val = func(sample, policy.annualisation_factor)
        if math.isfinite(val):
            replicates.append(val)

    if not replicates:
        return MetricEstimate(
            metric=metric,
            point_estimate=point,
            lower_ci=point,
            upper_ci=point,
            n_samples=n,
        )

    replicates.sort()
    alpha = 1.0 - policy.confidence_level
    lower_idx = int(alpha / 2 * len(replicates))
    upper_idx = int((1.0 - alpha / 2) * len(replicates)) - 1
    lower = replicates[max(0, lower_idx)]
    upper = replicates[min(len(replicates) - 1, upper_idx)]
    return MetricEstimate(
        metric=metric,
        point_estimate=point,
        lower_ci=lower,
        upper_ci=upper,
        n_samples=n,
    )


def report_with_ci(
    returns: Sequence[float],
    *,
    metrics: Sequence[Metric] = (
        Metric.SHARPE,
        Metric.SORTINO,
        Metric.WIN_RATE,
        Metric.PROFIT_FACTOR,
        Metric.MAX_DRAWDOWN,
    ),
    policy: BootstrapPolicy | None = None,
    seed: int | None = None,
) -> tuple[MetricEstimate, ...]:
    """Compute all metrics + CIs."""
    if not returns:
        raise ValueError("returns must be non-empty")
    pol = policy if policy is not None else BootstrapPolicy()
    return tuple(
        _bootstrap_metric(returns, m, policy=pol, seed=seed) for m in metrics
    )


def render_report(estimates: Sequence[MetricEstimate]) -> str:
    if not estimates:
        return "Backtest report: no metrics"
    head = f"Backtest report: n={estimates[0].n_samples}"
    lines = [head]
    for e in estimates:
        lines.append(
            f"  {e.metric.value:14s} = {e.point_estimate:+.4f} "
            f"[CI {e.lower_ci:+.4f}, {e.upper_ci:+.4f}]"
        )
    return "\n".join(lines)
