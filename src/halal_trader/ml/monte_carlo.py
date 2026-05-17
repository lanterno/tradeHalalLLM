"""Monte Carlo portfolio simulation — Round-5 Wave 14.B.

Pure-Python (numpy) Monte Carlo simulator for forward-looking portfolio
risk + return. Given current weights, an expected-return vector, a
covariance matrix, and a horizon, it samples N portfolio paths and
returns terminal-value distribution + path quantiles.

Pinned semantics:

- **Multi-variate normal sampling.** No fat-tail / copula model — the
  Round-4 ``ml/risk_aggregator.py`` covers the practical pre-trade
  gate; this module is the *exploratory* MC for "what's the
  distribution of my P&L over 252 trading days?".
- **Deterministic with seed.** Two runs with the same seed produce
  identical paths — replay-friendly.
- **Annualised return + covariance** — the math is in 1-day steps,
  so caller passes the daily covariance and expected daily return.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # type: ignore[import-untyped]


def _np():
    """Lazy numpy import; reused throughout."""
    import numpy as np  # type: ignore[import-untyped]

    return np


@dataclass(frozen=True)
class SimulationInputs:
    """Inputs for a Monte Carlo portfolio simulation."""

    weights: tuple[float, ...]
    expected_daily_returns: tuple[float, ...]
    daily_covariance: tuple[tuple[float, ...], ...]
    initial_value: float
    horizon_days: int

    def __post_init__(self) -> None:
        n = len(self.weights)
        if n == 0:
            raise ValueError("weights must be non-empty")
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("weights must sum to 1.0")
        if any(w < 0 for w in self.weights):
            raise ValueError("weights must be non-negative (long-only)")
        if len(self.expected_daily_returns) != n:
            raise ValueError("expected_daily_returns length mismatch")
        if len(self.daily_covariance) != n:
            raise ValueError("daily_covariance must be n x n")
        for row in self.daily_covariance:
            if len(row) != n:
                raise ValueError("daily_covariance must be square")
        if self.initial_value <= 0:
            raise ValueError("initial_value must be positive")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")


@dataclass(frozen=True)
class SimulationResult:
    """Result of a Monte Carlo run."""

    n_paths: int
    horizon_days: int
    initial_value: float
    terminal_values: tuple[float, ...]
    var_95: float  # 5th percentile loss (positive number = loss)
    cvar_95: float  # mean loss in worst 5% tail
    p50_terminal: float
    p10_terminal: float
    p90_terminal: float
    mean_terminal: float

    def __post_init__(self) -> None:
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")


def simulate(
    inputs: SimulationInputs,
    *,
    n_paths: int = 1000,
    seed: int | None = None,
) -> SimulationResult:
    """Run Monte Carlo simulation. Returns terminal-value distribution + risk metrics."""
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    np = _np()
    rng = np.random.default_rng(seed)

    weights = np.array(inputs.weights, dtype=float)
    means = np.array(inputs.expected_daily_returns, dtype=float)
    cov = np.array(inputs.daily_covariance, dtype=float)

    # Sample daily-return vectors: shape (n_paths, horizon_days, n_assets)
    samples = rng.multivariate_normal(
        mean=means, cov=cov, size=(n_paths, inputs.horizon_days)
    )

    # Project portfolio daily returns: shape (n_paths, horizon_days)
    portfolio_daily = samples @ weights

    # Compound to get terminal multiplier: shape (n_paths,)
    log_returns = np.log1p(portfolio_daily)
    cumulative_log = log_returns.sum(axis=1)
    terminal_multiplier = np.exp(cumulative_log)
    terminal_values = inputs.initial_value * terminal_multiplier

    # Risk metrics
    sorted_terminals = np.sort(terminal_values)
    p5_idx = max(0, int(0.05 * n_paths) - 1)
    p5_value = sorted_terminals[p5_idx]
    var_95 = max(0.0, inputs.initial_value - float(p5_value))
    tail = sorted_terminals[: p5_idx + 1]
    cvar_95 = max(0.0, inputs.initial_value - float(tail.mean()))

    p10 = float(np.percentile(terminal_values, 10))
    p50 = float(np.percentile(terminal_values, 50))
    p90 = float(np.percentile(terminal_values, 90))

    return SimulationResult(
        n_paths=n_paths,
        horizon_days=inputs.horizon_days,
        initial_value=inputs.initial_value,
        terminal_values=tuple(float(v) for v in terminal_values),
        var_95=var_95,
        cvar_95=cvar_95,
        p50_terminal=p50,
        p10_terminal=p10,
        p90_terminal=p90,
        mean_terminal=float(terminal_values.mean()),
    )


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_result(result: SimulationResult) -> str:
    head = (
        f"Monte Carlo: {result.n_paths} paths × {result.horizon_days} days "
        f"from ${result.initial_value:.2f}"
    )
    lines = [
        head,
        f"  • mean terminal: ${result.mean_terminal:.2f}",
        f"  • p10 / p50 / p90: ${result.p10_terminal:.2f} / "
        f"${result.p50_terminal:.2f} / ${result.p90_terminal:.2f}",
        f"  • VaR 95%: ${result.var_95:.2f}",
        f"  • CVaR 95%: ${result.cvar_95:.2f}",
    ]
    return _scrub("\n".join(lines))
