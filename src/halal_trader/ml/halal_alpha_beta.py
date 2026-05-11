"""Halal alpha/beta separation — Round-5 Wave 7.F.

Conventional alpha/beta separation isolates a portfolio's market
return (β times benchmark) from its idiosyncratic skill (α). Most
implementations use leverage:

    portfolio = α + β × benchmark
    long α    = long benchmark + α
    *but* this requires leveraging α with margin → riba.

The halal analogue substitutes the implicit leverage with an
**explicit Wa'd construct**: the operator commits cash equal to the
benchmark exposure, and a counterparty issues a Wa'd to deliver the
benchmark return (synthetic exposure) at maturity. The operator's
own portfolio runs the alpha overlay against the cash sleeve.

This module is the **separation accounting + planner**:

1. Decompose a portfolio's historical returns into α + β × R_b.
2. Plan a Wa'd-based replication strategy for the β leg.
3. Account for the α + β legs separately so the operator can attribute
   performance honestly (and audit Sharia compliance per leg).

Pinned semantics:

- **Decomposition uses OLS regression** on log-returns. Pure-Python
  closed-form (single-factor model — no matrix inversion needed).
  R² is computed for diagnostic.
- **Beta clipped to [-2, 3].** Outside this band the regression is
  unreliable; raise an `UnreliableBetaError` rather than silently
  returning bad numbers.
- **Wa'd-replication notional = β × portfolio_value.** Pin: never
  uses fractional/decimal share construction (the Wa'd is on a
  benchmark *index value*, not units).
- **No interest accrual on the α sleeve.** The α sleeve sits in cash
  + halal money-market substitute (Murabaha-T-bill); the planner
  records the sleeve's cash drag explicitly.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — only β / α / R² / amounts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date


class UnreliableBetaError(ValueError):
    """β estimate fell outside the [-2, 3] reliability band."""


@dataclass(frozen=True)
class AlphaBetaDecomposition:
    """Output of `decompose`."""

    alpha_per_period: float
    """Excess return per period — the constant in OLS y = α + β × x."""
    beta: float
    r_squared: float
    n_periods: int
    residual_std: float
    """σ of ε in y = α + β × x + ε."""

    def annualised_alpha(self, periods_per_year: int = 252) -> float:
        return self.alpha_per_period * periods_per_year


def decompose(
    portfolio_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    *,
    beta_band: tuple[float, float] = (-2.0, 3.0),
) -> AlphaBetaDecomposition:
    """Decompose portfolio_returns into α + β × benchmark_returns + ε.

    Inputs are aligned per-period log-returns. Pure-Python OLS
    (closed form for a single regressor + intercept).

    Raises `UnreliableBetaError` if β falls outside `beta_band`.
    """
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("portfolio_returns and benchmark_returns length mismatch")
    if len(portfolio_returns) < 5:
        raise ValueError("at least 5 return periods required for OLS")
    n = len(portfolio_returns)
    px = list(benchmark_returns)
    py = list(portfolio_returns)
    mean_x = sum(px) / n
    mean_y = sum(py) / n
    num = sum((px[i] - mean_x) * (py[i] - mean_y) for i in range(n))
    den = sum((px[i] - mean_x) ** 2 for i in range(n))
    if den < 1e-18:
        raise ValueError("benchmark variance is zero — cannot regress")
    beta = num / den
    if beta < beta_band[0] or beta > beta_band[1]:
        raise UnreliableBetaError(f"beta {beta:.3f} outside reliability band {beta_band}")
    alpha = mean_y - beta * mean_x
    # R² = 1 - SSres / SStot.
    ss_res = sum((py[i] - alpha - beta * px[i]) ** 2 for i in range(n))
    ss_tot = sum((py[i] - mean_y) ** 2 for i in range(n))
    if ss_tot < 1e-18:
        r_squared = 1.0  # constant portfolio return; trivial fit.
    else:
        r_squared = max(0.0, 1.0 - ss_res / ss_tot)
    residual_std = math.sqrt(ss_res / max(1, n - 2))
    return AlphaBetaDecomposition(
        alpha_per_period=alpha,
        beta=beta,
        r_squared=r_squared,
        n_periods=n,
        residual_std=residual_std,
    )


@dataclass(frozen=True)
class HalalReplicationPlan:
    """Plan to replicate a β exposure halally via a Wa'd construct."""

    portfolio_value: float
    target_beta: float
    benchmark_symbol: str
    waad_promisor_id: str
    waad_notional: float
    cash_sleeve_value: float
    """Cash held against the α sleeve. Equals portfolio_value −
    abs(waad_notional) when β ≥ 0; equals portfolio_value when β < 0
    (a synthetic short via reverse-Wa'd)."""
    issue_date: date
    expiry: date
    expected_alpha_drag_per_period: float = 0.0
    """Implicit cash-drag on the α sleeve from holding cash vs the
    benchmark. Operator-provided so the planner can include it in
    P&L attribution."""

    def __post_init__(self) -> None:
        if self.portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive")
        if not -2.0 <= self.target_beta <= 3.0:
            raise ValueError("target_beta must be in [-2, 3]")
        if not self.benchmark_symbol or not self.benchmark_symbol.strip():
            raise ValueError("benchmark_symbol must be non-empty")
        if not self.waad_promisor_id or not self.waad_promisor_id.strip():
            raise ValueError("waad_promisor_id must be non-empty")
        if self.cash_sleeve_value < 0:
            raise ValueError("cash_sleeve_value must be non-negative")
        if self.expiry <= self.issue_date:
            raise ValueError("expiry must be after issue_date")


def plan_replication(
    *,
    portfolio_value: float,
    target_beta: float,
    benchmark_symbol: str,
    waad_promisor_id: str,
    issue_date: date,
    expiry: date,
    expected_alpha_drag_per_period: float = 0.0,
) -> HalalReplicationPlan:
    """Build a Wa'd-replication plan for the β exposure.

    Long β (β > 0): Wa'd promises benchmark return on a notional of
    β × portfolio_value; cash sleeve is `portfolio − abs(notional)`
    (could be negative for β > 1, in which case the operator must add
    cash from elsewhere — the plan surfaces this explicitly via the
    `cash_sleeve_value` field).

    For 0 < β ≤ 1, cash sleeve absorbs (1 - β) of the portfolio.
    For β > 1, cash sleeve = 0 and the Wa'd is for β × portfolio
    (exceeds portfolio value — operator must verify halal funding).
    For β < 0, the Wa'd is a *short-benchmark* commitment via
    reverse-Wa'd; cash sleeve holds the full portfolio value.
    """
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")
    waad_notional = abs(target_beta) * portfolio_value
    if target_beta >= 0:
        cash_sleeve = max(0.0, portfolio_value - waad_notional)
    else:
        cash_sleeve = portfolio_value
    return HalalReplicationPlan(
        portfolio_value=portfolio_value,
        target_beta=target_beta,
        benchmark_symbol=benchmark_symbol,
        waad_promisor_id=waad_promisor_id,
        waad_notional=waad_notional,
        cash_sleeve_value=cash_sleeve,
        issue_date=issue_date,
        expiry=expiry,
        expected_alpha_drag_per_period=expected_alpha_drag_per_period,
    )


@dataclass(frozen=True)
class AttributionReport:
    """Output of `attribute` — period-level α + β + cash decomposition."""

    period_return: float
    benchmark_return: float
    beta_pnl: float
    """β × benchmark_return × portfolio_value."""
    alpha_pnl: float
    """The unexplained residual."""
    cash_drag: float
    """Cash sleeve carry penalty (negative = drag)."""
    total_attributed: float
    """beta_pnl + alpha_pnl + cash_drag."""


def attribute(
    plan: HalalReplicationPlan,
    *,
    realised_portfolio_return: float,
    realised_benchmark_return: float,
) -> AttributionReport:
    """Decompose realised P&L into β-leg + α-leg + cash-drag."""
    if not -1.0 <= realised_portfolio_return <= 5.0:
        raise ValueError("realised_portfolio_return outside reasonable bounds")
    if not -1.0 <= realised_benchmark_return <= 5.0:
        raise ValueError("realised_benchmark_return outside reasonable bounds")
    period_pnl = realised_portfolio_return * plan.portfolio_value
    beta_pnl = plan.target_beta * realised_benchmark_return * plan.portfolio_value
    cash_drag = plan.expected_alpha_drag_per_period * plan.cash_sleeve_value
    alpha_pnl = period_pnl - beta_pnl - cash_drag
    return AttributionReport(
        period_return=realised_portfolio_return,
        benchmark_return=realised_benchmark_return,
        beta_pnl=beta_pnl,
        alpha_pnl=alpha_pnl,
        cash_drag=cash_drag,
        total_attributed=beta_pnl + alpha_pnl + cash_drag,
    )


def render_decomposition(d: AlphaBetaDecomposition) -> str:
    """Operator-readable summary of an OLS decomposition."""
    return (
        f"📐 α/β decomposition: α={d.alpha_per_period * 1e4:+.2f} bps/period, "
        f"β={d.beta:+.3f}, R²={d.r_squared * 100:.2f}%, "
        f"σ_residual={d.residual_std * 1e4:.2f} bps, n={d.n_periods}"
    )


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


def render_plan(plan: HalalReplicationPlan) -> str:
    return (
        f"⚖️ Halal α/β plan: portfolio={plan.portfolio_value:.2f}, "
        f"β={plan.target_beta:+.3f} ({plan.benchmark_symbol})\n"
        f"  • Wa'd notional: {plan.waad_notional:.2f} via {_mask(plan.waad_promisor_id)}\n"
        f"  • Cash sleeve: {plan.cash_sleeve_value:.2f}\n"
        f"  • Expiry: {plan.expiry.isoformat()}"
    )


def render_attribution(rep: AttributionReport) -> str:
    return (
        f"📊 Attribution: portfolio {rep.period_return * 100:+.2f}% "
        f"(benchmark {rep.benchmark_return * 100:+.2f}%)\n"
        f"  • β leg: {rep.beta_pnl:+.2f}\n"
        f"  • α leg: {rep.alpha_pnl:+.2f}\n"
        f"  • cash drag: {rep.cash_drag:+.2f}\n"
        f"  • Total: {rep.total_attributed:+.2f}"
    )
