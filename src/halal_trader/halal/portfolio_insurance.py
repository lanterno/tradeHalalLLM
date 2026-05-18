"""Wa'd-based portfolio insurance — Round-5 Wave 13.B.

Conventional portfolio insurance buys put options against the
portfolio's market value. The halal alternative composes Wa'd
PROMISE_TO_SELL contracts (the Wave 4.A primitive) at downside
strikes — a portfolio of synthetic puts. This module is the
**composer**: given a portfolio + a target downside floor + a hedge
budget, it constructs a hedge plan via Wa'ds.

Pinned semantics:

- **Closed-set HedgeMode ladder** (FULL_FLOOR / PARTIAL_FLOOR /
  TAIL_ONLY).
- **Coverage ratio** is the fraction of portfolio value protected;
  `tail_only` covers only the tail beyond a threshold.
- **All issued Wa'ds delegate to `halal/waad.py`** for compliance
  validation; this module just composes.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from halal_trader.halal.waad import (
    StructuringPolicy,
    StructuringResult,
    WaadDirection,
    WaadInputs,
    structure_waad,
    synthetic_put_payoff,
)


class HedgeMode(str, Enum):
    """Closed-set hedge composition modes."""

    FULL_FLOOR = "full_floor"
    PARTIAL_FLOOR = "partial_floor"
    TAIL_ONLY = "tail_only"


@dataclass(frozen=True)
class PortfolioPosition:
    """A single position to insure."""

    symbol: str
    quantity: float
    market_price: float

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.market_price <= 0:
            raise ValueError("market_price must be positive")

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price


@dataclass(frozen=True)
class HedgePolicy:
    """Operator-tunable hedge policy."""

    mode: HedgeMode = HedgeMode.PARTIAL_FLOOR
    floor_pct: float = 0.90  # protect down to 90% of current value
    tail_threshold_pct: float = 0.80  # tail-only kicks in below 80%
    coverage_ratio: float = 1.0  # fraction of position covered
    hedge_term_days: int = 90

    def __post_init__(self) -> None:
        if not 0.0 < self.floor_pct < 1.0:
            raise ValueError("floor_pct must be in (0, 1)")
        if not 0.0 < self.tail_threshold_pct < 1.0:
            raise ValueError("tail_threshold_pct must be in (0, 1)")
        if not 0.0 < self.coverage_ratio <= 1.0:
            raise ValueError("coverage_ratio must be in (0, 1]")
        if self.hedge_term_days <= 0:
            raise ValueError("hedge_term_days must be positive")
        if self.tail_threshold_pct >= self.floor_pct:
            raise ValueError("tail_threshold_pct must be < floor_pct")


@dataclass(frozen=True)
class HedgePlan:
    """The hedge plan emitted by the composer."""

    waads: tuple[WaadInputs, ...]
    structuring_results: tuple[StructuringResult, ...]
    portfolio_value: float
    floor_value: float
    expected_payoff_at_floor: float

    def all_valid(self) -> bool:
        return all(r.is_valid for r in self.structuring_results)


def compose_hedge(
    positions: Iterable[PortfolioPosition],
    *,
    promisor: str,
    counterparty: str,
    today: date,
    policy: HedgePolicy | None = None,
) -> HedgePlan:
    """Compose a Wa'd-based hedge plan for the portfolio."""
    pol = policy if policy is not None else HedgePolicy()
    positions_t = tuple(positions)
    if not positions_t:
        return HedgePlan(
            waads=(),
            structuring_results=(),
            portfolio_value=0.0,
            floor_value=0.0,
            expected_payoff_at_floor=0.0,
        )

    portfolio_value = sum(p.market_value for p in positions_t)
    floor_value = portfolio_value * pol.floor_pct

    if pol.mode is HedgeMode.TAIL_ONLY:
        # Strike at the tail threshold rather than the floor.
        strike_pct = pol.tail_threshold_pct
    else:
        strike_pct = pol.floor_pct

    waad_policy = StructuringPolicy(max_term_days=max(pol.hedge_term_days * 2, 365))
    waads_list: list[WaadInputs] = []
    results: list[StructuringResult] = []
    expected_payoff = 0.0

    for i, pos in enumerate(positions_t):
        strike = pos.market_price * strike_pct
        hedged_qty = pos.quantity * pol.coverage_ratio
        if hedged_qty <= 0:
            continue
        waad = WaadInputs(
            waad_id=f"HEDGE-{i:03d}",
            direction=WaadDirection.PROMISE_TO_SELL,
            promisor=promisor,
            promisee=counterparty,
            underlying=pos.symbol,
            quantity=hedged_qty,
            strike_price=strike,
            market_price=pos.market_price,
            promise_date=today,
            exercise_date=today + timedelta(days=pol.hedge_term_days),
        )
        waads_list.append(waad)
        results.append(structure_waad(waad, policy=waad_policy))
        if pol.mode is HedgeMode.FULL_FLOOR:
            settlement = pos.market_price * pol.floor_pct
        else:
            settlement = pos.market_price * pol.tail_threshold_pct
        payoff = synthetic_put_payoff(waad, settlement_price=settlement).payoff
        expected_payoff += payoff

    return HedgePlan(
        waads=tuple(waads_list),
        structuring_results=tuple(results),
        portfolio_value=portfolio_value,
        floor_value=floor_value,
        expected_payoff_at_floor=expected_payoff,
    )


def render_plan(plan: HedgePlan) -> str:
    valid = plan.all_valid()
    emoji = "🛡️" if valid else "⚠️"
    lines = [
        f"{emoji} Hedge plan: {len(plan.waads)} Wa'd contracts, "
        f"portfolio=${plan.portfolio_value:.2f}, floor=${plan.floor_value:.2f}",
        f"  expected payoff at floor: ${plan.expected_payoff_at_floor:.2f}",
    ]
    for waad, result in zip(plan.waads, plan.structuring_results):
        marker = "✓" if result.is_valid else "✗"
        lines.append(
            f"  {marker} {waad.waad_id} promise-to-sell {waad.quantity:.2f} "
            f"{waad.underlying} strike={waad.strike_price:.2f}"
        )
    return "\n".join(lines)
