"""Sukuk pricing model — yield-curve-aware DCF — Round-5 Wave 3.B.

Sukuk price discovery follows the same DCF mechanic as bonds, with the
substitution that the cashflow being discounted is **profit
distribution** (Ijara rent / Mudarabah profit / Wakalah profit-share)
rather than coupon interest. Two wrinkles:

1. The reference rate is the **profit-rate yield curve** (Sukuk Bills
   / Islamic-bank profit rates) rather than the LIBOR / SOFR curve.
2. Sukuk that violate Standard 17 cl. 5.1.8 (pure-Murabaha / Salam)
   are not tradable on the secondary market — pricing them at face
   value is the only structurally-honest answer. The model surfaces
   this as a separate field rather than silently DCF-ing them.

This module is the pure-Python pricing primitive. No I/O, no DB —
the live profit-rate fetcher (IIFM / Bloomberg / GCC central-bank
feed) is a follow-up; this module exercises the math in isolation.

Pinned semantics:

- **Yield curve interpolation is linear in yield × log(time).**
  Standard market-conv. Operator-tunable via `Curve.interpolate`.
- **Cashflow timing in years from valuation date.** No business-day
  adjustment — the simple DCF's accuracy is set by the curve, not the
  day-count convention.
- **Tradable-secondary check** delegates to `aaoifi_standard_17`.
- **`yield_to_maturity` is bisection-based** — robust + monotone for
  reasonable price ranges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from halal_trader.halal.aaoifi_standard_17 import (
    SukukType,
    is_tradable_in_secondary,
)


@dataclass(frozen=True)
class CurvePoint:
    """A single (tenor in years, profit rate) point on the yield curve."""

    tenor_years: float
    rate: float

    def __post_init__(self) -> None:
        if self.tenor_years <= 0:
            raise ValueError("tenor_years must be positive")
        if not -0.10 < self.rate < 0.50:
            raise ValueError("rate outside reasonable bounds (-10%, 50%)")


@dataclass(frozen=True)
class ProfitRateCurve:
    """A profit-rate yield curve as a sorted sequence of (tenor, rate) points."""

    points: tuple[CurvePoint, ...]
    base_currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("curve must have at least one point")
        tenors = [p.tenor_years for p in self.points]
        if tenors != sorted(tenors):
            raise ValueError("curve points must be sorted by tenor")
        if len(set(tenors)) != len(tenors):
            raise ValueError("curve points must have unique tenors")
        if not self.base_currency or len(self.base_currency) > 8:
            raise ValueError("base_currency must be a non-empty short code")

    def interpolate(self, tenor_years: float) -> float:
        """Linear-in-tenor interpolation, flat extrapolation outside."""
        if tenor_years <= 0:
            raise ValueError("tenor_years must be positive")
        pts = self.points
        if tenor_years <= pts[0].tenor_years:
            return pts[0].rate
        if tenor_years >= pts[-1].tenor_years:
            return pts[-1].rate
        # Find segment
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if a.tenor_years <= tenor_years <= b.tenor_years:
                w = (tenor_years - a.tenor_years) / (b.tenor_years - a.tenor_years)
                return a.rate + w * (b.rate - a.rate)
        raise AssertionError("unreachable")  # pragma: no cover


@dataclass(frozen=True)
class Cashflow:
    """A scheduled sukuk cashflow."""

    amount: float
    time_years: float

    def __post_init__(self) -> None:
        if self.time_years <= 0:
            raise ValueError("time_years must be positive")


@dataclass(frozen=True)
class Sukuk:
    """A sukuk issuance for pricing purposes."""

    issuer: str
    sukuk_type: SukukType
    cashflows: tuple[Cashflow, ...]
    face_value: float

    def __post_init__(self) -> None:
        if not self.issuer or not self.issuer.strip():
            raise ValueError("issuer must be non-empty")
        if self.face_value <= 0:
            raise ValueError("face_value must be positive")
        if not self.cashflows:
            raise ValueError("sukuk must have at least one cashflow")
        times = [c.time_years for c in self.cashflows]
        if times != sorted(times):
            raise ValueError("cashflows must be sorted by time_years")


@dataclass(frozen=True)
class PricingResult:
    """Result of pricing a sukuk."""

    issuer: str
    present_value: float
    secondary_tradable: bool
    used_curve_rates: tuple[float, ...]
    accrued_profit: float

    def __post_init__(self) -> None:
        if self.present_value < 0:
            raise ValueError("present_value cannot be negative")


def price_sukuk(sukuk: Sukuk, curve: ProfitRateCurve) -> PricingResult:
    """Price a sukuk via DCF against the profit-rate curve.

    Pure-Murabaha / Salam sukuk are returned at face value with
    `secondary_tradable=False` — DCF is structurally inappropriate for
    debt instruments.
    """
    tradable = is_tradable_in_secondary(sukuk.sukuk_type)
    if not tradable:
        return PricingResult(
            issuer=sukuk.issuer,
            present_value=sukuk.face_value,
            secondary_tradable=False,
            used_curve_rates=(),
            accrued_profit=0.0,
        )

    pv = 0.0
    rates: list[float] = []
    for cf in sukuk.cashflows:
        rate = curve.interpolate(cf.time_years)
        rates.append(rate)
        # Continuous compounding for simplicity + stability across tenors.
        pv += cf.amount * math.exp(-rate * cf.time_years)

    return PricingResult(
        issuer=sukuk.issuer,
        present_value=pv,
        secondary_tradable=True,
        used_curve_rates=tuple(rates),
        accrued_profit=sum(cf.amount for cf in sukuk.cashflows) - sukuk.face_value,
    )


def _pv_at_yield(sukuk: Sukuk, ytm: float) -> float:
    return sum(cf.amount * math.exp(-ytm * cf.time_years) for cf in sukuk.cashflows)


def yield_to_maturity(
    sukuk: Sukuk,
    *,
    market_price: float,
    tolerance: float = 1e-7,
    max_iterations: int = 200,
) -> float:
    """Bisect for the yield that equates DCF to market price."""
    if market_price <= 0:
        raise ValueError("market_price must be positive")
    if not is_tradable_in_secondary(sukuk.sukuk_type):
        raise ValueError("yield_to_maturity not defined for non-tradable sukuk types")

    lo, hi = -0.05, 0.50
    f_lo = _pv_at_yield(sukuk, lo) - market_price
    f_hi = _pv_at_yield(sukuk, hi) - market_price
    if f_lo * f_hi > 0:
        # Edge case: market price outside DCF range at curve extremes.
        # Return whichever bound is closer.
        return lo if abs(f_lo) < abs(f_hi) else hi

    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        f_mid = _pv_at_yield(sukuk, mid) - market_price
        if abs(f_mid) < tolerance:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def render_pricing(result: PricingResult) -> str:
    if result.secondary_tradable:
        head = (
            f"💰 {result.issuer} PV={result.present_value:.4f} "
            f"(profit accrual: {result.accrued_profit:.4f})"
        )
    else:
        head = (
            f"⏸ {result.issuer} face-value-only: {result.present_value:.4f} "
            "(non-tradable on secondary)"
        )
    return head
