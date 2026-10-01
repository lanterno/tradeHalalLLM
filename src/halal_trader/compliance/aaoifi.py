"""AAOIFI-style Shariah screen: pure rules over public fundamentals.

Two screens, both must pass (AAOIFI Shariah Standard No. 21):

1. **Business activity.** The company's primary activity, read from its SEC
   SIC code, must not be prohibited: conventional finance and insurance
   (interest, gharar), alcohol, tobacco, pork processing, gambling. The SIC
   list below is deliberately conservative -- casino *and* general hotels
   are excluded, because SIC 7011 does not separate them -- and every
   exclusion names its reason. A SIC code is a coarse proxy for "primary
   activity"; that is the main reason this screen is validated against
   halal ETF holdings before it may gate trading.
2. **Financial ratios**, each against market capitalisation (shares
   outstanding x latest price) or revenue:
   * interest-bearing debt < 30% of market cap;
   * cash + interest-bearing securities < 30% of market cap;
   * interest (impermissible) income < 5% of revenue.

Anything that cannot be computed makes the verdict ``doubtful``, which is
not halal: the screen fails closed. Metrics are returned with every verdict
so each decision can be audited and the impure-income ratio reused for
dividend purification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["halal", "not_halal", "doubtful"]

DEBT_LIMIT = 0.30
CASH_LIMIT = 0.30
IMPURE_INCOME_LIMIT = 0.05

# (low, high, reason): SIC ranges whose primary activity is impermissible.
PROHIBITED_SIC: tuple[tuple[int, int, str], ...] = (
    (6000, 6299, "conventional banking, credit or securities dealing (riba)"),
    (6300, 6411, "conventional insurance (gharar, riba)"),
    (6700, 6799, "holding, investment and blank-check companies (financial)"),
    (2082, 2085, "alcoholic beverages"),
    (5181, 5182, "alcohol wholesale"),
    (5921, 5921, "liquor stores"),
    (2111, 2141, "tobacco products"),
    (5194, 5194, "tobacco wholesale"),
    (2011, 2013, "meat packing and processing (pork)"),
    (7011, 7011, "hotels and casinos (gambling, alcohol)"),
    (7993, 7993, "coin-operated amusement and gaming devices"),
)


def prohibited_activity(sic: int | None) -> str | None:
    """The reason a SIC code's activity is impermissible, or None if it is not."""
    if sic is None:
        return None
    for low, high, reason in PROHIBITED_SIC:
        if low <= sic <= high:
            return reason
    return None


@dataclass(frozen=True, slots=True)
class Fundamentals:
    """Inputs for one company. None = not reported / not found."""

    symbol: str
    sic: int | None
    shares_outstanding: float | None
    price: float | None
    interest_bearing_debt: float | None
    cash_and_securities: float | None
    interest_income: float | None
    revenue: float | None


@dataclass(frozen=True, slots=True)
class ScreenResult:
    symbol: str
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float | None] = field(default_factory=dict)


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return max(num, 0.0) / den


def screen(f: Fundamentals) -> ScreenResult:
    """Apply both screens to one company's fundamentals."""
    market_cap = (
        f.shares_outstanding * f.price
        if f.shares_outstanding is not None and f.price is not None
        else None
    )
    debt_ratio = _ratio(f.interest_bearing_debt, market_cap)
    cash_ratio = _ratio(f.cash_and_securities, market_cap)
    # No interest income reported at all is common for operating companies
    # that earn none worth tagging; treat it as zero only when revenue exists.
    impure = f.interest_income if f.interest_income is not None else 0.0
    impure_ratio = _ratio(impure, f.revenue) if f.revenue else None
    metrics: dict[str, float | None] = {
        "sic": float(f.sic) if f.sic is not None else None,
        "market_cap": market_cap,
        "debt_ratio": debt_ratio,
        "cash_ratio": cash_ratio,
        "impure_income_ratio": impure_ratio,
    }

    activity = prohibited_activity(f.sic)
    if activity is not None:
        return ScreenResult(f.symbol, "not_halal", [f"business activity: {activity}"], metrics)

    failures: list[str] = []
    missing: list[str] = []
    if f.sic is None:
        missing.append("industry code")
    for name, value, limit in (
        ("interest-bearing debt / market cap", debt_ratio, DEBT_LIMIT),
        ("cash and interest-bearing securities / market cap", cash_ratio, CASH_LIMIT),
        ("interest income / revenue", impure_ratio, IMPURE_INCOME_LIMIT),
    ):
        if value is None:
            missing.append(name)
        elif value >= limit:
            failures.append(f"{name} {value:.1%} >= {limit:.0%}")
    if failures:
        return ScreenResult(f.symbol, "not_halal", failures, metrics)
    if missing:
        return ScreenResult(
            f.symbol, "doubtful", [f"not computable: {m}" for m in missing], metrics
        )
    return ScreenResult(f.symbol, "halal", [], metrics)
