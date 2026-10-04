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
2. **Financial ratios**, each against market capitalisation or revenue.
   Market cap is shares outstanding x the 36-month average month-end
   price when that history exists (S&P's Shariah methodology, which SPUS
   follows: a spot price makes a 30-40% name flip verdict with every
   swing), else x the latest price:
   * interest-bearing debt < 30% of market cap;
   * cash + interest-bearing securities < 30% of market cap;
   * interest (impermissible) income < 5% of revenue (when not reported,
     estimated as cash and securities x 5%);
   * accounts receivable < 49% of market cap (S&P's fourth ratio).

The operator chose the strict option (2026-10-02): wherever AAOIFI and
S&P Shariah differ, the stricter rule applies, and an index Shariah
board's exclusion is a veto (compliance/index_veto.py).

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
RECEIVABLES_LIMIT = 0.49  # S&P Shariah: accounts receivable < 49% of market cap
# Yield assumed on cash and interest-bearing securities when a company does
# not report its interest income: at or above US bill yields in most years
# since 2016, so the estimate errs high.
ESTIMATED_YIELD = 0.05

# (low, high, reason): SIC ranges whose primary activity is impermissible.
PROHIBITED_SIC: tuple[tuple[int, int, str], ...] = (
    (6000, 6299, "conventional banking, credit or securities dealing (riba)"),
    (6300, 6411, "conventional insurance (gharar, riba)"),
    # 67xx is split: holding companies, funds, trusts and blank-check
    # shells stay out; real-asset businesses filed there (6792 oil
    # royalties, 6794 patent lessors, 6795 mineral royalties, 6798 REITs)
    # face the ratio screens instead. A mortgage REIT shares 6798 but earns
    # interest, so the impure-income ratio rejects it.
    (6700, 6791, "holding, investment and blank-check companies (financial)"),
    (6793, 6793, "commodity traders (financial)"),
    (6796, 6797, "investment offices (financial)"),
    (6799, 6799, "investors, not elsewhere classified (financial)"),
    (2082, 2085, "alcoholic beverages"),
    (5181, 5182, "alcohol wholesale"),
    (5921, 5921, "liquor stores"),
    (2111, 2141, "tobacco products"),
    (5194, 5194, "tobacco wholesale"),
    (2011, 2013, "meat packing and processing (pork)"),
    (7011, 7011, "hotels and casinos (gambling, alcohol)"),
    (7993, 7993, "coin-operated amusement and gaming devices"),
    # The strict option (operator, 2026-10-02): S&P Shariah's activity
    # exclusions on top of AAOIFI's. Advertising, media and entertainment
    # are out under S&P; a SIC code cannot see them inside a company filed
    # as software or retail, which is what the index veto is for.
    (7310, 7319, "advertising (S&P Shariah: advertising and media)"),
    (4830, 4841, "radio, television and cable broadcasting (S&P: media)"),
    (7810, 7833, "motion pictures and theaters (S&P: entertainment)"),
    (7841, 7841, "video rental and streaming (S&P: entertainment)"),
    (3652, 3652, "recorded music (S&P: entertainment)"),
    (7900, 7999, "amusement, recreation and gaming (S&P: entertainment, gambling)"),
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
    average_price: float | None = None  # 36-month mean month-end close
    foreign_filer: bool = False  # files 20-F/40-F: shares are ordinary shares, not the ADRs
    receivables: float | None = None  # accounts receivable; unreported counts as none


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
    price = f.average_price if f.average_price is not None else f.price
    market_cap = (
        f.shares_outstanding * price
        if f.shares_outstanding is not None and price is not None
        else None
    )
    debt_ratio = _ratio(f.interest_bearing_debt, market_cap)
    cash_ratio = _ratio(f.cash_and_securities, market_cap)
    receivables_ratio = _ratio(f.receivables or 0.0, market_cap)
    # Unreported interest income is not zero: Microsoft has not tagged it under
    # a standard concept since 2013, nor Apple since 2023 (they use their own
    # elements, which frames do not carry). It is then estimated as the cash and
    # interest-bearing securities times ESTIMATED_YIELD, so a missing figure can
    # only make the test, and purification, stricter.
    estimated = f.interest_income is None
    impure = (
        f.interest_income
        if f.interest_income is not None
        else (f.cash_and_securities or 0.0) * ESTIMATED_YIELD
    )
    impure_ratio = _ratio(impure, f.revenue) if f.revenue else None
    metrics: dict[str, float | None] = {
        "sic": float(f.sic) if f.sic is not None else None,
        "market_cap": market_cap,
        "market_cap_basis": 36.0 if f.average_price is not None else 0.0,
        "debt_ratio": debt_ratio,
        "cash_ratio": cash_ratio,
        "receivables_ratio": receivables_ratio,
        "impure_income_ratio": impure_ratio,
        "impure_income_estimated": 1.0 if estimated else 0.0,
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
        ("accounts receivable / market cap", receivables_ratio, RECEIVABLES_LIMIT),
    ):
        if value is None:
            missing.append(name)
        elif value >= limit:
            failures.append(f"{name} {value:.1%} >= {limit:.0%}")
    if failures:
        return ScreenResult(f.symbol, "not_halal", failures, metrics)
    if f.foreign_filer:
        # Ordinary shares x the ADR's price overstates market cap by the ADR
        # ratio (25x for NetEase), shrinking both balance-sheet ratios: a
        # pass here is not evidence. A failure above stands, since the true
        # ratio is only larger.
        missing.append("market cap (foreign issuer: ADR ratio unknown)")
    if missing:
        return ScreenResult(
            f.symbol, "doubtful", [f"not computable: {m}" for m in missing], metrics
        )
    return ScreenResult(f.symbol, "halal", [], metrics)
