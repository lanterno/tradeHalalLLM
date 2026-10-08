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
     estimated as cash and securities x 5%), lending income included: a
     mortgage REIT's loan interest, a captive finance arm's revenue;
   * accounts receivable < 49% of market cap (S&P's fourth ratio).

   Cross-checks that make a pass doubtful rather than trusting it: a share
   count so mis-scaled the market cap is implausible, and interest expense
   implying far more debt than the tags read (``IMPLIED_RATE``). A REIT
   whose assets are mostly loans fails as a lender.

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

from halal_trader.compliance.exclusions import NON_ALCOHOLIC_2080, denied

Verdict = Literal["halal", "not_halal", "doubtful"]

DEBT_LIMIT = 0.30
CASH_LIMIT = 0.30
IMPURE_INCOME_LIMIT = 0.05
RECEIVABLES_LIMIT = 0.49  # S&P Shariah: accounts receivable < 49% of market cap
# Yield assumed on cash and interest-bearing securities when a company does
# not report its interest income: at or above US bill yields in most years
# since 2016, so the estimate errs high.
ESTIMATED_YIELD = 0.05
# Rate at which interest expense is read back into the debt it implies.
# Investment-grade US borrowers paid about 4-5% on average in 2023-2026 and
# high yield 7-9%; 6% sits between, and the implied figure only matters
# when the debt read from the tags explains under half of it.
IMPLIED_RATE = 0.06
REIT_SIC = 6798
# A REIT whose loans and mortgages held are at least this share of its assets
# is a lender (mortgage REITs hold 70-100%; equity REITs' mezzanine loans
# are a few percent: SLG 1%, RHP 1%, STWD 36%).
LOANS_TO_ASSETS_LIMIT = 0.25
# Sanity bound: no balance-sheet ratio of a listed company comes near 50x
# its market cap (a distressed one at 10x is already rare). Beyond it the
# share count is mis-scaled and the verdict is doubtful, not a ratio failure.
IMPLAUSIBLE_RATIO = 50.0

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
    (2080, 2080, "beverages: SIC 2080 does not separate alcoholic drinks from soft drinks"),
    (2082, 2085, "alcoholic beverages"),
    (5181, 5182, "alcohol wholesale"),
    (5921, 5921, "liquor stores"),
    (5813, 5813, "drinking places (bars)"),
    (5993, 5993, "tobacco stores"),
    (2100, 2199, "tobacco products"),
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


# (low, high, reason): SIC ranges whose companies commonly earn impermissible
# revenue that SEC data does not separate: a restaurant's alcohol, a grocer's
# or a drugstore's alcohol and tobacco, pork in processed food. The ratios
# cannot see it, so a pass there stands only if a Shariah board that looked
# at the company's revenue (SPUS's or HLAL's) holds it; otherwise it is
# doubtful (index_veto.require_board). Prohibited ranges above take
# precedence.
MIXED_ACTIVITY_SIC: tuple[tuple[int, int, str], ...] = (
    # Food, not beverages (2080-2089): drink makers carry no pork, and the
    # alcoholic ones are prohibited above.
    (2000, 2079, "food manufacturing (pork products)"),
    (2090, 2099, "miscellaneous food preparations (pork products)"),
    (5140, 5149, "grocery wholesale (alcohol, pork)"),
    (5331, 5331, "variety stores (alcohol, tobacco)"),
    (5399, 5399, "general merchandise and warehouse clubs (alcohol, tobacco)"),
    (5411, 5412, "grocery and convenience stores (alcohol, tobacco)"),
    (5812, 5812, "restaurants (alcohol)"),
    (5912, 5912, "drug stores (alcohol, tobacco)"),
)


def mixed_activity(sic: int | None) -> str | None:
    """The hidden impermissible revenue a company's SIC code suggests, or None."""
    if sic is None:
        return None
    for low, high, reason in MIXED_ACTIVITY_SIC:
        if low <= sic <= high:
            return reason
    return None


def prohibited_activity(sic: int | None, cik: int | None = None) -> str | None:
    """The reason a company's activity is impermissible, or None if it is not.

    A company excluded by name (compliance/exclusions.py) is excluded
    whatever its SIC code; otherwise the code decides, except that the
    named non-alcoholic drink makers are exempt from SIC 2080.
    """
    if (reason := denied(cik)) is not None:
        return reason
    if sic is None:
        return None
    if sic == 2080 and cik in NON_ALCOHOLIC_2080:
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
    cik: int | None = None
    # Annual interest expense: a debt figure the balance-sheet tags may miss
    # (AES files ~$20B of debt under its own elements) shows up here.
    interest_expense: float | None = None
    # Interest and fees earned lending (a mortgage REIT's loan income, a
    # captive finance arm's revenue): impure income on top of interest_income.
    lender_income: float | None = None
    loans_receivable: float | None = None  # loans and mortgages held as assets
    total_assets: float | None = None
    # Why an input could not be trusted; each makes the verdict doubtful.
    data_issues: tuple[str, ...] = ()


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
    ) + max(f.lender_income or 0.0, 0.0)
    impure_ratio = _ratio(impure, f.revenue) if f.revenue else None
    implied_debt = (
        f.interest_expense / IMPLIED_RATE if f.interest_expense and f.interest_expense > 0 else None
    )
    implied_debt_ratio = _ratio(implied_debt, market_cap)
    loan_share = _ratio(f.loans_receivable, f.total_assets)
    metrics: dict[str, float | None] = {
        "sic": float(f.sic) if f.sic is not None else None,
        "market_cap": market_cap,
        "market_cap_basis": 36.0 if f.average_price is not None else 0.0,
        "debt_ratio": debt_ratio,
        "cash_ratio": cash_ratio,
        "receivables_ratio": receivables_ratio,
        "impure_income_ratio": impure_ratio,
        "impure_income_estimated": 1.0 if estimated else 0.0,
        "lender_income_ratio": _ratio(f.lender_income, f.revenue) if f.lender_income else None,
        "implied_debt_ratio": implied_debt_ratio,
        "loans_to_assets": loan_share,
    }

    activity = prohibited_activity(f.sic, f.cik)
    if activity is None and f.sic == REIT_SIC and loan_share is not None:
        if loan_share >= LOANS_TO_ASSETS_LIMIT:
            # SIC 6798 files mortgage REITs beside equity REITs. One whose
            # assets are mostly loans is a lender, whatever it calls itself.
            activity = (
                f"financial: loans and mortgages are {loan_share:.0%} of assets "
                f"(>= {LOANS_TO_ASSETS_LIMIT:.0%}): a mortgage REIT lends at interest"
            )
    if activity is not None:
        return ScreenResult(f.symbol, "not_halal", [f"business activity: {activity}"], metrics)

    failures: list[str] = []
    missing: list[str] = list(f.data_issues)
    if f.sic is None:
        missing.append("industry code")
    if (
        market_cap is not None
        and max(debt_ratio or 0.0, cash_ratio or 0.0, receivables_ratio or 0.0) > IMPLAUSIBLE_RATIO
    ):
        # McDonald's tagged 711 diluted shares (millions, unscaled): a market
        # cap of $0.2M and a debt ratio of 22,948,444%. That is a units error,
        # not a verdict on the company; it says nothing either way.
        return ScreenResult(
            f.symbol,
            "doubtful",
            [
                *(f"not computable: {m}" for m in missing),
                f"not computable: market cap {market_cap:,.0f} is implausible against the "
                f"balance sheet (over {IMPLAUSIBLE_RATIO:.0f}x): share count probably mis-scaled",
            ],
            metrics,
        )
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
    if (
        implied_debt is not None
        and implied_debt_ratio is not None
        and implied_debt_ratio >= DEBT_LIMIT
        and (f.interest_bearing_debt or 0.0) < implied_debt / 2
    ):
        # The interest bill says the debt is there even if no tag we read
        # does: at IMPLIED_RATE, interest expense implies debt over the limit,
        # and the reported debt accounts for under half of it. Doubtful, not a
        # failure: the true figure is unknown, but it is not evidence of a pass.
        missing.append(
            f"interest-bearing debt (interest expense implies about "
            f"{implied_debt_ratio:.0%} of market cap at {IMPLIED_RATE:.0%}; "
            f"the debt tags read explain under half)"
        )
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
