"""Gather fundamentals from SEC and the price store, screen, persist.

One screening run = one ``as_of`` date. Results go to
``halal_screen_results`` keyed (as_of, symbol, method), so every run is
kept: the screening history is point-in-time, and a backtest can ask what
the screen said on a given day instead of using today's verdicts on
yesterday's data. A re-screen under a newer method adds rows beside the old
ones (an order that cited a screen can still be audited against what it
said); readers take the newest method through the ``halal_screen_current``
view. Re-running the same method for the same date replaces its own rows.

Concept choices (each a judgment call, recorded here):

* interest-bearing debt is read two ways and the larger wins (``debt_of``):
  - **filed totals**: LongTermDebt; LongTermDebtNoncurrent + Current;
    LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities;
    LongTermDebtAndCapitalLeaseObligations + its Current part; the
    noncurrent figure + DebtCurrent; DebtAndCapitalLeaseObligations;
    DebtLongtermAndShorttermCombinedAmount; DebtInstrumentCarryingAmount;
    NotesAndLoansPayable. Filers pick one family: CVX files only the
    IncludingCurrentMaturities total, and before v3 its $37B of debt read as
    its $0.4B of short-term borrowings.
  - **the sum of five kinds of borrowing**, each read as the largest of its
    synonyms: notes (NotesPayable, SeniorNotes, SeniorLongTermNotes +
    SeniorNotesCurrent, LongTermNotesPayable + NotesPayableCurrent,
    UnsecuredDebt, UnsecuredLongTermDebt), secured debt (SecuredDebt,
    SecuredLongTermDebt), credit lines (LineOfCredit, LongTermLineOfCredit +
    LinesOfCreditCurrent), convertibles (ConvertibleDebt, its Noncurrent +
    Current, ConvertibleNotesPayable, ConvertibleLongTermNotesPayable +
    ConvertibleNotesPayableCurrent) and loans (LoansPayable, OtherLongTermDebt,
    OtherBorrowings, OtherLoansPayable). REITs and homebuilders file their debt
    only this way: KRC as SecuredDebt + UnsecuredDebt, BXP as SeniorNotes +
    SecuredDebt, NNN as NotesPayable + LoansPayable. Before v11 none of these
    were read, and 98 of 467 passes on 2026-10-05 read as debt-free.
  Synonyms within a kind are not added (MAA files the same notes as
  NotesPayable and UnsecuredDebt), kinds are, and the two readings are never
  added to each other. Where a filer tags one borrowing under two kinds the
  sum overstates (MAA's NotesPayable includes its $0.36B secured debt: +6%);
  the error is always upward, which the strict option accepts.
  ShortTermBorrowings, CommercialPaper and FinanceLeaseLiability are added
  on top, as before. Operating leases are excluded (not borrowing).
  A company that tags none of it reads as debt-free, and is then held to its
  interest expense (InterestExpense, InterestExpenseNonoperating,
  InterestExpenseDebt, InterestPaidNet; the largest): debt implied at 6%
  over the limit, with the tags explaining under half, is doubtful
  (aaoifi.IMPLIED_RATE). AES files ~$20B of debt under its own elements.
* cash and interest-bearing securities = CashAndCashEquivalentsAtCarryingValue
  (else CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents, which
  is larger, so strict, else Cash) plus the largest of ShortTermInvestments,
  MarketableSecuritiesCurrent and AvailableForSaleSecuritiesDebtSecuritiesCurrent
  (companies often tag the same holding under more than one, so they are not
  summed).
* interest income = InvestmentIncomeInterest, else InterestIncomeOther, else
  InvestmentIncomeInterestAndDividend; plus lending income, the largest of
  InterestAndDividendIncomeOperating, InterestAndFeeIncomeLoansAndLeases (and
  its Commercial, Consumer and RealEstate parts), InterestIncomeOperating,
  InterestIncomeSecuritiesMortgageBacked and FinancialServicesRevenue: 82%
  of STWD's revenue is InterestAndFeeIncomeLoansCommercial, which v10 did not
  read. Revenue = Revenues, else RevenueFromContractWithCustomerExcluding-
  AssessedTax, else the Including variant, else SalesRevenueNet. All from the
  latest calendar-year frame that has them.
* loans held = the largest of MortgageLoansOnRealEstate, LoansAndLeases-
  ReceivableNetReportedAmount, FinancingReceivableExcludingAccruedInterest-
  AfterAllowanceForCreditLoss and NotesReceivableNet, against Assets: a REIT
  (SIC 6798) whose assets are mostly loans is a lender.
* receivables = the largest of AccountsReceivableNetCurrent,
  ReceivablesNetCurrent (CAH and MCK file only this) and
  AccountsNotesAndLoansReceivableNetCurrent. Unreported still counts as
  none: the 49% test rarely binds, and an absent tag on a REIT or a
  software company is usually a real absence.
* price = the latest raw close in daily_bars (run `halal-trader data
  backfill` first). Shares = dei EntityCommonStockSharesOutstanding, else
  diluted weighted-average shares: multi-class filers (GOOG, META) report
  dei shares per class only, which the frames API leaves out. Average
  price = the mean of the last 36 month-end closes, split- and
  dividend-adjusted (needs at least 12) and then put back on the share
  basis of ``as_of`` (adjusted closes also carry every later split).
  Market cap = shares x the average price, else x the price. Adjusting
  for dividends within the window lowers past prices a little, so the
  average understates market cap and errs strict.
* splits after a share count was filed: the price is today's, the count is
  as of its filing, so each count is put on the price's basis
  (``corporate_actions``, ``rescale``). A split shows in the stored bars as
  a one-session jump in raw / adjusted close (adjusted bars carry every
  split up to the day they were fetched), and that jump is the split ratio:
  APH's 2:1 of 2026-09-03 and MNST's of 2026-08-11 halved their market caps
  until v11, and KLAC's pre-split diluted count won the mis-scale guard
  against its post-split cover page. A count is rescaled by the splits
  after it was known: the cover page's count from its own date, the
  financial statements' counts from the filing date (SEC.filed), since a
  filing issued after a split restates them (KLAC's 10-K restated its
  2025 balance-sheet count ten-fold). When the filing date is unknown only
  reverse splits apply (fewer shares: strict). A jump that is no split
  ratio (a spin-off, a large special dividend) leaves the count unknown
  and the verdict doubtful until the next filing. An adjusted series that
  jumps by a split ratio together with the raw one was fetched before the
  split and never re-adjusted: its average mixes share bases, so that name
  falls back to the spot price.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.aaoifi import Fundamentals, ScreenResult, screen
from halal_trader.compliance.sec import Company, Fact, SecClient, SecUnavailable
from halal_trader.compliance.successors import lineage
from halal_trader.data.store import last_closes

logger = logging.getLogger(__name__)

SUBMISSIONS_UNAVAILABLE = "SEC submissions record unavailable"
# A run tolerates this many companies whose submissions record EDGAR would
# not serve (each screened doubtful); beyond it the run aborts instead.
MAX_FAILED_NAMES = 5
MAX_FAILED_SHARE = 0.02

# v2: 36-month average market cap; REITs and royalties face the ratios.
# v3: every debt-concept family, and fallbacks for shares, cash and revenue.
# v4: a foreign issuer (20-F/40-F) never passes on ratios: its ADR ratio is unknown.
# v5: a share count the diluted count says is mis-scaled gives way to the smaller.
# v6: the 36-month average no longer carries splits made after the screen date.
# v7: the strict option -- S&P's activity exclusions and receivables test on top
#     of AAOIFI, and an index Shariah board's exclusion as a veto.
# v8: the veto recognises holdings named in fund-administrator style.
# v9: unreported interest income is estimated (cash and securities x 5%), not zero.
# v10: a successor registrant reads its predecessor's facts; two fallback tags.
# v11: debt read from the kinds REITs and homebuilders file (notes, secured,
#      credit lines, convertibles, loans) and checked against interest expense;
#      lending income is impure; a REIT that mostly holds loans is a lender;
#      more receivables tags; implausible share counts are doubtful; a CIK
#      deny-list for activities a SIC code hides; share counts rescaled for
#      splits after they were filed.
# v12: bars, tobacco stores and three cruise lines are excluded; in a mixed-activity sector
#      (restaurants, grocers, convenience, variety and warehouse stores, drug
#      stores, grocery wholesale, food manufacturing) a pass needs a Shariah
#      index's inclusion, else it is doubtful.
METHOD = "aaoifi-sec-v12"
UNMAPPED = "not an SEC registrant (or ticker not mapped)"
_MIN_MONTHS = 12

_DEBT_TOTAL = "LongTermDebt"
_DEBT_PARTS = ("LongTermDebtNoncurrent", "LongTermDebtCurrent")
_DEBT_WITH_LEASES_TOTAL = "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities"
_DEBT_WITH_LEASES = "LongTermDebtAndCapitalLeaseObligations"
_DEBT_WITH_LEASES_CURRENT = "LongTermDebtAndCapitalLeaseObligationsCurrent"
_DEBT_CURRENT = "DebtCurrent"
# Totals a filer may report instead of the families above.
_DEBT_OTHER_TOTALS = (
    "DebtAndCapitalLeaseObligations",
    "DebtLongtermAndShorttermCombinedAmount",
    "DebtInstrumentCarryingAmount",
    "NotesAndLoansPayable",
)
# (kind, alternatives): an alternative is concepts summed (noncurrent + current);
# a kind is the largest alternative; the kinds are added together.
DEBT_KINDS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    (
        "notes",
        (
            ("NotesPayable",),
            ("SeniorNotes",),
            ("SeniorLongTermNotes", "SeniorNotesCurrent"),
            ("LongTermNotesPayable", "NotesPayableCurrent"),
            ("UnsecuredDebt",),
            ("UnsecuredLongTermDebt",),
        ),
    ),
    ("secured", (("SecuredDebt",), ("SecuredLongTermDebt",))),
    ("credit lines", (("LineOfCredit",), ("LongTermLineOfCredit", "LinesOfCreditCurrent"))),
    (
        "convertibles",
        (
            ("ConvertibleDebt",),
            ("ConvertibleDebtNoncurrent", "ConvertibleDebtCurrent"),
            ("ConvertibleNotesPayable",),
            ("ConvertibleLongTermNotesPayable", "ConvertibleNotesPayableCurrent"),
        ),
    ),
    (
        "loans",
        (("LoansPayable",), ("OtherLongTermDebt",), ("OtherBorrowings",), ("OtherLoansPayable",)),
    ),
)
_DEBT_KIND_CONCEPTS = tuple(
    dict.fromkeys(c for _, alternatives in DEBT_KINDS for alt in alternatives for c in alt)
)
_CASH_WIDE = "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
_DILUTED_SHARES = "WeightedAverageNumberOfDilutedSharesOutstanding"
_DEBT_EXTRA = ("ShortTermBorrowings", "CommercialPaper", "FinanceLeaseLiability")
_CASH = "CashAndCashEquivalentsAtCarryingValue"
_CASH_PLAIN = "Cash"  # SLB tags its cash only as this
_BALANCE_SHARES = "CommonStockSharesOutstanding"  # balance-sheet count, one class only
_RECEIVABLES = (
    "AccountsReceivableNetCurrent",
    "ReceivablesNetCurrent",
    "AccountsNotesAndLoansReceivableNetCurrent",
)
_LOANS_HELD = (
    "MortgageLoansOnRealEstate",
    "LoansAndLeasesReceivableNetReportedAmount",
    "FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss",
    "NotesReceivableNet",
)
_ASSETS = "Assets"
_SECURITIES = (
    "ShortTermInvestments",
    "MarketableSecuritiesCurrent",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
)
_INTEREST = (
    "InvestmentIncomeInterest",
    "InterestIncomeOther",
    "InvestmentIncomeInterestAndDividend",
)
_LENDER_INCOME = (
    "InterestAndDividendIncomeOperating",
    "InterestAndFeeIncomeLoansAndLeases",
    "InterestAndFeeIncomeLoansCommercial",
    "InterestAndFeeIncomeLoansConsumer",
    "InterestAndFeeIncomeLoansRealEstate",
    "InterestIncomeOperating",
    "InterestIncomeSecuritiesMortgageBacked",
    "FinancialServicesRevenue",
)
_INTEREST_EXPENSE = (
    "InterestExpense",
    "InterestExpenseNonoperating",
    "InterestExpenseDebt",
    "InterestPaidNet",
)
_REVENUE = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)


def method_rank(method: str) -> int | None:
    """The version number of a method name ("aaoifi-sec-v11" -> 11), as the SQL
    function ``halal_screen_method_rank`` reads it; None when it has none."""
    match = re.search(r"-v([0-9]+)$", method)
    return int(match.group(1)) if match else None


SHARE_CONFLICT = 3.0  # cover-page and diluted counts further apart than this: a units error


def share_count(cover: float | None, diluted: float | None) -> float | None:
    """The cover-page share count, unless the diluted count says it is mis-scaled.

    Filers sometimes tag shares at the wrong scale (Alcoa once reported 185
    trillion shares, Woodward 62 billion). A count that large shrinks every
    ratio towards zero, so when the two sources disagree by more than
    ``SHARE_CONFLICT`` the smaller wins: the larger ratios fail closed.
    """
    if cover is None or diluted is None or diluted <= 0 or cover <= 0:
        return cover if cover is not None else diluted
    if max(cover, diluted) / min(cover, diluted) > SHARE_CONFLICT:
        return min(cover, diluted)
    return cover


def _sum(*values: float | None) -> float | None:
    """Sum of the values present; None if none is."""
    present = [x for x in values if x is not None]
    return sum(present) if present else None


def _largest(*values: float | None) -> float | None:
    return max((x for x in values if x is not None), default=None)


def debt_of(v: Callable[[str], float | None]) -> tuple[float | None, dict[str, float | None]]:
    """Interest-bearing debt from one company's facts (``v``: concept -> value).

    Returns the debt (None when no debt concept is filed at all) and the two
    readings behind it, for the record. See the module docstring for why
    the larger reading wins and what each can overstate.
    """
    noncurrent = v(_DEBT_PARTS[0])
    totals = _largest(
        v(_DEBT_TOTAL),
        _sum(noncurrent, v(_DEBT_PARTS[1])),
        v(_DEBT_WITH_LEASES_TOTAL),
        _sum(v(_DEBT_WITH_LEASES), v(_DEBT_WITH_LEASES_CURRENT)),
        _sum(noncurrent if noncurrent is not None else v(_DEBT_WITH_LEASES), v(_DEBT_CURRENT)),
        *(v(c) for c in _DEBT_OTHER_TOTALS),
    )
    kinds = _sum(
        *(
            _largest(*(_sum(*(v(c) for c in alt)) for alt in alternatives))
            for _, alternatives in DEBT_KINDS
        )
    )
    extras = _sum(*(v(e) for e in _DEBT_EXTRA))
    core = _largest(totals, kinds)
    debt = None if core is None and extras is None else (core or 0.0) + (extras or 0.0)
    return debt, {"debt_from_totals": totals, "debt_from_kinds": kinds, "debt_extras": extras}


def recent_quarter_instants(as_of: date, n: int = 5) -> list[str]:
    """SEC instant-frame periods for the last ``n`` completed quarters, newest first."""
    year, quarter = as_of.year, (as_of.month - 1) // 3  # quarters fully before as_of
    out = []
    for _ in range(n):
        if quarter == 0:
            year, quarter = year - 1, 4
        out.append(f"CY{year}Q{quarter}I")
        quarter -= 1
    return out


def _newest_fact(frames: list[dict[int, Fact]], cik: int) -> Fact | None:
    """The most recent fact across frames (ordered newest period first),
    from ``cik`` or, if it has none, its predecessor (compliance/successors.py)."""
    for filer in lineage(cik):
        best: Fact | None = None
        for frame in frames:
            fact = frame.get(filer)
            if fact is not None and (best is None or fact.end > best.end):
                best = fact
        if best is not None:
            return best
    return None


def _newest(frames: list[dict[int, Fact]], cik: int) -> float | None:
    fact = _newest_fact(frames, cik)
    return fact.val if fact is not None else None


async def _instant(
    sec: SecClient, concept: str, periods: list[str], *, dei: bool = False
) -> list[dict[int, Fact]]:
    taxonomy, unit = ("dei", "shares") if dei else ("us-gaap", "USD")
    return [await sec.frame(taxonomy, concept, unit, p) for p in periods]


async def _annual(sec: SecClient, concept: str, as_of: date) -> list[dict[int, Fact]]:
    return [
        await sec.frame("us-gaap", concept, "USD", f"CY{y}")
        for y in (as_of.year - 1, as_of.year - 2)
    ]


async def _mapped_by_name(engine: AsyncEngine) -> dict[str, tuple[int, str]]:
    from halal_trader.compliance.delisted import mapped_ciks

    return await mapped_ciks(engine)


async def company_map(sec: SecClient, engine: AsyncEngine) -> dict[str, Company]:
    """Ticker -> SEC company as a screen reads it: SEC's current ticker file, then
    the tickers it no longer lists, matched to their filer by name
    (compliance/delisted.py). SEC's own current mapping wins."""
    companies = await sec.companies()
    for sym, (cik, title) in (await _mapped_by_name(engine)).items():
        companies.setdefault(sym, Company(cik, sym, title))
    return companies


def company_of(companies: Mapping[str, Company], symbol: str) -> Company | None:
    """``symbol``'s company in a ``company_map`` (SEC writes BRK.B as BRK-B)."""
    sym = symbol.upper()
    return companies.get(sym) or companies.get(sym.replace(".", "-"))


# Ratios a split or reverse split is declared in. A one-session jump in raw /
# adjusted close within SPLIT_TOLERANCE of one is a split; any other jump over
# ADJUSTMENT_JUMP is an adjustment we cannot read (a spin-off, a big special
# dividend). 5:4 and 4:3 are left out on purpose: rare as splits, and near
# the size of a spin-off, where reading one as the other would add shares.
_FORWARD = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
_REVERSE = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100)
SPLIT_RATIOS = (*_FORWARD, *(1.0 / x for x in _REVERSE))
# Every split in the stored bars of 2025-26 jumps within 0.03% of its ratio
# (APH 1.9998, KLAC 10.0000, ORLY 15.0006); Fortive's spin-off of Ralliant
# jumped 1.4933, 0.45% off 3:2, and must not read as a split (it would add
# half again to the share count). A split on an ex-dividend day misses too,
# and is left unknown (doubtful) until the next filing.
SPLIT_TOLERANCE = 0.003
ADJUSTMENT_JUMP = 1.2  # raw / adjusted moving more than this in a session is not a dividend
_STALE_JUMP = 1.4  # an adjusted close moving this much in a session, raw / adjusted not at all


def split_ratio(jump: float) -> float | None:
    """The split ratio a one-session jump in raw / adjusted close is, or None."""
    for ratio in SPLIT_RATIOS:
        if abs(jump / ratio - 1.0) <= SPLIT_TOLERANCE:
            return ratio
    return None


@dataclass(frozen=True, slots=True)
class CorporateActions:
    """What the stored bars say happened to each symbol's share basis.

    ``splits``: symbol -> [(first session on the new basis, ratio or None)],
    a ratio of new shares per old (2.0 for 2:1, 0.1 for 1:10), None for an
    adjustment that is no split. ``stale``: symbol -> the session where the
    adjusted series jumps by a split ratio with the raw one (never
    re-adjusted after the split).
    """

    splits: dict[str, list[tuple[date, float | None]]] = field(default_factory=dict)
    stale: dict[str, date] = field(default_factory=dict)


async def corporate_actions(
    engine: AsyncEngine, symbols: Sequence[str], since: date, through: date
) -> CorporateActions:
    """Splits and unreadable adjustments in (since, through], from raw vs adjusted bars."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                """
                WITH s AS (
                    SELECT r.symbol, r.day, r.close AS raw, a.close AS adj,
                           lag(r.close) OVER w AS raw_prev, lag(a.close) OVER w AS adj_prev
                    FROM daily_bars r
                    JOIN daily_bars a
                      ON a.symbol = r.symbol AND a.day = r.day AND a.adjustment = 'all'
                    WHERE r.adjustment = 'raw' AND r.symbol = ANY(:s)
                      AND r.day >= :since AND r.day <= :through
                    WINDOW w AS (PARTITION BY r.symbol ORDER BY r.day)
                )
                SELECT symbol, day, raw, adj, raw_prev, adj_prev FROM s
                WHERE raw > 0 AND adj > 0 AND raw_prev > 0 AND adj_prev > 0
                  AND (abs(ln((raw_prev / adj_prev) / (raw / adj))) > ln(:jump)
                       OR abs(ln(adj_prev / adj)) > ln(:stale))
                ORDER BY symbol, day
                """
            ),
            {
                "s": [s.upper() for s in symbols],
                "since": since,
                "through": through,
                "jump": ADJUSTMENT_JUMP,
                "stale": _STALE_JUMP,
            },
        )
        found = CorporateActions()
        for r in rows:
            raw, adj, raw_prev, adj_prev = (
                float(r.raw),
                float(r.adj),
                float(r.raw_prev),
                float(r.adj_prev),
            )
            jump = (raw_prev / adj_prev) / (raw / adj)
            if max(jump, 1.0 / jump) > ADJUSTMENT_JUMP:
                found.splits.setdefault(r.symbol, []).append((r.day, split_ratio(jump)))
                continue
            # Raw / adjusted held still while the adjusted close fell by a
            # forward-split ratio: the adjusted bars before this day were
            # fetched before a split and never re-adjusted (or the stock
            # really halved; either way the spot price is the safer basis).
            fell = adj_prev / adj
            if fell > 1.0 and split_ratio(fell) is not None:
                found.stale.setdefault(r.symbol, r.day)
    return found


def rescale(
    fact: Fact | None,
    actions: list[tuple[date, float | None]],
    *,
    known: date | None,
    through: date,
) -> tuple[float | None, str | None]:
    """A share count put on the basis of ``through``, or (None, why) if it cannot be.

    ``known``: the date from which the count reflects every split (the cover
    page's date, or the filing date of a financial statement); None when the
    filing date is unknown, and then only reverse splits apply.
    """
    if fact is None:
        return None, None
    factor = 1.0
    for day, ratio in actions:
        if day <= (known or fact.end) or day > through:
            continue
        if ratio is None:
            return None, (
                f"share count (an adjustment on {day} that is no split: "
                "spin-off or special dividend?)"
            )
        factor *= ratio if known is not None else min(ratio, 1.0)
    return fact.val * factor, None


class _Rebase:
    """One company's share counts, each put on the basis of the price day."""

    def __init__(
        self,
        sec: SecClient,
        cik: int,
        actions: list[tuple[date, float | None]],
        through: date,
    ) -> None:
        self.sec, self.cik, self.actions, self.through = sec, cik, actions, through
        self.issues: list[str] = []

    def count(self, frames: list[dict[int, Fact]], *, cover: bool = False) -> float | None:
        fact = _newest_fact(frames, self.cik)
        if fact is None or not self.actions:
            return fact.val if fact is not None else None
        known = fact.end if cover else self.sec.filed(fact.accn)
        value, issue = rescale(fact, self.actions, known=known, through=self.through)
        if issue is not None and issue not in self.issues:
            self.issues.append(issue)
        return value


Audit = dict[str, dict[str, float | None]]


async def gather(
    sec: SecClient, engine: AsyncEngine, symbols: Sequence[str], as_of: date
) -> tuple[list[Fundamentals], dict[str, tuple[int | None, str]], dict[str, str], Audit]:
    """Fundamentals for each symbol, (cik, SIC description) for the record, company
    names, and the intermediate figures worth keeping beside each verdict."""
    companies = await company_map(sec, engine)
    periods = recent_quarter_instants(as_of)
    frames = {
        c: await _instant(sec, c, periods)
        for c in (
            _DEBT_TOTAL,
            *_DEBT_PARTS,
            _DEBT_WITH_LEASES_TOTAL,
            _DEBT_WITH_LEASES,
            _DEBT_WITH_LEASES_CURRENT,
            _DEBT_CURRENT,
            *_DEBT_OTHER_TOTALS,
            *_DEBT_KIND_CONCEPTS,
            *_DEBT_EXTRA,
            _CASH,
            _CASH_WIDE,
            _CASH_PLAIN,
            *_SECURITIES,
            *_RECEIVABLES,
            *_LOANS_HELD,
            _ASSETS,
        )
    }
    shares = await _instant(sec, "EntityCommonStockSharesOutstanding", periods, dei=True)
    diluted = [
        await sec.frame("us-gaap", _DILUTED_SHARES, "shares", p.removesuffix("I")) for p in periods
    ]
    balance_shares = [await sec.frame("us-gaap", _BALANCE_SHARES, "shares", p) for p in periods]
    annual = {
        c: await _annual(sec, c, as_of)
        for c in (*_INTEREST, *_LENDER_INCOME, *_INTEREST_EXPENSE, *_REVENUE)
    }
    priced = await last_closes(engine, [s.upper() for s in symbols], on_or_before=as_of)
    prices = {sym: close for sym, (_, close) in priced.items()}
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                """
                WITH month_end AS (
                    SELECT DISTINCT ON (symbol, date_trunc('month', day)) symbol, close
                    FROM daily_bars
                    WHERE adjustment = 'all' AND symbol = ANY(:s)
                      AND day <= :as_of AND day > :as_of - INTERVAL '36 months'
                    ORDER BY symbol, date_trunc('month', day), day DESC
                )
                SELECT symbol, avg(close) AS avg_close FROM month_end
                GROUP BY symbol HAVING count(*) >= :min_months
                """
            ),
            {"s": [s.upper() for s in symbols], "as_of": as_of, "min_months": _MIN_MONTHS},
        )
        adjusted_averages = {r.symbol: float(r.avg_close) for r in rows}
        rows = await conn.execute(
            text(
                "SELECT symbol, close FROM daily_bars WHERE adjustment = 'all' AND (symbol, day) "
                "IN (SELECT unnest(CAST(:s AS TEXT[])), unnest(CAST(:d AS DATE[])))"
            ),
            {
                "s": [sym for sym in adjusted_averages if sym in priced],
                "d": [priced[sym][0] for sym in adjusted_averages if sym in priced],
            },
        )
        adjusted_now = {r.symbol: float(r.close) for r in rows}  # same session as the raw price
    # Adjusted closes carry every split and dividend up to the day the bars
    # were fetched, including those after ``as_of``, which the share count
    # filed at ``as_of`` knows nothing of. Raw / adjusted on ``as_of`` is
    # exactly that later adjustment; dividing it back out puts the average
    # on the share basis of ``as_of`` (without it, AMZN's 2019 average was
    # its price after the 2022 20:1 split, and its market cap 20x too small).
    averages = {
        sym: avg * prices[sym] / adjusted_now[sym]
        for sym, avg in adjusted_averages.items()
        if sym in prices and adjusted_now.get(sym)
    }
    # The window covers the 36-month average and every share fact read
    # (five quarters back, plus the cover page's lag).
    actions = await corporate_actions(engine, symbols, as_of - timedelta(days=3 * 366), as_of)
    for sym, day in actions.stale.items():
        if averages.pop(sym, None) is not None:
            logger.warning(
                "halal screen: %s adjusted bars jump with the raw ones on %s (fetched before "
                "a split?): spot price used; re-fetch its adjusted bars",
                sym,
                day,
            )

    out: list[Fundamentals] = []
    meta: dict[str, tuple[int | None, str]] = {}
    titles: dict[str, str] = {}
    audit: Audit = {}
    failed: list[str] = []
    for symbol in symbols:
        sym = symbol.upper()
        company = company_of(companies, sym)
        if company is None:
            out.append(Fundamentals(sym, None, None, prices.get(sym), None, None, None, None))
            meta[sym] = (None, UNMAPPED)
            continue
        cik = company.cik
        try:
            sic, sic_desc = await sec.sic(cik)
            foreign = await sec.foreign_filer(cik)
        except SecUnavailable as exc:
            # One company's submissions record is not worth the night's
            # screen: without its industry code it is doubtful, never halal.
            logger.warning("halal screen: %s: %s", sym, exc)
            failed.append(sym)
            sic, sic_desc, foreign = None, SUBMISSIONS_UNAVAILABLE, False
        meta[sym] = (cik, sic_desc)
        titles[sym] = company.title

        def v(concept: str, cik: int = cik) -> float | None:
            return _newest(frames[concept], cik)

        def yearly(concepts: Sequence[str], cik: int = cik) -> list[float | None]:
            return [_newest(annual[c], cik) for c in concepts]

        cash = v(_CASH)
        if cash is None:
            cash = v(_CASH_WIDE)
        if cash is None:
            cash = v(_CASH_PLAIN)
        securities = _largest(*(v(s) for s in _SECURITIES)) or 0.0
        files_xbrl = cash is not None or _newest(shares, cik) is not None
        debt, readings = debt_of(v)
        if debt is None and files_xbrl:
            debt = 0.0  # tags no debt at all; interest expense is the check (aaoifi)
        interest = next((x for x in yearly(_INTEREST) if x is not None), None)
        revenue = next((x for x in yearly(_REVENUE) if x is not None), None)
        # Each share count on the basis of the price it multiplies (see the
        # module docstring): the cover page's from its own date, the
        # statements' from their filing date.
        on_basis = _Rebase(
            sec,
            cik,
            actions.splits.get(sym, []),
            priced[sym][0] if sym in priced else as_of,
        )
        # The cover page's count, else the diluted average, else the balance
        # sheet's (HSY tags the cover page per share class, which frames
        # omit; the balance-sheet count is its common stock alone, so the
        # market cap errs small: ratios fail closed).
        share_total = share_count(
            on_basis.count(shares, cover=True), on_basis.count(diluted)
        ) or on_basis.count(balance_shares)
        issues = on_basis.issues
        cover_fact = _newest_fact(shares, cik)
        audit[sym] = {
            **readings,
            "interest_expense": _largest(*yearly(_INTEREST_EXPENSE)),
            "lender_income": _largest(*yearly(_LENDER_INCOME)),
            "loans_receivable": _largest(*(v(c) for c in _LOANS_HELD)),
            "total_assets": v(_ASSETS),
            "receivables": _largest(*(v(c) for c in _RECEIVABLES)),
            "shares_as_filed": cover_fact.val if cover_fact is not None else None,
        }
        out.append(
            Fundamentals(
                symbol=sym,
                sic=sic,
                shares_outstanding=share_total,
                price=prices.get(sym),
                interest_bearing_debt=debt,
                cash_and_securities=(cash + securities) if cash is not None else None,
                interest_income=interest,
                revenue=revenue,
                average_price=averages.get(sym),
                foreign_filer=foreign,
                receivables=audit[sym]["receivables"],
                cik=cik,
                interest_expense=audit[sym]["interest_expense"],
                lender_income=audit[sym]["lender_income"],
                loans_receivable=audit[sym]["loans_receivable"],
                total_assets=audit[sym]["total_assets"],
                data_issues=tuple(issues) if share_total is None else (),
            )
        )
    mapped = sum(1 for cik, _ in meta.values() if cik is not None)
    if len(failed) > max(MAX_FAILED_NAMES, MAX_FAILED_SHARE * mapped):
        # EDGAR is down, not one record missing: a screen with this many
        # doubtful names would empty the universe. Abort; the last one stands.
        raise SecUnavailable(
            f"submissions unavailable for {len(failed)} of {mapped} companies "
            f"({', '.join(failed[:5])}, ...)"
        )
    return out, meta, titles, audit


async def run_screen(
    sec: SecClient, engine: AsyncEngine, symbols: Sequence[str], as_of: date
) -> list[ScreenResult]:
    """Screen ``symbols`` as of ``as_of`` and store every verdict with its inputs."""
    from halal_trader.compliance.index_veto import apply_veto, require_board, views_at

    fundamentals, meta, titles, audit = await gather(sec, engine, symbols, as_of)
    views = await views_at(engine, as_of)
    results = apply_veto([screen(f) for f in fundamentals], titles, views)
    results = require_board(results, {f.symbol: f.sic for f in fundamentals}, titles, views)
    async with engine.begin() as conn:
        for f, r in zip(fundamentals, results, strict=True):
            cik, sic_desc = meta.get(r.symbol, (None, ""))
            await conn.execute(
                text(
                    """
                    INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description,
                        verdict, reasons, metrics, method, screened_at)
                    VALUES (:as_of, :symbol, :cik, :sic_desc, :verdict, CAST(:reasons AS JSONB),
                        CAST(:metrics AS JSONB), :method, now())
                    ON CONFLICT (as_of, symbol, method) DO UPDATE SET cik = EXCLUDED.cik,
                        sic_description = EXCLUDED.sic_description, verdict = EXCLUDED.verdict,
                        reasons = EXCLUDED.reasons, metrics = EXCLUDED.metrics,
                        screened_at = EXCLUDED.screened_at
                    """
                ),
                {
                    "as_of": as_of,
                    "symbol": r.symbol,
                    "cik": cik,
                    "sic_desc": sic_desc,
                    "verdict": r.verdict,
                    "reasons": json.dumps(r.reasons),
                    "metrics": json.dumps(
                        {
                            **r.metrics,
                            **audit.get(r.symbol, {}),
                            "interest_bearing_debt": f.interest_bearing_debt,
                            "cash_and_securities": f.cash_and_securities,
                            "interest_income": f.interest_income,
                            "revenue": f.revenue,
                            "shares_outstanding": f.shares_outstanding,
                            "price": f.price,
                        }
                    ),
                    "method": METHOD,
                },
            )
    counts = {
        v: sum(1 for r in results if r.verdict == v) for v in ("halal", "not_halal", "doubtful")
    }
    logger.info("halal screen %s: %s", as_of, counts)
    return results
