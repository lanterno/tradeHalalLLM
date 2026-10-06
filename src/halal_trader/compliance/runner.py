"""Gather fundamentals from SEC and the price store, screen, persist.

One screening run = one ``as_of`` date. Results go to
``halal_screen_results`` keyed (as_of, symbol), so every run is kept: the
screening history is point-in-time, and a backtest can ask what the screen
said on a given day instead of using today's verdicts on yesterday's data.

Concept choices (each a judgment call, recorded here):

* interest-bearing debt = the largest of the long-term totals companies
  actually file (LongTermDebt; LongTermDebtNoncurrent + LongTermDebtCurrent;
  LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities;
  LongTermDebtAndCapitalLeaseObligations + its Current part; the noncurrent
  figure + DebtCurrent), plus ShortTermBorrowings, CommercialPaper and
  FinanceLeaseLiability. Filers pick one family: CVX files only the
  IncludingCurrentMaturities total, and before v3 its $37B of debt read as
  its $0.4B of short-term borrowings. Overlaps between families can count
  some debt twice; taking the largest errs strict, never lenient.
  Operating leases are excluded (not interest-bearing borrowing). A company
  that files XBRL but tags none of these is treated as debt-free.
* cash and interest-bearing securities = CashAndCashEquivalentsAtCarryingValue
  (else CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents, which
  is larger, so strict) plus the largest of ShortTermInvestments, MarketableSecuritiesCurrent and
  AvailableForSaleSecuritiesDebtSecuritiesCurrent (companies often tag the
  same holding under more than one, so they are not summed).
* interest income = InvestmentIncomeInterest, else InterestIncomeOther, else
  InvestmentIncomeInterestAndDividend; revenue = Revenues, else
  RevenueFromContractWithCustomerExcludingAssessedTax, else the Including
  variant, else SalesRevenueNet.
  Both from the latest calendar-year frame that has them.
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
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.aaoifi import Fundamentals, ScreenResult, screen
from halal_trader.compliance.sec import Company, Fact, SecClient, SecUnavailable
from halal_trader.compliance.successors import lineage

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
METHOD = "aaoifi-sec-v10"
UNMAPPED = "not an SEC registrant (or ticker not mapped)"
_MIN_MONTHS = 12

_DEBT_TOTAL = "LongTermDebt"
_DEBT_PARTS = ("LongTermDebtNoncurrent", "LongTermDebtCurrent")
_DEBT_WITH_LEASES_TOTAL = "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities"
_DEBT_WITH_LEASES = "LongTermDebtAndCapitalLeaseObligations"
_DEBT_WITH_LEASES_CURRENT = "LongTermDebtAndCapitalLeaseObligationsCurrent"
_DEBT_CURRENT = "DebtCurrent"
_CASH_WIDE = "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
_DILUTED_SHARES = "WeightedAverageNumberOfDilutedSharesOutstanding"
_DEBT_EXTRA = ("ShortTermBorrowings", "CommercialPaper", "FinanceLeaseLiability")
_CASH = "CashAndCashEquivalentsAtCarryingValue"
_CASH_PLAIN = "Cash"  # SLB tags its cash only as this
_BALANCE_SHARES = "CommonStockSharesOutstanding"  # balance-sheet count, one class only
_RECEIVABLES = "AccountsReceivableNetCurrent"
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
_REVENUE = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)


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


def _newest(frames: list[dict[int, Fact]], cik: int) -> float | None:
    """The most recent value across frames (ordered newest period first),
    from ``cik`` or, if it has none, its predecessor (compliance/successors.py)."""
    for filer in lineage(cik):
        best: Fact | None = None
        for frame in frames:
            fact = frame.get(filer)
            if fact is not None and (best is None or fact.end > best.end):
                best = fact
        if best is not None:
            return best.val
    return None


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


async def gather(
    sec: SecClient, engine: AsyncEngine, symbols: Sequence[str], as_of: date
) -> tuple[list[Fundamentals], dict[str, tuple[int | None, str]], dict[str, str]]:
    """Fundamentals for each symbol, (cik, SIC description) for the record, and company names."""
    companies = await sec.companies()
    # Tickers SEC no longer lists, matched to their filer by name
    # (compliance/delisted.py); SEC's own current mapping wins.
    for sym, (cik, title) in (await _mapped_by_name(engine)).items():
        companies.setdefault(sym, Company(cik, sym, title))
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
            *_DEBT_EXTRA,
            _CASH,
            _CASH_WIDE,
            _CASH_PLAIN,
            *_SECURITIES,
            _RECEIVABLES,
        )
    }
    shares = await _instant(sec, "EntityCommonStockSharesOutstanding", periods, dei=True)
    diluted = [
        await sec.frame("us-gaap", _DILUTED_SHARES, "shares", p.removesuffix("I")) for p in periods
    ]
    balance_shares = [await sec.frame("us-gaap", _BALANCE_SHARES, "shares", p) for p in periods]
    annual = {c: await _annual(sec, c, as_of) for c in (*_INTEREST, *_REVENUE)}
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                """
                SELECT DISTINCT ON (symbol) symbol, day, close FROM daily_bars
                WHERE adjustment = 'raw' AND symbol = ANY(:s) AND day <= :as_of
                ORDER BY symbol, day DESC
                """
            ),
            {"s": [s.upper() for s in symbols], "as_of": as_of},
        )
        priced = {r.symbol: (r.day, float(r.close)) for r in rows}
        prices = {sym: close for sym, (_, close) in priced.items()}
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

    out: list[Fundamentals] = []
    meta: dict[str, tuple[int | None, str]] = {}
    titles: dict[str, str] = {}
    failed: list[str] = []
    for symbol in symbols:
        sym = symbol.upper()
        company = companies.get(sym) or companies.get(sym.replace(".", "-"))
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

        noncurrent = v(_DEBT_PARTS[0])
        candidates = (
            v(_DEBT_TOTAL),
            _sum(noncurrent, v(_DEBT_PARTS[1])),
            v(_DEBT_WITH_LEASES_TOTAL),
            _sum(v(_DEBT_WITH_LEASES), v(_DEBT_WITH_LEASES_CURRENT)),
            _sum(noncurrent if noncurrent is not None else v(_DEBT_WITH_LEASES), v(_DEBT_CURRENT)),
        )
        extras = [v(e) for e in _DEBT_EXTRA]
        core = max((x for x in candidates if x is not None), default=None)
        cash = v(_CASH)
        if cash is None:
            cash = v(_CASH_WIDE)
        if cash is None:
            cash = v(_CASH_PLAIN)
        securities = max((x for x in (v(s) for s in _SECURITIES) if x is not None), default=0.0)
        files_xbrl = cash is not None or _newest(shares, cik) is not None
        debt: float | None
        if core is None and all(e is None for e in extras):
            debt = 0.0 if files_xbrl else None  # tags no debt at all: debt-free
        else:
            debt = (core or 0.0) + sum(e for e in extras if e is not None)
        interest = next(
            (x for x in (_newest(annual[c], cik) for c in _INTEREST) if x is not None), None
        )
        revenue = next(
            (x for x in (_newest(annual[c], cik) for c in _REVENUE) if x is not None), None
        )
        out.append(
            Fundamentals(
                symbol=sym,
                sic=sic,
                # The cover page's count, else the diluted average, else the
                # balance sheet's (HSY tags the cover page per share class,
                # which frames omit; the balance-sheet count is its common
                # stock alone, so the market cap errs small: ratios fail closed).
                shares_outstanding=share_count(_newest(shares, cik), _newest(diluted, cik))
                or _newest(balance_shares, cik),
                price=prices.get(sym),
                interest_bearing_debt=debt,
                cash_and_securities=(cash + securities) if cash is not None else None,
                interest_income=interest,
                revenue=revenue,
                average_price=averages.get(sym),
                foreign_filer=foreign,
                receivables=v(_RECEIVABLES),
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
    return out, meta, titles


async def run_screen(
    sec: SecClient, engine: AsyncEngine, symbols: Sequence[str], as_of: date
) -> list[ScreenResult]:
    """Screen ``symbols`` as of ``as_of`` and store every verdict with its inputs."""
    from halal_trader.compliance.index_veto import apply_veto, views_at

    fundamentals, meta, titles = await gather(sec, engine, symbols, as_of)
    results = apply_veto([screen(f) for f in fundamentals], titles, await views_at(engine, as_of))
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
                    ON CONFLICT (as_of, symbol) DO UPDATE SET cik = EXCLUDED.cik,
                        sic_description = EXCLUDED.sic_description, verdict = EXCLUDED.verdict,
                        reasons = EXCLUDED.reasons, metrics = EXCLUDED.metrics,
                        method = EXCLUDED.method, screened_at = EXCLUDED.screened_at
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
