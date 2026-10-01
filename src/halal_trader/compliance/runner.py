"""Gather fundamentals from SEC and the price store, screen, persist.

One screening run = one ``as_of`` date. Results go to
``halal_screen_results`` keyed (as_of, symbol), so every run is kept: the
screening history is point-in-time, and a backtest can ask what the screen
said on a given day instead of using today's verdicts on yesterday's data.

Concept choices (each a judgment call, recorded here):

* interest-bearing debt = the larger of LongTermDebt and
  (LongTermDebtNoncurrent + LongTermDebtCurrent), plus ShortTermBorrowings,
  CommercialPaper and FinanceLeaseLiability. Operating leases are excluded
  (not interest-bearing borrowing). A company that files XBRL but tags none
  of these is treated as debt-free: such companies often tag nothing.
* cash and interest-bearing securities = CashAndCashEquivalentsAtCarryingValue
  plus the largest of ShortTermInvestments, MarketableSecuritiesCurrent and
  AvailableForSaleSecuritiesDebtSecuritiesCurrent (companies often tag the
  same holding under more than one, so they are not summed).
* interest income = InvestmentIncomeInterest, else InterestIncomeOther, else
  InvestmentIncomeInterestAndDividend; revenue = Revenues, else
  RevenueFromContractWithCustomerExcludingAssessedTax, else SalesRevenueNet.
  Both from the latest calendar-year frame that has them.
* price = the latest raw close in daily_bars (run `halal-trader data
  backfill` first); market cap = dei EntityCommonStockSharesOutstanding x
  price.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.aaoifi import Fundamentals, ScreenResult, screen
from halal_trader.compliance.sec import Fact, SecClient

logger = logging.getLogger(__name__)

_DEBT_TOTAL = "LongTermDebt"
_DEBT_PARTS = ("LongTermDebtNoncurrent", "LongTermDebtCurrent")
_DEBT_EXTRA = ("ShortTermBorrowings", "CommercialPaper", "FinanceLeaseLiability")
_CASH = "CashAndCashEquivalentsAtCarryingValue"
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
    "SalesRevenueNet",
)


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
    """The most recent value across frames (ordered newest period first)."""
    best: Fact | None = None
    for frame in frames:
        fact = frame.get(cik)
        if fact is not None and (best is None or fact.end > best.end):
            best = fact
    return best.val if best is not None else None


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


async def gather(
    sec: SecClient, engine: AsyncEngine, symbols: Sequence[str], as_of: date
) -> tuple[list[Fundamentals], dict[str, tuple[int | None, str]]]:
    """Fundamentals for each symbol, plus (cik, SIC description) for the record."""
    companies = await sec.companies()
    periods = recent_quarter_instants(as_of)
    frames = {
        c: await _instant(sec, c, periods)
        for c in (_DEBT_TOTAL, *_DEBT_PARTS, *_DEBT_EXTRA, _CASH, *_SECURITIES)
    }
    shares = await _instant(sec, "EntityCommonStockSharesOutstanding", periods, dei=True)
    annual = {c: await _annual(sec, c, as_of) for c in (*_INTEREST, *_REVENUE)}
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                """
                SELECT DISTINCT ON (symbol) symbol, close FROM daily_bars
                WHERE adjustment = 'raw' AND symbol = ANY(:s) AND day <= :as_of
                ORDER BY symbol, day DESC
                """
            ),
            {"s": [s.upper() for s in symbols], "as_of": as_of},
        )
        prices = {r.symbol: float(r.close) for r in rows}

    out: list[Fundamentals] = []
    meta: dict[str, tuple[int | None, str]] = {}
    for symbol in symbols:
        sym = symbol.upper()
        company = companies.get(sym) or companies.get(sym.replace(".", "-"))
        if company is None:
            out.append(Fundamentals(sym, None, None, prices.get(sym), None, None, None, None))
            meta[sym] = (None, "not an SEC registrant (or ticker not mapped)")
            continue
        cik = company.cik
        sic, sic_desc = await sec.sic(cik)
        meta[sym] = (cik, sic_desc)

        def v(concept: str, cik: int = cik) -> float | None:
            return _newest(frames[concept], cik)

        total = v(_DEBT_TOTAL)
        parts = [v(p) for p in _DEBT_PARTS]
        parts_sum = (
            sum(p for p in parts if p is not None) if any(p is not None for p in parts) else None
        )
        extras = [v(e) for e in _DEBT_EXTRA]
        core = max((x for x in (total, parts_sum) if x is not None), default=None)
        cash = v(_CASH)
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
                shares_outstanding=_newest(shares, cik),
                price=prices.get(sym),
                interest_bearing_debt=debt,
                cash_and_securities=(cash + securities) if cash is not None else None,
                interest_income=interest,
                revenue=revenue,
            )
        )
    return out, meta


async def run_screen(
    sec: SecClient, engine: AsyncEngine, symbols: Sequence[str], as_of: date
) -> list[ScreenResult]:
    """Screen ``symbols`` as of ``as_of`` and store every verdict with its inputs."""
    fundamentals, meta = await gather(sec, engine, symbols, as_of)
    results = [screen(f) for f in fundamentals]
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
                    "method": "aaoifi-sec-v1",
                },
            )
    counts = {
        v: sum(1 for r in results if r.verdict == v) for v in ("halal", "not_halal", "doubtful")
    }
    logger.info("halal screen %s: %s", as_of, counts)
    return results
