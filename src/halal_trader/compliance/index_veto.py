"""The index veto: an index Shariah board's exclusion fails our screen too.

The operator chose the strict option (2026-10-02). SPUS (S&P 500 Shariah)
and HLAL (FTSE USA Shariah) hold every company their Shariah board passes
within their parent index. A company our screen passes, large enough to
be in that parent index, that the ETF does not hold was excluded by that
board, often for an activity a SIC code cannot see (Meta's advertising,
Mastercard's payments, a retailer's alcohol sales). Strict means the
exclusion stands here as well.

* **Large enough** is the ETF's own size range: at or above the 20th
  percentile market cap of the names it holds that this screen run also
  priced. Below that the ETF has no opinion, and none is taken.
* **Point in time:** holdings count from the day their N-PORT was filed
  (when they became public) and lapse after ``MAX_AGE``. There are none
  before mid-2020, so earlier screens carry no veto; backtests before then
  are correspondingly less strict.
* **Matching** is by ticker, else by company name: an N-PORT line without
  a ticker must not get a held company vetoed. A missing ticker is first
  recovered from the same CUSIP in any other filing (CUSIPs identify the
  security; SPUS's older filings omit most tickers, HLAL's carry them).
  Names are compared after stripping fund-administrator styling that
  SEC's company names do not use ("Cisco Systems Inc/Delaware",
  "salesforce.com Inc", "TJX Cos Inc/The"); without that, SPUS's 2021
  holdings of Cisco, Salesforce, Lowe's, TJX and Estee Lauder read as
  exclusions.
* **Renamed companies** are screened under today's ticker, which a holding
  from before the change does not carry, and their SEC name has usually
  changed too: SPUS held Meta as FB and HLAL held Corpay as FLT, and both
  read as exclusions. A view also holds a company under an old ticker
  (``events.renames.TICKER_RENAMES``) when its holdings are dated within
  the days that ticker named the company (``renames.news_window``: up to
  the last session under it, plus the grace sessions, never past its
  handover). A reused ticker therefore never counts for its former owner
  after the switch: IR in a 2020 holding is Gardner Denver, not Trane.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.aaoifi import ScreenResult, mixed_activity
from halal_trader.events.renames import ticker_history

MAX_AGE = timedelta(days=200)
SIZE_PERCENTILE = 20.0
MIN_PRICED = 30  # held names this run must price before the size range means anything


def name_key(name: str) -> str:
    """A company name in a form N-PORT holdings and SEC company names share."""
    from halal_trader.compliance.delisted import normalize

    base = name.split("/")[0]  # "Cisco Systems Inc/Delaware", "TJX COMPANIES INC /DE/"
    base = re.sub(r"\.com\b", "", base, flags=re.IGNORECASE).replace("'", "").replace("’", "")
    return re.sub(r"\bcos\b", "companies", normalize(base))


@dataclass(frozen=True, slots=True)
class IndexView:
    etf: str
    filed: date
    tickers: frozenset[str]
    names: frozenset[str]
    # The day the holdings are as of (N-PORT's period end). Without it an
    # old ticker cannot be dated, and none counts.
    period_end: date | None = None


def held_as(view: IndexView, symbol: str) -> str | None:
    """The old ticker the view holds ``symbol``'s company under, if any.

    One of the company's former tickers is among the view's, and the
    holdings are dated within the days it named the company.
    """
    if view.period_end is None:
        return None
    history = ticker_history()
    for old in history.olds.get(symbol, ()):
        _, first, last = history.windows[old]
        if old in view.tickers and first <= view.period_end <= last:
            return old
    return None


def held(view: IndexView, symbol: str, title: str) -> bool:
    """Whether the view holds the company trading as ``symbol``, SEC name ``title``:
    by ticker, by an old ticker of its own day (``held_as``), or by name."""
    if symbol in view.tickers or held_as(view, symbol) is not None:
        return True
    key = name_key(title)
    return bool(key) and key in view.names


async def views_at(engine: AsyncEngine, as_of: date) -> list[IndexView]:
    """Each ETF's newest holdings filed on or before ``as_of`` and not older than MAX_AGE."""
    async with engine.connect() as conn:
        # CUSIP -> ticker from every filing that carries both (a CUSIP names
        # one security for life; only reuse after a delisting could mislead,
        # and then towards "held", which a veto treats as no exclusion).
        known = {
            r.cusip: str(r.ticker).upper()
            for r in await conn.execute(
                text(
                    "SELECT DISTINCT ON (cusip) cusip, ticker FROM etf_holdings "
                    "WHERE ticker IS NOT NULL AND cusip <> '' ORDER BY cusip, filed DESC"
                )
            )
        }
        rows = await conn.execute(
            text(
                "SELECT h.etf, h.filed, h.period_end, h.ticker, h.name, h.cusip "
                "FROM etf_holdings h "
                "JOIN (SELECT etf, max(filed) AS filed FROM etf_holdings "
                "WHERE filed <= :d AND filed > :oldest GROUP BY etf) l "
                "ON l.etf = h.etf AND l.filed = h.filed"
            ),
            {"d": as_of, "oldest": as_of - MAX_AGE},
        )
        grouped: dict[tuple[str, date], tuple[set[str], set[str], set[date]]] = {}
        for r in rows:
            tickers, names, periods = grouped.setdefault((r.etf, r.filed), (set(), set(), set()))
            ticker = r.ticker or known.get(r.cusip)
            if ticker:
                tickers.add(str(ticker).upper())
            if key := name_key(str(r.name)):
                names.add(key)
            periods.add(r.period_end)
    # One filing reports one period (true of every stored filing on 2026-10-11).
    return [
        IndexView(etf, filed, frozenset(t), frozenset(n), max(p))
        for (etf, filed), (t, n, p) in sorted(grouped.items())
    ]


def apply_veto(
    results: Sequence[ScreenResult], titles: Mapping[str, str], views: Sequence[IndexView]
) -> list[ScreenResult]:
    """Fail every pass an index excluded within its size range."""
    out = list(results)
    caps = {r.symbol: r.metrics.get("market_cap") for r in results}
    for view in views:
        priced = [c for s, c in caps.items() if c and held(view, s, titles.get(s, ""))]
        if len(priced) < MIN_PRICED:
            continue
        floor = float(np.percentile(priced, SIZE_PERCENTILE))
        for i, r in enumerate(out):
            cap = caps.get(r.symbol)
            if (
                r.verdict != "halal"
                or not cap
                or cap < floor
                or held(view, r.symbol, titles.get(r.symbol, ""))
            ):
                continue
            reason = (
                f"excluded by {view.etf}'s Shariah index (holdings filed {view.filed}) "
                f"although within its size range (market cap >= {floor / 1e9:.1f}B)"
            )
            out[i] = replace(r, verdict="not_halal", reasons=[*r.reasons, reason])
    return out


def require_board(
    results: Sequence[ScreenResult],
    sics: Mapping[str, int | None],
    titles: Mapping[str, str],
    views: Sequence[IndexView],
) -> list[ScreenResult]:
    """A pass in a mixed-activity sector stands only if a Shariah index holds it.

    In those sectors (aaoifi.MIXED_ACTIVITY_SIC) impermissible revenue, such
    as a restaurant's alcohol, is not in SEC data, so the ratios cannot rule
    it out. An index Shariah board reviewed the company's revenue; without
    its inclusion (by either ETF, at any size) the verdict is doubtful. With
    no index holdings on file for the date, every such pass is doubtful.
    """

    out = []
    for r in results:
        activity = mixed_activity(sics.get(r.symbol))
        title = titles.get(r.symbol, "")
        if (
            r.verdict == "halal"
            and activity is not None
            and not any(held(v, r.symbol, title) for v in views)
        ):
            reason = (
                f"business activity unverified: {activity}; impermissible revenue is not "
                "in SEC data and no Shariah index (SPUS, HLAL) holds the company"
            )
            r = replace(r, verdict="doubtful", reasons=[*r.reasons, reason])
        out.append(r)
    return out
