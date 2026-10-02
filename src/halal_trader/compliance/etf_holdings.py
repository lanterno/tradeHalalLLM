"""Halal ETFs' holdings from their SEC N-PORT filings, to validate the screen.

SPUS (SP Funds, S&P 500 Shariah) and HLAL (Wahed, FTSE USA Shariah) are
each screened by their own Shariah board, and file complete holdings on
Form N-PORT, free on EDGAR. A trust files one N-PORT per fund series, so the
newest filing whose <seriesId> matches is the one we want.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.sec import SecClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HalalEtf:
    symbol: str
    cik: int
    series_id: str


# From sec.gov/files/company_tickers_mf.json (2026-10-01).
HALAL_ETFS: tuple[HalalEtf, ...] = (
    HalalEtf("SPUS", 1742912, "S000067283"),
    HalalEtf("HLAL", 1683471, "S000065986"),
)


@dataclass(frozen=True, slots=True)
class Holding:
    ticker: str | None
    name: str
    cusip: str
    weight_pct: float


@dataclass(frozen=True, slots=True)
class EtfHoldings:
    etf: str
    period_end: date | None
    holdings: list[Holding]

    @property
    def tickers(self) -> set[str]:
        return {h.ticker for h in self.holdings if h.ticker}


def parse_nport(xml_text: str) -> tuple[str | None, date | None, list[Holding]]:
    """(seriesId, reporting period end, common-equity holdings) from an N-PORT XML."""
    root = ET.fromstring(xml_text)
    series = root.findtext(".//{*}seriesId")
    end_txt = root.findtext(".//{*}repPdDate")
    period_end = date.fromisoformat(end_txt) if end_txt else None
    holdings = []
    for sec in root.iterfind(".//{*}invstOrSec"):
        if sec.findtext("{*}assetCat") != "EC":  # common equity only
            continue
        ticker_el = sec.find("{*}identifiers/{*}ticker")
        ticker = ticker_el.get("value") if ticker_el is not None else None
        holdings.append(
            Holding(
                ticker=ticker.upper().replace("/", ".") if ticker else None,
                name=sec.findtext("{*}title") or sec.findtext("{*}name") or "",
                cusip=sec.findtext("{*}cusip") or "",
                weight_pct=float(sec.findtext("{*}pctVal") or 0.0),
            )
        )
    return series, period_end, holdings


async def latest_holdings(sec: SecClient, etf: HalalEtf, *, max_filings: int = 60) -> EtfHoldings:
    """The newest N-PORT holdings for ``etf``'s series."""
    subs = await sec.submissions(etf.cik)
    if not subs:
        raise RuntimeError(f"no EDGAR submissions for {etf.symbol} (CIK {etf.cik})")
    recent = subs["filings"]["recent"]
    checked = 0
    for form, accession in zip(recent["form"], recent["accessionNumber"], strict=True):
        if not form.startswith("NPORT-P"):
            continue
        checked += 1
        if checked > max_filings:
            break
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{etf.cik}/"
            f"{accession.replace('-', '')}/primary_doc.xml"
        )
        xml_text = await sec.text(url)
        if not xml_text:
            continue
        series, period_end, holdings = parse_nport(xml_text)
        if series == etf.series_id:
            logger.info("%s: %d equity holdings as of %s", etf.symbol, len(holdings), period_end)
            return EtfHoldings(etf.symbol, period_end, holdings)
    raise RuntimeError(f"no N-PORT found for {etf.symbol} series {etf.series_id}")


_SERIES_FEED = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={series}"
    "&type=NPORT-P&dateb=&owner=include&count=100&output=atom"
)


async def nport_filings(sec: SecClient, etf: HalalEtf) -> list[tuple[date, str]]:
    """(filing date, primary_doc.xml URL) of every N-PORT the series has filed.

    EDGAR's per-series feed lists only this fund's filings, where the
    trust's own submissions mix in every other fund it runs.
    """
    feed = await sec.text(_SERIES_FEED.format(series=etf.series_id))
    if not feed:
        return []
    dates = re.findall(r"<filing-date>([^<]+)</filing-date>", feed)
    hrefs = re.findall(r"<filing-href>([^<]+)</filing-href>", feed)
    return [
        (date.fromisoformat(d), h.rsplit("/", 1)[0] + "/primary_doc.xml")
        for d, h in zip(dates, hrefs, strict=True)
    ]


async def sync_holdings(sec: SecClient, engine: AsyncEngine) -> int:
    """Store every N-PORT holdings filing of the halal ETFs not stored yet; returns filings."""
    async with engine.connect() as conn:
        rows = await conn.execute(text("SELECT DISTINCT etf, filed FROM etf_holdings"))
        done = {(r.etf, r.filed) for r in rows}
    stored = 0
    for etf in HALAL_ETFS:
        for filed, url in await nport_filings(sec, etf):
            if (etf.symbol, filed) in done:
                continue
            xml_text = await sec.text(url)
            if not xml_text:
                continue
            series, period_end, holdings = parse_nport(xml_text)
            if series != etf.series_id or period_end is None:
                continue
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO etf_holdings (etf, filed, period_end, ticker, name, "
                        "cusip, weight_pct) VALUES (:e, :f, :p, :t, :n, :c, :w)"
                    ),
                    [
                        {
                            "e": etf.symbol,
                            "f": filed,
                            "p": period_end,
                            "t": h.ticker,
                            "n": h.name,
                            "c": h.cusip,
                            "w": h.weight_pct,
                        }
                        for h in holdings
                    ],
                )
            stored += 1
            logger.info(
                "%s: N-PORT for %s (filed %s), %d holdings",
                etf.symbol,
                period_end,
                filed,
                len(holdings),
            )
    return stored
