"""SEC CIKs for tickers SEC's current ticker file no longer lists.

The screen maps ticker -> CIK through ``company_tickers.json``, which only
knows today's listings. A company that has since delisted is missing from
it, screens ``doubtful`` in every past quarter, and is never eligible in a
backtest: survivorship bias. This module recovers the mapping by name:

* the company's name comes from Alpaca's asset record for the ticker;
* candidate filers are every CIK that reported total assets in an XBRL
  frame since 2009, under its current name or any former one (EDGAR's
  cik-lookup-data.txt), so a company renamed since still matches;
* names are compared after normalising case, punctuation and corporate
  suffixes, and a ticker is mapped only when exactly one filer matches.

Every ticker's outcome is stored (``ticker_ciks``), so what stays unmapped
is counted rather than hidden: ``fund`` (an ETF or fund, never a company),
``no_name`` (Alpaca has no name for it), ``ambiguous``, ``no_match``.

Some delisted tickers name no company (Alpaca keeps no name for them) or
match several filers; ``HAND_CIKS`` maps those that matter by hand, each to
the filer that held the ticker over 2016-2024, checked against SEC's filer
names on 2026-10-10. A hand CIK is used only if it is an XBRL filer.

A mapped CIK is the company that last held the ticker. If a different
company held it earlier, that company's quarters have no facts under the
mapped CIK and still screen ``doubtful``: the screen fails closed.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.runner import UNMAPPED
from halal_trader.compliance.sec import SecClient
from halal_trader.data.alpaca_market import Asset

logger = logging.getLogger(__name__)

FIRST_XBRL_YEAR = 2009

# Ticker -> CIK for liquid delisted names the name match cannot resolve
# (no Alpaca name, or several filers under one name). Mostly 2016-2019
# acquisitions: leaving them out drops companies that did not survive.
HAND_CIKS: dict[str, int] = {
    "AABA": 1011006,  # Altaba (formerly Yahoo)
    "AGN": 1578845,  # Allergan plc
    "ANDV": 50104,  # Andeavor
    "ARRS": 1645494,  # ARRIS International plc
    "AZPN": 929940,  # Aspen Technology (before the 2022 Emerson deal)
    "BMS": 11199,  # Bemis
    "BPL": 805022,  # Buckeye Partners
    "CADE": 1614184,  # Cadence Bancorporation
    "CAVM": 1175609,  # Cavium
    "CLNS": 1679688,  # Colony NorthStar / Colony Capital
    "CONE": 1553023,  # CyrusOne
    "DATA": 1303652,  # Tableau Software
    "DNB": 1115222,  # Dun & Bradstreet (taken private 2019)
    "DVMT": 1571996,  # Dell Technologies class V
    "DWDP": 1666700,  # DowDuPont
    "ECYT": 1235007,  # Endocyte
    "EEP": 880285,  # Enbridge Energy Partners
    "ELLI": 1122388,  # Ellie Mae
    "ESL": 33619,  # Esterline Technologies
    "ESRX": 1532063,  # Express Scripts Holding
    "EVHC": 1678531,  # Envision Healthcare (after the 2016 AmSurg merger)
    "FDC": 883980,  # First Data
    "FNSR": 1094739,  # Finisar
    "GGP": 1496048,  # GGP
    "GHDX": 1131324,  # Genomic Health
    "HDS": 1573097,  # HD Supply Holdings
    "IDTI": 703361,  # Integrated Device Technology
    "ILG": 1434620,  # ILG
    "IMPV": 1364962,  # Imperva
    "ISBC": 1594012,  # Investors Bancorp
    "KLXI": 1617898,  # KLX
    "LOXO": 1581720,  # Loxo Oncology
    "MBFI": 1139812,  # MB Financial
    "MBT": 1115837,  # Mobile TeleSystems
    "MDSO": 1453814,  # Medidata Solutions
    "MGP": 1656936,  # MGM Growth Properties
    "MYL": 1623613,  # Mylan N.V.
    "OAK": 1403528,  # Oaktree Capital Group
    "OPHT": 1410939,  # Ophthotech
    "PF": 1564822,  # Pinnacle Foods
    "PNK": 1656239,  # Pinnacle Entertainment (after the 2016 spin-off)
    "RDC": 85408,  # Rowan Companies
    "REN": 1469510,  # Resolute Energy
    "RHT": 1087423,  # Red Hat
    "SCG": 754737,  # SCANA
    "SFLY": 1125920,  # Shutterfly
    "SHPG": 936402,  # Shire
    "STAY": 1581164,  # Extended Stay America
    "STL": 1070154,  # Sterling Bancorp
    "TCF": 814184,  # TCF Financial (before the 2019 Chemical merger)
    "TRCO": 726513,  # Tribune Media
    "TSRO": 1491576,  # Tesaro
    "TSS": 721683,  # Total System Services
    "TVPT": 1424755,  # Travelport Worldwide
    "VIAB": 1339947,  # Viacom
    "VSM": 1660690,  # Versum Materials
    "VVC": 1096385,  # Vectren
    "WMGI": 1492658,  # Wright Medical Group N.V.
    "WP": 1533932,  # Worldpay
    "ZAYO": 1608249,  # Zayo Group Holdings
}

_SUFFIXES = re.compile(
    r"\b(common|ordinary|stock|shares?|class [a-c]|series [a-c]|new|the|inc|incorporated|"
    r"corp|corporation|co|company|ltd|limited|plc|llc|lp|l p|holdings?|group|n v|nv|s a|sa|"
    r"ag|de|adr|american depositary|depositary|units?|representing)\b"
)
_FUND = re.compile(
    r"\b(etf|etn|fund|ishares|proshares|direxion|spdr|2x|3x|leveraged|inverse)\b",
    re.IGNORECASE,
)


def normalize(name: str) -> str:
    """Comparable form of a company name: lower case, no punctuation or corporate suffixes."""
    s = name.lower().replace("&", " and ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for _ in range(3):  # "Holdings, Inc. Class A Common Stock" peels in layers
        s = re.sub(r"\s+", " ", _SUFFIXES.sub(" ", s)).strip()
    return s


def is_fund(name: str) -> bool:
    return bool(_FUND.search(name))


@dataclass(frozen=True, slots=True)
class Match:
    symbol: str
    status: str  # mapped | fund | no_name | ambiguous | no_match
    cik: int | None = None
    asset_name: str | None = None
    filer_name: str | None = None


def match_symbols(
    symbols: Iterable[str],
    asset_names: dict[str, str],
    filers: dict[int, str],
    historical: Iterable[tuple[str, int]],
) -> list[Match]:
    """Match each symbol's asset name to exactly one XBRL filer, by any of its names."""
    by_name: dict[str, set[int]] = defaultdict(set)
    for cik, name in filers.items():
        by_name[normalize(name)].add(cik)
    for name, cik in historical:
        if cik in filers:  # a subsidiary or fund that never filed financials is no candidate
            by_name[normalize(name)].add(cik)
    by_name.pop("", None)

    out: list[Match] = []
    for symbol in sorted(set(symbols)):
        name = asset_names.get(symbol, "").strip()
        hand = HAND_CIKS.get(symbol)
        if hand is not None and hand in filers:
            out.append(Match(symbol, "mapped", hand, name or None, filers[hand]))
            continue
        key = normalize(name)
        ciks = by_name.get(key, set()) if key else set()
        if len(ciks) == 1:
            cik = next(iter(ciks))
            out.append(Match(symbol, "mapped", cik, name, filers[cik]))
        elif not key:
            out.append(Match(symbol, "no_name", asset_name=name or None))
        elif is_fund(name):  # only after the match: "Invesco Ltd." is a company
            out.append(Match(symbol, "fund", asset_name=name))
        else:
            out.append(Match(symbol, "ambiguous" if ciks else "no_match", asset_name=name))
    return out


def frame_periods(today: date) -> list[str]:
    """Every quarter-end instant frame from 2009 to the last complete quarter."""
    last = (today.month - 1) // 3  # quarters complete this year
    return [
        f"CY{y}Q{q}I"
        for y in range(FIRST_XBRL_YEAR, today.year + 1)
        for q in (1, 2, 3, 4)
        if y < today.year or q <= last
    ]


async def unmapped_symbols(engine: AsyncEngine) -> list[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT DISTINCT symbol FROM halal_screen_current WHERE sic_description = :u"),
            {"u": UNMAPPED},
        )
        return sorted(r.symbol for r in rows)


async def store_matches(engine: AsyncEngine, matches: Sequence[Match]) -> None:
    async with engine.begin() as conn:
        for m in matches:
            await conn.execute(
                text(
                    "INSERT INTO ticker_ciks (symbol, status, cik, asset_name, filer_name, "
                    "matched_at) VALUES (:s, :st, :c, :an, :fn, now()) "
                    "ON CONFLICT (symbol) DO UPDATE SET status = EXCLUDED.status, "
                    "cik = EXCLUDED.cik, asset_name = EXCLUDED.asset_name, "
                    "filer_name = EXCLUDED.filer_name, matched_at = EXCLUDED.matched_at"
                ),
                {
                    "s": m.symbol,
                    "st": m.status,
                    "c": m.cik,
                    "an": m.asset_name,
                    "fn": m.filer_name,
                },
            )


async def mapped_ciks(engine: AsyncEngine) -> dict[str, tuple[int, str]]:
    """symbol -> (CIK, filer name) for every ticker mapped by name."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT symbol, cik, filer_name FROM ticker_ciks WHERE status = 'mapped'")
        )
        return {r.symbol: (int(r.cik), str(r.filer_name or "")) for r in rows}


async def fund_symbols(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(text("SELECT symbol FROM ticker_ciks WHERE status = 'fund'"))
        return {r.symbol for r in rows}


async def map_unmapped(
    sec: SecClient, engine: AsyncEngine, assets: Sequence[Asset], *, today: date
) -> list[Match]:
    """Match every symbol a screen could not map, and store each outcome."""
    symbols = await unmapped_symbols(engine)
    if not symbols:
        return []
    # A ticker can have several Alpaca records (relisted, reused); the
    # active one names its current holder, which is what SEC lists too.
    names: dict[str, str] = {}
    for a in sorted(assets, key=lambda a: a.status == "active"):
        if a.name:
            names[a.symbol] = a.name
    filers: dict[int, str] = {}
    for period in frame_periods(today):
        filers.update(await sec.filers(period))
    logger.info("delisted: %d XBRL filers since %d", len(filers), FIRST_XBRL_YEAR)
    matches = match_symbols(symbols, names, filers, await sec.historical_names())
    await store_matches(engine, matches)
    return matches


async def rescreen_mapped(sec: SecClient, engine: AsyncEngine) -> dict[date, int]:
    """Re-screen every stored quarter for the newly mapped tickers screened as unmapped.

    The date's other stored rows size the index veto (``runner.stored_peers``):
    a few names alone price too few held ones and would never be vetoed.
    """
    from halal_trader.compliance.runner import company_map, run_screen, stored_peers

    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT r.as_of, array_agg(r.symbol) AS symbols FROM halal_screen_current r "
                "JOIN ticker_ciks t ON t.symbol = r.symbol AND t.status = 'mapped' "
                "WHERE r.sic_description = :u GROUP BY r.as_of ORDER BY r.as_of"
            ),
            {"u": UNMAPPED},
        )
        todo = [(r.as_of, list(r.symbols)) for r in rows]
    companies = await company_map(sec, engine) if todo else {}
    out: dict[date, int] = {}
    for as_of, symbols in todo:
        peers = await stored_peers(engine, companies, as_of, symbols)
        results = await run_screen(sec, engine, symbols, as_of, peers=peers)
        out[as_of] = sum(1 for r in results if r.verdict == "halal")
        logger.info("rescreen %s: %d mapped, %d halal", as_of, len(symbols), out[as_of])
    return out
