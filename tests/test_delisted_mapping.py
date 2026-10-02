"""Delisted tickers matched to their SEC filer by name, and the screen using the match."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.delisted import (
    Match,
    frame_periods,
    mapped_ciks,
    match_symbols,
    normalize,
    rescreen_mapped,
    store_matches,
)
from halal_trader.compliance.runner import UNMAPPED, run_screen
from halal_trader.compliance.sec import Company, Fact


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Celgene Corporation Common Stock", "celgene"),
        ("CELGENE CORP /DE/", "celgene"),
        ("Dunkin' Brands Group, Inc. Common Stock", "dunkin brands"),
        ("AK Steel Holding Corp.", "ak steel"),
        ("Discovery, Inc. Series C Common Stock", "discovery"),
        ("Johnson & Johnson", "johnson and johnson"),
        ("Inc.", ""),
        ("", ""),
    ],
)
def test_normalize_drops_case_punctuation_and_corporate_suffixes(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def _by_symbol(matches: list[Match]) -> dict[str, Match]:
    return {m.symbol: m for m in matches}


def test_matching_maps_only_a_unique_xbrl_filer_by_current_or_former_name() -> None:
    filers = {
        1: "CELGENE CORP /DE/",
        2: "Warner Bros. Discovery, Inc.",
        3: "ACME INC",
        4: "ACME CORP",
    }
    historical = [
        ("DISCOVERY, INC.", 2),  # a former name of filer 2
        ("CELGENE SUBSIDIARY LLC", 99),  # never filed financials: no candidate
        ("BETA HOLDINGS", 99),
    ]
    names = {
        "CELG": "Celgene Corporation Common Stock",
        "DISCA": "Discovery, Inc. Series A Common Stock",
        "ACME": "Acme Inc.",
        "BETA": "Beta Holdings, Inc.",
        "BLANK": "",
        "QQQX": "Some Nasdaq 100 ETF",
        "IVZ": "Invesco Ltd.",
    }
    filers[5] = "INVESCO LTD."
    by = _by_symbol(match_symbols([*names, "NONAME"], names, filers, historical))

    assert (by["CELG"].status, by["CELG"].cik) == ("mapped", 1)
    assert (by["DISCA"].status, by["DISCA"].cik) == ("mapped", 2)
    assert by["DISCA"].filer_name == "Warner Bros. Discovery, Inc."
    assert by["ACME"].status == "ambiguous"  # two filers normalise to "acme"
    assert by["BETA"].status == "no_match"  # only a non-filer carries the name
    assert by["BLANK"].status == "no_name"  # a blank name never matches anything
    assert by["NONAME"].status == "no_name"
    assert by["QQQX"].status == "fund"
    assert (by["IVZ"].status, by["IVZ"].cik) == ("mapped", 5)  # a company named like a fund house


def test_frame_periods_run_from_2009_to_the_last_complete_quarter() -> None:
    periods = frame_periods(date(2026, 10, 2))
    assert periods[0] == "CY2009Q1I"
    assert periods[-1] == "CY2026Q3I"
    assert len(periods) == 17 * 4 + 3


class GoneSec:
    """SEC as of today: GONE (CIK 50) has left company_tickers.json but filed for years."""

    async def companies(self) -> dict[str, Company]:
        return {}

    async def sic(self, cik: int) -> tuple[int | None, str]:
        return 3674, "Semiconductors"

    async def frame(self, taxonomy: str, concept: str, unit: str, period: str) -> dict[int, Fact]:
        end = date(2019, 9, 30)
        instant = period.endswith("I")
        table = {
            "EntityCommonStockSharesOutstanding": 1_000.0 if instant else None,
            "CashAndCashEquivalentsAtCarryingValue": 1_000.0 if instant else None,
            "Revenues": 50_000.0 if not instant else None,
        }
        val = table.get(concept)
        return {} if val is None else {50: Fact(val, end, "a")}


async def test_a_mapped_ticker_is_screened_under_its_filer_and_rescreened(
    engine: AsyncEngine,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                "volume, fetched_at) VALUES ('GONE', '2019-12-30', 'raw', 100, 100, 100, 100, "
                "1, now())"
            )
        )
    as_of = date(2019, 12, 31)
    sec = GoneSec()
    first = await run_screen(sec, engine, ["GONE"], as_of)  # type: ignore[arg-type]
    assert first[0].verdict == "doubtful"
    async with engine.connect() as conn:
        desc = (
            await conn.execute(text("SELECT sic_description FROM halal_screen_results"))
        ).scalar()
    assert desc == UNMAPPED

    await store_matches(engine, [Match("GONE", "mapped", 50, "Gone Inc.", "GONE INC")])
    assert await mapped_ciks(engine) == {"GONE": (50, "GONE INC")}

    assert await rescreen_mapped(sec, engine) == {as_of: 1}  # type: ignore[arg-type]
    async with engine.connect() as conn:
        row = (await conn.execute(text("SELECT cik, verdict FROM halal_screen_results"))).one()
    assert (row.cik, row.verdict) == (50, "halal")
    assert (
        await rescreen_mapped(sec, engine) == {}
    )  # nothing left screened as unmapped  # type: ignore[arg-type]
