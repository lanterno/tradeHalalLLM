"""The index veto counts today's ticker for today's company only from the day it took it,
and a ticker recovered from a CUSIP for today's company at any date, never as an old ticker."""

from __future__ import annotations

from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.aaoifi import ScreenResult
from halal_trader.compliance.index_veto import (
    IndexView,
    apply_veto,
    held,
    held_as,
    held_by_ticker,
    require_board,
    views_at,
)

# 40 held names from 10B to 49B: the size range starts near 17.8B.
HELD = {f"H{i}": 10e9 + i * 1e9 for i in range(40)}
TITLES = {
    "IR": "Ingersoll Rand Inc.",  # Gardner Denver until 2020-02-28
    "TT": "Trane Technologies plc",  # Ingersoll-Rand plc, IR until 2020-02-28
    "META": "Meta Platforms, Inc.",
    "RVTY": "Revvity, Inc.",
}


def _spus(
    filed: date, period_end: date | None, *tickers: str, recovered: tuple[str, ...] = ()
) -> IndexView:
    return IndexView(
        "SPUS",
        filed,
        frozenset({*HELD, *tickers}),
        frozenset(),
        period_end,
        recovered=frozenset(recovered),
    )


def _verdicts(view: IndexView, **caps: float) -> dict[str, str]:
    results = [ScreenResult(s, "halal", [], {"market_cap": c}) for s, c in {**HELD, **caps}.items()]
    return {r.symbol: r.verdict for r in apply_veto(results, TITLES, [view]) if r.symbol in caps}


def test_ir_counts_for_gardner_denver_only_from_the_day_it_took_it() -> None:
    # Ingersoll-Rand plc traded as IR through 2020-02-28 and became Trane (TT);
    # Gardner Denver took IR at that close.
    before = _spus(date(2020, 4, 28), date(2020, 2, 28), "IR")
    after = _spus(date(2020, 4, 28), date(2020, 2, 29), "IR")
    assert not held_by_ticker(before, "IR")
    assert held_as(before, "TT") == "IR"
    assert _verdicts(before, IR=20e9, TT=30e9) == {"IR": "not_halal", "TT": "halal"}
    assert held_by_ticker(after, "IR")
    assert held_as(after, "TT") is None
    assert _verdicts(after, IR=20e9, TT=30e9) == {"IR": "halal", "TT": "not_halal"}


def test_the_board_rule_dates_todays_ticker_too() -> None:
    result = ScreenResult("IR", "halal", [], {"market_cap": 20e9})
    before = _spus(date(2020, 4, 28), date(2020, 2, 28), "IR")
    after = _spus(date(2020, 4, 28), date(2020, 2, 29), "IR")
    (r,) = require_board([result], {"IR": 5812}, TITLES, [before])
    assert r.verdict == "doubtful"
    (r,) = require_board([result], {"IR": 5812}, TITLES, [after])
    assert r.verdict == "halal"


def test_an_undated_view_dates_todays_ticker_by_its_filing() -> None:
    # Its holdings precede the filing: a filing before Gardner Denver took IR
    # cannot be holding it as IR.
    filed_before = IndexView("SPUS", date(2019, 11, 27), frozenset({"IR", "AAPL"}), frozenset())
    filed_after = IndexView("SPUS", date(2020, 7, 28), frozenset({"IR"}), frozenset())
    assert not held(filed_before, "IR", TITLES["IR"])
    assert held(filed_after, "IR", TITLES["IR"])
    assert held(filed_before, "AAPL", "")  # a ticker no other company held counts at any date


def test_a_reported_ticker_before_its_company_took_it_is_not_that_company() -> None:
    # In 2021 Meta traded as FB and Revvity as PKI: a filing reporting META or
    # RVTY then is not reporting them.
    spus = _spus(date(2021, 7, 28), date(2021, 5, 31), "META", "RVTY")
    assert not held_by_ticker(spus, "META") and not held_by_ticker(spus, "RVTY")
    assert _verdicts(spus, META=900e9, RVTY=25e9) == {"META": "not_halal", "RVTY": "not_halal"}


def test_a_ticker_recovered_from_a_cusip_counts_for_todays_company_at_any_date() -> None:
    # SPUS's filings of 2021 list Facebook and PerkinElmer without a ticker;
    # their CUSIPs' newest tickers are META (from 2022-06-09) and RVTY (from
    # 2023-05-16).
    spus = _spus(date(2021, 7, 28), date(2021, 5, 31), recovered=("META", "RVTY"))
    assert held_by_ticker(spus, "META") and held_by_ticker(spus, "RVTY")
    assert _verdicts(spus, META=900e9, RVTY=25e9) == {"META": "halal", "RVTY": "halal"}
    undated = _spus(date(2021, 7, 28), None, recovered=("META",))
    assert held_by_ticker(undated, "META")


def test_a_recovered_ticker_is_never_an_old_one() -> None:
    # Gardner Denver's CUSIP, recovered as IR in holdings of IR's last day as
    # Ingersoll-Rand plc's: the security is Gardner Denver's, not Trane's.
    spus = _spus(date(2020, 4, 28), date(2020, 2, 28), recovered=("IR",))
    assert held_as(spus, "TT") is None
    assert held(spus, "IR", TITLES["IR"])
    assert _verdicts(spus, IR=20e9, TT=30e9) == {"IR": "halal", "TT": "not_halal"}


async def test_views_keep_the_tickers_recovered_from_cusips_apart(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO etf_holdings (etf, filed, period_end, ticker, name, cusip, "
                "weight_pct) VALUES (:e, :f, :p, :t, :n, :c, 1.0)"
            ),
            [
                # SPUS's 2021 shape: Facebook and PerkinElmer without a ticker.
                {
                    "e": "SPUS",
                    "f": date(2021, 7, 28),
                    "p": date(2021, 5, 31),
                    "t": None,
                    "n": "Facebook Inc",
                    "c": "30303M102",
                },
                {
                    "e": "SPUS",
                    "f": date(2021, 7, 28),
                    "p": date(2021, 5, 31),
                    "t": None,
                    "n": "PerkinElmer Inc",
                    "c": "714046109",
                },
                {
                    "e": "SPUS",
                    "f": date(2021, 7, 28),
                    "p": date(2021, 5, 31),
                    "t": "IR",
                    "n": "Ingersoll Rand Inc",
                    "c": "45687V106",
                },
                # HLAL's later filings carry the tickers.
                {
                    "e": "HLAL",
                    "f": date(2022, 10, 24),
                    "p": date(2022, 8, 31),
                    "t": "META",
                    "n": "Meta Platforms Inc",
                    "c": "30303M102",
                },
                {
                    "e": "HLAL",
                    "f": date(2023, 7, 27),
                    "p": date(2023, 5, 31),
                    "t": "RVTY",
                    "n": "Revvity Inc",
                    "c": "714046109",
                },
            ],
        )
    (spus,) = await views_at(engine, date(2021, 9, 30))
    assert spus.tickers == frozenset({"IR"})
    assert spus.recovered == frozenset({"META", "RVTY"})
    assert held(spus, "META", TITLES["META"]) and held(spus, "RVTY", TITLES["RVTY"])
    assert held(spus, "IR", TITLES["IR"])  # reported, and Gardner Denver's since 2020-02-29


async def _holdings(
    engine: AsyncEngine, *rows: tuple[str, date, date, str | None, str, str]
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO etf_holdings (etf, filed, period_end, ticker, name, cusip, "
                "weight_pct) VALUES (:e, :f, :p, :t, :n, :c, 1.0)"
            ),
            [dict(zip("efptnc", r, strict=True)) for r in rows],
        )


async def test_a_recovered_ticker_names_its_company_as_of_the_filing_it_came_from(
    engine: AsyncEngine,
) -> None:
    spus_2021 = ("SPUS", date(2021, 10, 28), date(2021, 8, 31))
    hlal_2020 = ("HLAL", date(2020, 7, 7), date(2020, 5, 31))
    await _holdings(
        engine,
        # SPUS's 2021 lines without a ticker.
        (*spus_2021, None, "Fortune Brands Home & Security Inc", "34964C106"),
        (*spus_2021, None, "Gap Inc/The", "364760108"),
        (*spus_2021, None, "IAC/InterActiveCorp", "44919P508"),
        (*spus_2021, None, "Trane Technologies PLC", "G47791101"),
        # The newest lines that carry a ticker for those CUSIPs. Fortune Brands
        # (FBIN from 2022-12-15) and Gap (GAP from 2024-08-22) were last
        # reported under their old tickers.
        ("HLAL", date(2022, 10, 24), date(2022, 8, 31), "FBHS", "Fortune Brands", "34964C106"),
        (*hlal_2020, "GPS", "Gap Inc/The", "364760108"),
        # IAC was the old IAC (Match Group today) until 2020-06-30.
        (*hlal_2020, "IAC", "IAC/InterActiveCorp", "44919P508"),
        # Ingersoll-Rand plc, Trane today, as a filing of 2019 would carry it.
        ("HLAL", date(2020, 1, 27), date(2019, 11, 30), "IR", "Ingersoll-Rand PLC", "G47791101"),
    )
    (spus,) = await views_at(engine, date(2021, 12, 31))
    assert spus.tickers == frozenset()
    assert spus.recovered == frozenset({"FBIN", "GAP", "TT"})  # the old IAC is nobody today
    assert held(spus, "FBIN", "Fortune Brands Innovations, Inc.")
    assert held_by_ticker(spus, "TT") and not held(spus, "IR", TITLES["IR"])
    assert not held(spus, "PPLI", "People Inc.")
