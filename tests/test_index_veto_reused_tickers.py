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
