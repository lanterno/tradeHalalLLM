"""The strict option: S&P activity exclusions, receivables, and the index veto."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.aaoifi import Fundamentals, ScreenResult, screen
from halal_trader.compliance.index_veto import IndexView, apply_veto, views_at


def _f(**kw: object) -> Fundamentals:
    base: dict[str, object] = dict(
        symbol="CO",
        sic=3571,
        shares_outstanding=1_000.0,
        price=100.0,
        interest_bearing_debt=10_000.0,
        cash_and_securities=20_000.0,
        interest_income=100.0,
        revenue=50_000.0,
    )
    base.update(kw)
    return Fundamentals(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("sic", "fragment"),
    [
        (7311, "advertising"),
        (4841, "broadcasting"),
        (7812, "motion pictures"),
        (7841, "streaming"),
        (3652, "music"),
        (7990, "gaming"),
    ],
)
def test_s_and_p_activity_exclusions_apply(sic: int, fragment: str) -> None:
    result = screen(_f(sic=sic))
    assert result.verdict == "not_halal"
    assert fragment in result.reasons[0]


def test_receivables_above_49_percent_of_market_cap_fail_and_unreported_count_as_none() -> None:
    assert screen(_f(receivables=50_000.0)).verdict == "not_halal"
    assert screen(_f(receivables=40_000.0)).verdict == "halal"
    assert screen(_f()).verdict == "halal"


def _passes(caps: dict[str, float]) -> list[ScreenResult]:
    return [ScreenResult(s, "halal", [], {"market_cap": c}) for s, c in caps.items()]


def test_the_veto_fails_a_large_pass_the_index_left_out_and_nothing_else() -> None:
    held = {f"H{i}": 10e9 + i * 1e9 for i in range(40)}  # the ETF's size range starts near 18B
    results = _passes({**held, "META": 1_500e9, "SMALL": 2e9, "GOOG": 2_000e9})
    titles = {"GOOG": "Alphabet Inc.", "META": "Meta Platforms, Inc."}
    view = IndexView(
        "SPUS",
        date(2025, 11, 25),
        frozenset(held),
        frozenset({"alphabet"}),  # held as GOOGL, matched by name
    )
    by = {r.symbol: r for r in apply_veto(results, titles, [view])}
    assert by["META"].verdict == "not_halal"
    assert "SPUS" in by["META"].reasons[-1]
    assert by["GOOG"].verdict == "halal"  # same company as the held class
    assert by["SMALL"].verdict == "halal"  # below the index's size range: no opinion
    assert by["H3"].verdict == "halal"


def test_the_veto_needs_enough_held_names_priced_to_know_the_size_range() -> None:
    results = _passes({"A": 10e9, "BIG": 900e9})
    view = IndexView("SPUS", date(2025, 11, 25), frozenset({"A"}), frozenset())
    assert [r.verdict for r in apply_veto(results, {}, [view])] == ["halal", "halal"]


async def test_holdings_count_from_their_filing_date_and_lapse(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO etf_holdings (etf, filed, period_end, ticker, name, cusip, "
                "weight_pct) VALUES (:e, :f, :p, :t, :n, 'x', 1.0)"
            ),
            [
                {
                    "e": "SPUS",
                    "f": date(2025, 1, 28),
                    "p": date(2024, 11, 30),
                    "t": "OLD",
                    "n": "Old Co",
                },
                {
                    "e": "SPUS",
                    "f": date(2025, 4, 28),
                    "p": date(2025, 2, 28),
                    "t": "NEW",
                    "n": "Newco Corp",
                },
            ],
        )
    assert [v.tickers for v in await views_at(engine, date(2025, 4, 27))] == [frozenset({"OLD"})]
    (view,) = await views_at(engine, date(2025, 4, 28))
    assert view.tickers == frozenset({"NEW"}) and view.names == frozenset({"newco"})


@pytest.mark.parametrize(
    ("nport", "sec"),
    [
        ("Cisco Systems Inc/Delaware", "CISCO SYSTEMS, INC."),
        ("salesforce.com Inc", "Salesforce, Inc."),
        ("TJX Cos Inc/The", "TJX COMPANIES INC /DE/"),
        ("Lowe's Cos Inc", "LOWES COMPANIES INC"),
        ("Estee Lauder Cos Inc/The", "ESTEE LAUDER COMPANIES INC"),
        ("Coca-Cola Co/The", "COCA COLA CO"),
    ],
)
def test_nport_and_sec_names_of_one_company_match(nport: str, sec: str) -> None:
    from halal_trader.compliance.index_veto import name_key

    assert name_key(nport) == name_key(sec)


async def test_a_missing_ticker_is_recovered_from_the_same_cusip(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO etf_holdings (etf, filed, period_end, ticker, name, cusip, "
                "weight_pct) VALUES (:e, :f, '2025-01-31', :t, 'Some Name', :c, 1.0)"
            ),
            [
                {"e": "HLAL", "f": date(2025, 3, 1), "t": "CSCO", "c": "17275R102"},
                {"e": "SPUS", "f": date(2025, 3, 2), "t": None, "c": "17275R102"},
            ],
        )
    views = {v.etf: v for v in await views_at(engine, date(2025, 4, 1))}
    assert "CSCO" in views["SPUS"].tickers
    assert await views_at(engine, date(2026, 1, 1)) == []  # older than MAX_AGE: no opinion


async def test_nport_filings_come_from_the_series_feed() -> None:
    import httpx

    from halal_trader.compliance.etf_holdings import HALAL_ETFS, nport_filings
    from halal_trader.compliance.sec import SecClient

    feed = (
        "<feed><entry><filing-date>2026-07-29</filing-date><filing-href>"
        "https://www.sec.gov/Archives/edgar/data/1/0001/0001-26-1-index.htm</filing-href>"
        "</entry></feed>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert "CIK=S000067283" in str(request.url)
        return httpx.Response(200, text=feed)

    sec = SecClient(
        "test test@example.invalid",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        min_interval_s=0.0,
    )
    assert await nport_filings(sec, HALAL_ETFS[0]) == [
        (date(2026, 7, 29), "https://www.sec.gov/Archives/edgar/data/1/0001/primary_doc.xml")
    ]
