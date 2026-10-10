"""Phase A backfills: parsing each source, point-in-time stamps, resumability."""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data.alpaca_market import NewsArticle
from halal_trader.events.history import (
    backfill_news,
    eps_rows,
    filing_records,
    insider_records,
    mark_units,
    news_records,
    quarters,
)


def _article(n: int, symbols: tuple[str, ...], when: datetime) -> NewsArticle:
    return NewsArticle(n, f"h{n}", "s", f"https://x/{n}", "benzinga", symbols, when)


def test_news_keeps_one_event_per_covered_symbol() -> None:
    when = datetime(2024, 3, 5, 14, tzinfo=UTC)
    recs = news_records([_article(1, ("AAPL", "MSFT", "ZZZZ"), when)], {"AAPL", "MSFT"})
    assert [(r.symbol, r.source_id, r.published_at) for r in recs] == [
        ("AAPL", "alpaca:1", when),
        ("MSFT", "alpaca:1", when),
    ]


class _Market:
    def __init__(self) -> None:
        self.days: list[date] = []

    async def news(self, symbols, *, start, end, max_pages):
        self.days.append(start.date())
        return [_article(start.toordinal(), ("AAPL",), start)]


async def test_news_backfill_runs_newest_first_and_resumes(engine: AsyncEngine) -> None:
    market = _Market()
    n = await backfill_news(
        engine, market, symbols={"AAPL"}, start=date(2024, 1, 1), end=date(2024, 1, 3)
    )
    assert n == 3
    assert market.days == [date(2024, 1, 3), date(2024, 1, 2), date(2024, 1, 1)]
    again = _Market()
    assert (
        await backfill_news(
            engine, again, symbols={"AAPL"}, start=date(2023, 12, 31), end=date(2024, 1, 3)
        )
        == 1
    )
    assert again.days == [date(2023, 12, 31)]  # finished days are skipped


def test_filings_are_stamped_at_sec_acceptance_with_their_items() -> None:
    page = {
        "form": ["8-K", "4", "10-Q", "8-K"],
        "filingDate": ["2024-02-21", "2024-02-20", "2024-02-21", "2014-01-01"],
        "acceptanceDateTime": [
            "2024-02-21T21:20:05.000Z",
            "2024-02-20T21:00:00.000Z",
            "2024-02-21T21:30:00.000Z",
            "2014-01-01T12:00:00.000Z",
        ],
        "accessionNumber": ["a1", "a2", "a3", "a4"],
        "items": ["2.02,9.01", "", "", "8.01"],
        "reportDate": ["2024-02-21", "", "2024-01-28", ""],
        "primaryDocument": ["d.htm", "", "q.htm", ""],
    }
    recs = filing_records(page, "NVDA")
    assert [(r.kind, r.source_id) for r in recs] == [("8-k", "a1"), ("10-q", "a3")]
    assert recs[0].published_at == datetime(2024, 2, 21, 21, 20, 5, tzinfo=UTC)
    assert recs[0].payload["items"] == ["2.02", "9.01"]


def _zip(files: dict[str, list[list[str]]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, rows in files.items():
            zf.writestr(name, "\n".join("\t".join(r) for r in rows) + "\n")
    return buf.getvalue()


def test_insider_purchases_and_sales_of_covered_companies_from_the_data_set() -> None:
    blob = _zip(
        {
            "SUBMISSION.tsv": [
                ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "AFF10B5ONE"],
                ["acc1", "31-JAN-2024", "4", "0000320193", "0"],
                ["acc2", "31-JAN-2024", "4", "0000999999", "0"],  # not covered
                ["acc3", "01-FEB-2024", "4", "0000320193", "1"],
            ],
            "REPORTINGOWNER.tsv": [
                ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"],
                ["acc1", "0001", "Officer", "CEO"],
                ["acc3", "0002", "Director", ""],
            ],
            "NONDERIV_TRANS.tsv": [
                [
                    "ACCESSION_NUMBER",
                    "NONDERIV_TRANS_SK",
                    "TRANS_DATE",
                    "TRANS_CODE",
                    "TRANS_SHARES",
                    "TRANS_PRICEPERSHARE",
                    "SHRS_OWND_FOLWNG_TRANS",
                    "DIRECT_INDIRECT_OWNERSHIP",
                ],
                ["acc1", "1", "29-JAN-2024", "P", "1000", "10.5", "5000", "D"],
                ["acc1", "2", "29-JAN-2024", "A", "50", "0", "5050", "D"],  # a grant: ignored
                ["acc2", "3", "29-JAN-2024", "P", "1", "1", "1", "D"],
                ["acc3", "4", "30-JAN-2024", "S", "200", "20", "0", "I"],
            ],
        }
    )
    recs = insider_records(blob, {320193: "AAPL"})
    assert [(r.kind, r.symbol, r.source_id) for r in recs] == [
        ("insider_buy", "AAPL", "acc1:1"),
        ("insider_sell", "AAPL", "acc3:4"),
    ]
    buy, sell = recs
    assert buy.payload["value"] == 10_500.0 and buy.payload["titles"] == ["CEO"]
    # Filing date only: stamped 17:00 New York, so usable from the next session.
    assert buy.published_at == datetime(2024, 1, 31, 22, tzinfo=UTC)
    assert sell.payload["plan_10b5_1"] is True and sell.payload["direct"] is False


def test_eps_rows_keep_durations_with_their_filing_date() -> None:
    payload = {
        "units": {
            "USD/shares": [
                {
                    "start": "2024-01-29",
                    "end": "2024-04-28",
                    "val": 5.98,
                    "accn": "a",
                    "fy": 2025,
                    "fp": "Q1",
                    "form": "10-Q",
                    "filed": "2024-05-29",
                },
                {
                    "end": "2024-04-28",
                    "val": 1.0,
                    "accn": "b",
                    "form": "10-Q",
                    "filed": "2024-05-29",
                },
            ]
        }
    }
    (row,) = eps_rows(1045810, "EarningsPerShareDiluted", payload)
    assert row["filed"] == date(2024, 5, 29) and row["val"] == 5.98


def test_quarters_cover_the_window() -> None:
    assert quarters(date(2025, 11, 1), date(2026, 4, 2)) == ["2026q1", "2026q2"]


async def test_backfilled_events_are_queryable(engine: AsyncEngine) -> None:
    await backfill_news(
        engine, _Market(), symbols={"AAPL"}, start=date(2024, 1, 1), end=date(2024, 1, 1)
    )
    async with engine.connect() as conn:
        row = (await conn.execute(text("SELECT payload->>'backfill' AS b FROM events"))).one()
    assert row.b == "true"


async def _units(engine: AsyncEngine) -> dict[str, int]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT unit, items FROM backfill_progress WHERE task = 'test'")
        )
        return {r.unit: r.items for r in rows}


async def test_marks_on_an_open_connection_commit_and_roll_back_with_it(
    engine: AsyncEngine,
) -> None:
    async with engine.begin() as conn:
        await mark_units(object(), "test", {"a": 1, "b": 2}, conn=conn)  # type: ignore[arg-type]
        assert await _units(engine) == {}  # the caller's transaction, not committed yet
    assert await _units(engine) == {"a": 1, "b": 2}
    with pytest.raises(RuntimeError, match="the caller failed"):
        async with engine.begin() as conn:
            await mark_units(engine, "test", {"c": 3}, conn=conn)
            raise RuntimeError("the caller failed")
    assert await _units(engine) == {"a": 1, "b": 2}
    await mark_units(engine, "test", {"a": 5})  # a transaction of its own, upserting
    await mark_units(engine, "test", {})
    assert await _units(engine) == {"a": 5, "b": 2}
