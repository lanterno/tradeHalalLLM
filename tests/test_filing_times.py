"""Stored filings restamped from their EDGAR header (`events filings fix-times`), and new
ones stamped from it before they are stored (the evening filings refresh)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import pytest
from click.testing import CliRunner
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from halal_trader.cli import cli
from halal_trader.compliance.sec import SecClient, SecUnavailable
from halal_trader.events import daily, history
from halal_trader.events.store import EventRecord, EventRecorder
from halal_trader.market_hours import MARKET_TZ

UA = "test test@example.invalid"
AGENT = 1193125  # a filing agent: EDGAR has no folder for its clients' filings under it
AAPL, MSFT, NVDA = 320193, 789019, 1045810


def accession(filer: int, year: int, n: int) -> str:
    return f"{filer:010d}-{year % 100:02d}-{n:06d}"


def ny(y: int, mo: int, d: int, h: int, mi: int, s: int = 0) -> datetime:
    """A New York wall time, as UTC."""
    return datetime(y, mo, d, h, mi, s, tzinfo=MARKET_TZ).astimezone(UTC)


def header_page(when: datetime) -> str:
    stamp = when.astimezone(MARKET_TZ).strftime("%Y%m%d%H%M%S")
    return f"<HTML>\n<!--\n<SEC-HEADER>x.hdr.sgml\n<ACCEPTANCE-DATETIME>{stamp}\n<TYPE>8-K\n-->"


class Edgar:
    """A fake EDGAR: index headers under the CIKs given, submissions per CIK."""

    def __init__(self) -> None:
        self.headers: dict[tuple[int, str], datetime] = {}
        self.submissions: dict[int, dict[str, Any]] = {}
        self.down: set[str] = set()  # accessions whose header answers down_status
        self.down_status = 503  # EDGAR's outage; 403 is its rate block
        self.requests: list[str] = []

    def header(self, cik: int, acc: str, when: datetime) -> None:
        self.headers[(cik, acc)] = when

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if "/submissions/CIK" in url:
            cik = int(url.rsplit("CIK", 1)[1].removesuffix(".json"))
            sub = self.submissions.get(cik)
            return httpx.Response(200, json=sub) if sub else httpx.Response(404)
        # /Archives/edgar/data/{cik}/{nodash}/{acc}-index-headers.html
        parts = request.url.path.split("/")
        cik, acc = int(parts[4]), parts[6].removesuffix("-index-headers.html")
        if acc in self.down:
            return httpx.Response(self.down_status)
        when = self.headers.get((cik, acc))
        return httpx.Response(200, text=header_page(when)) if when else httpx.Response(404)

    def header_requests(self) -> list[str]:
        """(cik/accession) of each header request, in order."""
        out = []
        for url in self.requests:
            if "index-headers" in url:
                parts = url.split("/")  # https: '' host Archives edgar data cik nodash page
                out.append(f"{parts[6]}/{parts[8].removesuffix('-index-headers.html')}")
        return out

    def client(self) -> SecClient:
        return SecClient(
            UA,
            client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)),
            min_interval_s=0.0,
            retries=0,
        )


async def screen(engine: AsyncEngine, ciks: dict[str, int]) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) "
                "VALUES ('2024-06-28', :s, :c, '', 'halal', '[]', '{}', 't', now())"
            ),
            [{"s": s, "c": c} for s, c in ciks.items()],
        )


async def store(engine: AsyncEngine, rows: list[tuple[str, str, str, datetime, list[str]]]) -> None:
    """Filing rows as the backfill stores them: (accession, symbol, kind, time, items)."""
    await EventRecorder(engine, raise_errors=True).record(
        [
            EventRecord("sec", acc, kind, sym, at, at, {"form": kind.upper(), "items": items})
            for acc, sym, kind, at, items in rows
        ]
    )


async def stored(engine: AsyncEngine) -> dict[tuple[str, str], tuple[datetime, datetime, Any]]:
    """(accession, symbol) -> (published_at, seen_at, time_source)."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT source_id, symbol, published_at, seen_at, payload->>'time_source' AS ts "
                "FROM events WHERE source = 'sec'"
            )
        )
        return {(r.source_id, r.symbol): (r.published_at, r.seen_at, r.ts) for r in rows}


async def progress(engine: AsyncEngine, task: str) -> dict[str, int]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT unit, items FROM backfill_progress WHERE task = :t"), {"t": task}
        )
        return {r.unit: r.items for r in rows}


# ── ordering ──────────────────────────────────────────────────


def test_structural_items_come_first_then_results_then_deals_then_the_rest() -> None:
    p = history.filing_priority
    assert p("8-k", ["4.02", "9.01"]) == p("8-k/a", ["1.02"]) == 0
    assert p("8-k", ["2.02", "9.01"]) == 1
    assert p("8-k", ["2.02", "3.02"]) == 0  # a structural item wins
    assert p("8-k", ["5.02"]) == p("8-k", ["2.05", "7.01"]) == 2
    assert p("8-k", ["8.01", "9.01"]) == p("8-k", []) == 3
    assert p("10-q", ["2.02"]) == 4  # other forms after every 8-K


def test_a_record_without_a_header_keeps_the_json_time_and_says_so() -> None:
    at = datetime(2023, 11, 3, 0, 30, 32, tzinfo=UTC)
    rec = EventRecord("sec", accession(AAPL, 2023, 104), "8-k", "AAPL", at, at, {"items": []})
    json_kept = history.header_stamped(rec, None)
    assert json_kept.published_at == at and json_kept.payload["time_source"] == "json"
    true = ny(2023, 11, 2, 16, 30, 32)
    fixed = history.header_stamped(rec, true)
    assert fixed.published_at == fixed.seen_at == true
    assert fixed.payload == {"items": [], "time_source": "header"}
    assert rec.payload == {"items": []}  # the original is untouched
    assert history.time_delta(at, true) == 4 * 3600


# ── correct_filing_times ──────────────────────────────────────


async def test_stored_filings_take_the_header_time_and_the_delta_is_recorded(
    engine: AsyncEngine,
) -> None:
    edgar = Edgar()
    await screen(engine, {"AAPL": AAPL, "MSFT": MSFT})
    late = accession(AAPL, 2023, 104)  # summer: the JSON is 4 h late
    agent = accession(AGENT, 2024, 7)  # filed by an agent; winter: 5 h late
    right = accession(MSFT, 2024, 1)
    lost = accession(AGENT, 2024, 9)  # no header anywhere
    quarterly = accession(AAPL, 2023, 200)
    await store(
        engine,
        [
            (late, "AAPL", "8-k", ny(2023, 11, 2, 20, 30, 32), ["2.02", "9.01"]),
            (agent, "MSFT", "8-k/a", ny(2024, 1, 15, 13, 0), ["4.02"]),
            (right, "MSFT", "8-k", ny(2024, 1, 30, 16, 5), ["2.02"]),
            (lost, "MSFT", "8-k", ny(2024, 2, 1, 9, 0), ["8.01"]),
            (quarterly, "AAPL", "10-q", ny(2023, 11, 3, 1, 0), []),
        ],
    )
    edgar.header(AAPL, late, ny(2023, 11, 2, 16, 30, 32))
    edgar.header(MSFT, agent, ny(2024, 1, 15, 8, 0))
    edgar.header(MSFT, right, ny(2024, 1, 30, 16, 5))
    edgar.header(AAPL, quarterly, ny(2023, 11, 2, 21, 0))

    times = await history.correct_filing_times(
        engine, edgar.client(), start=date(2016, 1, 1), end=date(2026, 10, 10), concurrency=1
    )

    assert (times.checked, times.corrected, times.rows, times.missing) == (3, 2, 2, 1)
    assert times.deltas == {14400: 1, 18000: 1, 0: 1}
    rows = await stored(engine)
    assert rows[(late, "AAPL")] == (ny(2023, 11, 2, 16, 30, 32),) * 2 + ("header",)
    assert rows[(agent, "MSFT")] == (ny(2024, 1, 15, 8, 0),) * 2 + ("header",)
    assert rows[(right, "MSFT")] == (ny(2024, 1, 30, 16, 5),) * 2 + ("header",)
    assert rows[(lost, "MSFT")] == (ny(2024, 2, 1, 9, 0),) * 2 + (None,)
    assert rows[(quarterly, "AAPL")][2] is None  # 10-Qs are not in the default forms
    assert await progress(engine, history.TIMES_TASK) == {late: 14400, agent: 18000, right: 0}
    assert await progress(engine, history.TIMES_MISSING_TASK) == {lost: 0}
    # The company's CIK first; the agent's folder only when the company's has nothing.
    assert f"{MSFT}/{agent}" in edgar.header_requests()
    assert f"{AGENT}/{agent}" not in edgar.header_requests()
    assert edgar.header_requests()[-2:] == [f"{MSFT}/{lost}", f"{AGENT}/{lost}"]

    # A rerun reads nothing again and changes nothing.
    edgar.requests.clear()
    again = await history.correct_filing_times(
        engine, edgar.client(), start=date(2016, 1, 1), end=date(2026, 10, 10)
    )
    assert edgar.requests == []
    assert (again.checked, again.done_before) == (0, 4)
    assert await stored(engine) == rows


async def test_filings_go_in_priority_order_newest_first_and_stop_at_the_limit(
    engine: AsyncEngine,
) -> None:
    edgar = Edgar()
    filings = {
        "rest": (ny(2024, 3, 1, 8, 0), ["8.01"]),
        "earn_old": (ny(2023, 11, 2, 16, 30), ["2.02"]),
        "earn_new": (ny(2024, 2, 1, 16, 5), ["2.02", "9.01"]),
        "struct": (ny(2022, 5, 5, 7, 0), ["4.02"]),
        "mgmt": (ny(2024, 1, 10, 17, 0), ["5.02"]),
    }
    acc = {name: accession(NVDA, 2020, i) for i, name in enumerate(filings, 1)}
    await store(engine, [(acc[n], "NVDA", "8-k", at, items) for n, (at, items) in filings.items()])
    for n, (at, _) in filings.items():
        edgar.header(NVDA, acc[n], at)  # no screen row: the accession's filer is NVDA itself

    first = await history.correct_filing_times(
        engine,
        edgar.client(),
        start=date(2016, 1, 1),
        end=date(2026, 10, 10),
        limit=3,
        concurrency=1,
    )
    rest = await history.correct_filing_times(
        engine, edgar.client(), start=date(2016, 1, 1), end=date(2026, 10, 10), concurrency=1
    )

    order = ["struct", "earn_new", "earn_old", "mgmt", "rest"]
    assert edgar.header_requests() == [f"{NVDA}/{acc[n]}" for n in order]
    assert (first.checked, rest.checked, rest.done_before) == (3, 2, 3)


async def test_the_window_is_new_york_days_and_ticker_ciks_find_delisted_names(
    engine: AsyncEngine,
) -> None:
    edgar = Edgar()
    gone = 1_111_111
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO ticker_ciks (symbol, status, cik, matched_at) "
                "VALUES ('GONE', 'mapped', :c, now())"
            ),
            {"c": gone},
        )
    inside = accession(AGENT, 2023, 1)
    after = accession(AGENT, 2023, 2)
    await store(
        engine,
        [
            (inside, "GONE", "8-k", ny(2023, 12, 31, 23, 30), []),  # Jan 1 in UTC
            (after, "GONE", "8-k", ny(2024, 1, 1, 0, 30), []),
        ],
    )
    edgar.header(gone, inside, ny(2023, 12, 31, 18, 30))
    edgar.header(gone, after, ny(2023, 12, 31, 19, 30))

    times = await history.correct_filing_times(
        engine, edgar.client(), start=date(2023, 12, 1), end=date(2023, 12, 31)
    )

    assert (times.checked, times.missing) == (1, 0)
    assert edgar.header_requests() == [f"{gone}/{inside}"]
    assert (await stored(engine))[(inside, "GONE")][0] == ny(2023, 12, 31, 18, 30)


async def test_an_edgar_outage_keeps_what_was_read_and_a_rerun_resumes(
    engine: AsyncEngine,
) -> None:
    edgar = Edgar()
    accs = [accession(AAPL, 2024, n) for n in (1, 2, 3)]
    await store(
        engine,
        [(a, "AAPL", "8-k", ny(2024, 5, 3 - i, 20, 0), []) for i, a in enumerate(accs)],
    )
    for i, a in enumerate(accs):
        edgar.header(AAPL, a, ny(2024, 5, 3 - i, 16, 0))
    edgar.down.add(accs[2])  # the oldest, read last

    with pytest.raises(SecUnavailable):
        await history.correct_filing_times(
            engine, edgar.client(), start=date(2024, 1, 1), end=date(2024, 12, 31), concurrency=1
        )
    assert await progress(engine, history.TIMES_TASK) == {accs[0]: 14400, accs[1]: 14400}
    assert (await stored(engine))[(accs[0], "AAPL")][0] == ny(2024, 5, 3, 16, 0)

    edgar.down.clear()
    edgar.requests.clear()
    times = await history.correct_filing_times(
        engine, edgar.client(), start=date(2024, 1, 1), end=date(2024, 12, 31)
    )
    assert edgar.header_requests() == [f"{AAPL}/{accs[2]}"]
    assert (times.checked, times.corrected, times.done_before) == (1, 1, 2)


async def test_a_row_stored_late_after_its_filing_was_read_takes_its_time_unrequested(
    engine: AsyncEngine,
) -> None:
    edgar = Edgar()
    read = accession(AAPL, 2023, 104)
    lost_rows = accession(AAPL, 2023, 105)  # read once, but no stamped row is left
    unread = accession(AAPL, 2023, 106)
    true = ny(2023, 11, 2, 16, 30, 32)
    late = true + timedelta(hours=4)
    for acc in (read, lost_rows, unread):
        edgar.header(AAPL, acc, true)
    await store(engine, [(read, "AAPL", "8-k", late, ["2.02"])])
    first = await history.correct_filing_times(
        engine, edgar.client(), start=date(2023, 1, 1), end=date(2023, 12, 31)
    )
    assert first.corrected == 1
    # Another path stores the filing under a second symbol at the JSON's time (a
    # company whose covered symbol changed); the second filing's unit is done, but
    # its stamped rows are gone.
    await store(
        engine,
        [
            (read, "AAPL2", "8-k", late, ["2.02"]),
            (lost_rows, "AAPL", "8-k", late, []),
            (unread, "AAPL", "8-k", late, []),
        ],
    )
    await history.mark_units(engine, history.TIMES_TASK, {lost_rows: 14400})
    edgar.requests.clear()

    again = await history.correct_filing_times(
        engine, edgar.client(), start=date(2023, 1, 1), end=date(2023, 12, 31), limit=1
    )

    rows = await stored(engine)
    assert rows[(read, "AAPL2")] == rows[(read, "AAPL")] == (true, true, "header")
    assert (again.copied, again.done_before) == (1, 1)  # no request, outside the limit
    assert edgar.header_requests() == [f"{AAPL}/{lost_rows}"]  # read again, in the limit
    assert rows[(lost_rows, "AAPL")] == (true, true, "header")
    assert rows[(unread, "AAPL")][2] is None  # past the limit
    assert (await progress(engine, history.TIMES_TASK))[read] == 14400  # the first read's

    edgar.requests.clear()
    third = await history.correct_filing_times(
        engine, edgar.client(), start=date(2023, 1, 1), end=date(2023, 12, 31), limit=1
    )
    assert (third.copied, third.done_before, third.checked) == (0, 2, 1)
    assert edgar.header_requests() == [f"{AAPL}/{unread}"]


async def test_concurrent_workers_read_each_filing_once(engine: AsyncEngine) -> None:
    edgar = Edgar()
    accs = [accession(AAPL, 2024, n) for n in range(1, 31)]
    at = ny(2024, 6, 3, 21, 0)
    await store(engine, [(a, "AAPL", "8-k", at, []) for a in accs])
    for a in accs:
        edgar.header(AAPL, a, ny(2024, 6, 3, 16, 0))

    times = await history.correct_filing_times(
        engine, edgar.client(), start=date(2024, 6, 1), end=date(2024, 6, 30), concurrency=4
    )

    assert sorted(edgar.header_requests()) == sorted(f"{AAPL}/{a}" for a in accs)
    assert (times.checked, times.corrected, times.deltas) == (30, 30, {18000: 30})
    assert set(await progress(engine, history.TIMES_TASK)) == set(accs)


# ── the evening refresh ───────────────────────────────────────


def submissions(rows: list[tuple[str, str, str, str, str]]) -> dict[str, Any]:
    """A submissions record: (form, filing date, JSON acceptance, accession, items)."""
    return {
        "filings": {
            "recent": {
                "form": [r[0] for r in rows],
                "filingDate": [r[1] for r in rows],
                "acceptanceDateTime": [r[2] for r in rows],
                "accessionNumber": [r[3] for r in rows],
                "items": [r[4] for r in rows],
                "reportDate": ["" for _ in rows],
                "primaryDocument": ["d.htm" for _ in rows],
            }
        }
    }


async def test_new_filings_are_stamped_from_their_header_highest_priority_first(
    engine: AsyncEngine,
) -> None:
    edgar = Edgar()
    old = accession(AAPL, 2026, 1)
    earn, struct, quarterly = (accession(AAPL, 2026, n) for n in (2, 3, 4))
    edgar.submissions[AAPL] = submissions(
        [
            ("8-K", "2026-07-30", "2026-07-31T00:30:00.000Z", earn, "2.02,9.01"),
            ("8-K", "2026-07-29", "2026-07-29T21:00:00.000Z", struct, "4.02"),
            ("10-Q", "2026-07-31", "2026-07-31T22:00:00.000Z", quarterly, ""),
            ("8-K", "2026-06-01", "2026-06-01T12:00:00.000Z", old, "8.01"),
        ]
    )
    edgar.header(AAPL, earn, ny(2026, 7, 30, 16, 30))
    edgar.header(AAPL, struct, ny(2026, 7, 29, 17, 0))
    edgar.header(AAPL, quarterly, ny(2026, 7, 31, 18, 0))
    # Stored before by another path (the holdings check), at the JSON's time.
    await store(engine, [(old, "AAPL", "8-k", datetime(2026, 6, 1, 12, tzinfo=UTC), ["8.01"])])

    written = await daily.refresh_filings(engine, edgar.client(), {AAPL: "AAPL"}, header_budget=2)

    assert written == 3
    assert edgar.header_requests() == [f"{AAPL}/{struct}", f"{AAPL}/{earn}"]  # not the old one
    rows = await stored(engine)
    assert rows[(struct, "AAPL")] == (ny(2026, 7, 29, 17, 0),) * 2 + ("header",)
    assert rows[(earn, "AAPL")] == (ny(2026, 7, 30, 16, 30),) * 2 + ("header",)
    json_time = datetime(2026, 7, 31, 22, tzinfo=UTC)
    assert rows[(quarterly, "AAPL")] == (json_time, json_time, "json")  # past the budget
    assert rows[(old, "AAPL")][2] is None
    assert await progress(engine, history.TIMES_TASK) == {struct: 0, earn: 4 * 3600}


@pytest.mark.parametrize("status", [503, 403])
async def test_an_outage_stops_header_requests_but_not_the_refresh(
    engine: AsyncEngine, status: int
) -> None:
    edgar = Edgar()
    a, b = accession(MSFT, 2026, 1), accession(MSFT, 2026, 2)
    edgar.submissions[MSFT] = submissions(
        [
            ("8-K", "2026-07-30", "2026-07-30T20:05:00.000Z", a, "2.02"),
            ("8-K", "2026-07-29", "2026-07-29T21:00:00.000Z", b, "8.01"),
        ]
    )
    edgar.down, edgar.down_status = {a, b}, status

    written = await daily.refresh_filings(
        engine, edgar.client(), {MSFT: "MSFT"}, today=date(2026, 8, 1)
    )

    assert written == 2
    assert len(edgar.header_requests()) == 1  # no second try, no catch-up
    assert {v[2] for v in (await stored(engine)).values()} == {"json"}
    assert await progress(engine, history.TIMES_TASK) == {}


async def test_the_budget_left_corrects_the_last_month_stored_with_the_json_time(
    engine: AsyncEngine,
) -> None:
    edgar = Edgar()
    await screen(engine, {"AAPL": AAPL})
    recent, older = accession(AGENT, 2026, 1), accession(AGENT, 2026, 2)
    await store(
        engine,
        [
            (recent, "AAPL", "8-k", ny(2026, 7, 20, 20, 0), ["2.02"]),
            (older, "AAPL", "8-k", ny(2026, 5, 20, 20, 0), ["2.02"]),
        ],
    )
    edgar.header(AAPL, recent, ny(2026, 7, 20, 16, 0))
    edgar.header(AAPL, older, ny(2026, 5, 20, 16, 0))
    edgar.submissions[AAPL] = submissions([])

    await daily.refresh_filings(engine, edgar.client(), {AAPL: "AAPL"}, today=date(2026, 8, 1))

    rows = await stored(engine)
    assert rows[(recent, "AAPL")][0] == ny(2026, 7, 20, 16, 0)
    assert rows[(older, "AAPL")][0] == ny(2026, 5, 20, 20, 0)  # outside the 30 days
    assert edgar.header_requests() == [f"{AAPL}/{recent}"]
    assert daily.CATCH_UP == timedelta(days=30)


async def test_a_run_stamps_an_earnings_week_and_the_next_run_its_overflow(
    engine: AsyncEngine,
) -> None:
    """The default budget is 1,000 requests a run: a week's new filings reach that
    at the 90th percentile, and the next week's catch-up takes the rest."""
    edgar = Edgar()
    late = datetime(2026, 7, 28, 20, 30, tzinfo=UTC)  # 4 h after the header's time
    overflow = accession(MSFT, 2026, 401)
    for cik, n in ((AAPL, 600), (MSFT, 401)):
        filings = []
        for i in range(1, n + 1):
            acc, at = accession(cik, 2026, i), late + timedelta(minutes=i)
            items = "8.01" if acc == overflow else "2.02"  # the one left: last in priority
            filings.append(("8-K", "2026-07-28", f"{at:%Y-%m-%dT%H:%M:%S}.000Z", acc, items))
            edgar.header(cik, acc, at - timedelta(hours=4))
        edgar.submissions[cik] = submissions(filings)
    companies = {AAPL: "AAPL", MSFT: "MSFT"}

    written = await daily.refresh_filings(engine, edgar.client(), companies, today=date(2026, 8, 1))

    assert written == 1001
    assert len(edgar.header_requests()) == daily.HEADER_BUDGET == 1000
    sources = {key: ts for key, (_, _, ts) in (await stored(engine)).items()}
    assert [key for key, ts in sources.items() if ts != "header"] == [(overflow, "MSFT")]

    edgar.requests.clear()
    assert (
        await daily.refresh_filings(engine, edgar.client(), companies, today=date(2026, 8, 8)) == 0
    )
    assert edgar.header_requests() == [f"{MSFT}/{overflow}"]
    at = late + timedelta(minutes=401, hours=-4)
    assert (await stored(engine))[(overflow, "MSFT")] == (at, at, "header")


# ── the command ───────────────────────────────────────────────


def _run(database_url: str, work: Callable[[AsyncEngine], Awaitable[Any]]) -> Any:
    async def go() -> Any:
        engine = create_async_engine(database_url)
        try:
            return await work(engine)
        finally:
            await engine.dispose()

    return asyncio.run(go())


def test_fix_times_prints_counts_by_delta_and_resumes(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    edgar = Edgar()
    late, right = accession(AAPL, 2023, 104), accession(AAPL, 2023, 105)
    edgar.header(AAPL, late, ny(2023, 11, 2, 16, 30, 32))
    edgar.header(AAPL, right, ny(2023, 11, 3, 8, 0))

    async def seed(engine: AsyncEngine) -> None:
        await screen(engine, {"AAPL": AAPL})
        await store(
            engine,
            [
                (late, "AAPL", "8-k", ny(2023, 11, 2, 20, 30, 32), ["2.02"]),
                (right, "AAPL", "8-k", ny(2023, 11, 3, 8, 0), ["8.01"]),
            ],
        )

    _run(database_url, seed)
    monkeypatch.setenv("EDGAR_USER_AGENT", UA)
    monkeypatch.setattr("halal_trader.compliance.sec.SecClient", lambda ua: edgar.client())
    args = ["events", "filings", "fix-times", "--start", "2023-01-01", "--end", "2023-12-31"]

    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert (
        "2023-01-01..2023-12-31: 2 filing(s) read, 1 corrected (1 row(s)), "
        "0 without a header; 0 done before"
    ) in result.output
    lines = [line.strip() for line in result.output.splitlines()]
    assert "+4h00m00s  1" in lines and "0s  1" in lines

    again = CliRunner().invoke(cli, args)
    assert again.exit_code == 0, again.output
    assert "0 filing(s) read, 0 corrected (0 row(s)), 0 without a header; 2 done before" in (
        again.output
    )

    # The filing stored since under a second symbol, at the JSON's time.
    _run(
        database_url,
        lambda e: store(e, [(late, "AAPL2", "8-k", ny(2023, 11, 2, 20, 30, 32), ["2.02"])]),
    )
    third = CliRunner().invoke(cli, args)
    assert third.exit_code == 0, third.output
    assert "; 2 done before (1 row(s) stored late since, retimed)" in third.output


@pytest.mark.parametrize("status", [503, 403])
def test_fix_times_reports_an_outage_without_a_traceback(
    database_url: str, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    edgar = Edgar()
    acc = accession(AAPL, 2023, 104)
    edgar.down, edgar.down_status = {acc}, status
    _run(
        database_url,
        lambda e: store(e, [(acc, "AAPL", "8-k", ny(2023, 11, 2, 20, 30), [])]),
    )
    monkeypatch.setenv("EDGAR_USER_AGENT", UA)
    monkeypatch.setattr("halal_trader.compliance.sec.SecClient", lambda ua: edgar.client())

    result = CliRunner().invoke(cli, ["events", "filings", "fix-times", "--limit", "5"])

    assert result.exit_code == 1
    assert "EDGAR stopped answering" in result.output and "run it again" in result.output
    assert "Traceback" not in result.output
