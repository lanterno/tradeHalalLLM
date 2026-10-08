"""EDGAR requests are retried; one company EDGAR won't serve doesn't sink the screen."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.runner import SUBMISSIONS_UNAVAILABLE, run_screen
from halal_trader.compliance.sec import Company, Fact, SecClient, SecUnavailable
from halal_trader.core.http import retry_after

UA = "test test@example.invalid"


def _client(handler: Callable[[httpx.Request], httpx.Response], **kw: float | int) -> SecClient:
    return SecClient(
        UA,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        min_interval_s=0.0,
        **kw,  # type: ignore[arg-type]
    )


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the client's waits instead of sleeping them."""
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("halal_trader.core.http.asyncio.sleep", fake_sleep)
    return waits


async def test_a_429_waits_as_retry_after_asks_then_succeeds(slept: list[float]) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(
            200, json={"data": [{"cik": 1, "val": 5, "end": "2026-06-30", "accn": "a"}]}
        )

    sec = _client(handler, backoff_s=1.0)
    facts = await sec.frame("us-gaap", "LongTermDebt", "USD", "CY2026Q2I")

    assert facts[1].val == 5.0 and len(calls) == 2
    assert slept == [7.0]  # Retry-After (7 s) beat the 1 s backoff


async def test_5xx_and_timeouts_back_off_and_eventually_raise(slept: list[float]) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls % 2:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(503)

    sec = _client(handler, retries=3, backoff_s=1.0)
    with pytest.raises(SecUnavailable):
        await sec.frame("us-gaap", "LongTermDebt", "USD", "CY2026Q2I")

    assert calls == 4  # one try and three retries, no more
    assert slept == [1.0, 2.0, 4.0]  # exponential backoff between them


async def test_a_client_error_is_not_retried_and_a_404_is_no_data(slept: list[float]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404 if "Missing" in str(request.url) else 403)

    sec = _client(handler)
    assert await sec.frame("us-gaap", "Missing", "USD", "CY2026Q2I") == {}
    with pytest.raises(httpx.HTTPStatusError):
        await sec.frame("us-gaap", "LongTermDebt", "USD", "CY2026Q2I")
    assert slept == []


async def test_every_try_carries_the_user_agent(slept: list[float]) -> None:
    agents: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        agents.append(request.headers["User-Agent"])
        return httpx.Response(500 if len(agents) < 3 else 200, json={})

    await _client(handler).submissions(1)
    assert agents == [UA] * 3


async def test_a_missing_ticker_map_raises_instead_of_mapping_nothing(slept: list[float]) -> None:
    sec = _client(lambda request: httpx.Response(404))
    with pytest.raises(SecUnavailable):
        await sec.companies()


def test_retry_after_reads_seconds_and_http_dates() -> None:
    assert retry_after(httpx.Response(429, headers={"Retry-After": "30"})) == 30.0
    later = format_datetime(datetime.now(UTC) + timedelta(seconds=90), usegmt=True)
    assert 80 <= (retry_after(httpx.Response(429, headers={"Retry-After": later})) or 0) <= 91
    assert retry_after(httpx.Response(429)) is None
    assert retry_after(httpx.Response(429, headers={"Retry-After": "soon"})) is None


async def test_the_submissions_record_gives_filing_dates_by_accession() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "sic": "3571",
                "sicDescription": "Computers",
                "filings": {
                    "recent": {
                        "form": ["10-Q", "10-K"],
                        "accessionNumber": ["0000000001-26-000002", "0000000001-26-000001"],
                        "filingDate": ["2026-07-31", "2026-02-11"],
                    }
                },
            },
        )

    sec = _client(handler)
    await sec.sic(1)
    assert sec.filed("0000000001-26-000002") == date(2026, 7, 31)
    assert sec.filed("unknown") is None


class FlakySubmissions:
    """CIKs in ``down`` have a submissions record EDGAR will not serve."""

    def __init__(self, n: int, down: set[int]) -> None:
        self.n, self.down = n, down

    async def companies(self) -> dict[str, Company]:
        return {f"C{c}": Company(c, f"C{c}", f"C{c}") for c in range(1, self.n + 1)}

    async def sic(self, cik: int) -> tuple[int | None, str]:
        if cik in self.down:
            raise SecUnavailable("submissions: HTTP 503 after 5 tries")
        return 7372, "Prepackaged software"

    async def foreign_filer(self, cik: int) -> bool:
        return False

    def filed(self, accn: str) -> date | None:
        return None

    async def frame(self, taxonomy: str, concept: str, unit: str, period: str) -> dict[int, Fact]:
        end = date(2026, 6, 30)
        if concept == "EntityCommonStockSharesOutstanding":
            return {c: Fact(1_000.0, end, "a") for c in range(1, self.n + 1)}
        if concept == "CashAndCashEquivalentsAtCarryingValue":
            return {c: Fact(1_000.0, end, "a") for c in range(1, self.n + 1)}
        if concept == "Revenues" and not period.endswith("I"):
            return {c: Fact(10_000.0, date(2025, 12, 31), "a") for c in range(1, self.n + 1)}
        return {}


async def _prices(engine: AsyncEngine, symbols: list[str]) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                "volume, fetched_at) VALUES (:s, '2026-09-30', 'raw', 100, 100, 100, 100, 1, now())"
            ),
            [{"s": s} for s in symbols],
        )


async def test_one_company_edgar_will_not_serve_is_doubtful_and_the_run_goes_on(
    engine: AsyncEngine,
) -> None:
    symbols = [f"C{c}" for c in range(1, 11)]
    await _prices(engine, symbols)

    results = await run_screen(
        FlakySubmissions(10, down={3}),  # type: ignore[arg-type]
        engine,
        symbols,
        date(2026, 10, 1),
    )

    by = {r.symbol: r for r in results}
    assert by["C3"].verdict == "doubtful"
    assert all(by[s].verdict == "halal" for s in symbols if s != "C3")
    async with engine.connect() as conn:
        desc = (
            await conn.execute(
                text("SELECT sic_description FROM halal_screen_results WHERE symbol = 'C3'")
            )
        ).scalar()
    assert desc == SUBMISSIONS_UNAVAILABLE


async def test_edgar_down_for_many_companies_aborts_the_run_and_writes_nothing(
    engine: AsyncEngine,
) -> None:
    symbols = [f"C{c}" for c in range(1, 11)]
    await _prices(engine, symbols)

    with pytest.raises(SecUnavailable):
        await run_screen(
            FlakySubmissions(10, down=set(range(1, 8))),  # type: ignore[arg-type]
            engine,
            symbols,
            date(2026, 10, 1),
        )
    async with engine.connect() as conn:
        n = (await conn.execute(text("SELECT count(*) FROM halal_screen_results"))).scalar()
    assert n == 0  # the last screen stays in force
