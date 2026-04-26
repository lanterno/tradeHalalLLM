"""Tests for the SEC EDGAR 8-K catalyst source."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from halal_trader.trading.catalysts import StockCatalystFeed
from halal_trader.trading.edgar_catalysts import (
    ITEM_LABELS,
    EDGAREightKSource,
)


def _date(days_from_now: float) -> str:
    return (datetime.now(UTC) + timedelta(days=days_from_now)).date().isoformat()


def _ticker_map_payload() -> dict:
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc"},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    }


def _submissions_payload(filings: list[dict]) -> dict:
    """Build an EDGAR-shaped submissions response from a flat filings list."""
    return {
        "filings": {
            "recent": {
                "form": [f["form"] for f in filings],
                "filingDate": [f["filingDate"] for f in filings],
                "primaryDocument": [f.get("primaryDocument", "") for f in filings],
                "accessionNumber": [f.get("accessionNumber", "") for f in filings],
                "items": [f.get("items", "") for f in filings],
            }
        }
    }


def _mock_handler(submissions_by_cik: dict[str, dict]):
    """Return a transport that serves the ticker map + each CIK's submissions."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "company_tickers.json" in url:
            return httpx.Response(200, json=_ticker_map_payload())
        for cik, payload in submissions_by_cik.items():
            if cik in url:
                return httpx.Response(200, json=payload)
        return httpx.Response(404, json={})

    return handler


def _client_with(submissions_by_cik: dict[str, dict]) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(_mock_handler(submissions_by_cik)),
        headers={"User-Agent": "Test (test@example.com)"},
    )


# ── Disabled paths ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_user_agent_returns_empty() -> None:
    src = EDGAREightKSource(user_agent="")
    assert await src.fetch(["AAPL"]) == []


@pytest.mark.asyncio
async def test_no_symbols_returns_empty() -> None:
    src = EDGAREightKSource(user_agent="Test (t@e.com)")
    src._client = _client_with({})
    assert await src.fetch([]) == []
    await src.aclose()


# ── Happy path ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetches_8k_for_known_ticker() -> None:
    submissions = {
        "0000320193": _submissions_payload(
            [
                {
                    "form": "8-K",
                    "filingDate": _date(0),
                    "primaryDocument": "aapl-20260427.htm",
                    "accessionNumber": "0000320193-26-000010",
                    "items": "2.02,9.01",
                },
            ]
        ),
    }
    src = EDGAREightKSource(user_agent="Test (t@e.com)")
    src._client = _client_with(submissions)
    out = await src.fetch(["AAPL"])
    assert len(out) == 1
    c = out[0]
    assert c.symbol == "AAPL"
    assert c.source == "edgar"
    assert c.kind.startswith("8-k:")
    # 2.02 = earnings results
    assert "earnings" in c.title.lower() or "results" in c.title.lower()
    assert c.url.startswith("https://www.sec.gov/Archives/edgar/data/")
    await src.aclose()


@pytest.mark.asyncio
async def test_skips_non_8k_filings() -> None:
    submissions = {
        "0000320193": _submissions_payload(
            [
                {"form": "10-K", "filingDate": _date(0), "items": ""},
                {"form": "8-K", "filingDate": _date(0), "items": "5.02"},
                {"form": "10-Q", "filingDate": _date(0), "items": ""},
            ]
        ),
    }
    src = EDGAREightKSource(user_agent="Test (t@e.com)")
    src._client = _client_with(submissions)
    out = await src.fetch(["AAPL"])
    assert len(out) == 1
    assert "executive" in out[0].title.lower() or "departure" in out[0].title.lower()
    await src.aclose()


@pytest.mark.asyncio
async def test_skips_filings_outside_lookback() -> None:
    submissions = {
        "0000320193": _submissions_payload(
            [
                {"form": "8-K", "filingDate": _date(-10), "items": "2.02"},  # too old
                {"form": "8-K", "filingDate": _date(0), "items": "2.02"},  # fresh
            ]
        ),
    }
    src = EDGAREightKSource(user_agent="Test (t@e.com)", look_back_hours=24)
    src._client = _client_with(submissions)
    out = await src.fetch(["AAPL"])
    assert len(out) == 1
    await src.aclose()


@pytest.mark.asyncio
async def test_unknown_ticker_silently_dropped() -> None:
    src = EDGAREightKSource(user_agent="Test (t@e.com)")
    src._client = _client_with({})
    out = await src.fetch(["UNKNOWN_TICKER"])
    assert out == []
    await src.aclose()


@pytest.mark.asyncio
async def test_handles_unparseable_filing_date() -> None:
    submissions = {
        "0000320193": _submissions_payload(
            [
                {"form": "8-K", "filingDate": "not-a-date", "items": "2.02"},
                {"form": "8-K", "filingDate": _date(0), "items": "5.02"},
            ]
        ),
    }
    src = EDGAREightKSource(user_agent="Test (t@e.com)")
    src._client = _client_with(submissions)
    out = await src.fetch(["AAPL"])
    assert len(out) == 1
    await src.aclose()


# ── Caching ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_filings_cached_per_symbol() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        url = str(request.url)
        if "company_tickers.json" in url:
            return httpx.Response(200, json=_ticker_map_payload())
        call_count += 1
        return httpx.Response(
            200,
            json=_submissions_payload([{"form": "8-K", "filingDate": _date(0), "items": "8.01"}]),
        )

    src = EDGAREightKSource(user_agent="Test (t@e.com)")
    src._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await src.fetch(["AAPL"])
    await src.fetch(["AAPL"])  # second call — cached
    await src.fetch(["AAPL"])
    assert call_count == 1
    await src.aclose()


@pytest.mark.asyncio
async def test_ticker_map_cached() -> None:
    map_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal map_calls
        url = str(request.url)
        if "company_tickers.json" in url:
            map_calls += 1
            return httpx.Response(200, json=_ticker_map_payload())
        return httpx.Response(
            200,
            json=_submissions_payload([{"form": "8-K", "filingDate": _date(0), "items": "8.01"}]),
        )

    src = EDGAREightKSource(user_agent="Test (t@e.com)")
    src._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await src.fetch(["AAPL"])
    await src.fetch(["MSFT"])  # different ticker — but map cached
    assert map_calls == 1
    await src.aclose()


# ── Error handling ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ticker_map_failure_returns_empty() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    src = EDGAREightKSource(user_agent="Test (t@e.com)")
    src._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = await src.fetch(["AAPL"])
    assert out == []
    await src.aclose()


@pytest.mark.asyncio
async def test_per_cik_failure_only_drops_that_ticker() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "company_tickers.json" in url:
            return httpx.Response(200, json=_ticker_map_payload())
        if "0000789019" in url:  # MSFT — fail
            return httpx.Response(500)
        # AAPL — succeed
        return httpx.Response(
            200,
            json=_submissions_payload([{"form": "8-K", "filingDate": _date(0), "items": "2.02"}]),
        )

    src = EDGAREightKSource(user_agent="Test (t@e.com)")
    src._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = await src.fetch(["AAPL", "MSFT"])
    symbols = {c.symbol for c in out}
    assert symbols == {"AAPL"}
    await src.aclose()


# ── Feed integration ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_integrates_with_StockCatalystFeed() -> None:
    submissions = {
        "0000320193": _submissions_payload(
            [{"form": "8-K", "filingDate": _date(0), "items": "5.02"}]
        ),
    }
    src = EDGAREightKSource(user_agent="Test (t@e.com)")
    src._client = _client_with(submissions)
    feed = StockCatalystFeed(sources=[src])
    out = await feed.fetch_all(["AAPL"])
    assert len(out) >= 1
    assert out[0].source == "edgar"
    await src.aclose()


# ── Smoke ────────────────────────────────────────────────────────


def test_item_labels_cover_high_impact() -> None:
    for high_impact in ("2.02", "5.02", "1.01", "2.05"):
        assert high_impact in ITEM_LABELS
