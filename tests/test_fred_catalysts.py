"""Tests for the FRED release-calendar catalyst source."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from halal_trader.trading.catalysts import StockCatalystFeed
from halal_trader.trading.fred_catalysts import (
    FRED_RELEASE_IDS,
    FREDReleaseCalendarSource,
)


def _mock_transport(payload_by_id: dict[int, dict]) -> httpx.MockTransport:
    """Mock transport that returns scripted JSON keyed on release_id."""

    def handler(request: httpx.Request) -> httpx.Response:
        rid = int(request.url.params["release_id"])
        body = payload_by_id.get(rid, {"release_dates": []})
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def _mock_source(
    payload_by_id: dict[int, dict],
    *,
    api_key: str = "test-key",
    enabled: tuple[str, ...] = ("cpi", "fomc"),
    look_ahead_days: int = 30,
) -> FREDReleaseCalendarSource:
    src = FREDReleaseCalendarSource(
        api_key=api_key,
        enabled_releases=enabled,
        look_ahead_days=look_ahead_days,
    )
    src._client = httpx.AsyncClient(transport=_mock_transport(payload_by_id))
    return src


def _date(days_from_now: int) -> str:
    return (datetime.now(UTC) + timedelta(days=days_from_now)).date().isoformat()


# ── Empty / disabled paths ───────────────────────────────────────


@pytest.mark.asyncio
async def test_no_api_key_returns_empty() -> None:
    src = FREDReleaseCalendarSource(api_key="")
    out = await src.fetch(["AAPL"])
    assert out == []


@pytest.mark.asyncio
async def test_no_symbols_returns_empty() -> None:
    src = _mock_source({10: {"release_dates": [{"date": _date(7)}]}})
    out = await src.fetch([])
    assert out == []
    await src.aclose()


# ── Release-date parsing ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetches_and_emits_per_symbol_catalyst() -> None:
    cpi_id = FRED_RELEASE_IDS["cpi"]
    fomc_id = FRED_RELEASE_IDS["fomc"]
    src = _mock_source(
        {
            cpi_id: {"release_dates": [{"date": _date(5)}]},
            fomc_id: {"release_dates": [{"date": _date(14)}]},
        }
    )
    out = await src.fetch(["AAPL", "MSFT"])
    # 2 releases × 2 symbols = 4 catalysts
    assert len(out) == 4
    kinds = {c.kind for c in out}
    assert kinds == {"cpi", "fomc"}
    symbols = {c.symbol for c in out}
    assert symbols == {"AAPL", "MSFT"}
    assert all(c.source == "fred" for c in out)
    await src.aclose()


@pytest.mark.asyncio
async def test_skips_past_dates() -> None:
    cpi_id = FRED_RELEASE_IDS["cpi"]
    src = _mock_source(
        {
            cpi_id: {
                "release_dates": [
                    {"date": _date(-5)},  # past — drop
                    {"date": _date(7)},  # future — keep
                ]
            }
        },
        enabled=("cpi",),
    )
    out = await src.fetch(["AAPL"])
    assert len(out) == 1
    assert out[0].timestamp.date() > datetime.now(UTC).date()
    await src.aclose()


@pytest.mark.asyncio
async def test_skips_unparseable_dates() -> None:
    cpi_id = FRED_RELEASE_IDS["cpi"]
    src = _mock_source(
        {
            cpi_id: {
                "release_dates": [
                    {"date": "not-a-date"},
                    {"date": _date(3)},
                ]
            }
        },
        enabled=("cpi",),
    )
    out = await src.fetch(["AAPL"])
    assert len(out) == 1
    await src.aclose()


@pytest.mark.asyncio
async def test_unknown_release_skipped() -> None:
    src = _mock_source({}, enabled=("cpi", "made_up_release"))  # type: ignore[arg-type]
    out = await src.fetch(["AAPL"])
    # Only 'cpi' is mapped, but we returned no dates for it -> empty
    assert out == []
    await src.aclose()


# ── HTTP error handling ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_http_error_degrades_to_empty() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    src = FREDReleaseCalendarSource(api_key="bad-key", enabled_releases=("cpi",))
    src._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = await src.fetch(["AAPL"])
    assert out == []
    await src.aclose()


@pytest.mark.asyncio
async def test_transport_exception_degrades_to_empty() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    src = FREDReleaseCalendarSource(api_key="key", enabled_releases=("cpi",))
    src._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = await src.fetch(["AAPL"])
    assert out == []
    await src.aclose()


# ── Caching ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_avoids_repeat_calls() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"release_dates": [{"date": _date(7)}]})

    src = FREDReleaseCalendarSource(api_key="key", enabled_releases=("cpi",))
    src._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await src.fetch(["AAPL"])
    await src.fetch(["MSFT"])  # second call — should hit cache
    await src.fetch(["GOOG"])  # third — still cached
    assert call_count == 1
    await src.aclose()


# ── Feed integration ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_integrates_with_StockCatalystFeed() -> None:
    cpi_id = FRED_RELEASE_IDS["cpi"]
    fomc_id = FRED_RELEASE_IDS["fomc"]
    src = _mock_source(
        {
            cpi_id: {"release_dates": [{"date": _date(2)}]},
            fomc_id: {"release_dates": [{"date": _date(8)}]},
        },
        enabled=("cpi", "fomc"),
    )
    feed = StockCatalystFeed(sources=[src])
    out = await feed.fetch_all(["AAPL"])
    kinds = {c.kind for c in out}
    assert "cpi" in kinds
    assert "fomc" in kinds
    await src.aclose()


# ── Smoke ────────────────────────────────────────────────────────


def test_release_id_table_has_expected_keys() -> None:
    assert {"cpi", "nfp", "fomc", "gdp"} <= set(FRED_RELEASE_IDS)
