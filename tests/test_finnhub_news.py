"""The one Finnhub company-news request (sentiment/finnhub.py)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
import pytest

from halal_trader.sentiment import finnhub


async def test_the_key_travels_in_a_header_and_the_window_in_days() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"headline": "x"}, "junk"])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    items = await finnhub.company_news(client, "secret", "AAPL", days=3, today=date(2026, 10, 8))

    assert items == [{"headline": "x"}]
    (req,) = seen
    assert req.headers["X-Finnhub-Token"] == "secret"
    assert "secret" not in str(req.url)
    assert req.url.params["from"] == "2026-10-05" and req.url.params["to"] == "2026-10-08"


async def test_an_http_error_is_the_callers_to_handle() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    with pytest.raises(httpx.HTTPStatusError):
        await finnhub.company_news(client, "k", "AAPL")


def test_published_at() -> None:
    assert finnhub.published_at(1_760_000_000) == datetime.fromtimestamp(1_760_000_000, UTC)
    aware = datetime(2026, 10, 8, 12, tzinfo=UTC)
    assert finnhub.published_at(aware) == aware
    for bad in (None, True, 0, -5, "soon", 1e30):
        assert finnhub.published_at(bad) is None
