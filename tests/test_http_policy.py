"""The one retry policy every outbound client uses (core/http.py), and what
the Alpaca clients share (execution/alpaca_http.py)."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from halal_trader.core import http
from halal_trader.execution.alpaca_http import (
    LIVE_URL,
    PAPER_URL,
    auth_headers,
    iso_z,
    trading_url,
)


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(http.asyncio, "sleep", fake_sleep)
    return waits


def _client(*answers: int | Exception) -> tuple[httpx.AsyncClient, list[str]]:
    seen: list[str] = []
    queue = list(answers)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        headers = {"Retry-After": "7"} if answer == 429 else {}
        return httpx.Response(answer, headers=headers, json={})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


async def test_retryable_answers_back_off_then_succeed(slept: list[float]) -> None:
    client, seen = _client(503, 429, 200)
    r = await http.request(client, "GET", "https://x/y", label="t", retries=3, backoff_s=1.0)
    assert r.status_code == 200 and len(seen) == 3
    assert slept == [1.0, 7.0]  # backoff, then what Retry-After asked (longer than 2 s)


async def test_the_last_answer_is_returned_once_tries_are_spent(slept: list[float]) -> None:
    client, _ = _client(503, 503)
    r = await http.request(client, "GET", "https://x/y", label="t", retries=1)
    assert r.status_code == 503


async def test_other_errors_are_not_retried(slept: list[float]) -> None:
    client, seen = _client(400)
    r = await http.request(client, "GET", "https://x/y", label="t", retries=3)
    assert r.status_code == 400 and len(seen) == 1 and slept == []


async def test_a_transport_error_is_raised_after_the_last_try(slept: list[float]) -> None:
    boom = httpx.ConnectError("down")
    client, seen = _client(boom, boom)
    with pytest.raises(httpx.ConnectError):
        await http.request(client, "GET", "https://x/y", label="t", retries=1, backoff_s=0.5)
    assert len(seen) == 2 and slept == [0.5]


async def test_waits_are_capped(slept: list[float]) -> None:
    client, _ = _client(429, 200)
    await http.request(client, "GET", "https://x/y", label="t", retries=1, max_wait_s=3.0)
    assert slept == [3.0]


async def test_the_broker_sends_an_order_once(slept: list[float]) -> None:
    from halal_trader.execution.alpaca_broker import AlpacaRestBroker, BrokerError

    client, seen = _client(503)
    broker = AlpacaRestBroker("k", "s", client=client)
    with pytest.raises(BrokerError):
        await broker._request("POST", "https://x/v2/orders", json={"symbol": "AAPL"})
    assert seen == ["POST"] and slept == []


def test_redact_hides_every_secret() -> None:
    assert http.redact("GET /x?api_key=abc&t=abc", "abc", "") == "GET /x?api_key=***&t=***"


def test_alpaca_shared_pieces() -> None:
    assert trading_url(True) == PAPER_URL and trading_url(False) == LIVE_URL
    assert auth_headers("k", "s") == {"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"}
    with pytest.raises(ValueError):
        auth_headers("k", "")
    assert iso_z(datetime(2026, 10, 8, 13, 30, tzinfo=UTC)) == "2026-10-08T13:30:00Z"


async def test_market_data_lists_assets_in_its_accounts_environment() -> None:
    from types import SimpleNamespace

    from halal_trader.data.alpaca_market import AlpacaMarketData

    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(200, json=[])

    settings = SimpleNamespace(
        alpaca=SimpleNamespace(api_key="k", secret_key="s", paper_trade=False)
    )
    market = AlpacaMarketData.from_settings(
        settings, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), min_interval_s=0
    )
    await market.assets()
    assert hosts == ["api.alpaca.markets"]
