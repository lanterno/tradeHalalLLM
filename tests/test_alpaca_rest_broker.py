"""The Alpaca REST broker adapter: strict parsing, retries, exactly-once orders."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from halal_trader.execution import alpaca_broker
from halal_trader.execution.alpaca_broker import AlpacaRestBroker, BrokerError

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr(alpaca_broker.asyncio, "sleep", instant)


def _broker(handler: Handler) -> AlpacaRestBroker:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AlpacaRestBroker("key", "secret", paper=True, client=client)


ACCOUNT = {
    "equity": "105028.70",
    "buying_power": "210057.40",
    "cash": "40000.00",
    "portfolio_value": "105028.70",
    "status": "ACTIVE",
}


async def test_account_is_parsed_from_alpaca_strings() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ACCOUNT)

    account = await _broker(handler).get_account_info()

    assert account.equity == 105028.70 and account.status == "ACTIVE"
    assert str(seen[0].url) == "https://paper-api.alpaca.markets/v2/account"
    assert seen[0].headers["APCA-API-KEY-ID"] == "key"


async def test_a_missing_account_field_raises_instead_of_reading_zero() -> None:
    partial = {k: v for k, v in ACCOUNT.items() if k != "equity"}
    with pytest.raises(KeyError):
        await _broker(lambda r: httpx.Response(200, json=partial)).get_account_info()


async def test_positions_are_parsed_and_a_non_list_raises() -> None:
    position = {
        "symbol": "MSFT",
        "qty": "24",
        "avg_entry_price": "500.1",
        "current_price": "510.0",
        "unrealized_pl": "237.6",
        "unrealized_plpc": "0.0198",
    }
    positions = await _broker(lambda r: httpx.Response(200, json=[position])).get_all_positions()
    assert [(p.symbol, p.qty) for p in positions] == [("MSFT", 24.0)]

    with pytest.raises(BrokerError):
        await _broker(lambda r: httpx.Response(200, json={"oops": 1})).get_all_positions()


async def test_http_errors_raise_with_status_and_body() -> None:
    broker = _broker(lambda r: httpx.Response(403, json={"message": "forbidden"}))
    with pytest.raises(BrokerError) as caught:
        await broker.get_clock()
    assert caught.value.status == 403 and "forbidden" in caught.value.body


async def test_gets_retry_transport_errors() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("reset", request=request)
        return httpx.Response(
            200,
            json={"is_open": False, "next_open": "2026-10-02T09:30:00-04:00", "next_close": "x"},
        )

    clock = await _broker(handler).get_clock()

    assert calls == 3 and clock.is_open is False


async def test_bars_ask_for_sip_history_outside_the_embargo() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"bars": {"AAPL": []}, "next_page_token": None})

    out = await _broker(handler).get_stock_bars("AAPL", days=60)

    assert out == {"bars": {"AAPL": []}, "next_page_token": None}
    params = seen[0].url.params
    assert params["feed"] == "sip" and params["symbols"] == "AAPL"
    assert seen[0].url.host == "data.alpaca.markets"


# ── orders: exactly once ──────────────────────────────────────────────


def _order(body: dict[str, Any]) -> dict[str, Any]:
    return {"id": "o-1", "status": "accepted", **body}


async def test_order_body_carries_a_client_order_id_and_string_qty() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=_order(sent[-1]))

    result = await _broker(handler).place_order("AMD", "buy", 8)

    assert result["id"] == "o-1"
    assert sent[0]["qty"] == "8" and sent[0]["type"] == "market"
    assert sent[0]["client_order_id"].startswith("ht-")


async def test_a_timed_out_order_that_landed_is_not_resubmitted() -> None:
    posts: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posts.append(json.loads(request.content))
            raise httpx.ReadTimeout("slow", request=request)
        assert request.url.path == "/v2/orders:by_client_order_id"
        assert request.url.params["client_order_id"] == posts[0]["client_order_id"]
        return httpx.Response(200, json=_order(posts[0]))

    result = await _broker(handler).place_order("AMD", "buy", 8)

    assert len(posts) == 1 and result["id"] == "o-1"


async def test_a_timed_out_order_that_never_landed_is_resubmitted_once_same_id() -> None:
    posts: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posts.append(json.loads(request.content))
            if len(posts) == 1:
                raise httpx.ReadTimeout("slow", request=request)
            return httpx.Response(200, json=_order(posts[-1]))
        return httpx.Response(404, json={"message": "order not found"})

    await _broker(handler).place_order("AMD", "buy", 8)

    assert len(posts) == 2
    assert posts[0]["client_order_id"] == posts[1]["client_order_id"]


async def test_an_order_refusal_comes_back_in_the_shape_the_monitor_reads() -> None:
    """Not retried, not raised: the monitor's wash-trade recovery parses it."""
    from halal_trader.trading.executor import _extract_order_id
    from halal_trader.trading.monitor import StockPositionMonitor

    posts = 0
    wash = {"code": 40310000, "message": "potential wash trade", "existing_order_id": "o-9"}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        posts += 1
        return httpx.Response(403, json=wash)

    result = await _broker(handler).place_order("AMD", "sell", 8)

    assert posts == 1
    assert _extract_order_id(result) == ""
    assert StockPositionMonitor._wash_trade_conflict_id(result) == "o-9"


async def test_a_server_error_on_an_order_raises() -> None:
    with pytest.raises(BrokerError):
        await _broker(lambda r: httpx.Response(500, text="boom")).place_order("AMD", "buy", 8)


async def test_close_position_and_flatten_use_alpacas_endpoints() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v2/positions":
            return httpx.Response(207, json=[{"symbol": "AMD", "status": 200, "body": {}}])
        return httpx.Response(200, json={"id": "o-2"})

    broker = _broker(handler)
    await broker.close_position("ORCL")
    flattened = await broker.close_all_positions()

    assert (seen[0].method, seen[0].url.path) == ("DELETE", "/v2/positions/ORCL")
    assert seen[1].url.params["cancel_orders"] == "true"
    assert flattened == [{"symbol": "AMD", "status": 200, "body": {}}]
