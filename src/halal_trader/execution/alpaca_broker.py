"""Alpaca over REST: the Broker port without the MCP subprocess (plan 3.2 / 1.6).

The MCP adapter (mcp/client.py) is the boundary that broke the bot most
often: an unpinned subprocess, schema changes upstream, no timeouts, parse
failures turned into made-up defaults. This adapter talks to Alpaca's REST
API directly with httpx:

* every request has a timeout; idempotent GETs retry on transport errors;
* orders carry a ``client_order_id``. If a submission times out, the order
  is looked up by that id *before* anything is resubmitted, so a network
  blip can never place the same order twice;
* market data and orders come back in the shapes Alpaca's REST API uses,
  which are also the shapes the MCP server passed through -- callers parse
  them unchanged (see tests/fixtures/alpaca_mcp_2_3_2/);
* account, clock and positions are parsed strictly into domain models: a
  missing field raises instead of becoming 0.

The bot trades through the MCP server; this adapter serves the core
portfolio and the side-by-side check `halal-trader broker compare`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from halal_trader.domain.models import Account, MarketClock, Position
from halal_trader.market_hours import now_eastern, today_eastern

logger = logging.getLogger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_GET_RETRIES = 3
_SIP_EMBARGO = timedelta(minutes=16)


class BrokerError(RuntimeError):
    """Alpaca refused or failed a request; carries the HTTP status and body."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Alpaca HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


def _f(raw: dict[str, Any], key: str) -> float:
    value = raw[key]  # KeyError on a missing field: fail loudly, not as 0
    return float(value)


def _json_or_text(body: str) -> Any:
    try:
        return json.loads(body)
    except ValueError:
        return body


class AlpacaRestBroker:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        paper: bool = True,
        client: httpx.AsyncClient | None = None,
        data_feed: str = "iex",
    ) -> None:
        if not api_key or not secret_key:
            raise ValueError("Alpaca API key and secret are required")
        self._client = client or httpx.AsyncClient(timeout=_TIMEOUT)
        self._owns_client = client is None
        self._base = PAPER_URL if paper else LIVE_URL
        self._headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
        self._feed = data_feed

    # ── lifecycle (parity with the MCP client) ────────────────────

    async def connect(self) -> None:
        account = await self.get_account_info()
        logger.info("Connected to Alpaca REST (%s, account %s)", self._base, account.status)

    async def disconnect(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ── transport ─────────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        allow_404: bool = False,
    ) -> Any:
        retries = _GET_RETRIES if method == "GET" else 1
        for attempt in range(retries):
            try:
                response = await self._client.request(
                    method, url, params=params, json=json, headers=self._headers
                )
            except httpx.TransportError:
                if attempt + 1 >= retries:
                    raise
                await asyncio.sleep(0.5 * 2**attempt)
                continue
            if allow_404 and response.status_code == 404:
                return None
            if response.status_code >= 400:
                raise BrokerError(response.status_code, response.text)
            if response.status_code == 204 or not response.content:
                return {}
            return response.json()
        raise RuntimeError("unreachable")

    def _trading(self, path: str) -> str:
        return f"{self._base}{path}"

    # ── account / clock / positions (strict) ──────────────────────

    async def get_account_info(self) -> Account:
        raw = await self._request("GET", self._trading("/v2/account"))
        multiplier = raw.get("multiplier")
        shorting = raw.get("shorting_enabled")
        return Account(
            equity=_f(raw, "equity"),
            buying_power=_f(raw, "buying_power"),
            cash=_f(raw, "cash"),
            portfolio_value=_f(raw, "portfolio_value"),
            status=str(raw["status"]),
            account_id=str(raw["id"]) if raw.get("id") else None,
            account_number=str(raw["account_number"]) if raw.get("account_number") else None,
            multiplier=float(multiplier) if multiplier not in (None, "") else None,
            # A string "false" would be truthy: only a JSON boolean counts.
            shorting_enabled=shorting if isinstance(shorting, bool) else None,
        )

    async def get_clock(self) -> MarketClock:
        raw = await self._request("GET", self._trading("/v2/clock"))
        return MarketClock(
            is_open=bool(raw["is_open"]),
            next_open=str(raw["next_open"]),
            next_close=str(raw["next_close"]),
            timestamp=now_eastern(),
        )

    async def get_calendar(self, start: str | None = None, end: str | None = None) -> Any:
        today = today_eastern()
        params = {
            "start": start or today.isoformat(),
            "end": end or (today + timedelta(days=30)).isoformat(),
        }
        return await self._request("GET", self._trading("/v2/calendar"), params=params)

    async def get_all_positions(self) -> list[Position]:
        raw = await self._request("GET", self._trading("/v2/positions"))
        if not isinstance(raw, list):
            raise BrokerError(200, f"positions: expected a list, got {type(raw).__name__}")
        return [
            Position(
                symbol=str(p["symbol"]),
                qty=_f(p, "qty"),
                avg_entry_price=_f(p, "avg_entry_price"),
                current_price=_f(p, "current_price"),
                unrealized_pl=_f(p, "unrealized_pl"),
                unrealized_plpc=_f(p, "unrealized_plpc"),
            )
            for p in raw
        ]

    async def account_payload(self) -> dict[str, Any]:
        """``/v2/account`` as Alpaca returns it (``last_equity`` included)."""
        return dict(await self._request("GET", self._trading("/v2/account")))

    async def positions_payload(self) -> list[dict[str, Any]]:
        """``/v2/positions`` as Alpaca returns it (``change_today``, ``lastday_price``...)."""
        raw = await self._request("GET", self._trading("/v2/positions"))
        if not isinstance(raw, list):
            raise BrokerError(200, f"positions: expected a list, got {type(raw).__name__}")
        return [dict(p) for p in raw]

    # ── market data (REST shapes, as the MCP server passed through) ──

    async def get_stock_snapshot(self, symbols: str) -> Any:
        """``{"AAPL": {"latestTrade": .., "latestQuote": .., "dailyBar": .., ...}}``."""
        return await self._request(
            "GET",
            f"{DATA_URL}/v2/stocks/snapshots",
            params={"symbols": symbols, "feed": self._feed},
        )

    async def get_stock_bars(self, symbol: str, days: int = 5, timeframe: str = "1Day") -> Any:
        """``{"bars": {"AAPL": [...]}}``: consolidated (SIP) history, as the
        free plan allows up to 15 minutes ago."""
        end = datetime.now(UTC) - _SIP_EMBARGO
        params = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": (end - timedelta(days=days)).isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "feed": "sip",
            "limit": 10_000,
        }
        return await self._request("GET", f"{DATA_URL}/v2/stocks/bars", params=params)

    async def get_option_chain(
        self,
        underlying: str,
        *,
        feed: str = "indicative",
        expiration_date_gte: str | None = None,
        expiration_date_lte: str | None = None,
        strike_price_gte: float | None = None,
        strike_price_lte: float | None = None,
        limit: int = 200,
    ) -> Any:
        optional = {
            "expiration_date_gte": expiration_date_gte,
            "expiration_date_lte": expiration_date_lte,
            "strike_price_gte": strike_price_gte,
            "strike_price_lte": strike_price_lte,
        }
        params: dict[str, Any] = {"feed": feed, "limit": limit}
        params.update({k: v for k, v in optional.items() if v is not None})
        return await self._request(
            "GET", f"{DATA_URL}/v1beta1/options/snapshots/{underlying}", params=params
        )

    # ── orders ────────────────────────────────────────────────────

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        time_in_force: str = "day",
        *,
        client_order_id: str | None = None,
    ) -> Any:
        """Submit an order exactly once, even across a timeout.

        A refusal (4xx) is returned, not raised, as ``{"error": {"status",
        "detail"}}`` with Alpaca's body as the detail: the shape the MCP
        server used, which the executor (no order id = rejected) and the
        monitor (wash-trade code 40310000, then cancel and retry) read.

        ``client_order_id``: a caller-chosen id makes the order idempotent
        across runs, not only across a timeout. Alpaca refuses a second order
        with an id the account has already used (the core sends
        ``core-<date>-<symbol>-<side>``, so a repeated run cannot buy twice).
        """
        client_order_id = client_order_id or f"ht-{uuid.uuid4().hex}"
        qty = str(int(quantity)) if float(quantity).is_integer() else repr(float(quantity))
        body = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "type": order_type,
            "time_in_force": time_in_force,
            "client_order_id": client_order_id,
        }
        for attempt in range(2):
            try:
                return await self._request("POST", self._trading("/v2/orders"), json=body)
            except BrokerError as refused:
                if not 400 <= refused.status < 500:
                    raise
                return {"error": {"status": refused.status, "detail": _json_or_text(refused.body)}}
            except httpx.TransportError:
                # Did the first attempt reach Alpaca? Ask before resubmitting.
                existing = await self._request(
                    "GET",
                    self._trading("/v2/orders:by_client_order_id"),
                    params={"client_order_id": client_order_id},
                    allow_404=True,
                )
                if existing:
                    logger.warning(
                        "order %s landed despite a timeout; not resubmitting", client_order_id
                    )
                    return existing
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable")

    async def get_open_orders(self) -> list[dict[str, Any]]:
        """Every order not yet final (new, accepted, partially filled, ...)."""
        raw = await self._request(
            "GET", self._trading("/v2/orders"), params={"status": "open", "limit": 500}
        )
        if not isinstance(raw, list):
            raise BrokerError(200, f"orders: expected a list, got {type(raw).__name__}")
        return [dict(o) for o in raw]

    async def get_order_by_id(self, order_id: str) -> dict[str, Any]:
        raw = await self._request("GET", self._trading(f"/v2/orders/{order_id}"))
        return raw if isinstance(raw, dict) else {}

    async def cancel_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        result = await self._request("DELETE", self._trading(f"/v2/orders/{order_id}"))
        return result if isinstance(result, dict) else {"result": result}

    async def close_position(self, symbol: str) -> Any:
        return await self._request("DELETE", self._trading(f"/v2/positions/{symbol}"))

    async def close_all_positions(self) -> Any:
        return await self._request(
            "DELETE", self._trading("/v2/positions"), params={"cancel_orders": "true"}
        )
