"""Read-only Alpaca REST client for the broker ledger.

Only what the books need: account activities (fills, dividends, fees, ...)
and the daily portfolio-history equity curve. Orders still go through the
MCP adapter (mcp/client.py) until the plan replaces it; nothing here can
place, change or cancel an order.

Parsing is strict on purpose: a payload that does not have the fields the
ledger records raises, so an upstream schema change shows up as a failed
sync -- not as a silently wrong P&L.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import httpx

from halal_trader.market_hours import MARKET_TZ

logger = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"
_TIMEOUT_S = 20.0
_PAGE_SIZE = 100


@dataclass(frozen=True, slots=True)
class BrokerActivity:
    """One Alpaca account activity. Trade activities (FILL) carry symbol,
    side, qty and price; non-trade ones (DIV, FEE, CSD, ...) carry net_amount."""

    id: str
    activity_type: str
    transaction_time: datetime
    symbol: str | None
    side: str | None
    qty: float | None
    price: float | None
    net_amount: float | None
    order_id: str | None
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EquityPoint:
    day: date
    equity: float
    profit_loss: float
    profit_loss_pct: float


def _opt_float(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _timestamp(value: Any) -> datetime:
    """Alpaca timestamps: RFC 3339 for trades, a bare date for non-trade activities."""
    text = str(value)
    if len(text) == 10:  # "2026-10-01"
        return datetime.fromisoformat(text).replace(tzinfo=UTC)
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def parse_activity(raw: dict[str, Any]) -> BrokerActivity:
    activity_type = str(raw["activity_type"])
    when = raw.get("transaction_time") or raw["date"]
    return BrokerActivity(
        id=str(raw["id"]),
        activity_type=activity_type,
        transaction_time=_timestamp(when),
        symbol=raw.get("symbol"),
        side=raw.get("side"),
        qty=_opt_float(raw.get("qty")),
        price=_opt_float(raw.get("price")),
        net_amount=_opt_float(raw.get("net_amount")),
        order_id=raw.get("order_id"),
        raw=raw,
    )


def parse_portfolio_history(payload: dict[str, Any]) -> list[EquityPoint]:
    stamps = payload["timestamp"]
    equity = payload["equity"]
    pnl = payload["profit_loss"]
    pnl_pct = payload["profit_loss_pct"]
    if not len(stamps) == len(equity) == len(pnl) == len(pnl_pct):
        raise ValueError("portfolio history arrays differ in length")
    points = []
    for ts, eq, pl, plp in zip(stamps, equity, pnl, pnl_pct, strict=True):
        if eq is None:  # Alpaca pads days before the account existed with nulls
            continue
        points.append(
            EquityPoint(
                # Each 1D point is stamped 00:00 UTC *after* the session it
                # closes (e.g. 2026-09-26 00:00 UTC, a Saturday, for Friday
                # 2026-09-25). Converting to US/Eastern gives the session date;
                # the UTC date would book every day's equity one day late.
                day=datetime.fromtimestamp(int(ts), MARKET_TZ).date(),
                equity=float(eq),
                profit_loss=float(pl or 0.0),
                profit_loss_pct=float(plp or 0.0),
            )
        )
    return points


class AlpacaRestClient:
    """Thin async client over the two read-only endpoints the ledger uses."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        paper: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key or not secret_key:
            raise ValueError("Alpaca API key and secret are required")
        self._client = client or httpx.AsyncClient(
            base_url=PAPER_BASE_URL if paper else LIVE_BASE_URL,
            timeout=_TIMEOUT_S,
        )
        self._owns_client = client is None
        self._headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        response = await self._client.get(path, params=params, headers=self._headers)
        response.raise_for_status()
        return response.json()

    async def activities(self, *, after: datetime | None = None) -> list[BrokerActivity]:
        """Every account activity after ``after`` (all of them when None), oldest first."""
        params: dict[str, Any] = {"direction": "asc", "page_size": _PAGE_SIZE}
        if after is not None:
            params["after"] = after.astimezone(UTC).isoformat().replace("+00:00", "Z")
        out: list[BrokerActivity] = []
        while True:
            page = await self._get("/v2/account/activities", params)
            if not isinstance(page, list):
                raise ValueError(f"activities: expected a list, got {type(page).__name__}")
            out.extend(parse_activity(item) for item in page)
            if len(page) < _PAGE_SIZE:
                return out
            params["page_token"] = page[-1]["id"]

    async def equity_history(self, *, start: date, end: date | None = None) -> list[EquityPoint]:
        """Daily closing equity from ``start`` through ``end`` (default: today).

        ``end`` is always sent: with ``start`` alone Alpaca silently caps the
        window at one month.
        """
        payload = await self._get(
            "/v2/account/portfolio/history",
            {
                "timeframe": "1D",
                "start": start.isoformat(),
                "end": (end or datetime.now(UTC).date()).isoformat(),
            },
        )
        return parse_portfolio_history(payload)
