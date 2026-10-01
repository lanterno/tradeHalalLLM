"""Alpaca market-data client for research: daily bars and the asset list.

The free data plan serves *historical* consolidated (SIP) bars back to 2016
-- only the most recent 15 minutes are off limits -- so research can use the
full market's volume rather than the IEX slice the live bot quotes from. The
client asks for at most ``end = now - 16 min`` to stay inside that rule.

Throttled to stay under the free tier's 200 requests/minute, with backoff
on 429. Strict parsing, as in execution/alpaca_rest.py: a payload without the
fields we store raises instead of storing nonsense.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

DATA_URL = "https://data.alpaca.markets"
TRADING_URL = "https://paper-api.alpaca.markets"
Adjustment = Literal["raw", "all"]

_SYMBOLS_PER_REQUEST = 100
_PAGE_LIMIT = 10_000
_MIN_INTERVAL_S = 0.35  # ~170 requests/min, under the free tier's 200
_MAX_RETRIES = 5
_SIP_EMBARGO = timedelta(minutes=16)


@dataclass(frozen=True, slots=True)
class DailyBar:
    symbol: str
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None
    trades: int | None


@dataclass(frozen=True, slots=True)
class Asset:
    symbol: str
    name: str
    exchange: str
    tradable: bool
    fractionable: bool
    status: str


def parse_bar(symbol: str, raw: dict[str, Any]) -> DailyBar:
    return DailyBar(
        symbol=symbol,
        # Daily bars are stamped 04:00Z/05:00Z = midnight US/Eastern of the session.
        day=date.fromisoformat(str(raw["t"])[:10]),
        open=float(raw["o"]),
        high=float(raw["h"]),
        low=float(raw["l"]),
        close=float(raw["c"]),
        volume=float(raw["v"]),
        vwap=float(raw["vw"]) if raw.get("vw") is not None else None,
        trades=int(raw["n"]) if raw.get("n") is not None else None,
    )


def parse_asset(raw: dict[str, Any]) -> Asset:
    return Asset(
        symbol=str(raw["symbol"]),
        name=str(raw.get("name") or ""),
        exchange=str(raw["exchange"]),
        tradable=bool(raw["tradable"]),
        fractionable=bool(raw.get("fractionable", False)),
        status=str(raw["status"]),
    )


class AlpacaMarketData:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        min_interval_s: float = _MIN_INTERVAL_S,
    ) -> None:
        if not api_key or not secret_key:
            raise ValueError("Alpaca API key and secret are required")
        self._client = client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = client is None
        self._headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
        self._min_interval = min_interval_s
        self._last_request = 0.0

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(self, url: str, params: dict[str, Any]) -> Any:
        loop = asyncio.get_running_loop()
        for attempt in range(_MAX_RETRIES):
            wait = self._last_request + self._min_interval - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = loop.time()
            response = await self._client.get(url, params=params, headers=self._headers)
            if response.status_code == 429:
                backoff = 2.0 ** (attempt + 1)
                logger.warning("market data rate-limited; backing off %.0fs", backoff)
                await asyncio.sleep(backoff)
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"market data request still rate-limited after {_MAX_RETRIES} tries")

    async def assets(self) -> list[Asset]:
        """Every active US equity Alpaca lists (stocks and ETFs)."""
        payload = await self._get(
            f"{TRADING_URL}/v2/assets", {"status": "active", "asset_class": "us_equity"}
        )
        return [parse_asset(a) for a in payload]

    async def daily_bars(
        self,
        symbols: Iterable[str],
        *,
        start: date,
        end: date | None = None,
        adjustment: Adjustment = "raw",
    ) -> list[DailyBar]:
        """Daily SIP bars for ``symbols`` from ``start`` through ``end`` (default: now)."""
        end_ts = datetime.now(UTC) - _SIP_EMBARGO
        if end is not None:
            end_ts = min(end_ts, datetime.combine(end, datetime.max.time(), UTC))
        wanted = sorted({s.upper() for s in symbols})
        out: list[DailyBar] = []
        for i in range(0, len(wanted), _SYMBOLS_PER_REQUEST):
            batch = wanted[i : i + _SYMBOLS_PER_REQUEST]
            params: dict[str, Any] = {
                "symbols": ",".join(batch),
                "timeframe": "1Day",
                "start": start.isoformat(),
                "end": end_ts.isoformat().replace("+00:00", "Z"),
                "adjustment": adjustment,
                "feed": "sip",
                "limit": _PAGE_LIMIT,
            }
            while True:
                page = await self._get(f"{DATA_URL}/v2/stocks/bars", params)
                for sym, rows in (page.get("bars") or {}).items():
                    out.extend(parse_bar(sym, r) for r in rows)
                token = page.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
        return out
