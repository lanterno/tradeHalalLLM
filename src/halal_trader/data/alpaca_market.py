"""Alpaca market-data client: daily and monthly bars, the asset list, and news.

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


@dataclass(frozen=True, slots=True)
class NewsArticle:
    id: int
    headline: str
    summary: str
    url: str
    source: str
    symbols: tuple[str, ...]
    created_at: datetime  # UTC


def parse_news(raw: dict[str, Any]) -> NewsArticle:
    return NewsArticle(
        id=int(raw["id"]),
        headline=str(raw.get("headline") or ""),
        summary=str(raw.get("summary") or ""),
        url=str(raw.get("url") or ""),
        source=str(raw.get("source") or "benzinga"),
        symbols=tuple(str(s).upper() for s in raw.get("symbols") or []),
        created_at=datetime.fromisoformat(str(raw["created_at"]).replace("Z", "+00:00")),
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

    async def news(
        self,
        symbols: Iterable[str] | None,
        *,
        start: datetime,
        end: datetime | None = None,
        max_pages: int = 20,
    ) -> list[NewsArticle]:
        """Benzinga articles naming any of ``symbols`` (every article when None)
        published in [start, end], newest first.

        Up to ``_SYMBOLS_PER_REQUEST`` symbols per request; an article naming
        several of them is returned once.
        """
        wanted = sorted({s.upper() for s in symbols}) if symbols is not None else []
        chunks = (
            [
                ",".join(wanted[i : i + _SYMBOLS_PER_REQUEST])
                for i in range(0, len(wanted), _SYMBOLS_PER_REQUEST)
            ]
            if symbols is not None
            else [None]
        )
        out: dict[int, NewsArticle] = {}
        for chunk in chunks:
            params: dict[str, Any] = {
                "start": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "limit": 50,
                "sort": "desc",
            }
            if chunk is not None:
                params["symbols"] = chunk
            if end is not None:
                params["end"] = end.astimezone(UTC).isoformat().replace("+00:00", "Z")
            for _ in range(max_pages):
                payload = await self._get(f"{DATA_URL}/v1beta1/news", params)
                for raw in payload.get("news") or []:
                    article = parse_news(raw)
                    out[article.id] = article
                token = payload.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
        return sorted(out.values(), key=lambda a: a.created_at, reverse=True)

    async def minute_bars(
        self, symbol: str, *, start: datetime, end: datetime
    ) -> list[dict[str, Any]]:
        """Raw SIP minute bars of one symbol in [start, end], as Alpaca returns them."""
        out: list[dict[str, Any]] = []
        params: dict[str, Any] = {
            "symbols": symbol,
            "timeframe": "1Min",
            "start": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "end": min(end, datetime.now(UTC) - _SIP_EMBARGO)
            .astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "feed": "sip",
            "adjustment": "raw",
            "limit": _PAGE_LIMIT,
        }
        while True:
            payload = await self._get(f"{DATA_URL}/v2/stocks/bars", params)
            out += (payload.get("bars") or {}).get(symbol) or []
            token = payload.get("next_page_token")
            if not token:
                return out
            params["page_token"] = token

    async def cash_dividends(
        self, symbols: Iterable[str], *, start: date, end: date
    ) -> list[dict[str, Any]]:
        """Cash dividends (ex-date in [start, end]) from Alpaca's corporate actions."""
        wanted = sorted({s.upper() for s in symbols})
        out: list[dict[str, Any]] = []
        for i in range(0, len(wanted), _SYMBOLS_PER_REQUEST):
            params: dict[str, Any] = {
                "symbols": ",".join(wanted[i : i + _SYMBOLS_PER_REQUEST]),
                "types": "cash_dividend",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": 1000,
            }
            while True:
                payload = await self._get(f"{DATA_URL}/v1/corporate-actions", params)
                out += (payload.get("corporate_actions") or {}).get("cash_dividends") or []
                token = payload.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
        return out

    async def assets(self) -> list[Asset]:
        """Every active US equity Alpaca lists (stocks and ETFs)."""
        payload = await self._get(
            f"{TRADING_URL}/v2/assets", {"status": "active", "asset_class": "us_equity"}
        )
        return [parse_asset(a) for a in payload]

    async def inactive_assets(self) -> list[Asset]:
        """US equities Alpaca no longer lists: delisted, merged, or renamed.

        Their prices are still served, which is what lets a backtest include
        the companies that did not survive.
        """
        payload = await self._get(
            f"{TRADING_URL}/v2/assets", {"status": "inactive", "asset_class": "us_equity"}
        )
        return [parse_asset(a) for a in payload]

    async def daily_bars(
        self,
        symbols: Iterable[str],
        *,
        start: date,
        end: date | None = None,
        adjustment: Adjustment = "raw",
        timeframe: str = "1Day",
    ) -> list[DailyBar]:
        """SIP bars for ``symbols`` from ``start`` through ``end`` (default: now).

        ``timeframe`` is Alpaca's: "1Day", or "1Month" (stamped on the first
        of the month) for cheap long-horizon liquidity history.
        """
        end_ts = datetime.now(UTC) - _SIP_EMBARGO
        if end is not None:
            end_ts = min(end_ts, datetime.combine(end, datetime.max.time(), UTC))
        wanted = sorted({s.upper() for s in symbols})
        out: list[DailyBar] = []
        for i in range(0, len(wanted), _SYMBOLS_PER_REQUEST):
            batch = wanted[i : i + _SYMBOLS_PER_REQUEST]
            params: dict[str, Any] = {
                "symbols": ",".join(batch),
                "timeframe": timeframe,
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
