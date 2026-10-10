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

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

import httpx

from halal_trader.core import http
from halal_trader.execution.alpaca_http import (
    DATA_URL,
    SIP_EMBARGO,
    auth_headers,
    iso_z,
    trading_url,
)

logger = logging.getLogger(__name__)

Adjustment = Literal["raw", "all"]
MinuteFeed = Literal["sip", "iex"]

_SYMBOLS_PER_REQUEST = 100
_PAGE_LIMIT = 10_000
_MIN_INTERVAL_S = 0.35  # ~170 requests/min, under the free tier's 200
_MAX_RETRIES = 5


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
        paper: bool = True,
        client: httpx.AsyncClient | None = None,
        min_interval_s: float = _MIN_INTERVAL_S,
    ) -> None:
        self._headers = auth_headers(api_key, secret_key)
        self._client = client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = client is None
        self._trading = trading_url(paper)  # the asset list is a trading-API call
        self._pacer = http.Pacer(min_interval_s)

    @classmethod
    def from_settings(cls, settings: Any, **kwargs: Any) -> AlpacaMarketData:
        """The day-trader's keys (ALPACA_*), in their account's environment."""
        a = settings.alpaca
        return cls(a.api_key, a.secret_key, paper=a.paper_trade, **kwargs)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(self, url: str, params: dict[str, Any]) -> Any:
        response = await http.request(
            self._client,
            "GET",
            url,
            label="Alpaca market data",
            retries=_MAX_RETRIES - 1,
            backoff_s=2.0,
            pacer=self._pacer,
            params=params,
            headers=self._headers,
        )
        response.raise_for_status()
        return response.json()

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
                "start": iso_z(start),
                "limit": 50,
                "sort": "desc",
            }
            if chunk is not None:
                params["symbols"] = chunk
            if end is not None:
                params["end"] = iso_z(end)
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
        self, symbol: str, *, start: datetime, end: datetime, feed: MinuteFeed = "sip"
    ) -> list[dict[str, Any]]:
        """Raw minute bars of one symbol in [start, end], as Alpaca returns them."""
        bars = await self.minute_bars_many([symbol], start=start, end=end, feed=feed)
        return bars.get(symbol, [])

    async def minute_bars_many(
        self,
        symbols: Iterable[str],
        *,
        start: datetime,
        end: datetime,
        feed: MinuteFeed = "sip",
    ) -> dict[str, list[dict[str, Any]]]:
        """Raw minute bars of several symbols in [start, end], by symbol.

        ``feed`` is SIP (the consolidated tape research stores) unless asked
        otherwise; ``"iex"`` is only probed (the news engine's D8), never
        stored as history. One request carries up to ``_SYMBOLS_PER_REQUEST``
        symbols and a page holds ``_PAGE_LIMIT`` bars across them, so a
        session of a hundred names takes about four requests instead of a
        hundred.
        """
        wanted = sorted({s.upper() for s in symbols})
        out: dict[str, list[dict[str, Any]]] = {}
        for i in range(0, len(wanted), _SYMBOLS_PER_REQUEST):
            params: dict[str, Any] = {
                "symbols": ",".join(wanted[i : i + _SYMBOLS_PER_REQUEST]),
                "timeframe": "1Min",
                "start": iso_z(start),
                "end": iso_z(min(end, datetime.now(UTC) - SIP_EMBARGO)),
                "feed": feed,
                "adjustment": "raw",
                "limit": _PAGE_LIMIT,
            }
            while True:
                payload = await self._get(f"{DATA_URL}/v2/stocks/bars", params)
                for sym, rows in (payload.get("bars") or {}).items():
                    out.setdefault(sym, []).extend(rows or [])
                token = payload.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
        return out

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
            f"{self._trading}/v2/assets", {"status": "active", "asset_class": "us_equity"}
        )
        return [parse_asset(a) for a in payload]

    async def inactive_assets(self) -> list[Asset]:
        """US equities Alpaca no longer lists: delisted, merged, or renamed.

        Their prices are still served, which is what lets a backtest include
        the companies that did not survive.
        """
        payload = await self._get(
            f"{self._trading}/v2/assets", {"status": "inactive", "asset_class": "us_equity"}
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
        end_ts = datetime.now(UTC) - SIP_EMBARGO
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
                "end": iso_z(end_ts),
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
