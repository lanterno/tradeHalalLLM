"""Equities news collector — Yahoo Finance search endpoint, no API key.

The crypto pipeline gets per-cycle news via :class:`NewsEventReactor`
+ :class:`RecentNewsFeed`. The reactor polls CryptoPanic which is
crypto-only; stocks need an equivalent source. This module fills the
gap with Yahoo Finance's public ``v1/finance/search`` endpoint —
already used elsewhere in the codebase (``trading/options_iv.py``,
``trading/options_catalyst_adapter.py``) so the dep surface and
caching pattern is familiar.

Design:

* No long-running reactor — the stocks cycle is 15-minute cadence so
  per-cycle pulls are cheap enough. ``fetch_news`` is async and
  batches a symbol list with bounded concurrency.
* Returns :class:`NewsEvent` (the same shape the crypto pipeline
  feeds into :class:`RecentNewsFeed`) so :class:`BuildNewsStage`
  consumes either path identically.
* Sentiment is classified per-headline via
  :func:`sentiment.headline_polarity.classify_headline` — a small
  curated lexicon (same pattern as ``trading/fed_speak.py``'s
  hawkish/dovish scorer). Yahoo's response has no polarity field;
  the classifier reads the headline string and emits
  ``"positive"`` / ``"negative"`` / ``"neutral"`` (the same
  literals the CryptoPanic path uses).
* 15-minute per-symbol cache — same TTL the options IV adapter uses,
  matching the 15-minute stocks cycle so each pass hits the cache
  once and the network once.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from halal_trader.sentiment.events import NewsEvent
from halal_trader.sentiment.headline_polarity import classify_headline

logger = logging.getLogger(__name__)

# Yahoo's public search endpoint. The ``newsCount`` knob caps the news
# items returned per symbol; ``quotesCount=0`` keeps the response small
# since we don't want the quote panel.
_API_BASE = "https://query2.finance.yahoo.com/v1/finance/search"
_CACHE_TTL_S = 15 * 60  # 15 min — matches the stocks-cycle cadence
_DEFAULT_NEWS_PER_SYMBOL = 5
_HTTP_TIMEOUT_S = 10.0
# Yahoo blocks Python's default user-agent. A plain browser UA is the
# minimum that gets a 200 back; matches what trading/options_iv.py uses.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class StockNewsCollector:
    """Per-symbol news fetcher backed by Yahoo Finance search.

    Returns :class:`NewsEvent` objects so the existing
    :class:`RecentNewsFeed` + :class:`BuildNewsStage` render them
    identically to crypto-side CryptoPanic items. Cache is bounded by
    symbol+TTL — repeated calls within the TTL window hit memory only.
    """

    def __init__(
        self,
        *,
        news_per_symbol: int = _DEFAULT_NEWS_PER_SYMBOL,
        cache_ttl_seconds: int = _CACHE_TTL_S,
    ) -> None:
        self._news_per_symbol = news_per_symbol
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, list[NewsEvent]]] = {}
        self._client: httpx.AsyncClient | None = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT_S,
                headers={"User-Agent": _UA},
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_for_symbols(self, symbols: list[str]) -> list[NewsEvent]:
        """Return up to ``news_per_symbol`` recent news items per symbol.

        Failures (network, parse, malformed Yahoo response) degrade to
        an empty list for that symbol — never raises. The aggregate
        result is concatenated and sorted newest-first.
        """
        out: list[NewsEvent] = []
        for sym in symbols:
            try:
                out.extend(await self._fetch_one(sym))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Yahoo news fetch failed for %s: %s", sym, exc)
        # Newest first so the LLM prompt's ``limit=6`` truncates the
        # oldest. ``published_at`` is an ISO 8601 string from Yahoo;
        # lexical sort is correct.
        out.sort(key=lambda e: e.published_at, reverse=True)
        return out

    async def _fetch_one(self, symbol: str) -> list[NewsEvent]:
        sym = symbol.upper()
        now = time.monotonic()
        cached = self._cache.get(sym)
        if cached is not None and now - cached[0] < self._cache_ttl:
            return cached[1]

        client = await self._http()
        params = {
            "q": sym,
            "newsCount": self._news_per_symbol,
            "quotesCount": 0,
            "lang": "en-US",
            "region": "US",
        }
        try:
            r = await client.get(_API_BASE, params=params)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Yahoo news request failed for %s: %s", sym, exc)
            self._cache[sym] = (now, [])
            return []

        events = _parse_news_payload(sym, data)
        self._cache[sym] = (now, events)
        return events


def _parse_news_payload(symbol: str, payload: dict[str, Any]) -> list[NewsEvent]:
    """Map Yahoo's ``news[]`` array onto :class:`NewsEvent`.

    The Yahoo response shape (partial — only the fields we read):
    ``{"news": [{"title": str, "publisher": str, "link": str,
    "providerPublishTime": int (epoch s)}]}``. Items missing required
    fields are skipped silently — the upstream API occasionally
    returns sponsored entries with a different shape.
    """
    items = payload.get("news") or []
    out: list[NewsEvent] = []
    for item in items:
        title = item.get("title")
        link = item.get("link")
        publisher = item.get("publisher") or "Yahoo Finance"
        ts = item.get("providerPublishTime")
        if not isinstance(title, str) or not isinstance(link, str):
            continue
        # Yahoo's timestamp is epoch seconds. Render as ISO 8601 so the
        # downstream lexical-sort + formatter work the same as the
        # CryptoPanic path (which already emits ISO strings).
        published = _epoch_to_iso(ts) if isinstance(ts, (int, float)) else ""
        clean_title = title.strip()
        out.append(
            NewsEvent(
                title=clean_title,
                source=publisher,
                url=link,
                published_at=published,
                sentiment=classify_headline(clean_title),
                affected_pairs=[symbol],
                importance="normal",
            )
        )
    return out


def _epoch_to_iso(ts: float) -> str:
    """Convert an epoch-seconds timestamp to ISO 8601 UTC."""
    from datetime import UTC, datetime

    return datetime.fromtimestamp(ts, tz=UTC).isoformat()
