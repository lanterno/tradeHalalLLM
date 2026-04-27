"""Reddit public-JSON fetcher — no OAuth required.

Reddit's compliance gate (Responsible Builder Policy + Free-Tier API
Form) makes the OAuth path painful for a single-operator personal
bot. For our use case — **counting and timestamping recent mentions
of a ticker across a few subreddits** — we don't actually need OAuth.
The public JSON endpoints expose:

* search-within-subreddit:
  ``https://www.reddit.com/r/<sub>/search.json?q=<ticker>&restrict_sr=1&sort=new``
* new posts in a subreddit:
  ``https://www.reddit.com/r/<sub>/new.json``

Rate limits without OAuth: 60 req/min for an unidentified caller,
100 req/min with a descriptive User-Agent. Our cadence (one query per
(sub, ticker) every ~5 minutes for ~10 tickers across ~5 subreddits)
sits well under that.

This module produces ``Mention`` rows that feed straight into
:func:`halal_trader.sentiment.velocity.compute_velocity`. The
sentiment manager swaps from PRAW-style auth to this fetcher with no
downstream change.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from halal_trader.sentiment.velocity import Mention

logger = logging.getLogger(__name__)


# Default subreddit set — small + high signal-to-noise. Operator can
# extend per fetcher instance. Crypto-side defaults; stock subs are
# wired via the optional `subreddits` constructor arg.
DEFAULT_CRYPTO_SUBS: tuple[str, ...] = (
    "CryptoCurrency",
    "Bitcoin",
    "ethfinance",
    "CryptoMarkets",
)
DEFAULT_STOCK_SUBS: tuple[str, ...] = (
    "wallstreetbets",
    "stocks",
    "investing",
    "StockMarket",
)

_API_BASE = "https://www.reddit.com"
_CACHE_TTL_S = 5 * 60  # 5 minutes — Reddit results don't churn that fast


@dataclass
class _CacheEntry:
    fetched_at: float
    mentions: list[Mention]


@dataclass
class RedditPublicFetcher:
    """Pulls recent ticker mentions from Reddit's public JSON endpoints.

    No OAuth, no client_id, no secret. Reddit's API ToS still requires
    a unique ``User-Agent`` that identifies you — pass it as
    ``user_agent``. Empty user_agent → fetch returns ``[]`` so the
    cycle degrades cleanly (Reddit will 429 a generic UA on the third
    request anyway).
    """

    user_agent: str = "halal-trader/0.1"
    subreddits: tuple[str, ...] = DEFAULT_CRYPTO_SUBS
    limit_per_sub: int = 100
    _client: Any | None = None
    _cache: dict[str, _CacheEntry] = field(default_factory=dict)

    async def fetch_for_symbols(self, symbols: Sequence[str]) -> list[Mention]:
        """Pull recent mentions for every (subreddit, symbol) pair.

        Errors per (sub, symbol) are isolated — a flaky subreddit
        doesn't take out the whole feed. Cached for 5 minutes per pair.
        """
        if not symbols or not self.user_agent:
            return []

        out: list[Mention] = []
        for sym in symbols:
            for sub in self.subreddits:
                cache_key = f"{sub}:{sym.upper()}"
                cached = self._cache.get(cache_key)
                if cached and (time.monotonic() - cached.fetched_at) < _CACHE_TTL_S:
                    out.extend(cached.mentions)
                    continue
                try:
                    rows = await self._search(sub, sym)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("reddit fetch %s/%s failed: %s", sub, sym, exc)
                    continue
                self._cache[cache_key] = _CacheEntry(
                    fetched_at=time.monotonic(), mentions=list(rows)
                )
                out.extend(rows)
        return out

    async def _search(self, subreddit: str, symbol: str) -> list[Mention]:
        """One ``r/<sub>/search.json?q=<sym>`` request → list[Mention]."""
        client = await self._get_client()
        url = f"{_API_BASE}/r/{subreddit}/search.json"
        params = {
            "q": symbol,
            "restrict_sr": "1",
            "sort": "new",
            "limit": str(self.limit_per_sub),
            "t": "day",  # search within the last 24h
        }
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            logger.debug("reddit %s/%s returned %d", subreddit, symbol, resp.status_code)
            return []
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            return []
        children = (data.get("data", {}) or {}).get("children", []) or []
        out: list[Mention] = []
        for child in children:
            d = child.get("data", {}) or {}
            ts_raw = d.get("created_utc")
            if not isinstance(ts_raw, int | float):
                continue
            ts = datetime.fromtimestamp(float(ts_raw), tz=UTC)
            score = float(d.get("score") or 0)
            out.append(
                Mention(
                    symbol=symbol.upper(),
                    timestamp=ts,
                    source=f"reddit:{subreddit}",
                    score=score,
                )
            )
        return out

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                timeout=10.0,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json",
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
