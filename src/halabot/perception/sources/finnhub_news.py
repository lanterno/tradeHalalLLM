"""Finnhub news source — emits observation.news for the halal universe.

Polls Finnhub company-news per symbol and emits one ``observation.news`` per
new headline (deduped by ``asset:url``), attaching a cheap lexicon polarity so
cognition's ``NewsLexiconInterpreter`` can turn it into evidence immediately.
LLM headline scoring stays the sparse cognition path — perception just reports
"we saw news" (always works, even if the LLM is down — INV-1). Read-only.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from halabot.perception.poll import PollingSource
from halabot.platform.clock import Clock
from halabot.platform.events import Event, EventType, new_event

logger = logging.getLogger(__name__)

_FINNHUB_NEWS = "https://finnhub.io/api/v1/company-news"
_HTTP_TIMEOUT_S = 10.0
# Lexicon tag → directional polarity. "neutral" → None so the interpreter
# abstains (no fabricated signal); the magnitude is modest so one headline
# can't dominate a belief on its own.
_POLARITY: dict[str, float | None] = {"positive": 0.5, "negative": -0.5, "neutral": None}

UniverseProvider = Callable[[], Awaitable[list[str]]]


class FinnhubNewsSource(PollingSource):
    def __init__(
        self,
        api_key: str,
        universe: UniverseProvider,
        clock: Clock,
        *,
        lookback_days: int = 1,
        per_symbol_spacing_s: float = 0.2,
        interval_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        dedup_store: Any | None = None,
    ) -> None:
        super().__init__(
            "finnhub-news", interval_s=interval_s, sleep=sleep, dedup_store=dedup_store
        )
        self._api_key = api_key
        self._universe = universe
        self._clock = clock
        self._lookback = lookback_days
        self._spacing = per_symbol_spacing_s
        self._client = client or httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S)

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    async def fetch(self) -> list[Any]:
        if not self.enabled:
            return []
        symbols = await self._universe()
        today = datetime.now(UTC).date()
        params_base = {
            "from": (today - timedelta(days=self._lookback)).isoformat(),
            "to": today.isoformat(),
            "token": self._api_key,
        }
        out: list[dict[str, Any]] = []
        for sym in symbols:
            try:
                resp = await self._client.get(
                    _FINNHUB_NEWS, params={**params_base, "symbol": sym}
                )
                resp.raise_for_status()
                items = resp.json()
            except Exception as exc:  # noqa: BLE001 — one symbol's failure skips it
                logger.warning("finnhub-news fetch failed for %s: %r", sym, exc)
                items = []
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        out.append({"_asset": sym, **item})
            if self._spacing > 0:
                await self._sleep(self._spacing)
        return out

    def to_event(self, raw: dict[str, Any]) -> Event | None:
        headline = str(raw.get("headline") or "").strip()
        url = str(raw.get("url") or "")
        if not headline or not url:
            return None
        polarity = _lexicon_polarity(headline)
        return new_event(
            self._clock,
            EventType.OBSERVATION_NEWS,
            source="finnhub-news",
            asset=raw["_asset"],
            payload={
                "headline": headline[:300],
                "summary": str(raw.get("summary") or "")[:500],
                "url": url,
                "published_at": str(raw.get("datetime") or ""),
                "source": str(raw.get("source") or "Finnhub"),
                "lexicon_polarity": polarity,
            },
        )

    def dedup_key(self, raw: dict[str, Any]) -> str | None:
        return f"{raw['_asset']}:{raw.get('url', '')}"

    async def aclose(self) -> None:
        await self._client.aclose()


def _lexicon_polarity(headline: str) -> float | None:
    """Cheap, deterministic polarity via the legacy lexicon classifier."""
    from halal_trader.sentiment.headline_polarity import classify_headline

    return _POLARITY.get(classify_headline(headline))
