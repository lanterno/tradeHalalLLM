"""Stocks news-momentum event reactor.

Ports the crypto :class:`NewsEventReactor` pattern (CryptoPanic →
emergency-cycle callback) to the stocks side, with two key changes:

1. **Per-symbol polling.** Finnhub's company-news endpoint is keyed
   on a single ticker, so the reactor polls each watchlist symbol in
   turn rather than the bulk-currency CryptoPanic feed.

2. **LLM-scored headlines, not exchange-vote sentiment.** Each new
   headline goes through a :class:`HeadlineClassifier` (default is
   the GPT-4o-mini wrapper) and only events with ``score >= threshold``
   trigger callbacks. The keyword-lexicon
   :func:`sentiment.headline_polarity.classify_headline` stays as the
   sentiment field for downstream prompt rendering, but it's too
   noisy to drive an entry trigger by itself.

The reactor is event-driven on the entry side per the operator's
"fast in, slow out" direction (memory: strategy-fast-in-slow-out):
the slower 15-min cron stays for scheduled scans / risk pruning,
the reactor wins for time-sensitive moves.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

import httpx

logger = logging.getLogger(__name__)

# Finnhub company-news endpoint. Free tier: 60 req/min, far above the
# reactor's actual demand (~1 call per symbol every 60s = 1 call/s on
# a 10-symbol watchlist).
_FINNHUB_API_BASE = "https://finnhub.io/api/v1/company-news"
_HTTP_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class HeadlineClassification:
    """LLM-rendered judgement of a single headline.

    ``score`` is the momentum-impact estimate in ``[0.0, 1.0]``: how
    likely is this headline to drive the stock up in the next 15-30
    minutes. ``tag`` is a coarse bucket (``earnings`` / ``M&A`` /
    ``FDA`` / ``regulatory`` / ``guidance`` / ``other``) used for
    later analytics / per-tag policy tuning. ``rationale`` is a short
    string the LLM emits for the trade journal — never shown to the
    operator at decision time, but logged for post-hoc review.
    """

    score: float
    tag: str = "other"
    rationale: str = ""


class HeadlineClassifier(Protocol):
    """Stable surface the reactor depends on. Production impl wraps
    GPT-4o-mini; tests stub with deterministic returns."""

    async def classify(
        self, *, symbol: str, headline: str, summary: str = ""
    ) -> HeadlineClassification: ...


@dataclass
class StockNewsEvent:
    """A scored news event the reactor decided is worth acting on."""

    symbol: str
    title: str
    source: str
    url: str
    published_at: str
    classification: HeadlineClassification


EventCallback = Callable[[StockNewsEvent], Awaitable[None]]


class StockNewsEventReactor:
    """Polls Finnhub per-symbol, scores headlines via an injected
    classifier, fires callbacks on high-score events.

    Lifecycle mirrors the crypto reactor:
      * :meth:`run` — long-lived poll loop, supervised by the bot.
      * :meth:`stop` — graceful shutdown, closes the HTTP client.
      * :meth:`on_event` — register a callback (e.g. the scheduler's
        emergency-cycle trigger).

    Dedup tracking: keeps the last 1000 (symbol, url) tuples in
    memory. The reactor lives for the process lifetime, so this is
    sufficient — restart clears.

    Operator escape: empty ``api_key`` disables the reactor entirely
    (``enabled`` returns False, :meth:`run` exits immediately). The
    cron cycle keeps working without the reactor.
    """

    _DEFAULT_POLL_INTERVAL_S = 60
    _DEFAULT_SCORE_THRESHOLD = 0.7
    _SEEN_CAP = 1000

    def __init__(
        self,
        *,
        api_key: str,
        symbols: list[str],
        classifier: HeadlineClassifier,
        poll_interval_seconds: int = _DEFAULT_POLL_INTERVAL_S,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
        per_symbol_request_spacing_s: float = 0.5,
    ) -> None:
        self._api_key = api_key
        self._symbols = [s.upper() for s in symbols]
        self._classifier = classifier
        self._poll_interval = poll_interval_seconds
        self._score_threshold = score_threshold
        self._spacing = per_symbol_request_spacing_s
        self._seen: set[tuple[str, str]] = set()
        self._callbacks: list[EventCallback] = []
        self._running = False
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._api_key) and bool(self._symbols)

    def on_event(self, callback: EventCallback) -> None:
        """Register an async callback fired once per scored event."""
        self._callbacks.append(callback)

    async def run(self) -> None:
        """Supervisor entry point — runs the poll loop until cancelled."""
        if not self.enabled:
            logger.info(
                "StockNewsEventReactor disabled — no Finnhub key or empty watchlist"
            )
            return
        self._running = True
        logger.info(
            "StockNewsEventReactor started (symbols=%d, poll=%ds, threshold=%.2f)",
            len(self._symbols),
            self._poll_interval,
            self._score_threshold,
        )
        try:
            await self._poll_loop()
        finally:
            self._running = False
            if self._client is not None and not self._client.is_closed:
                await self._client.aclose()
                self._client = None

    async def stop(self) -> None:
        """External cancel — sets the flag; the in-flight poll exits on
        the next ``asyncio.sleep`` boundary."""
        self._running = False

    async def _poll_loop(self) -> None:
        # Stagger the very first pull so we don't fire the moment
        # the bot starts (gives the scheduler time to wire callbacks).
        await asyncio.sleep(5)
        while self._running:
            try:
                events = await self._scan_all_symbols()
                for event in events:
                    logger.info(
                        "Scored stock-news event: [%s score=%.2f tag=%s] %s — %s",
                        event.symbol,
                        event.classification.score,
                        event.classification.tag,
                        event.title[:80],
                        event.classification.rationale[:120],
                    )
                    for cb in self._callbacks:
                        try:
                            await cb(event)
                        except Exception as exc:  # noqa: BLE001
                            logger.error("StockNewsEvent callback failed: %s", exc)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("StockNewsEventReactor poll error: %s", exc)

            await asyncio.sleep(self._poll_interval)

    async def _scan_all_symbols(self) -> list[StockNewsEvent]:
        """Sweep every watchlist symbol, classify, return high-score events."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S)

        out: list[StockNewsEvent] = []
        for sym in self._symbols:
            try:
                items = await self._fetch_for_symbol(sym)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Finnhub news fetch failed for %s: %s", sym, exc)
                items = []
            for item in items:
                event = await self._maybe_emit(sym, item)
                if event is not None:
                    out.append(event)
            # Polite spacing between per-symbol requests to stay
            # well under the 60 req/min free-tier ceiling.
            if self._spacing > 0:
                await asyncio.sleep(self._spacing)
        return out

    async def _fetch_for_symbol(self, symbol: str) -> list[dict[str, Any]]:
        """Hit Finnhub's company-news endpoint for the last 24h."""
        from datetime import UTC, datetime, timedelta

        assert self._client is not None  # set by _scan_all_symbols
        today = datetime.now(UTC).date()
        params = {
            "symbol": symbol,
            "from": (today - timedelta(days=1)).isoformat(),
            "to": today.isoformat(),
            "token": self._api_key,
        }
        resp = await self._client.get(_FINNHUB_API_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    async def _maybe_emit(
        self, symbol: str, item: dict[str, Any]
    ) -> StockNewsEvent | None:
        """Dedup + classify + threshold-filter a single headline.

        Returns the event when worth firing callbacks, ``None`` otherwise.
        """
        url = str(item.get("url") or "")
        if not url:
            return None
        key = (symbol, url)
        if key in self._seen:
            return None
        self._seen.add(key)
        # Bound the dedup set so a long-running reactor doesn't leak.
        if len(self._seen) > self._SEEN_CAP:
            extras = list(self._seen)[: self._SEEN_CAP // 2]
            for k in extras:
                self._seen.discard(k)

        title = str(item.get("headline") or "").strip()
        if not title:
            return None
        summary = str(item.get("summary") or "")[:500]
        try:
            cls = await self._classifier.classify(
                symbol=symbol, headline=title, summary=summary
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("classifier failed for %s '%s': %s", symbol, title[:60], exc)
            return None

        if cls.score < self._score_threshold:
            logger.debug(
                "Skipping low-score headline (%s, score=%.2f): %s",
                symbol,
                cls.score,
                title[:80],
            )
            return None

        return StockNewsEvent(
            symbol=symbol,
            title=title,
            source=str(item.get("source") or "Finnhub"),
            url=url,
            published_at=str(item.get("datetime") or ""),
            classification=cls,
        )


# ── Default LLM classifier (GPT-4o-mini wrapper) ────────────────


class GPTHeadlineClassifier:
    """LLM-backed classifier — wraps any object exposing
    ``generate_json(prompt, system=...)`` returning a dict with
    ``score`` / ``tag`` / ``rationale`` fields.

    The strategy's existing :class:`LLMBackend` already satisfies
    this surface, so composition reuses one LLM client across the
    bot. Costs at GPT-4o-mini: ~$0.0005 per headline, well inside
    operator budgets at the reactor's ~100-300 classified events/day.
    """

    _SYSTEM_PROMPT = (
        "You are a fast headline scorer for an intraday momentum trader. "
        "Given a single stock headline (+ optional summary), output JSON: "
        '{"score": float in [0,1], "tag": "earnings|M&A|FDA|regulatory|'
        'guidance|macro|product|other", "rationale": "<one short clause>"}. '
        "Score 0.0 = no expected move (rumor / repackaged / dated). "
        "Score 0.5 = mild positive (in-line guidance, small contract). "
        "Score 0.7 = material positive likely to push the stock up 1-3% "
        "in the next 15-30 min (earnings beat, real M&A, FDA approval). "
        "Score 0.9+ = breaking, high-confidence catalyst (surprise beat, "
        "acquisition announcement, large contract win, major upgrade). "
        "Negative-headline scores stay LOW even if material — this is a "
        "LONG-ONLY momentum reactor; we only act on bullish catalysts. "
        "Be conservative: rumours and analyst notes that just repackage "
        "old news score <= 0.3."
    )

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def classify(
        self, *, symbol: str, headline: str, summary: str = ""
    ) -> HeadlineClassification:
        prompt = f"Symbol: {symbol}\nHeadline: {headline}"
        if summary:
            prompt += f"\nSummary: {summary[:500]}"
        try:
            raw = await self._llm.generate_json(prompt, system=self._SYSTEM_PROMPT)
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM classify failed for %s: %s", symbol, exc)
            return HeadlineClassification(score=0.0)
        try:
            score = float(raw.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        tag = str(raw.get("tag", "other"))
        rationale = str(raw.get("rationale", ""))[:200]
        return HeadlineClassification(score=score, tag=tag, rationale=rationale)


__all__ = [
    "GPTHeadlineClassifier",
    "HeadlineClassification",
    "HeadlineClassifier",
    "StockNewsEvent",
    "StockNewsEventReactor",
    "EventCallback",
]
