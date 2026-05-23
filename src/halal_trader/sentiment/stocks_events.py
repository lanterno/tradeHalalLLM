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
    # Raised from 0.7 to 0.85 on 2026-05-22 after the NVDA earnings
    # cascade flooded the reactor — the classifier was clustering
    # everything at exactly 0.70 (threshold floor), so 0.7 acted as
    # an "any AI-positive headline" filter, not a "material catalyst"
    # filter. 0.85 forces the LLM to express higher conviction.
    _DEFAULT_SCORE_THRESHOLD = 0.85
    # Per-symbol notification cooldown: even with a tighter threshold,
    # a single catalyst (earnings, M&A) reliably generates 10-30
    # repackaged headlines across publishers. This caps callbacks at
    # one per (symbol, window) so the operator's Telegram doesn't
    # flood and we don't burn budget on stale duplicates.
    # Raised 15 → 30 min on 2026-05-22 19:03 after observing the
    # same NVDA Q1 catalyst re-firing every ~15 min via new
    # headlines (analyst notes, recaps). 30 min still lets a
    # genuinely-new catalyst (different tag, different rationale)
    # land within the cycle cadence.
    _DEFAULT_NOTIFY_COOLDOWN_S = 1800  # 30 min
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
        per_symbol_notify_cooldown_s: int = _DEFAULT_NOTIFY_COOLDOWN_S,
    ) -> None:
        self._api_key = api_key
        self._symbols = [s.upper() for s in symbols]
        self._classifier = classifier
        self._poll_interval = poll_interval_seconds
        self._score_threshold = score_threshold
        self._spacing = per_symbol_request_spacing_s
        self._notify_cooldown_s = per_symbol_notify_cooldown_s
        self._seen: set[tuple[str, str]] = set()
        # symbol → monotonic timestamp of last callback fire
        self._last_notify: dict[str, float] = {}
        self._callbacks: list[EventCallback] = []
        self._running = False
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._api_key) and bool(self._symbols)

    @property
    def classifier(self) -> HeadlineClassifier:
        """Read-only handle to the injected classifier — used by the
        scheduler's EOD hook to pull ``get_telemetry()`` without
        reaching into reactor internals."""
        return self._classifier

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
                import time as _t

                for event in events:
                    # Audit log at DEBUG: every scored event flows
                    # through here. Operators tailing the log can
                    # adjust the level if they want to see the full
                    # classifier output for drift analysis.
                    logger.debug(
                        "Scored stock-news event (audit): [%s score=%.2f tag=%s] %s",
                        event.symbol,
                        event.classification.score,
                        event.classification.tag,
                        event.title[:80],
                    )
                    # Per-symbol notification cooldown — a single
                    # catalyst (earnings, M&A) reliably re-fires across
                    # 10+ repackaged headlines. Only the FIRST event
                    # per (symbol, window) gets the INFO log + callback
                    # dispatch, so external watchers (Monitor, Telegram)
                    # see one signal per real catalyst.
                    last = self._last_notify.get(event.symbol, 0.0)
                    now_t = _t.monotonic()
                    if (
                        self._notify_cooldown_s > 0
                        and (now_t - last) < self._notify_cooldown_s
                    ):
                        continue
                    self._last_notify[event.symbol] = now_t
                    logger.info(
                        "Reactor dispatch: [%s score=%.2f tag=%s] %s — %s",
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


_QUOTA_ERROR_MARKERS = ("insufficient_quota", "exceeded your current quota")
# Default per-UTC-day call ceiling. Yesterday's reactor scored 188
# headlines in a 6h session; 250 leaves room for an active day without
# silently slipping into an unbounded spend (the trigger for tonight's
# 3,736-call 429 hammering).
_DEFAULT_DAILY_CLASSIFY_CAP = 250


def _is_quota_exhausted(error: Exception) -> bool:
    """True when the provider says we're out of credits (vs a rate
    limit / transient 5xx). Quota exhaustion is irrecoverable inside a
    session — the credit gets topped up by a human, not by waiting."""
    msg = str(error).lower()
    return any(marker in msg for marker in _QUOTA_ERROR_MARKERS)


class GPTHeadlineClassifier:
    """LLM-backed classifier — wraps any object exposing
    ``generate_json(prompt, system=...)`` returning a dict with
    ``score`` / ``tag`` / ``rationale`` fields.

    The strategy's existing :class:`LLMBackend` already satisfies
    this surface, so composition reuses one LLM client across the
    bot. Costs at GPT-4o-mini: ~$0.0005 per headline, well inside
    operator budgets at the reactor's ~100-300 classified events/day.

    Two guards added 2026-05-23 after a quota-exhaustion incident
    triggered 3,736 wasted 429 calls to OpenAI in ~9.5h:

    * **Insufficient-quota circuit breaker.** First ``insufficient_quota``
      response trips a session flag; subsequent calls short-circuit to
      score=0.0 without hitting the API. One Telegram alert fires via
      the injected ``AlertSink`` (deduped by its own cooldown).
      Auto-resets on bot restart — a human top-up is the only valid
      recovery path.
    * **Daily classify ceiling.** Hard ceiling on attempted API calls
      per UTC day (default 250). When tripped, classify returns 0.0
      silently and logs once per day. Resets at UTC midnight or
      restart.
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

    def __init__(
        self,
        llm: Any,
        *,
        alert_sink: Any = None,
        daily_classify_cap: int = _DEFAULT_DAILY_CLASSIFY_CAP,
    ) -> None:
        self._llm = llm
        self._alert_sink = alert_sink
        self._daily_cap = max(0, daily_classify_cap)
        # Quota-exhausted session flag — set on first insufficient_quota
        # and never cleared (a process restart is the only reset).
        self._quota_exhausted = False
        # Daily counter — counts every attempted API call (success or
        # failure) since both consume rate budget and we want a single
        # truth-source for "how many calls did we make today".
        self._daily_count = 0
        self._daily_reset_date = ""
        # Day-scoped log dedup so we only emit one "cap reached" warning
        # per UTC day instead of one per skipped headline.
        self._cap_log_date = ""
        # Telemetry — cumulative counters since process start. The
        # 2026-05-22 quota incident proved we couldn't see classifier
        # spend at all; this surfaces enough to answer "where did the
        # budget go" from the EOD report without standing up a separate
        # persistence layer for what's already a noisy stream.
        self._total_calls = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_short_circuits = 0
        self._calls_by_provider: dict[str, int] = {}
        self._cost_usd_total: float = 0.0

    def _roll_daily_counter(self) -> None:
        from datetime import UTC, datetime

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_count = 0
            self._daily_reset_date = today

    def _daily_cap_reached(self) -> bool:
        if self._daily_cap == 0:
            return False
        self._roll_daily_counter()
        if self._daily_count < self._daily_cap:
            return False
        if self._cap_log_date != self._daily_reset_date:
            logger.warning(
                "Classifier daily cap reached (%d calls); skipping further classify "
                "calls until UTC midnight",
                self._daily_cap,
            )
            self._cap_log_date = self._daily_reset_date
        return True

    async def _trip_quota_breaker(self, *, symbol: str, headline: str) -> None:
        """Fire the one-shot alert + flip the session flag."""
        self._quota_exhausted = True
        logger.warning(
            "LLM quota exhausted (first detected on %s '%s') — classifier will "
            "short-circuit to score=0.0 until bot restart; top up provider quota "
            "to revive the reactor",
            symbol,
            headline[:80],
        )
        if self._alert_sink is None:
            return
        try:
            await self._alert_sink.notify(
                "classifier.quota_exhausted",
                (
                    "Reactor classifier hit insufficient_quota on the LLM "
                    f"provider (first headline: {symbol} '{headline[:80]}'). "
                    "All further classifies will return score=0.0 until you "
                    "top up quota and restart the bot. The 'fast in' half of "
                    "the strategy is OFFLINE."
                ),
                market="stocks",
                severity="critical",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("AlertSink notify failed: %s", exc)

    async def classify(
        self, *, symbol: str, headline: str, summary: str = ""
    ) -> HeadlineClassification:
        # Guard 1: session-level quota breaker. No API call.
        if self._quota_exhausted:
            self._total_short_circuits += 1
            return HeadlineClassification(score=0.0)
        # Guard 2: daily ceiling. No API call.
        if self._daily_cap_reached():
            self._total_short_circuits += 1
            return HeadlineClassification(score=0.0)

        prompt = f"Symbol: {symbol}\nHeadline: {headline}"
        if summary:
            prompt += f"\nSummary: {summary[:500]}"
        # Count the attempt BEFORE the call so a hanging provider can't
        # let us burn past the cap while in flight.
        self._roll_daily_counter()
        self._daily_count += 1
        self._total_calls += 1
        try:
            raw = await self._llm.generate_json(prompt, system=self._SYSTEM_PROMPT)
        except Exception as exc:  # noqa: BLE001
            self._total_failures += 1
            if _is_quota_exhausted(exc):
                await self._trip_quota_breaker(symbol=symbol, headline=headline)
            else:
                logger.debug("LLM classify failed for %s: %s", symbol, exc)
            return HeadlineClassification(score=0.0)
        # Success path: roll up per-provider usage from the LLM's
        # ``last_usage`` if the provider populated it (every BaseLLM
        # subclass calls _record_usage on success, so this is reliable
        # for the cloud providers; Ollama populates provider="ollama"
        # too but cost stays 0).
        self._total_successes += 1
        last_usage = getattr(self._llm, "last_usage", None)
        if last_usage is not None:
            provider = getattr(last_usage, "provider", "") or "unknown"
            self._calls_by_provider[provider] = (
                self._calls_by_provider.get(provider, 0) + 1
            )
            cost = getattr(last_usage, "cost_usd", 0)
            try:
                self._cost_usd_total += float(cost)
            except (TypeError, ValueError):
                pass
        try:
            score = float(raw.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        tag = str(raw.get("tag", "other"))
        rationale = str(raw.get("rationale", ""))[:200]
        return HeadlineClassification(score=score, tag=tag, rationale=rationale)

    def get_telemetry(self) -> dict[str, Any]:
        """Snapshot of classifier health + cumulative spend since start.

        Read by the EOD routine to enrich the daily Telegram summary so
        the operator can see classifier spend without grepping logs.
        Numbers are cumulative across the bot's process lifetime —
        bot restart resets, same as the quota breaker.
        """
        self._roll_daily_counter()
        return {
            "total_calls": self._total_calls,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "total_short_circuits": self._total_short_circuits,
            "calls_today": self._daily_count,
            "daily_cap": self._daily_cap,
            "quota_exhausted": self._quota_exhausted,
            "calls_by_provider": dict(self._calls_by_provider),
            "cost_usd_total": round(self._cost_usd_total, 4),
        }


__all__ = [
    "GPTHeadlineClassifier",
    "HeadlineClassification",
    "HeadlineClassifier",
    "StockNewsEvent",
    "StockNewsEventReactor",
    "EventCallback",
]
