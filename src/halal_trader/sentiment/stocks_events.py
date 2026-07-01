"""Stocks news-momentum event reactor.

Ports the crypto :class:`NewsEventReactor` pattern (CryptoPanic →
emergency-cycle callback) to the stocks side, with two key changes:

1. **Per-symbol polling.** Finnhub's company-news endpoint is keyed
   on a single ticker, so the reactor polls each watchlist symbol in
   turn rather than the bulk-currency CryptoPanic feed.

2. **LLM-scored headlines, not exchange-vote sentiment.** Each new
   headline goes through a :class:`HeadlineClassifier` (default is
   the GLM wrapper) and only events with ``score >= threshold``
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
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
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
    GLM-5.2; tests stub with deterministic returns."""

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
    # Dedup capacity. Must comfortably exceed a full session's DISTINCT
    # (symbol, url) headlines across the watchlist (~700-1000/day observed), or
    # entries thrash out of the dedup and get re-classified — which burns the
    # daily classify cap on duplicates and floods the logs (live 2026-06-09:
    # ~1000 distinct headlines re-scored ~190x -> 233k DEBUG lines + 5 rotations).
    _SEEN_CAP = 8000

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
        state_path: Path | str | None = None,
        halt_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        self._api_key = api_key
        # Optional kill-switch probe. When engaged, the entry path is already
        # blocked downstream — but classification runs BEFORE that gate, so an
        # ungated reactor burns LLM calls + Finnhub quota scoring catalysts it
        # can never act on (≈850 wasted classify calls over the 06-15..17
        # halt). Gating the whole sweep skips fetch+classify entirely while
        # halted; unseen headlines are left unscored so they classify once on
        # resume. Best-effort: a flaky halt read degrades to "proceed".
        self._halt_check = halt_check
        self._symbols = [s.upper() for s in symbols]
        self._classifier = classifier
        self._poll_interval = poll_interval_seconds
        self._score_threshold = score_threshold
        self._spacing = per_symbol_request_spacing_s
        self._notify_cooldown_s = per_symbol_notify_cooldown_s
        # Insertion-ordered dedup (dict preserves order) so eviction is FIFO —
        # drop the OLDEST seen headlines, not arbitrary ones (a set evicts
        # randomly, which is what let still-relevant headlines fall back out and
        # get re-classified). Value is unused; only the key set matters.
        self._seen: dict[tuple[str, str], None] = {}
        # symbol → wall-clock epoch of last callback fire. Wall-clock
        # (not monotonic) so the cooldown survives a restart — the
        # 2026-05-22 session restarted 5+ times and re-fired the same
        # catalyst's Telegram alert on every boot. 30-min cooldown vs
        # clock skew is a non-issue on a stable host.
        self._last_notify: dict[str, float] = {}
        self._callbacks: list[EventCallback] = []
        self._running = False
        self._client: httpx.AsyncClient | None = None
        # Optional cross-restart persistence. None disables it (tests,
        # ad-hoc runs); the scheduler points it at data/reactor_state.json.
        self._state_path = Path(state_path) if state_path else None
        # Set whenever _seen / _last_notify mutate so we only rewrite the
        # state file when there's something new to persist.
        self._state_dirty = False

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
            logger.info("StockNewsEventReactor disabled — no Finnhub key or empty watchlist")
            return
        self._load_state()
        self._running = True
        logger.info(
            "StockNewsEventReactor started (symbols=%d, poll=%ds, threshold=%.2f, "
            "seen=%d, cooldowns=%d)",
            len(self._symbols),
            self._poll_interval,
            self._score_threshold,
            len(self._seen),
            len(self._last_notify),
        )
        try:
            await self._poll_loop()
        finally:
            self._running = False
            self._save_state()
            if self._client is not None and not self._client.is_closed:
                await self._client.aclose()
                self._client = None

    def _load_state(self) -> None:
        """Restore ``_seen`` + ``_last_notify`` from disk so a restart
        doesn't re-fire alerts for catalysts already handled this window.

        Best-effort: a missing / corrupt / unreadable file resets to
        empty state rather than blocking startup."""
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("reactor state unreadable (%s) — starting fresh", exc)
            return
        seen = raw.get("seen", [])
        self._seen = {
            (str(pair[0]), str(pair[1])): None
            for pair in seen
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        }
        last = raw.get("last_notify", {})
        if isinstance(last, dict):
            self._last_notify = {
                str(k): float(v) for k, v in last.items() if isinstance(v, (int, float))
            }

    def _save_state(self) -> None:
        """Atomically write ``_seen`` + ``_last_notify`` to disk.

        Atomic (temp file + ``os.replace``) so a crash mid-write can't
        leave a half-written file that fails ``_load_state``. No-op when
        persistence is disabled or nothing changed since the last save."""
        if self._state_path is None or not self._state_dirty:
            return
        payload = {
            "seen": [list(pair) for pair in self._seen],
            "last_notify": self._last_notify,
            "saved_at": time.time(),
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self._state_path)
            self._state_dirty = False
        except OSError as exc:
            logger.debug("reactor state save failed: %s", exc)

    async def stop(self) -> None:
        """External cancel — sets the flag; the in-flight poll exits on
        the next ``asyncio.sleep`` boundary."""
        self._running = False

    async def _poll_loop(self) -> None:
        # Stagger the very first pull so we don't fire the moment
        # the bot starts (gives the scheduler time to wire callbacks).
        await asyncio.sleep(5)
        while self._running:
            # Skip the entire sweep (fetch + classify) while the kill-switch
            # is engaged — entries are blocked downstream anyway, so scoring
            # is pure wasted LLM/Finnhub spend. Keep polling at the normal
            # cadence so the reactor wakes within one interval of resume.
            if self._halt_check is not None:
                try:
                    halted = await self._halt_check()
                except Exception as exc:  # noqa: BLE001 — halt read flaked → proceed
                    logger.debug("reactor halt check failed: %s — proceeding", exc)
                    halted = False
                if halted:
                    logger.debug("reactor sweep skipped — kill-switch engaged")
                    await asyncio.sleep(self._poll_interval)
                    continue
            try:
                events = await self._scan_all_symbols()
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
                    now_t = time.time()
                    if self._notify_cooldown_s > 0 and (now_t - last) < self._notify_cooldown_s:
                        continue
                    self._last_notify[event.symbol] = now_t
                    self._state_dirty = True
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
                # Persist after each full sweep so an unclean exit
                # (SIGKILL, watchdog restart) still keeps most of the
                # dedup + cooldown state. ``_save_state`` is a no-op when
                # nothing changed this iteration.
                self._save_state()
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

    async def _maybe_emit(self, symbol: str, item: dict[str, Any]) -> StockNewsEvent | None:
        """Dedup + classify + threshold-filter a single headline.

        Returns the event when worth firing callbacks, ``None`` otherwise.
        """
        url = str(item.get("url") or "")
        if not url:
            return None
        key = (symbol, url)
        if key in self._seen:
            return None
        self._seen[key] = None
        self._state_dirty = True
        # Bound the dedup map so a long-running reactor doesn't leak — evict the
        # OLDEST entries first (insertion order) rather than arbitrary ones, so
        # still-relevant recent headlines stay deduped and aren't re-classified.
        while len(self._seen) > self._SEEN_CAP:
            self._seen.pop(next(iter(self._seen)))

        title = str(item.get("headline") or "").strip()
        if not title:
            return None
        summary = str(item.get("summary") or "")[:500]
        try:
            cls = await self._classifier.classify(symbol=symbol, headline=title, summary=summary)
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


# ── Default LLM classifier (GLM wrapper) ────────────────────────


# OpenAI-compat shape ("insufficient_quota") plus OpenRouter's 402
# "Insufficient credits" — matched case-insensitively below.
_QUOTA_ERROR_MARKERS = ("insufficient_quota", "exceeded your current quota", "insufficient credits")
# Default per-UTC-day call ceiling — pure cost backstop (see config.py).
# Raised 250 -> 1500 on 2026-06-09: a full day is ~700-1000 DISTINCT
# headlines, so 250 silently starved the reactor once dedup stopped
# re-scoring duplicates. Quota-exhaustion call-spam is handled by the
# half-open quota breaker now, not this ceiling.
_DEFAULT_DAILY_CLASSIFY_CAP = 1500
# Half-open recovery window for the quota breaker: after tripping, the breaker
# allows ONE probe call per this interval to test whether quota recovered
# (period reset / operator top-up) WITHOUT a restart. Bounds wasted calls to
# ~1/window while letting the reactor self-heal — the original design latched
# until restart, which left the reactor dead for days after a transient dip.
_QUOTA_RECOVERY_COOLDOWN_S = 3600.0  # 1 hour


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
    bot. Costs at GLM-5.2: ~$0.001 per headline, well inside
    operator budgets at the reactor's ~100-300 classified events/day.

    Two guards added 2026-05-23 after a quota-exhaustion incident
    triggered 3,736 wasted 429 calls to OpenAI in ~9.5h:

    * **Insufficient-quota circuit breaker (half-open).** First
      ``insufficient_quota`` response trips a session flag; subsequent calls
      short-circuit to score=0.0 without hitting the API. One Telegram alert
      fires via the injected ``AlertSink``. After ``_QUOTA_RECOVERY_COOLDOWN_S``
      the breaker goes half-open: it allows ONE probe call to test whether quota
      recovered (period reset / top-up) and clears itself on success — so the
      reactor self-heals without a restart, while a real outage still costs at
      most ~1 probe/hour (not the 3,736-call hammering that motivated the guard).
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
        # Quota-exhausted session flag — set on insufficient_quota. Cleared
        # automatically when a half-open probe succeeds (see classify); a
        # restart also resets it. ``_quota_tripped_at`` is a monotonic stamp
        # used to gate the half-open probe cadence.
        self._quota_exhausted = False
        self._quota_tripped_at: float | None = None
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

    def _quota_recovery_due(self) -> bool:
        """True when the breaker is tripped but the half-open probe window has
        elapsed, so one call should be allowed through to test for recovery."""
        if self._quota_tripped_at is None:
            return True
        return (time.monotonic() - self._quota_tripped_at) >= _QUOTA_RECOVERY_COOLDOWN_S

    async def _trip_quota_breaker(self, *, symbol: str, headline: str) -> None:
        """Flip the session flag + (on the FIRST trip) fire the one-shot alert.
        Re-trips (a failed half-open probe) only re-arm the cooldown, silently."""
        already = self._quota_exhausted
        self._quota_exhausted = True
        self._quota_tripped_at = time.monotonic()
        if already:
            return  # re-armed the half-open cooldown; don't re-log / re-alert
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
        # Guard 1: session-level quota breaker. Short-circuit (no API call)
        # unless the half-open window has elapsed — then fall through to probe.
        if self._quota_exhausted and not self._quota_recovery_due():
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
        # by the GLM provider; the telemetry degrades gracefully
        # too but cost stays 0).
        self._total_successes += 1
        # Half-open probe succeeded → quota recovered; close the breaker so the
        # reactor goes fully live again without needing a restart.
        if self._quota_exhausted:
            self._quota_exhausted = False
            self._quota_tripped_at = None
            logger.info("LLM quota recovered — classifier breaker reset; news reactor live again")
        last_usage = getattr(self._llm, "last_usage", None)
        if last_usage is not None:
            provider = getattr(last_usage, "provider", "") or "unknown"
            self._calls_by_provider[provider] = self._calls_by_provider.get(provider, 0) + 1
            cost = getattr(last_usage, "cost_usd", 0)
            try:
                self._cost_usd_total += float(cost)
            except TypeError, ValueError:
                pass
        try:
            score = float(raw.get("score", 0.0))
        except TypeError, ValueError:
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
