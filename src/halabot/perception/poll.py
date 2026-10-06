"""Polling source base — periodic fetch → map → emit, with dedup.

Subclasses implement ``fetch`` (hit the feed), ``to_event`` (map one raw item
to an :class:`Event`, or None to drop), and optionally ``dedup_key`` (so a
re-seen item isn't re-emitted — the reactor's seen-set, generalized). A fetch
or map failure is logged and skipped for that tick; the loop continues (INV-2).

The seen set forgets a key only once the feed has stopped returning it for
``seen_ttl_s``. It used to drop an arbitrary half of itself at 2,000 keys while
the feed still returned those items, and the persisted store was read only at
start-up, so each prune re-published about a thousand headlines as new: 535,746
news events for 22,900 distinct (asset, url) pairs. A key missing from memory
is now also looked up in the persisted store before its item is re-published.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from halabot.perception.base import Emit
from halabot.perception.dedup import DedupStore
from halabot.platform.events import Event

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]
# Keep a key this long after the feed last returned its item. Three days is
# the persisted store's retention and covers every source's lookback window.
_SEEN_TTL_S = 3 * 86400.0
# A memory bound only, far above any feed's working set (oldest go first).
_SEEN_HARD_CAP = 100_000


class PollingSource(ABC):
    def __init__(
        self,
        name: str,
        *,
        interval_s: float,
        sleep: Sleep = asyncio.sleep,
        dedup_store: DedupStore | None = None,
        seen_ttl_s: float = _SEEN_TTL_S,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._interval = interval_s
        self._sleep = sleep
        # key -> when the feed last returned it (monotonic seconds).
        self._seen: dict[str, float] = {}
        self._seen_ttl = seen_ttl_s
        self._monotonic = monotonic
        # Persisted dedup (survives restarts) — keyed by this source's name as the
        # namespace. None = in-memory only (tests / no DB).
        self._dedup = dedup_store
        self._primed = False

    @abstractmethod
    async def fetch(self) -> list[Any]:
        """Return the current batch of raw items from the feed."""

    @abstractmethod
    def to_event(self, raw: Any) -> Event | None:
        """Map one raw item to an observation Event, or None to drop it."""

    def dedup_key(self, raw: Any) -> str | None:
        """Stable key to suppress re-emitting a seen item; None = never dedup."""
        return None

    def emitted(self, raw: Any) -> None:
        """Called once ``raw``'s event has been published (a source's own bookkeeping)."""
        return None

    async def _prime(self) -> None:
        """Load persisted seen-keys once, so a restart doesn't re-emit (INV-2)."""
        if self._primed:
            return
        self._primed = True
        if self._dedup is not None:
            try:
                keys = await self._dedup.load(self.name)
            except Exception as exc:  # noqa: BLE001 — a dedup-store hiccup must not block the feed
                logger.warning("source %s dedup load failed: %r", self.name, exc)
                return
            now = self._monotonic()
            self._seen.update(dict.fromkeys(keys, now))

    async def _known_elsewhere(self, keys: list[str]) -> set[str]:
        """Which of ``keys`` (absent from memory) the persisted store has seen."""
        if not keys or self._dedup is None:
            return set()
        try:
            return await self._dedup.contains(self.name, keys)
        except Exception as exc:  # noqa: BLE001 — on a store hiccup, memory decides alone
            logger.warning("source %s dedup lookup failed: %r", self.name, exc)
            return set()

    async def poll_once(self, emit: Emit) -> int:
        """One fetch → emit cycle. Returns the number of events emitted.

        Swallows fetch/map errors (logged) so a transient feed hiccup skips the
        tick rather than crashing the source (INV-2)."""
        await self._prime()
        try:
            items = await self.fetch()
        except Exception as exc:  # noqa: BLE001
            logger.warning("source %s fetch failed: %r", self.name, exc)
            return 0

        now = self._monotonic()
        keyed = [(raw, self.dedup_key(raw)) for raw in items]
        misses = list({k for _, k in keyed if k is not None and k not in self._seen})
        for known in await self._known_elsewhere(misses):
            self._seen[known] = now

        emitted = 0
        new_keys: list[str] = []
        try:
            for raw, key in keyed:
                if key is not None and key in self._seen:
                    self._seen[key] = now  # still in the feed: keep remembering it
                    continue
                try:
                    event = self.to_event(raw)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("source %s failed to map an item: %r", self.name, exc)
                    continue
                if event is None:
                    continue
                if key is not None:
                    self._seen[key] = now
                    new_keys.append(key)
                await emit(event)
                self.emitted(raw)
                emitted += 1
        finally:
            if new_keys and self._dedup is not None:
                try:
                    await self._dedup.add(self.name, new_keys)
                except Exception as exc:  # noqa: BLE001 — persistence is best-effort; in-memory still dedups
                    logger.warning("source %s dedup persist failed: %r", self.name, exc)

        self._prune_seen(now)
        return emitted

    def _prune_seen(self, now: float) -> None:
        """Forget keys the feed hasn't returned for the TTL (it has moved on)."""
        cutoff = now - self._seen_ttl
        stale = [k for k, t in self._seen.items() if t < cutoff]
        for key in stale:
            del self._seen[key]
        if len(self._seen) > _SEEN_HARD_CAP:
            by_age = sorted(self._seen, key=self._seen.__getitem__)
            for key in by_age[: len(self._seen) - _SEEN_HARD_CAP]:
                del self._seen[key]

    async def run(self, emit: Emit) -> None:
        while True:
            await self.poll_once(emit)
            await self._sleep(self._interval)
