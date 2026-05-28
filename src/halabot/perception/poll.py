"""Polling source base — periodic fetch → map → emit, with dedup.

Subclasses implement ``fetch`` (hit the feed), ``to_event`` (map one raw item
to an :class:`Event`, or None to drop), and optionally ``dedup_key`` (so a
re-seen item isn't re-emitted — the reactor's seen-set, generalized). A fetch
or map failure is logged and skipped for that tick; the loop continues (INV-2).
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from halabot.perception.base import Emit
from halabot.platform.events import Event

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]
_SEEN_CAP = 2000


class PollingSource(ABC):
    def __init__(self, name: str, *, interval_s: float, sleep: Sleep = asyncio.sleep) -> None:
        self.name = name
        self._interval = interval_s
        self._sleep = sleep
        self._seen: set[str] = set()

    @abstractmethod
    async def fetch(self) -> list[Any]:
        """Return the current batch of raw items from the feed."""

    @abstractmethod
    def to_event(self, raw: Any) -> Event | None:
        """Map one raw item to an observation Event, or None to drop it."""

    def dedup_key(self, raw: Any) -> str | None:
        """Stable key to suppress re-emitting a seen item; None = never dedup."""
        return None

    async def poll_once(self, emit: Emit) -> int:
        """One fetch → emit cycle. Returns the number of events emitted.

        Swallows fetch/map errors (logged) so a transient feed hiccup skips the
        tick rather than crashing the source (INV-2)."""
        try:
            items = await self.fetch()
        except Exception as exc:  # noqa: BLE001
            logger.warning("source %s fetch failed: %r", self.name, exc)
            return 0

        emitted = 0
        for raw in items:
            key = self.dedup_key(raw)
            if key is not None and key in self._seen:
                continue
            try:
                event = self.to_event(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning("source %s failed to map an item: %r", self.name, exc)
                continue
            if event is None:
                continue
            if key is not None:
                self._seen.add(key)
            await emit(event)
            emitted += 1

        self._prune_seen()
        return emitted

    def _prune_seen(self) -> None:
        if len(self._seen) > _SEEN_CAP:
            # Drop an arbitrary half — bounded memory; exact identity of dropped
            # keys doesn't matter (worst case: one duplicate re-emit far later).
            for key in list(self._seen)[: _SEEN_CAP // 2]:
                self._seen.discard(key)

    async def run(self, emit: Emit) -> None:
        while True:
            await self.poll_once(emit)
            await self._sleep(self._interval)
