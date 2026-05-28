"""CognitionRouter — observation stream → evidence → belief updates (L2 → L3).

Subscribes to ``observation.*`` (and ``system.heartbeat``), runs the matching
interpreters, and feeds their evidence to the :class:`BeliefUpdater`. This is
the "always-on understanding" loop: beliefs form continuously from live events
with no fixed cycle. The router subscribes only to observations/heartbeat (never
``belief.*``), so the updater publishing ``belief.updated`` can't loop back in.

Read-only by construction (Phase 2): it produces beliefs, never orders.
"""

from __future__ import annotations

import logging
from datetime import datetime

from halabot.belief.schema import EvidenceItem
from halabot.belief.updater import BeliefUpdater
from halabot.cognition.bars import Bar, BarBuffer
from halabot.cognition.base import Interpreter
from halabot.platform.bus import EventBus, Subscription
from halabot.platform.events import Event, EventType

logger = logging.getLogger(__name__)


class CognitionRouter:
    def __init__(
        self,
        *,
        bus: EventBus,
        updater: BeliefUpdater,
        buffer: BarBuffer,
        interpreters: list[Interpreter],
    ) -> None:
        self._bus = bus
        self._updater = updater
        self._buffer = buffer
        self._by_type: dict[EventType, list[Interpreter]] = {}
        for itp in interpreters:
            for t in itp.consumes:
                self._by_type.setdefault(t, []).append(itp)
        self._known_assets: set[str] = set()
        self._subs: list[Subscription] = []

    def start(self) -> None:
        types = set(self._by_type) | {
            EventType.OBSERVATION_BAR,
            EventType.SYSTEM_HEARTBEAT,
        }
        self._subs.append(self._bus.subscribe(types, self._on_event))

    def stop(self) -> None:
        for sub in self._subs:
            sub.unsubscribe()
        self._subs.clear()

    @property
    def known_assets(self) -> frozenset[str]:
        return frozenset(self._known_assets)

    async def _on_event(self, event: Event) -> None:
        if event.type == EventType.SYSTEM_HEARTBEAT:
            await self._on_heartbeat(event)
            return

        asset = event.asset
        if event.type == EventType.OBSERVATION_BAR and asset is not None:
            self._buffer.append(asset, _parse_bar(event))
        if asset is not None:
            self._known_assets.add(asset)

        evidence: list[EvidenceItem] = []
        for itp in self._by_type.get(event.type, []):
            try:
                evidence.extend(await itp.interpret(event))
            except Exception as exc:  # noqa: BLE001 — a bad interpreter yields no evidence (INV-1)
                logger.error(
                    "interpreter %s failed on %s (%s): %r",
                    type(itp).__name__,
                    event.type,
                    event.id,
                    exc,
                )

        # Update on new evidence, or on any bar (the buffer + levels changed).
        if asset is not None and (evidence or event.type == EventType.OBSERVATION_BAR):
            await self._updater.apply_evidence(asset, evidence, now=event.ts)

    async def _on_heartbeat(self, event: Event) -> None:
        # Decay-only pass for every known asset so conviction fades on the
        # passage of time even with no new data (fix R-08).
        for asset in sorted(self._known_assets):
            await self._updater.apply_evidence(asset, [], now=event.ts)


def _parse_bar(event: Event) -> Bar:
    p = event.payload
    return Bar(
        o=float(p["o"]),
        h=float(p["h"]),
        low=float(p["low"]),
        c=float(p["c"]),
        v=float(p.get("v", 0.0)),
        ts=_parse_dt(p.get("bar_ts")) or event.ts,
    )


def _parse_dt(value: object) -> datetime | None:
    return datetime.fromisoformat(value) if isinstance(value, str) else None
