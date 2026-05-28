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

from halabot.belief.schema import ComplianceVerdict, EvidenceItem
from halabot.belief.updater import BeliefUpdater
from halabot.cognition.bars import Bar, BarBuffer
from halabot.cognition.base import Interpreter
from halabot.cognition.worker import BeliefSink, InlineBeliefSink
from halabot.platform.bus import EventBus, Subscription
from halabot.platform.events import Event, EventType

logger = logging.getLogger(__name__)

# Observation event types replayed during bootstrap (compliance is re-established
# live by the seed / Zoya source, not replayed).
_OBSERVATION_TYPES = frozenset(
    {EventType.OBSERVATION_BAR, EventType.OBSERVATION_NEWS, EventType.OBSERVATION_PRICE}
)


class CognitionRouter:
    def __init__(
        self,
        *,
        bus: EventBus,
        buffer: BarBuffer,
        interpreters: list[Interpreter],
        updater: BeliefUpdater | None = None,
        sink: BeliefSink | None = None,
    ) -> None:
        if sink is None:
            if updater is None:
                raise ValueError("CognitionRouter requires a sink or an updater")
            sink = InlineBeliefSink(updater)
        self._bus = bus
        self._sink = sink
        # Bootstrap replay must apply synchronously to completion before the live
        # stream starts (Appendix F), even when the live sink is the async worker.
        # An inline sink over the updater guarantees that; fall back to the live
        # sink when no updater was injected.
        self._replay_sink: BeliefSink = InlineBeliefSink(updater) if updater else sink
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
            EventType.COMPLIANCE_VERDICT,
        }
        self._subs.append(self._bus.subscribe(types, self._on_event))

    def stop(self) -> None:
        for sub in self._subs:
            sub.unsubscribe()
        self._subs.clear()

    @property
    def known_assets(self) -> frozenset[str]:
        return frozenset(self._known_assets)

    async def bootstrap(
        self, *, since: datetime, until: datetime, now: datetime
    ) -> frozenset[str]:
        """Warm beliefs by replaying ``observation.*`` from the event log (Appendix F).

        Each event is interpreted at its OWN ts (event-time) with ``is_replay=True``
        so decay ages it relative to *then* and NO invalidation/order side-effects
        fire (replay must never trade against historical prices). After the window
        replays, each warmed belief is decayed forward to ``now``. Applied inline
        and synchronously, so it completes before the live stream is subscribed;
        ``merge``'s event_id dedup absorbs any overlap with buffered live events.

        Call BEFORE :meth:`start`. Returns the set of warmed assets.
        """
        warmed: set[str] = set()
        count = 0
        async for event in self._bus.replay(since=since, until=until):
            if event.type not in _OBSERVATION_TYPES or event.asset is None:
                continue
            asset = event.asset
            if event.type == EventType.OBSERVATION_BAR:
                self._buffer.append(asset, _parse_bar(event))
            self._known_assets.add(asset)
            warmed.add(asset)
            evidence: list[EvidenceItem] = []
            for itp in self._by_type.get(event.type, []):
                try:
                    evidence.extend(await itp.interpret(event))
                except Exception as exc:  # noqa: BLE001 — a bad interpreter yields no evidence
                    logger.error("replay interpreter %s failed: %r", type(itp).__name__, exc)
            if evidence or event.type == EventType.OBSERVATION_BAR:
                await self._replay_sink.evidence(asset, event.ts, evidence, is_replay=True)
            count += 1
        # Bring each warmed belief to the present (decay-only, still suppressed).
        for asset in sorted(warmed):
            await self._replay_sink.evidence(asset, now, [], is_replay=True)
        if warmed:
            logger.info(
                "bootstrap warmed %d assets from %d replayed observations", len(warmed), count
            )
        return frozenset(warmed)

    async def _on_event(self, event: Event) -> None:
        if event.type == EventType.SYSTEM_HEARTBEAT:
            await self._on_heartbeat(event)
            return
        if event.type == EventType.COMPLIANCE_VERDICT:
            await self._on_compliance(event)
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
            await self._sink.evidence(asset, event.ts, evidence)

    async def _on_heartbeat(self, event: Event) -> None:
        # Decay-only pass for every known asset so conviction fades on the
        # passage of time even with no new data (fix R-08).
        for asset in sorted(self._known_assets):
            await self._sink.evidence(asset, event.ts, [])

    async def _on_compliance(self, event: Event) -> None:
        asset = event.asset
        if asset is None:
            return
        self._known_assets.add(asset)
        p = event.payload
        verdict = ComplianceVerdict(
            asset=asset,
            status=p.get("status", "doubtful"),
            detail=str(p.get("detail", "")),
            screened_at=event.ts,
            screening_id=p.get("screening_id"),
            transient_error=bool(p.get("transient_error", False)),
        )
        await self._sink.compliance(asset, verdict, now=event.ts)


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
