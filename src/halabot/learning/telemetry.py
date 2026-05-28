"""Decision telemetry writers — persist conviction.scored + policy.target_changed.

Two read-only subscribers that durably record the decision stream (INV-5) so the
dashboard can replay a decision by ``correlation_id`` and the learning loop can
study scoring/sizing behavior. Neither is a calibration *input* (the calibrator
trains on ``outcome.entry_belief`` only — no mid-trade leakage); these are history.

Writes are best-effort: a DB hiccup is logged and skipped (the bus already
isolates handler failures), never blocking dispatch.
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.platform.bus import EventBus, Subscription
from halabot.platform.db import conviction_score as _conviction_table
from halabot.platform.db import target_weight as _target_table
from halabot.platform.events import Event, EventType

logger = logging.getLogger(__name__)


class ConvictionScoreWriter:
    """Persists one ``hb_conviction_score`` row per ``conviction.scored`` event."""

    def __init__(self, *, bus: EventBus, engine: AsyncEngine) -> None:
        self._bus = bus
        self._engine = engine
        self._subs: list[Subscription] = []
        self.written = 0

    def start(self) -> None:
        self._subs.append(self._bus.subscribe({EventType.CONVICTION_SCORED}, self._on_scored))

    def stop(self) -> None:
        for sub in self._subs:
            sub.unsubscribe()
        self._subs.clear()

    async def _on_scored(self, event: Event) -> None:
        asset = event.asset
        if asset is None:
            return
        p = event.payload
        try:
            async with self._engine.begin() as conn:
                await conn.execute(
                    sa.insert(_conviction_table).values(
                        asset=asset,
                        ts=event.ts,
                        raw_score=float(p.get("raw", 0.0)),
                        calibrated=float(p.get("calibrated", 0.0)),
                        features=p.get("features", {}),
                        belief_version=int(p.get("belief_version", 0)),
                    )
                )
            self.written += 1
        except Exception as exc:  # noqa: BLE001 — telemetry write is best-effort
            logger.warning("conviction_score write failed for %s: %r", asset, exc)


class TargetWeightWriter:
    """Persists one ``hb_target_weight`` row per ``policy.target_changed`` event."""

    def __init__(self, *, bus: EventBus, engine: AsyncEngine) -> None:
        self._bus = bus
        self._engine = engine
        self._subs: list[Subscription] = []
        self.written = 0

    def start(self) -> None:
        self._subs.append(
            self._bus.subscribe({EventType.POLICY_TARGET_CHANGED}, self._on_target)
        )

    def stop(self) -> None:
        for sub in self._subs:
            sub.unsubscribe()
        self._subs.clear()

    async def _on_target(self, event: Event) -> None:
        asset = event.asset
        if asset is None:
            return
        p = event.payload
        try:
            async with self._engine.begin() as conn:
                await conn.execute(
                    sa.insert(_target_table).values(
                        asset=asset,
                        ts=event.ts,
                        target_weight=float(p.get("target_weight", 0.0)),
                        current_weight=float(p.get("current_weight", 0.0)),
                        reason=str(p.get("reason", "")),
                        belief_version=int(p.get("belief_version", 0)),
                    )
                )
            self.written += 1
        except Exception as exc:  # noqa: BLE001 — telemetry write is best-effort
            logger.warning("target_weight write failed for %s: %r", asset, exc)
