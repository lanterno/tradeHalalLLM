"""Shadow outcome tracker — hypothetical fills marked to price → outcomes.

Subscribes to ``policy.trade_proposed`` (which now carries the decision price),
maintains a per-asset hypothetical position (VWAP entry), and on each
reduce/close writes an ``hb_outcome`` row with the realized return and the
entry-belief snapshot. Read-only and hypothetical — no broker, no orders.

In-memory positions reset on restart (a half-open hypothetical position is
dropped) — acceptable for the shadow; the durable record is the closed outcomes.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.belief.store import BeliefStore
from halabot.platform.bus import EventBus, Subscription
from halabot.platform.db import outcome as _outcome_table
from halabot.platform.events import Event, EventType

logger = logging.getLogger(__name__)

_EPS = 1e-9


@dataclass
class _Position:
    weight: float
    entry_vwap: float
    open_ts: datetime
    belief_version: int


class ShadowOutcomeTracker:
    def __init__(
        self,
        *,
        bus: EventBus,
        engine: AsyncEngine,
        store: BeliefStore,
        win_threshold_pct: float = 0.002,
        on_close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._bus = bus
        self._engine = engine
        self._store = store
        self._win_threshold = win_threshold_pct
        # Called after each closed outcome is written (the calibrator retrainer
        # hooks here to refit off accumulated outcomes — L8).
        self._on_close = on_close
        self._positions: dict[str, _Position] = {}
        self._subs: list[Subscription] = []
        self.closed_count = 0

    def start(self) -> None:
        self._subs.append(self._bus.subscribe({EventType.POLICY_TRADE_PROPOSED}, self._on_proposal))

    def stop(self) -> None:
        for sub in self._subs:
            sub.unsubscribe()
        self._subs.clear()

    async def _on_proposal(self, event: Event) -> None:
        p = event.payload
        asset = event.asset
        price = p.get("price")
        if asset is None or price is None or price <= 0:
            return  # can't mark without a price
        delta = float(p.get("weight_delta", 0.0))
        belief_version = int(p.get("belief_version", 0))
        ts = event.ts

        pos = self._positions.get(asset)
        if delta > 0:  # buy / add → blend VWAP
            if pos is None or pos.weight <= _EPS:
                self._positions[asset] = _Position(
                    weight=delta, entry_vwap=price, open_ts=ts, belief_version=belief_version
                )
            else:
                total = pos.weight + delta
                pos.entry_vwap = (pos.entry_vwap * pos.weight + price * delta) / total
                pos.weight = total
            return

        # sell / reduce → realize on the closed portion
        if pos is None or pos.weight <= _EPS:
            return
        closed = min(abs(delta), pos.weight)
        return_pct = (price - pos.entry_vwap) / pos.entry_vwap if pos.entry_vwap > 0 else 0.0
        hold_seconds = max(0, int((ts - pos.open_ts).total_seconds()))
        await self._write_outcome(
            asset=asset,
            entry_ts=pos.open_ts,
            exit_ts=ts,
            entry_price=pos.entry_vwap,
            exit_price=price,
            closed_weight=closed,
            return_pct=return_pct,
            hold_seconds=hold_seconds,
            belief_version=pos.belief_version,
            reason=str(p.get("reason", "")),
        )
        self.closed_count += 1
        pos.weight -= closed
        if pos.weight <= _EPS:
            self._positions.pop(asset, None)
        if self._on_close is not None:
            try:
                await self._on_close()
            except Exception as exc:  # noqa: BLE001 — a retrain failure must not break the tracker
                logger.warning("on_close hook failed: %r", exc)

    async def _write_outcome(
        self,
        *,
        asset: str,
        entry_ts: datetime,
        exit_ts: datetime,
        entry_price: float,
        exit_price: float,
        closed_weight: float,
        return_pct: float,
        hold_seconds: int,
        belief_version: int,
        reason: str,
    ) -> None:
        entry_belief = await self._entry_belief_snapshot(asset, belief_version)
        label = 1 if return_pct > self._win_threshold else 0
        async with self._engine.begin() as conn:
            await conn.execute(
                sa.insert(_outcome_table).values(
                    asset=asset,
                    entry_ts=entry_ts,
                    exit_ts=exit_ts,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    closed_weight=closed_weight,
                    return_pct=return_pct,
                    hold_seconds=hold_seconds,
                    belief_version=belief_version,
                    entry_belief=entry_belief,
                    label=label,
                    reason=reason,
                    created_at=datetime.now(UTC),
                )
            )

    async def _entry_belief_snapshot(self, asset: str, version: int) -> dict[str, object] | None:
        b = await self._store.get_version(asset, version)
        if b is None:
            return None
        # Compact entry-time features (the full belief is recoverable by version).
        return {
            "regime": str(b.regime),
            "regime_confidence": b.regime_confidence,
            "direction": str(b.direction),
            "conviction": b.conviction,
            "conviction_raw": b.conviction_raw,
        }
