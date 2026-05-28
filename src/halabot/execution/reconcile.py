"""Continuous reconciliation (REARCHITECTURE L6, INV-3, fix R-02). DORMANT.

The DB reconciles *to the broker*, never the reverse. During migration two
engines share one broker account, so reconciliation is **scoped by engine_owner**:
the belief engine reconciles only positions it opened and treats ``legacy``
broker positions as out-of-scope (and vice-versa) — otherwise each engine would
import the other's position as a phantom and they'd fight over the same shares.

``reconcile_plan`` is pure + tested; :class:`Reconciler` emits ``position.reconciled``
events per action. INV-2: a missing quote/position is skipped, never invented.
NEVER instantiated by ``app.build_engine`` — dormant until Phase-4.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from halabot.execution.venue import Venue
from halabot.platform.bus import EventBus
from halabot.platform.clock import Clock
from halabot.platform.events import EventType, new_event

logger = logging.getLogger(__name__)

_EPS = 1e-9
Action = Literal["none", "adjustment", "import", "neutralize", "skip_other_engine"]


@dataclass(frozen=True)
class ReconcileAction:
    asset: str
    db_net: float
    broker_qty: float
    action: Action
    adjustment_qty: float  # signed delta the DB must apply to match broker (0 for skips/none)


def reconcile_plan(
    broker_qty: dict[str, float],
    db_net: dict[str, float],
    owner_of: Callable[[str], str | None],
    *,
    engine_owner: str = "belief",
) -> list[ReconcileAction]:
    """Compute the reconcile action per asset across broker ∪ DB.

    ``owner_of(asset)`` returns the engine that owns the position (from the trade
    rows), or None if unknown/unowned. The engine acts ONLY on positions it
    explicitly owns: a different-engine OR unknown-owner position is skipped
    (fail-safe — never adopt/adjust shares we can't prove are ours, fix R-02)."""
    out: list[ReconcileAction] = []
    for asset in sorted(set(broker_qty) | set(db_net)):
        bq = broker_qty.get(asset, 0.0)
        dq = db_net.get(asset, 0.0)
        owner = owner_of(asset)
        if owner != engine_owner:
            out.append(ReconcileAction(asset, dq, bq, "skip_other_engine", 0.0))
            continue
        if abs(bq - dq) < _EPS:
            out.append(ReconcileAction(asset, dq, bq, "none", 0.0))
        elif abs(dq) > _EPS and abs(bq) < _EPS:
            # DB thinks we hold it; broker is flat → phantom: neutralize the DB.
            out.append(ReconcileAction(asset, dq, bq, "neutralize", -dq))
        elif abs(bq) > _EPS and abs(dq) < _EPS:
            # Broker holds it; no DB record → import (adopt broker truth).
            out.append(ReconcileAction(asset, dq, bq, "import", bq))
        else:
            # Both non-zero but differ → adjust the DB net toward broker.
            out.append(ReconcileAction(asset, dq, bq, "adjustment", bq - dq))
    return out


class Reconciler:
    def __init__(
        self,
        *,
        venue: Venue,
        bus: EventBus,
        clock: Clock,
        db_net: Callable[[], dict[str, float]],
        owner_of: Callable[[str], str | None],
        engine_owner: str = "belief",
    ) -> None:
        self._venue = venue
        self._bus = bus
        self._clock = clock
        self._db_net = db_net
        self._owner_of = owner_of
        self._engine_owner = engine_owner

    async def run_once(self) -> list[ReconcileAction]:
        positions = await self._venue.positions()
        broker = {p.asset: p.quantity for p in positions}
        plan = reconcile_plan(
            broker, self._db_net(), self._owner_of, engine_owner=self._engine_owner
        )
        for a in plan:
            if a.action == "none":
                continue
            await self._bus.publish(
                new_event(
                    self._clock,
                    EventType.POSITION_RECONCILED,
                    source="execution.reconcile",
                    asset=a.asset,
                    payload={
                        "db_net": a.db_net,
                        "broker_qty": a.broker_qty,
                        "engine_owner": self._engine_owner,
                        "action": a.action,
                        "adjustment_qty": a.adjustment_qty,
                    },
                )
            )
            if a.action != "skip_other_engine":
                logger.info(
                    "reconcile %s: %s (db=%.4f broker=%.4f)",
                    a.asset,
                    a.action,
                    a.db_net,
                    a.broker_qty,
                )
        return plan
