"""The event model — the system's spine.

Everything that enters or moves through the engine is an immutable
:class:`Event` appended to a durable log (INV-5: replayability). Events are
classified into two **durability tiers** (Appendix E): CONTROL events
(exits, halts, heartbeat) dispatch best-effort even if the durable append
fails, so risk-reducing actions never block on the DB; DURABLE events
(observations, belief versions, outcomes) require the append.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from halabot.platform.clock import Clock

SCHEMA_VERSION = 1


class EventType(StrEnum):
    # ── perception ──
    OBSERVATION_PRICE = "observation.price"
    OBSERVATION_BAR = "observation.bar"
    OBSERVATION_NEWS = "observation.news"
    OBSERVATION_MACRO = "observation.macro"
    OBSERVATION_ONCHAIN = "observation.onchain"
    OBSERVATION_SENTIMENT = "observation.sentiment"
    # ── belief ──
    BELIEF_UPDATED = "belief.updated"
    BELIEF_THESIS_REFRESHED = "belief.thesis_refreshed"
    BELIEF_INVALIDATED = "belief.invalidated"
    # ── conviction / policy ──
    CONVICTION_SCORED = "conviction.scored"
    POLICY_TARGET_CHANGED = "policy.target_changed"
    POLICY_TRADE_PROPOSED = "policy.trade_proposed"
    # ── execution ──
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    ORDER_REJECTED = "order.rejected"
    POSITION_RECONCILED = "position.reconciled"
    # ── risk / ops ──
    RISK_STATE = "risk.state"
    RISK_HALT = "risk.halt"
    COMPLIANCE_VERDICT = "compliance.verdict"
    SYSTEM_HEARTBEAT = "system.heartbeat"


# Two-tier durability (Appendix E, fix R DB-down deadlock). CONTROL events
# carry risk-reducing intent (exits, halts) or the time-decay tick; they must
# dispatch even when the durable append fails, so the monitor can keep closing
# risk during a DB outage. Everything else is DURABLE: the append must succeed
# (it is the training corpus + replay source) before dispatch.
CONTROL_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.ORDER_SUBMITTED,
        EventType.ORDER_FILLED,
        EventType.ORDER_REJECTED,
        EventType.BELIEF_INVALIDATED,
        EventType.RISK_HALT,
        EventType.SYSTEM_HEARTBEAT,
    }
)


@dataclass(frozen=True, slots=True)
class Event:
    """An immutable fact on the bus / in the log.

    ``ts`` is *event time* (when the thing happened / was observed), stamped
    from the injected clock — distinct from ``ingested_at`` which the log
    records on append. ``causation_id`` is the event that directly caused this
    one; ``correlation_id`` groups a whole causal chain (e.g. news → belief
    update → trade) so the dashboard can reconstruct "why" (INV-5).
    """

    id: UUID
    type: EventType
    ts: datetime
    source: str
    asset: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    causation_id: UUID | None = None
    correlation_id: UUID | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def is_control(self) -> bool:
        """True for best-effort-dispatch events (exits/halts/heartbeat)."""
        return self.type in CONTROL_EVENT_TYPES


def new_event(
    clock: Clock,
    type: EventType,
    *,
    source: str,
    asset: str | None = None,
    payload: dict[str, Any] | None = None,
    causation: Event | None = None,
) -> Event:
    """Construct an :class:`Event`, stamping ``id`` and ``ts`` from the clock.

    When ``causation`` is supplied, the new event inherits its
    ``correlation_id`` (continuing the causal chain) and records it as
    ``causation_id``; otherwise it starts a fresh chain with a new
    ``correlation_id``.
    """
    correlation_id = causation.correlation_id if causation is not None else None
    return Event(
        id=uuid4(),
        type=type,
        ts=clock.now(),
        source=source,
        asset=asset,
        payload=payload or {},
        causation_id=causation.id if causation is not None else None,
        correlation_id=correlation_id or uuid4(),
        schema_version=SCHEMA_VERSION,
    )
