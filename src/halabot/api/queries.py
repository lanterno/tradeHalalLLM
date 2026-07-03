"""Read queries + control writes for the API (REARCHITECTURE L9).

Pure async functions over the shared engine — unit-tested directly against the
test DB, independent of FastAPI. Everything is read-only EXCEPT ``set_halt``,
which toggles the operator kill-switch (``hb_control``)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.belief.store import PgBeliefStore
from halabot.platform.db import control, conviction_score, event_log, outcome
from halabot.platform.events import EventType


def _belief_dict(b: Any) -> dict[str, Any]:
    return {
        "asset": b.asset,
        "version": b.version,
        "regime": str(b.regime),
        "regime_confidence": round(b.regime_confidence, 4),
        "direction": str(b.direction),
        "conviction": round(b.conviction, 4),
        "conviction_raw": round(b.conviction_raw, 4),
        "thesis": b.thesis,
        "invalidation": b.levels.invalidation,
        "stop": b.levels.stop,
        "support": b.levels.support,
        "resistance": b.levels.resistance,
        "horizon": str(b.horizon),
        # Persisted all along; surfaced for the belief board (Task D) now
        # that Task B slice 1 populates it.
        "catalysts_pending": [
            {
                "kind": c.kind,
                "scheduled_for": c.scheduled_for.isoformat(),
                "expected_impact": round(c.expected_impact, 3),
                "detail": c.detail,
            }
            for c in b.catalysts_pending
        ],
        "halal": (b.halal.status if b.halal else None),
        "n_evidence": len(b.evidence),
        "top_evidence": [
            {"source": e.source, "direction": round(e.direction, 3), "weight": round(e.weight, 3),
             "detail": e.detail}
            for e in sorted(b.evidence, key=lambda e: -abs(e.direction * e.weight))[:5]
        ],
        "last_updated": b.last_updated.isoformat() if b.last_updated else None,
    }


async def list_beliefs(engine: AsyncEngine) -> list[dict[str, Any]]:
    """The belief board: every active belief, conviction-ranked."""
    beliefs = await PgBeliefStore(engine).all_active()
    return [_belief_dict(b) for b in sorted(beliefs, key=lambda b: -b.conviction)]


async def get_belief(engine: AsyncEngine, asset: str) -> dict[str, Any] | None:
    b = await PgBeliefStore(engine).get(asset)
    return _belief_dict(b) if b is not None else None


def _event_dict(row: Any) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "type": row.type,
        "asset": row.asset,
        "ts": row.ts.isoformat(),
        "source": row.source,
        "payload": row.payload,
        "causation_id": str(row.causation_id) if row.causation_id else None,
        "correlation_id": str(row.correlation_id) if row.correlation_id else None,
    }


async def decision_chain(engine: AsyncEngine, correlation_id: UUID) -> list[dict[str, Any]]:
    """Replay one decision's causal chain (news → belief → conviction → policy →
    order), every event sharing the correlation_id, in time order (INV-5)."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            sa.select(event_log)
            .where(event_log.c.correlation_id == correlation_id)
            .order_by(event_log.c.ts)
        )
        return [_event_dict(r) for r in rows]


async def recent_decisions(engine: AsyncEngine, *, limit: int = 50) -> list[dict[str, Any]]:
    """Recent policy proposals (the decision-stream feed). Each carries the
    correlation_id you can expand via ``decision_chain``."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            sa.select(event_log)
            .where(event_log.c.type == str(EventType.POLICY_TRADE_PROPOSED))
            .order_by(event_log.c.ts.desc())
            .limit(limit)
        )
        return [_event_dict(r) for r in rows]


async def latest_risk(engine: AsyncEngine) -> dict[str, Any] | None:
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                sa.select(event_log)
                .where(event_log.c.type == str(EventType.RISK_STATE))
                .order_by(event_log.c.ts.desc())
                .limit(1)
            )
        ).first()
        return _event_dict(row) if row is not None else None


async def conviction_history(
    engine: AsyncEngine, asset: str, *, limit: int = 100
) -> list[dict[str, Any]]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            sa.select(conviction_score)
            .where(conviction_score.c.asset == asset)
            .order_by(conviction_score.c.ts.desc())
            .limit(limit)
        )
        return [
            {"ts": r.ts.isoformat(), "raw": r.raw_score, "calibrated": r.calibrated,
             "belief_version": r.belief_version}
            for r in rows
        ]


async def system_health(engine: AsyncEngine) -> dict[str, Any]:
    async with engine.connect() as conn:
        n_events = (await conn.execute(sa.select(sa.func.count()).select_from(event_log))).scalar()
        n_outcomes = (await conn.execute(sa.select(sa.func.count()).select_from(outcome))).scalar()
        last_ts = (await conn.execute(sa.select(sa.func.max(event_log.c.ts)))).scalar()
    # CURRENT active beliefs (latest version per asset) — not the all-time count
    # of every asset that ever had a target change (which only grows).
    active_beliefs = len(await PgBeliefStore(engine).all_active())
    halt = await get_halt(engine)
    return {
        "events": int(n_events or 0),
        "active_beliefs": active_beliefs,
        "outcomes": int(n_outcomes or 0),
        "last_event_ts": last_ts.isoformat() if last_ts else None,
        "halted": halt["halted"],
    }


# ── control / kill-switch ──
async def get_halt(engine: AsyncEngine) -> dict[str, Any]:
    async with engine.connect() as conn:
        row = (await conn.execute(sa.select(control).where(control.c.id == 1))).first()
    if row is None:
        return {"halted": False, "reason": None, "updated_at": None}
    return {
        "halted": bool(row.halted),
        "reason": row.reason,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


async def set_halt(engine: AsyncEngine, *, halted: bool, reason: str | None) -> dict[str, Any]:
    now = datetime.now(UTC)
    async with engine.begin() as conn:
        existing = (await conn.execute(sa.select(control.c.id).where(control.c.id == 1))).first()
        if existing is None:
            await conn.execute(
                sa.insert(control).values(id=1, halted=halted, reason=reason, updated_at=now)
            )
        else:
            await conn.execute(
                sa.update(control)
                .where(control.c.id == 1)
                .values(halted=halted, reason=reason, updated_at=now)
            )
    return {"halted": halted, "reason": reason, "updated_at": now.isoformat()}
