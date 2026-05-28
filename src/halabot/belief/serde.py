"""JSON (de)serialization for ``BeliefState`` and its nested parts.

Used by the Postgres store to round-trip beliefs through JSONB columns.
Datetimes are stored as ISO-8601 strings and UUIDs as strings (JSONB-native),
and parsed back on read.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from halabot.belief.schema import (
    BeliefState,
    Catalyst,
    ComplianceVerdict,
    Direction,
    EvidenceItem,
    Horizon,
    Levels,
    Regime,
)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _parse_dt(value: Any) -> datetime | None:
    return datetime.fromisoformat(value) if isinstance(value, str) else None


def evidence_to_json(e: EvidenceItem) -> dict[str, Any]:
    return {
        "source": e.source,
        "direction": e.direction,
        "weight": e.weight,
        "detail": e.detail,
        "ts": _iso(e.ts),
        "event_id": str(e.event_id) if e.event_id is not None else None,
        "directional": e.directional,
    }


def evidence_from_json(d: dict[str, Any]) -> EvidenceItem:
    eid = d.get("event_id")
    return EvidenceItem(
        source=d["source"],
        direction=d["direction"],
        weight=d["weight"],
        detail=d.get("detail", ""),
        ts=_parse_dt(d.get("ts")),
        event_id=UUID(eid) if eid else None,
        directional=d.get("directional", True),
    )


def catalyst_to_json(c: Catalyst) -> dict[str, Any]:
    return {
        "kind": c.kind,
        "scheduled_for": _iso(c.scheduled_for),
        "expected_impact": c.expected_impact,
        "detail": c.detail,
    }


def catalyst_from_json(d: dict[str, Any]) -> Catalyst:
    scheduled = _parse_dt(d.get("scheduled_for"))
    assert scheduled is not None  # catalysts always carry a time
    return Catalyst(
        kind=d["kind"],
        scheduled_for=scheduled,
        expected_impact=d["expected_impact"],
        detail=d.get("detail", ""),
    )


def levels_to_json(lv: Levels) -> dict[str, Any]:
    return {
        "support": lv.support,
        "resistance": lv.resistance,
        "stop": lv.stop,
        "invalidation": lv.invalidation,
    }


def levels_from_json(d: dict[str, Any]) -> Levels:
    return Levels(
        support=d.get("support"),
        resistance=d.get("resistance"),
        stop=d.get("stop"),
        invalidation=d.get("invalidation"),
    )


def verdict_to_json(v: ComplianceVerdict | None) -> dict[str, Any] | None:
    if v is None:
        return None
    return {
        "asset": v.asset,
        "status": v.status,
        "detail": v.detail,
        "screened_at": _iso(v.screened_at),
        "screening_id": v.screening_id,
        "transient_error": v.transient_error,
    }


def verdict_from_json(d: dict[str, Any] | None) -> ComplianceVerdict | None:
    if d is None:
        return None
    return ComplianceVerdict(
        asset=d["asset"],
        status=d["status"],
        detail=d.get("detail", ""),
        screened_at=_parse_dt(d.get("screened_at")),
        screening_id=d.get("screening_id"),
        transient_error=d.get("transient_error", False),
    )


def belief_to_values(b: BeliefState) -> dict[str, Any]:
    """Column values for an insert into ``hb_belief_state`` (excludes id/version,
    which the store assigns)."""
    return {
        "asset": b.asset,
        "regime": str(b.regime),
        "regime_confidence": b.regime_confidence,
        "direction": str(b.direction),
        "conviction": b.conviction,
        "conviction_raw": b.conviction_raw,
        "horizon": str(b.horizon),
        "thesis": b.thesis,
        "levels": levels_to_json(b.levels),
        "catalysts": [catalyst_to_json(c) for c in b.catalysts_pending],
        "evidence": [evidence_to_json(e) for e in b.evidence],
        "halal_verdict": verdict_to_json(b.halal),
        "opened_trade_ids": list(b.opened_trade_ids),
        "last_thesis_refresh": b.last_thesis_refresh,
        "last_updated": b.last_updated,
    }


def belief_from_row(m: dict[str, Any]) -> BeliefState:
    """Reconstruct a :class:`BeliefState` from a ``hb_belief_state`` row mapping."""
    return BeliefState(
        asset=m["asset"],
        regime=Regime(m["regime"]),
        regime_confidence=m["regime_confidence"],
        direction=Direction(m["direction"]),
        conviction=m["conviction"],
        conviction_raw=m["conviction_raw"],
        horizon=Horizon(m["horizon"]),
        thesis=m["thesis"] or "",
        levels=levels_from_json(m["levels"]),
        catalysts_pending=[catalyst_from_json(c) for c in (m["catalysts"] or [])],
        evidence=[evidence_from_json(e) for e in (m["evidence"] or [])],
        halal=verdict_from_json(m["halal_verdict"]),
        opened_trade_ids=list(m["opened_trade_ids"] or []),
        last_updated=m["last_updated"],
        last_thesis_refresh=m["last_thesis_refresh"],
        version=m["version"],
    )
