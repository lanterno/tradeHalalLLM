"""Read queries + control writes for the API (REARCHITECTURE L9).

Pure async functions over the shared engine — unit-tested directly against the
test DB, independent of FastAPI. Everything is read-only EXCEPT ``set_halt``,
which toggles the operator kill-switch (``hb_control``)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.api import plain
from halabot.belief.serde import belief_from_row
from halabot.belief.store import PgBeliefStore
from halabot.platform.db import (
    control,
    conviction_score,
    event_log,
    open_position,
    outcome,
)
from halabot.platform.events import EventType


def _evidence_dict(e: Any) -> dict[str, Any]:
    return {
        "source": e.source,
        "label": plain.evidence_label(e.source),
        "direction": round(e.direction, 3),
        "weight": round(e.weight, 3),
        "detail": e.detail,
        "plain": plain.evidence_sentence(e.source, e.direction, e.detail),
    }


def _strongest(evidence: Sequence[Any], sign: int) -> str | None:
    """The plain sentence of the strongest item pointing ``sign`` (+1 for, -1 against)."""
    side = [e for e in evidence if e.direction * sign > 0]
    if not side:
        return None
    e = max(side, key=lambda e: abs(e.direction) * e.weight)
    return plain.evidence_sentence(e.source, e.direction, e.detail)


def _belief_dict(b: Any) -> dict[str, Any]:
    top = sorted(b.evidence, key=lambda e: -abs(e.direction * e.weight))[:5]
    top_dicts = [_evidence_dict(e) for e in top]
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
        "top_evidence": top_dicts,
        # The board's one-line case: the strongest item for and against, over
        # ALL the evidence (the top five can hold no counter at all).
        "main_reason": _strongest(b.evidence, +1),
        "counter_reason": _strongest(b.evidence, -1),
        "caution": plain.rsi_caution(top_dicts),
        "last_updated": b.last_updated.isoformat() if b.last_updated else None,
    }


# The latest version of each asset's belief, one index probe per asset down
# (asset, version desc). The store's DISTINCT ON reads every stored version:
# about a second on 900k rows, where this takes milliseconds.
_LATEST_BELIEFS = sa.text(
    "WITH RECURSIVE a AS (SELECT min(asset) AS asset FROM hb_belief_state "
    "UNION ALL SELECT (SELECT min(asset) FROM hb_belief_state b WHERE b.asset > a.asset) "
    "FROM a WHERE a.asset IS NOT NULL) "
    "SELECT s.* FROM a CROSS JOIN LATERAL (SELECT * FROM hb_belief_state b "
    "WHERE b.asset = a.asset ORDER BY b.version DESC LIMIT 1) s WHERE a.asset IS NOT NULL"
)


async def list_beliefs(engine: AsyncEngine) -> list[dict[str, Any]]:
    """The belief board: every active belief, conviction-ranked."""
    async with engine.connect() as conn:
        rows = (await conn.execute(_LATEST_BELIEFS)).all()
    beliefs = [belief_from_row(dict(r._mapping)) for r in rows]
    return [_belief_dict(b) for b in sorted(beliefs, key=lambda b: (-b.conviction, b.asset))]


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


async def recent_decisions(
    engine: AsyncEngine, *, limit: int = 50, asset: str | None = None
) -> list[dict[str, Any]]:
    """Recent policy proposals (the decision-stream feed), newest first, of one
    asset when ``asset`` is given. Each carries the correlation_id you can
    expand via ``decision_chain``."""
    stmt = sa.select(event_log).where(event_log.c.type == str(EventType.POLICY_TRADE_PROPOSED))
    if asset is not None:
        stmt = stmt.where(event_log.c.asset == asset)
    async with engine.connect() as conn:
        rows = await conn.execute(stmt.order_by(event_log.c.ts.desc()).limit(limit))
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
            {
                "ts": r.ts.isoformat(),
                "raw": r.raw_score,
                "calibrated": r.calibrated,
                "belief_version": r.belief_version,
            }
            for r in rows
        ]


# Above this many rows the planner's estimate stands in for count(*): an exact
# count of hb_event_log (2.5M rows) took most of a 2 s health call.
_EXACT_COUNT_BELOW = 100_000


async def _row_count(conn: Any, table: sa.Table) -> tuple[int, bool]:
    """(rows, estimated): ``pg_class.reltuples`` when the table is large and
    analysed, an exact count otherwise (-1 means never analysed)."""
    estimate = (
        await conn.execute(
            sa.text("SELECT reltuples::bigint FROM pg_class WHERE oid = to_regclass(:t)"),
            {"t": table.name},
        )
    ).scalar()
    if estimate is not None and estimate >= _EXACT_COUNT_BELOW:
        return int(estimate), True
    exact = (await conn.execute(sa.select(sa.func.count()).select_from(table))).scalar()
    return int(exact or 0), False


async def _last_event_ts(conn: Any) -> datetime | None:
    # max(ts) has no index of its own. Walk the (type, ts) index instead:
    # each distinct type, then its newest ts, one index probe apiece.
    value = (
        await conn.execute(
            sa.text(
                "WITH RECURSIVE t AS (SELECT min(type) AS type FROM hb_event_log "
                "UNION ALL SELECT (SELECT min(type) FROM hb_event_log e "
                "WHERE e.type > t.type) FROM t WHERE t.type IS NOT NULL) "
                "SELECT max((SELECT max(ts) FROM hb_event_log e WHERE e.type = t.type)) "
                "FROM t WHERE t.type IS NOT NULL"
            )
        )
    ).scalar()
    return value if isinstance(value, datetime) else None


async def last_event_at(engine: AsyncEngine) -> datetime | None:
    """When the engine last logged anything."""
    async with engine.connect() as conn:
        return await _last_event_ts(conn)


async def system_health(engine: AsyncEngine) -> dict[str, Any]:
    async with engine.connect() as conn:
        n_events, events_estimated = await _row_count(conn, event_log)
        n_outcomes, _ = await _row_count(conn, outcome)
        last_ts = await _last_event_ts(conn)
        # CURRENT active beliefs (latest version per asset): a count of the
        # distinct assets, walked down the (asset, version) index one asset at
        # a time rather than a DISTINCT ON over every stored version.
        active_beliefs = (
            await conn.execute(
                sa.text(
                    "WITH RECURSIVE a AS (SELECT min(asset) AS asset FROM hb_belief_state "
                    "UNION ALL SELECT (SELECT min(asset) FROM hb_belief_state b "
                    "WHERE b.asset > a.asset) FROM a WHERE a.asset IS NOT NULL) "
                    "SELECT count(asset) FROM a"
                )
            )
        ).scalar()
    halt = await get_halt(engine)
    return {
        "events": n_events,
        "events_estimated": events_estimated,
        "active_beliefs": int(active_beliefs or 0),
        "outcomes": n_outcomes,
        "last_event_ts": last_ts.isoformat() if last_ts else None,
        "halted": halt["halted"],
    }


# ── belief board: prices, the shadow book, the track record ──
_BAR = str(EventType.OBSERVATION_BAR)
# How far back a name's latest bar is looked for: a long weekend plus a holiday.
PRICE_LOOKBACK = timedelta(days=7)


async def latest_prices(
    engine: AsyncEngine, assets: Sequence[str], *, now: datetime
) -> dict[str, tuple[float, datetime]]:
    """Each asset's latest bar close and when the engine saw it.

    One index probe per asset walking ``observation.bar`` back from now, so
    the cost is the number of names, not the size of the event log.
    """
    if not assets:
        return {}
    async with engine.connect() as conn:
        rows = await conn.execute(
            sa.text(
                "SELECT a.asset, b.ts, b.c FROM unnest(CAST(:assets AS text[])) AS a(asset) "
                "CROSS JOIN LATERAL (SELECT e.ts, (e.payload->>'c')::float AS c "
                "FROM hb_event_log e WHERE e.type = :t AND e.asset = a.asset "
                "AND e.ts > :since AND jsonb_typeof(e.payload->'c') = 'number' "
                "ORDER BY e.ts DESC LIMIT 1) b"
            ),
            {"assets": list(assets), "t": _BAR, "since": now - PRICE_LOOKBACK},
        )
        return {r.asset: (float(r.c), r.ts) for r in rows}


async def last_bar_at(engine: AsyncEngine) -> datetime | None:
    """When the newest bar arrived (one probe of the (type, ts) index)."""
    async with engine.connect() as conn:
        value = (
            await conn.execute(
                sa.select(sa.func.max(event_log.c.ts)).where(event_log.c.type == _BAR)
            )
        ).scalar()
    return value if isinstance(value, datetime) else None


async def open_positions(engine: AsyncEngine, *, cohort: int) -> list[dict[str, Any]]:
    """The shadow book of ``cohort``: what it holds, marked at its last price."""
    p = open_position
    async with engine.connect() as conn:
        rows = await conn.execute(
            sa.select(p).where(p.c.cohort == cohort, p.c.weight > 0).order_by(p.c.weight.desc())
        )
        return [
            {
                "asset": r.asset,
                "weight": round(float(r.weight), 4),
                "entry_price": round(float(r.entry_vwap), 4),
                "last_price": round(float(r.last_price), 4),
                "return_pct": round(float(r.unrealized_return_pct), 5),
                "opened_at": r.entry_ts.isoformat(),
                "marked_at": r.updated_at.isoformat(),
            }
            for r in rows
        ]


def _stats(n: int, wins: int, mean: float | None) -> dict[str, Any]:
    return {
        "closed": n,
        "wins": wins,
        "win_rate": round(wins / n, 4) if n else None,
        "mean_return_pct": round(float(mean), 5) if n and mean is not None else None,
    }


async def outcome_stats(engine: AsyncEngine, *, cohort: int) -> dict[str, Any]:
    """Closed-trade record of ``cohort`` and of every earlier outcome.

    ``started`` is the cohort's first entry, open or closed; ``median_hold_s``
    the median holding time of its closed trades (what a random entry is held
    for when the baseline is measured).
    """
    o, p = outcome, open_position
    is_current = sa.func.coalesce(o.c.cohort == cohort, False).label("cur")
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                sa.select(
                    is_current,
                    sa.func.count().label("n"),
                    sa.func.coalesce(sa.func.sum(o.c.label), 0).label("wins"),
                    sa.func.avg(o.c.return_pct).label("mean"),
                    sa.func.min(o.c.entry_ts).label("started"),
                    sa.func.percentile_cont(0.5)
                    .within_group(o.c.hold_seconds)
                    .label("median_hold"),
                ).group_by(is_current)
            )
        ).all()
        open_started = (
            await conn.execute(sa.select(sa.func.min(p.c.entry_ts)).where(p.c.cohort == cohort))
        ).scalar()
    by = {bool(r.cur): r for r in rows}
    cur, old = by.get(True), by.get(False)
    starts = [t for t in (cur.started if cur else None, open_started) if t is not None]
    return {
        "cohort": cohort,
        "started": min(starts).isoformat() if starts else None,
        "current": _stats(cur.n, int(cur.wins), cur.mean) if cur else _stats(0, 0, None),
        "earlier": _stats(old.n, int(old.wins), old.mean) if old else _stats(0, 0, None),
        "median_hold_s": float(cur.median_hold) if cur and cur.median_hold else None,
    }


async def calibration_probe(
    engine: AsyncEngine, assets: Sequence[str], *, since: datetime
) -> tuple[int, int]:
    """(scorings since ``since``, how many the calibrator moved off the raw score).

    The fitted calibrator lives in the shadow's memory only, so the database
    says whether it is on by its output: identity returns the raw score
    clamped to [0, 1], anything else is a fitted model at work.
    """
    if not assets:
        return 0, 0
    c = conviction_score
    clamped = sa.func.least(1.0, sa.func.greatest(0.0, c.c.raw_score))
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                sa.select(
                    sa.func.count(),
                    sa.func.count().filter(sa.func.abs(c.c.calibrated - clamped) > 1e-6),
                ).where(c.c.asset.in_(list(assets)), c.c.ts >= since)
            )
        ).one()
    return int(row[0]), int(row[1])


async def bar_closes(
    engine: AsyncEngine, assets: Sequence[str], *, since: datetime
) -> dict[str, list[tuple[datetime, float]]]:
    """Each asset's bar closes since ``since``, one per bar (its last print),
    oldest first. The baseline's raw material; callers cache it."""
    if not assets:
        return {}
    async with engine.connect() as conn:
        rows = await conn.execute(
            sa.text(
                "SELECT DISTINCT ON (asset, payload->>'bar_ts') asset, "
                "(payload->>'bar_ts')::timestamptz AS bar_ts, (payload->>'c')::float AS c "
                "FROM hb_event_log WHERE type = :t AND ts >= :since "
                "AND asset = ANY(CAST(:assets AS text[])) AND payload ? 'bar_ts' "
                "AND jsonb_typeof(payload->'c') = 'number' "
                "ORDER BY asset, payload->>'bar_ts', ts DESC"
            ),
            {"t": _BAR, "since": since, "assets": list(assets)},
        )
        out: dict[str, list[tuple[datetime, float]]] = {}
        for r in rows:
            out.setdefault(r.asset, []).append((r.bar_ts, float(r.c)))
    for series in out.values():
        series.sort()
    return out


def random_entry_win_rate(
    closes: dict[str, list[tuple[datetime, float]]],
    *,
    hold: timedelta,
    threshold: float,
    in_session: Callable[[datetime], bool] | None = None,
) -> tuple[float | None, int]:
    """Win rate of entering at every bar close and holding for ``hold``.

    The honest bar for the shadow's record: the same names over the same
    period, with no judgement at all. A trade wins when the close of the first
    bar at least ``hold`` later is more than ``threshold`` above the entry
    (the outcome label's rule). ``in_session`` filters entry bars (a fill is
    only possible in the regular session). Returns (rate, trades).
    """
    wins = n = 0
    for series in closes.values():
        j = 0
        for i, (t, c) in enumerate(series):
            if c <= 0 or (in_session is not None and not in_session(t)):
                continue
            j = max(j, i + 1)
            while j < len(series) and series[j][0] - t < hold:
                j += 1
            if j >= len(series):
                break
            n += 1
            if series[j][1] / c - 1 > threshold:
                wins += 1
    return (wins / n if n else None), n


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
