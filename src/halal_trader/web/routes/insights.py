"""Insights routes — read-only views over the new analytics modules.

Each endpoint computes its result fresh from recent closed trades plus
in-process state (drift monitor, shadow ledger, regime memory) the
cycle has been populating. None of these endpoints write — that's the
cycle's job.

Exposed routes:

* ``GET /api/insights/regret``       — hindsight regret summary
* ``GET /api/insights/thesis``       — thesis attribution table
* ``GET /api/insights/drift``        — concept-drift state
* ``GET /api/insights/stress``       — last stress harness verdicts (in-memory)
* ``GET /api/insights/shadow``       — shadow ledger + alert level
* ``GET /api/insights/calibration``  — current calibration curve

The cycle stashes any live state under ``app_state["insights"]`` (a
dict). This lets the routes avoid hard imports of cycle-side modules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    @app.get("/api/insights/regret")
    async def api_regret(limit: int = 200) -> JSONResponse:
        from halal_trader.cli.insights import (
            _load_closed_crypto_trades,
            _trades_to_closed_views,
        )
        from halal_trader.core.regret import aggregate_regret, hindsight_regret

        try:
            trades = await _load_closed_crypto_trades(limit)
            views = _trades_to_closed_views(trades)
            records = [hindsight_regret(v) for v in views]
            summary = aggregate_regret(records)
            return JSONResponse(
                {
                    "n": summary.n,
                    "mean_regret": summary.mean_regret,
                    "median_regret": summary.median_regret,
                    "pct_high_regret": summary.pct_high_regret,
                    "missed_edge_count": summary.missed_edge_count,
                    "tail_loss_count": summary.tail_loss_count,
                    "by_symbol": summary.by_symbol,
                }
            )
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/insights/thesis")
    async def api_thesis(limit: int = 200) -> JSONResponse:
        from halal_trader.cli.insights import (
            _load_closed_crypto_trades,
            _trades_to_tagged,
        )
        from halal_trader.core.thesis import (
            attribute_pnl_by_thesis,
            deprecated_thesis_kill_list,
        )

        try:
            trades = await _load_closed_crypto_trades(limit)
            views = _trades_to_tagged(trades)
            rows = attribute_pnl_by_thesis(views)
            return JSONResponse(
                {
                    "rows": [
                        {
                            "tag": r.tag,
                            "n_trades": r.n_trades,
                            "wins": r.wins,
                            "losses": r.losses,
                            "win_rate": r.win_rate,
                            "avg_pnl_pct": r.avg_pnl_pct,
                        }
                        for r in rows.values()
                    ],
                    "kill_candidates": deprecated_thesis_kill_list(rows),
                }
            )
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/insights/drift")
    async def api_drift() -> JSONResponse:
        insights = app_state.get("insights") or {}
        mon = insights.get("drift_monitor")
        if mon is None:
            return JSONResponse({"available": False})
        return JSONResponse(
            {
                "available": True,
                "state": mon.state,
                "n": mon.n,
                "drift_count": mon.drift_count,
                "last_drift_at": mon.last_drift_at,
            }
        )

    @app.get("/api/insights/stress")
    async def api_stress() -> JSONResponse:
        insights = app_state.get("insights") or {}
        verdicts = insights.get("stress_verdicts")
        if not verdicts:
            return JSONResponse({"available": False})
        return JSONResponse(
            {
                "available": True,
                "ts": insights.get("stress_ts"),
                "verdicts": [
                    {
                        "scenario_name": v.scenario_name,
                        "severity": v.severity,
                        "passed": v.passed,
                        "buys": v.buys,
                        "sells": v.sells,
                        "holds": v.holds,
                        "notes": v.notes,
                    }
                    for v in verdicts
                ],
            }
        )

    @app.get("/api/insights/shadow")
    async def api_shadow() -> JSONResponse:
        insights = app_state.get("insights") or {}
        ledger = insights.get("shadow_ledger")
        if ledger is None or ledger.size == 0:
            return JSONResponse({"available": False})
        from halal_trader.core.shadow import (
            divergence_metrics,
            shadow_alert_state,
        )

        metrics = divergence_metrics(ledger.entries)
        level = shadow_alert_state(metrics)
        return JSONResponse(
            {
                "available": True,
                "n": ledger.size,
                "level": level,
                "metrics": (
                    {
                        "n": metrics.n,
                        "mean_diff_pct": metrics.mean_diff_pct,
                        "last_diff_pct": metrics.last_diff_pct,
                        "max_drawdown_diff": metrics.max_drawdown_diff,
                        "paired_t_score": metrics.paired_t_score,
                        "direction": metrics.direction,
                    }
                    if metrics is not None
                    else None
                ),
                "ts": datetime.now(UTC).isoformat(),
            }
        )

    @app.get("/api/insights/regime")
    async def api_regime() -> JSONResponse:
        insights = app_state.get("insights") or {}
        mem = insights.get("regime_memory")
        if mem is None:
            return JSONResponse({"available": False})
        size = await mem.size()
        if size == 0:
            return JSONResponse({"available": False})
        recent = await mem.recent(limit=10)
        return JSONResponse(
            {
                "available": True,
                "size": size,
                "recent": [
                    {
                        "date": s.date,
                        "outcome_pnl_pct": s.outcome_pnl_pct,
                        "outcome_win_rate": s.outcome_win_rate,
                        "outcome_n_trades": s.outcome_n_trades,
                        "note": s.note,
                    }
                    for s in recent
                ],
            }
        )

    @app.get("/api/insights/basis")
    async def api_basis() -> JSONResponse:
        insights = app_state.get("insights") or {}
        tracker = insights.get("basis_tracker")
        if tracker is None or not tracker.history_by_pair:
            return JSONResponse({"available": False})
        out = {}
        for pair, hist in tracker.history_by_pair.items():
            if not hist:
                continue
            out[pair] = {
                "n": len(hist),
                "last_basis_bps": hist[-1] if hist else 0.0,
                "mean_basis_bps": (sum(hist) / len(hist)) if hist else 0.0,
            }
        return JSONResponse({"available": True, "pairs": out})

    @app.get("/api/insights/treasury")
    async def api_treasury() -> JSONResponse:
        # Pull current account from app_state if the cycle has cached it,
        # otherwise emit an "unavailable" response.
        try:
            from halal_trader.core.treasury import (
                TreasuryPolicy,
                estimate_annual_yield_usd,
                plan_idle_cash,
            )
        except Exception:  # noqa: BLE001
            return JSONResponse({"available": False})
        cached = app_state.get("account_snapshot")
        if not cached:
            return JSONResponse({"available": False})
        policy = TreasuryPolicy()
        plan = plan_idle_cash(
            cash_balance=float(cached.get("cash", 0)),
            positions_value=float(cached.get("positions_value", 0)),
            current_treasury_value=float(cached.get("treasury_value", 0)),
            policy=policy,
        )
        return JSONResponse(
            {
                "available": True,
                "action": plan.action,
                "amount_usd": plan.amount_usd,
                "instrument": plan.instrument,
                "reason": plan.reason,
                "estimated_yield_usd_year": estimate_annual_yield_usd(
                    cached.get("treasury_value", 0)
                ),
            }
        )

    @app.get("/api/insights/purification")
    async def api_purification() -> JSONResponse:
        from halal_trader.halal.round_trip_purification import (
            RoundTripLedger,
            outstanding_round_trip_due,
        )

        engine = app_state.get("engine")
        if engine is None:
            return JSONResponse({"available": False})
        ledger = RoundTripLedger(engine=engine)
        if await ledger.count() == 0:
            return JSONResponse({"available": False})
        return JSONResponse({"available": True, **(await outstanding_round_trip_due(ledger))})

    @app.get("/api/insights/replay")
    async def api_replay(limit: int = 50) -> JSONResponse:
        from halal_trader.core.replay import ReplayStore

        engine = app_state.get("engine")
        if engine is None:
            return JSONResponse({"available": False})
        store = ReplayStore(engine=engine)
        cycle_ids = await store.list_cycle_ids(limit=limit)
        return JSONResponse(
            {
                "available": True,
                "n": len(cycle_ids),
                "cycle_ids": cycle_ids,
            }
        )

    @app.get("/api/insights/exceptions")
    async def api_exceptions(status: str = "pending") -> JSONResponse:
        from halal_trader.halal.exception_queue import ExceptionQueue

        engine = app_state.get("engine")
        if engine is None:
            return JSONResponse({"available": False})
        if status not in ("pending", "approved", "rejected", "deferred", "all"):
            return JSONResponse({"error": f"unknown status {status!r}"}, status_code=400)
        q = ExceptionQueue(engine=engine)
        rows = await q.all() if status == "all" else await q.by_status(status)  # type: ignore[arg-type]
        return JSONResponse(
            {
                "available": True,
                "n": len(rows),
                "entries": [
                    {
                        "entry_id": e.entry_id,
                        "instrument": e.instrument,
                        "kind": e.kind,
                        "reasoning": e.reasoning,
                        "status": e.status,
                        "created_at": e.created_at,
                        "decided_at": e.decided_at,
                        "decided_by": e.decided_by,
                        "operator_note": e.operator_note,
                    }
                    for e in rows
                ],
            }
        )

    @app.post("/api/insights/exceptions/{entry_id}/decide")
    async def api_exceptions_decide(
        entry_id: str,
        status: str,
        decided_by: str = "",
        note: str = "",
    ) -> JSONResponse:
        from halal_trader.halal.exception_queue import ExceptionQueue

        engine = app_state.get("engine")
        if engine is None:
            return JSONResponse({"error": "no engine"}, status_code=503)
        q = ExceptionQueue(engine=engine)
        try:
            ok = await q.decide(
                entry_id,
                status=status,
                decided_by=decided_by,
                operator_note=note,  # type: ignore[arg-type]
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if not ok:
            return JSONResponse({"error": "entry not found"}, status_code=404)
        return JSONResponse({"ok": True, "entry_id": entry_id, "status": status})

    @app.get("/api/insights/velocity")
    async def api_velocity() -> JSONResponse:
        insights = app_state.get("insights") or {}
        velocity = insights.get("velocity") or {}
        if not velocity:
            return JSONResponse({"available": False})
        return JSONResponse(
            {
                "available": True,
                "results": [
                    {
                        "symbol": r.symbol,
                        "n_recent": r.n_recent,
                        "n_older": r.n_older,
                        "n_total": r.n_total,
                        "velocity": r.velocity,
                        "novelty": r.novelty,
                        "label": r.label,
                    }
                    for r in velocity.values()
                ],
            }
        )

    @app.get("/api/insights/whale")
    async def api_whale() -> JSONResponse:
        insights = app_state.get("insights") or {}
        flows = insights.get("whale_flows") or {}
        if not flows:
            return JSONResponse({"available": False})
        return JSONResponse(
            {
                "available": True,
                "flows": [
                    {
                        "symbol": sig.symbol,
                        "inflow_to_exchange_usd": sig.inflow_to_exchange_usd,
                        "outflow_from_exchange_usd": sig.outflow_from_exchange_usd,
                        "inflow_pressure": sig.inflow_pressure,
                        "n_transfers": sig.n_transfers,
                        "label": sig.label,
                    }
                    for sig in flows.values()
                ],
            }
        )

    @app.get("/api/insights/rag")
    async def api_rag(query: str = "", k: int = 5) -> JSONResponse:
        from halal_trader.core.llm.rag_db import DBRationaleStore

        engine = app_state.get("engine")
        if engine is None:
            return JSONResponse({"available": False})
        store = DBRationaleStore(engine=engine)
        size = await store.size()
        if size == 0:
            return JSONResponse({"available": False})
        if not query:
            return JSONResponse({"available": True, "size": size, "hits": []})
        hits = await store.query(query, k=k, min_similarity=0.0)
        return JSONResponse(
            {
                "available": True,
                "size": size,
                "hits": [
                    {
                        "trade_id": r.trade_id,
                        "symbol": r.symbol,
                        "text": r.text[:200],
                        "outcome_pnl_pct": r.outcome_pnl_pct,
                        "outcome_win": r.outcome_win,
                        "similarity": sim,
                    }
                    for r, sim in hits
                ],
                "aggregate": await store.aggregate(hits),
            }
        )

    @app.get("/api/insights/calibration")
    async def api_calibration() -> JSONResponse:
        insights = app_state.get("insights") or {}
        curve = insights.get("calibration_curve")
        if curve is None:
            return JSONResponse({"available": False})
        return JSONResponse(
            {
                "available": True,
                "method": curve.method,
                "n_samples": curve.n_samples,
                "anchors": curve.anchors,
            }
        )
