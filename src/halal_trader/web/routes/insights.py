"""Insights routes — read-only views over the new analytics modules."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/insights/drift")
    async def api_drift(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        mon = ctx.hub.drift
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

    @app.get("/api/insights/shadow")
    async def api_shadow(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        ledger = ctx.hub.shadow
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
    async def api_regime(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        mem = ctx.hub.regime
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

    @app.get("/api/insights/treasury")
    async def api_treasury(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        try:
            from halal_trader.core.treasury import (
                TreasuryPolicy,
                estimate_annual_yield_usd,
                plan_idle_cash,
            )
        except Exception:  # noqa: BLE001
            return JSONResponse({"available": False})
        cached = ctx.runtime.account_snapshot
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
    async def api_purification(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halal_trader.halal.round_trip_purification import (
            RoundTripLedger,
            outstanding_round_trip_due,
        )

        ledger = RoundTripLedger(engine=ctx.engine)
        if await ledger.count() == 0:
            return JSONResponse({"available": False})
        return JSONResponse({"available": True, **(await outstanding_round_trip_due(ledger))})

    @app.get("/api/insights/replay")
    async def api_replay(limit: int = 50, ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        from halal_trader.core.replay import ReplayStore

        store = ReplayStore(engine=ctx.engine)
        cycle_ids = await store.list_cycle_ids(limit=limit)
        return JSONResponse(
            {
                "available": True,
                "n": len(cycle_ids),
                "cycle_ids": cycle_ids,
            }
        )

    @app.get("/api/insights/exceptions")
    async def api_exceptions(
        status: str = "pending", ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        from halal_trader.halal.exception_queue import ExceptionQueue

        if status not in ("pending", "approved", "rejected", "deferred", "all"):
            return JSONResponse({"error": f"unknown status {status!r}"}, status_code=400)
        q = ExceptionQueue(engine=ctx.engine)
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
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halal_trader.halal.exception_queue import ExceptionQueue

        if status not in ("pending", "approved", "rejected", "deferred"):
            return JSONResponse(
                {"error": f"invalid status: {status}"},
                status_code=400,
            )

        q = ExceptionQueue(engine=ctx.engine)
        try:
            ok = await q.decide(
                entry_id,
                status=status,  # type: ignore[arg-type]
                decided_by=decided_by,
                operator_note=note,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if not ok:
            return JSONResponse({"error": "entry not found"}, status_code=404)
        return JSONResponse({"ok": True, "entry_id": entry_id, "status": status})

    @app.get("/api/insights/rag")
    async def api_rag(
        query: str = "",
        k: int = 5,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halal_trader.core.llm.rag_db import DBRationaleStore

        store = DBRationaleStore(engine=ctx.engine)
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
    async def api_calibration(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        curve = ctx.hub.calibration
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
