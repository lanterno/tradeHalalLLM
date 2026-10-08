"""Insights routes — read-only views over the new analytics modules."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.portfolio.core_account import BROKER_ACCOUNTS, DAY_TRADER
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/insights/purification")
    async def api_purification(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        """Purification still owed, by account and by symbol.

        Reads the dividend ledger the research run keeps for both broker
        accounts (``purification_accruals``) plus the older round-trip
        ledger. It used to read only the round-trip ledger, which nothing
        has written since the dividend ledger replaced it, and so said "no
        closed wins yet" while dividends were owed.
        """
        from sqlalchemy import text

        from halal_trader.halal.round_trip_purification import (
            RoundTripLedger,
            outstanding_round_trip_due,
        )

        async with ctx.engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT account, symbol, amount, paid_at IS NOT NULL AS paid "
                        "FROM purification_accruals WHERE account = ANY(:a)"
                    ),
                    {"a": list(BROKER_ACCOUNTS)},
                )
            ).all()
        ledger = RoundTripLedger(engine=ctx.engine)
        legacy = await outstanding_round_trip_due(ledger) if await ledger.count() else None
        if not rows and legacy is None:
            return JSONResponse({"available": False})

        by_account: dict[str, float] = {}
        by_symbol: dict[str, float] = {}
        disbursed = 0.0
        unpaid_entries = 0
        for r in rows:
            if r.paid:
                disbursed += float(r.amount)
                continue
            unpaid_entries += 1
            by_account[r.account] = by_account.get(r.account, 0.0) + float(r.amount)
            by_symbol[r.symbol] = by_symbol.get(r.symbol, 0.0) + float(r.amount)
        if legacy is not None:
            by_account[DAY_TRADER] = by_account.get(DAY_TRADER, 0.0) + legacy["total_usd"]
            for symbol, amount in legacy["by_symbol"].items():
                by_symbol[symbol] = by_symbol.get(symbol, 0.0) + amount
            disbursed += legacy["disbursed_total_usd"]
            unpaid_entries += legacy["n_entries"]
        return JSONResponse(
            {
                "available": True,
                "total_usd": round(sum(by_account.values()), 2),
                "by_account": {a: round(v, 2) for a, v in by_account.items()},
                "by_symbol": {s: round(v, 2) for s, v in by_symbol.items()},
                "disbursed_total_usd": round(disbursed, 2),
                "n_entries": unpaid_entries,
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
