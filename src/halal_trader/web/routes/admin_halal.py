"""Halal & compliance admin endpoints.

Surfaces the purification ledger, on-demand cache refresh, and current
sector allocation so the operator can drive compliance workflows from
the dashboard:

* GET / POST / DELETE on the purification ledger.
"""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx
from halal_trader.web.middleware.confirm import require_confirmation

logger = logging.getLogger(__name__)


class RecordPurificationRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12)
    dividend_usd: float = Field(ge=0)
    haram_pct: float = Field(ge=0, le=1)
    notes: str | None = Field(default=None, max_length=500)


def register(app: FastAPI) -> None:
    @app.get("/api/halal/explain/{trade_id}")
    async def explain_trade(
        trade_id: int,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        """Operator-readable Sharia-compliance explanation.

        Pulls the trade + its ``halal_screenings`` receipt and renders
        the criteria blob as Markdown with citations to
        ``docs/halal_jurisprudence.md``. Returns 404 when the trade
        doesn't exist; renders an "unattested" body for legacy trades
        that pre-date the screening FK.
        """
        from halal_trader.halal.audit import export_receipt
        from halal_trader.halal.explainer import explain_screening

        receipt = await export_receipt(ctx.engine, trade_id=trade_id)
        if receipt is None:
            raise HTTPException(404, f"trade {trade_id} not found")
        explanation = explain_screening(receipt.payload)
        return JSONResponse(
            {
                "trade_id": trade_id,
                "decision": explanation.decision,
                "body_md": explanation.body_md,
                "sources": explanation.sources,
            }
        )

    @app.get("/api/admin/purification")
    async def list_purification(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        outstanding = await ctx.repo.get_outstanding_purification()
        totals = await ctx.repo.get_purification_totals()
        return JSONResponse(
            {
                "outstanding": serialize(outstanding),
                "totals": totals,
            }
        )

    @app.post(
        "/api/admin/purification",
        dependencies=[Depends(require_confirmation)],
    )
    async def record_purification(
        req: RecordPurificationRequest,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halal_trader.halal.purification import compute_purification

        entry = compute_purification(
            symbol=req.symbol,
            dividend_usd=req.dividend_usd,
            haram_revenue_pct=req.haram_pct,
            notes=req.notes or "",
        )
        eid = await ctx.repo.record_purification(
            symbol=entry.symbol,
            dividend_usd=float(entry.dividend_usd),
            haram_pct=float(entry.haram_pct),
            purification_usd=float(entry.purification_usd),
            notes=entry.notes,
        )
        return JSONResponse(
            {
                "id": eid,
                "symbol": entry.symbol,
                "purification_usd": float(entry.purification_usd),
            }
        )

    @app.post(
        "/api/admin/purification/{entry_id}/mark_paid",
        dependencies=[Depends(require_confirmation)],
    )
    async def mark_paid(entry_id: int, ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        ok = await ctx.repo.mark_purification_paid(entry_id)
        if not ok:
            raise HTTPException(404, f"purification entry {entry_id} not found")
        return JSONResponse({"id": entry_id, "paid": True})
