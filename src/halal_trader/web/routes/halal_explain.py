"""GET /api/halal/explain/{trade_id}: why a trade was halal, with citations.

The API twin of ``halal-trader halal explain``.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


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
