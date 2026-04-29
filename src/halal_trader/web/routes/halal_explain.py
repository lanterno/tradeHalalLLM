"""Wave L — /api/halal/explain/{asset_class}/{trade_id}."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.halal.audit import export_receipt
from halal_trader.halal.explainer import explain_screening
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/halal/explain/{asset_class}/{trade_id}")
    async def explain(
        asset_class: str,
        trade_id: int,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        if asset_class not in ("stock", "crypto"):
            raise HTTPException(400, "asset_class must be 'stock' or 'crypto'")
        receipt = await export_receipt(ctx.engine, trade_id=trade_id, asset_class=asset_class)
        if receipt is None:
            raise HTTPException(404, f"no {asset_class} trade with id={trade_id}")
        explanation = explain_screening(receipt.payload)
        return JSONResponse(
            {
                "decision": explanation.decision,
                "markdown": explanation.body_md,
                "sources": explanation.sources,
            }
        )
