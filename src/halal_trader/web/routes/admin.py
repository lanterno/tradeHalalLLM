"""Operator lifecycle endpoints — halt, resume, cancel, force-close."""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from halal_trader.core.context import DashboardContext
from halal_trader.core.halt import clear_halt, get_status, set_halt
from halal_trader.web.dependencies import get_ctx
from halal_trader.web.middleware.confirm import require_confirmation

logger = logging.getLogger(__name__)


class HaltRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


class ForceCloseRequest(BaseModel):
    asset_class: str = Field(default="stock", pattern="^stock$")
    reason: str = Field(default="operator_intervention", min_length=1, max_length=100)


def register(app: FastAPI) -> None:
    @app.get("/api/admin/halt")
    async def admin_halt_status(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        status = await get_status(ctx.engine)
        return JSONResponse(
            {
                "enabled": status.enabled,
                "reason": status.reason,
                "set_by": status.set_by,
                "set_at": status.set_at.isoformat() if status.set_at else None,
            }
        )

    @app.post("/api/admin/halt", dependencies=[Depends(require_confirmation)])
    async def admin_halt(
        req: HaltRequest, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        status = await set_halt(ctx.engine, reason=req.reason, set_by="dashboard")
        return JSONResponse(
            {
                "enabled": status.enabled,
                "reason": status.reason,
                "set_by": status.set_by,
                "set_at": status.set_at.isoformat() if status.set_at else None,
            }
        )

    @app.post("/api/admin/resume", dependencies=[Depends(require_confirmation)])
    async def admin_resume(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        status = await clear_halt(ctx.engine)
        return JSONResponse(
            {
                "enabled": status.enabled,
                "reason": status.reason,
                "set_by": status.set_by,
                "set_at": status.set_at.isoformat() if status.set_at else None,
            }
        )

    @app.delete(
        "/api/admin/orders",
        dependencies=[Depends(require_confirmation)],
    )
    async def admin_cancel_all_orders(
        asset_class: str = "stock", ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        if asset_class != "stock":
            raise HTTPException(400, "asset_class must be 'stock'")
        broker = ctx.runtime.stock_broker
        if broker is None:
            raise HTTPException(503, f"{asset_class} broker not bound to web app")
        from halal_trader.core.shutdown import cancel_all_open_orders

        result = await cancel_all_open_orders(broker)
        return JSONResponse(
            {
                "cancelled": result.cancelled,
                "failed": [{"order_id": oid, "error": err} for oid, err in result.failed],
            }
        )

    @app.delete(
        "/api/admin/orders/{order_id}",
        dependencies=[Depends(require_confirmation)],
    )
    async def admin_cancel_one_order(
        order_id: str,
        symbol: str,
        asset_class: str = "stock",
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        if asset_class != "stock":
            raise HTTPException(400, "asset_class must be 'stock'")
        broker = ctx.runtime.stock_broker
        if broker is None:
            raise HTTPException(503, f"{asset_class} broker not bound to web app")
        try:
            await broker.cancel_order(symbol=symbol, order_id=order_id)
        except Exception as e:
            raise HTTPException(502, f"broker rejected cancel: {e}")
        return JSONResponse({"order_id": order_id, "cancelled": True})

    @app.post(
        "/api/admin/positions/{symbol}/close",
        dependencies=[Depends(require_confirmation)],
    )
    async def admin_force_close(
        symbol: str,
        req: ForceCloseRequest,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        broker = ctx.runtime.stock_broker
        if broker is None:
            raise HTTPException(503, f"{req.asset_class} broker not bound to web app")
        try:
            await broker.close_position(symbol)
        except Exception as e:
            raise HTTPException(502, f"broker rejected close: {e}")
        return JSONResponse({"symbol": symbol, "closed": True, "reason": req.reason})
