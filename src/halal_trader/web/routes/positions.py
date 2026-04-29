"""GET /api/positions."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/positions")
    async def api_positions(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        open_trades = await ctx.repo.get_open_crypto_trades()
        positions = []
        for t in open_trades:
            d = t.model_dump()
            current = None
            ws_mgr = ctx.runtime.ws_manager
            if ws_mgr and hasattr(ws_mgr, "get_latest_price"):
                current = ws_mgr.get_latest_price(t.pair)

            entry = t.entry_price or t.price
            d["entry_price"] = entry
            if current and entry:
                d["current_price"] = current
                d["unrealized_pnl"] = (current - entry) * t.quantity
                d["unrealized_pnl_pct"] = (current - entry) / entry
            else:
                d["current_price"] = entry
                d["unrealized_pnl"] = 0.0
                d["unrealized_pnl_pct"] = 0.0

            positions.append(d)
        return JSONResponse(serialize(positions))
