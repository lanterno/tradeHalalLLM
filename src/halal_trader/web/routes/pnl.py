"""GET /api/pnl/daily."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/pnl/daily")
    async def api_daily_pnl(
        days: int = 30, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        pnl = await ctx.repo.get_crypto_pnl_history(limit=days)
        return JSONResponse(serialize(pnl))
