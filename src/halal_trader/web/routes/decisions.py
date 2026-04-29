"""GET /api/decisions, /api/adjustments."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/decisions")
    async def api_decisions(
        limit: int = 50, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        decisions = await ctx.repo.get_recent_decisions(limit=limit)
        return JSONResponse(serialize(decisions))

    @app.get("/api/adjustments")
    async def api_adjustments(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        adjustments = await ctx.repo.get_recent_adjustments()
        return JSONResponse(serialize(adjustments))
