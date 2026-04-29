"""GET /api/activity — recent web mutation audit feed."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/activity")
    async def api_activity(
        limit: int = 50, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        rows = await ctx.repo.get_recent_web_actions(limit=limit)
        return JSONResponse(serialize(rows))
