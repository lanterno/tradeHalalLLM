"""GET /api/operations — the Operations page (web/operations.py builds it)."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/operations")
    async def api_operations(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        from halal_trader.web.operations import build

        return JSONResponse(
            await build(ctx.engine, ctx.settings, web_started=ctx.runtime.started_at)
        )
