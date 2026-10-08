"""GET /api/core — the Core page (portfolio/core_view.py builds it)."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/core")
    async def api_core(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        from halal_trader.portfolio.core_view import build

        return JSONResponse(await build(ctx.engine, ctx.settings))
