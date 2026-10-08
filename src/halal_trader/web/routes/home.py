"""GET /api/home — the dashboard's home page: money, the core portfolio, the
market and what the automation does next (assembled in portfolio/home.py)."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/home")
    async def api_home(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        from halal_trader.portfolio.home import build

        return JSONResponse(await build(ctx.engine, ctx.settings))
