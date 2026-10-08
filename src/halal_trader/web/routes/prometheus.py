"""GET /metrics — Prometheus exposition endpoint (web/prometheus.py), read
from the database at scrape time."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import PlainTextResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx
from halal_trader.web.prometheus import collect, render_metrics


def register(app: FastAPI) -> None:
    @app.get("/metrics")
    async def metrics(ctx: DashboardContext = Depends(get_ctx)) -> PlainTextResponse:
        return PlainTextResponse(
            content=render_metrics(await collect(ctx.engine)),
            media_type="text/plain; version=0.0.4",
        )
