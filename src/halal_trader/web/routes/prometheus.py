"""GET /metrics — Prometheus exposition endpoint."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import PlainTextResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx
from halal_trader.web.prometheus import collect_default_snapshots, render_metrics


def register(app: FastAPI) -> None:
    @app.get("/metrics")
    async def metrics(ctx: DashboardContext = Depends(get_ctx)) -> PlainTextResponse:
        snapshots = collect_default_snapshots(ctx.runtime)
        body = render_metrics(snapshots)
        return PlainTextResponse(content=body, media_type="text/plain; version=0.0.4")
