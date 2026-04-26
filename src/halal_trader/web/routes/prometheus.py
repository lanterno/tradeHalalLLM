"""GET /metrics — Prometheus exposition endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from halal_trader.web.prometheus import collect_default_snapshots, render_metrics


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        snapshots = collect_default_snapshots(app_state)
        body = render_metrics(snapshots)
        # Prometheus expects text/plain with a specific version directive in
        # production; it tolerates the default content-type fine for scraping.
        return PlainTextResponse(content=body, media_type="text/plain; version=0.0.4")
