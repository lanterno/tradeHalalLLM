"""GET /metrics — Prometheus exposition endpoint.

Combines two metric sources:
  1. The hand-rolled gauge snapshots from ``web/prometheus.py``
     (bot_running, drawdown_pct, etc) — built from the runtime view.
  2. The native ``prometheus_client`` histograms from
     ``core/metrics.py`` (per-stage cycle latency, LLM call latency,
     broker call latency).

Both formats are append-friendly so we just concatenate.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import PlainTextResponse

from halal_trader.core.context import DashboardContext
from halal_trader.core.metrics import render_prometheus_text
from halal_trader.web.dependencies import get_ctx
from halal_trader.web.prometheus import collect_default_snapshots, render_metrics


def register(app: FastAPI) -> None:
    @app.get("/metrics")
    async def metrics(ctx: DashboardContext = Depends(get_ctx)) -> PlainTextResponse:
        snapshots = collect_default_snapshots(ctx.runtime)
        body = render_metrics(snapshots)
        # Append the prometheus_client histograms (Wave J).
        histograms = render_prometheus_text().decode("utf-8", errors="replace")
        if histograms:
            body = body.rstrip("\n") + "\n" + histograms
        return PlainTextResponse(content=body, media_type="text/plain; version=0.0.4")
