"""GET /api/metrics/cycles, /api/metrics/llm."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/metrics/cycles")
    async def api_metrics_cycles(
        window: int = 3600, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        from halal_trader.web.metrics import cycle_metrics

        log_path = ctx.settings.log.dir / "halal_trader.log"
        m = cycle_metrics(log_path, window_seconds=window)
        return JSONResponse(
            {
                "window_seconds": m.window_seconds,
                "count": m.count,
                "p50_ms": m.p50_ms,
                "p95_ms": m.p95_ms,
                "p99_ms": m.p99_ms,
                "failed": m.failed,
                "halted": m.halted,
            }
        )

    @app.get("/api/metrics/llm")
    async def api_metrics_llm(
        window: int = 86400, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        from halal_trader.web.metrics import llm_metrics

        log_path = ctx.settings.log.dir / "halal_trader.log"
        m = llm_metrics(log_path, window_seconds=window)
        return JSONResponse(
            {
                "window_seconds": m.window_seconds,
                "calls": m.calls,
                "total_tokens": m.total_tokens,
                "total_cost_usd": m.total_cost_usd,
                "p50_ms": m.p50_ms,
                "p95_ms": m.p95_ms,
                "by_provider": m.by_provider,
            }
        )

    @app.get("/api/metrics/rejections")
    async def api_metrics_rejections(
        window: int = 86400, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        """Guard/rejection reasons (cycle.no_action) so the operator can see
        why the bot didn't trade — concentration cap, cooldown, SL re-entry
        gate, halal screen, etc."""
        from halal_trader.web.metrics import recent_rejections

        log_path = ctx.settings.log.dir / "halal_trader.log"
        return JSONResponse(recent_rejections(log_path, window_seconds=window))
