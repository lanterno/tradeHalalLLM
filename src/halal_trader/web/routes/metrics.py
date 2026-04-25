"""GET /api/metrics/cycles, /api/metrics/llm."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from halal_trader.config import get_settings


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    del app_state  # metrics read directly from log files

    @app.get("/api/metrics/cycles")
    async def api_metrics_cycles(window: int = 3600) -> JSONResponse:
        from halal_trader.web.metrics import cycle_metrics

        settings = get_settings()
        log_path = settings.log.dir / "halal_trader.log"
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
    async def api_metrics_llm(window: int = 86400) -> JSONResponse:
        from halal_trader.web.metrics import llm_metrics

        settings = get_settings()
        log_path = settings.log.dir / "halal_trader.log"
        m = llm_metrics(log_path, window_seconds=window)
        return JSONResponse(
            {
                "window_seconds": m.window_seconds,
                "calls": m.calls,
                "total_tokens": m.total_tokens,
                "p50_ms": m.p50_ms,
                "p95_ms": m.p95_ms,
                "by_provider": m.by_provider,
            }
        )
