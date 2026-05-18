"""System endpoints: /api/health, /api/system/{status,halt,reconcile,backups}."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import Body, Depends, FastAPI, Header
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/health")
    async def api_health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "running",
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "0.3.0",
            }
        )

    @app.get("/api/system/status")
    async def api_system_status(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        started = ctx.runtime.started_at
        uptime = (datetime.now(UTC) - started).total_seconds() if started else None

        ws_health: dict[str, Any] = {}
        ws_mgr = ctx.runtime.ws_manager
        if ws_mgr and hasattr(ws_mgr, "health_status"):
            ws_health = ws_mgr.health_status()

        return JSONResponse(
            {
                "bot_running": ctx.runtime.bot_running,
                "last_cycle": ctx.runtime.last_cycle,
                "cycle_interval_seconds": ctx.settings.crypto.trading_interval_seconds,
                "ws_health": ws_health,
                "uptime_seconds": uptime,
            }
        )

    @app.get("/api/system/halt")
    async def api_get_halt(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        from halal_trader.core.halt import get_status

        s = await get_status(ctx.engine)
        return JSONResponse(
            {
                "enabled": s.enabled,
                "reason": s.reason,
                "set_by": s.set_by,
                "set_at": s.set_at.isoformat() if s.set_at else None,
            }
        )

    @app.post("/api/system/halt")
    async def api_set_halt(
        body: dict[str, Any] | None = Body(default=None),
        x_halt_confirm: str = Header(default=""),
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        if x_halt_confirm.lower() != "yes":
            return JSONResponse(
                {"error": "X-Halt-Confirm: yes header required"},
                status_code=400,
            )
        from halal_trader.core.halt import set_halt

        reason = (body or {}).get("reason") or "dashboard"
        s = await set_halt(ctx.engine, reason=reason, set_by="dashboard")
        return JSONResponse(
            {
                "enabled": s.enabled,
                "reason": s.reason,
                "set_by": s.set_by,
                "set_at": s.set_at.isoformat() if s.set_at else None,
            }
        )

    @app.delete("/api/system/halt")
    async def api_clear_halt(
        x_halt_confirm: str = Header(default=""),
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        if x_halt_confirm.lower() != "yes":
            return JSONResponse(
                {"error": "X-Halt-Confirm: yes header required"},
                status_code=400,
            )
        from halal_trader.core.halt import clear_halt

        s = await clear_halt(ctx.engine)
        return JSONResponse(
            {
                "enabled": s.enabled,
                "reason": s.reason,
                "set_by": s.set_by,
                "set_at": s.set_at.isoformat() if s.set_at else None,
            }
        )

    @app.get("/api/system/reconcile/recent")
    async def api_reconcile_recent(
        limit: int = 25, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        from halal_trader.core.reconcile import get_recent_logs

        rows = await get_recent_logs(ctx.engine, limit=max(1, min(limit, 200)))
        return JSONResponse(rows)

    @app.get("/api/system/backups")
    async def api_backups() -> JSONResponse:
        # Postgres baseline — backups happen via pg_dump or managed-DB
        # snapshot tooling, not via this endpoint.
        return JSONResponse([])
