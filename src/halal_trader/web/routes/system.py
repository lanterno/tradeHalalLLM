"""System endpoints: /api/health, /api/system/{status,halt,reconcile,backups}."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import Body, FastAPI, Header
from fastapi.responses import JSONResponse

from halal_trader.config import get_settings


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    @app.get("/api/health")
    async def api_health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "running",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "0.2.0",
            }
        )

    @app.get("/api/system/status")
    async def api_system_status() -> JSONResponse:
        started = app_state.get("started_at")
        uptime = None
        if started:
            uptime = (datetime.now(timezone.utc) - started).total_seconds()

        ws_health: dict[str, Any] = {}
        ws_mgr = app_state.get("ws_manager")
        if ws_mgr and hasattr(ws_mgr, "health_status"):
            ws_health = ws_mgr.health_status()

        return JSONResponse(
            {
                "bot_running": app_state.get("bot_running", False),
                "last_cycle": app_state.get("last_cycle"),
                "cycle_interval_seconds": get_settings().crypto.trading_interval_seconds,
                "ws_health": ws_health,
                "uptime_seconds": uptime,
            }
        )

    @app.get("/api/system/halt")
    async def api_get_halt() -> JSONResponse:
        from halal_trader.core.halt import get_status

        engine = app_state["engine"]
        s = await get_status(engine)
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
        body: dict | None = Body(default=None),
        x_halt_confirm: str = Header(default=""),
    ) -> JSONResponse:
        if x_halt_confirm.lower() != "yes":
            return JSONResponse(
                {"error": "X-Halt-Confirm: yes header required"},
                status_code=400,
            )
        from halal_trader.core.halt import set_halt

        reason = (body or {}).get("reason") or "dashboard"
        engine = app_state["engine"]
        s = await set_halt(engine, reason=reason, set_by="dashboard")
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
    ) -> JSONResponse:
        if x_halt_confirm.lower() != "yes":
            return JSONResponse(
                {"error": "X-Halt-Confirm: yes header required"},
                status_code=400,
            )
        from halal_trader.core.halt import clear_halt

        engine = app_state["engine"]
        s = await clear_halt(engine)
        return JSONResponse(
            {
                "enabled": s.enabled,
                "reason": s.reason,
                "set_by": s.set_by,
                "set_at": s.set_at.isoformat() if s.set_at else None,
            }
        )

    @app.get("/api/system/reconcile/recent")
    async def api_reconcile_recent(limit: int = 25) -> JSONResponse:
        from halal_trader.core.reconcile import get_recent_logs

        engine = app_state["engine"]
        rows = await get_recent_logs(engine, limit=max(1, min(limit, 200)))
        return JSONResponse(rows)

    @app.get("/api/system/backups")
    async def api_backups() -> JSONResponse:
        from halal_trader.db.backup import list_backups

        settings = get_settings()
        rows = list_backups(settings.backup.dir)
        return JSONResponse(
            [
                {
                    "path": str(r.path),
                    "size_bytes": r.size_bytes,
                    "backed_up_at": r.backed_up_at.isoformat(),
                }
                for r in rows
            ]
        )
