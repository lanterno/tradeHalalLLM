"""System endpoints: /api/health, /api/system/{status,halt,reconcile,backups}."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import Body, Depends, FastAPI, Header
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


async def _bot_liveness(ctx: DashboardContext) -> tuple[bool, dict[str, Any] | None]:
    """Is the stock bot alive, judged from its heartbeat rows (core/heartbeat).

    The web runs in its own container, so this is the only honest answer it
    can give. ``(False, None)`` when the heartbeats cannot be read at all.
    """
    from halal_trader.core.heartbeat import STOCK_PROCESS, read_beats

    try:
        beats = await read_beats(ctx.engine)
    except Exception:  # noqa: BLE001 -- health must answer even when the DB can't
        return False, None
    now = datetime.now(UTC)
    components = {
        name: {
            "beat_at": b.beat_at.isoformat(),
            "age_seconds": round(b.age(now).total_seconds(), 1),
            "stale": b.is_stale(now),
            "detail": b.detail,
        }
        for name, b in beats.items()
    }
    process = beats.get(STOCK_PROCESS)
    return (process is not None and not process.is_stale(now)), components


def register(app: FastAPI) -> None:
    @app.get("/api/health")
    async def api_health(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        """Web liveness (always 200 while the web serves) plus the bot's beats.

        `status` used to be a hard-coded "running" that drove the dashboard's
        "Bot Running" badge whether or not any bot existed. `bot_alive` is now
        the real signal; /api/health/bot turns it into an HTTP status.
        """
        alive, components = await _bot_liveness(ctx)
        return JSONResponse(
            {
                "status": "running",
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "0.3.0",
                "bot_alive": alive,
                "bot": components,
            }
        )

    @app.get("/api/health/bot")
    async def api_health_bot(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        """200 if the stock bot's process heartbeat is fresh, else 503."""
        alive, components = await _bot_liveness(ctx)
        return JSONResponse(
            {"bot_alive": alive, "bot": components}, status_code=200 if alive else 503
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

        # Classifier health — added after the 2026-05-22 quota incident
        # so "is the brain healthy" is one HTTP call instead of grepping
        # JSON logs. None when bot is dashboard-only or reactor isn't
        # configured (no Finnhub key / empty halal watchlist).
        classifier_health: dict[str, Any] | None = None
        reactor = getattr(ctx.runtime, "stocks_news_reactor", None)
        if reactor is not None:
            classifier = getattr(reactor, "classifier", None)
            if classifier is not None and hasattr(classifier, "get_telemetry"):
                try:
                    classifier_health = classifier.get_telemetry()
                except Exception:  # noqa: BLE001
                    classifier_health = None

        # Both market cadences — crypto runs every 60s by default,
        # stocks on a 15-min cron. Frontends used to read only the
        # crypto value and showed "60s" for stocks operators.
        # ``cycle_interval_seconds`` is preserved (= crypto) so legacy
        # frontends keep working; ``stocks_cycle_interval_seconds`` is
        # the new field a market-aware dashboard reads.
        from halal_trader.core.heartbeat import STOCK_CYCLE

        alive, components = await _bot_liveness(ctx)
        cycle_beat = (components or {}).get(STOCK_CYCLE)
        return JSONResponse(
            {
                # From the bot's heartbeat rows, not in-process state this
                # container never had (which always read "Bot Running: No").
                "bot_running": alive,
                "last_cycle": ctx.runtime.last_cycle
                or (cycle_beat["beat_at"] if cycle_beat else None),
                "cycle_interval_seconds": ctx.settings.crypto.trading_interval_seconds,
                "crypto_cycle_interval_seconds": ctx.settings.crypto.trading_interval_seconds,
                "stocks_cycle_interval_seconds": ctx.settings.stocks.trading_interval_minutes * 60,
                "ws_health": ws_health,
                "classifier_health": classifier_health,
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
