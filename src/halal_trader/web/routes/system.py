"""System endpoints: /api/health, /api/system/{halt,risk-meter,reconcile}.

The Operations page reads /api/operations (web/operations.py).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import Body, Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from halal_trader import __version__
from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx
from halal_trader.web.middleware.confirm import require_confirmation


class RiskMeterChange(BaseModel):
    level: int
    reason: str = ""


async def _bot_liveness(ctx: DashboardContext) -> tuple[bool, dict[str, Any] | None]:
    """Is the stock bot alive, judged from its heartbeat rows (core/heartbeat).

    The web runs in its own container, so this is the only honest answer it
    can give. ``(False, None)`` when the heartbeats cannot be read at all.
    """
    from halal_trader.core.heartbeat import (
        assess,
        bot_liveness,
        cycles_due_at,
        describe,
        read_beats,
    )

    try:
        beats = await read_beats(ctx.engine)
    except Exception:  # noqa: BLE001 -- health must answer even when the DB can't
        return False, None
    now = datetime.now(UTC)
    cycles_due = cycles_due_at(now)
    statuses = assess(beats, now=now, cycles_due=cycles_due)
    components: dict[str, Any] = describe(beats, statuses, now=now)
    alive, reason = bot_liveness(beats, now=now, cycles_due=cycles_due)
    components["_verdict"] = {"alive": alive, "reason": reason, "cycles_due": cycles_due}
    return alive, components


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
                "version": __version__,
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

    @app.get("/api/system/halt")
    async def api_get_halt(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        from halal_trader.core.halt import get_status

        return JSONResponse((await get_status(ctx.engine)).to_json())

    # Engaging and clearing the kill-switch: the token (auth middleware) and
    # the confirm header every destructive route needs (middleware/confirm.py).
    @app.post("/api/system/halt", dependencies=[Depends(require_confirmation)])
    async def api_set_halt(
        body: dict[str, Any] | None = Body(default=None),
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halal_trader.core.halt import set_halt

        reason = (body or {}).get("reason") or "dashboard"
        s = await set_halt(ctx.engine, reason=reason, set_by="dashboard")
        return JSONResponse(s.to_json())

    @app.delete("/api/system/halt", dependencies=[Depends(require_confirmation)])
    async def api_clear_halt(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        from halal_trader.core.halt import clear_halt

        return JSONResponse((await clear_halt(ctx.engine)).to_json())

    @app.get("/api/system/risk-meter")
    async def api_get_risk_meter(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        from halal_trader.core.risk_meter import get_status, levels_json

        return JSONResponse({**(await get_status(ctx.engine)).to_json(), "levels": levels_json()})

    # Changing the day trader's risk level: confirmed like the kill-switch,
    # recorded by the audit trail, and announced on Telegram.
    @app.put("/api/system/risk-meter", dependencies=[Depends(require_confirmation)])
    async def api_set_risk_meter(
        body: RiskMeterChange, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        from halal_trader.core.risk_meter import MeterRefused, announce, levels_json, set_level

        reason = body.reason.strip() or "dashboard"
        try:
            s = await set_level(ctx.engine, body.level, reason=reason, set_by="dashboard")
        except MeterRefused as e:
            return JSONResponse({"detail": str(e)}, status_code=409)
        await announce(ctx.settings, s)
        return JSONResponse({**s.to_json(), "levels": levels_json()})

    @app.get("/api/system/reconcile/recent")
    async def api_reconcile_recent(
        limit: int = 25, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        from halal_trader.core.reconcile import get_recent_logs

        rows = await get_recent_logs(ctx.engine, limit=max(1, min(limit, 200)))
        return JSONResponse(rows)
