"""System endpoints: /api/health, /api/system/{status,halt,reconcile,backups}."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import Body, Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx
from halal_trader.web.middleware.confirm import require_confirmation


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

        from halal_trader.core.heartbeat import STOCK_CYCLE, core_running

        alive, components = await _bot_liveness(ctx)
        cycle_beat = (components or {}).get(STOCK_CYCLE)
        return JSONResponse(
            {
                # From the bot's heartbeat rows, not in-process state this
                # container never had (which always read "Bot Running: No").
                "bot_running": alive,
                "last_cycle": ctx.runtime.last_cycle
                or (cycle_beat["beat_at"] if cycle_beat else None),
                "stocks_cycle_interval_seconds": ctx.settings.stocks.trading_interval_minutes * 60,
                # The core runs only with its own account's keys set; the bot
                # says so in its process beat (the web never sees those keys).
                "core_enabled": await core_running(ctx.engine),
                "classifier_health": classifier_health,
                "uptime_seconds": uptime,
            }
        )

    @app.get("/api/system/core-config")
    async def api_core_config(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        """The core portfolio's parameters for the System page: its settings
        (never its keys) and the rule constants it trades by."""
        from halal_trader.core.heartbeat import core_running
        from halal_trader.portfolio import core_executor, strict_core
        from halal_trader.portfolio.home import CORE_TRADE

        core = ctx.settings.core
        return JSONResponse(
            {
                "core_enabled": await core_running(ctx.engine),
                "core_paper": core.paper,
                "core_top_n": core.top_n,
                "core_rebalance_band": strict_core.BAND,
                "core_band_floor": strict_core.BAND_FLOOR,
                "core_min_trade_usd": core_executor.MIN_TRADE_FLOOR,
                "core_min_trade_fraction": core_executor.MIN_TRADE_FRACTION,
                "core_cash_buffer": core_executor.CASH_BUFFER,
                "core_max_screen_age_days": core_executor.MAX_SCREEN_AGE.days,
                "core_trades_at_et": CORE_TRADE.strftime("%H:%M"),
            }
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
