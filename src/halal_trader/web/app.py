"""FastAPI dashboard application — REST API + WebSocket + React SPA serving."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from halal_trader.config import get_settings
from halal_trader.crypto.analytics import PerformanceAnalytics
from halal_trader.db.models import init_db
from halal_trader.db.repository import Repository

logger = logging.getLogger(__name__)

_DASHBOARD_DIST = Path(__file__).resolve().parent.parent.parent.parent / "dashboard" / "dist"


def _serialize(obj: Any) -> Any:
    """Convert datetime values in dicts/lists to ISO strings for JSON."""
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


app_state: dict[str, Any] = {}


def create_app() -> Any:
    """Create and configure the FastAPI application."""
    from fastapi import Body, FastAPI, Header, Query, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    from halal_trader.core.observability import new_id, request_id_var

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        settings = get_settings()
        engine = await init_db(str(settings.resolve_db_path()))
        app_state["engine"] = engine
        app_state["repo"] = Repository(engine)
        app_state["analytics"] = PerformanceAnalytics(app_state["repo"])
        app_state["started_at"] = datetime.now(timezone.utc)
        yield
        if "engine" in app_state:
            await app_state["engine"].dispose()

    app = FastAPI(title="Halal Trader Dashboard", version="0.2.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def correlate_request(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or new_id("req")
        token = request_id_var.set(rid)
        try:
            response: Response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response

    # ── REST API ───────────────────────────────────────────────

    @app.get("/api/trades")
    async def api_trades(
        limit: int = 100,
        offset: int = 0,
        pair: str | None = None,
        side: str | None = None,
        status: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> JSONResponse:
        repo: Repository = app_state["repo"]
        trades = await repo.get_recent_crypto_trades(limit=limit + offset)
        result = trades[offset:]

        if pair:
            result = [t for t in result if t.get("pair") == pair]
        if side:
            result = [t for t in result if t.get("side") == side]
        if status:
            result = [t for t in result if t.get("status") == status]
        if from_date:
            result = [t for t in result if (t.get("timestamp") or "") >= from_date]
        if to_date:
            result = [t for t in result if (t.get("timestamp") or "") <= to_date]

        return JSONResponse(_serialize(result[:limit]))

    @app.get("/api/pnl/daily")
    async def api_daily_pnl(days: int = 30) -> JSONResponse:
        repo: Repository = app_state["repo"]
        pnl = await repo.get_crypto_pnl_history(limit=days)
        return JSONResponse(_serialize(pnl))

    @app.get("/api/analytics")
    async def api_analytics(days: int = 7) -> JSONResponse:
        analytics: PerformanceAnalytics = app_state["analytics"]
        stats = await analytics.compute_stats(lookback_days=days)
        pf = stats.profit_factor
        if pf == float("inf"):
            pf = 999999.0
        return JSONResponse(
            {
                "total_trades": stats.total_trades,
                "wins": stats.wins,
                "losses": stats.losses,
                "win_rate": stats.win_rate,
                "avg_win_pct": stats.avg_win_pct,
                "avg_loss_pct": stats.avg_loss_pct,
                "total_pnl": stats.total_pnl,
                "profit_factor": pf,
                "max_drawdown_pct": stats.max_drawdown_pct,
                "avg_hold_minutes": stats.avg_hold_minutes,
                "best_pair": stats.best_pair,
                "worst_pair": stats.worst_pair,
                "streak": stats.streak,
                "streak_type": stats.streak_type,
                "by_exit_reason": stats.by_exit_reason,
            }
        )

    @app.get("/api/positions")
    async def api_positions() -> JSONResponse:
        repo: Repository = app_state["repo"]
        open_trades = await repo.get_open_crypto_trades()
        positions = []
        for t in open_trades:
            d = t.model_dump()
            current = None
            ws_mgr = app_state.get("ws_manager")
            if ws_mgr and hasattr(ws_mgr, "get_latest_price"):
                current = ws_mgr.get_latest_price(t.pair)

            entry = t.entry_price or t.price
            d["entry_price"] = entry
            if current and entry:
                d["current_price"] = current
                d["unrealized_pnl"] = (current - entry) * t.quantity
                d["unrealized_pnl_pct"] = (current - entry) / entry
            else:
                d["current_price"] = entry
                d["unrealized_pnl"] = 0.0
                d["unrealized_pnl_pct"] = 0.0

            positions.append(d)
        return JSONResponse(_serialize(positions))

    @app.get("/api/decisions")
    async def api_decisions(limit: int = 50) -> JSONResponse:
        repo: Repository = app_state["repo"]
        decisions = await repo.get_recent_decisions(limit=limit)
        return JSONResponse(_serialize(decisions))

    @app.get("/api/adjustments")
    async def api_adjustments() -> JSONResponse:
        repo: Repository = app_state["repo"]
        adjustments = await repo.get_recent_adjustments()
        return JSONResponse(_serialize(adjustments))

    @app.get("/api/sentiment")
    async def api_sentiment() -> JSONResponse:
        mgr = app_state.get("sentiment_manager")
        if not mgr or not hasattr(mgr, "latest_signals"):
            return JSONResponse([])

        signals = []
        for pair, sig in mgr.latest_signals.items():
            signals.append(
                {
                    "pair": pair,
                    "score": getattr(sig, "score", 0),
                    "buzz": getattr(sig, "buzz", 0),
                    "confidence": getattr(sig, "confidence", 0),
                    "top_narratives": getattr(sig, "top_narratives", []) or [],
                    "news_headlines": getattr(sig, "news_headlines", []) or [],
                    "data_sources": getattr(sig, "data_sources", []) or [],
                }
            )
        return JSONResponse(signals)

    @app.get("/api/config")
    async def api_config() -> JSONResponse:
        settings = get_settings()
        return JSONResponse(
            {
                "llm_provider": settings.llm_provider.value,
                "llm_model": settings.llm_model,
                "crypto_pairs": settings.crypto_pairs,
                "crypto_trading_interval_seconds": settings.crypto_trading_interval_seconds,
                "crypto_max_position_pct": settings.crypto_max_position_pct,
                "crypto_daily_loss_limit": settings.crypto_daily_loss_limit,
                "crypto_daily_return_target": settings.crypto_daily_return_target,
                "db_path": str(settings.db_path),
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
                "cycle_interval_seconds": get_settings().crypto_trading_interval_seconds,
                "ws_health": ws_health,
                "uptime_seconds": uptime,
            }
        )

    @app.get("/api/health")
    async def api_health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "running",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "0.2.0",
            }
        )

    @app.get("/api/metrics/cycles")
    async def api_metrics_cycles(window: int = 3600) -> JSONResponse:
        from halal_trader.web.metrics import cycle_metrics

        settings = get_settings()
        log_path = settings.log_dir / "halal_trader.log"
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
        log_path = settings.log_dir / "halal_trader.log"
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

    # ── Risk + system-status ───────────────────────────────────

    @app.get("/api/risk/state")
    async def api_risk_state() -> JSONResponse:
        """Return the most recent PortfolioRiskState (cached by the cycle)."""
        state = app_state.get("risk_state")
        if state is None:
            return JSONResponse({"available": False})
        return JSONResponse({"available": True, **state})

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
        rows = list_backups(settings.backup_dir)
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

    # ── SSE Live Updates ───────────────────────────────────────

    @app.get("/api/sse")
    async def sse() -> StreamingResponse:
        async def event_stream():
            while True:
                repo: Repository = app_state["repo"]
                trades = await repo.get_recent_crypto_trades(5)
                data = json.dumps({"trades": trades}, default=str)
                yield f"data: {data}\n\n"
                await asyncio.sleep(10)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ── WebSocket Live Prices ──────────────────────────────────

    @app.websocket("/ws/prices")
    async def ws_prices(
        websocket: WebSocket,
        symbols: list[str] = Query(default=[]),
    ) -> None:
        await websocket.accept()
        try:
            while True:
                prices: dict[str, float] = {}
                ws_mgr = app_state.get("ws_manager")
                for sym in symbols:
                    price = None
                    if ws_mgr and hasattr(ws_mgr, "get_latest_price"):
                        price = ws_mgr.get_latest_price(sym)
                    if price is not None:
                        prices[sym] = price
                if prices:
                    await websocket.send_json(prices)
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("WebSocket connection closed")

    # ── Static Files / SPA ─────────────────────────────────────

    if _DASHBOARD_DIST.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(_DASHBOARD_DIST / "assets")),
            name="static",
        )

        @app.get("/{full_path:path}")
        async def spa_catch_all(full_path: str) -> FileResponse:
            file = _DASHBOARD_DIST / full_path
            if file.exists() and file.is_file():
                return FileResponse(str(file))
            return FileResponse(str(_DASHBOARD_DIST / "index.html"))

    return app
