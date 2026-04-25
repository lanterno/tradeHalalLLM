"""FastAPI dashboard application — REST API + WebSocket + React SPA serving.

The route handlers live in ``halal_trader.web.routes.*`` modules; this file
just composes them with lifespan, middleware, and the SPA static fallback.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from halal_trader.config import get_settings
from halal_trader.crypto.analytics import PerformanceAnalytics
from halal_trader.db.models import init_db
from halal_trader.db.repository import Repository
from halal_trader.web.routes import register_all

logger = logging.getLogger(__name__)

_DASHBOARD_DIST = Path(__file__).resolve().parent.parent.parent.parent / "dashboard" / "dist"


app_state: dict[str, Any] = {}


def create_app() -> Any:
    """Create and configure the FastAPI application."""
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, Response
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

    register_all(app, app_state)

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
