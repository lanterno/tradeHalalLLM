"""FastAPI dashboard application — REST API + WebSocket + React SPA serving.

The route handlers live in ``halal_trader.web.routes.*`` modules; this file
just composes them with lifespan, middleware, and the SPA static fallback.

Each route takes its dependencies via ``Depends(get_ctx)`` (see
``web/dependencies.py``). The single source of truth for state is the
:class:`~halal_trader.core.context.DashboardContext` attached to
``app.state.ctx`` at lifespan start.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from halal_trader.config import get_settings
from halal_trader.core.context import DashboardContext, RuntimeView
from halal_trader.core.event_bus import EventBus
from halal_trader.core.insights_hub import InsightsHub
from halal_trader.crypto.analytics import PerformanceAnalytics
from halal_trader.db.models import init_db
from halal_trader.db.repository import Repository
from halal_trader.web.routes import register_all

logger = logging.getLogger(__name__)

_DASHBOARD_DIST = Path(__file__).resolve().parent.parent.parent.parent / "dashboard" / "dist"


def create_app() -> Any:
    """Create and configure the FastAPI application."""
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles

    from halal_trader.core.observability import new_id, request_id_var

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Co-host pattern: when the bot ran ``attach_to_app`` before the
        # dashboard's lifespan fired, ``_app.state.ctx`` is already set
        # to the bot's projected DashboardContext. In that mode the bot
        # owns the engine, hub, and event bus — the lifespan must NOT
        # build a parallel set or dispose the engine on shutdown.
        if getattr(_app.state, "ctx", None) is not None:
            yield
            return

        settings = get_settings()
        engine = await init_db(settings.database_url)
        repo = Repository(engine)
        analytics = PerformanceAnalytics(repo)
        # The standalone dashboard process builds an empty hub —
        # DB-backed insights (regime, replay, exception queue) come
        # through the engine directly; the in-memory ones stay empty
        # until a co-hosted bot writes to them.
        from halal_trader.ml.regime_memory import RegimeMemory

        hub = InsightsHub(regime=RegimeMemory(engine=engine))
        runtime = RuntimeView(started_at=datetime.now(UTC))
        ctx = DashboardContext(
            engine=engine,
            repo=repo,
            hub=hub,
            analytics=analytics,
            settings=settings,
            bus=EventBus(),
            runtime=runtime,
        )
        _app.state.ctx = ctx
        yield
        await engine.dispose()

    app = FastAPI(title="Halal Trader Dashboard", version="0.3.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware execution is LIFO — the LAST registered runs FIRST
    # on inbound. To get the documented flow
    #   correlate (outermost) → auth → audit (innermost) → handler
    # the registrations must run audit → auth → correlate so the stack
    # is built with audit at the bottom and correlate at the top.
    #
    # Why this order matters:
    # * correlate must run before auth so a rejected auth request
    #   still gets an ``X-Request-ID`` header (operator-facing 401s
    #   were previously untraceable).
    # * correlate must run before audit so ``request_id_var`` is set
    #   when audit writes its row (every ``web_actions.actor`` was
    #   previously the default ``"anon"``).
    from halal_trader.web.audit import audit_middleware
    from halal_trader.web.middleware.auth import auth_middleware

    app.middleware("http")(audit_middleware)
    app.middleware("http")(auth_middleware)

    @app.middleware("http")
    async def correlate_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get("X-Request-ID") or new_id("req")
        token = request_id_var.set(rid)
        try:
            response: Response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response

    register_all(app)

    if _DASHBOARD_DIST.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(_DASHBOARD_DIST / "assets")),
            name="static",
        )

        # Excluded from the OpenAPI schema so the FileResponse return-type
        # forward ref doesn't crash pydantic's schema generation at
        # /openapi.json (and therefore /docs).
        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_catch_all(full_path: str) -> FileResponse:
            file = _DASHBOARD_DIST / full_path
            if file.exists() and file.is_file():
                return FileResponse(str(file))
            return FileResponse(str(_DASHBOARD_DIST / "index.html"))

    return app
