"""FastAPI dashboard application — REST API + WebSocket + React SPA serving.

The route handlers live in ``halal_trader.web.routes.*`` modules; this file
just composes them with lifespan, middleware, and the SPA static fallback.

Each route takes its dependencies via ``Depends(get_ctx)`` (see
``web/dependencies.py``). The single source of truth for state is the
:class:`~halal_trader.core.context.DashboardContext` attached to
``app.state.ctx`` at lifespan start.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.config import Settings, get_settings
from halal_trader.core.context import DashboardContext, RuntimeView
from halal_trader.core.event_bus import EventBus
from halal_trader.db.models import init_db
from halal_trader.db.repository import Repository
from halal_trader.portfolio.analytics import PerformanceAnalytics
from halal_trader.web.routes import register_all

logger = logging.getLogger(__name__)

_DASHBOARD_DIST = Path(__file__).resolve().parent.parent.parent.parent / "dashboard" / "dist"


def _resolve_static(dist_root: Path, full_path: str) -> Path | None:
    """Resolve ``full_path`` under ``dist_root`` for the SPA catch-all.

    Returns the file only if it stays *inside* ``dist_root`` and is a regular
    file; otherwise ``None`` (the caller serves ``index.html`` for client-side
    routing). This blocks path-traversal (``../../etc/passwd``) from escaping
    the static root — a plain ``dist_root / full_path`` + ``exists()`` check
    would happily serve any file outside dist that the traversal reaches.
    """
    candidate = (dist_root / full_path).resolve()
    if candidate.is_relative_to(dist_root) and candidate.is_file():
        return candidate
    return None


def _start_watchdog(
    engine: AsyncEngine, settings: Settings
) -> tuple[asyncio.Task[None] | None, Any]:
    """Spawn the fleet watchdog (web/watchdog.py) unless it is switched off."""
    interval = settings.web.watchdog_interval_seconds
    if interval <= 0:
        return None, None
    from halal_trader.notifications.telegram import TelegramNotifier
    from halal_trader.web import watchdog

    notifier = TelegramNotifier(
        bot_token=settings.telegram.bot_token, chat_id=settings.telegram.chat_id
    )
    if not notifier.enabled:
        logger.warning("watchdog: Telegram is not configured, so it can only log")
    task = asyncio.create_task(
        watchdog.run(
            engine,
            notifier,
            core_enabled=settings.core.enabled,
            interval_s=float(interval),
        ),
        name="fleet-watchdog",
    )
    return task, notifier


def create_app() -> Any:
    """Create and configure the FastAPI application."""
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles

    from halal_trader.core.observability import new_id, request_id_var

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        engine = await init_db(settings.database_url)
        repo = Repository(engine)
        analytics = PerformanceAnalytics(repo)
        runtime = RuntimeView(started_at=datetime.now(UTC))
        ctx = DashboardContext(
            engine=engine,
            repo=repo,
            analytics=analytics,
            settings=settings,
            bus=EventBus(),
            runtime=runtime,
        )
        _app.state.ctx = ctx
        watchdog_task, notifier = _start_watchdog(engine, settings)
        try:
            yield
        finally:
            if watchdog_task is not None:
                watchdog_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watchdog_task
            if notifier is not None:
                await notifier.close()
            await engine.dispose()

    app = FastAPI(title="Halal Trader Dashboard", version="0.3.0", lifespan=lifespan)

    # No CORS: the built SPA is same-origin, and the Vite dev server
    # (`npm run dev`) proxies /api and /ws.

    # The container healthcheck and the home stack's probe poll /api/health*
    # constantly; their access-log lines drowned everything else.
    from halal_trader.logging import install_health_access_filter

    install_health_access_filter()

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
        _dist_root = _DASHBOARD_DIST.resolve()

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_catch_all(full_path: str) -> FileResponse:
            served = _resolve_static(_dist_root, full_path)
            if served is not None:
                return FileResponse(str(served))
            return FileResponse(str(_dist_root / "index.html"))

    return app
