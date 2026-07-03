"""GET /api/halabot/* — the shadow engine's belief board (Task D slice 1).

Read-only bridge from the :8082 dashboard to the halabot shadow engine's
data. Reuses ``halabot.api.queries`` (pure async functions over any
AsyncEngine) against the shared ctx engine — the hb_ tables live in the
same Postgres, so no second server, proxy, or CORS is needed.

Fail-soft: the hb_ tables are created by the shadow daemon's
``bootstrap_schema`` (outside Alembic by design). On a DB where the
shadow has never run, every endpoint degrades to an empty/available:false
payload instead of a 500 — the dashboard renders an honest empty board.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx

logger = logging.getLogger(__name__)


async def _soft(coro: Any, fallback: Any) -> Any:
    """Run a halabot query, degrading to *fallback* when hb_ tables are absent."""
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001 — missing hb_ schema must not 500 the dashboard
        logger.debug("halabot belief query degraded: %r", exc)
        return fallback


def register(app: FastAPI) -> None:
    @app.get("/api/halabot/beliefs")
    async def api_halabot_beliefs(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halabot.api import queries

        rows = await _soft(queries.list_beliefs(ctx.engine), [])
        return JSONResponse({"available": bool(rows), "beliefs": rows})

    @app.get("/api/halabot/beliefs/{asset}")
    async def api_halabot_belief(
        asset: str, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        from halabot.api import queries

        row = await _soft(queries.get_belief(ctx.engine, asset.upper()), None)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no belief for {asset}")
        return JSONResponse(row)

    @app.get("/api/halabot/decisions")
    async def api_halabot_decisions(
        limit: int = 50, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        from halabot.api import queries

        limit = max(1, min(limit, 200))
        rows = await _soft(queries.recent_decisions(ctx.engine, limit=limit), [])
        return JSONResponse(rows)

    @app.get("/api/halabot/decisions/{correlation_id}")
    async def api_halabot_decision_chain(
        correlation_id: str, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        from halabot.api import queries

        try:
            cid = UUID(correlation_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid correlation id") from exc
        rows = await _soft(queries.decision_chain(ctx.engine, cid), [])
        return JSONResponse(rows)

    @app.get("/api/halabot/health")
    async def api_halabot_health(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halabot.api import queries

        health = await _soft(queries.system_health(ctx.engine), None)
        if health is None:
            return JSONResponse({"available": False})
        return JSONResponse({"available": True, **health})
