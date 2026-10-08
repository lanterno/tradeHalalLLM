"""GET /api/halabot/* — the shadow engine's belief board.

``/beliefs`` and ``/overview`` are what the Belief Board page polls; their
payloads are assembled in ``web/belief_board.py``.

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
from datetime import UTC, datetime
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx
from halal_trader.web.soft import soft

logger = logging.getLogger(__name__)


def register(app: FastAPI) -> None:
    @app.get("/api/halabot/beliefs")
    async def api_halabot_beliefs(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halal_trader.portfolio.core_account import core_account
        from halal_trader.web import belief_board

        body = await belief_board.board(
            ctx.engine,
            core_account=core_account(ctx.settings.core.paper),
            now=datetime.now(UTC),
        )
        return JSONResponse(body)

    @app.get("/api/halabot/overview")
    async def api_halabot_overview(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        """The board's trust strip and side column in one read."""
        from halal_trader.web import belief_board

        return JSONResponse(await belief_board.overview(ctx.engine, now=datetime.now(UTC)))

    @app.get("/api/halabot/beliefs/{asset}")
    async def api_halabot_belief(
        asset: str, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        from halabot.api import queries

        row = await soft("halabot query", queries.get_belief(ctx.engine, asset.upper()), None)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no belief for {asset}")
        return JSONResponse(row)

    @app.get("/api/halabot/decisions")
    async def api_halabot_decisions(
        limit: int = 50,
        asset: str | None = None,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halabot.api import queries
        from halal_trader.web import belief_board

        limit = max(1, min(limit, 200))
        rows = await soft(
            "halabot query",
            queries.recent_decisions(
                ctx.engine, limit=limit, asset=asset.upper() if asset else None
            ),
            [],
        )
        return JSONResponse(belief_board.decorate_decisions(rows))

    @app.get("/api/halabot/decisions/{correlation_id}")
    async def api_halabot_decision_chain(
        correlation_id: str, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        from halabot.api import queries

        try:
            cid = UUID(correlation_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid correlation id") from exc
        rows = await soft("halabot query", queries.decision_chain(ctx.engine, cid), [])
        return JSONResponse(rows)

    @app.get("/api/halabot/health")
    async def api_halabot_health(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halabot.api import queries

        health = await soft("halabot query", queries.system_health(ctx.engine), None)
        if health is None:
            return JSONResponse({"available": False})
        return JSONResponse({"available": True, **health})
