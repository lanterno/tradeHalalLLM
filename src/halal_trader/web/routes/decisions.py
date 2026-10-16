"""GET /api/decisions, /api/decisions/quality, /api/adjustments."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx
from halal_trader.web.soft import soft


def register(app: FastAPI) -> None:
    @app.get("/api/decisions")
    async def api_decisions(
        limit: int = 50, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        decisions = await ctx.repo.get_recent_decisions(limit=limit)
        return JSONResponse(serialize(decisions))

    @app.get("/api/decisions/quality")
    async def api_decision_quality(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        """The evening run's score of the last 30 days' round trips against SPY
        (trading/decision_quality.py), with when it was computed; null before the
        first run."""
        from halal_trader.core.heartbeat import read_beat
        from halal_trader.trading.decision_quality import BEAT

        b = await soft("decision quality", read_beat(ctx.engine, BEAT), None)
        if b is None:
            return JSONResponse(None)
        return JSONResponse({"computed_at": b.beat_at.isoformat(), **(b.detail or {})})

    @app.get("/api/adjustments")
    async def api_adjustments(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        adjustments = await ctx.repo.get_recent_adjustments()
        return JSONResponse(serialize(adjustments))
