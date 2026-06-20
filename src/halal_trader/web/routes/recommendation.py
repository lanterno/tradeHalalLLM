"""GET /api/recommendation, /api/recommendation/history; POST /api/recommendation/generate.

Advisory "stock of the day" surface. GET endpoints are public-read; the POST
regenerate is auth-gated + audited automatically by the middleware (non-GET
/api/ path). Generation never trades — it only runs the recommendation engine.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/recommendation")
    async def api_recommendation_latest(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        rec = await ctx.repo.get_latest_recommendation()
        if rec is None:
            # 200 with a sentinel (not 404) so the SPA can render an empty
            # state without its apiFetch wrapper throwing.
            return JSONResponse({"available": False})
        return JSONResponse(serialize({"available": True, **rec}))

    @app.get("/api/recommendation/history")
    async def api_recommendation_history(
        limit: int = 30, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        limit = max(1, min(limit, 200))
        rows = await ctx.repo.get_recent_recommendations(limit=limit)
        return JSONResponse(serialize(rows))

    @app.post("/api/recommendation/generate")
    async def api_recommendation_generate(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halal_trader.recommendation.engine import DailyRecommendationEngine

        # Prefer the live bot's broker if co-hosted; else spin up a throwaway
        # Alpaca MCP client for this request (the web process usually has none).
        broker = ctx.runtime.stock_broker
        own_broker = None
        try:
            if broker is None:
                from halal_trader.mcp.client import AlpacaMCPClient

                own_broker = AlpacaMCPClient()
                await own_broker.connect()
                broker = own_broker
            engine = DailyRecommendationEngine(
                broker=broker, repo=ctx.repo, settings=ctx.settings
            )
            rec = await engine.generate()
        except Exception as exc:  # noqa: BLE001 — surface as a structured 502
            raise HTTPException(
                status_code=502, detail=f"recommendation generation failed: {exc}"
            ) from exc
        finally:
            if own_broker is not None:
                await own_broker.disconnect()
        return JSONResponse(serialize({"available": True, **rec}))
