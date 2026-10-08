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

    @app.get("/api/recommendation/scorecard")
    async def api_recommendation_scorecard(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        # Read-only aggregate over already-labeled picks; the daily job does
        # the (broker-bound) forward-return backfill, not this endpoint.
        from halal_trader.recommendation.scorecard import compute_scorecard

        sc = await compute_scorecard(ctx.repo)
        return JSONResponse(serialize(sc))

    @app.get("/api/recommendation/whatif")
    async def api_recommendation_whatif(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        # Equity curve of taking every scored pick vs the halal benchmark.
        from halal_trader.recommendation.scorecard import whatif_equity_curve

        curve = await whatif_equity_curve(ctx.repo)
        return JSONResponse(serialize(curve))

    @app.post("/api/recommendation/generate")
    async def api_recommendation_generate(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halal_trader.mcp.client import AlpacaMCPClient
        from halal_trader.recommendation.engine import DailyRecommendationEngine

        # The web has no broker of its own: a throwaway client for this request.
        own_broker = AlpacaMCPClient()
        try:
            await own_broker.connect()
            engine = DailyRecommendationEngine(
                broker=own_broker, repo=ctx.repo, settings=ctx.settings
            )
            rec = await engine.generate()
        except Exception as exc:  # noqa: BLE001 — surface as a structured 502
            raise HTTPException(
                status_code=502, detail=f"recommendation generation failed: {exc}"
            ) from exc
        finally:
            await own_broker.disconnect()
        return JSONResponse(serialize({"available": True, **rec}))
