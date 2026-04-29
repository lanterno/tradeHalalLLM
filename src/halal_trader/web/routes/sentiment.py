"""GET /api/sentiment."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/sentiment")
    async def api_sentiment(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        mgr = ctx.runtime.sentiment_manager
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
