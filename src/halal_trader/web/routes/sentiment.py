"""GET /api/sentiment."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
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
