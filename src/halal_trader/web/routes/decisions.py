"""GET /api/decisions, /api/adjustments."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from halal_trader.db.repository import Repository
from halal_trader.web._serializer import serialize


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    @app.get("/api/decisions")
    async def api_decisions(limit: int = 50) -> JSONResponse:
        repo: Repository = app_state["repo"]
        decisions = await repo.get_recent_decisions(limit=limit)
        return JSONResponse(serialize(decisions))

    @app.get("/api/adjustments")
    async def api_adjustments() -> JSONResponse:
        repo: Repository = app_state["repo"]
        adjustments = await repo.get_recent_adjustments()
        return JSONResponse(serialize(adjustments))
