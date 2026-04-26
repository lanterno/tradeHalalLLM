"""GET /api/activity — recent web mutation audit feed."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from halal_trader.db.repository import Repository
from halal_trader.web._serializer import serialize


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    @app.get("/api/activity")
    async def api_activity(limit: int = 50) -> JSONResponse:
        """Recent dashboard mutations, newest first."""
        repo: Repository = app_state["repo"]
        rows = await repo.get_recent_web_actions(limit=limit)
        return JSONResponse(serialize(rows))
