"""GET /api/pnl/daily."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from halal_trader.db.repository import Repository
from halal_trader.web._serializer import serialize


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    @app.get("/api/pnl/daily")
    async def api_daily_pnl(days: int = 30) -> JSONResponse:
        repo: Repository = app_state["repo"]
        pnl = await repo.get_crypto_pnl_history(limit=days)
        return JSONResponse(serialize(pnl))
