"""GET /api/trades."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from halal_trader.db.repository import Repository
from halal_trader.web._serializer import serialize


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    @app.get("/api/trades")
    async def api_trades(
        limit: int = 100,
        offset: int = 0,
        pair: str | None = None,
        side: str | None = None,
        status: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> JSONResponse:
        repo: Repository = app_state["repo"]
        trades = await repo.get_recent_crypto_trades(limit=limit + offset)
        result = trades[offset:]
        if pair:
            result = [t for t in result if t.get("pair") == pair]
        if side:
            result = [t for t in result if t.get("side") == side]
        if status:
            result = [t for t in result if t.get("status") == status]
        if from_date:
            result = [t for t in result if (t.get("timestamp") or "") >= from_date]
        if to_date:
            result = [t for t in result if (t.get("timestamp") or "") <= to_date]
        return JSONResponse(serialize(result[:limit]))
