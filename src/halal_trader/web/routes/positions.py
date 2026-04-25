"""GET /api/positions."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from halal_trader.db.repository import Repository
from halal_trader.web._serializer import serialize


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    @app.get("/api/positions")
    async def api_positions() -> JSONResponse:
        repo: Repository = app_state["repo"]
        open_trades = await repo.get_open_crypto_trades()
        positions = []
        for t in open_trades:
            d = t.model_dump()
            current = None
            ws_mgr = app_state.get("ws_manager")
            if ws_mgr and hasattr(ws_mgr, "get_latest_price"):
                current = ws_mgr.get_latest_price(t.pair)

            entry = t.entry_price or t.price
            d["entry_price"] = entry
            if current and entry:
                d["current_price"] = current
                d["unrealized_pnl"] = (current - entry) * t.quantity
                d["unrealized_pnl_pct"] = (current - entry) / entry
            else:
                d["current_price"] = entry
                d["unrealized_pnl"] = 0.0
                d["unrealized_pnl_pct"] = 0.0

            positions.append(d)
        return JSONResponse(serialize(positions))
