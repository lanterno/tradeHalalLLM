"""GET /api/risk/state."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    @app.get("/api/risk/state")
    async def api_risk_state() -> JSONResponse:
        """Return the most recent PortfolioRiskState (cached by the cycle)."""
        state = app_state.get("risk_state")
        if state is None:
            return JSONResponse({"available": False})
        return JSONResponse({"available": True, **state})
