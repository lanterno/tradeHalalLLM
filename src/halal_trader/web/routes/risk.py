"""GET /api/risk/state."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/risk/state")
    async def api_risk_state(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        """Return the most recent PortfolioRiskState (cached by the cycle)."""
        state = ctx.runtime.risk_state
        if state is None:
            return JSONResponse({"available": False})
        return JSONResponse({"available": True, **state})
