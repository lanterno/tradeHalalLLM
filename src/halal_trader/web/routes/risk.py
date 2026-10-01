"""GET /api/risk/state."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/risk/state")
    async def api_risk_state(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        """The risk read the last trading cycle published (via its heartbeat).

        Used to read in-process state the web container never had, so it
        answered {"available": false} in every deployment.
        """
        from halal_trader.core.heartbeat import cycle_risk

        try:
            state, at = await cycle_risk(ctx.engine)
        except Exception:  # noqa: BLE001 -- an unreadable table is "not available"
            state, at = None, None
        if state is None:
            return JSONResponse({"available": False})
        return JSONResponse(
            {
                "available": True,
                "market": "stocks",
                "pushed_at": at.isoformat() if at else None,
                **state,
            }
        )
