"""GET /api/pnl/daily."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/pnl/daily")
    async def api_daily_pnl(
        days: int = 30,
        market: str = "crypto",
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        """Daily P&L ledger for one market.

        ``market="crypto"`` (default, back-compat) reads
        ``crypto_daily_pnl`` via :meth:`get_crypto_pnl_history`.
        ``market="stocks"`` reads the stocks-side ``daily_pnl`` table
        via :meth:`get_pnl_history`. Anything else 400s — silent
        empties were how the stocks-side day-end row was previously
        invisible to the dashboard.
        """
        market = market.lower()
        if market == "crypto":
            pnl = await ctx.repo.get_crypto_pnl_history(limit=days)
        elif market in ("stock", "stocks"):
            pnl = await ctx.repo.get_pnl_history(limit=days)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"market must be 'crypto' or 'stocks', got {market!r}",
            )
        return JSONResponse(serialize(pnl))
