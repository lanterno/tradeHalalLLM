"""GET /api/positions."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/positions")
    async def api_positions(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        """Open stock positions from the ``trades`` table.

        The stocks broker is REST-only (no streaming price feed on the
        dashboard side), so ``current_price`` falls back to the entry
        price and ``unrealized_pnl`` is 0.
        """
        open_stock_trades = await ctx.repo.get_open_trades()
        positions = []
        for stock_trade in open_stock_trades:
            d = stock_trade.model_dump()
            # Stocks rows have ``symbol``; also surface it as ``pair``,
            # which the dashboard's position template keys on.
            d.setdefault("pair", d.get("symbol"))
            entry = d.get("filled_price") or d.get("price")
            d["entry_price"] = entry
            d["current_price"] = entry
            d["unrealized_pnl"] = 0.0
            d["unrealized_pnl_pct"] = 0.0
            positions.append(d)
        return JSONResponse(serialize(positions))
