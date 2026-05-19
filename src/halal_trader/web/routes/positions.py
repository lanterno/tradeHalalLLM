"""GET /api/positions."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/positions")
    async def api_positions(
        market: str = "crypto",
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        """Open positions for one market.

        ``market="crypto"`` (default, back-compat) reads
        ``get_open_crypto_trades`` and enriches each row with the
        latest WebSocket price for unrealized P&L. ``market="stocks"``
        reads ``get_open_trades`` from the stocks ``trades`` table;
        the WS-price enrichment is skipped because the stocks broker
        is REST-only (no streaming price feed on the dashboard side
        today), so ``current_price`` falls back to entry and
        ``unrealized_pnl`` is 0. The fields surface the same shape
        either way so the frontend renders both with one template.
        """
        market = market.lower()
        if market == "crypto":
            open_trades = await ctx.repo.get_open_crypto_trades()
            ws_mgr = ctx.runtime.ws_manager
            positions = []
            for t in open_trades:
                d = t.model_dump()
                current = None
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

        if market in ("stock", "stocks"):
            open_stock_trades = await ctx.repo.get_open_trades()
            positions = []
            for stock_trade in open_stock_trades:
                d = stock_trade.model_dump()
                # Stocks have ``symbol`` not ``pair``; surface ``pair``
                # too so the frontend's existing crypto-shape templates
                # render without a discriminator.
                d.setdefault("pair", d.get("symbol"))
                entry = d.get("filled_price") or d.get("price")
                d["entry_price"] = entry
                d["current_price"] = entry
                d["unrealized_pnl"] = 0.0
                d["unrealized_pnl_pct"] = 0.0
                positions.append(d)
            return JSONResponse(serialize(positions))

        raise HTTPException(
            status_code=400,
            detail=f"market must be 'crypto' or 'stocks', got {market!r}",
        )
