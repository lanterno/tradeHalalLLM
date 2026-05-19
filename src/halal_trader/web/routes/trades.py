"""GET /api/trades."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/trades")
    async def api_trades(
        limit: int = 100,
        offset: int = 0,
        pair: str | None = None,
        symbol: str | None = None,
        side: str | None = None,
        status: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        market: str = "crypto",
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        """Recent trades for one market.

        ``market="crypto"`` (default, back-compat) reads
        ``crypto_trades`` via ``get_recent_crypto_trades``; rows are
        keyed by ``pair``. ``market="stocks"`` reads ``trades`` via
        ``get_recent_trades``; rows are keyed by ``symbol``. The
        ``pair`` and ``symbol`` filters are both accepted and applied
        to whichever key the row carries, so a frontend that doesn't
        know the market discriminator still gets sensible filtering.
        """
        market = market.lower()
        trades: list[dict[str, Any]]
        if market == "crypto":
            trades = await ctx.repo.get_recent_crypto_trades(limit=limit + offset)
        elif market in ("stock", "stocks"):
            trades = await ctx.repo.get_recent_trades(limit=limit + offset)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"market must be 'crypto' or 'stocks', got {market!r}",
            )
        result = trades[offset:]
        # Stocks rows expose ``symbol``; crypto rows expose ``pair``.
        # Both filters work on either by checking whichever key the
        # row has — saves the frontend a market switch.
        entity_filter = pair or symbol
        if entity_filter:
            result = [
                t
                for t in result
                if t.get("pair") == entity_filter or t.get("symbol") == entity_filter
            ]
        if side:
            result = [t for t in result if t.get("side") == side]
        if status:
            result = [t for t in result if t.get("status") == status]
        if from_date:
            result = [t for t in result if (t.get("timestamp") or "") >= from_date]
        if to_date:
            result = [t for t in result if (t.get("timestamp") or "") <= to_date]
        return JSONResponse(serialize(result[:limit]))
