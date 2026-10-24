"""GET /api/trades."""

from __future__ import annotations

from datetime import date

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/trades")
    async def api_trades(
        limit: int = 100,
        offset: int = 0,
        symbol: str | None = None,
        side: str | None = None,
        status: str | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        """Stock trades from the ``trades`` table, newest first.

        The filters run in the query, before the page is cut, so a page of
        one symbol is that symbol's trades and not the newest page's share
        of them. ``from_date``/``to_date`` are Eastern trading days, inclusive.
        """
        trades = await ctx.repo.find_trades(
            limit=max(1, min(limit, 1000)),
            offset=max(0, offset),
            symbol=symbol,
            side=side,
            status=status,
            first_day=from_date,
            last_day=to_date,
        )
        return JSONResponse(serialize(trades))
