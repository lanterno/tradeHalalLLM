"""GET /api/pnl/daily — the retired day-trader's daily ledger (``daily_pnl``)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx

MAX_ROWS = 2000


def with_equity_change(row: dict[str, Any]) -> dict[str, Any]:
    """Add ``equity_change``: the day's ending minus starting equity, None
    while the day is open.

    The ledger's ``realized_pnl`` is not realized P&L: the writer
    (trading/portfolio.record_day_end) stores the equity change, which
    includes the open positions' marks, and some older rows hold something
    else again (2026-07-29: +$5,429 against a +$274 equity change). The
    dashboard plots this computed figure and calls it what it is.
    """
    start, end = row.get("starting_equity"), row.get("ending_equity")
    change = None if start is None or end is None else round(float(end) - float(start), 2)
    return {**row, "equity_change": change}


def register(app: FastAPI) -> None:
    @app.get("/api/pnl/daily")
    async def api_daily_pnl(
        days: int = 30,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        """The ledger's days in the last ``days`` calendar days (New York),
        newest first."""
        from halal_trader.market_hours import today_eastern

        days = max(1, min(days, 3660))
        since = today_eastern() - timedelta(days=days)
        pnl = await ctx.repo.get_pnl_history(limit=MAX_ROWS, since=since)
        return JSONResponse(serialize([with_equity_change(r) for r in pnl]))
