"""GET /api/analytics."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/analytics")
    async def api_analytics(
        days: int = 7,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        """Rolling P&L / win-rate / drawdown stats over closed stock round-trips.

        Reads ``ctx.analytics`` (the :class:`PerformanceAnalytics` built
        at dashboard lifespan).
        """
        stats = await ctx.analytics.compute_stats(lookback_days=days)
        pf = stats.profit_factor
        if pf == float("inf"):
            pf = 999999.0
        return JSONResponse(
            {
                "total_trades": stats.total_trades,
                "wins": stats.wins,
                "losses": stats.losses,
                "win_rate": stats.win_rate,
                "avg_win_pct": stats.avg_win_pct,
                "avg_loss_pct": stats.avg_loss_pct,
                "total_pnl": stats.total_pnl,
                "profit_factor": pf,
                "max_drawdown_pct": stats.max_drawdown_pct,
                "avg_hold_minutes": stats.avg_hold_minutes,
                "best_symbol": stats.best_symbol,
                "worst_symbol": stats.worst_symbol,
                "streak": stats.streak,
                "streak_type": stats.streak_type,
                "by_exit_reason": stats.by_exit_reason,
                "exits": [asdict(e) for e in stats.exits],
            }
        )
