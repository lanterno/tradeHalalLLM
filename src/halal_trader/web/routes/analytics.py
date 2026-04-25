"""GET /api/analytics."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from halal_trader.crypto.analytics import PerformanceAnalytics


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    @app.get("/api/analytics")
    async def api_analytics(days: int = 7) -> JSONResponse:
        analytics: PerformanceAnalytics = app_state["analytics"]
        stats = await analytics.compute_stats(lookback_days=days)
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
                "best_pair": stats.best_pair,
                "worst_pair": stats.worst_pair,
                "streak": stats.streak,
                "streak_type": stats.streak_type,
                "by_exit_reason": stats.by_exit_reason,
            }
        )
