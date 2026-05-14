"""GET /api/halal/compliance — AAOIFI summary tile data.

Round-4 wave 2.E: returns the AAOIFI compliance summary in a shape
the dashboard tile renders. Read-only; reads from
``halal_screenings``, ``trades``, ``crypto_trades``,
``purification_entries``, and ``round_trip_purification``.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.halal.aaoifi_summary import compute_aaoifi_summary
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/halal/compliance")
    async def api_halal_compliance(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        """Return the live AAOIFI compliance summary.

        Shape:

        ```
        {
          "status": "compliant" | "attention" | "violation",
          "is_compliant": bool,
          "quarter_start": ISO timestamp,
          "month_start": ISO timestamp,
          "today_start": ISO timestamp,
          "trades_today": int,
          "trades_this_month": int,
          "trades_this_quarter": int,
          "halal_screenings_quarter": int,
          "doubtful_screenings_quarter": int,
          "not_halal_screenings_quarter": int,
          "non_halal_fills_quarter": int,
          "purification_accrued_usd": float,
          "purification_disbursed_usd": float,
          "purification_outstanding_usd": float
        }
        ```
        """
        summary = await compute_aaoifi_summary(ctx.engine)
        return JSONResponse(
            {
                "status": summary.status,
                "is_compliant": summary.is_compliant,
                "quarter_start": summary.quarter_start.isoformat(),
                "month_start": summary.month_start.isoformat(),
                "today_start": summary.today_start.isoformat(),
                "trades_today": summary.trades_today,
                "trades_this_month": summary.trades_this_month,
                "trades_this_quarter": summary.trades_this_quarter,
                "halal_screenings_quarter": summary.halal_screenings_quarter,
                "doubtful_screenings_quarter": summary.doubtful_screenings_quarter,
                "not_halal_screenings_quarter": summary.not_halal_screenings_quarter,
                "non_halal_fills_quarter": summary.non_halal_fills_quarter,
                "purification_accrued_usd": summary.purification_accrued_usd,
                "purification_disbursed_usd": summary.purification_disbursed_usd,
                "purification_outstanding_usd": summary.purification_outstanding_usd,
            }
        )
