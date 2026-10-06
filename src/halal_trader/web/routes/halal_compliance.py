"""GET /api/halal/compliance — the Halal page's compliance summary.

Read-only; see ``halal/aaoifi_summary.py`` for what is counted and how a
buy is judged. Shape:

```
{
  "status": "compliant" | "attention" | "violation",   # the worst account's
  "is_compliant": bool,
  "quarter_start" / "month_start" / "today_start": "YYYY-MM-DD" (New York),
  "trades_today" / "trades_this_month" / "trades_this_quarter": int (all accounts),
  "non_halal_fills_quarter": int,      # buys the screen did not hold halal
  "accounts": [                         # the core first
    {"account", "label", "status", "trades_today", "trades_this_month",
     "trades_this_quarter", "buys_this_quarter",
     "buy_verdicts": {"halal", "doubtful", "not_halal", "unscreened"},
     "non_halal_buys_quarter", "non_halal_buys": [{symbol, day, verdict, screen_as_of}]}
  ],
  "purification_accrued_usd" / "purification_disbursed_usd": float (this quarter),
  "purification_outstanding_usd": float (all time),
  "purification_unpaid_by_account": {account: float}
}
```
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
        """Return the compliance summary, per account."""
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
                "non_halal_fills_quarter": summary.non_halal_fills_quarter,
                "accounts": [
                    {
                        "account": a.account,
                        "label": a.label,
                        "status": a.status,
                        "trades_today": a.trades_today,
                        "trades_this_month": a.trades_this_month,
                        "trades_this_quarter": a.trades_this_quarter,
                        "buys_this_quarter": a.buys_this_quarter,
                        "buy_verdicts": a.buy_verdicts,
                        "non_halal_buys_quarter": a.non_halal_buys_quarter,
                        "non_halal_buys": a.non_halal_buys,
                    }
                    for a in summary.accounts
                ],
                "purification_accrued_usd": summary.purification_accrued_usd,
                "purification_disbursed_usd": summary.purification_disbursed_usd,
                "purification_outstanding_usd": summary.purification_outstanding_usd,
                "purification_unpaid_by_account": summary.purification_unpaid_by_account,
            }
        )
