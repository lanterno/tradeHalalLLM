"""GET /api/halal/zakat and /api/halal/purification — the Halal page's money tiles.

Read-only. Zakat: the next hawl, the latest recorded assessment, and both of
Dar al-Ifta's methods "if it were due today" for the real (paper) account.
Purification: the dividend ledger per holding for a payment year.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


def _assessment_json(a: Any) -> dict[str, Any]:
    return {
        "period_start": a.period_start.isoformat(),
        "hawl_date": a.hawl_date.isoformat(),
        "market_value": round(a.market_value, 2),
        "trade_goods_zakat": round(a.trade_goods, 2),
        "dividends": round(a.dividends, 2),
        "purified": round(a.purified, 2),
        "income_zakat": round(a.income, 2),
        "chosen": a.chosen,
        "amount": round(a.amount, 2),
    }


def register(app: FastAPI) -> None:
    @app.get("/api/halal/zakat")
    async def api_halal_zakat(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        from halal_trader.compliance import zakat as z
        from halal_trader.config import get_settings
        from halal_trader.market_hours import today_eastern

        hawl_hijri = get_settings().zakat.hawl_hijri
        if not hawl_hijri:
            return JSONResponse({"configured": False, "source": z.SOURCE})
        hawl = z.parse_hawl(hawl_hijri)
        today = today_eastern()
        last_start, last_hawl = z.hawl_period(hawl, today)
        # The next hawl: the one on or before a day a lunar year ahead.
        _, next_hawl = z.hawl_period(hawl, date.fromordinal(last_hawl.toordinal() + 360))
        accounts = []
        for account, label in (("core", "Core portfolio"), ("paper", "Day-trader")):
            now = await z.assess(ctx.engine, account, period_start=last_hawl, hawl_date=today)
            async with ctx.engine.connect() as conn:
                row = (
                    await conn.execute(
                        text(
                            "SELECT hawl_date, hawl_hijri, amount, chosen FROM zakat_assessments "
                            "WHERE account = :a ORDER BY hawl_date DESC LIMIT 1"
                        ),
                        {"a": account},
                    )
                ).first()
            accounts.append(
                {
                    "account": account,
                    "label": label,
                    "if_due_today": _assessment_json(now),
                    "last_recorded": None
                    if row is None
                    else {
                        "hawl_date": row.hawl_date.isoformat(),
                        "hawl_hijri": row.hawl_hijri,
                        "amount": row.amount,
                        "chosen": row.chosen,
                    },
                }
            )
        return JSONResponse(
            {
                "configured": True,
                "source": z.SOURCE,
                "hawl_hijri": hawl_hijri,
                "last_hawl": last_hawl.isoformat(),
                "next_hawl": next_hawl.isoformat(),
                "next_hawl_hijri": z.hijri_label(next_hawl),
                "days_to_next": (next_hawl - today).days,
                "accounts": accounts,
            }
        )

    @app.get("/api/halal/purification")
    async def api_halal_purification(
        year: int | None = None, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        from halal_trader.compliance.purification import report

        y = year or date.today().year
        lines = [
            (account, line)
            for account in ("core", "paper")
            for line in await report(ctx.engine, account, y)
        ]
        return JSONResponse(
            {
                "year": y,
                "lines": [
                    {
                        "account": account,
                        "symbol": line.symbol,
                        "dividends": round(line.dividends, 2),
                        "amount": round(line.amount, 2),
                        "payments": line.payments,
                        "assumed": line.assumed,
                    }
                    for account, line in lines
                ],
                "dividends": round(sum(x.dividends for _, x in lines), 2),
                "amount": round(sum(x.amount for _, x in lines), 2),
            }
        )
