"""GET /api/core — the Core page: the strict-halal core portfolio's own account.

Read-only, from the database the core's jobs fill: holdings (rebuilt from
the account's fills, valued at the last close) against the forward book's
targets, the account's equity against the book's NAV, the live-money gate
and the recent orders and runs.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


def _f(x: Any, digits: int = 2) -> float | None:
    return None if x is None else round(float(x), digits)


def _fill(o: Any) -> dict[str, Any]:
    if o is None:
        return {"fill_price": None, "fill_status": None, "vs_arrival_bps": None}
    return {
        "fill_price": _f(o.fill_price, 4),
        "fill_status": o.status,
        "vs_arrival_bps": _f(o.vs_arrival_bps, 1),
    }


def register(app: FastAPI) -> None:
    @app.get("/api/core")
    async def api_core(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        from halal_trader.compliance.purification import paper_positions
        from halal_trader.config import get_settings
        from halal_trader.market_hours import today_eastern
        from halal_trader.portfolio import readiness as gate

        settings = get_settings()
        today = today_eastern()
        engine = ctx.engine
        async with engine.connect() as conn:
            equity_rows = (
                await conn.execute(
                    text(
                        "SELECT day, equity FROM broker_equity WHERE account = 'core' "
                        "AND equity > 0 ORDER BY day"
                    )
                )
            ).all()
            navs = {
                r.day: float(r.nav)
                for r in await conn.execute(
                    text("SELECT day, nav FROM forward_book_days WHERE book = 'core' ORDER BY day")
                )
            }
            book_weights = (
                await conn.execute(
                    text(
                        "SELECT weights FROM forward_book_days WHERE book = 'core' "
                        "ORDER BY day DESC LIMIT 1"
                    )
                )
            ).scalar()
            orders = (
                await conn.execute(
                    text(
                        "SELECT submitted_at, symbol, side, qty, est_price, notional, reason, "
                        "screen_as_of, status FROM core_orders ORDER BY submitted_at DESC "
                        "LIMIT 60"
                    )
                )
            ).all()
            runs = (
                await conn.execute(
                    text(
                        "SELECT run_on, monthly, executed, equity, cash, orders, halted, "
                        "screen_as_of FROM core_runs ORDER BY recorded_at DESC LIMIT 12"
                    )
                )
            ).all()
            shares = await paper_positions(engine, today + timedelta(days=1), "core")
            closes = {
                r.symbol: float(r.close)
                for r in await conn.execute(
                    text(
                        "SELECT DISTINCT ON (symbol) symbol, close FROM daily_bars "
                        "WHERE adjustment = 'raw' AND symbol = ANY(:s) "
                        "ORDER BY symbol, day DESC"
                    ),
                    {"s": sorted(shares)},
                )
            }

        equity = float(equity_rows[-1].equity) if equity_rows else None
        targets = dict(book_weights or {})
        holdings: list[dict[str, Any]] = []
        for symbol in sorted(set(shares) | set(targets)):
            value = shares.get(symbol, 0.0) * closes.get(symbol, 0.0)
            weight = value / equity if equity else None
            target = float(targets.get(symbol, 0.0))
            holdings.append(
                {
                    "symbol": symbol,
                    "shares": _f(shares.get(symbol), 6),
                    "value": _f(value),
                    "weight": _f(weight, 5),
                    "target": _f(target, 5),
                    "drift": _f(weight - target, 5) if weight is not None else None,
                }
            )
        holdings.sort(key=lambda h: -float(h["target"] or 0) - float(h["weight"] or 0))

        # Account and book, both rebased to 100 at the account's first day.
        series = []
        if equity_rows:
            first = equity_rows[0]
            base_nav = navs.get(first.day)
            for r in equity_rows:
                nav = navs.get(r.day)
                series.append(
                    {
                        "date": r.day.isoformat(),
                        "account": _f(100 * float(r.equity) / float(first.equity), 3),
                        "book": _f(100 * nav / base_nav, 3) if nav and base_nav else None,
                    }
                )

        from halal_trader.portfolio.execution_quality import report as fills

        executed = await fills(engine, today - timedelta(days=30), today)
        fill_of = {(o.submitted_at, o.symbol): o for o in executed.orders}
        ready = await gate.check(engine, today=today)
        return JSONResponse(
            {
                "enabled": settings.core.enabled,
                "paper": settings.core.paper,
                "equity": _f(equity),
                "equity_day": equity_rows[-1].day.isoformat() if equity_rows else None,
                "holdings": holdings,
                "series": series,
                "readiness": {
                    "ready": ready.ready,
                    "days": ready.days,
                    "min_days": gate.MIN_DAYS,
                    "monthly_runs": ready.monthly_runs,
                    "tracking_error": _f(ready.tracking_error, 5),
                    "max_tracking_error": gate.MAX_TRACKING_ERROR,
                    "gap": _f(ready.gap, 5),
                    "max_gap": gate.MAX_GAP,
                    "refused": ready.refused,
                    "halted": ready.halted,
                    "failures": ready.failures,
                },
                "execution": executed.summary() if executed.orders else None,
                "orders": [
                    {
                        "at": o.submitted_at.isoformat(),
                        "symbol": o.symbol,
                        "side": o.side,
                        "qty": _f(o.qty, 6),
                        "price": _f(o.est_price),
                        "notional": _f(o.notional),
                        "reason": o.reason,
                        "screen_as_of": o.screen_as_of.isoformat() if o.screen_as_of else None,
                        "status": o.status,
                        **_fill(fill_of.get((o.submitted_at, o.symbol))),
                    }
                    for o in orders
                ],
                "runs": [
                    {
                        "run_on": r.run_on.isoformat(),
                        "monthly": r.monthly,
                        "executed": r.executed,
                        "equity": _f(r.equity),
                        "cash": _f(r.cash),
                        "orders": r.orders,
                        "halted": r.halted,
                        "screen_as_of": r.screen_as_of.isoformat() if r.screen_as_of else None,
                    }
                    for r in runs
                ],
            }
        )
