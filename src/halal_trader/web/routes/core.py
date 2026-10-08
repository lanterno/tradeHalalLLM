"""GET /api/core — the Core page: the strict-halal core portfolio's own account.

Read-only, from the database the core's jobs fill: holdings against the
forward book's targets, the account's equity against the book's NAV, the
live-money gate and the recent orders and runs, all for the core's account
in the configured environment ("core" on paper, "core-live" live).

Current equity and holdings come from the bot's newest account snapshot
(``account_snapshots``: the broker's own positions and values) when there
is one, else from the ledger: the last daily equity, and shares rebuilt
from fills valued at the last close. The daily series is the ledger's.
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
        from halal_trader.core.heartbeat import core_running
        from halal_trader.market_hours import today_eastern
        from halal_trader.portfolio import readiness as gate
        from halal_trader.portfolio.core_account import core_account

        settings = get_settings()
        account = core_account(settings.core.paper)
        today = today_eastern()
        engine = ctx.engine
        async with engine.connect() as conn:
            equity_rows = (
                await conn.execute(
                    text(
                        "SELECT day, equity FROM broker_equity WHERE account = :a "
                        "AND equity > 0 ORDER BY day"
                    ),
                    {"a": account},
                )
            ).all()
            snap = (
                await conn.execute(
                    text(
                        "SELECT taken_at, equity, positions FROM account_snapshots "
                        "WHERE account = :a"
                    ),
                    {"a": account},
                )
            ).first()
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
                        "screen_as_of, screen_method, status FROM core_orders WHERE account = :a "
                        "ORDER BY submitted_at DESC LIMIT 60"
                    ),
                    {"a": account},
                )
            ).all()
            runs = (
                await conn.execute(
                    text(
                        "SELECT run_on, monthly, executed, equity, cash, orders, halted, "
                        "screen_as_of, notes FROM core_runs WHERE account = :a "
                        "ORDER BY recorded_at DESC LIMIT 12"
                    ),
                    {"a": account},
                )
            ).all()
            shares = await paper_positions(engine, today + timedelta(days=1), account)
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
        equity_day = equity_rows[-1].day.isoformat() if equity_rows else None
        values = {s: q * closes.get(s, 0.0) for s, q in shares.items()}
        source = "ledger"
        if snap is not None and snap.equity > 0:
            # The broker's own figures, minutes old in the session: newer than
            # the ledger's last close, and they count what fills cannot (splits).
            equity, equity_day, source = float(snap.equity), snap.taken_at.isoformat(), "live"
            shares, values = {}, {}
            for p in snap.positions or []:
                if p.get("qty"):
                    shares[p["symbol"]] = float(p["qty"])
                    values[p["symbol"]] = float(p.get("market_value") or 0.0)
        targets = dict(book_weights or {})
        holdings: list[dict[str, Any]] = []
        for symbol in sorted(set(shares) | set(targets)):
            value = values.get(symbol, 0.0)
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

        executed = await fills(engine, today - timedelta(days=30), today, account=account)
        fill_of = {(o.submitted_at, o.symbol): o for o in executed.orders}
        ready = await gate.check(engine, today=today)  # the paper rehearsal's gate
        return JSONResponse(
            {
                "enabled": await core_running(engine),
                "paper": settings.core.paper,
                "account": account,
                "equity": _f(equity),
                "equity_day": equity_day,
                "equity_source": source,
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
                    "unfilled": ready.unfilled,
                    "partial": ready.partial,
                    "missing_runs": [d.isoformat() for d in ready.missing_runs],
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
                        "screen_method": o.screen_method,
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
                        "notes": list(r.notes or []),
                    }
                    for r in runs
                ],
            }
        )
