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
from halal_trader.core.num import money, ratio, rounded
from halal_trader.web.dependencies import get_ctx


def _fill(o: Any) -> dict[str, Any]:
    if o is None:
        return {"fill_price": None, "fill_status": None, "vs_arrival_bps": None}
    return {
        "fill_price": rounded(o.fill_price, 4),
        "fill_status": o.status,
        "vs_arrival_bps": rounded(o.vs_arrival_bps, 1),
    }


def register(app: FastAPI) -> None:
    @app.get("/api/core")
    async def api_core(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        from halal_trader.compliance.purification import paper_positions
        from halal_trader.core.heartbeat import core_running
        from halal_trader.data.store import last_closes
        from halal_trader.execution.ledger import equity_history
        from halal_trader.market_hours import today_eastern
        from halal_trader.portfolio import readiness as gate
        from halal_trader.portfolio.core_account import core_account
        from halal_trader.portfolio.snapshots import read_snapshot
        from halal_trader.research.forward_book import latest_weights, nav_series

        settings = ctx.settings
        account = core_account(settings.core.paper)
        today = today_eastern()
        engine = ctx.engine
        equity_rows = await equity_history(engine, account)
        navs = dict(await nav_series(engine, "core"))
        targets = await latest_weights(engine, "core")
        snap = await read_snapshot(engine, account)
        async with engine.connect() as conn:
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
            closes = {s: c for s, (_, c) in (await last_closes(engine, sorted(shares))).items()}

        equity = equity_rows[-1][1] if equity_rows else None
        equity_day = equity_rows[-1][0].isoformat() if equity_rows else None
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
        holdings: list[dict[str, Any]] = []
        for symbol in sorted(set(shares) | set(targets)):
            value = values.get(symbol, 0.0)
            weight = value / equity if equity else None
            target = float(targets.get(symbol, 0.0))
            holdings.append(
                {
                    "symbol": symbol,
                    "shares": rounded(shares.get(symbol), 6),
                    "value": money(value),
                    "weight": ratio(weight),
                    "target": ratio(target),
                    "drift": ratio(weight - target) if weight is not None else None,
                }
            )
        holdings.sort(key=lambda h: -float(h["target"] or 0) - float(h["weight"] or 0))

        # Account and book, both rebased to 100 at the account's first day.
        series = []
        if equity_rows:
            first_day, first_equity = equity_rows[0]
            base_nav = navs.get(first_day)
            for day, equity_then in equity_rows:
                nav = navs.get(day)
                series.append(
                    {
                        "date": day.isoformat(),
                        "account": rounded(100 * equity_then / first_equity, 3),
                        "book": rounded(100 * nav / base_nav, 3) if nav and base_nav else None,
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
                "equity": money(equity),
                "equity_day": equity_day,
                "equity_source": source,
                "holdings": holdings,
                "series": series,
                "readiness": {
                    "ready": ready.ready,
                    "days": ready.days,
                    "min_days": gate.MIN_DAYS,
                    "monthly_runs": ready.monthly_runs,
                    "tracking_error": ratio(ready.tracking_error),
                    "max_tracking_error": gate.MAX_TRACKING_ERROR,
                    "gap": ratio(ready.gap),
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
                        "qty": rounded(o.qty, 6),
                        "price": money(o.est_price),
                        "notional": money(o.notional),
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
                        "equity": money(r.equity),
                        "cash": money(r.cash),
                        "orders": r.orders,
                        "halted": r.halted,
                        "screen_as_of": r.screen_as_of.isoformat() if r.screen_as_of else None,
                        "notes": list(r.notes or []),
                    }
                    for r in runs
                ],
            }
        )
