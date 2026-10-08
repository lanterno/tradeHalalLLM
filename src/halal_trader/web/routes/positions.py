"""GET /api/positions — what each broker account holds, marked to the market.

Both accounts, the core first, then the day-trader. The marks come from the bot's
minute snapshot of each account (``account_snapshots``, written by
portfolio/snapshots.py): the broker's own price, market value and unrealized
P&L, the same figures the home page shows. The web process has no broker
connection, so ``as_of``/``age_seconds`` say how old those marks are.

Before an account's first snapshot the holdings are rebuilt from its ledger
and marked at the last daily close (``source: "ledger"``); never at the entry
price, which showed a $2,963 gain as $0 and a holding under its own stop.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.context import DashboardContext
from halal_trader.portfolio.core_account import DAY_TRADER, core_account
from halal_trader.web.dependencies import get_ctx


def _f(value: Any, digits: int = 2) -> float | None:
    try:
        return None if value is None else round(float(value), digits)
    except TypeError, ValueError:
        return None


def broker_position(p: dict[str, Any]) -> dict[str, Any]:
    """One snapshot position (portfolio/snapshots.position_row) with its cost.

    The snapshot carries the broker's market value and unrealized P&L, so the
    cost basis is their difference and the average entry that over the qty.
    """
    qty = float(p.get("qty") or 0.0)
    value = p.get("market_value")
    upl = p.get("unrealized_pl")
    cost = float(value) - float(upl) if value is not None and upl is not None else None
    return {
        "symbol": str(p["symbol"]),
        "qty": _f(qty, 6),
        "avg_entry": _f(cost / qty, 4) if cost is not None and qty else None,
        "price": _f(p.get("price"), 4),
        "market_value": _f(value),
        "cost_basis": _f(cost),
        "unrealized_pl": _f(upl),
        "unrealized_pl_pct": _f(float(upl) / cost, 5) if cost and upl is not None else None,
        "change_today": _f(p.get("change_today"), 5),
    }


async def _last_closes(engine: AsyncEngine, symbols: list[str]) -> dict[str, tuple[Any, float]]:
    if not symbols:
        return {}
    async with engine.connect() as conn:
        return {
            r.symbol: (r.day, float(r.close))
            for r in await conn.execute(
                text(
                    "SELECT DISTINCT ON (symbol) symbol, day, close FROM daily_bars "
                    "WHERE adjustment = 'raw' AND symbol = ANY(:s) ORDER BY symbol, day DESC"
                ),
                {"s": symbols},
            )
        }


async def ledger_positions(
    engine: AsyncEngine, account: str, open_trades: list[Any]
) -> tuple[list[dict[str, Any]], str | None]:
    """Holdings without a snapshot: the ledger's quantities at the last close."""
    from halal_trader.compliance.purification import paper_positions
    from halal_trader.market_hours import today_eastern

    if account == DAY_TRADER:
        qty: dict[str, float] = {}
        cost: dict[str, float] = {}
        for t in open_trades:
            q = float(t.filled_quantity or t.quantity or 0.0)
            qty[t.symbol] = qty.get(t.symbol, 0.0) + q
            cost[t.symbol] = cost.get(t.symbol, 0.0) + q * float(t.filled_price or t.price or 0.0)
    else:
        qty = await paper_positions(engine, today_eastern() + timedelta(days=1), account)
        cost = {}
    closes = await _last_closes(engine, sorted(qty))
    rows = []
    for symbol, q in qty.items():
        day_close = closes.get(symbol)
        price = day_close[1] if day_close else None
        value = q * price if price is not None else None
        basis = cost.get(symbol)
        upl = value - basis if value is not None and basis else None
        rows.append(
            {
                "symbol": symbol,
                "qty": _f(q, 6),
                "avg_entry": _f(basis / q, 4) if basis and q else None,
                "price": _f(price, 4),
                "market_value": _f(value),
                "cost_basis": _f(basis),
                "unrealized_pl": _f(upl),
                "unrealized_pl_pct": _f(upl / basis, 5) if upl is not None and basis else None,
                "change_today": None,
            }
        )
    days = [c[0] for c in closes.values()]
    return rows, (max(days).isoformat() if days else None)


def _ledger_levels(open_trades: list[Any]) -> dict[str, dict[str, Any]]:
    """The day-trader's own exit levels per symbol: the newest entry's stop and
    target, and when the position was first opened."""
    levels: dict[str, dict[str, Any]] = {}
    for t in sorted(open_trades, key=lambda t: t.timestamp):
        lv = levels.setdefault(t.symbol, {"opened_at": t.timestamp.isoformat()})
        lv["stop_loss"] = t.stop_loss
        lv["target_price"] = t.target_price
    return levels


def register(app: FastAPI) -> None:
    @app.get("/api/positions")
    async def api_positions(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        from halal_trader.core.heartbeat import core_running

        now = datetime.now(UTC)
        async with ctx.engine.connect() as conn:
            snaps = {
                r.account: r
                for r in await conn.execute(
                    text("SELECT account, taken_at, equity, cash, positions FROM account_snapshots")
                )
            }
        open_trades = await ctx.repo.get_open_trades()
        core_name = core_account(ctx.settings.core.paper)
        status = {
            core_name: "active" if await core_running(ctx.engine) else "disabled",
            DAY_TRADER: "active",
        }
        out = []
        for account, label in ((core_name, "Core portfolio"), (DAY_TRADER, "Day-trader")):
            snap = snaps.get(account)
            equity: float | None
            cash: float | None
            age: int | None
            if snap is not None:
                positions = [broker_position(p) for p in snap.positions or []]
                equity, cash = float(snap.equity), float(snap.cash)
                as_of, source = snap.taken_at.isoformat(), "snapshot"
                age = round((now - snap.taken_at).total_seconds())
            else:
                positions, as_of = await ledger_positions(
                    ctx.engine, account, open_trades if account == DAY_TRADER else []
                )
                equity = cash = None
                source, age = "ledger", None
                if not positions:
                    continue
            if account == DAY_TRADER:
                levels = _ledger_levels(open_trades)
                for p in positions:
                    p.update(levels.get(p["symbol"], {}))
            invested = sum(p["market_value"] or 0.0 for p in positions)
            for p in positions:
                denominator = equity or invested
                p["weight"] = (
                    _f((p["market_value"] or 0.0) / denominator, 5) if denominator else None
                )
            positions.sort(key=lambda p: -(p["market_value"] or 0.0))
            upl = [p["unrealized_pl"] for p in positions if p["unrealized_pl"] is not None]
            out.append(
                {
                    "account": account,
                    "label": label,
                    "status": status[account],
                    "source": source,
                    "as_of": as_of,
                    "age_seconds": age,
                    "equity": _f(equity),
                    "cash": _f(cash),
                    "invested": _f(invested),
                    "unrealized_pl": _f(sum(upl)) if upl else None,
                    "positions": positions,
                }
            )
        return JSONResponse({"accounts": out})
