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

from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.core.num import money, ratio
from halal_trader.portfolio.core_account import DAY_TRADER, core_account
from halal_trader.portfolio.holdings import broker_position, ledger_positions
from halal_trader.web.dependencies import get_ctx


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
        from halal_trader.portfolio.snapshots import read_snapshots

        snaps = await read_snapshots(ctx.engine)
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
            as_of: str | None
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
                    ratio((p["market_value"] or 0.0) / denominator) if denominator else None
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
                    "equity": money(equity),
                    "cash": money(cash),
                    "invested": money(invested),
                    "unrealized_pl": money(sum(upl)) if upl else None,
                    "positions": positions,
                }
            )
        return JSONResponse({"accounts": out})
