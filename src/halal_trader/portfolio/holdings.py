"""What an account holds, marked to the market, and how concentrated it is.

The bot's minute snapshot (portfolio/snapshots.py) marks holdings with the
broker's own price, market value and unrealized P&L. Before an account's
first snapshot they are rebuilt from its ledger and marked at the last daily
close (``source: "ledger"``); never at the entry price, which showed a $2,963
gain as $0 and a holding under its own stop. The positions page, the core's
risk page and the home page all read holdings, and the screen's view of
them, here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.sectors import sector_of
from halal_trader.core.num import money, ratio, rounded
from halal_trader.data.store import last_closes
from halal_trader.halal import strict
from halal_trader.portfolio.core_account import DAY_TRADER


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
        "qty": rounded(qty, 6),
        "avg_entry": rounded(cost / qty, 4) if cost is not None and qty else None,
        "price": rounded(p.get("price"), 4),
        "market_value": money(value),
        "cost_basis": money(cost),
        "unrealized_pl": money(upl),
        "unrealized_pl_pct": ratio(float(upl) / cost) if cost and upl is not None else None,
        "change_today": ratio(p.get("change_today")),
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
    closes = await last_closes(engine, sorted(qty))
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
                "qty": rounded(q, 6),
                "avg_entry": rounded(basis / q, 4) if basis and q else None,
                "price": rounded(price, 4),
                "market_value": money(value),
                "cost_basis": money(basis),
                "unrealized_pl": money(upl),
                "unrealized_pl_pct": ratio(upl / basis) if upl is not None and basis else None,
                "change_today": None,
            }
        )
    days = [c[0] for c in closes.values()]
    return rows, (max(days).isoformat() if days else None)


async def screen_view(
    engine: AsyncEngine, symbols: Sequence[str]
) -> dict[str, tuple[str, str | None]]:
    """symbol -> (verdict, SIC description) on the newest screen, as the order
    boundary reads it; a symbol it does not hold is absent."""
    as_of = await strict.newest_screen(engine)
    if as_of is None:
        return {}
    return {
        r.symbol: (r.verdict, r.sic_description)
        for r in await strict.screen_rows(engine, as_of, symbols=symbols)
    }


def failing_screen(
    symbols: Iterable[str], screen: Mapping[str, tuple[str, str | None]]
) -> list[str]:
    """The holdings the newest screen does not pass (or does not hold)."""
    return [s for s in symbols if screen.get(s, ("", None))[0] != "halal"]


def sector_values(
    values: Mapping[str, float], screen: Mapping[str, tuple[str, str | None]]
) -> list[tuple[str, float]]:
    """Each sector's total value, largest first (sectors from the screen's SIC)."""
    sectors: dict[str, float] = {}
    for symbol, value in values.items():
        sector = sector_of(screen.get(symbol, ("", None))[1])
        sectors[sector] = sectors.get(sector, 0.0) + value
    return sorted(sectors.items(), key=lambda kv: -kv[1])
