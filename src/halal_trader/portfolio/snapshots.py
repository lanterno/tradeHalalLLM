"""Live account values and prices for the home page, taken by the bot.

The web process has no broker connection (the database is the only
contract between the processes), so the bot writes what the home page
shows: each account's equity, cash, previous close and positions, and the
benchmarks' prices. One row per account and per symbol, overwritten each
minute of the session; the daily history stays in broker_equity and
daily_bars.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

BENCHMARKS = ("SPUS", "HLAL", "SPY")


def _f(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except TypeError, ValueError:
        return None


def position_row(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": str(p["symbol"]),
        "qty": _f(p.get("qty")),
        "market_value": _f(p.get("market_value")),
        "price": _f(p.get("current_price")),
        "prev_close": _f(p.get("lastday_price")),
        "change_today": _f(p.get("change_today")),  # a fraction: 0.012 is +1.2%
        "unrealized_pl": _f(p.get("unrealized_pl")),
    }


def quote_row(
    symbol: str, snap: dict[str, Any] | None, *, in_session: bool = True
) -> dict[str, Any] | None:
    """The price to show and the close it moved from. In the regular session
    that is the latest trade; outside it, the session's official close, so the
    change is the familiar close-to-close one and not moved by thin
    extended-hours trades."""
    snap = snap or {}
    trade = _f((snap.get("latestTrade") or {}).get("p"))
    close = _f((snap.get("dailyBar") or {}).get("c"))
    price = (trade or close) if in_session else (close or trade)
    if not price:
        return None
    return {
        "symbol": symbol,
        "price": price,
        "prev_close": _f((snap.get("prevDailyBar") or {}).get("c")),
    }


async def snapshot_account(engine: AsyncEngine, account: str, broker: Any) -> None:
    raw = await broker.account_payload()
    positions = [position_row(p) for p in await broker.positions_payload()]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO account_snapshots (account, taken_at, equity, cash, last_equity, "
                "positions) VALUES (:a, :t, :e, :c, :l, CAST(:p AS JSONB)) "
                "ON CONFLICT (account) DO UPDATE SET taken_at = EXCLUDED.taken_at, "
                "equity = EXCLUDED.equity, cash = EXCLUDED.cash, "
                "last_equity = EXCLUDED.last_equity, positions = EXCLUDED.positions"
            ),
            {
                "a": account,
                "t": datetime.now(UTC),
                "e": _f(raw.get("equity")) or 0.0,
                "c": _f(raw.get("cash")) or 0.0,
                "l": _f(raw.get("last_equity")),
                "p": json.dumps(positions),
            },
        )


async def snapshot_quotes(
    engine: AsyncEngine, broker: Any, symbols: tuple[str, ...] = BENCHMARKS
) -> int:
    from halal_trader.market_hours import is_market_open_local

    payload = await broker.get_stock_snapshot(",".join(symbols)) or {}
    session = is_market_open_local()
    rows = [r for s in symbols if (r := quote_row(s, payload.get(s), in_session=session))]
    if rows:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO quotes (symbol, taken_at, price, prev_close) "
                    "VALUES (:symbol, :t, :price, :prev_close) ON CONFLICT (symbol) DO UPDATE "
                    "SET taken_at = EXCLUDED.taken_at, price = EXCLUDED.price, "
                    "prev_close = EXCLUDED.prev_close"
                ),
                [{**r, "t": datetime.now(UTC)} for r in rows],
            )
    return len(rows)
