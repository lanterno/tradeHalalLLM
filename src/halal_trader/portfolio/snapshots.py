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
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.num import to_float

BENCHMARKS = ("SPUS", "HLAL", "SPY")


def position_row(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": str(p["symbol"]),
        "qty": to_float(p.get("qty")),
        "market_value": to_float(p.get("market_value")),
        "price": to_float(p.get("current_price")),
        "prev_close": to_float(p.get("lastday_price")),
        "change_today": to_float(p.get("change_today")),  # a fraction: 0.012 is +1.2%
        "unrealized_pl": to_float(p.get("unrealized_pl")),
    }


def quote_row(
    symbol: str, snap: dict[str, Any] | None, *, in_session: bool = True
) -> dict[str, Any] | None:
    """The price to show and the close it moved from. In the regular session
    that is the latest trade; outside it, the session's official close, so the
    change is the familiar close-to-close one and not moved by thin
    extended-hours trades."""
    from halal_trader.trading.bars import parse_snapshot

    s = parse_snapshot(snap or {}, symbol)
    if s is None:
        return None
    trade, close = s.last_trade, s.daily_close
    price = (trade or close) if in_session else (close or trade)
    if not price:
        return None
    return {"symbol": symbol, "price": price, "prev_close": s.prev_close}


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    account: str
    taken_at: datetime
    equity: float
    cash: float
    last_equity: float | None
    positions: list[dict[str, Any]]


async def read_snapshots(engine: AsyncEngine) -> dict[str, AccountSnapshot]:
    """Every account's latest snapshot, by account name."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT account, taken_at, equity, cash, last_equity, positions "
                "FROM account_snapshots"
            )
        )
        return {
            r.account: AccountSnapshot(
                r.account,
                r.taken_at,
                float(r.equity),
                float(r.cash),
                float(r.last_equity) if r.last_equity is not None else None,
                list(r.positions or []),
            )
            for r in rows
        }


async def read_snapshot(engine: AsyncEngine, account: str) -> AccountSnapshot | None:
    """One account's latest snapshot (None before the first)."""
    return (await read_snapshots(engine)).get(account)


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
                "e": to_float(raw.get("equity")) or 0.0,
                "c": to_float(raw.get("cash")) or 0.0,
                "l": to_float(raw.get("last_equity")),
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


async def benchmark_moves(engine: AsyncEngine, today: date) -> list[dict[str, Any]]:
    """Each benchmark's price and move: the bot's quote while it is today's
    (``live``), else the last two stored closes."""
    from halal_trader.market_hours import MARKET_TZ

    async with engine.connect() as conn:
        quotes = {r.symbol: r for r in await conn.execute(text("SELECT * FROM quotes"))}
        closes: dict[str, list[tuple[date, float]]] = {}
        for r in await conn.execute(
            text(
                "SELECT symbol, day, close FROM ("
                " SELECT symbol, day, close, row_number() OVER "
                " (PARTITION BY symbol ORDER BY day DESC) AS n FROM daily_bars "
                " WHERE adjustment = 'raw' AND symbol = ANY(:s)) x WHERE n <= 2"
            ),
            {"s": list(BENCHMARKS)},
        ):
            closes.setdefault(r.symbol, []).append((r.day, float(r.close)))
    out = []
    for symbol in BENCHMARKS:
        q = quotes.get(symbol)
        hist = sorted(closes.get(symbol, []))
        if q is not None and q.taken_at.astimezone(MARKET_TZ).date() == today:
            price, prev, as_of, live = float(q.price), q.prev_close, q.taken_at.isoformat(), True
        elif q is not None and hist and q.taken_at.astimezone(MARKET_TZ).date() > hist[-1][0]:
            price, prev, as_of, live = float(q.price), q.prev_close, q.taken_at.isoformat(), False
        elif hist:
            price, as_of, live = hist[-1][1], hist[-1][0].isoformat(), False
            prev = hist[-2][1] if len(hist) > 1 else None
        else:
            continue
        out.append(
            {
                "symbol": symbol,
                "price": round(price, 2),
                "change_pct": round(price / float(prev) - 1, 5) if prev else None,
                "as_of": as_of,
                "live": live,
            }
        )
    return out
