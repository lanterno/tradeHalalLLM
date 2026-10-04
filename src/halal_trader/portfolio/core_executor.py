"""Trading the strict-halal core portfolio on its own Alpaca account.

The rules are portfolio/strict_core.py's, applied to the live account:

* **monthly** (the first run of a calendar month): the banded rebalance to
  cap weights of the largest names the screen passes;
* **every other run**: only forced sales, of holdings the screen no longer
  passes.

Safety, in order of precedence:

* **Halal at the order boundary.** Every buy is re-checked against the
  strict in-house screen at the moment of the order; a symbol it does not
  hold halal is refused. The screen must be fresh (``MAX_SCREEN_AGE``): a
  stale or missing screen means no orders at all, not orders on old data.
* **Cash only, long only.** Sells go first; buys are scaled to the cash on
  hand plus the sells' proceeds, less a buffer. Nothing is ever sold beyond
  the quantity held.
* **Small trades are skipped** (under ``MIN_TRADE``), so bands do their job.

Orders are fractional market orders, placed late in the session. Each one,
filled or refused, is recorded in ``core_orders`` with the screen verdict it
relied on: the receipt for every fill.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.portfolio.strict_core import (
    TOP_N,
    rebalance,
    split_share_classes,
    targets,
)

logger = logging.getLogger(__name__)

MAX_SCREEN_AGE = timedelta(days=10)
MIN_TRADE = 5.0  # dollars
CASH_BUFFER = 0.01  # of equity, left uninvested


@dataclass(frozen=True, slots=True)
class PlannedOrder:
    symbol: str
    side: str  # buy | sell
    qty: float
    price: float
    reason: str  # rebalance | forced sale (screen) | exit (left the target)

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass
class Plan:
    monthly: bool
    screen_as_of: date | None
    equity: float
    cash: float
    orders: list[PlannedOrder] = field(default_factory=list)
    halted: str | None = None  # why nothing will be traded, if so
    notes: list[str] = field(default_factory=list)


async def _screen(
    engine: AsyncEngine, day: date
) -> tuple[date | None, dict[str, tuple[float, float, int | None]]]:
    """(screen date, symbol -> (screen price, shares, CIK)) for the newest screen's halal names."""
    async with engine.connect() as conn:
        as_of = (
            await conn.execute(
                text("SELECT max(as_of) FROM halal_screen_results WHERE as_of <= :d"), {"d": day}
            )
        ).scalar()
        if as_of is None:
            return None, {}
        rows = await conn.execute(
            text(
                "SELECT symbol, cik, (metrics->>'price')::float AS p, "
                "(metrics->>'shares_outstanding')::float AS sh FROM halal_screen_results "
                "WHERE as_of = :a AND verdict = 'halal'"
            ),
            {"a": as_of},
        )
        return as_of, {r.symbol: (r.p, r.sh, r.cik) for r in rows if r.p and r.sh}


async def is_halal_now(engine: AsyncEngine, symbol: str, day: date) -> tuple[bool, date | None]:
    """The order-boundary check: does the newest fresh screen hold ``symbol`` halal?"""
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT as_of, verdict FROM halal_screen_results WHERE symbol = :s "
                    "AND as_of = (SELECT max(as_of) FROM halal_screen_results WHERE as_of <= :d)"
                ),
                {"s": symbol, "d": day},
            )
        ).first()
    if row is None or day - row.as_of > MAX_SCREEN_AGE:
        return False, row.as_of if row else None
    return row.verdict == "halal", row.as_of


def _prices(snapshot: Any) -> dict[str, float]:
    out = {}
    for symbol, snap in (snapshot or {}).items():
        trade = (snap or {}).get("latestTrade") or {}
        bar = (snap or {}).get("dailyBar") or {}
        price = trade.get("p") or bar.get("c")
        if price:
            out[symbol] = float(price)
    return out


async def plan(
    engine: AsyncEngine, broker: Any, *, today: date, monthly: bool, top_n: int = TOP_N
) -> Plan:
    from halal_trader.data.store import BENCHMARKS
    from halal_trader.data.universe import universe_at

    account = await broker.get_account_info()
    positions = {p.symbol: p for p in await broker.get_all_positions() if p.qty > 0}
    equity = account.effective_equity
    result = Plan(monthly, None, equity, account.cash)
    if equity <= 0:
        result.halted = "account has no equity"
        return result

    screened, screen = await _screen(engine, today)
    result.screen_as_of = screened
    if screened is None or today - screened > MAX_SCREEN_AGE:
        result.halted = f"halal screen missing or stale ({screened}): no orders on old data"
        return result
    members = set(await universe_at(engine, today, top_n=1000))
    eligible = set(screen) - set(BENCHMARKS)
    if members:
        eligible &= members

    current = {s: p.qty * p.current_price / equity for s, p in positions.items()}
    if monthly:
        candidates = sorted(eligible)
        prices = (
            _prices(await broker.get_stock_snapshot(",".join(candidates))) if candidates else {}
        )
        caps = {
            s: screen[s][1] * prices[s]  # shares (as screened) x the price now
            for s in candidates
            if s in prices
        }
        missing = len(candidates) - len(caps)
        if missing:
            result.notes.append(f"{missing} eligible name(s) without a live price, left out")
        goal = targets(split_share_classes(caps, {s: screen[s][2] for s in caps}), top_n)
    else:
        prices = {}
        goal = {s: w for s, w in current.items() if s in eligible}
    new = rebalance(current, goal, eligible)

    price_of = {**{s: p.current_price for s, p in positions.items()}, **prices}
    sells, buys = [], []
    for symbol in sorted(set(current) | set(new)):
        before, after = current.get(symbol, 0.0), new.get(symbol, 0.0)
        delta = (after - before) * equity
        price = price_of.get(symbol)
        if not price or abs(delta) < MIN_TRADE and after > 0:
            continue
        if after == 0 and symbol in positions:  # a full exit sells exactly what is held
            reason = "forced sale (screen)" if symbol not in eligible else "exit (left the target)"
            sells.append(PlannedOrder(symbol, "sell", positions[symbol].qty, price, reason))
        elif delta < 0:
            qty = min(-delta / price, positions[symbol].qty if symbol in positions else 0.0)
            if qty > 0:
                sells.append(PlannedOrder(symbol, "sell", qty, price, "rebalance"))
        elif delta > 0:
            buys.append(PlannedOrder(symbol, "buy", delta / price, price, "rebalance"))

    budget = account.cash + sum(o.notional for o in sells) - CASH_BUFFER * equity
    wanted = sum(o.notional for o in buys)
    if wanted > budget > 0:
        scale = budget / wanted
        buys = [PlannedOrder(o.symbol, o.side, o.qty * scale, o.price, o.reason) for o in buys]
        result.notes.append(f"buys scaled to {scale:.0%} to stay within cash")
    elif budget <= 0:
        result.notes.append("no cash for buys")
        buys = []
    result.orders = sells + [o for o in buys if o.notional >= MIN_TRADE]
    return result


async def execute(
    engine: AsyncEngine, broker: Any, p: Plan, *, today: date
) -> list[dict[str, Any]]:
    """Place the plan's orders, sells first, re-checking every buy at the order boundary."""
    results: list[dict[str, Any]] = []
    if p.halted:
        return results
    for order in p.orders:
        verdict_date = p.screen_as_of
        if order.side == "buy":
            ok, verdict_date = await is_halal_now(engine, order.symbol, today)
            if not ok:
                results.append(
                    await _record(engine, order, None, "refused: not halal now", verdict_date)
                )
                continue
        qty = round(order.qty, 6)
        response = await broker.place_order(order.symbol, order.side, qty)
        status = "refused" if isinstance(response, dict) and "error" in response else "submitted"
        results.append(await _record(engine, order, response, status, verdict_date))
    return results


async def _record(
    engine: AsyncEngine, order: PlannedOrder, response: Any, status: str, screen_as_of: date | None
) -> dict[str, Any]:
    import json

    broker_id = response.get("id") if isinstance(response, dict) else None
    row = {
        "at": datetime.now(UTC),
        "s": order.symbol,
        "side": order.side,
        "q": order.qty,
        "p": order.price,
        "n": order.notional,
        "r": order.reason,
        "scr": screen_as_of,
        "st": status,
        "bid": broker_id,
        "resp": json.dumps(response, default=str) if response is not None else None,
    }
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO core_orders (submitted_at, symbol, side, qty, est_price, notional, "
                "reason, screen_as_of, status, broker_order_id, response) VALUES (:at, :s, :side, "
                ":q, :p, :n, :r, :scr, :st, :bid, CAST(:resp AS JSONB))"
            ),
            row,
        )
    logger.info(
        "core order %s %s %.6f %s: %s", order.side, order.symbol, order.qty, order.reason, status
    )
    return {k: row[k] for k in ("s", "side", "q", "n", "r", "st")}


async def monthly_due(engine: AsyncEngine, today: date) -> bool:
    """True when no monthly rebalance has been recorded this calendar month."""
    async with engine.connect() as conn:
        n = (
            await conn.execute(
                text("SELECT count(*) FROM core_runs WHERE monthly AND run_on >= :m AND executed"),
                {"m": today.replace(day=1)},
            )
        ).scalar()
    return not n


async def record_run(engine: AsyncEngine, p: Plan, *, today: date, executed: bool) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO core_runs (run_on, monthly, executed, equity, cash, orders, halted, "
                "screen_as_of, recorded_at) VALUES (:d, :m, :e, :eq, :c, :n, :h, :s, now())"
            ),
            {
                "d": today,
                "m": p.monthly,
                "e": executed,
                "eq": p.equity,
                "c": p.cash,
                "n": len(p.orders),
                "h": p.halted,
                "s": p.screen_as_of,
            },
        )
