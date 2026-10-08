"""Trading the strict-halal core portfolio on its own Alpaca account.

The rules are portfolio/strict_core.py's, applied to the live account:

* **monthly** (the first run of a calendar month that actually trades): the
  banded rebalance to cap weights of the largest names the screen passes;
* **every other run**: sells only, of holdings the screen no longer holds
  halal. No buys: renormalising what is left would buy whenever cash drifts,
  which is a rebalance by another name.

Safety, in order of precedence:

* **Halal at the order boundary.** Every buy is re-checked against the
  strict in-house screen at the moment of the order; a symbol it does not
  hold halal is refused. The screen must be fresh (``MAX_SCREEN_AGE``): a
  stale or missing screen means no orders at all, not orders on old data.
* **One run's orders at a time.** Nothing is planned while any order is still
  open at the broker, and every order carries a deterministic
  ``client_order_id`` (``core-<date>-<symbol>-<side>``): Alpaca refuses an id
  it has seen, so a repeated run cannot place the same buys twice.
* **Cash only, long only.** Sells go first, and buys wait for them to finish
  (``SELL_FILL_TIMEOUT_S``). Buys are then sized to what the account holds
  after the sells: ``min(cash, buying_power)`` (a margin account's buying
  power is never borrowed against; any debit would be riba), less open buy
  orders, a buffer, and any purification still owed. A sell that has not
  filled by then contributes nothing, and the run says so. Nothing is ever
  sold beyond the quantity held.
* **Small trades are skipped** (under :func:`min_trade`), so bands do their job.
* **Live money** has a ceiling on what the core holds invested
  (``CORE_LIVE_MAX_NOTIONAL``); its other gates are core/safeguards.py's.

Orders are fractional market orders, placed late in the session. Each one,
filled or refused, is recorded in ``core_orders`` with the screen verdict it
relied on; each run in ``core_runs`` with its notes. Both are keyed by the
account name (portfolio/core_account.py), so paper and live never mix.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance import purification
from halal_trader.data.store import last_closes
from halal_trader.halal import strict
from halal_trader.portfolio.core_account import CORE_PAPER, core_account
from halal_trader.portfolio.strict_core import (
    TOP_N,
    rebalance,
    split_share_classes,
    targets,
)

logger = logging.getLogger(__name__)

MAX_SCREEN_AGE = strict.MAX_SCREEN_AGE
# The smallest trade worth placing: Alpaca's fractional minimum ($1), or a
# quarter of the band floor (strict_core.BAND_FLOOR = 0.2% of equity),
# whichever is larger. $52 on the paper account's $105k, $1 on a $1k live stage:
# a fixed $5 was noise at the first and a fifth of a position at the second.
MIN_TRADE_FLOOR = 1.0
MIN_TRADE_FRACTION = 0.0005
CASH_BUFFER = 0.01  # of equity, left uninvested
SELL_FILL_TIMEOUT_S = 120.0
SELL_POLL_S = 2.0
# Order states after which nothing more fills today.
TERMINAL = frozenset({"filled", "canceled", "expired", "rejected", "done_for_day", "replaced"})

REBALANCE = "rebalance"
SCREEN_SALE = "forced sale (screen)"  # the screen no longer holds it halal
UNIVERSE_EXIT = "exit (left the universe)"  # halal, but no longer a universe member
TARGET_EXIT = "exit (left the target)"  # halal and in the universe, not in the top N


def min_trade(equity: float) -> float:
    return max(MIN_TRADE_FLOOR, MIN_TRADE_FRACTION * equity)


def client_order_id(today: date, symbol: str, side: str) -> str:
    return f"core-{today:%Y%m%d}-{symbol}-{side}"


@dataclass(frozen=True, slots=True)
class PlannedOrder:
    symbol: str
    side: str  # buy | sell
    qty: float
    price: float
    reason: str  # REBALANCE | SCREEN_SALE | UNIVERSE_EXIT | TARGET_EXIT

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
    account: str = CORE_PAPER
    live_ceiling: float | None = None  # dollars invested at most (live only)


async def _screen(
    engine: AsyncEngine, day: date
) -> tuple[date | None, dict[str, tuple[float, float, int | None]], set[str]]:
    """(screen date, symbol -> (screen price, shares, CIK) for the halal names
    that can be cap-weighted, every symbol the newest screen holds halal)."""
    as_of = await strict.newest_screen(engine, on_or_before=day)
    if as_of is None:
        return None, {}, set()
    rows = await strict.screen_rows(engine, as_of, halal_only=True)
    return (
        as_of,
        {r.symbol: (r.price, r.shares, r.cik) for r in rows if r.price and r.shares},
        {r.symbol for r in rows},
    )


async def is_halal_now(engine: AsyncEngine, symbol: str, day: date) -> tuple[bool, date | None]:
    """The order-boundary check: does the newest fresh screen hold ``symbol`` halal?
    The day-trader's own gate (halal/strict.py), so both strategies apply one rule."""
    v = await strict.verdict(engine, symbol, today=day)
    return v.halal, v.screen_as_of


def _prices(snapshot: Any) -> dict[str, float]:
    out = {}
    for symbol, snap in (snapshot or {}).items():
        trade = (snap or {}).get("latestTrade") or {}
        bar = (snap or {}).get("dailyBar") or {}
        price = trade.get("p") or bar.get("c")
        if price:
            out[symbol] = float(price)
    return out


def _open_buy_notional(orders: list[dict[str, Any]]) -> float:
    """What open buy orders may still spend: their notional, else the unfilled
    quantity at the limit price. An open buy whose size cannot be told (a
    market order by quantity) counts as unbounded, so no buy goes out beside it."""
    total = 0.0
    for o in orders:
        if o.get("side") != "buy":
            continue
        if o.get("notional"):
            total += float(o["notional"])
            continue
        remaining = float(o.get("qty") or 0) - float(o.get("filled_qty") or 0)
        if remaining <= 0:
            continue
        if not o.get("limit_price"):
            return math.inf
        total += remaining * float(o["limit_price"])
    return total


def _floor6(qty: float) -> float:
    """Down to Alpaca's six decimals: rounding up could sell a sliver more than held."""
    return math.floor(qty * 1e6 + 1e-4) / 1e6  # the 1e-4: binary noise, e.g. 0.3 * 1e6


def _budget(
    *,
    cash: float,
    buying_power: float,
    proceeds: float,
    open_buys: float,
    equity: float,
    reserved: float,
    ceiling: float | None,
    invested: float,
    notes: list[str],
) -> float:
    """Dollars the buys may spend: a cash account's figure on any account.

    ``min(cash, buying_power)``: on a margin account (the paper account is
    4x) buying power is borrowable money, and a margin debit is riba.
    """
    budget = min(cash, buying_power) + proceeds - open_buys - CASH_BUFFER * equity - reserved
    if ceiling is not None:
        room = ceiling - invested
        if room < budget:
            notes.append(
                f"live ceiling ${ceiling:,.2f}: ${max(invested, 0):,.2f} invested, "
                f"buys capped at ${max(room, 0):,.2f}"
            )
            budget = room
    return budget


def _scaled(
    buys: list[PlannedOrder], budget: float, equity: float, notes: list[str]
) -> list[PlannedOrder]:
    wanted = sum(o.notional for o in buys)
    if not buys:
        return []
    if budget <= 0:
        notes.append("no cash for buys")
        return []
    if wanted > budget:
        scale = budget / wanted
        buys = [PlannedOrder(o.symbol, o.side, o.qty * scale, o.price, o.reason) for o in buys]
        notes.append(f"buys scaled to {scale:.0%} to stay within cash")
    floor = min_trade(equity)
    kept = [o for o in buys if o.notional >= floor]
    if len(kept) < len(buys):
        notes.append(f"{len(buys) - len(kept)} buy(s) under ${floor:,.2f} skipped")
    return kept


async def plan(
    engine: AsyncEngine,
    broker: Any,
    *,
    today: date,
    monthly: bool,
    top_n: int = TOP_N,
    account: str = CORE_PAPER,
    live_ceiling: float | None = None,
) -> Plan:
    """The orders the rule wants now. Reads only: nothing is placed or recorded."""
    from halal_trader.data.store import BENCHMARKS
    from halal_trader.data.universe import universe_at

    acct = await broker.get_account_info()
    positions = {p.symbol: p for p in await broker.get_all_positions() if p.qty > 0}
    equity = acct.effective_equity
    result = Plan(monthly, None, equity, acct.cash, account=account, live_ceiling=live_ceiling)
    if equity <= 0:
        result.halted = "account has no equity"
        return result
    open_orders = await broker.get_open_orders()
    if open_orders:
        # A run's orders still working: planning again would count their
        # cash twice and repeat their buys (one run placed $104k twice).
        result.halted = (
            f"{len(open_orders)} order(s) still open at the broker: wait for them to fill "
            "or cancel them before the core trades again"
        )
        return result

    screened, screen, halal = await _screen(engine, today)
    result.screen_as_of = screened
    if screened is None or today - screened > MAX_SCREEN_AGE:
        result.halted = f"halal screen missing or stale ({screened}): no orders on old data"
        return result
    members = set(await universe_at(engine, today, top_n=1000))
    eligible = set(screen) - set(BENCHMARKS)
    if members:
        eligible &= members

    current = {s: p.qty * p.current_price / equity for s, p in positions.items()}
    sells: list[PlannedOrder] = []
    buys: list[PlannedOrder] = []
    if not monthly:
        # Sells only, and only what the screen no longer holds halal. A
        # universe or target exit waits for the month's rebalance.
        for symbol, position in sorted(positions.items()):
            if symbol not in halal:
                sells.append(
                    PlannedOrder(symbol, "sell", position.qty, position.current_price, SCREEN_SALE)
                )
        result.orders = sells
        return result

    candidates = sorted(eligible)
    prices = _prices(await broker.get_stock_snapshot(",".join(candidates))) if candidates else {}
    # A holding without a live (IEX) trade is still a holding: it is priced
    # from the position, else the last close, rather than dropped from the
    # targets and sold as if it had left them.
    unpriced_held = [s for s in candidates if s not in prices and s in positions]
    if unpriced_held:
        closes = {
            s: c
            for s, (_, c) in (await last_closes(engine, unpriced_held, on_or_before=today)).items()
        }
        for s in unpriced_held:
            fallback = positions[s].current_price or closes.get(s)
            if fallback:
                prices[s] = float(fallback)
        result.notes.append(
            f"{len(unpriced_held)} holding(s) without a live price, priced from the position "
            f"or the last close: {', '.join(unpriced_held[:10])}"
        )
    caps = {s: screen[s][1] * prices[s] for s in candidates if s in prices}
    missing = sorted(set(candidates) - set(caps))
    if missing:
        result.notes.append(
            f"{len(missing)} eligible name(s) without a price, left out: {', '.join(missing[:10])}"
        )
    goal = targets(split_share_classes(caps, {s: screen[s][2] for s in caps}), top_n)
    new = rebalance(current, goal, eligible)

    floor = min_trade(equity)
    price_of = {**{s: p.current_price for s, p in positions.items()}, **prices}
    for symbol in sorted(set(current) | set(new)):
        before, after = current.get(symbol, 0.0), new.get(symbol, 0.0)
        delta = (after - before) * equity
        price = price_of.get(symbol)
        if not price or abs(delta) < floor and after > 0:
            continue
        if after == 0 and symbol in positions:  # a full exit sells exactly what is held
            if symbol not in halal:
                reason = SCREEN_SALE
            elif symbol not in eligible:
                reason = UNIVERSE_EXIT
            else:
                reason = TARGET_EXIT
            sells.append(PlannedOrder(symbol, "sell", positions[symbol].qty, price, reason))
        elif delta < 0:
            qty = min(-delta / price, positions[symbol].qty if symbol in positions else 0.0)
            if qty > 0:
                sells.append(PlannedOrder(symbol, "sell", qty, price, REBALANCE))
        elif delta > 0:
            buys.append(PlannedOrder(symbol, "buy", delta / price, price, REBALANCE))

    # Purification owed by this account is not the portfolio's to invest: it
    # stays as cash until it is given away (purify paid).
    reserved = await purification.unpaid(engine, [account])
    if reserved > 0:
        result.notes.append(f"${reserved:,.2f} held back for purification")
    proceeds = sum(o.notional for o in sells)
    invested = sum(p.qty * p.current_price for p in positions.values()) - proceeds
    budget = _budget(
        cash=acct.cash,
        buying_power=acct.buying_power,
        proceeds=proceeds,
        open_buys=0.0,
        equity=equity,
        reserved=reserved,
        ceiling=live_ceiling,
        invested=invested,
        notes=result.notes,
    )
    result.orders = sells + _scaled(buys, budget, equity, result.notes)
    return result


async def _await_sells(
    broker: Any,
    ids: dict[str, str],
    *,
    timeout: float,
    poll: float,
    sleep: Callable[[float], Awaitable[Any]],
) -> list[str]:
    """Poll the placed sells until each is final or ``timeout`` passes.

    ``ids``: broker order id -> symbol. Returns the symbols not fully filled.
    """
    pending = dict(ids)
    status: dict[str, str] = {}
    for attempt in range(max(1, math.ceil(timeout / poll)) + 1):
        for oid in list(pending):
            try:
                order = await broker.get_order_by_id(oid)
            except Exception as exc:  # noqa: BLE001 -- a failed poll is "not known final"
                logger.warning("core: polling sell %s failed: %r", oid, exc)
                continue
            state = str(order.get("status") or "")
            if state in TERMINAL:
                status[oid] = state
                pending.pop(oid)
        if not pending or attempt * poll >= timeout:
            break
        await sleep(poll)
    return sorted(sym for oid, sym in ids.items() if oid in pending or status.get(oid) != "filled")


async def execute(
    engine: AsyncEngine,
    broker: Any,
    p: Plan,
    *,
    today: date,
    sell_timeout: float = SELL_FILL_TIMEOUT_S,
    poll: float = SELL_POLL_S,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> list[dict[str, Any]]:
    """Place the plan's orders: sells, then (once they are final) buys within cash.

    Every buy is re-checked at the order boundary, and re-sized to the cash
    the account actually holds after its sells, never more than planned.
    """
    results: list[dict[str, Any]] = []
    if p.halted:
        return results
    sells = [o for o in p.orders if o.side == "sell"]
    buys = [o for o in p.orders if o.side == "buy"]

    placed: dict[str, str] = {}
    for order in sells:
        row, broker_id = await _place(engine, broker, order, p, today, p.screen_as_of)
        results.append(row)
        if broker_id:
            placed[broker_id] = order.symbol
    if not buys:
        return results

    if placed:
        unfilled = await _await_sells(broker, placed, timeout=sell_timeout, poll=poll, sleep=sleep)
        if unfilled:
            p.notes.append(
                f"{len(unfilled)} sell(s) not filled after {sell_timeout:.0f}s "
                f"({', '.join(unfilled[:10])}): buys use only the cash already held"
            )
    acct = await broker.get_account_info()
    positions = await broker.get_all_positions()
    invested = sum(x.qty * x.current_price for x in positions if x.qty > 0)
    budget = _budget(
        cash=acct.cash,
        buying_power=acct.buying_power,
        proceeds=0.0,
        open_buys=_open_buy_notional(await broker.get_open_orders()),
        equity=acct.effective_equity,
        reserved=await purification.unpaid(engine, [p.account]),
        ceiling=p.live_ceiling,
        invested=invested,
        notes=p.notes,
    )
    for order in _scaled(buys, budget, acct.effective_equity, p.notes):
        ok, verdict_date = await is_halal_now(engine, order.symbol, today)
        if not ok:
            results.append(
                await _record(
                    engine, p.account, order, None, "refused: not halal now", verdict_date
                )
            )
            continue
        row, _ = await _place(engine, broker, order, p, today, verdict_date)
        results.append(row)
    return results


async def _place(
    engine: AsyncEngine,
    broker: Any,
    order: PlannedOrder,
    p: Plan,
    today: date,
    verdict_date: date | None,
) -> tuple[dict[str, Any], str | None]:
    response = await broker.place_order(
        order.symbol,
        order.side,
        _floor6(order.qty),
        client_order_id=client_order_id(today, order.symbol, order.side),
    )
    refused = not isinstance(response, dict) or "error" in response or not response.get("id")
    row = await _record(
        engine, p.account, order, response, "refused" if refused else "submitted", verdict_date
    )
    return row, None if refused else str(response["id"])


async def _record(
    engine: AsyncEngine,
    account: str,
    order: PlannedOrder,
    response: Any,
    status: str,
    screen_as_of: date | None,
) -> dict[str, Any]:
    broker_id = response.get("id") if isinstance(response, dict) else None
    row = {
        "acc": account,
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
                "INSERT INTO core_orders (account, submitted_at, symbol, side, qty, est_price, "
                "notional, reason, screen_as_of, screen_method, status, broker_order_id, "
                "response) VALUES (:acc, :at, :s, :side, :q, :p, :n, :r, :scr, "
                "(SELECT method FROM halal_screen_current "
                "WHERE as_of = CAST(:scr AS DATE) AND symbol = CAST(:s AS VARCHAR)), "
                ":st, :bid, CAST(:resp AS JSONB))"
            ),
            row,
        )
    logger.info(
        "core order %s %s %.6f %s: %s", order.side, order.symbol, order.qty, order.reason, status
    )
    return {k: row[k] for k in ("s", "side", "q", "n", "r", "st")}


async def monthly_due(engine: AsyncEngine, today: date, account: str = CORE_PAPER) -> bool:
    """True until a monthly rebalance has actually run on ``account`` this month.

    A halted run (stale screen, open orders) and a plan-only run do not count:
    the month is used up only by a rebalance that traded or found nothing to do.
    """
    async with engine.connect() as conn:
        n = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM core_runs WHERE account = :a AND monthly AND executed "
                    "AND halted IS NULL AND run_on >= :m"
                ),
                {"a": account, "m": today.replace(day=1)},
            )
        ).scalar()
    return not n


async def record_run(
    engine: AsyncEngine, p: Plan, *, today: date, executed: bool, halted: str | None = None
) -> None:
    """One row per run. ``executed`` only when orders went out (or none were
    needed); ``halted`` is why the run was stopped or restricted, if it was."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO core_runs (account, run_on, monthly, executed, equity, cash, orders, "
                "halted, screen_as_of, notes, recorded_at) VALUES (:a, :d, :m, :e, :eq, :c, :n, "
                ":h, :s, CAST(:notes AS JSONB), now())"
            ),
            {
                "a": p.account,
                "d": today,
                "m": p.monthly,
                "e": executed,
                "eq": p.equity,
                "c": p.cash,
                "n": len(p.orders),
                "h": halted or p.halted,
                "s": p.screen_as_of,
                "notes": json.dumps(p.notes),
            },
        )


# ── one run, end to end: the scheduled job and `halal-trader core run` ──


@dataclass
class RunOutcome:
    account: str
    plan: Plan | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)  # preflight: nothing was planned
    market_closed: bool = False
    kill_switch: bool = False

    @property
    def submitted(self) -> list[dict[str, Any]]:
        return [r for r in self.results if r["st"] == "submitted"]

    @property
    def rejected(self) -> list[dict[str, Any]]:
        return [r for r in self.results if r["st"] != "submitted"]


KILL_SWITCH_NOTE = "kill-switch engaged: forced sales (screen) only, no buys"


async def run(
    engine: AsyncEngine,
    broker: Any,
    settings: Any,
    *,
    today: date,
    execute_orders: bool,
    monthly: bool | None = None,
    check_token: bool = True,
    day_trader: Callable[[Any], Awaitable[Any]] | None = None,
    now: datetime | None = None,
    sell_timeout: float = SELL_FILL_TIMEOUT_S,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> RunOutcome:
    """Preflight, plan, and (``execute_orders``) place and record one core run.

    Refused outright, before any plan: a closed market (when executing), and
    any core/safeguards.py preflight problem (other account's keys, and for
    live money the token, the paper gate and a cash account). With the
    kill-switch engaged the run is restricted to forced sales: selling what
    the screen no longer holds halal is the one trade an emergency stop must
    not block. A plan-only call records nothing.
    """
    from halal_trader.core import safeguards
    from halal_trader.core.halt import is_halted

    core = settings.core
    out = RunOutcome(core_account(core.paper))
    if execute_orders and not (await broker.get_clock()).is_open:
        out.market_closed = True
        return out
    problems = await safeguards.core_preflight(
        settings,
        engine=engine,
        core=await broker.get_account_info(),
        day_trader=day_trader or safeguards.day_trader_account,
        today=today,
        now=now,
        check_token=check_token,
    )
    if problems and execute_orders:
        out.refused = problems
        return out

    out.kill_switch = await is_halted(engine)
    if out.kill_switch:
        is_monthly = False
    elif monthly is not None:
        is_monthly = monthly
    else:
        is_monthly = await monthly_due(engine, today, out.account)
    p = await plan(
        engine,
        broker,
        today=today,
        monthly=is_monthly,
        top_n=core.top_n,
        account=out.account,
        live_ceiling=None if core.paper else core.live_max_notional,
    )
    if out.kill_switch:
        p.notes.insert(0, KILL_SWITCH_NOTE)
    p.notes.extend(f"would be refused: {x}" for x in problems)
    out.plan = p
    if not execute_orders:
        return out
    out.results = await execute(
        engine, broker, p, today=today, sell_timeout=sell_timeout, sleep=sleep
    )
    await record_run(
        engine,
        p,
        today=today,
        executed=ran(p, out.results),
        halted=p.halted or (KILL_SWITCH_NOTE if out.kill_switch else None),
    )
    return out


def ran(p: Plan, results: list[dict[str, Any]]) -> bool:
    """Did the run trade? Not when halted, and not when every order it tried
    was refused; yes when it placed orders, or legitimately had none to place."""
    if p.halted is not None:
        return False
    return not results or any(r["st"] == "submitted" for r in results)
