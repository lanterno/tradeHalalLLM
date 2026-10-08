"""Purification: the share of each dividend that came from impermissible income.

A company passing the screen may still earn up to 5% of its revenue from
interest. That share of every dividend it pays is not ours to keep and is
given away (AAOIFI Shariah Standard No. 21, the dividend method):

    amount = shares held x dividend per share x impure-income ratio

* **shares held** at the close before the ex-date (the holder of record
  since T+1 settlement). For a broker account they come from the broker's
  position snapshot moved to that close by the fills in between, so a
  split or spin-off is counted as the broker counts it; from the fills
  alone where no snapshot can be used (:func:`held_shares`). Alpaca's paper
  accounts credit no dividends, so the shares, not DIV activities, are the
  basis. For a forward book they are the book's weight x NAV x a notional
  amount / the close.
* **impure-income ratio** from the screen in force at the ex-date
  (interest income / revenue). Where the screen has none, 5% -- the most
  a passing company may have -- so an unknown errs towards giving more.

The ratio sees interest income only. Revenue from impermissible lines of
business inside an otherwise permissible company is not in SEC data, so
this is a floor, and the report says so.

Accruals are written once per (account, dividend) and never rewritten.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.market_hours import MARKET_TZ
from halal_trader.portfolio.core_account import BROKER_ACCOUNTS

logger = logging.getLogger(__name__)

DEFAULT_RATIO = 0.05
METHOD = "dividend x impure-income ratio (interest income / revenue), screen at ex-date"
BOOK_NOTIONAL = 10_000.0  # forward books are unitless: purification reported per $10,000


async def sync_dividends(
    engine: AsyncEngine, market: Any, symbols: Iterable[str], *, start: date, end: date
) -> int:
    rows = await market.cash_dividends(symbols, start=start, end=end)
    stored = 0
    async with engine.begin() as conn:
        for r in rows:
            result = await conn.execute(
                text(
                    "INSERT INTO dividends (source_id, symbol, ex_date, payable_date, record_date, "
                    "rate, special) VALUES (:id, :s, :ex, :pay, :rec, :rate, :sp) "
                    "ON CONFLICT (source_id) DO NOTHING"
                ),
                {
                    "id": str(r["id"]),
                    "s": str(r["symbol"]).upper(),
                    "ex": date.fromisoformat(r["ex_date"]),
                    "pay": date.fromisoformat(r["payable_date"]) if r.get("payable_date") else None,
                    "rec": date.fromisoformat(r["record_date"]) if r.get("record_date") else None,
                    "rate": float(r["rate"]),
                    "sp": bool(r.get("special")),
                },
            )
            stored += result.rowcount or 0
    return stored


async def impure_ratio(
    engine: AsyncEngine, symbol: str, day: date
) -> tuple[float, date | None, bool]:
    """(ratio, screen date, known) from the newest screen of ``symbol`` on or before ``day``."""
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT as_of, (metrics->>'impure_income_ratio')::float AS r "
                    "FROM halal_screen_current WHERE symbol = :s AND as_of <= :d "
                    "ORDER BY as_of DESC LIMIT 1"
                ),
                {"s": symbol, "d": day},
            )
        ).first()
    if row is None or row.r is None:
        return DEFAULT_RATIO, row.as_of if row else None, False
    return max(float(row.r), 0.0), row.as_of, True


async def paper_positions(
    engine: AsyncEngine, before: date, account: str = "paper"
) -> dict[str, float]:
    """Shares ``account`` held at the close of the last session before ``before``
    (New York), rebuilt from the broker ledger's fills (flat when the ledger began)."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT symbol, sum(CASE WHEN side = 'buy' THEN qty ELSE -qty END) AS q "
                "FROM broker_activities WHERE activity_type = 'FILL' AND account = :a "
                "AND (transaction_time AT TIME ZONE 'America/New_York')::date < :d "
                "GROUP BY symbol"
            ),
            {"d": before, "a": account},
        )
        return {r.symbol: float(r.q) for r in rows if r.q and float(r.q) > 1e-9}


SNAPSHOT_FILL_MARGIN = timedelta(minutes=2)


async def held_shares(engine: AsyncEngine, before: date, account: str) -> dict[str, float]:
    """Shares ``account`` held at the close of the last session before ``before``.

    The broker's own position snapshot (``account_snapshots``, the newest one)
    is the count of record: it carries splits, spin-offs and anything else
    that changes a holding without a fill. It is moved to that close by the
    fills between the two moments, in whichever direction. Rebuilding from
    fills alone (:func:`paper_positions`) counts a 2-for-1 split as half the
    shares, and is the fallback when there is no usable snapshot: none taken,
    the ledger not yet synced past it (its fills would be missing), or a fill
    within ``SNAPSHOT_FILL_MARGIN`` of it (the two clocks cannot order them).
    """
    from halal_trader.core.heartbeat import STOCK_LEDGER

    async with engine.connect() as conn:
        snap = (
            await conn.execute(
                text("SELECT taken_at, positions FROM account_snapshots WHERE account = :a"),
                {"a": account},
            )
        ).first()
        synced = (
            await conn.execute(
                text("SELECT beat_at FROM heartbeats WHERE component = :c"), {"c": STOCK_LEDGER}
            )
        ).scalar()
        if snap is None or synced is None or synced < snap.taken_at:
            return await paper_positions(engine, before, account)
        near = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM broker_activities WHERE account = :a "
                    "AND activity_type = 'FILL' AND transaction_time BETWEEN :lo AND :hi"
                ),
                {
                    "a": account,
                    "lo": snap.taken_at - SNAPSHOT_FILL_MARGIN,
                    "hi": snap.taken_at + SNAPSHOT_FILL_MARGIN,
                },
            )
        ).scalar()
        if near:
            return await paper_positions(engine, before, account)
        # The close before ``before``: every fill dated (New York) before it.
        cutoff = datetime.combine(before, time(), MARKET_TZ)
        lo, hi, sign = (
            (cutoff, snap.taken_at, -1.0)
            if snap.taken_at >= cutoff
            else (snap.taken_at, cutoff, 1.0)
        )
        moves = await conn.execute(
            text(
                "SELECT symbol, sum(CASE WHEN side = 'buy' THEN qty ELSE -qty END) AS q "
                "FROM broker_activities WHERE account = :a AND activity_type = 'FILL' "
                "AND transaction_time >= :lo AND transaction_time < :hi GROUP BY symbol"
            ),
            {"a": account, "lo": lo, "hi": hi},
        )
        shares: dict[str, float] = {}
        for p in snap.positions or []:
            if p.get("qty") is not None:
                shares[str(p["symbol"])] = float(p["qty"])
        for r in moves:
            shares[r.symbol] = shares.get(r.symbol, 0.0) + sign * float(r.q or 0.0)
    return {s: q for s, q in shares.items() if q > 1e-9}


@dataclass(frozen=True, slots=True)
class Accrual:
    account: str
    dividend_id: str
    symbol: str
    ex_date: date
    payable_date: date | None
    shares: float
    dividend: float
    ratio: float
    screen_as_of: date | None
    known: bool

    @property
    def amount(self) -> float:
        return self.dividend * self.ratio


async def _pending(engine: AsyncEngine, account: str, through: date) -> list[Any]:
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT d.source_id, d.symbol, d.ex_date, d.payable_date, d.rate "
                "FROM dividends d WHERE d.ex_date <= :t AND NOT EXISTS ("
                "SELECT 1 FROM purification_accruals a WHERE a.account = :acc "
                "AND a.dividend_id = d.source_id) ORDER BY d.ex_date"
            ),
            {"t": through, "acc": account},
        )
        return list(result.all())


async def _store(engine: AsyncEngine, accruals: list[Accrual]) -> None:
    if not accruals:
        return
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO purification_accruals (account, dividend_id, symbol, ex_date, "
                "payable_date, shares, dividend, impure_ratio, screen_as_of, amount, method, "
                "accrued_at) VALUES (:acc, :id, :s, :ex, :pay, :sh, :div, :r, :scr, :amt, :m, :at) "
                "ON CONFLICT (account, dividend_id) DO NOTHING"
            ),
            [
                {
                    "acc": a.account,
                    "id": a.dividend_id,
                    "s": a.symbol,
                    "ex": a.ex_date,
                    "pay": a.payable_date,
                    "sh": a.shares,
                    "div": a.dividend,
                    "r": a.ratio,
                    "scr": a.screen_as_of,
                    "amt": a.amount,
                    "m": METHOD
                    if a.known
                    else f"{METHOD}; ratio unknown, {DEFAULT_RATIO:.0%} assumed",
                    "at": datetime.now(UTC),
                }
                for a in accruals
            ],
        )


async def accrue_account(engine: AsyncEngine, account: str, *, through: date) -> list[Accrual]:
    """Accrue every stored dividend a broker account ("paper", "core", "core-live")
    held at its ex-date, counting the shares from the broker's snapshot where it can."""
    out = []
    for d in await _pending(engine, account, through):
        shares = (await held_shares(engine, d.ex_date, account)).get(d.symbol, 0.0)
        if shares <= 0:
            continue
        ratio, as_of, known = await impure_ratio(engine, d.symbol, d.ex_date)
        out.append(
            Accrual(
                account,
                d.source_id,
                d.symbol,
                d.ex_date,
                d.payable_date,
                shares,
                shares * d.rate,
                ratio,
                as_of,
                known,
            )
        )
    await _store(engine, out)
    return out


async def accrue_paper(engine: AsyncEngine, *, through: date) -> list[Accrual]:
    """The day-trader's account (the ledger's original, "paper")."""
    return await accrue_account(engine, "paper", through=through)


async def accrue_book(
    engine: AsyncEngine, book: str, *, through: date, notional: float = BOOK_NOTIONAL
) -> list[Accrual]:
    """Accrue a forward book's dividends as if ``notional`` dollars had followed it."""
    account = f"book:{book}"
    async with engine.connect() as conn:
        days = (
            await conn.execute(
                text(
                    "SELECT day, nav, weights FROM forward_book_days WHERE book = :b ORDER BY day"
                ),
                {"b": book},
            )
        ).all()
    if not days:
        return []
    out = []
    for d in await _pending(engine, account, through):
        held = [r for r in days if r.day < d.ex_date]
        if not held:
            continue
        last = held[-1]
        weight = dict(last.weights).get(d.symbol, 0.0)
        if weight <= 0:
            continue
        async with engine.connect() as conn:
            close = (
                await conn.execute(
                    text(
                        "SELECT close FROM daily_bars WHERE symbol = :s AND adjustment = 'raw' "
                        "AND day = :d"
                    ),
                    {"s": d.symbol, "d": last.day},
                )
            ).scalar()
        if not close:
            continue
        shares = weight * last.nav * notional / float(close)
        ratio, as_of, known = await impure_ratio(engine, d.symbol, d.ex_date)
        out.append(
            Accrual(
                account,
                d.source_id,
                d.symbol,
                d.ex_date,
                d.payable_date,
                shares,
                shares * d.rate,
                ratio,
                as_of,
                known,
            )
        )
    await _store(engine, out)
    return out


async def held_symbols(engine: AsyncEngine, since: date) -> set[str]:
    """Every symbol the paper account or any forward book held since ``since``."""
    async with engine.connect() as conn:
        paper = {
            r.symbol
            for r in await conn.execute(
                text(
                    "SELECT DISTINCT symbol FROM broker_activities WHERE activity_type = 'FILL' "
                    "AND transaction_time >= :d AND symbol IS NOT NULL"
                ),
                {"d": since},
            )
        }
        books = {
            k
            for r in await conn.execute(
                text("SELECT weights FROM forward_book_days WHERE day >= :d"), {"d": since}
            )
            for k in dict(r.weights)
        }
    return paper | books


@dataclass(frozen=True, slots=True)
class PurificationLine:
    symbol: str
    dividends: float
    amount: float
    payments: int
    assumed: int  # payments whose ratio was unknown (5% assumed)


async def unpaid(engine: AsyncEngine, accounts: Iterable[str] = BROKER_ACCOUNTS) -> float:
    """What the accounts still owe to charity. The default is every broker
    account: a forward book's accruals ("book:*") are per a notional $10,000
    that nobody holds, so nothing is set aside for them."""
    async with engine.connect() as conn:
        total = (
            await conn.execute(
                text(
                    "SELECT coalesce(sum(amount), 0) FROM purification_accruals "
                    "WHERE paid_at IS NULL AND account = ANY(:a)"
                ),
                {"a": list(accounts)},
            )
        ).scalar()
    return float(total or 0.0)


async def report(engine: AsyncEngine, account: str, year: int) -> list[PurificationLine]:
    """Per symbol, for dividends payable in ``year`` (by ex-date when no pay date)."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT symbol, sum(dividend) AS d, sum(amount) AS a, count(*) AS n, "
                "count(*) FILTER (WHERE method LIKE '%assumed%') AS u "
                "FROM purification_accruals WHERE account = :acc "
                "AND extract(year FROM coalesce(payable_date, ex_date)) = :y "
                "GROUP BY symbol ORDER BY sum(amount) DESC"
            ),
            {"acc": account, "y": year},
        )
        return [
            PurificationLine(r.symbol, float(r.d), float(r.a), int(r.n), int(r.u)) for r in rows
        ]


async def mark_paid(engine: AsyncEngine, account: str, *, through: date, paid_to: str) -> float:
    """Record a donation covering every unpaid accrual payable on or before ``through``.

    Returns the amount marked paid. Accruals are never edited otherwise.
    """
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                "UPDATE purification_accruals SET paid_at = now(), paid_to = :to "
                "WHERE account = :acc AND paid_at IS NULL "
                "AND coalesce(payable_date, ex_date) <= :t RETURNING amount"
            ),
            {"acc": account, "t": through, "to": paid_to},
        )
        return float(sum(r.amount for r in result))


async def clear_unpaid(engine: AsyncEngine) -> int:
    """Drop every accrual not yet paid, so it is recomputed under the current screen.

    The correction path when the screen's ratios change (v9 stopped reading an
    unreported interest income as zero). Paid accruals are settled history and
    stay exactly as they were.
    """
    async with engine.begin() as conn:
        result = await conn.execute(text("DELETE FROM purification_accruals WHERE paid_at IS NULL"))
        return result.rowcount or 0
