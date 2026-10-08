"""Paper-forward books: a strategy run forward on real closing prices, no orders.

A backtest of S1 cannot escape hindsight (today's universe, today's screen).
A forward book can: every decision is made from data that existed at the
time, and is recorded the evening it is made, so the record cannot be
revised afterwards. Three months of it is the minimum the capital gates ask
for (plan §7.5, G1).

Rules (S1, the only strategy so far):

* rebalance at the close of the first session of each month, plus the first
  session after the book is created;
* the target comes from ``scores_at`` on the previous session's closes, over
  names in that session's point-in-time liquidity universe (the top 1,000
  by trailing dollar volume, as `factor-backtest --pit` uses) that the
  newest halal screen *as of that session* holds halal: the screen history
  is point in time, so a later re-screen never rewrites a decision;
* filled at that session's close (a market-on-close order), paying
  ``cost_bps`` per side on turnover;
* between rebalances the weights drift with prices; a missing bar holds its
  value.

One row per (book, session) in ``forward_book_days``. Rows are append-only
and a session is never recomputed once written.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data.store import BENCHMARKS, stored_symbols
from halal_trader.data.universe import universe_at
from halal_trader.halal import strict
from halal_trader.research.factor_backtest import (
    Prices,
    Stats,
    benchmark_returns,
    load_prices,
    scores_at,
    split_reused_tickers,
    stats,
)

logger = logging.getLogger(__name__)

STRATEGIES = ("s1-momentum-lowvol", "core-strict-cap")
UNIVERSE = 1000  # the point-in-time liquidity universe, as in `factor-backtest --pit`
_HISTORY = timedelta(days=550)  # 252 + 21 sessions of lookback, with room for holidays


@dataclass(frozen=True, slots=True)
class BookDay:
    day: date
    nav: float
    day_return: float
    turnover: float
    weights: dict[str, float]
    rebalance_next: bool


@dataclass(frozen=True, slots=True)
class BookReport:
    name: str
    started: date
    days: int
    nav: float
    stats: Stats | None
    benchmarks: dict[str, Stats | None]
    holdings: dict[str, float]


async def create_book(
    engine: AsyncEngine,
    name: str,
    *,
    strategy: str = "s1-momentum-lowvol",
    top_n: int = 30,
    cost_bps: float = 10.0,
) -> None:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; known: {', '.join(STRATEGIES)}")
    params = {"top_n": top_n, "cost_bps": cost_bps, "strategy": strategy}
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO forward_books (name, strategy, params, created_at) "
                "VALUES (:n, :s, CAST(:p AS JSONB), now())"
            ),
            {"n": name, "s": strategy, "p": json.dumps(params)},
        )


async def book_names(engine: AsyncEngine) -> list[str]:
    async with engine.connect() as conn:
        return [r.name for r in await conn.execute(text("SELECT name FROM forward_books"))]


async def _params(engine: AsyncEngine, name: str) -> dict[str, Any]:
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT strategy, params FROM forward_books WHERE name = :n"), {"n": name}
            )
        ).first()
    if row is None:
        raise KeyError(f"no forward book named {name!r}")
    return {**dict(row.params), "strategy": row.strategy}


async def _screen_caps(
    engine: AsyncEngine, day: date
) -> tuple[date, dict[str, tuple[float, int | None]]] | None:
    """(screen date, symbol -> (market cap on that date, CIK)) for the halal names of
    the newest screen on or before ``day``: price x shares as the screen read them."""
    as_of = await strict.newest_screen(engine, on_or_before=day)
    if as_of is None:
        return None
    rows = await strict.screen_rows(engine, as_of, halal_only=True)
    return as_of, {r.symbol: (r.price * r.shares, r.cik) for r in rows if r.price and r.shares}


def _core_target(
    prices: Prices,
    t: int,
    screened: date,
    caps: dict[str, tuple[float, int | None]],
    eligible: set[str],
    top_n: int,
) -> dict[str, float]:
    """Cap weights of the largest ``top_n`` eligible names, caps rolled from the
    screen date to session ``t`` by price (adjusted closes: splits cancel out)."""
    from halal_trader.portfolio.strict_core import split_share_classes, targets

    index = {s: i for i, s in enumerate(prices.symbols)}
    then = max((i for i, d in enumerate(prices.days) if d <= screened), default=None)
    rolled: dict[str, float] = {}
    for symbol, (cap, _) in caps.items():
        j = index.get(symbol)
        if symbol not in eligible or j is None or then is None:
            continue
        p0, p1 = prices.close[then, j], prices.close[t, j]
        if np.isnan(p0) or np.isnan(p1) or p0 <= 0:
            continue
        rolled[symbol] = cap * float(p1 / p0)
    ciks = {s: caps[s][1] for s in rolled}
    return targets(split_share_classes(rolled, ciks), top_n)


async def _last_day(engine: AsyncEngine, name: str) -> BookDay | None:
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT day, nav, day_return, turnover, weights, rebalance_next "
                    "FROM forward_book_days WHERE book = :n ORDER BY day DESC LIMIT 1"
                ),
                {"n": name},
            )
        ).first()
    if row is None:
        return None
    return BookDay(
        row.day, row.nav, row.day_return, row.turnover, dict(row.weights), row.rebalance_next
    )


async def _halal_as_of(engine: AsyncEngine, day: date) -> set[str] | None:
    """What the newest screen run on or before ``day`` held halal (None: no run yet)."""
    as_of = await strict.newest_screen(engine, on_or_before=day)
    if as_of is None:
        return None
    return {r.symbol for r in await strict.screen_rows(engine, as_of, halal_only=True)}


def target_weights(prices: Prices, t: int, eligible: set[str], top_n: int) -> dict[str, float]:
    """Equal weights over the top ``top_n`` eligible names scored at session ``t``."""
    score = scores_at(prices.close, t)
    allowed = np.array([s in eligible for s in prices.symbols])
    score[~allowed] = np.nan
    ranked = np.argsort(np.where(np.isnan(score), -np.inf, score))[::-1]
    picks = [int(i) for i in ranked[:top_n] if not np.isnan(score[i])]
    return {prices.symbols[i]: 1.0 / len(picks) for i in picks}


def _drift(weights: dict[str, float], prices: Prices, t: int) -> tuple[dict[str, float], float]:
    """Carry ``weights`` from session t-1 to t; returns new weights and the day's return."""
    if not weights:
        return {}, 0.0
    index = {s: i for i, s in enumerate(prices.symbols)}
    grown: dict[str, float] = {}
    for symbol, w in weights.items():
        j = index.get(symbol)
        r = 0.0
        if j is not None:
            prev, now = prices.close[t - 1, j], prices.close[t, j]
            if not (np.isnan(prev) or np.isnan(now)):
                r = float(now / prev - 1.0)
        grown[symbol] = w * (1.0 + r)
    total = sum(grown.values())
    gross = total - sum(weights.values())
    return {s: w / total for s, w in grown.items()} if total > 0 else {}, gross


async def _core_step(
    engine: AsyncEngine,
    prices: Prices,
    t: int,
    weights: dict[str, float],
    rebalance: bool,
    top_n: int,
    universe_size: int,
) -> tuple[dict[str, float], float]:
    """One session of the core: forced sales any day, banded rebalance monthly."""
    from halal_trader.portfolio.strict_core import rebalance as banded
    from halal_trader.portfolio.strict_core import turnover as traded

    day, prev_day = prices.days[t], prices.days[t - 1]
    eligible = await _halal_as_of(engine, prev_day)
    if eligible is None:
        return weights, 0.0
    members = await universe_at(engine, prev_day, top_n=universe_size)
    if members:
        eligible &= set(members)
    eligible -= set(BENCHMARKS)
    failed = set(weights) - eligible
    if not (rebalance or failed or day.month != prev_day.month):
        return weights, 0.0
    if rebalance or day.month != prev_day.month:
        screen = await _screen_caps(engine, prev_day)
        if screen is None:
            return weights, 0.0
        target = _core_target(prices, t - 1, screen[0], screen[1], eligible, top_n)
    else:
        # A mid-month forced sale: the failed names go, the rest are held as they are.
        target = {s: w for s, w in weights.items() if s in eligible}
    new = banded(weights, target, eligible)
    return new, traded(weights, new)


async def advance_book(engine: AsyncEngine, name: str, *, through: date) -> list[BookDay]:
    """Append every stored session after the book's last row, up to ``through``."""
    params = await _params(engine, name)
    top_n, cost = int(params["top_n"]), float(params["cost_bps"]) / 10_000.0
    universe_size = int(params.get("universe", UNIVERSE))
    last = await _last_day(engine, name)
    since = (last.day if last else through) - _HISTORY
    prices, _ = split_reused_tickers(
        await load_prices(engine, await stored_symbols(engine, "all"), since=since)
    )
    sessions = [i for i, d in enumerate(prices.days) if d <= through]
    if not sessions:
        return []

    added: list[BookDay] = []
    if last is None:
        # Genesis on the latest stored session: empty, rebalancing at the next close.
        t0 = sessions[-1]
        added.append(BookDay(prices.days[t0], 1.0, 0.0, 0.0, {}, True))
        logger.info("forward book %s starts on %s", name, prices.days[t0])
    else:
        nav, weights, rebalance = last.nav, last.weights, last.rebalance_next
        for t in (i for i in sessions if prices.days[i] > last.day and i > 0):
            day, prev_day = prices.days[t], prices.days[t - 1]
            weights, gross = _drift(weights, prices, t)
            turnover = 0.0
            if params.get("strategy") == "core-strict-cap":
                weights, turnover = await _core_step(
                    engine, prices, t, weights, rebalance, top_n, universe_size
                )
                rebalance = False
            elif rebalance or day.month != prev_day.month:
                eligible = await _halal_as_of(engine, prev_day)
                members = await universe_at(engine, prev_day, top_n=universe_size)
                if eligible is not None and members:  # no monthly bars yet: no restriction
                    eligible &= set(members)
                if eligible is None:
                    logger.warning(
                        "forward book %s: no halal screen by %s; holding", name, prev_day
                    )
                else:
                    target = target_weights(prices, t - 1, eligible - set(BENCHMARKS), top_n)
                    names = set(target) | set(weights)
                    turnover = sum(abs(target.get(s, 0.0) - weights.get(s, 0.0)) for s in names)
                    weights = target
                rebalance = False
            day_return = (1.0 + gross) * (1.0 - turnover * cost) - 1.0
            nav *= 1.0 + day_return
            added.append(BookDay(day, nav, day_return, turnover, weights, rebalance))

    async with engine.begin() as conn:
        for row in added:
            await conn.execute(
                text(
                    "INSERT INTO forward_book_days (book, day, nav, day_return, turnover, "
                    "weights, rebalance_next, recorded_at) VALUES (:b, :d, :nav, :r, :to, "
                    "CAST(:w AS JSONB), :rb, now()) ON CONFLICT (book, day) DO NOTHING"
                ),
                {
                    "b": name,
                    "d": row.day,
                    "nav": row.nav,
                    "r": row.day_return,
                    "to": row.turnover,
                    "w": json.dumps(row.weights),
                    "rb": row.rebalance_next,
                },
            )
    return added


async def report(engine: AsyncEngine, name: str) -> BookReport:
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT day, nav, day_return, weights FROM forward_book_days "
                    "WHERE book = :n ORDER BY day"
                ),
                {"n": name},
            )
        ).all()
    if not rows:
        raise KeyError(f"forward book {name!r} has no days yet")
    days = [r.day for r in rows]
    returns = np.array([r.day_return for r in rows[1:]])  # the genesis row has no return
    traded = days[1:]
    bench: dict[str, Stats | None] = {}
    if traded:
        prices = await load_prices(engine, list(BENCHMARKS), since=days[0])
        for symbol in BENCHMARKS:
            b = benchmark_returns(prices, symbol, traded)
            bench[symbol] = stats(b) if b is not None and len(b) > 1 else None
    return BookReport(
        name=name,
        started=days[0],
        days=len(traded),
        nav=rows[-1].nav,
        stats=stats(returns) if len(returns) > 1 else None,
        benchmarks=bench,
        holdings=dict(rows[-1].weights),
    )
