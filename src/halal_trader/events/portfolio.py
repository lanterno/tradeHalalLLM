"""Event portfolios: what trading a signal's events would have earned (Phase C).

A fixed number of equal **slots**: each entry takes 1/slots of the
portfolio's value at that moment, the rest is cash earning nothing. When
more candidates arrive than slots are free, the strongest signals go
first. Entry follows the harness's clock rule (study.entry_point); a
position exits at the close ``hold`` sessions later. Every fill pays the
one-way cost of its liquidity bucket. Only names eligible at entry (that
month's point-in-time universe x the screen of the time) may be bought.

Within a session: open entries, then the mark at the close, then exits at
the close (freeing slots), then close entries.

Known bias, stated: a position whose stock stops trading is carried at
its last price, so a delisting loss after the last bar is missed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
from numpy.typing import NDArray

from halal_trader.events.study import Bars, entry_point

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Candidate:
    symbol: str
    published_at: datetime
    score: float


@dataclass
class _Position:
    symbol: str
    shares: float
    exit_i: int
    last: float


@dataclass
class EventBook:
    days: list[date]
    returns: FloatArray
    trades: int
    skipped_full: int  # candidates turned away because every slot was taken
    skipped_ineligible: int
    exposure: float  # mean share of value invested
    holdings_count: list[int] = field(default_factory=list)


def _eligible(schedule: Sequence[tuple[date, set[str]]], day: date) -> set[str]:
    current = [names for d, names in schedule if d <= day]
    return current[-1] if current else set()


def simulate(
    bars: Bars,
    candidates: Sequence[Candidate],
    *,
    hold: int,
    slots: int,
    start: date,
    eligible_from: Mapping[date, set[str]],
    cost_bps: Callable[[str, date], float],
) -> EventBook:
    sessions = bars.sessions
    first = next((i for i, d in enumerate(sessions) if d >= start), len(sessions))
    entries: dict[tuple[int, str], list[Candidate]] = defaultdict(list)
    for c in candidates:
        point = entry_point(c.published_at, sessions)
        if point is not None and point[0] >= first:
            entries[point].append(c)
    schedule = sorted(eligible_from.items())
    cash, positions = 1.0, []
    nav_prev = 1.0
    returns, holdings, invested = [], [], []
    trades = full = ineligible = 0

    def nav_at(i: int) -> float:
        total = cash
        for p in positions:
            price = bars.close.get(p.symbol, {}).get(sessions[i])
            if price:
                p.last = price
            total += p.shares * p.last
        return total

    def enter(i: int, at: str, value: float) -> None:
        nonlocal cash, trades, full, ineligible
        day = sessions[i]
        allowed = _eligible(schedule, day)
        held = {p.symbol for p in positions}
        prices = bars.open if at == "open" else bars.close
        for c in sorted(entries.get((i, at), []), key=lambda c: -c.score):
            if c.symbol in held:
                continue
            if c.symbol not in allowed:
                ineligible += 1
                continue
            if len(positions) >= slots:
                full += 1
                continue
            price = prices.get(c.symbol, {}).get(day)
            stake = value / slots
            if not price or cash < stake * 0.999:
                continue
            fill = price * (1 + cost_bps(c.symbol, day) / 10_000)
            exit_i = i + hold - (1 if at == "open" else 0)
            positions.append(_Position(c.symbol, stake / fill, exit_i, price))
            cash -= stake
            held.add(c.symbol)
            trades += 1

    for i in range(first, len(sessions)):
        enter(i, "open", nav_prev)
        nav = nav_at(i)
        for p in [p for p in positions if p.exit_i <= i]:
            cash += p.shares * p.last * (1 - cost_bps(p.symbol, sessions[i]) / 10_000)
            positions.remove(p)
        enter(i, "close", nav_at(i))
        nav = nav_at(i)
        returns.append(nav / nav_prev - 1.0)
        holdings.append(len(positions))
        invested.append(1 - cash / nav if nav > 0 else 0.0)
        nav_prev = nav
    return EventBook(
        days=sessions[first:],
        returns=np.array(returns),
        trades=trades,
        skipped_full=full,
        skipped_ineligible=ineligible,
        exposure=float(np.mean(invested)) if invested else 0.0,
        holdings_count=holdings,
    )
