"""Event studies: what happened after events, by signal strength (roadmap Phase B).

**Entry** follows the clock, with daily bars only:

* before 09:30 New York on a session day: that session's **open**;
* during the session: that session's **close** (the first price a daily
  bar guarantees after the event; minute-bar entries are a later,
  separate refinement);
* after 16:00, or on a closed day: the **next** session's open.

**Exit** is the close ``h`` sessions after entry (an open entry on day i
with h = 1 exits at day i's close). Returns are **abnormal** (minus SPY,
which spans the whole history, over the same prices) and **net**: each
round trip pays twice the one-way cost of its liquidity bucket (half-spread + impact), taken from
the symbol's rank in that month's point-in-time universe.

**Summary:** mean net abnormal return per signal decile and horizon,
with its t-statistic, plus the rank IC; optionally split by year or by
liquidity bucket. A harness is trusted only after it reproduces a known
effect (SUE drift in 2016-2019), the T3 lesson.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.signal_eval import information_coefficient

# The market an event study measures against, and its calendar. SPY, not
# SPUS: SPUS starts 2019-12-18, and a calendar taken from its bars once
# dated every 2016-2019 event to its first session (the harness then
# "found" -15% for all deciles: the 2020 crash). Strategies are still
# judged against SPUS in the trials ledger.
BENCHMARK = "SPY"
HORIZONS = (1, 5, 20, 60)
_ET = ZoneInfo("America/New_York")
_OPEN, _CLOSE = time(9, 30), time(16, 0)
# One-way cost in basis points by liquidity rank in the month's universe:
# half the typical quoted spread plus 5 bps of impact for a small order.
COST_BPS = ((300, 7.0), (1000, 15.0), (10**9, 30.0))


@dataclass(frozen=True, slots=True)
class Observation:
    symbol: str
    published_at: datetime
    signal: float
    tags: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Bars:
    sessions: list[date]
    open: dict[str, dict[date, float]]
    close: dict[str, dict[date, float]]


def entry_point(published: datetime, sessions: Sequence[date]) -> tuple[int, str] | None:
    """(session index, "open" | "close") of the first price tradable after ``published``."""
    local = published.astimezone(_ET)
    # Before the calendar starts (beyond a weekend or holiday gap) there is no
    # price to enter at; the next stored session may be months away.
    if not sessions or (sessions[0] - local.date()).days > 5:
        return None
    i = bisect_left(sessions, local.date())
    if i < len(sessions) and sessions[i] == local.date():
        if local.time() < _OPEN:
            return i, "open"
        if local.time() < _CLOSE:
            return i, "close"
        i += 1
    return (i, "open") if i < len(sessions) else None


def cost_bps(liquidity_rank: int | None) -> float:
    rank = liquidity_rank if liquidity_rank is not None else 10**8
    return next(bps for limit, bps in COST_BPS if rank < limit)


async def load_bars(engine: AsyncEngine, symbols: Sequence[str]) -> Bars:
    """Adjusted daily opens and closes for ``symbols`` and the benchmark."""
    wanted = sorted(set(symbols) | {BENCHMARK})
    opens: dict[str, dict[date, float]] = defaultdict(dict)
    closes: dict[str, dict[date, float]] = defaultdict(dict)
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT symbol, day, open, close FROM daily_bars "
                "WHERE adjustment = 'all' AND symbol = ANY(:s)"
            ),
            {"s": wanted},
        )
        for r in rows:
            opens[r.symbol][r.day] = float(r.open)
            closes[r.symbol][r.day] = float(r.close)
    return Bars(sorted(closes.get(BENCHMARK, {})), opens, closes)


def outcome(
    bars: Bars, symbol: str, entry: tuple[int, str], horizon: int, one_way_bps: float
) -> float | None:
    """Net abnormal return of holding ``symbol`` from ``entry`` for ``horizon`` sessions."""
    i, at = entry
    exit_i = i + horizon - (1 if at == "open" else 0)
    if exit_i >= len(bars.sessions):
        return None
    start_day, end_day = bars.sessions[i], bars.sessions[exit_i]
    prices = bars.open if at == "open" else bars.close
    p0 = prices.get(symbol, {}).get(start_day)
    p1 = bars.close.get(symbol, {}).get(end_day)
    b0 = prices.get(BENCHMARK, {}).get(start_day)
    b1 = bars.close.get(BENCHMARK, {}).get(end_day)
    if not (p0 and p1 and b0 and b1):
        return None
    return (p1 / p0 - 1.0) - (b1 / b0 - 1.0) - 2 * one_way_bps / 10_000


@dataclass(frozen=True, slots=True)
class DecileRow:
    group: str
    horizon: int
    decile: int  # 1 = weakest signal, 10 = strongest
    n: int
    mean: float
    t: float


@dataclass(frozen=True, slots=True)
class StudyResult:
    rows: list[DecileRow]
    ic: dict[tuple[str, int], float]
    n: dict[str, int]


def summarise(
    outcomes: Sequence[tuple[Observation, int, float]], *, by: str | None = None
) -> StudyResult:
    """Deciles of signal (within each group) -> mean net abnormal return and its t-stat.

    ``outcomes`` are (observation, horizon, net abnormal return); ``by`` names a
    tag to split on (e.g. "year", "bucket").
    """
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for obs, horizon, ret in outcomes:
        group = str(obs.tags.get(by, "all")) if by else "all"
        grouped[(group, horizon)].append((obs.signal, ret))
    rows: list[DecileRow] = []
    ic: dict[tuple[str, int], float] = {}
    counts: dict[str, int] = defaultdict(int)
    for (group, horizon), pairs in sorted(grouped.items()):
        signal = np.array([p[0] for p in pairs])
        rets = np.array([p[1] for p in pairs])
        ic[(group, horizon)] = information_coefficient(signal, rets)
        counts[group] = max(counts[group], len(pairs))
        if len(pairs) < 50:
            continue
        ranks = signal.argsort().argsort()
        deciles = np.minimum(ranks * 10 // len(pairs), 9)
        for d in range(10):
            sample = rets[deciles == d]
            if len(sample) < 2:
                continue
            sd = float(sample.std(ddof=1))
            t = float(sample.mean()) / (sd / math.sqrt(len(sample))) if sd > 0 else 0.0
            rows.append(DecileRow(group, horizon, d + 1, len(sample), float(sample.mean()), t))
    return StudyResult(rows, ic, dict(counts))


def bucket_of(rank: int | None) -> str:
    if rank is None or rank >= 1000:
        return "small (1000+)"
    return "large (<300)" if rank < 300 else "mid (300-1000)"


async def evaluate(
    engine: AsyncEngine, observations: Sequence[Observation], horizons: Sequence[int] = HORIZONS
) -> list[tuple[Observation, int, float]]:
    """Net abnormal outcome of every observation at every horizon that has elapsed.

    Each observation is tagged with its year and its liquidity bucket in the
    point-in-time universe of its month (ranked by mean dollar volume).
    """
    from halal_trader.data.universe import universe_at

    bars = await load_bars(engine, sorted({o.symbol for o in observations}))
    ranks: dict[date, dict[str, int]] = {}
    out: list[tuple[Observation, int, float]] = []
    for obs in observations:
        month = obs.published_at.astimezone(_ET).date().replace(day=1)
        if month not in ranks:
            names = await universe_at(engine, month, top_n=3000)
            ranks[month] = {s: i for i, s in enumerate(names)}
        rank = ranks[month].get(obs.symbol)
        entry = entry_point(obs.published_at, bars.sessions)
        if entry is None:
            continue
        tagged = Observation(
            obs.symbol,
            obs.published_at,
            obs.signal,
            {**obs.tags, "year": obs.published_at.year, "bucket": bucket_of(rank)},
        )
        for h in horizons:
            ret = outcome(bars, obs.symbol, entry, h, cost_bps(rank))
            if ret is not None:
                out.append((tagged, h, ret))
    return out
