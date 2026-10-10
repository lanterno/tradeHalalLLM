"""Random synthetic markets for the look-ahead and determinism tests (seeded)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from halabot.playbooks.types import PathData, SpyData
from halal_trader.data.minutes import BarArrays
from halal_trader.market_hours import next_trading_day
from tests.halabot.playbooks._support import (
    Context,
    Item,
    Story,
    daily,
    epoch,
    et,
    minutes_of,
    path,
)

WEEK = [date(2016, 3, d) for d in (7, 8, 9, 10, 11)] + [date(2016, 3, 14), date(2016, 3, 15)]


def random_session(
    rng: np.random.Generator, day: date, p0: float, *, holes: bool = True
) -> BarArrays:
    """A random walk over ``day``'s minutes, some missing, a gap now and then, some VWAPs null."""
    mins = minutes_of(day)
    keep = np.ones(len(mins), dtype=bool)
    if holes:
        keep &= rng.random(len(mins)) > 0.05
        if rng.random() < 0.5:  # one halt-like gap of 6-12 minutes
            at = int(rng.integers(10, len(mins) - 20))
            keep[at : at + int(rng.integers(6, 13))] = False
        if rng.random() < 0.2:  # a late first print
            keep[: int(rng.integers(5, 12))] = False
    steps = rng.normal(0, 0.002, len(mins))
    c = p0 * np.exp(np.cumsum(steps))
    o = np.concatenate([[p0], c[:-1]])
    spread = np.abs(rng.normal(0, 0.001, len(mins))) + 0.0002
    h = np.maximum(o, c) * (1 + spread)
    low = np.minimum(o, c) * (1 - spread)
    vw = (o + c + h + low) / 4
    vw[rng.random(len(mins)) < 0.1] = np.nan
    v = rng.integers(100, 5_000, len(mins)).astype(np.float64)
    ts = np.asarray([epoch(day, hh, mm) for hh, mm in mins], dtype=np.int64)
    return BarArrays(ts[keep], o[keep], h[keep], low[keep], c[keep], v[keep], vw[keep])


@dataclass
class Market:
    stories: list[Story]
    paths: dict[str, PathData]
    spy: SpyData
    ctx: Context
    sessions: int


def random_market(
    seed: int, symbols: int, *, sessions: int | None = None, prefix: str = "S"
) -> Market:
    """``symbols`` names, each with a story on WEEK[0] and often one on WEEK[1] or WEEK[2]."""
    rng = np.random.default_rng(seed)
    n = sessions or int(rng.choice([1, 3]))
    spy = SpyData({d: random_session(rng, d, 200.0, holes=False) for d in WEEK})
    stories: list[Story] = []
    paths: dict[str, PathData] = {}
    daily_points = {}
    for d in WEEK:
        daily_points[("SPY", d)] = daily(float(spy.days[d].c[-1]), float(spy.days[d].o[0]))
    for j in range(symbols):
        symbol = f"{prefix}{j:03d}"
        p = float(rng.uniform(10, 300))
        bars = {}
        for d in WEEK:
            bars[d] = random_session(rng, d, p)
            p = float(bars[d].c[-1]) if len(bars[d]) else p
            if len(bars[d]):
                daily_points[(symbol, d)] = daily(float(bars[d].c[-1]), float(bars[d].o[0]))
        days = [WEEK[0]]
        if rng.random() < 0.6:
            days.append(WEEK[int(rng.integers(1, 3))])
        for s in days:
            items = []
            if rng.random() < 0.5:  # pre-open news
                items.append(Item(et(s, int(rng.integers(6, 9)), int(rng.integers(0, 60)))))
            else:  # in-session news, before the cutoff
                items.append(Item(et(s, int(rng.integers(10, 15)), int(rng.integers(0, 60)))))
            if rng.random() < 0.3:  # a structural item later
                hh = int(rng.integers(10, 16))
                items.append(Item(et(s, hh, int(rng.integers(0, 60))), "offering", structural=True))
            if rng.random() < 0.3:  # noise
                items.append(
                    Item(et(s, int(rng.integers(9, 16)), int(rng.integers(0, 60))), "noise")
                )
            st = Story(symbol, s, items)
            stories.append(st)
            pdays = [s]
            while len(pdays) < n:
                pdays.append(next_trading_day(pdays[-1]))
            paths[st.story_id] = path(st.story_id, symbol, pdays, [bars[d] for d in pdays])
    verdicts = {}
    for st in stories[::7]:
        verdicts[(st.symbol, next_trading_day(st.session))] = "not_halal"  # some compliance exits
    return Market(stories, paths, spy, Context(daily=daily_points, verdicts=verdicts), n)
