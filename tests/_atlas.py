"""A synthetic week for the path atlas: stored news, stories, screens, ranks, bars, H1 rows.

March 2017 (inside the train window). Five names, one week of minute bars:

* ALFA (Technology, rank 0): a pre-open downgrade on Tue 03-07 (NSN_CORE,
  "out"). It falls to 94 by 09:40, holds, reclaims its VWAP at 10:06 and
  holds 95.2; Wednesday it rallies to 97.6 from noon (a 60% retrace),
  Thursday it trades at 93.5 (a new low: a fade). On Thursday at noon an
  SEC probe makes a second, structural story.
* BRVO: an in-session offering on Wed 03-08 at 11:00 (dilution): 50 -> 47.
* CHRL: a product launch on Tue 03-07, flat at 30 (no drop).
* DLTA: a downgrade, but not halal (not PRIMARY).
* ECHO: an earnings preview only (noise: no unit).

SPY trades flat at 200 through March. Every price is synthetic.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, timedelta

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data.minutes import BarArrays
from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl
from halal_trader.events.h1 import NAME as H1_NAME
from halal_trader.events.h1 import STAGE_A_FAIL
from halal_trader.events.stories import build_range, pins
from halal_trader.market_hours import is_trading_day
from tests._renames import mark_renamed_news_done
from tests._stories import add_aliases, news_row, ny, store
from tests.halabot.playbooks._seed import mark_done, seed_bars
from tests.halabot.playbooks._support import Row, minutes_of, session_bars

START, END = date(2017, 3, 6), date(2017, 3, 10)
MON, S1, S2, S3, FRI = (date(2017, 3, d) for d in (6, 7, 8, 9, 10))
MINUTE_DAYS = [MON, S1, S2, S3, FRI, date(2017, 3, 13), date(2017, 3, 14)]
SCREEN_AS_OF = date(2017, 2, 15)
SOFTWARE = "SERVICES-PREPACKAGED SOFTWARE"
RETAIL = "RETAIL-VARIETY STORES"

# symbol: (alias, base price, cik, sic, verdict)
NAMES: dict[str, tuple[str, float, int, str, str]] = {
    "ALFA": ("Alfa", 100.0, 11, SOFTWARE, "halal"),
    "BRVO": ("Bravo", 50.0, 12, RETAIL, "halal"),
    "CHRL": ("Charlie", 30.0, 13, RETAIL, "halal"),
    "DLTA": ("Delta", 40.0, 14, RETAIL, "not_halal"),
    "ECHO": ("Echo", 20.0, 15, RETAIL, "halal"),
}
HEADLINES = [
    news_row(1, "ALFA", ny(S1, 8), "Morgan Stanley Downgrades Alfa to Equal-Weight"),
    news_row(2, "CHRL", ny(S1, 7), "Charlie Launches New Chip"),
    news_row(3, "DLTA", ny(S1, 8), "Barclays Downgrades Delta to Underweight"),
    news_row(4, "ECHO", ny(S1, 9), "Earnings Preview: Echo"),
    news_row(5, "BRVO", ny(S2, 11), "Bravo Prices $50M Public Offering"),
    news_row(6, "ALFA", ny(S3, 12), "Alfa Faces SEC Probe Into Disclosures"),
]

# ALFA's daily closes from S-1 on (the minute bars' closes); SPY's are 200 in March.
ALFA_CLOSES = {MON: 100.0, S1: 95.2, S2: 97.6, S3: 93.5}
BRVO_CLOSES = {S2: 47.3}


def _sessions(lo: date, hi: date) -> list[date]:
    out, d = [], lo
    while d <= hi:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def daily_closes(symbol: str) -> dict[date, float]:
    """The name's daily closes (raw = all-adjusted: no corporate action)."""
    rng = np.random.default_rng(sum(map(ord, symbol)))
    if symbol == "SPY":
        days = _sessions(date(2016, 1, 4), date(2021, 12, 31))
        out = {d: 200.0 * (1.0 + float(rng.normal(0.0, 0.005))) for d in days}
        out.update({d: 200.0 for d in days if (d.year, d.month) == (2017, 3)})
        return out
    base = NAMES[symbol][1]
    days = _sessions(date(2016, 1, 4), date(2017, 4, 28))
    out = {d: base * (1.0 + float(rng.normal(0.0, 0.008))) for d in days}
    out[MON] = base
    tail = {"ALFA": 93.5, "BRVO": 47.3}.get(symbol, base)
    for d in days:
        if d > MON:
            out[d] = tail
    if symbol == "ALFA":
        out.update(ALFA_CLOSES)
    if symbol == "BRVO":
        out[S1] = base
    return out


def _ramp(
    day: date, start: tuple[int, int], prices: list[float], volume: float
) -> dict[tuple[int, int], Row]:
    """Bars from ``start`` on, one a minute, closing at ``prices`` (opened at the previous)."""
    rows: dict[tuple[int, int], Row] = {}
    hh, mm = start
    prev = prices[0]
    for p in prices:
        rows[(hh, mm)] = (prev, max(prev, p), min(prev, p), p, volume, p)
        prev = p
        mm += 1
        if mm == 60:
            hh, mm = hh + 1, 0
    return rows


def minute_bars(symbol: str, day: date) -> BarArrays:
    if symbol == "SPY":
        return session_bars(day, price=200.0)
    if symbol == "ALFA":
        if day == S1:
            rows = _ramp(day, (9, 30), [97.0 - 0.2 * k for k in range(10)], 100.0)
            rows[(9, 40)] = (95.2, 95.2, 94.0, 94.5, 10_000.0, 94.6)
            for m in range(41, 66):
                rows[(9 + m // 60, m % 60)] = (94.6, 94.7, 94.5, 94.6, 10_000.0, 94.6)
            rows[(10, 6)] = (94.6, 95.1, 94.6, 95.0, 1_000.0, 95.0)
            return session_bars(day, price=95.2, rows=rows)
        if day == S2:
            noon = {
                (h, m): (97.6, 97.6, 97.6, 97.6, 1_000.0, 97.6)
                for h in range(12, 16)
                for m in range(60)
            }
            return session_bars(day, price=95.2, rows=noon)
        if day == MON:
            return session_bars(day, price=100.0)
        return session_bars(day, price=93.5, rows=_flat_band(day, 93.5, 93.6, 93.4))
    if symbol == "BRVO":
        if day == S2:
            rows = {
                (h, m): (50.0, 50.0, 50.0, 50.0, 1_000.0, 50.0)
                for h in (9, 10)
                for m in range(60)
                if (h, m) >= (9, 30)
            }
            rows[(11, 0)] = (50.0, 50.0, 48.0, 48.5, 5_000.0, 49.0)
            rows.update(_ramp(day, (11, 1), [48.2, 47.9, 47.6, 47.3, 47.1], 2_000.0))
            rows[(11, 5)] = (47.3, 47.3, 47.0, 47.1, 2_000.0, 47.1)
            return session_bars(day, price=47.3, rows=rows)
        return session_bars(day, price=50.0 if day < S2 else 47.3)
    return session_bars(day, price=NAMES[symbol][1])


def _flat_band(day: date, c: float, h: float, low: float) -> dict[tuple[int, int], Row]:
    return dict.fromkeys(minutes_of(day), (c, h, low, c, 1_000.0, c))


async def seed_daily(engine: AsyncEngine) -> None:
    rows = []
    for symbol in [*NAMES, "SPY"]:
        for d, c in daily_closes(symbol).items():
            for adjustment in ("raw", "all"):
                rows.append({"s": symbol, "d": d, "a": adjustment, "c": c})
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES (:s, :d, :a, :c, :c, :c, :c, 1000000, now())"
            ),
            rows,
        )


async def seed_universe_and_screen(engine: AsyncEngine) -> None:
    months = [date(2016 + (k // 12), k % 12 + 1, 1) for k in range(15)]  # 2016-01 .. 2017-03
    monthly = [
        {"s": s, "m": m, "c": NAMES[s][1], "v": 1e9 / (rank + 1)}
        for rank, s in enumerate(NAMES)
        for m in months
    ]
    screen = [
        {"a": SCREEN_AS_OF, "s": s, "v": verdict, "k": cik, "d": sic, "r": json.dumps([])}
        for s, (_, _, cik, sic, verdict) in NAMES.items()
    ]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO monthly_bars (symbol, month, close, volume, vwap) "
                "VALUES (:s, :m, :c, :v, :c)"
            ),
            monthly,
        )
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) "
                "VALUES (:a, :s, :k, :d, :v, CAST(:r AS JSONB), '{}', 'v12', now())"
            ),
            screen,
        )


async def seed_minutes(engine: AsyncEngine) -> None:
    units = []
    for symbol in ("ALFA", "BRVO", "CHRL", "SPY"):
        for d in MINUTE_DAYS:
            await seed_bars(engine, symbol, minute_bars(symbol, d))
            units.append((symbol, d))
    await mark_done(engine, units)


async def seed_stories(engine: AsyncEngine) -> None:
    """The headlines, the aliases, every renamed month done (``small_map``), the stories."""
    await store(engine, HEADLINES)
    await add_aliases(
        engine,
        [(s, alias, "name") for s, (alias, *_) in NAMES.items()]
        + [(s, s, "ticker") for s in NAMES],
    )
    await mark_renamed_news_done(engine)
    await build_range(engine, start=date(2017, 3, 1), end=date(2017, 3, 31))


CONFIG = {"hypothesis": "news.h1.overreaction_bounce", "version": 1, "test": True}


async def register_h1(
    engine: AsyncEngine,
    *,
    closing: str | None = "verdict",
    pins: Mapping[str, str] | None = None,
) -> int:
    """The H1 registration; ``closing`` "verdict", "stage-a" (insufficient events) or None.

    With ``pins`` the configuration records them, as H1's PREREG does.
    """
    config = CONFIG if pins is None else {**CONFIG, "pins": dict(pins)}
    repo = QuantTrialRepoImpl(engine)
    reg = await repo.record_trial(name=H1_NAME, kind="preregistration", config=config)
    if closing == "verdict":
        await repo.record_trial(name=H1_NAME, kind="verdict", config=config, verdict="fail")
    elif closing == "stage-a":
        await repo.record_trial(name=H1_NAME, kind="stage-a", config=config, verdict=STAGE_A_FAIL)
    return reg


async def seed_world(engine: AsyncEngine) -> int:
    """Everything the atlas reads, H1 closed by a verdict under the story pins in force;
    returns the registration id."""
    await seed_daily(engine)
    await seed_universe_and_screen(engine)
    await seed_minutes(engine)
    await seed_stories(engine)
    return await register_h1(engine, pins=await pins(engine))
