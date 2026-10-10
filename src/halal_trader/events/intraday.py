"""The reactor's "fast in" thesis, tested on minute bars (Phase D, pre-registered).

Daily bars showed the LLM reads news well but that the move is priced by
the first daily-bar entry. What remains is speed: buying within a minute
of an in-session headline. For the first in-session headline of each
symbol and day (09:30-15:30 New York, leaving time to trade):

* **entry** at the VWAP of the first minute bar starting at least
  ``LATENCY`` after publication (ingest, decide, route), plus the bucket's
  one-way cost;
* **exits** at the same session's last regular minute close, and at the raw
  daily close one and five sessions later, paying the cost again;
* **abnormal** against SPY over the same timestamps.

Minute bars are stored in ``minute_bars`` (data/minutes.py: raw, regular
session) and fetched once per (symbol, day).
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.study import cost_bps
from halal_trader.market_hours import MARKET_OPEN, MARKET_TZ

logger = logging.getLogger(__name__)

LATENCY = timedelta(seconds=60)
SESSION = (MARKET_OPEN, time(15, 30))
STRONG = 0.4
CONTROL_SAMPLE = 1500
_PLAUSIBLE = (0.5, 2.0)


@dataclass(frozen=True, slots=True)
class Headline:
    symbol: str
    published_at: datetime
    score: float


async def first_in_session(
    engine: AsyncEngine,
    scorer_prefix: str = "llm-batch:",
    *,
    scored_before: datetime | None = None,
) -> list[Headline]:
    """The first scored in-session headline of each (symbol, New York day).

    ``scored_before`` pins the set to the scores that existed then, so a
    result can be reproduced after later scoring runs add headlines.
    """
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT e.symbol, e.published_at, s.score FROM event_scores s "
                "JOIN events e ON e.id = s.event_id WHERE s.scorer LIKE :p AND e.kind = 'news' "
                "AND (CAST(:before AS timestamptz) IS NULL OR s.scored_at < :before)"
            ),
            {"p": scorer_prefix + "%", "before": scored_before},
        )
        first: dict[tuple[str, date], Headline] = {}
        for r in rows:
            local = r.published_at.astimezone(MARKET_TZ)
            if local.weekday() >= 5 or not SESSION[0] <= local.time() < SESSION[1]:
                continue
            key = (r.symbol, local.date())
            if key not in first or r.published_at < first[key].published_at:
                first[key] = Headline(r.symbol, r.published_at, float(r.score))
    return sorted(first.values(), key=lambda h: h.published_at)


def selection(headlines: list[Headline], seed: int = 11) -> list[Headline]:
    strong = [h for h in headlines if abs(h.score) >= STRONG]
    rest = [h for h in headlines if abs(h.score) < STRONG]
    return strong + random.Random(seed).sample(rest, min(CONTROL_SAMPLE, len(rest)))


async def minute_series(engine: AsyncEngine, market: Any, symbol: str, day: date) -> list[Any]:
    """``symbol``'s regular-session bars on ``day``, fetched once and stored."""
    from halal_trader.data import minutes

    rows = await minutes.read(engine, symbol, day)
    if not rows:
        await minutes.backfill(engine, market, [(symbol, day)])
        rows = await minutes.read(engine, symbol, day)
    return rows


def entry_and_close(rows: list[Any], at: datetime) -> tuple[float, float] | None:
    """(entry VWAP of the first bar starting >= at, last regular-session close)."""
    later = [r for r in rows if r.ts >= at]
    if not later or not rows:
        return None
    entry = later[0].vwap or later[0].open
    return float(entry), float(rows[-1].close)


@dataclass(frozen=True, slots=True)
class Outcome:
    headline: Headline
    same_day: float
    next_day: float | None
    five_day: float | None


async def _raw_closes(engine: AsyncEngine, symbols: set[str]) -> dict[str, dict[date, float]]:
    out: dict[str, dict[date, float]] = defaultdict(dict)
    async with engine.connect() as conn:
        for r in await conn.execute(
            text(
                "SELECT symbol, day, close FROM daily_bars WHERE adjustment = 'raw' "
                "AND symbol = ANY(:s) AND day >= '2025-11-01'"
            ),
            {"s": sorted(symbols)},
        ):
            out[r.symbol][r.day] = float(r.close)
    return out


async def run(
    engine: AsyncEngine, market: Any, headlines: list[Headline], *, cost_sides: int = 2
) -> list[Outcome]:
    """Each headline's abnormal return from 60 s after it, less ``cost_sides`` one-way
    costs: 2 for an entry and its exit, 0 for an exit's gross move (exit_test)."""
    from halal_trader.data.universe import universe_at

    closes = await _raw_closes(engine, {h.symbol for h in headlines} | {"SPY"})
    sessions = sorted(closes.get("SPY", {}))
    ranks: dict[date, dict[str, int]] = {}
    out = []
    for n, h in enumerate(headlines):
        day = h.published_at.astimezone(MARKET_TZ).date()
        if day not in sessions:
            continue
        month = day.replace(day=1)
        if month not in ranks:
            ranks[month] = {
                s: i for i, s in enumerate(await universe_at(engine, month, top_n=3000))
            }
        cost = cost_sides * cost_bps(ranks[month].get(h.symbol)) / 10_000
        at = h.published_at + LATENCY
        stock = entry_and_close(await minute_series(engine, market, h.symbol, day), at)
        spy = entry_and_close(await minute_series(engine, market, "SPY", day), at)
        if stock is None or spy is None:
            continue
        (e, c), (se, sc) = stock, spy
        if not _PLAUSIBLE[0] < c / e < _PLAUSIBLE[1]:
            continue
        same = (c / e - 1) - (sc / se - 1) - cost
        i = sessions.index(day)
        later: list[float | None] = []
        for k in (1, 5):
            d = sessions[i + k] if i + k < len(sessions) else None
            p, q = closes.get(h.symbol, {}).get(d), closes["SPY"].get(d)
            if not p or not q or not _PLAUSIBLE[0] < p / e < _PLAUSIBLE[1]:
                later.append(None)
            else:
                later.append((p / e - 1) - (q / se - 1) - cost)

        out.append(Outcome(h, same, later[0], later[1]))
        if n % 250 == 0:
            logger.info("intraday study: %d/%d headlines", n, len(headlines))
    return out


@dataclass(frozen=True, slots=True)
class Bucket:
    label: str
    horizon: str
    n: int
    mean: float
    t: float


def summarise(outcomes: list[Outcome]) -> list[Bucket]:
    groups = {
        f"score >= {STRONG}": lambda s: s >= STRONG,
        "neutral (control)": lambda s: abs(s) < STRONG,
        f"score <= -{STRONG}": lambda s: s <= -STRONG,
    }
    rows = []
    for label, keep in groups.items():
        for horizon in ("same_day", "next_day", "five_day"):
            values = np.array(
                [
                    getattr(o, horizon)
                    for o in outcomes
                    if keep(o.headline.score) and getattr(o, horizon) is not None
                ]
            )
            if len(values) < 2:
                continue
            sd = float(values.std(ddof=1))
            t = float(values.mean()) / (sd / math.sqrt(len(values))) if sd > 0 else 0.0
            rows.append(Bucket(label, horizon, len(values), float(values.mean()), t))
    return rows


# ── Negative news as an exit (the reactor's third rebuilt test, pre-registered) ──
#
# A holder who sells 60 s after a strongly negative headline, rather than at
# the close, pays the same one sale's cost either way, so what decides it is
# the gross abnormal move after the exit point. Registered before any result:
# the post-cutoff window is split in two (TRAIN_UNTIL); the exit is kept only
# if, in the first half, the gross same-day move after score <= -STRONG
# headlines is negative with t <= -2 and the next day has not recovered above
# the exit point (next_day <= 0), and the second half agrees on both.

TRAIN_UNTIL = date(2026, 6, 1)


def exit_test(outcomes: list[Outcome]) -> dict[str, list[Bucket]]:
    """The exit test's buckets for the first ("train") and second ("holdout") half."""
    halves = {
        "train": [o for o in outcomes if o.headline.published_at.date() < TRAIN_UNTIL],
        "holdout": [o for o in outcomes if o.headline.published_at.date() >= TRAIN_UNTIL],
    }
    return {k: summarise(v) for k, v in halves.items()}
