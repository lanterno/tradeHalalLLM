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

Minute bars are stored in ``minute_bars`` (raw, regular session) and
fetched once per (symbol, day).
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.study import cost_bps

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
LATENCY = timedelta(seconds=60)
SESSION = (time(9, 30), time(15, 30))
STRONG = 0.4
CONTROL_SAMPLE = 1500
_PLAUSIBLE = (0.5, 2.0)


@dataclass(frozen=True, slots=True)
class Headline:
    symbol: str
    published_at: datetime
    score: float


async def first_in_session(
    engine: AsyncEngine, scorer_prefix: str = "llm-batch:"
) -> list[Headline]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT e.symbol, e.published_at, s.score FROM event_scores s "
                "JOIN events e ON e.id = s.event_id WHERE s.scorer LIKE :p AND e.kind = 'news'"
            ),
            {"p": scorer_prefix + "%"},
        )
        first: dict[tuple[str, date], Headline] = {}
        for r in rows:
            local = r.published_at.astimezone(_ET)
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


async def _stored(engine: AsyncEngine, symbol: str, day: date) -> list[Any]:
    lo = datetime.combine(day, time(9, 30), _ET)
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT ts, close, vwap, open FROM minute_bars WHERE symbol = :s "
                    "AND ts >= :lo AND ts < :hi ORDER BY ts"
                ),
                {"s": symbol, "lo": lo, "hi": lo + timedelta(hours=6, minutes=30)},
            )
        ).all()


async def minute_series(engine: AsyncEngine, market: Any, symbol: str, day: date) -> list[Any]:
    rows = await _stored(engine, symbol, day)
    if rows:
        return rows
    lo = datetime.combine(day, time(9, 30), _ET)
    raw = await market.minute_bars(symbol, start=lo, end=lo + timedelta(hours=6, minutes=30))
    if raw:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO minute_bars (symbol, ts, open, high, low, close, volume, vwap) "
                    "VALUES (:s, :t, :o, :h, :l, :c, :v, :vw) ON CONFLICT DO NOTHING"
                ),
                [
                    {
                        "s": symbol,
                        "t": datetime.fromisoformat(b["t"].replace("Z", "+00:00")),
                        "o": b["o"],
                        "h": b["h"],
                        "l": b["l"],
                        "c": b["c"],
                        "v": b["v"],
                        "vw": b.get("vw"),
                    }
                    for b in raw
                ],
            )
    return await _stored(engine, symbol, day)


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


async def run(engine: AsyncEngine, market: Any, headlines: list[Headline]) -> list[Outcome]:
    from halal_trader.data.universe import universe_at

    closes = await _raw_closes(engine, {h.symbol for h in headlines} | {"SPY"})
    sessions = sorted(closes.get("SPY", {}))
    ranks: dict[date, dict[str, int]] = {}
    out = []
    for n, h in enumerate(headlines):
        day = h.published_at.astimezone(_ET).date()
        if day not in sessions:
            continue
        month = day.replace(day=1)
        if month not in ranks:
            ranks[month] = {
                s: i for i, s in enumerate(await universe_at(engine, month, top_n=3000))
            }
        cost = 2 * cost_bps(ranks[month].get(h.symbol)) / 10_000
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

        def later(k: int) -> float | None:
            if i + k >= len(sessions):
                return None
            d = sessions[i + k]
            p, q = closes.get(h.symbol, {}).get(d), closes["SPY"].get(d)
            if not p or not q or not _PLAUSIBLE[0] < p / e < _PLAUSIBLE[1]:
                return None
            return (p / e - 1) - (q / se - 1) - cost

        out.append(Outcome(h, same, later(1), later(5)))
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
