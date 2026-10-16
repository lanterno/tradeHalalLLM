"""Does the day-trader's entry rule work? Buying intraday momentum, tested on 15-minute bars.
(pre-registered 2026-10-16)

The day-trader's prompt (trading/strategy.py) tells GLM to find one to four
entries every cycle and names its setup: "intraday momentum (>0.5% with
volume confirmation)". Its first six server days (2026-10-08..15, 104 round
trips) look like the opposite of an edge: the names it bought trailed SPY by
0.23% from the fill to the close (t -2.1), and the first hour's buys did
worst. Six days prove nothing, and they are the days the idea was spotted
in, so this tests the rule itself, mechanically, where it was not seen.

Pre-registered (written before any result was seen):

* **universe:** each session, the 20 largest names (by market cap) that the
  newest point-in-time screen dated before the session holds halal, the
  live cycle's rule (``halal_universe``, ``_MAX_SYMBOLS_PER_CYCLE``);
* **bars:** Alpaca SIP 15-minute bars, raw prices, regular session only; a
  (symbol, session) without all 26 bars is skipped, and so is every session
  where SPY lacks one (early closes);
* **decision times:** the cycle's, 09:45 to 15:30 ET. At time T the day's
  move is the last price before T over the session's open, the relative
  volume is the volume traded before T over its mean at the same time of
  day across the symbol's previous 20 sessions (20 needed);
* **groups:** CHASE (move > +0.5% and relative volume > 1: the prompt's
  setup), DIP (move < -0.5%), ALL (every symbol at every decision time);
* **trade:** enter at T (the open of the bar starting at T), exit at the
  close (the day-trader flattens at the end of the day). The outcome is the
  return minus SPY's over the same span, minus ``COST`` for the round trip;
* **statistic:** observations within a session move together, so each
  session's mean is one draw; the mean and t are taken across sessions;
* **rule, CHASE:** trained on 2016-10..2021, then held out on 2022-2024.
  *Refuted* if the net excess is negative with t <= -2 in both windows;
  *supported* if positive with t >= 2 in both; otherwise inconclusive.
  2025-2026 (which holds the six days that raised the question) stays
  untouched. DIP and ALL are reported, not judged.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.market_hours import MARKET_TZ

BARS = 26  # 15-minute bars from 09:30 to 16:00
BAR_MINUTES = 15
FIRST_DECISION, LAST_DECISION = 1, 24  # bar index of T: 09:45 .. 15:30
VOLUME_LOOKBACK = 20
MOVE = 0.005
COST = 0.0010  # round trip on a megacap: half the prompt's own 0.1-0.3% estimate
UNIVERSE = 20
TRAIN = (date(2016, 10, 1), date(2021, 12, 31))
HOLDOUT = (date(2022, 1, 1), date(2024, 12, 31))
T_BAR = 2.0
GROUPS = ("CHASE", "DIP", "ALL")
BENCHMARK = "SPY"
_OPEN = time(9, 30)
WARM_UP = timedelta(days=45)  # calendar days of bars before the first session: 20+ sessions
CHUNK = timedelta(days=92)  # bars fetched a quarter at a time (extended hours included)


class IntradayBars(Protocol):
    async def minute_bars_many(
        self, symbols: Iterable[str], *, start: datetime, end: datetime, timeframe: Any
    ) -> dict[str, list[dict[str, Any]]]: ...


@dataclass(frozen=True, slots=True)
class Session:
    """One symbol's regular session in 15-minute bars (index 0 = 09:30)."""

    open: tuple[float, ...]
    close: tuple[float, ...]
    cumvol: tuple[float, ...]  # volume through the end of each bar


@dataclass(frozen=True, slots=True)
class Observation:
    day: date
    bar: int  # index of the bar starting at T
    symbol: str
    move: float
    rvol: float
    excess: float  # gross, vs SPY, entry at T to the close


@dataclass(frozen=True, slots=True)
class Summary:
    group: str
    observations: int
    sessions: int
    mean: float  # net, the mean of session means
    t: float


def bar_index(ts: datetime) -> int | None:
    """Which regular-session 15-minute bar a bar start falls in, if any."""
    local = ts.astimezone(MARKET_TZ)
    minutes = (local.hour - _OPEN.hour) * 60 + local.minute - _OPEN.minute
    if minutes < 0 or minutes % BAR_MINUTES:
        return None
    k = minutes // BAR_MINUTES
    return k if k < BARS else None


def sessions_of(raw: Iterable[Mapping[str, object]]) -> dict[date, Session]:
    """Alpaca's raw bars of one symbol, as complete sessions by New York date."""
    by_day: dict[date, dict[int, tuple[float, float, float]]] = defaultdict(dict)
    for b in raw:
        ts = datetime.fromisoformat(str(b["t"]).replace("Z", "+00:00"))
        k = bar_index(ts)
        if k is None:
            continue
        by_day[ts.astimezone(MARKET_TZ).date()][k] = (
            float(str(b["o"])),
            float(str(b["c"])),
            float(str(b["v"])),
        )
    out: dict[date, Session] = {}
    for day, bars in by_day.items():
        if len(bars) != BARS:
            continue
        opens, closes, cum = [], [], []
        total = 0.0
        for k in range(BARS):
            o, c, v = bars[k]
            total += v
            opens.append(o)
            closes.append(c)
            cum.append(total)
        out[day] = Session(tuple(opens), tuple(closes), tuple(cum))
    return out


def universes(
    screens: Mapping[date, Sequence[tuple[str, float | None]]],
    days: Iterable[date],
    *,
    limit: int = UNIVERSE,
) -> dict[date, list[str]]:
    """Each day's ``limit`` largest halal names on the newest screen dated before it.

    ``screens`` holds each screen's halal (symbol, market cap) pairs.
    """
    dated = sorted(screens)
    out: dict[date, list[str]] = {}
    for day in days:
        known = [d for d in dated if d < day]
        if not known:
            continue
        rows = sorted(screens[known[-1]], key=lambda r: (r[1] is None, -(r[1] or 0.0), r[0]))
        out[day] = [s for s, _ in rows[:limit]]
    return out


def observations(
    bars: Mapping[str, Mapping[date, Session]],
    universe: Mapping[date, Sequence[str]],
) -> list[Observation]:
    """Every (session, decision time, universe symbol) with a volume baseline."""
    spy = bars.get(BENCHMARK, {})
    out: list[Observation] = []
    for symbol, sessions in bars.items():
        if symbol == BENCHMARK:
            continue
        history: deque[Session] = deque(maxlen=VOLUME_LOOKBACK)
        for day in sorted(sessions):
            s = sessions[day]
            bench = spy.get(day)
            if (
                bench is not None
                and len(history) == VOLUME_LOOKBACK
                and symbol in universe.get(day, ())
            ):
                for k in range(FIRST_DECISION, LAST_DECISION + 1):
                    base = sum(h.cumvol[k - 1] for h in history) / VOLUME_LOOKBACK
                    if base <= 0 or s.open[0] <= 0 or s.open[k] <= 0 or bench.open[k] <= 0:
                        continue
                    out.append(
                        Observation(
                            day,
                            k,
                            symbol,
                            s.close[k - 1] / s.open[0] - 1,
                            s.cumvol[k - 1] / base,
                            (s.close[-1] / s.open[k] - 1) - (bench.close[-1] / bench.open[k] - 1),
                        )
                    )
            history.append(s)
    return out


def in_group(o: Observation, group: str) -> bool:
    if group == "CHASE":
        return o.move > MOVE and o.rvol > 1.0
    if group == "DIP":
        return o.move < -MOVE
    return group == "ALL"


def summarise(obs: Iterable[Observation], group: str, *, cost: float = COST) -> Summary:
    """Net excess of one group: the mean of session means, and its t across sessions."""
    per_day: dict[date, list[float]] = defaultdict(list)
    n = 0
    for o in obs:
        if in_group(o, group):
            per_day[o.day].append(o.excess - cost)
            n += 1
    means = [sum(v) / len(v) for v in per_day.values()]
    if len(means) < 2:
        return Summary(group, n, len(means), means[0] if means else math.nan, math.nan)
    mu = sum(means) / len(means)
    sd = math.sqrt(sum((m - mu) ** 2 for m in means) / (len(means) - 1))
    t = mu / (sd / math.sqrt(len(means))) if sd > 0 else math.nan
    return Summary(group, n, len(means), mu, t)


def window(obs: Iterable[Observation], span: tuple[date, date]) -> list[Observation]:
    return [o for o in obs if span[0] <= o.day <= span[1]]


def verdict(train: Summary, holdout: Summary) -> str:
    """The pre-registered rule on CHASE."""
    if all(s.mean < 0 and s.t <= -T_BAR for s in (train, holdout)):
        return "refuted"
    if all(s.mean > 0 and s.t >= T_BAR for s in (train, holdout)):
        return "supported"
    return "inconclusive"


async def load(
    engine: AsyncEngine, market: IntradayBars, *, start: date, end: date
) -> tuple[dict[str, dict[date, Session]], dict[date, list[str]]]:
    """Every session's universe in [start, end] and the 15-minute bars it needs."""
    from halal_trader.halal.strict import all_screens

    screens = {
        d: [(r.symbol, r.market_cap) for r in rows if r.verdict == "halal"]
        for d, rows in (await all_screens(engine)).items()
    }
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    universe = universes(screens, (d for d in days if d.weekday() < 5))
    symbols = sorted({s for names in universe.values() for s in names} | {BENCHMARK})
    bars: dict[str, dict[date, Session]] = {s: {} for s in symbols}
    lo = datetime.combine(start - WARM_UP, time(), UTC)
    hi = datetime.combine(end + timedelta(days=1), time(), UTC)
    while lo < hi:
        raw = await market.minute_bars_many(
            symbols, start=lo, end=min(lo + CHUNK, hi), timeframe="15Min"
        )
        for symbol, rows in raw.items():
            bars.setdefault(symbol, {}).update(sessions_of(rows))
        lo += CHUNK
    return bars, universe
