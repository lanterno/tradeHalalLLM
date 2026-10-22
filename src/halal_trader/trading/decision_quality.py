"""How good the day-trader's decisions were: each closed round trip against SPY.

Profit alone cannot say whether a decision was good: on a rising day every
buy makes money. Each round trip is scored against SPY instead:

* **entry edge:** from the fill to that session's close, the name's return
  minus SPY's. Positive: the name beat the market after it was bought. This
  is the buy decision's own quality, whatever the exit did.
* **exit edge:** from the sale to that session's close, SPY's return minus
  the name's. Positive: after it was sold the name did worse than the
  market, so selling was right. A sale at the close scores about zero.
* **excess:** from the fill to the sale, the name's return minus SPY's.
* **plan:** the buy's stop and target as stored, read against the fill:
  reward at least twice the risk (the prompt's rule), below that, a stop
  at or above the fill, no target, or no stop. Entry edge by plan says
  whether the buys that kept the rule did better.

Trips in one session move together, so means and t-statistics treat each
session's mean as one draw. SPY's price at a fill or a sale is the close of
the last minute bar that ended by then (Alpaca, SIP); closes are the stored
as-traded daily closes. The evening run keeps the summary of the last 30
days in the ``stock.decisions`` heartbeat for the Decisions page.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.market_hours import MARKET_TZ

BENCHMARK = "SPY"
WINDOW_DAYS = 30
BEAT = "stock.decisions"
# Entry-time buckets (New York): the open's first hour has done worst so far.
HOURS = (("09:30-10:30", time(9, 30)), ("10:30-13:00", time(10, 30)), ("13:00-16:00", time(13)))
REWARD_RISK = 2.0  # the prompt's rule 3: a target at least twice as far as the stop
PLANS = ("2:1 or better", "below 2:1", "stop at or above fill", "no target", "no stop")


class MinuteSource(Protocol):
    async def minute_bars(
        self, symbol: str, *, start: datetime, end: datetime
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class Trip:
    symbol: str
    filled_at: datetime
    closed_at: datetime
    fill: float
    exit: float
    quantity: float
    exit_reason: str
    entry_type: str
    stop_loss: float | None = None
    target: float | None = None

    @property
    def session(self) -> date:
        return self.filled_at.astimezone(MARKET_TZ).date()

    @property
    def exit_session(self) -> date:
        return self.closed_at.astimezone(MARKET_TZ).date()

    @property
    def hour(self) -> str:
        at = self.filled_at.astimezone(MARKET_TZ).time()
        label = HOURS[0][0]
        for name, start in HOURS:
            if at >= start:
                label = name
        return label

    @property
    def pnl(self) -> float:
        return (self.exit - self.fill) * self.quantity

    @property
    def plan(self) -> str:
        """Which of ``PLANS`` the buy's stop and target make against its fill (a zero is
        none, as the monitor reads it)."""
        if not self.stop_loss:
            return PLANS[4]
        if not self.target:
            return PLANS[3]
        risk = self.fill - self.stop_loss
        if risk <= 0:
            return PLANS[2]
        return PLANS[0] if self.target - self.fill >= REWARD_RISK * risk else PLANS[1]


@dataclass(frozen=True, slots=True)
class Score:
    trip: Trip
    entry_edge: float
    exit_edge: float
    excess: float


@dataclass(frozen=True, slots=True)
class Stat:
    trips: int
    sessions: int
    mean: float | None
    t: float | None

    def as_dict(self) -> dict[str, object]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(x, 6)

        return {
            "trips": self.trips,
            "sessions": self.sessions,
            "mean": r(self.mean),
            "t": r(self.t),
        }


def clustered(pairs: Iterable[tuple[date, float]]) -> Stat:
    """Mean of session means and its t across sessions."""
    by_day: dict[date, list[float]] = defaultdict(list)
    n = 0
    for day, value in pairs:
        by_day[day].append(value)
        n += 1
    means = [sum(v) / len(v) for v in by_day.values()]
    if not means:
        return Stat(0, 0, None, None)
    mu = sum(means) / len(means)
    if len(means) < 2:
        return Stat(n, 1, mu, None)
    sd = math.sqrt(sum((m - mu) ** 2 for m in means) / (len(means) - 1))
    return Stat(n, len(means), mu, mu / (sd / math.sqrt(len(means))) if sd > 0 else None)


def price_at(bars: Sequence[tuple[datetime, float]], ts: datetime) -> float | None:
    """Close of the last minute bar that ended by ``ts`` (bars sorted by start)."""
    i = bisect_right(bars, ts - timedelta(minutes=1), key=lambda b: b[0])
    return bars[i - 1][1] if i else None


def score(
    trips: Iterable[Trip],
    closes: Mapping[tuple[str, date], float],
    spy: Sequence[tuple[datetime, float]],
) -> list[Score]:
    """Every trip whose prices are all known, scored against SPY."""
    out: list[Score] = []
    for t in trips:
        if (
            (close_in := closes.get((t.symbol, t.session))) is None
            or (close_out := closes.get((t.symbol, t.exit_session))) is None
            or (spy_close_in := closes.get((BENCHMARK, t.session))) is None
            or (spy_close_out := closes.get((BENCHMARK, t.exit_session))) is None
            or (spy_in := price_at(spy, t.filled_at)) is None
            or (spy_out := price_at(spy, t.closed_at)) is None
        ):
            continue
        out.append(
            Score(
                t,
                (close_in / t.fill - 1) - (spy_close_in / spy_in - 1),
                (spy_close_out / spy_out - 1) - (close_out / t.exit - 1),
                (t.exit / t.fill - 1) - (spy_out / spy_in - 1),
            )
        )
    return out


@dataclass
class Report:
    start: date
    end: date
    scores: list[Score] = field(default_factory=list)
    unscored: int = 0

    def summary(self) -> dict[str, object]:
        s = self.scores
        sells = [x for x in s if x.trip.exit_reason != "eod_close_all"]
        by_reason: dict[str, list[Score]] = defaultdict(list)
        for x in sells:
            by_reason[x.trip.exit_reason].append(x)
        days = sorted({x.trip.session for x in s})
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "trips": len(s),
            "unscored": self.unscored,
            "pnl_usd": round(sum(x.trip.pnl for x in s), 2),
            "entry_edge": clustered((x.trip.session, x.entry_edge) for x in s).as_dict(),
            "exit_edge": clustered((x.trip.session, x.exit_edge) for x in sells).as_dict(),
            "excess": clustered((x.trip.session, x.excess) for x in s).as_dict(),
            "by_hour": {
                name: clustered(
                    (x.trip.session, x.entry_edge) for x in s if x.trip.hour == name
                ).as_dict()
                for name, _ in HOURS
            },
            "by_exit": {
                reason: clustered((x.trip.session, x.exit_edge) for x in xs).as_dict()
                for reason, xs in sorted(by_reason.items())
            },
            # A list, in PLANS order: the heartbeat's JSONB reorders an object's keys.
            "by_plan": [
                {
                    "plan": plan,
                    **clustered(
                        (x.trip.session, x.entry_edge) for x in s if x.trip.plan == plan
                    ).as_dict(),
                }
                for plan in PLANS
            ],
            "days": [
                {
                    "day": d.isoformat(),
                    "trips": sum(1 for x in s if x.trip.session == d),
                    "entry_edge": clustered(
                        (d, x.entry_edge) for x in s if x.trip.session == d
                    ).as_dict()["mean"],
                    "pnl_usd": round(sum(x.trip.pnl for x in s if x.trip.session == d), 2),
                }
                for d in days
            ],
        }


def _bounds(start: date, end: date) -> tuple[datetime, datetime]:
    lo = datetime.combine(start, time(), MARKET_TZ).astimezone(UTC)
    hi = datetime.combine(end + timedelta(days=1), time(), MARKET_TZ).astimezone(UTC)
    return lo, hi


async def closed_trips(engine: AsyncEngine, start: date, end: date) -> list[Trip]:
    """The day-trader's round trips bought in [start, end] (New York dates) and closed."""
    lo, hi = _bounds(start, end)
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT symbol, filled_at, closed_at, filled_price, exit_price, "
                "coalesce(filled_quantity, quantity) AS qty, coalesce(exit_reason, '') AS why, "
                "coalesce(entry_type, 'cycle') AS kind, stop_loss, target_price FROM trades "
                "WHERE side = 'buy' AND filled_price > 0 AND exit_price > 0 "
                "AND filled_at >= :lo AND filled_at < :hi AND closed_at IS NOT NULL "
                "ORDER BY filled_at"
            ),
            {"lo": lo, "hi": hi},
        )
        return [
            Trip(
                r.symbol,
                r.filled_at,
                r.closed_at,
                float(r.filled_price),
                float(r.exit_price),
                float(r.qty),
                r.why,
                r.kind,
                None if r.stop_loss is None else float(r.stop_loss),
                None if r.target_price is None else float(r.target_price),
            )
            for r in rows
        ]


async def _closes(
    engine: AsyncEngine, symbols: Sequence[str], start: date, end: date
) -> dict[tuple[str, date], float]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT symbol, day, close FROM daily_bars WHERE adjustment = 'raw' "
                "AND symbol = ANY(:s) AND day BETWEEN :a AND :b AND close > 0"
            ),
            {"s": list(symbols), "a": start, "b": end},
        )
        return {(r.symbol, r.day): float(r.close) for r in rows}


async def report(
    engine: AsyncEngine, market: MinuteSource, today: date, *, days: int = WINDOW_DAYS
) -> Report:
    """The round trips bought in the ``days`` calendar days through ``today``, scored."""
    start = today - timedelta(days=days - 1)
    trips = await closed_trips(engine, start, today)
    if not trips:
        return Report(start, today)
    last = max(t.exit_session for t in trips)
    closes = await _closes(engine, sorted({t.symbol for t in trips} | {BENCHMARK}), start, last)
    lo, hi = _bounds(start, last)
    raw = await market.minute_bars(BENCHMARK, start=lo, end=hi)
    spy = sorted(
        (datetime.fromisoformat(str(b["t"]).replace("Z", "+00:00")), float(b["c"])) for b in raw
    )
    scores = score(trips, closes, spy)
    return Report(start, today, scores, len(trips) - len(scores))
