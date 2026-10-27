"""Which of the day-trader's buys would three rules have kept? A replay of its scored trips.
(pre-registered 2026-10-27, HAL-15)

The weak point is the buy: over the 147 round trips bought 2026-10-08..26,
the decision scorecard's entry edge averaged -0.22% a trip (t -1.23), while
the exit edge was +0.22% (t +1.82). Each rule below says which buys to skip,
from data known before the buy; the replay asks whether the skipped ones
were the worse ones. Nothing here trades.

Pre-registered (written before any result was seen):

* **trips:** the scorecard's (trading/decision_quality.py): the day-trader's
  buys filled 2026-10-08..2026-10-26 (New York) and closed, each with its
  entry edge (the name's return minus SPY's, from the fill to the close);
* **hourly bars:** clock hours of the regular session (09:30-16:00 New York;
  the first is 09:30-10:00), each closing at its last SIP 15-minute bar's
  close; a bar counts once it has ended by the fill;
* **rules,** each dropping a buy when:

  - ``market_trend``: SPY's last hourly close is below the mean of its last
    ``SMA_BARS`` (the shadow engine's regime gate, ``sma_trend_state`` at
    ``market_sma_window`` 50);
  - ``strength``: the name's return over its last ``STRENGTH_BARS`` hourly
    closes is below SPY's over its own (the shadow engine's
    ``RelativeStrengthInterpreter``: ``closes[-1] / closes[-20] - 1``);
  - ``confidence``: the model's 0-1 confidence in the buy is below the median
    of every matched buy's. A buy's confidence is the one in the latest
    decision recorded at most ``MATCH_WINDOW`` before its trade row that buys
    the same symbol with the same reasoning;

  a trip a rule can't judge (too few bars, no decision found) is left out
  of that rule and counted;
* **statistic:** the kept and the dropped trips' entry edge, each the mean
  of session means; the difference dropped - kept, its t by Welch over the
  two groups' session means;
* **rule:** *pass* (a candidate for a forward test after the 11-02 review)
  if the difference is below zero with t <= -2; otherwise *no pass*. With 13
  sessions a pass points at a candidate, nothing more.
"""

from __future__ import annotations

import json
import math
import statistics
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.market_hours import MARKET_TZ
from halal_trader.trading import decision_quality as dq

START = date(2026, 10, 8)
END = date(2026, 10, 26)
WARM_UP = timedelta(days=21)  # calendar days of bars before START: 50 hourly bars and more
SMA_BARS = 50
STRENGTH_BARS = 20
MATCH_WINDOW = timedelta(minutes=30)
T_PASS = -2.0
RULES = ("market_trend", "strength", "confidence")
TRIAL = "research.day_trader.replay"
_OPEN, _CLOSE = time(9, 30), time(16)


class IntradayBars(Protocol):
    async def minute_bars_many(
        self, symbols: Iterable[str], *, start: datetime, end: datetime, timeframe: Any
    ) -> dict[str, list[dict[str, Any]]]: ...


Bar = tuple[datetime, float]  # an hourly bar: its end and its close
Hourly = Sequence[Bar]  # oldest first


@dataclass(frozen=True, slots=True)
class Split:
    rule: str
    kept: dq.Stat
    dropped: dq.Stat
    difference: float | None  # dropped - kept
    t: float | None  # Welch, over the two groups' session means
    unjudged: int

    @property
    def verdict(self) -> str:
        if self.difference is not None and self.t is not None:
            if self.difference < 0 and self.t <= T_PASS:
                return "pass"
        return "no pass"

    def as_dict(self) -> dict[str, object]:
        return {
            "kept": self.kept.as_dict(),
            "dropped": self.dropped.as_dict(),
            "difference": None if self.difference is None else round(self.difference, 6),
            "t": None if self.t is None else round(self.t, 6),
            "unjudged": self.unjudged,
        }


def hourly_closes(raw: Iterable[Mapping[str, object]]) -> list[Bar]:
    """Alpaca's 15-minute bars of one symbol as regular-session clock-hour bars."""
    buckets: dict[tuple[date, int], tuple[datetime, float]] = {}
    for b in raw:
        start = datetime.fromisoformat(str(b["t"]).replace("Z", "+00:00"))
        local = start.astimezone(MARKET_TZ)
        if not _OPEN <= local.time() < _CLOSE:
            continue
        key = (local.date(), local.hour)
        if key not in buckets or buckets[key][0] < start:
            buckets[key] = (start, float(str(b["c"])))
    out = []
    for (day, hour), (_, close) in buckets.items():
        end = datetime.combine(day, time(hour), MARKET_TZ) + timedelta(hours=1)
        out.append((end.astimezone(UTC), close))
    return sorted(out)


def completed(bars: Hourly, at: datetime) -> list[float]:
    """The closes of the hourly bars that had ended by ``at``."""
    return [c for _, c in bars[: bisect_right(bars, at, key=lambda b: b[0])]]


def below_trend(spy: Hourly, at: datetime) -> bool | None:
    closes = completed(spy, at)
    if len(closes) < SMA_BARS:
        return None
    return closes[-1] < sum(closes[-SMA_BARS:]) / SMA_BARS


def strength(name: Hourly, spy: Hourly, at: datetime) -> float | None:
    """The name's return minus SPY's over each one's last ``STRENGTH_BARS`` closes."""
    a, b = completed(name, at), completed(spy, at)
    if len(a) < STRENGTH_BARS or len(b) < STRENGTH_BARS:
        return None
    if a[-STRENGTH_BARS] <= 0 or b[-STRENGTH_BARS] <= 0:
        return None
    return (a[-1] / a[-STRENGTH_BARS] - 1) - (b[-1] / b[-STRENGTH_BARS] - 1)


@dataclass(frozen=True, slots=True)
class Decision:
    at: datetime
    symbol: str
    reasoning: str
    confidence: float


def buys_in(at: datetime, raw_response: str | None) -> list[Decision]:
    """The buys of one recorded plan; a response that isn't a plan has none."""
    try:
        plan = json.loads(raw_response or "")
    except ValueError:
        return []
    if not isinstance(plan, dict) or not isinstance(plan.get("decisions"), list):
        return []
    out = []
    for d in plan["decisions"]:
        if not isinstance(d, dict) or d.get("action") != "buy":
            continue
        try:
            confidence = float(d["confidence"])
        except KeyError, TypeError, ValueError:
            continue
        out.append(
            Decision(at, str(d.get("symbol", "")).upper(), str(d.get("reasoning")), confidence)
        )
    return out


def confidence_of(
    symbol: str, reasoning: str | None, placed_at: datetime, decisions: Sequence[Decision]
) -> float | None:
    """The confidence of the latest matching buy recorded within ``MATCH_WINDOW`` before."""
    found = [
        d
        for d in decisions
        if d.symbol == symbol
        and d.reasoning == reasoning
        and placed_at - MATCH_WINDOW <= d.at <= placed_at
    ]
    return max(found, key=lambda d: d.at).confidence if found else None


def welch(kept: Sequence[float], dropped: Sequence[float]) -> tuple[float | None, float | None]:
    """mean(dropped) - mean(kept) and its Welch t (None with under two of either)."""
    if not kept or not dropped:
        return None, None
    diff = statistics.fmean(dropped) - statistics.fmean(kept)
    if len(kept) < 2 or len(dropped) < 2:
        return diff, None
    se = math.sqrt(
        statistics.variance(dropped) / len(dropped) + statistics.variance(kept) / len(kept)
    )
    return diff, (diff / se if se > 0 else None)


def split(rule: str, judged: Iterable[tuple[dq.Score, bool | None]]) -> Split:
    """Kept and dropped entry edge for one rule (``True``: the rule drops the buy)."""
    kept: list[tuple[date, float]] = []
    dropped: list[tuple[date, float]] = []
    unjudged = 0
    for s, drop in judged:
        if drop is None:
            unjudged += 1
        else:
            (dropped if drop else kept).append((s.trip.session, s.entry_edge))
    diff, t = welch(_session_means(kept), _session_means(dropped))
    return Split(rule, dq.clustered(kept), dq.clustered(dropped), diff, t, unjudged)


def _session_means(pairs: Iterable[tuple[date, float]]) -> list[float]:
    by_day: dict[date, list[float]] = defaultdict(list)
    for day, value in pairs:
        by_day[day].append(value)
    return [statistics.fmean(v) for v in by_day.values()]


def _below(x: float | None, bar: float | None) -> bool | None:
    return None if x is None or bar is None else x < bar


def judge(
    scores: Sequence[dq.Score],
    hourly: Mapping[str, Hourly],
    confidences: Mapping[tuple[str, datetime], float | None],
) -> list[Split]:
    """The three rules over the scored trips."""
    spy = hourly.get(dq.BENCHMARK, ())
    rel = [strength(hourly.get(s.trip.symbol, ()), spy, s.trip.filled_at) for s in scores]
    conf = [confidences.get((s.trip.symbol, s.trip.filled_at)) for s in scores]
    known = [c for c in conf if c is not None]
    median = statistics.median(known) if known else None
    return [
        split("market_trend", [(s, below_trend(spy, s.trip.filled_at)) for s in scores]),
        split("strength", [(s, _below(r, 0.0)) for s, r in zip(scores, rel, strict=True)]),
        split("confidence", [(s, _below(c, median)) for s, c in zip(scores, conf, strict=True)]),
    ]


WINDOW = f"trips bought {START}..{END} (New York), as the decision scorecard scores them"
CRITERIA = {
    "market_trend": (
        f"drop a buy when SPY's last regular-session hourly close is below the mean of its last "
        f"{SMA_BARS}; pass if the dropped trips' entry edge (mean of session means) is below "
        f"the kept ones' with Welch t <= {T_PASS:g} over the two groups' session means"
    ),
    "strength": (
        f"drop a buy when the name's return over its last {STRENGTH_BARS} regular-session hourly "
        f"closes is below SPY's; pass if the dropped trips' entry edge (mean of session means) "
        f"is below the kept ones' with Welch t <= {T_PASS:g} over the two groups' session means"
    ),
    "confidence": (
        "drop a buy whose model confidence is below the median of the matched buys'; pass if "
        "the dropped trips' entry edge (mean of session means) is below the kept ones' with "
        f"Welch t <= {T_PASS:g} over the two groups' session means"
    ),
}


class AlreadyJudged(RuntimeError):
    """A rule's result is on the ledger: the replay runs once."""


def trial_name(rule: str) -> str:
    return f"{TRIAL}.{rule}"


def config(rule: str) -> dict[str, object]:
    common: dict[str, object] = {"start": START.isoformat(), "end": END.isoformat()}
    if rule == "market_trend":
        return {**common, "sma_bars": SMA_BARS, "bars": "1Hour regular session, from 15Min sip"}
    if rule == "strength":
        return {
            **common,
            "strength_bars": STRENGTH_BARS,
            "bars": "1Hour regular session, from 15Min sip",
        }
    return {**common, "match_window_min": MATCH_WINDOW.total_seconds() / 60, "split": "median"}


async def register(engine: AsyncEngine) -> dict[str, int]:
    """Each rule's registration row, recorded now unless it is already: before any data is read.

    Raises ``AlreadyJudged`` when a rule's result is recorded already.
    """
    from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl

    repo = QuantTrialRepoImpl(engine)
    ids: dict[str, int] = {}
    for rule in RULES:
        rows = [
            r
            for r in await repo.get_trials(name_prefix=trial_name(rule))
            if r["name"] == trial_name(rule)
        ]
        if judged := [r for r in rows if r["kind"] == "verdict"]:
            raise AlreadyJudged(
                f"{trial_name(rule)} is judged already (quant_trials {judged[0]['id']})"
            )
        if registered := [r for r in rows if r["kind"] == "registration"]:
            ids[rule] = registered[0]["id"]
            continue
        ids[rule] = await repo.record_trial(
            name=trial_name(rule),
            kind="registration",
            config=config(rule),
            window=WINDOW,
            criterion=CRITERIA[rule],
            verdict="registered",
        )
    return ids


async def record(
    engine: AsyncEngine, report: dq.Report, splits: Sequence[Split], ids: Mapping[str, int]
) -> None:
    """Each rule's result, after its registration."""
    from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl

    repo = QuantTrialRepoImpl(engine)
    for sp in splits:
        await repo.record_trial(
            name=trial_name(sp.rule),
            kind="verdict",
            config=config(sp.rule),
            window=WINDOW,
            metrics={
                **sp.as_dict(),
                "registration": ids[sp.rule],
                "trips": len(report.scores),
                "unscored": report.unscored,
            },
            criterion=CRITERIA[sp.rule],
            verdict=sp.verdict,
        )


async def _confidences(engine: AsyncEngine) -> dict[tuple[str, datetime], float | None]:
    """Each buy's confidence, keyed as its trip is (symbol, fill time)."""
    lo, hi = dq.session_bounds(START, END)
    async with engine.connect() as conn:
        trades = (
            await conn.execute(
                text(
                    "SELECT symbol, filled_at, timestamp, llm_reasoning FROM trades "
                    "WHERE side = 'buy' AND filled_at >= :lo AND filled_at < :hi"
                ),
                {"lo": lo, "hi": hi},
            )
        ).all()
        rows = await conn.execute(
            text(
                "SELECT timestamp, raw_response FROM llm_decisions "
                "WHERE timestamp >= :lo AND timestamp < :hi"
            ),
            {"lo": lo - MATCH_WINDOW, "hi": hi},
        )
        decisions = [d for r in rows for d in buys_in(r.timestamp, r.raw_response)]
    return {
        (t.symbol, t.filled_at): confidence_of(t.symbol, t.llm_reasoning, t.timestamp, decisions)
        for t in trades
    }


async def hourly_bars(bars: IntradayBars, symbols: Iterable[str]) -> dict[str, list[Bar]]:
    """Each symbol's hourly bars, SPY's included, from WARM_UP before START through END."""
    lo = datetime.combine(START - WARM_UP, time(), UTC)
    hi = datetime.combine(END + timedelta(days=1), time(), UTC)
    raw = await bars.minute_bars_many(
        sorted(set(symbols) | {dq.BENCHMARK}), start=lo, end=hi, timeframe="15Min"
    )
    return {symbol: hourly_closes(rows) for symbol, rows in raw.items()}


async def run(engine: AsyncEngine, market: Any) -> tuple[dq.Report, list[Split]]:
    """Score the trips, fetch their hourly bars, read their confidences, judge the rules."""
    report = await dq.report(engine, market, END, days=(END - START).days + 1)
    hourly = await hourly_bars(market, {s.trip.symbol for s in report.scores})
    return report, judge(report.scores, hourly, await _confidences(engine))
