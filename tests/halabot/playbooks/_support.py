"""Synthetic sessions, stories, context and a toy playbook for the simulator tests.

The toy playbook buys on the first visible bar at or after open + 20 min
(or on the bar starting at ``entry_bar``), sells when a bar closes at its
target (in S units), aborts on a structural item, and finishes on its exit
fill. ``bounce.py`` is another workstream; the toy only exercises the
simulator.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta

import numpy as np

from halabot.playbooks.clock import from_us
from halabot.playbooks.playbook import TRIGGERED, Ctx
from halabot.playbooks.records import StoryOutcome
from halabot.playbooks.sim import SymbolState, simulate_symbol
from halabot.playbooks.types import (
    BarIn,
    ComplianceIn,
    FillIn,
    Finish,
    Input,
    Intent,
    NewsIn,
    OrderClosedIn,
    PathData,
    PathSkip,
    Session,
    SessionIn,
    SimConfig,
    SpyData,
    Submit,
    TradeFacts,
    Transition,
)
from halal_trader.data.minutes import BarArrays
from halal_trader.market_hours import MARKET_TZ, effective_close_time

MON = date(2016, 3, 7)
TUE = date(2016, 3, 8)
WED = date(2016, 3, 9)
THU = date(2016, 3, 10)
FRI = date(2016, 3, 11)
HALF = date(2016, 11, 25)  # closes at 13:00
NEWS_LAG = timedelta(seconds=600)

Row = tuple[float, float, float, float, float, float]  # o, h, l, c, v, vw


def et(day: date, hh: int, mm: int, ss: int = 0) -> datetime:
    """A New York wall-clock time on ``day``, as UTC."""
    return datetime.combine(day, time(hh, mm, ss), MARKET_TZ).astimezone(UTC)


def epoch(day: date, hh: int, mm: int) -> int:
    return int(et(day, hh, mm).timestamp())


def minutes_of(day: date) -> list[tuple[int, int]]:
    """Every regular-session minute of ``day`` as (hour, minute)."""
    close = effective_close_time(day)
    out = []
    t = datetime.combine(day, time(9, 30))
    end = datetime.combine(day, close)
    while t < end:
        out.append((t.hour, t.minute))
        t += timedelta(minutes=1)
    return out


def session_bars(
    day: date,
    *,
    price: float = 100.0,
    rows: Mapping[tuple[int, int], Row] | None = None,
    skip: Iterable[tuple[int, int]] = (),
    first: tuple[int, int] = (9, 30),
    last: tuple[int, int] | None = None,
    volume: float = 1_000.0,
) -> BarArrays:
    """One bar a minute in [first, last] (default: the whole session), flat at ``price``.

    ``rows`` overrides single minutes with (o, h, l, c, v, vw); ``skip``
    leaves minutes out (a gap).
    """
    rows = rows or {}
    gone = set(skip)
    keep = [
        m for m in minutes_of(day) if m >= first and (last is None or m <= last) and m not in gone
    ]
    ts, o, h, low, c, v, vw = [], [], [], [], [], [], []
    for hh, mm in keep:
        r = rows.get((hh, mm), (price, price, price, price, volume, price))
        ts.append(epoch(day, hh, mm))
        o.append(r[0])
        h.append(r[1])
        low.append(r[2])
        c.append(r[3])
        v.append(r[4])
        vw.append(r[5])
    return BarArrays(
        np.asarray(ts, dtype=np.int64),
        np.asarray(o, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
        np.asarray(low, dtype=np.float64),
        np.asarray(c, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
        np.asarray(vw, dtype=np.float64),
    )


def spy_data(days: Iterable[date], *, price: float = 200.0, **per_day: BarArrays) -> SpyData:
    """Flat SPY sessions (every minute), with ``per_day`` overrides keyed ``d<isoformat>``."""
    out = SpyData()
    for d in days:
        out.days[d] = per_day.get(f"d{d:%Y%m%d}") or session_bars(d, price=price)
    return out


def path(
    story_id: str,
    symbol: str,
    days: Sequence[date],
    bars: Sequence[BarArrays],
    *,
    spare: date | None = None,
    spare_bars: BarArrays | None = None,
) -> PathData:
    return PathData(
        story_id=story_id,
        symbol=symbol,
        sessions=tuple(Session.of(d) for d in days),
        bars=tuple(bars),
        spare=Session.of(spare) if spare else None,
        spare_bars=spare_bars,
    )


# ── stories ──


@dataclass(frozen=True, slots=True)
class Card:
    family: str | None
    type: str
    structural: bool = False
    vetoes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Item:
    available_at: datetime
    itype: str = "analyst_downgrade"
    structural: bool = False


FAMILY = frozenset({"analyst_downgrade", "earnings_miss"})


@dataclass(slots=True)
class Story:
    """A StoryView over a list of items: NSN_CORE while a family item is known."""

    symbol: str
    session: date
    items: list[Item] = field(default_factory=list)

    @property
    def story_id(self) -> str:
        return f"{self.symbol}:{self.session.isoformat()}"

    def _known(self, t: datetime) -> list[Item]:
        return [i for i in self.items if i.available_at <= t]

    def card_at(self, t: datetime) -> Card:
        known = self._known(t)
        family = [i for i in known if i.itype in FAMILY]
        structural = any(i.structural for i in known)
        if family and not structural:
            return Card("NSN_CORE", family[0].itype, False)
        return Card(None, known[-1].itype if known else "none", structural)

    def nsn_at(self, cutoff: datetime) -> datetime | None:
        for i in sorted(self.items, key=lambda i: i.available_at):
            if i.available_at > cutoff:
                return None
            if self.card_at(i.available_at).family == "NSN_CORE":
                return i.available_at
        return None

    def at_news(self) -> datetime | None:
        nsn = self.nsn_at(Session.of(self.session).entry_cutoff)
        return nsn - NEWS_LAG if nsn is not None else None

    def start_case(self) -> str:
        at = self.at_news()
        s = Session.of(self.session)
        return "in" if at is not None and s.open <= at < s.close else "out"

    def news_times(self) -> list[datetime]:
        return [i.available_at for i in self.items]

    def before(self, t: datetime) -> Story:
        """The same story with every item after ``t`` deleted."""
        return Story(self.symbol, self.session, [i for i in self.items if i.available_at <= t])


def story(symbol: str, session: date, *items: Item) -> Story:
    return Story(symbol, session, list(items))


def downgrade(day: date, hh: int = 8, mm: int = 0) -> Item:
    return Item(et(day, hh, mm))


# ── context ──


@dataclass(frozen=True, slots=True)
class Daily:
    """A raw daily bar shaped like ``context.DailyPoint`` (the fields the simulator reads)."""

    open: float
    high: float
    low: float
    close: float
    volume: float


class Context:
    """A ContextView: A-factors (1.0 unless set), verdicts (halal unless set), daily bars."""

    def __init__(
        self,
        *,
        adj: Mapping[tuple[str, date], float | None] | None = None,
        verdicts: Mapping[tuple[str, date], str] | None = None,
        daily: Mapping[tuple[str, date], Daily] | None = None,
        sessions: Sequence[date] = (),
    ) -> None:
        self._adj = dict(adj or {})
        self._verdicts = dict(verdicts or {})
        self._daily = dict(daily or {})
        self.sessions = list(sessions)

    def adj(self, symbol: str, day: date) -> float | None:
        return self._adj.get((symbol, day), 1.0)

    def screen_verdict(self, symbol: str, day: date) -> str:
        return self._verdicts.get((symbol, day), "halal")

    def daily(self, symbol: str, day: date) -> Daily | None:
        return self._daily.get((symbol, day))


def daily(c: float, o: float | None = None) -> Daily:
    o = c if o is None else o
    return Daily(o, max(o, c), min(o, c), c, 1e6)


# ── the toy playbook ──

LIVE = frozenset({"WATCHING", "ARMED", "ENTERING", "ENTERED"})  # the toy's own live states

FACTS = TradeFacts(
    family_type="analyst_downgrade",
    cell="NSN_CORE/ID",
    variant="ID",
    cost_bps=15.0,
    rank=500,
    tech=False,
    beta=1.0,
)


class Toy:
    """Buy at the first visible bar after open + 20 min, sell at target or the time stop."""

    name = "toy"
    version = "1"

    def __init__(
        self,
        story: object,
        *,
        sessions: int = 1,
        target: float = 0.01,
        entry_bar: tuple[int, int] | None = None,
        facts: TradeFacts = FACTS,
        sell_at: tuple[int, int, int] | None = None,
        finish_after_entry: bool = False,
    ) -> None:
        self.path_sessions = sessions
        self.seen: list[Input] = []
        self._finish_after_entry = finish_after_entry
        self._target_pct = target
        self._entry_bar = entry_bar
        self._facts = facts
        self._sell_at = sell_at  # (path session, hour, minute) of a bar that forces a "stop" sell
        self._state = "DETECTED"
        self._target = math.inf

    def state(self) -> str:
        return self._state

    def live(self) -> bool:
        return self._state in LIVE

    def _go(self, to: str, reason: str) -> Transition:
        self._state = to
        return Transition(to, reason)

    def start(self, ctx: Ctx) -> list[Intent]:
        return [self._go("WATCHING", "start")]

    def on(self, ev: Input, ctx: Ctx) -> list[Intent]:
        self.seen.append(ev)
        if isinstance(ev, BarIn) and ev.symbol == ctx.story.symbol:
            return self._bar(ev, ctx)
        if isinstance(ev, FillIn):
            if ev.side == "buy":
                self._target = ev.price_s * (1.0 + self._target_pct)
                if self._finish_after_entry:
                    return [self._go("ENTERED", "filled"), Finish("done")]
                return [self._go("ENTERED", "filled")]
            return [self._go("EXITED", ev.reason), Finish(ev.reason)]
        if isinstance(ev, OrderClosedIn) and ev.side == "buy":
            return [self._go("EXPIRED", "entry_unfilled"), Finish("entry_unfilled")]
        if isinstance(ev, NewsIn) and self._state == "ENTERED":
            if ev.story.card_at(ev.at).structural:
                return [self._go("EXITING", "abort"), Submit("sell", reason="abort")]
        if isinstance(ev, ComplianceIn) and self._state == "ENTERED":
            return [self._go("EXITING", "compliance")]
        if isinstance(ev, SessionIn):
            if ev.kind == "entry_cutoff" and ev.k == 0 and self._state == "WATCHING":
                return [self._go("EXPIRED", "cutoff"), Finish("cutoff")]
            if ev.kind == "flatten" and ev.k == self.path_sessions - 1 and self._state == "ENTERED":
                return [self._go("EXITING", "time_stop")]
        return []

    def _bar(self, ev: BarIn, ctx: Ctx) -> list[Intent]:
        bars = ctx.market.bars(ev.symbol)
        start = from_us(int(bars.ts[ev.i]) * 1_000_000)
        if self._state == "WATCHING":
            wanted = (
                start == et(ctx.session.day, *self._entry_bar)
                if self._entry_bar is not None
                else ev.at >= ctx.session.entry_start
            )
            if wanted:
                return [
                    self._go("ARMED", TRIGGERED),
                    self._go("ENTERING", "entry"),
                    Submit("buy", facts=self._facts),
                ]
        elif self._state == "ENTERED":
            if self._sell_at is not None:
                k, hh, mm = self._sell_at
                if start == et(ctx.sessions[k].day, hh, mm):
                    return [self._go("EXITING", "stop"), Submit("sell", reason="stop")]
            close_s = float(bars.c[ev.i] * bars.scale[ev.i])
            if close_s >= self._target:
                return [self._go("EXITING", "target"), Submit("sell", reason="target")]
        return []


class Recorder:
    """A toy factory that keeps every playbook it made, and the story each got."""

    name = Toy.name
    version = Toy.version

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.made: list[Toy] = []
        self.stories: list[object] = []

    @property
    def path_sessions(self) -> int:
        return int(self.kwargs.get("sessions", 1))  # type: ignore[call-overload]

    def __call__(self, story: object) -> Toy:
        toy = Toy(story, **self.kwargs)  # type: ignore[arg-type]
        self.made.append(toy)
        self.stories.append(story)
        return toy


@dataclass(frozen=True, slots=True)
class ToyFactory:
    """A picklable PlaybookFactory of toys (a spawn pool needs one)."""

    sessions: int = 1
    target: float = 0.01
    name: str = Toy.name
    version: str = Toy.version

    @property
    def path_sessions(self) -> int:
        return self.sessions

    def __call__(self, story: object) -> Toy:
        return Toy(story, sessions=self.sessions, target=self.target)


DEFAULT_CFG = SimConfig()


def run_one(
    st: Story,
    pd: PathData | PathSkip,
    *,
    spy: SpyData | None = None,
    ctx: Context | None = None,
    cfg: SimConfig = DEFAULT_CFG,
    toy: Mapping[str, object] | None = None,
    factory: object | None = None,
    stop_at: str = "end",
    assume_full_hold: bool = False,
    keep: bool = True,
    news_from: Sequence[Story] | None = None,
    state: SymbolState | None = None,
) -> StoryOutcome:
    """Simulate one story with the toy playbook; returns its outcome."""
    days = [s.day for s in pd.sessions] if isinstance(pd, PathData) else [st.session]
    if isinstance(pd, PathData) and pd.spare is not None:
        days.append(pd.spare.day)
    spy = spy or spy_data(days)
    out = simulate_symbol(
        st.symbol,
        [st],
        factory or (lambda s: Toy(s, **dict(toy or {}))),  # type: ignore[arg-type]
        {st.story_id: pd},
        spy,
        ctx or Context(),
        cfg,
        stop_at=stop_at,  # type: ignore[arg-type]
        assume_full_hold=assume_full_hold,
        keep_transitions=keep,
        news_from=news_from,
        state=state,
    )
    assert len(out) == 1
    return out[0]
