"""The simulator's data types: sessions, bar series, paths, inputs, intents and orders.

Prices in a :class:`BarSeries` are **raw** (what traded). A path spanning
several sessions carries ``scale = A(d) / A(S)`` per bar, with ``A(d) =
close_all(d) / close_raw(d)``, so ``price * scale`` is in session-S units
and a split or a dividend inside the path is not a price move
(:meth:`BarSeries.in_s_units`).

A playbook receives **inputs** (``BarIn``, ``GapIn``, ``NewsIn``,
``SessionIn``, ``FillIn``, ``OrderClosedIn``, ``ComplianceIn``, ``TimerIn``)
and answers with **intents** (``Submit``, ``Cancel``, ``SetTimer``,
``Transition``, ``Finish``). Every input carries the instant it is
delivered, ``at``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from halabot.playbooks.clock import (
    ENTRY_CUTOFF_BEFORE_CLOSE,
    ENTRY_START_AFTER_OPEN,
    FLATTEN_BEFORE_CLOSE,
    GAP,
    ORDER_LAG,
    PRE_OPEN,
    SIP_RT,
    FeedProfile,
    from_us,
)
from halabot.playbooks.interfaces import StoryView
from halal_trader.data.minutes import BarArrays, session_bounds
from halal_trader.market_hours import (
    EARLY_CLOSE_DATES,
    MARKET_TZ,
    is_trading_day,
    next_trading_day,
)

if TYPE_CHECKING:
    from halabot.playbooks.exchange import FillModel

# ── sessions ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Session:
    """One regular session: ``[open, close)`` in UTC, early closes honoured."""

    day: date
    open: datetime
    close: datetime
    early: bool

    @classmethod
    def of(cls, day: date) -> Session:
        if not is_trading_day(day):
            raise ValueError(f"{day} is not a trading session")
        lo, hi = session_bounds(day)
        return cls(day, lo.astimezone(UTC), hi.astimezone(UTC), day in EARLY_CLOSE_DATES)

    @property
    def pre_open(self) -> datetime:
        return datetime.combine(self.day, PRE_OPEN, MARKET_TZ).astimezone(UTC)

    @property
    def entry_start(self) -> datetime:
        return self.open + ENTRY_START_AFTER_OPEN

    @property
    def entry_cutoff(self) -> datetime:
        return self.close - ENTRY_CUTOFF_BEFORE_CLOSE

    @property
    def flatten(self) -> datetime:
        return self.close - FLATTEN_BEFORE_CLOSE


def path_days(session: date, n: int) -> list[date]:
    """``n`` consecutive sessions from ``session`` (included), by ``market_hours``."""
    if not is_trading_day(session):
        raise ValueError(f"{session} is not a trading session")
    days = [session]
    while len(days) < n:
        days.append(next_trading_day(days[-1]))
    return days


# ── bars ──────────────────────────────────────────────────────


def _frozen[A: np.generic](a: NDArray[A]) -> NDArray[A]:
    a.setflags(write=False)
    return a


@dataclass(frozen=True, slots=True)
class BarSeries:
    """One symbol's bars over a path, ascending ``ts``, as read-only numpy arrays.

    ``ts`` and ``visible_at`` are epoch seconds (the minute's start, and the
    instant the bar reaches the playbook); ``k`` is each bar's index among
    the path's sessions; ``scale`` is ``A(day) / A(S)``.
    """

    ts: NDArray[np.int64]
    o: NDArray[np.float64]
    h: NDArray[np.float64]
    l: NDArray[np.float64]  # noqa: E741 - the bar's low, named as in o/h/l/c
    c: NDArray[np.float64]
    v: NDArray[np.float64]
    vw: NDArray[np.float64]
    visible_at: NDArray[np.int64]
    k: NDArray[np.int64]
    scale: NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.ts)

    @classmethod
    def build(cls, parts: Sequence[BarArrays], scales: Sequence[float], lag_s: int) -> BarSeries:
        """One series from per-session arrays (path order) and their ``A(d)/A(S)``."""
        if len(parts) != len(scales):
            raise ValueError("one scale per session")
        if not parts:
            return cls.empty()
        ts = np.concatenate([p.ts for p in parts]).astype(np.int64)
        k = np.concatenate(
            [np.full(len(p), j, dtype=np.int64) for j, p in enumerate(parts)]
        ).astype(np.int64)
        scale = np.concatenate(
            [np.full(len(p), s, dtype=np.float64) for p, s in zip(parts, scales)]
        ).astype(np.float64)
        return cls(
            _frozen(ts),
            _frozen(np.concatenate([p.o for p in parts]).astype(np.float64)),
            _frozen(np.concatenate([p.h for p in parts]).astype(np.float64)),
            _frozen(np.concatenate([p.l for p in parts]).astype(np.float64)),
            _frozen(np.concatenate([p.c for p in parts]).astype(np.float64)),
            _frozen(np.concatenate([p.v for p in parts]).astype(np.float64)),
            _frozen(np.concatenate([p.vw for p in parts]).astype(np.float64)),
            _frozen(ts + np.int64(60 + lag_s)),
            _frozen(k),
            _frozen(scale),
        )

    @classmethod
    def empty(cls) -> BarSeries:
        i = _frozen(np.empty(0, dtype=np.int64))
        f = _frozen(np.empty(0, dtype=np.float64))
        return cls(i, f, f, f, f, f, f, i, i, f)

    def columns(self) -> tuple[NDArray[Any], ...]:
        """The arrays in field order (``BarSeries(*columns)`` rebuilds the series)."""
        return (
            self.ts,
            self.o,
            self.h,
            self.l,
            self.c,
            self.v,
            self.vw,
            self.visible_at,
            self.k,
            self.scale,
        )

    def in_s_units(self) -> BarSeries:
        """Prices times ``scale`` (session-S units), volume divided by it; ``scale`` becomes 1."""
        s = self.scale
        return BarSeries(
            self.ts,
            _frozen(self.o * s),
            _frozen(self.h * s),
            _frozen(self.l * s),
            _frozen(self.c * s),
            _frozen(self.v / s),
            _frozen(self.vw * s),
            self.visible_at,
            self.k,
            _frozen(np.ones(len(s), dtype=np.float64)),
        )

    def session_start(self, k: int) -> int:
        """Index of the first bar of path session ``k`` (``len`` if there is none after it)."""
        return int(np.searchsorted(self.k, k, side="left"))

    def bar_time(self, i: int) -> datetime:
        """Bar ``i``'s start as a UTC datetime."""
        return from_us(int(self.ts[i]) * 1_000_000)


class VisibleBars:
    """The visible head of one path's :class:`BarSeries`, grown in place.

    :meth:`head` returns read-only views ``buf[:n]`` of buffers the size of
    the path. A buffer is filled only up to the largest ``n`` asked so far,
    and ``n`` never goes back (a run's clock only moves forward), so a
    view's ``.base`` holds the bars visible by then and zeros after them:
    no later bar is reachable through it. Each bar is copied once, when it
    becomes visible, so a playbook reading its bars on every bar costs
    linear time over a path (a fresh copy per bar cost quadratic time).
    The head of the current ``n`` is cached: asking twice returns the same
    object.

    The buffers themselves are read-only too, except while :meth:`head`
    copies the new bars in: a write through a view's ``.base`` is refused,
    so a head a playbook keeps never changes under it.
    """

    __slots__ = ("_bufs", "_head", "_n", "_src")

    def __init__(self, series: BarSeries) -> None:
        self._src = series.columns()
        # Read-only buffers: their slices are read-only views whose ``.base``
        # is the buffer (numpy points a view at the owner).
        self._bufs = tuple(_frozen(np.zeros_like(a)) for a in self._src)
        self._n = 0
        self._head: BarSeries | None = None

    def __len__(self) -> int:
        """How many bars are visible (filled) so far."""
        return self._n

    def head(self, n: int) -> BarSeries:
        """The first ``n`` bars as read-only views; ``n`` may not go back."""
        if n < self._n:
            raise ValueError(f"visible bars only grow: {n} asked after {self._n}")
        if n > len(self._bufs[0]):
            raise ValueError(f"{n} bars asked of a path of {len(self._bufs[0])}")
        if n > self._n:
            for buf, src in zip(self._bufs, self._src):
                buf.flags.writeable = True  # the buffer owns its memory: allowed
                try:
                    buf[self._n : n] = src[self._n : n]
                finally:
                    buf.flags.writeable = False
            self._n = n
            self._head = None
        if self._head is None:
            self._head = BarSeries(*(b[:n] for b in self._bufs))
        return self._head


@dataclass(slots=True)
class SpyData:
    """SPY's regular-session bars by session, filled by the loader as paths load."""

    days: dict[date, BarArrays] = field(default_factory=dict)

    def get(self, day: date) -> BarArrays | None:
        return self.days.get(day)

    def subset(self, days: Iterable[date]) -> SpyData:
        return SpyData({d: self.days[d] for d in days if d in self.days})


SkipReason = Literal[
    "units_missing", "spy_missing", "adjust_defect", "bad_bars", "halted_all_day", "no_daily"
]
DATA_SKIPS: frozenset[str] = frozenset(
    {"units_missing", "spy_missing", "adjust_defect", "bad_bars"}
)


@dataclass(frozen=True, slots=True)
class PathData:
    """One story's raw minute bars over its path (S .. S + n - 1), sanity-filtered.

    ``spare`` is the next session when it is inside the window and loaded:
    it is never shown to the playbook, and only serves an exit that found
    no market on the deadline session (``no_market``).
    """

    story_id: str
    symbol: str
    sessions: tuple[Session, ...]
    bars: tuple[BarArrays, ...]
    spare: Session | None = None
    spare_bars: BarArrays | None = None
    dropped: int = 0  # rows dropped by the bar sanity rule


@dataclass(frozen=True, slots=True)
class PathSkip:
    """A story the loader could not give a path, and why."""

    story_id: str
    reason: SkipReason


# ── orders ────────────────────────────────────────────────────


class OrderKind(StrEnum):
    MARKET = "market"  # time in force: day; the only kind H1 uses
    LIMIT = "limit"  # reserved: a later trial, tagged .research
    MOO = "moo"  # reserved: a later trial, tagged .port
    MOC = "moc"  # reserved: a later trial, tagged .port


Side = Literal["buy", "sell"]
CostMode = Literal["study", "study_x1.5", "study_x2", "surcharge"]


@dataclass(frozen=True, slots=True)
class TradeFacts:
    """What the trade record needs from the playbook, sent with the entry ``Submit``.

    Levels are in session-S units; ``cost_bps`` is the one-way cost
    (``study.cost_bps(rank)``).
    """

    family_type: str
    cell: str
    variant: str
    cost_bps: float
    rank: int
    tech: bool
    beta: float = math.nan
    p0: float = math.nan
    spy0: float = math.nan
    sigma: float = math.nan
    thr: float = math.nan
    low_star: float = math.nan
    target: float = math.nan
    anchor_ts: datetime | None = None
    adv20_usd: float = math.nan  # the "surcharge" cost sensitivity only


@dataclass(frozen=True, slots=True)
class WorkingOrder:
    """An admitted order; times are epoch microseconds.

    ``k`` is the path session it works in; ``qty`` is None for a buy (sized
    at the fill as ``reference_notional / price``).
    """

    order_id: str
    side: Side
    kind: OrderKind
    k: int
    decided_at_us: int
    active_at_us: int
    qty: float | None
    reason: str = ""
    tag: str = ""
    by_sim: bool = False

    @property
    def decided_at(self) -> datetime:
        return from_us(self.decided_at_us)

    @property
    def active_at(self) -> datetime:
        return from_us(self.active_at_us)


@dataclass(frozen=True, slots=True)
class Execution:
    """A whole order filled on one bar (or by a fallback, ``i == -1``)."""

    order: WorkingOrder
    k: int  # path session of the fill (``len(sessions)`` for the spare)
    i: int  # bar index in the path series (-1: official close or last trade)
    bar_ts: int | None  # epoch seconds of the filling bar's start
    price: float  # raw
    filled_at_us: int
    flags: tuple[str, ...] = ()
    participation: float = math.nan


# ── inputs ────────────────────────────────────────────────────

SessionKind = Literal["pre_open", "open", "entry_start", "entry_cutoff", "flatten", "close"]


@dataclass(frozen=True, slots=True)
class BarIn:
    """Bar ``i`` of ``market.bars(symbol)`` just became visible."""

    at: datetime
    symbol: str
    i: int


@dataclass(frozen=True, slots=True)
class GapIn:
    """No print for 5 minutes or more before ``until`` (or a late first bar)."""

    at: datetime
    symbol: str
    since: datetime | None  # the previous bar's start; None for a session's first bar
    until: datetime  # the start of the bar that ends the gap
    late_open: bool


@dataclass(frozen=True, slots=True)
class NewsIn:
    """An item of ``story`` became available (``own``: the playbook's own story)."""

    at: datetime
    story: StoryView
    own: bool


@dataclass(frozen=True, slots=True)
class SessionIn:
    at: datetime
    kind: SessionKind
    k: int  # path session index
    day: date


@dataclass(frozen=True, slots=True)
class FillIn:
    """An order filled: ``price`` raw, ``price_s`` in session-S units."""

    at: datetime
    order_id: str
    side: Side
    price: float
    price_s: float
    qty: float
    bar_ts: datetime | None
    flags: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class OrderClosedIn:
    at: datetime
    order_id: str
    side: Side
    status: Literal["rejected", "expired", "cancelled"]
    reason: str


@dataclass(frozen=True, slots=True)
class ComplianceIn:
    """At pre_open of a held session the screen is not halal; the simulator sells at the open."""

    at: datetime
    day: date
    verdict: str


@dataclass(frozen=True, slots=True)
class TimerIn:
    at: datetime
    tag: str


Input = BarIn | GapIn | NewsIn | SessionIn | FillIn | OrderClosedIn | ComplianceIn | TimerIn


# ── intents ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Submit:
    """A new order. A buy needs ``facts``; a sell's ``qty`` None means the whole position."""

    side: Side
    kind: OrderKind = OrderKind.MARKET
    qty: float | None = None
    reason: str = ""  # a sell's exit reason: target | stop | abort | compliance | time_stop
    tag: str = ""
    facts: TradeFacts | None = None


@dataclass(frozen=True, slots=True)
class Cancel:
    order_id: str | None = None
    tag: str | None = None


@dataclass(frozen=True, slots=True)
class SetTimer:
    at: datetime
    tag: str


@dataclass(frozen=True, slots=True)
class Transition:
    """The playbook moved to state ``to``; ``reason`` says why.

    ``reason == "triggered"`` (``playbook.TRIGGERED``) marks the trigger, a self-transition.
    """

    to: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Finish:
    """The playbook is done. A held position is flattened by the simulator (time_stop)."""

    reason: str = ""


Intent = Submit | Cancel | SetTimer | Transition | Finish


# ── configuration ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SimConfig:
    """The simulator's constants.

    ``fill`` is the fill model (``exchange.FillModel``). None means the D.5
    market rule (``exchange.MARKET_FILL``), the only one a trial uses;
    ``legacy.py`` holds the gate-only models that replicate the legacy
    studies (``legacy.reactor_config()``, ``legacy.daily_config()``).
    """

    feed: FeedProfile = SIP_RT
    order_lag: timedelta = ORDER_LAG
    gap: timedelta = GAP
    cost: CostMode = "study"
    reference_notional: float = 10_000.0
    allowed_kinds: frozenset[OrderKind] = frozenset({OrderKind.MARKET})
    fill: FillModel | None = None

    def as_config(self) -> dict[str, object]:
        """The constants as plain JSON (for the run row and trial configs)."""
        return {
            "feed": self.feed.name,
            "bar_visible_s": 60 + int(self.feed.bar_lag.total_seconds()),
            "order_lag_s": self.order_lag.total_seconds(),
            "gap_s": self.gap.total_seconds(),
            "cost": self.cost,
            "reference_notional": self.reference_notional,
            "allowed_kinds": sorted(str(k) for k in self.allowed_kinds),
            "fill": self.fill.name if self.fill is not None else "market",
        }


__all__ = [
    "DATA_SKIPS",
    "BarIn",
    "BarSeries",
    "Cancel",
    "ComplianceIn",
    "CostMode",
    "Execution",
    "FillIn",
    "Finish",
    "GapIn",
    "Input",
    "Intent",
    "NewsIn",
    "OrderClosedIn",
    "OrderKind",
    "PathData",
    "PathSkip",
    "Session",
    "SessionIn",
    "SessionKind",
    "SetTimer",
    "Side",
    "SimConfig",
    "SkipReason",
    "SpyData",
    "Submit",
    "TimerIn",
    "TradeFacts",
    "Transition",
    "VisibleBars",
    "WorkingOrder",
    "path_days",
]
