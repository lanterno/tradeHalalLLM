"""The minute simulator: playbooks run against minute bars, one story at a time per symbol.

**A story** starts at ``max(nsn_at(entry cutoff of S), open(S))``
(:func:`start_time`); a story that never becomes NSN_CORE by the cutoff is
not started (its items still reach a live playbook as news). From the
start, one :class:`~halabot.playbooks.clock.EventHeap` merges, in
``(time, priority, seq)`` order:

* the exchange testing a working order on its first eligible bar, at
  ``ts + 60 s`` (fills: ``exchange.py``), and the ``FillIn`` 1 s later;
* ``SessionIn`` markers of each path session (pre_open 09:20, open,
  entry_start, entry_cutoff, flatten, close);
* SPY's bars, then the symbol's (a ``GapIn`` first when a gap ends), each
  at its visible time;
* ``NewsIn`` at each ``available_at`` of every story of the symbol;
* the playbook's timers.

Bars already visible at the start are history, readable through
``ctx.market``. The playbook sees only ``Ctx``: stories through
:class:`PitStory` (items available by ``now`` only); daily bars, the screen
and bars not yet visible stay with the simulator.

**The simulator enforces** (``rules.py`` admits each order):

* compliance: at pre_open of each later held session, a screen that is not
  halal sends a market sell at the open + ``ORDER_LAG`` (``compliance``),
  and the playbook gets a ``ComplianceIn``;
* flatten: at close - 5 min of the deadline session, working buys are
  cancelled and a held position gets a market sell (``time_stop``); an exit
  already working is kept (it fills no later than a new one would).
  ``Finish`` while holding does the same;
* day orders: at each close, a working buy expires; a working sell is
  carried to the next path session's open + ``ORDER_LAG``, and on the
  deadline session falls back to the official close (``close_fallback``),
  else the spare session's market (``no_market``), else the last trade
  (``unresolved``); the trade is always kept.

**Per symbol**, stories run in start order and only one playbook may be
live (``Playbook.live()``) at a time: a story starting meanwhile is
``blocked_open``, before any eligibility its own playbook would judge (it
never gets one). ``stop_at="entry"`` stops a story at its entry fill
(Stage A); ``assume_full_hold`` then keeps the symbol blocked through the
deadline session.

**Fill models.** The exchange fills by ``SimConfig.fill`` (the D.5 market
rule unless a gate-only model from ``legacy.py`` is given; :func:`run`
accepts those only under a gate unlock). A gate-only model replicates a
legacy study, which held through its exit whatever the screen said: under
one the simulator skips the screen at entry, the flatten and the pre-open
compliance exit, so the exit is the model's own (``time_stop``).

**Determinism.** Symbols are independent, so they are split over workers by
``crc32(symbol) % workers`` and the records are identical for any worker
count (``records.outcomes_sha256``). :func:`run` keeps no module state:
two runs may share an event loop.
"""

from __future__ import annotations

import asyncio
import bisect
import logging
import math
import multiprocessing
import sys
import warnings
import zlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, Final, Literal

import numpy as np
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks import rules
from halabot.playbooks.clock import US, EventHeap, Priority, from_us, span_us, to_us
from halabot.playbooks.exchange import (
    Exchange,
    fill_model,
    in_surcharge_window,
    one_way_bps,
    spy_fill,
    unfilled_exit,
)
from halabot.playbooks.interfaces import CardView, ContextView, StoryView
from halabot.playbooks.loader import (
    SPY,
    MinuteBarLoader,
    PathRequest,
    Window,
    WindowUnlock,
)
from halabot.playbooks.playbook import ARMED, TRIGGERED, Ctx, Playbook, PlaybookFactory
from halabot.playbooks.records import Leg, OutcomeSink, RunInfo, StoryOutcome, TradeRecord
from halabot.playbooks.types import (
    BarIn,
    BarSeries,
    Cancel,
    ComplianceIn,
    Execution,
    FillIn,
    Finish,
    GapIn,
    Input,
    Intent,
    NewsIn,
    OrderClosedIn,
    OrderKind,
    PathData,
    PathSkip,
    Session,
    SessionIn,
    SessionKind,
    SetTimer,
    SimConfig,
    SpyData,
    Submit,
    TimerIn,
    TradeFacts,
    Transition,
    VisibleBars,
    WorkingOrder,
)
from halal_trader.core import events
from halal_trader.data.minutes import BarArrays

logger = logging.getLogger(__name__)

StopAt = Literal["entry", "end"]
Parallel = bool | Literal["fork", "spawn"] | None
MakePlaybook = Callable[[StoryView], Playbook]

# Heap payload kinds.
_EXCH, _NOTE, _SESSION, _SPY, _BAR, _NEWS, _TIMER = range(7)
_SESSION_KINDS: tuple[SessionKind, ...] = (
    "pre_open",
    "open",
    "entry_start",
    "entry_cutoff",
    "flatten",
    "close",
)


def start_time(story: StoryView) -> datetime | None:
    """When a story's playbook starts: ``max(nsn_at(entry cutoff of S), open(S))``."""
    s = Session.of(story.session)
    nsn = story.nsn_at(s.entry_cutoff)
    return None if nsn is None else max(nsn, s.open)


@dataclass(slots=True)
class SymbolState:
    """What a symbol's earlier stories leave for its later ones."""

    busy_until_us: int = 0  # a playbook on the symbol is live until then
    held: list[tuple[int, int]] = field(default_factory=list)  # [entry fill, exit fill)

    def holds(self, at_us: int) -> bool:
        return any(a <= at_us < b for a, b in self.held)


class _Market:
    """``MarketView`` over one path: bars visible at ``now`` only.

    ``bars`` hands out read-only views of a buffer filled as bars become
    visible (:class:`~halabot.playbooks.types.VisibleBars`, one per symbol,
    made on first use): nothing past ``now`` is behind them, and reading
    them on every bar costs linear time over the path.
    """

    __slots__ = ("_a_s", "_b_s", "_bars", "_ctx", "_now_s", "_spy", "_symbol", "_visible")

    def __init__(
        self, symbol: str, bars: BarSeries, spy: BarSeries, ctx: ContextView, s: date
    ) -> None:
        self._symbol = symbol
        self._bars = bars
        self._spy = spy
        self._ctx = ctx
        self._a_s = ctx.adj(symbol, s)
        self._b_s = ctx.adj(SPY, s)
        self._now_s = 0
        self._visible: dict[str, VisibleBars] = {}

    def set_now(self, now_us: int) -> None:
        self._now_s = now_us // US

    def _series(self, symbol: str) -> BarSeries:
        if symbol == self._symbol:
            return self._bars
        if symbol == SPY:
            return self._spy
        raise KeyError(f"no bars for {symbol} in this path")

    def bars(self, symbol: str) -> BarSeries:
        series = self._series(symbol)
        n = int(np.searchsorted(series.visible_at, self._now_s, side="right"))
        visible = self._visible.get(symbol)
        if visible is None:
            visible = self._visible[symbol] = VisibleBars(series)
        return visible.head(n)

    def last(self, symbol: str) -> float | None:
        series = self._series(symbol)
        n = int(np.searchsorted(series.visible_at, self._now_s, side="right"))
        return float(series.c[n - 1]) if n else None

    def to_s_units(self, price: float, day: date, *, symbol: str | None = None) -> float:
        sym = symbol or self._symbol
        if sym not in (self._symbol, SPY):
            raise KeyError(f"no A-factors for {sym} in this path")
        a_s = self._a_s if sym == self._symbol else self._b_s
        a = self._ctx.adj(sym, day)
        if a is None or a_s is None:
            raise KeyError(f"no A-factor for {sym} on {day}")
        return price * a / a_s


class PitStory:
    """A StoryView that answers only from the items available by its run's ``now``.

    What a playbook gets as ``ctx.story`` and ``NewsIn.story``: ``card_at(t)``
    and ``nsn_at(cutoff)`` are clamped to ``now``, ``news_times()`` lists only
    the items already available, and ``at_news()`` / ``start_case()`` are
    known once the NSN item is (always by the story's start).
    """

    __slots__ = ("_now", "_story")

    def __init__(self, story: StoryView, now: Callable[[], datetime]) -> None:
        self._story = story
        self._now = now

    @property
    def story_id(self) -> str:
        return self._story.story_id

    @property
    def symbol(self) -> str:
        return self._story.symbol

    @property
    def session(self) -> date:
        return self._story.session

    def card_at(self, t: datetime) -> CardView:
        return self._story.card_at(min(t, self._now()))

    def nsn_at(self, cutoff: datetime) -> datetime | None:
        return self._story.nsn_at(min(cutoff, self._now()))

    def at_news(self) -> datetime | None:
        if self.nsn_at(Session.of(self.session).entry_cutoff) is None:
            return None
        return self._story.at_news()

    def start_case(self) -> str:
        return self._story.start_case() if self.at_news() is not None else "out"

    def news_times(self) -> list[datetime]:
        now = self._now()
        return [t for t in self._story.news_times() if t <= now]


class _Position:
    """``PositionView`` of one story run."""

    __slots__ = ("_run",)

    def __init__(self, run: _StoryRun) -> None:
        self._run = run

    @property
    def held(self) -> bool:
        return self._run.qty > 0

    @property
    def qty(self) -> float:
        return self._run.qty

    @property
    def entry_price(self) -> float | None:
        e = self._run.entry
        return e.price if e is not None else None

    @property
    def entry_price_s(self) -> float | None:
        e = self._run.entry
        return e.price * self._run.scale_of(e.k) if e is not None else None

    @property
    def entry_at(self) -> datetime | None:
        e = self._run.entry
        return from_us(e.filled_at_us) if e is not None else None

    @property
    def working(self) -> tuple[WorkingOrder, ...]:
        return self._run.exchange.working


def _spy_series(spy: SpyData, days: Sequence[date], scales: Sequence[float], lag: int) -> BarSeries:
    return BarSeries.build([spy.get(d) or BarArrays.empty() for d in days], scales, lag)


class _StoryRun:
    """One story's simulation (see the module docstring)."""

    def __init__(
        self,
        story: StoryView,
        make_playbook: MakePlaybook,
        path: PathData,
        spy: SpyData,
        ctx: ContextView,
        cfg: SimConfig,
        start_us: int,
        news: Sequence[tuple[int, StoryView]],
        state: SymbolState,
        *,
        stop_at: StopAt,
        assume_full_hold: bool,
        keep_transitions: bool,
        run_id: str,
    ) -> None:
        if path.sessions[0].day != story.session or path.symbol != story.symbol:
            raise ValueError(
                f"{story.story_id}: the path is {path.symbol} from {path.sessions[0].day}"
            )
        self.story = story
        self.path = path
        self.ctx = ctx
        self.cfg = cfg
        self.symbol = story.symbol
        self.sessions = path.sessions
        self.n = len(path.sessions)
        self.deadline = self.n - 1
        self.state = state
        self.stop_at = stop_at
        self.assume_full_hold = assume_full_hold
        self.keep = keep_transitions
        self.run_id = run_id
        self.lag_us = span_us(cfg.order_lag)
        lag_s = int(cfg.feed.bar_lag.total_seconds())
        s_day = self.sessions[0].day
        a_s = ctx.adj(self.symbol, s_day)
        b_s = ctx.adj(SPY, s_day)
        if a_s is None or b_s is None:
            raise ValueError(f"{story.story_id}: no A-factor on S")
        self.a_s, self.b_s = a_s, b_s
        days = [s.day for s in self.sessions]
        self.scales = [(ctx.adj(self.symbol, d) or a_s) / a_s for d in days]
        self.spy_scales = [(ctx.adj(SPY, d) or b_s) / b_s for d in days]
        self.bars = BarSeries.build(path.bars, self.scales, lag_s)
        self.spy = _spy_series(spy, days, self.spy_scales, lag_s)
        self.spy_data = spy
        self.market = _Market(self.symbol, self.bars, self.spy, ctx, s_day)
        self.exchange = Exchange(cfg)
        self.fill = fill_model(cfg)
        self.position = _Position(self)
        self.open_us = [to_us(s.open) for s in self.sessions]
        self.close_us = [to_us(s.close) for s in self.sessions]
        self.pre_open_us = [to_us(s.pre_open) for s in self.sessions]
        self.start_us = start_us
        self.end_us = self.close_us[-1]
        self.news = [(t, s) for t, s in news if start_us < t <= self.end_us]

        self.heap: EventHeap[tuple[int, Any]] = EventHeap()
        self._pit: dict[str, PitStory] = {}
        self.now_us = start_us
        self.qty = 0.0
        self.entry: Execution | None = None
        self.exit: Execution | None = None
        self.entry_spy = math.nan
        self.exit_spy = math.nan
        self.entry_flags: tuple[str, ...] = ()
        self.exit_flags: tuple[str, ...] = ()
        self.exit_k = -1
        self.exit_spare: BarSeries | None = None
        self.facts: TradeFacts | None = None
        self.due: dict[str, int] = {}
        self.n_orders = 0
        self.transitions: list[tuple[datetime, str, str, str]] = []
        self.intents: list[tuple[datetime, Intent]] = []
        self.last_reason = ""
        self.triggered_at: datetime | None = None
        self.armed_at: datetime | None = None
        self.entry_decided_at: datetime | None = None
        self.finished = False
        self.finish_reason: str | None = None
        self.done = False
        self.stop_after_entry = False
        self.unlive_us: int | None = None
        self.bar_i = int(np.searchsorted(self.bars.visible_at, start_us // US, side="right"))
        self.spy_i = int(np.searchsorted(self.spy.visible_at, start_us // US, side="right"))
        # Built last, from the story as known at the start (the factory may keep it).
        self.pb: Playbook = make_playbook(self.pit(story))
        if self.pb.path_sessions != self.n:
            raise ValueError(
                f"{story.story_id}: the playbook runs {self.pb.path_sessions} sessions, "
                f"the path has {self.n}"
            )

    # ── helpers ──

    def pit(self, story: StoryView) -> PitStory:
        """The point-in-time view of ``story`` a playbook gets (one per story per run)."""
        view = self._pit.get(story.story_id)
        if view is None:
            view = self._pit[story.story_id] = PitStory(story, self._now_dt)
        return view

    def _now_dt(self) -> datetime:
        return from_us(self.now_us)

    def scale_of(self, k: int) -> float:
        if k < self.n:
            return self.scales[k]
        assert self.path.spare is not None
        a = self.ctx.adj(self.symbol, self.path.spare.day)
        return (a or self.a_s) / self.a_s

    def day_of(self, k: int) -> date:
        if k < self.n:
            return self.sessions[k].day
        assert self.path.spare is not None
        return self.path.spare.day

    def _ctx(self) -> Ctx:
        self.market.set_now(self.now_us)
        k = max(bisect.bisect_right(self.pre_open_us, self.now_us) - 1, 0)
        return Ctx(
            now=from_us(self.now_us),
            story=self.pit(self.story),
            sessions=self.sessions,
            k=k,
            market=self.market,
            position=self.position,
        )

    def _working_sell(self) -> bool:
        return any(o.side == "sell" for o in self.exchange.working)

    def _next_id(self) -> str:
        self.n_orders += 1
        return f"{self.story.story_id}:{self.n_orders}"

    def _note(self, inp: Input) -> None:
        """Deliver ``inp`` after the current event (rejections and cancels from intents)."""
        self.heap.push(self.now_us, Priority.FILL, (_NOTE, inp))

    # ── the loop ──

    def run(self) -> StoryOutcome:
        for k, s in enumerate(self.sessions):
            times = (
                self.pre_open_us[k],
                self.open_us[k],
                to_us(s.entry_start),
                to_us(s.entry_cutoff),
                to_us(s.flatten),
                self.close_us[k],
            )
            for what, at in zip(_SESSION_KINDS, times):
                if at >= self.start_us:
                    self.heap.push(at, Priority.SESSION, (_SESSION, (k, what)))
        for j, (t, _) in enumerate(self.news):
            self.heap.push(t, Priority.NEWS, (_NEWS, j))
        self._push_bar()
        self._push_spy()

        self._apply(self.pb.start(self._ctx()), self.pb.state())
        while self.heap and not self.done:
            at, _, (kind, data) = self.heap.pop()
            self.now_us = at
            self._handle(kind, data)
            if self.finished and self.qty == 0 and not self.exchange.working:
                self.done = True
        return self._outcome()

    def _push_bar(self) -> None:
        if self.bar_i < len(self.bars):
            at = int(self.bars.visible_at[self.bar_i]) * US
            if at <= self.end_us + 60 * US:
                self.heap.push(at, Priority.BAR, (_BAR, self.bar_i))

    def _push_spy(self) -> None:
        if self.spy_i < len(self.spy):
            at = int(self.spy.visible_at[self.spy_i]) * US
            if at <= self.end_us + 60 * US:
                self.heap.push(at, Priority.SPY_BAR, (_SPY, self.spy_i))

    def _handle(self, kind: int, data: Any) -> None:
        now = from_us(self.now_us)
        if kind == _EXCH:
            order_id, i = data
            self._exchange_test(order_id, i)
        elif kind == _NOTE:
            assert isinstance(data, FillIn | OrderClosedIn)
            self._deliver(data)
            if isinstance(data, FillIn) and data.side == "buy" and self.stop_after_entry:
                self.done = True
        elif kind == _SESSION:
            k, what = data
            self._session(k, what)
        elif kind == _SPY:
            i = int(data)
            self.spy_i = i + 1
            self._push_spy()
            self._deliver(BarIn(now, SPY, i))
        elif kind == _BAR:
            i = int(data)
            self.bar_i = i + 1
            self._push_bar()
            gap = self._gap_before(i)
            if gap is not None:
                self._deliver(gap)
            self._deliver(BarIn(now, self.symbol, i))
        elif kind == _NEWS:
            _, source = self.news[int(data)]
            own = source.story_id == self.story.story_id
            self._deliver(NewsIn(now, self.pit(source), own))
        elif kind == _TIMER:
            self._deliver(TimerIn(now, str(data)))

    def _gap_before(self, i: int) -> GapIn | None:
        b = self.bars
        k = int(b.k[i])
        ts = int(b.ts[i])
        gap_s = int(self.cfg.gap.total_seconds())
        now = from_us(self.now_us)
        if i == 0 or int(b.k[i - 1]) != k:
            if ts * US >= self.open_us[k] + gap_s * US:
                return GapIn(now, self.symbol, None, from_us(ts * US), True)
            return None
        prev = int(b.ts[i - 1])
        if ts - prev >= gap_s:
            return GapIn(now, self.symbol, from_us(prev * US), from_us(ts * US), False)
        return None

    def _deliver(self, inp: Input) -> None:
        if self.finished or self.done:
            return
        before = self.pb.state()
        self._apply(self.pb.on(inp, self._ctx()), before)

    def _apply(self, intents: Sequence[Intent], before: str) -> None:
        now = from_us(self.now_us)
        label = before
        for it in intents:
            if self.keep:
                self.intents.append((now, it))
            if isinstance(it, Transition):
                if self.keep:
                    self.transitions.append((now, str(label), str(it.to), it.reason))
                if self.armed_at is None and str(it.to).casefold() == ARMED:
                    self.armed_at = now
                if it.reason == TRIGGERED and self.triggered_at is None:
                    self.triggered_at = now
                label = it.to
                self.last_reason = it.reason
            elif isinstance(it, Submit):
                self._submit(it, by_sim=False)
            elif isinstance(it, Cancel):
                self._cancel(it)
            elif isinstance(it, SetTimer):
                self.heap.push(max(to_us(it.at), self.now_us), Priority.TIMER, (_TIMER, it.tag))
            elif isinstance(it, Finish):
                self._finish(it.reason)
        if self.unlive_us is None and (self.finished or not self.pb.live()):
            self.unlive_us = self.now_us

    # ── orders ──

    def _reject(self, order_id: str, side: Literal["buy", "sell"], why: str) -> None:
        self._note(OrderClosedIn(from_us(self.now_us), order_id, side, "rejected", why))

    def _submit(self, sub: Submit, *, by_sim: bool) -> None:
        order_id = self._next_id()
        if (why := rules.kind_reason(sub.kind, self.cfg.allowed_kinds)) is not None:
            self._reject(order_id, sub.side, why)
            return
        if sub.side == "buy":
            if sub.facts is None:
                self._reject(order_id, "buy", rules.NO_FACTS)
                return
            s = self.sessions[0]
            held = (
                self.qty > 0
                or any(o.side == "buy" for o in self.exchange.working)
                or self.state.holds(self.now_us)
            )
            if self.fill.gate_only:
                # A legacy study's replication (sim.run allows it under a gate unlock only).
                why = rules.POSITION_HELD if held else None
            else:
                why = rules.admit_buy(
                    verdict=self.ctx.screen_verdict(self.symbol, s.day),
                    decided_us=self.now_us,
                    session=s,
                    held=held,
                )
            active = rules.buy_active_at(self.now_us, self.lag_us, s)
            if why is not None or active is None:
                self._reject(order_id, "buy", why or rules.MARKET_CLOSED)
                return
            order = WorkingOrder(
                order_id, "buy", sub.kind, 0, self.now_us, active, None, sub.reason, sub.tag
            )
            self.facts = sub.facts
            self.entry_decided_at = from_us(self.now_us)
        else:
            working = sum(o.qty or 0.0 for o in self.exchange.working if o.side == "sell")
            qty = rules.clamp_sell(sub.qty, self.qty, working)
            if self.qty <= 0 or qty <= 0:
                self._reject(order_id, "sell", rules.NO_POSITION)
                return
            if qty < self.qty - working:
                # A trade record holds one exit fill; scaling out is a later trial.
                self._reject(order_id, "sell", rules.PARTIAL_EXIT)
                return
            placed = rules.sell_active_at(self.now_us, self.lag_us, self.sessions)
            if placed is None:
                self._reject(order_id, "sell", rules.MARKET_CLOSED)
                return
            k, active = placed
            order = WorkingOrder(
                order_id, "sell", sub.kind, k, self.now_us, active, qty, sub.reason, sub.tag, by_sim
            )
        self.exchange.submit(order)
        self._schedule(order)

    def _schedule(self, order: WorkingOrder) -> None:
        i = self.exchange.due(self.bars, order)
        if i is None:
            self.due.pop(order.order_id, None)
            return
        self.due[order.order_id] = i
        at = (int(self.bars.ts[i]) + 60) * US
        self.heap.push(at, Priority.EXCHANGE, (_EXCH, (order.order_id, i)))

    def _cancel(self, c: Cancel) -> None:
        for o in self.exchange.working:
            if (c.order_id is not None and o.order_id == c.order_id) or (
                c.tag is not None and o.tag == c.tag
            ):
                self.exchange.cancel(o.order_id)
                self.due.pop(o.order_id, None)
                self._note(
                    OrderClosedIn(from_us(self.now_us), o.order_id, o.side, "cancelled", "cancel")
                )

    def _cancel_buys(self, why: str, *, inline: bool) -> None:
        for o in self.exchange.working:
            if o.side == "buy":
                self.exchange.cancel(o.order_id)
                self.due.pop(o.order_id, None)
                note = OrderClosedIn(from_us(self.now_us), o.order_id, "buy", "cancelled", why)
                if inline:
                    self._deliver(note)
                else:
                    self._note(note)

    def _finish(self, reason: str) -> None:
        self.finished = True
        self.finish_reason = reason
        self._cancel_buys("finish", inline=False)
        if self.qty > 0 and not self._working_sell():
            self._submit(Submit("sell", reason="time_stop"), by_sim=True)

    def _exchange_test(self, order_id: str, i: int) -> None:
        if self.due.get(order_id) != i or self.exchange.get(order_id) is None:
            return  # cancelled, filled, or re-timed since
        k = int(self.bars.k[i])
        for ex in self.exchange.on_bar(i, self.bars, session_open_us=self.open_us[k]):
            self.due.pop(ex.order.order_id, None)
            spy_px, spy_flags = self.fill.spy_price(self.spy, ex)
            self._executed(ex, spy_px, spy_flags, ex.k)
            fill = self._fill_in(ex, ex.filled_at_us + US)
            self.heap.push(ex.filled_at_us + US, Priority.FILL, (_NOTE, fill))

    def _fill_in(self, ex: Execution, at_us: int) -> FillIn:
        o = ex.order
        qty = self.cfg.reference_notional / ex.price if o.side == "buy" else (o.qty or 0.0)
        return FillIn(
            from_us(at_us),
            o.order_id,
            o.side,
            ex.price,
            ex.price * self.scale_of(ex.k),
            qty,
            from_us(ex.bar_ts * US) if ex.bar_ts is not None else None,
            ex.flags,
            o.reason,
        )

    def _executed(self, ex: Execution, spy_px: float, spy_flags: tuple[str, ...], k: int) -> None:
        if ex.order.side == "buy":
            self.qty = self.cfg.reference_notional / ex.price
            self.entry = ex
            self.entry_spy = spy_px
            self.entry_flags = (*ex.flags, *spy_flags)
            if self.stop_at == "entry":
                self.stop_after_entry = True
        else:
            self.qty = max(self.qty - (ex.order.qty or 0.0), 0.0)
            self.exit = ex
            self.exit_spy = spy_px
            self.exit_flags = (*ex.flags, *spy_flags)
            self.exit_k = k

    # ── sessions ──

    def _session(self, k: int, kind: SessionKind) -> None:
        s = self.sessions[k]
        now = from_us(self.now_us)
        if kind == "pre_open" and k >= 1 and self.qty > 0 and not self.fill.gate_only:
            verdict = self.ctx.screen_verdict(self.symbol, s.day)
            if verdict != "halal":
                self._cancel_buys("compliance", inline=True)
                if not self._working_sell():
                    self._submit(Submit("sell", reason="compliance"), by_sim=True)
                self._deliver(ComplianceIn(now, s.day, verdict))
        elif kind == "flatten" and k == self.deadline and not self.fill.gate_only:
            self._cancel_buys("flatten", inline=True)
            if self.qty > 0 and not self._working_sell():
                self._submit(Submit("sell", reason="time_stop"), by_sim=True)
        elif kind == "close":
            self._close(k)
        self._deliver(SessionIn(now, kind, k, s.day))
        if kind == "close" and k == self.deadline:
            self.done = True

    def _close(self, k: int) -> None:
        now = from_us(self.now_us)
        for o in self.exchange.expire(k):
            self.due.pop(o.order_id, None)
            if o.side == "buy":
                self._deliver(OrderClosedIn(now, o.order_id, "buy", "expired", "close"))
            elif k < self.deadline:
                carried = replace(o, k=k + 1, active_at_us=self.open_us[k + 1] + self.lag_us)
                self.exchange.submit(carried)
                self._schedule(carried)
            else:
                self._fallback(o, k)
        if k == self.deadline and self.qty > 0:
            # Held at the last close with no exit working: the time stop, by the fallback.
            order = WorkingOrder(
                self._next_id(),
                "sell",
                OrderKind.MARKET,
                k,
                self.now_us,
                self.now_us,
                self.qty,
                "time_stop",
                "",
                True,
            )
            self._fallback(order, k)

    def _fallback(self, o: WorkingOrder, k: int) -> None:
        """Fill an exit still open at the deadline's close: the fill model's way, else
        the market rule's fallback (``exchange.unfilled_exit``)."""
        close = self.close_us[k]
        own = self.fill.at_close(self.bars, self.spy, k)
        if own is not None:
            ex = Execution(o, k, own.i, own.bar_ts, own.price, close, own.flags)
            self._executed(ex, own.spy_price, (), k)
            self._deliver(self._fill_in(ex, self.now_us))
            return
        day = self.sessions[k].day
        dp = self.ctx.daily(self.symbol, day)
        spy_dp = self.ctx.daily(SPY, day)
        spare, spy_spare = self._spare_series()
        spare_open = to_us(self.path.spare.open) if self.path.spare is not None else None
        last = len(self.bars) - 1
        lk = int(self.bars.k[last])
        fb = unfilled_exit(
            official_close=float(dp.close) if dp is not None else None,
            spy_close=float(spy_dp.close) if spy_dp is not None else self._spy_last_close(k),
            spare=spare,
            spy_spare=spy_spare,
            spare_open_us=spare_open,
            lag_us=self.lag_us,
            gap_us=span_us(self.cfg.gap),
            last_close=float(self.bars.c[last]),
            spy_last=self._spy_mark(int(self.bars.ts[last]), lk),
        )
        if fb.rule == "close_fallback":
            ex = Execution(o, k, -1, None, fb.price, close, fb.flags)
            exit_k = k
        elif fb.rule == "no_market":
            assert fb.bar_ts is not None and spare is not None and spare_open is not None
            moved = replace(o, k=self.n, active_at_us=spare_open + self.lag_us)
            ex = Execution(moved, self.n, -1, fb.bar_ts, fb.price, (fb.bar_ts + 60) * US, fb.flags)
            self.exit_spare = spare
            exit_k = self.n
        else:  # marked at the last trade
            ex = Execution(o, lk, last, None, fb.price, close, fb.flags)
            exit_k = lk
        self._executed(ex, fb.spy_price, (), exit_k)
        assert self.exit is not None
        self._deliver(self._fill_in(self.exit, self.now_us))

    def _spare_series(self) -> tuple[BarSeries | None, BarSeries | None]:
        """The spare session's bars (stock, SPY) in S units' scale, if loaded and adjustable."""
        spare, arrays = self.path.spare, self.path.spare_bars
        if spare is None or arrays is None or len(arrays) == 0:
            return None, None
        a = self.ctx.adj(self.symbol, spare.day)
        b = self.ctx.adj(SPY, spare.day)
        spy_arrays = self.spy_data.get(spare.day)
        if a is None or b is None or spy_arrays is None:
            return None, None
        lag_s = int(self.cfg.feed.bar_lag.total_seconds())
        return (
            BarSeries.build([arrays], [a / self.a_s], lag_s),
            BarSeries.build([spy_arrays], [b / self.b_s], lag_s),
        )

    def _spy_mark(self, ts: int, k: int) -> tuple[float, tuple[str, ...]]:
        """SPY's close of the bar starting at ``ts`` (else its proxy by ``spy_fill``)."""
        j = int(np.searchsorted(self.spy.ts, ts, side="left"))
        if j < len(self.spy) and int(self.spy.ts[j]) == ts:
            return float(self.spy.c[j]), ()
        return spy_fill(self.spy, ts, k, use_open=False)

    def _spy_last_close(self, k: int) -> float:
        idx = np.nonzero(self.spy.k == k)[0]
        return float(self.spy.c[idx[-1]]) if len(idx) else math.nan

    # ── the record ──

    def _outcome(self) -> StoryOutcome:
        nsn = self.story.nsn_at(self.sessions[0].entry_cutoff)
        trade: TradeRecord | None = None
        legs: tuple[Leg, ...] = ()
        if self.stop_at == "end" and self.entry is not None and self.exit is not None:
            trade, legs = self._trade(nsn)
        reason = self.finish_reason if self.finish_reason else self.last_reason
        entry_bar = (
            from_us(self.entry.bar_ts * US)
            if self.entry is not None and self.entry.bar_ts is not None
            else None
        )
        return StoryOutcome(
            story_id=self.story.story_id,
            symbol=self.symbol,
            session=self.sessions[0].day,
            terminal_state=str(self.pb.state()),
            reason=reason,
            skip=None,
            triggered_at=self.triggered_at,
            armed_at=self.armed_at,
            trade=trade,
            transitions=tuple(self.transitions),
            start_at=from_us(self.start_us),
            nsn_at=nsn,
            entry_decided_at=self.entry_decided_at,
            entry_bar_ts=entry_bar,
            legs=legs,
            intents=tuple(self.intents),
        )

    @property
    def busy_until_us(self) -> int:
        """Until when this story kept its symbol busy."""
        until = self.unlive_us if self.unlive_us is not None else self.now_us
        if self.assume_full_hold and self.entry is not None:
            until = max(until, self.end_us)
        return until

    @property
    def held_interval(self) -> tuple[int, int] | None:
        if self.entry is None:
            return None
        if self.exit is not None:
            return self.entry.filled_at_us, self.exit.filled_at_us
        if self.assume_full_hold:
            return self.entry.filled_at_us, self.end_us
        return None

    def _close_s(self, k: int, *, spy: bool) -> float | None:
        """Session k's official close in S units (the last bar's close without a daily bar)."""
        symbol = SPY if spy else self.symbol
        day = self.day_of(k)
        base = self.b_s if spy else self.a_s
        a = self.ctx.adj(symbol, day)
        dp = self.ctx.daily(symbol, day)
        if dp is not None and a is not None:
            return float(dp.close) * a / base
        series = self.spy if spy else self.bars
        idx = np.nonzero(series.k == k)[0]
        if len(idx):
            return float(series.c[idx[-1]] * series.scale[idx[-1]])
        return None

    def _trade(self, nsn: datetime | None) -> tuple[TradeRecord, tuple[Leg, ...]]:
        entry, exit_ = self.entry, self.exit
        assert entry is not None and exit_ is not None
        facts = self.facts
        assert facts is not None
        e_day, x_day = self.day_of(entry.k), self.day_of(self.exit_k)
        a_e = self.ctx.adj(self.symbol, e_day) or self.a_s
        a_x = self.ctx.adj(self.symbol, x_day) or a_e
        b_e = self.ctx.adj(SPY, e_day) or self.b_s
        b_x = self.ctx.adj(SPY, x_day) or b_e
        r_gross = (exit_.price * a_x) / (entry.price * a_e) - 1.0
        r_spy = (self.exit_spy * b_x) / (self.entry_spy * b_e) - 1.0
        notional = self.cfg.reference_notional

        def side_bps(ex: Execution, series: BarSeries | None, i: int) -> float:
            window = series is not None and i >= 0 and in_surcharge_window(series, i)
            return one_way_bps(
                facts.cost_bps,
                self.cfg.cost,
                rank=facts.rank,
                surcharge_window=window,
                sigma=facts.sigma,
                adv20_usd=facts.adv20_usd,
                notional=notional,
            )

        c_e = side_bps(entry, self.bars, entry.i)
        if exit_.i >= 0:
            c_x = side_bps(exit_, self.bars, exit_.i)
        elif "no_market" in exit_.flags and self.exit_spare is not None and exit_.bar_ts:
            j = int(np.searchsorted(self.exit_spare.ts, exit_.bar_ts, side="left"))
            c_x = side_bps(exit_, self.exit_spare, j)
        else:
            c_x = side_bps(exit_, None, -1)
        total = c_e + c_x
        r_net = r_gross - r_spy - total / 1e4
        r_beta = r_gross - facts.beta * r_spy - total / 1e4

        flags: list[str] = []
        for f in (*self.entry_flags, *self.exit_flags):
            if f not in flags:
                flags.append(f)

        entry_bar = from_us((entry.bar_ts or 0) * US)
        exit_bar = from_us(exit_.bar_ts * US) if exit_.bar_ts is not None else None
        exit_mark_us = exit_.bar_ts * US if exit_.bar_ts is not None else exit_.filled_at_us
        legs = self._legs(entry, exit_, a_e, a_x, b_e, b_x, c_e / 1e4, c_x / 1e4)
        mae, mfe = self._excursions(entry, exit_)
        participation = entry.participation
        if math.isfinite(exit_.participation):
            participation = max(participation, exit_.participation)
        trade = TradeRecord(
            run_id=self.run_id,
            story_id=self.story.story_id,
            symbol=self.symbol,
            family_type=facts.family_type,
            cell=facts.cell,
            variant=facts.variant,
            feed=self.cfg.feed.name,
            session=self.sessions[0].day,
            exit_session=x_day,
            sessions_held=self.exit_k - entry.k + 1,
            start_case=self.story.start_case(),
            at_news=self.story.at_news(),
            nsn_at=nsn,
            anchor_ts=facts.anchor_ts,
            entry_decided_at=entry.order.decided_at,
            entry_active_at=entry.order.active_at,
            entry_bar_ts=entry_bar,
            exit_decided_at=exit_.order.decided_at,
            exit_active_at=exit_.order.active_at,
            exit_bar_ts=exit_bar,
            p0=facts.p0,
            spy0=facts.spy0,
            sigma=facts.sigma,
            thr=facts.thr,
            low_star=facts.low_star,
            target=facts.target,
            entry_px=entry.price,
            exit_px=exit_.price,
            adj_entry=a_e,
            adj_exit=a_x,
            spy_entry_px=self.entry_spy,
            spy_exit_px=self.exit_spy,
            spy_adj_entry=b_e,
            spy_adj_exit=b_x,
            cost_bps=total / 2.0,
            rank=facts.rank,
            tech=facts.tech,
            beta=facts.beta,
            r_gross=r_gross,
            r_spy=r_spy,
            r_net_abn=r_net,
            r_beta_adj=r_beta,
            exit_reason=exit_.order.reason or "unspecified",
            mae=mae,
            mfe=mfe,
            hold_minutes=self._hold_minutes((entry.bar_ts or 0) * US, exit_mark_us),
            participation=participation,
            flags=tuple(flags),
        )
        return trade, legs

    def _legs(
        self,
        entry: Execution,
        exit_: Execution,
        a_e: float,
        a_x: float,
        b_e: float,
        b_x: float,
        c_e: float,
        c_x: float,
    ) -> tuple[Leg, ...]:
        k0, k1 = entry.k, self.exit_k
        out: list[Leg] = []
        prev_a = entry.price * a_e / self.a_s
        prev_b = self.entry_spy * b_e / self.b_s
        for k in range(k0, k1 + 1):
            if k == k1:
                z = exit_.price * a_x / self.a_s
                zb = self.exit_spy * b_x / self.b_s
            else:
                z = self._close_s(k, spy=False) or prev_a
                zb = self._close_s(k, spy=True) or prev_b
            cost = (c_e if k == k0 else 0.0) + (c_x if k == k1 else 0.0)
            out.append(Leg(self.day_of(k), prev_a, z, prev_b, zb, cost))
            prev_a, prev_b = z, zb
        return tuple(out)

    def _excursions(self, entry: Execution, exit_: Execution) -> tuple[float, float]:
        """Worst and best excursion from the entry, in S units, over the bars held."""
        i0 = entry.i
        if exit_.i >= 0:
            i1 = exit_.i
        else:
            idx = np.nonzero(self.bars.k <= min(self.exit_k, self.n - 1))[0]
            i1 = int(idx[-1]) if len(idx) else i0
        b = self.bars
        entry_s = entry.price * float(b.scale[i0])
        lows = b.l[i0 : i1 + 1] * b.scale[i0 : i1 + 1]
        highs = b.h[i0 : i1 + 1] * b.scale[i0 : i1 + 1]
        return float(lows.min()) / entry_s - 1.0, float(highs.max()) / entry_s - 1.0

    def _hold_minutes(self, t0: int, t1: int) -> int:
        """Regular-session minutes between two instants across the path's sessions."""
        spans = list(zip(self.open_us, self.close_us))
        if self.path.spare is not None:
            spans.append((to_us(self.path.spare.open), to_us(self.path.spare.close)))
        total = sum(max(0, min(t1, c) - max(t0, o)) for o, c in spans)
        return round(total / (60 * US))


# ── per symbol ────────────────────────────────────────────────


def _story_key(s: StoryView) -> tuple[date, str]:
    return s.session, s.story_id


def news_items(stories: Sequence[StoryView]) -> list[tuple[int, StoryView]]:
    """Every item time of ``stories``, as ``(available_at_us, story)``, in time order."""
    out = [(to_us(t), s) for s in sorted(stories, key=_story_key) for t in s.news_times()]
    out.sort(key=lambda x: (x[0], x[1].story_id))
    return out


def _closed(
    story: StoryView, state: str, reason: str, skip: str | None, start: datetime | None
) -> StoryOutcome:
    return StoryOutcome(
        story_id=story.story_id,
        symbol=story.symbol,
        session=story.session,
        terminal_state=state,
        reason=reason,
        skip=skip,
        triggered_at=None,
        armed_at=None,
        trade=None,
        start_at=start,
        nsn_at=story.nsn_at(Session.of(story.session).entry_cutoff),
    )


def simulate_symbol(
    symbol: str,
    stories: Sequence[StoryView],
    make_playbook: MakePlaybook,
    paths: Mapping[str, PathData | PathSkip],
    spy: SpyData,
    ctx: ContextView,
    cfg: SimConfig,
    *,
    stop_at: StopAt = "end",
    assume_full_hold: bool = False,
    keep_transitions: bool = False,
    news_from: Sequence[StoryView] | None = None,
    state: SymbolState | None = None,
    run_id: str = "",
) -> list[StoryOutcome]:
    """Simulate ``symbol``'s stories in start order; one outcome per started story.

    ``stories`` are the stories to run (the ones without a start are
    skipped); ``news_from`` (default: ``stories``) are every story of the
    symbol whose items a live playbook hears. ``state`` carries the
    symbol's blocking across calls (batches) and is updated in place.

    ``make_playbook`` is called once per story that runs (never for a
    blocked or skipped one), with the story as :class:`PitStory` at its
    start; a :class:`~halabot.playbooks.playbook.PlaybookFactory` or any
    such callable.
    """
    state = state if state is not None else SymbolState()
    news = news_items(news_from if news_from is not None else stories)
    started: list[tuple[int, StoryView, datetime]] = []
    for story in stories:
        if story.symbol != symbol:
            raise ValueError(f"{story.story_id} is not a story of {symbol}")
        st = start_time(story)
        if st is not None:
            started.append((to_us(st), story, st))
    started.sort(key=lambda x: (x[0], x[1].story_id))
    out: list[StoryOutcome] = []
    for start_us, story, start in started:
        if start_us < state.busy_until_us:
            out.append(_closed(story, "DISMISSED", "blocked_open", None, start))
            continue
        item = paths.get(story.story_id) or PathSkip(story.story_id, "units_missing")
        if isinstance(item, PathSkip):
            out.append(_closed(story, "SKIPPED", item.reason, item.reason, start))
            continue
        run = _StoryRun(
            story,
            make_playbook,
            item,
            spy,
            ctx,
            cfg,
            start_us,
            news,
            state,
            stop_at=stop_at,
            assume_full_hold=assume_full_hold,
            keep_transitions=keep_transitions,
            run_id=run_id,
        )
        out.append(run.run())
        state.busy_until_us = max(state.busy_until_us, run.busy_until_us)
        if (held := run.held_interval) is not None:
            state.held.append(held)
    return out


def worker_of(symbol: str, workers: int) -> int:
    """The worker a symbol runs in: ``crc32(symbol) % workers`` (stable across runs)."""
    return zlib.crc32(symbol.encode()) % max(workers, 1)


def simulate_many(
    stories: Sequence[StoryView],
    make_playbook: MakePlaybook,
    paths: Mapping[str, PathData | PathSkip],
    spy: SpyData,
    ctx: ContextView,
    cfg: SimConfig,
    *,
    workers: int = 1,
    stop_at: StopAt = "end",
    assume_full_hold: bool = False,
    keep_transitions: bool = False,
    news_from: Sequence[StoryView] | None = None,
    states: dict[str, SymbolState] | None = None,
    run_id: str = "",
) -> list[StoryOutcome]:
    """Every symbol of ``stories``, partitioned over ``workers`` (in this process).

    The partition is the one :func:`run` uses; outcomes come back sorted by
    (session, story_id), identical for any ``workers``.
    """
    states = states if states is not None else {}
    by_symbol = _group(stories)
    carriers = _group(news_from) if news_from is not None else by_symbol
    out: list[StoryOutcome] = []
    for w in range(max(workers, 1)):
        for symbol in sorted(s for s in by_symbol if worker_of(s, workers) == w):
            out += simulate_symbol(
                symbol,
                by_symbol[symbol],
                make_playbook,
                paths,
                spy,
                ctx,
                cfg,
                stop_at=stop_at,
                assume_full_hold=assume_full_hold,
                keep_transitions=keep_transitions,
                news_from=carriers.get(symbol, by_symbol[symbol]),
                state=states.setdefault(symbol, SymbolState()),
                run_id=run_id,
            )
    out.sort(key=lambda o: (o.session, o.story_id))
    return out


def _group(stories: Sequence[StoryView]) -> dict[str, list[StoryView]]:
    out: dict[str, list[StoryView]] = {}
    for s in sorted(stories, key=_story_key):
        out.setdefault(s.symbol, []).append(s)
    return out


# ── the driver ────────────────────────────────────────────────


# The loader's data rules (``loader.py``), which no legacy study applied: a skip for a
# stale or unreadable adjustment, and one for more than 5 bars failing the sanity rule.
DATA_SKIPS: Final = ("adjust_defect", "bad_bars")


@dataclass(frozen=True, slots=True)
class RunSummary:
    """What a run did, by count, and the story ids a legacy replication must set apart.

    ``skip_ids`` lists every skipped story by its reason, and
    ``bar_drop_ids`` the stories simulated although the bar sanity rule
    removed bars from their path (1 to 5 of the symbol's, or any of SPY's on
    a path session). The legacy studies read every stored row and skip
    nothing for data reasons, so these are the stories where a gate-only
    replication can differ from its study by construction (a missing entry
    or exit bar, another first bar after the decision): :meth:`data_filtered`
    gathers them for the R1 gate (spec §E.2), which sets them aside and
    counts them before it compares dropped sets and returns. The coverage
    skips (``units_missing``, ``halted_all_day``, ``spy_missing``,
    ``no_daily``) are listed too: a study run on the same stored rows drops
    or computes those headlines on its own terms, and the gate reports any
    it kept.
    """

    run_id: str
    stories: int  # stories given
    started: int  # with a start time (NSN by the entry cutoff)
    outcomes: int
    entries: int
    trades: int
    terminal: dict[str, int]  # "STATE/reason" -> count
    skips: dict[str, int]
    loader: dict[str, int]
    skip_ids: dict[str, tuple[str, ...]]  # reason -> sorted story ids
    bar_drop_ids: tuple[str, ...]  # sorted; simulated with bars dropped by the sanity rule

    def data_filtered(self) -> frozenset[str]:
        """The stories this run's data rules treated unlike a study reading every row:
        skipped as ``bad_bars`` or ``adjust_defect``, or simulated with bars dropped."""
        ids = {i for reason in DATA_SKIPS for i in self.skip_ids.get(reason, ())}
        return frozenset(ids | set(self.bar_drop_ids))

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "stories": self.stories,
            "started": self.started,
            "outcomes": self.outcomes,
            "entries": self.entries,
            "trades": self.trades,
            "terminal": dict(sorted(self.terminal.items())),
            "skips": dict(sorted(self.skips.items())),
            "loader": dict(sorted(self.loader.items())),
            "skip_ids": {k: list(v) for k, v in sorted(self.skip_ids.items())},
            "bar_drop_ids": list(self.bar_drop_ids),
        }


@dataclass(frozen=True, slots=True)
class _Shared:
    """What every worker needs for a whole run, passed explicitly (never module state).

    The serial path hands it to :func:`_work`; a process pool gives it to each
    worker once, through the pool's initializer (inherited under fork,
    unpickled once per worker under spawn), never per task. It holds no
    run id, so the pool can start before the run row is written.
    """

    by_id: dict[str, StoryView]
    carriers: dict[str, list[StoryView]]
    make_playbook: MakePlaybook
    ctx: ContextView
    cfg: SimConfig
    stop_at: StopAt
    assume_full_hold: bool
    keep_transitions: bool


_Job = tuple[
    str,  # the run id
    list[tuple[str, list[str]]],  # (symbol, story ids) in order
    dict[str, PathData | PathSkip],
    SpyData,
    dict[str, SymbolState],
]


def _work(shared: _Shared, job: _Job) -> tuple[list[StoryOutcome], dict[str, SymbolState]]:
    run_id, symbols, paths, spy, states = job
    out: list[StoryOutcome] = []
    for symbol, ids in symbols:
        out += simulate_symbol(
            symbol,
            [shared.by_id[i] for i in ids],
            shared.make_playbook,
            paths,
            spy,
            shared.ctx,
            shared.cfg,
            stop_at=shared.stop_at,
            assume_full_hold=shared.assume_full_hold,
            keep_transitions=shared.keep_transitions,
            news_from=shared.carriers[symbol],
            state=states.setdefault(symbol, SymbolState()),
            run_id=run_id,
        )
    return out, states


# A pool worker's run state: set once by its initializer, read by its tasks. Each run
# has its own pool, so a worker serves one run; the parent process never sets it.
_WORKER_STATE: _Shared | None = None


def _init_worker(shared: _Shared) -> None:
    global _WORKER_STATE
    _WORKER_STATE = shared


def _work_in_worker(job: _Job) -> tuple[list[StoryOutcome], dict[str, SymbolState]]:
    shared = _WORKER_STATE
    if shared is None:
        raise RuntimeError("a pool worker without its run's state (its initializer did not run)")
    return _work(shared, job)


def _noop() -> None:
    return None


def pool_method(parallel: Parallel, workers: int) -> Literal["fork", "spawn"] | None:
    """The start method of a run's process pool, or None to simulate in this process.

    * ``None`` (the default): a fork pool when ``workers > 1`` on Linux, and
      **serial on macOS**: forking a process that already runs asyncio and
      the database driver's threads is unsafe there (Python says so), and
      the fleet runs on a Mac until the server move;
    * ``False``: serial. ``True``: a pool by the platform's safe method
      (fork on Linux, spawn on macOS);
    * ``"fork"`` or ``"spawn"``: that method; fork is refused on macOS.

    Under spawn each worker unpickles the run's state once, so the factory,
    the stories and the context must pickle: a module-level factory, not a
    closure. The records are identical whichever is used.
    """
    if parallel is False or workers <= 1:
        return None
    darwin = sys.platform == "darwin"
    methods = multiprocessing.get_all_start_methods()
    if parallel is None:
        return "fork" if not darwin and "fork" in methods else None
    if parallel is True:
        return "fork" if not darwin and "fork" in methods else "spawn"
    if parallel == "fork" and darwin:
        raise ValueError(
            "a fork pool is unsafe on macOS: use parallel='spawn' (a picklable factory)"
        )
    if parallel not in methods:
        raise ValueError(f"process start method {parallel!r} is not available here")
    return parallel


def check_picklable(shared: _Shared) -> None:
    """Raise ``ValueError`` unless the run's state pickles, as a spawn pool needs.

    Under spawn each worker unpickles the factory, the context, the config
    and the stories; :func:`run` checks this before it writes anything, and
    names the part that fails.
    """
    from multiprocessing.reduction import ForkingPickler

    try:
        ForkingPickler.dumps(shared)
    except Exception as exc:
        parts: dict[str, object] = {
            "make_playbook": shared.make_playbook,
            "context": shared.ctx,
            "cfg": shared.cfg,
            "stories": list(shared.by_id.values()),
        }
        bad = []
        for name, obj in parts.items():
            try:
                ForkingPickler.dumps(obj)
            except Exception:
                bad.append(name)
        raise ValueError(
            f"a spawn pool pickles the run's state for each worker, and "
            f"{', '.join(bad) or 'it'} cannot be pickled ({exc!s}): use module-level "
            f"classes, not closures, or run serially (parallel=False)"
        ) from exc


def _pool(workers: int, method: Literal["fork", "spawn"], shared: _Shared) -> ProcessPoolExecutor:
    """A pool whose workers each hold ``shared`` from their start (``_init_worker``).

    Under fork all workers start at once, by the first submit, before the
    pool's own manager thread exists. The parent may still have idle threads
    (asyncio's resolver, the engine's pool), which is what Python's fork
    warning is about; the workers never touch them or the database: they
    only simulate and return records. Under spawn the first submit starts a
    worker, which unpickles ``shared``: a state that does not load there
    breaks the pool here, before the caller writes anything.
    """
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context(method),
        initializer=_init_worker,
        initargs=(shared,),
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*fork\(\) may lead to deadlocks", category=DeprecationWarning
            )
            pool.submit(_noop).result()
    except BaseException:
        pool.shutdown(cancel_futures=True)
        raise
    return pool


async def run(
    engine: AsyncEngine,
    stories: Sequence[StoryView],
    make_playbook: PlaybookFactory,
    *,
    context: ContextView,
    window: Window,
    window_end: date,
    cfg: SimConfig,
    unlock: WindowUnlock,
    sink: OutcomeSink,
    workers: int = 6,
    stop_at: StopAt = "end",
    assume_full_hold: bool = False,
    keep_transitions: bool = False,
    batch_paths: int = 250,
    parallel: Parallel = None,
) -> RunSummary:
    """Load paths behind the window guard, simulate every started story, write the outcomes.

    ``stories`` are every story of the symbols concerned (the ones that never
    start still bring their news to a live playbook). ``make_playbook`` gives
    the paths' length (``path_sessions``) and the playbook's name and version
    for the run row, and builds one playbook per story that runs.

    **Nothing is written until the run may proceed:** the unlock is verified
    against the ledger, the calendar is checked, and every day of every
    requested path is checked against the window guard (a path crossing
    ``window_end`` raises ``loader.WindowLocked`` here), and a process pool
    is started, all before ``sink.begin``; under spawn the run's state (the
    factory, the context, the config and the stories) must pickle
    (:func:`check_picklable`, a ``ValueError`` naming the part that does
    not) and load in a worker. A gate-only fill model (``legacy.py``) needs
    a gate unlock.

    Batches are simulated in session order; inside a batch the symbols are
    split over ``workers`` (:func:`pool_method` chooses processes or this
    one), and each symbol's blocking state carries from batch to batch. The
    run keeps no module state, so several runs may share an event loop.

    The summary lists the skipped story ids by reason and the stories
    simulated with bars dropped by the sanity rule (:class:`RunSummary`):
    a legacy replication (R1) sets :meth:`RunSummary.data_filtered` apart
    before it compares dropped sets.
    """
    fill = fill_model(cfg)
    if fill.gate_only and unlock.gate is None:
        raise ValueError(f"the {fill.name!r} fill model replicates a legacy study: gate runs only")
    method = pool_method(parallel, workers)
    by_symbol = _group(stories)
    started = [s for s in stories if start_time(s) is not None]
    n_sessions = make_playbook.path_sessions
    requests = [PathRequest(s.story_id, s.symbol, s.session, n_sessions) for s in started]
    loader = MinuteBarLoader(
        engine, window=window, window_end=window_end, unlock=unlock, batch_paths=batch_paths
    )
    await loader.prepare()
    loader.check(requests)
    shared = _Shared(
        by_id={s.story_id: s for s in stories},
        carriers=by_symbol,
        make_playbook=make_playbook,
        ctx=context,
        cfg=cfg,
        stop_at=stop_at,
        assume_full_hold=assume_full_hold,
        keep_transitions=keep_transitions,
    )
    if method == "spawn":
        check_picklable(shared)
    pool = _pool(workers, method, shared) if method is not None else None
    states: dict[str, SymbolState] = {}
    terminal: Counter[str] = Counter()
    skips: Counter[str] = Counter()
    skip_ids: dict[str, list[str]] = {}
    bar_drop_ids: list[str] = []
    n_out = entries = trades = n_batches = 0
    loop = asyncio.get_running_loop()
    try:
        run_id = await sink.begin(
            RunInfo(
                window=str(window),
                window_end=window_end,
                feed=cfg.feed.name,
                stop_at=stop_at,
                playbook=make_playbook.name,
                playbook_version=make_playbook.version,
                sim=cfg.as_config(),
            )
        )
        logger.info(
            "sim run %s: %d stories, %d paths",
            run_id,
            len(stories),
            len(requests),
            extra={
                "event": events.SIM_RUN_START,
                "run_id": run_id,
                "window": str(window),
                "stories": len(stories),
                "paths": len(requests),
                "pool": method or "serial",
            },
        )
        spy_all = await loader.spy()
        async for batch in loader.batches(requests, context):
            items = {x.story_id: x for x in batch}
            bar_drop_ids += [
                x.story_id
                for x in batch
                if isinstance(x, PathData)
                and (x.dropped or any(s.day in loader.spy_dropped for s in x.sessions))
            ]
            ids_by_symbol: dict[str, list[str]] = {}
            for sid in sorted(items, key=lambda i: _story_key(shared.by_id[i])):
                ids_by_symbol.setdefault(shared.by_id[sid].symbol, []).append(sid)
            jobs: list[_Job] = []
            for w in range(max(workers, 1)):
                symbols = sorted(s for s in ids_by_symbol if worker_of(s, workers) == w)
                if not symbols:
                    continue
                ids = [i for s in symbols for i in ids_by_symbol[s]]
                days = {
                    d
                    for i in ids
                    if isinstance(p := items[i], PathData)
                    for d in [s.day for s in p.sessions] + ([p.spare.day] if p.spare else [])
                }
                spy = spy_all.subset(days)
                jobs.append(
                    (
                        run_id,
                        [(s, ids_by_symbol[s]) for s in symbols],
                        {i: items[i] for i in ids},
                        spy,
                        {s: states.get(s, SymbolState()) for s in symbols},
                    )
                )
            if pool is not None:
                results = await asyncio.gather(
                    *(loop.run_in_executor(pool, _work_in_worker, j) for j in jobs)
                )
            else:
                results = [_work(shared, j) for j in jobs]
            outcomes: list[StoryOutcome] = []
            for out, new_states in results:
                outcomes += out
                states.update(new_states)
            outcomes.sort(key=lambda o: (o.session, o.story_id))
            for o in outcomes:
                terminal[f"{o.terminal_state}/{o.reason}"] += 1
                if o.skip is not None:
                    skips[o.skip] += 1
                    skip_ids.setdefault(o.skip, []).append(o.story_id)
                entries += o.entered
                trades += o.trade is not None
            n_out += len(outcomes)
            await sink.write(outcomes)
            n_batches += 1
            logger.info(
                "sim run %s: batch %d, %d outcomes so far",
                run_id,
                n_batches,
                n_out,
                extra={
                    "event": events.SIM_RUN_BATCH,
                    "run_id": run_id,
                    "batch": n_batches,
                    "outcomes": n_out,
                    "trades": trades,
                },
            )
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    summary = RunSummary(
        run_id=run_id,
        stories=len(stories),
        started=len(started),
        outcomes=n_out,
        entries=entries,
        trades=trades,
        terminal=dict(terminal),
        skips=dict(skips),
        loader=dict(loader.counts),
        skip_ids={k: tuple(sorted(v)) for k, v in skip_ids.items()},
        bar_drop_ids=tuple(sorted(bar_drop_ids)),
    )
    await sink.finish(summary.as_dict())
    logger.info(
        "sim run %s: %d outcomes, %d trades",
        run_id,
        n_out,
        trades,
        extra={
            "event": events.SIM_RUN_DONE,
            "run_id": run_id,
            "outcomes": n_out,
            "entries": entries,
            "trades": trades,
        },
    )
    return summary


__all__ = [
    "DATA_SKIPS",
    "MakePlaybook",
    "Parallel",
    "PitStory",
    "RunSummary",
    "SimConfig",
    "StopAt",
    "SymbolState",
    "news_items",
    "pool_method",
    "run",
    "simulate_many",
    "simulate_symbol",
    "start_time",
    "worker_of",
]
