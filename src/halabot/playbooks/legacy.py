"""Gate-only fill models and the daily-bar source (spec §E.2 R1, §E.3 S1).

They replicate two legacy studies inside the simulator, to show that its
machinery (the clock, the heap, the exchange, the records) reproduces them
to 1e-10 when given their rules. No trial uses them: their ``gate_only``
flag makes ``sim.run`` refuse them without a gate unlock.

**R1, the reactor study** (``events/intraday.run``): :func:`reactor_config`
(the HARNESS feed, no order lag) with :class:`LegacyReactorFill`:

* a buy fills on the first bar of its session with ``ts >= active_at``, at
  that bar's VWAP, or its open when the VWAP is missing: no clamp to the
  bar's range, no gap rule, no stale limit;
* SPY's leg is SPY's own first bar with ``ts >= active_at``, same rule;
* the exit is the session's last regular bar's close (and SPY's), taken at
  the deadline's close: sells never fill on a bar and there is no flatten.

Each headline is a :func:`reactor_story`, deciding at ``published + 60 s``
(the news-lag override, ``intraday.LATENCY``); the playbook is :class:`Hold`
(buy at the start, hold), built by :class:`HoldFactory`.

**How R1 compares dropped sets** (:func:`r1_dropped`). The pinned headline
set H (R0's, headline id -> (symbol, New York day)) runs as an explicit
set: ``sim.run(..., expected=H)`` with one :func:`reactor_story` per
headline, so every id of H is accounted for. The study (``intraday.run``
on H) drops a headline when its day is not a session, when no stock bar or
no SPY bar starts at or after its decision, or when the move is
implausible; the simulator gives each id of H without a finite return its
reason in ``RunSummary.dropped`` (``no_session``, ``entry_unfilled`` for a
decision with no stock bar after it, ``no_spy`` for one with no SPY bar
after it, ``halted_all_day``, ...). Then:

1. **Set aside** (:func:`r1_set_aside`, on both sides) the headlines the
   loader treated by rules the study never had (:data:`R1_SET_ASIDE`):

   * the skips ``adjust_defect`` (a stale or unreadable adjustment),
     ``bad_bars`` (more than 5 bars failing the sanity rule), ``no_daily``
     (no A-factor for the stock: the study never reads daily bars) and
     ``spy_thin`` (a SPY session with bars, but fewer than the loader's
     300: the study needs one after its decision);
   * ``bars_cut`` (``RunSummary.bar_drop_ids``): the sanity rule cut 1 to
     5 of the stock's bars on S, so the entry or the last bar can differ;
   * ``spy_bars_cut`` (``RunSummary.spy_drop_ids``): it cut any of SPY's
     bars on S. The whole day is set aside, wherever the cut bar lies
     relative to the decision: conservative, and its own reason so its
     cost in coverage shows per headline.

   A headline with several reasons lists them all. The set-aside is
   capped, pre-stated: more than :data:`R1_SET_ASIDE_CAP` (1%) of H set
   aside fails R1, whatever the comparison. Its count, by reason, goes
   with the gate row (:meth:`R1Dropped.as_config`).
2. **The study's plausibility filter** (``intraday._PLAUSIBLE`` on the
   raw exit/entry ratio) applies to the simulator's trades
   (:func:`reactor_plausible`): an implausible trade is dropped as
   ``implausible``.
3. **Compare** the rest: the simulator's dropped ids must equal the
   study's exactly. Each id dropped by one side only is listed
   (``sim_only`` with the simulator's reason, ``study_only``), and R1
   passes the dropped-set check only when both lists are empty and the
   set-aside is within its cap. The coverage skips (``units_missing``,
   ``spy_missing``, ``halted_all_day``) and ``blocked_open`` are compared,
   not set aside: the study on the same rows must drop those headlines too
   (no stock bar, no SPY bar), or the lists show them.
4. On the headlines both sides kept (:attr:`R1Dropped.kept`), the gate
   requires ``|Δr| <= 1e-10``.

``RunSummary.spare_drop_ids`` is not set aside: R1's exit is S's last bar
(``GateFill.at_close``), so the spare session is never read.

**S1, the daily-bar study** (``events/study.evaluate``): :func:`daily_config`
with :class:`DailyBarSource`, which turns an observation into a pseudo path
over the sessions from its entry to its exit. Each session has an 09:30
pseudo-bar at the adjusted open and a last pseudo-bar (close - 1 min) at
the adjusted close, every price of a pseudo-bar equal. The clock is
``study.entry_point``: an "open" entry decides at its session's open (so
the 09:30 pseudo-bar fills it), a "close" entry one second later (the close
pseudo-bar). The exit is the close pseudo-bar ``h`` sessions on
(``exit_i = i + h - 1`` for an open entry, ``i + h`` for a close entry).
The pseudo-bars are in adjusted ('all') prices, so the source is also the
run's ``ContextView``, with every A-factor 1. An event on an early-close
day published between that close and 16:00 gets ``entry_point``'s
"close" of a session already shut: :attr:`DailyEntry.lookahead` marks it,
for S1 to exclude and count.

Both models admit a buy without the halal screen or the entry window, and
a held position is never sold by the pre-open compliance check (R1's
headlines and S1's complement universe are not H1's universe, and a
replication takes the study's trades as they were): every exit is the
model's, recorded as ``time_stop``. Nothing here places an order.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Final, Literal

import numpy as np
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks.clock import HARNESS
from halabot.playbooks.exchange import FallbackFill, first_eligible
from halabot.playbooks.interfaces import CardView, StoryView
from halabot.playbooks.playbook import TRIGGERED, Ctx
from halabot.playbooks.records import StoryOutcome, TradeRecord
from halabot.playbooks.sim import RunSummary, simulate_symbol
from halabot.playbooks.types import (
    BarSeries,
    Execution,
    FillIn,
    Finish,
    Input,
    Intent,
    OrderClosedIn,
    PathData,
    PathSkip,
    Session,
    SimConfig,
    SkipReason,
    SpyData,
    Submit,
    TradeFacts,
    Transition,
    WorkingOrder,
)
from halal_trader.data.minutes import BarArrays
from halal_trader.events import study
from halal_trader.events.intraday import _PLAUSIBLE as STUDY_PLAUSIBLE  # read, not copied
from halal_trader.events.intraday import LATENCY
from halal_trader.market_hours import MARKET_TZ

SPY: Final = "SPY"
CLOSE_ENTRY_AFTER_OPEN: Final = timedelta(seconds=1)  # past the 09:30 pseudo-bar's start

# ── fill models ───────────────────────────────────────────────


def _vwap_or_open(bars: BarSeries, i: int) -> float:
    """A bar's price as the legacy studies read it: ``vwap or open``."""
    vw = float(bars.vw[i])
    return float(bars.o[i]) if math.isnan(vw) or vw == 0.0 else vw


def _last_of(bars: BarSeries, k: int) -> int | None:
    """Index of path session ``k``'s last bar, None when it has none."""
    lo, hi = bars.session_start(k), bars.session_start(k + 1)
    return hi - 1 if hi > lo else None


class GateFill:
    """The fill rules both legacy studies share (module docstring); gate runs only."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def gate_only(self) -> bool:
        return True

    def due(self, bars: BarSeries, order: WorkingOrder) -> int | None:
        if order.side != "buy":
            return None  # exits wait for the deadline's close (at_close)
        return first_eligible(bars, order.active_at_us, order.k)

    def price(
        self, bars: BarSeries, i: int, order: WorkingOrder, *, gap_us: int, session_open_us: int
    ) -> tuple[float, tuple[str, ...]]:
        return _vwap_or_open(bars, i), ()

    def spy_price(self, spy: BarSeries, ex: Execution) -> tuple[float, tuple[str, ...]]:
        j = first_eligible(spy, ex.order.active_at_us, ex.k)
        if j is None:
            return math.nan, ("no_spy",)
        return _vwap_or_open(spy, j), ()

    def at_close(self, bars: BarSeries, spy: BarSeries, k: int) -> FallbackFill | None:
        i = _last_of(bars, k)
        if i is None:
            return None  # no bar that session: the market rule's fallback
        j = _last_of(spy, k)
        spy_px = float(spy.c[j]) if j is not None else math.nan
        flags = ("last_close",) if j is not None else ("last_close", "no_spy")
        return FallbackFill("last_close", float(bars.c[i]), spy_px, int(bars.ts[i]), i, flags)


class LegacyReactorFill(GateFill):
    """R1: the reactor study's fills (``events/intraday.entry_and_close``)."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("legacy-reactor")


class DailyBarFill(GateFill):
    """S1: fills on :class:`DailyBarSource` pseudo-bars (every price of a bar is equal)."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("daily-bars")


def reactor_config() -> SimConfig:
    """R1's simulator: HARNESS bars, no order lag, the legacy fills, study costs."""
    return SimConfig(feed=HARNESS, order_lag=timedelta(0), fill=LegacyReactorFill())


def daily_config() -> SimConfig:
    """S1's simulator: no order lag (an open entry fills on the 09:30 pseudo-bar)."""
    return SimConfig(feed=HARNESS, order_lag=timedelta(0), fill=DailyBarFill())


# ── the harness: one observation, one story, one held position ──


@dataclass(frozen=True, slots=True)
class HarnessCard:
    family: str | None = "NSN_CORE"
    type: str = "harness"
    structural: bool = False
    vetoes: tuple[str, ...] = ()


_CARD: Final = HarnessCard()


@dataclass(frozen=True, slots=True)
class HarnessStory:
    """One observation of a legacy study as a story; it starts, and buys, at ``decide_at``.

    ``nsn_at`` answers ``decide_at`` whatever cutoff it is asked about: the
    study took every observation (a reactor headline after 15:00 still
    enters), so the start rule's entry cutoff does not apply. It carries no
    news items.
    """

    story_id: str
    symbol: str
    session: date
    decide_at: datetime
    published_at: datetime

    def card_at(self, t: datetime) -> CardView:
        return _CARD

    def nsn_at(self, cutoff: datetime) -> datetime | None:
        return self.decide_at

    def at_news(self) -> datetime | None:
        return self.published_at

    def start_case(self) -> str:
        s = Session.of(self.session)
        return "in" if s.open <= self.published_at < s.close else "out"

    def news_times(self) -> Sequence[datetime]:
        return ()


def reactor_story(story_id: str, symbol: str, published_at: datetime) -> HarnessStory:
    """R1's story of one headline: S is its New York day; it decides 60 s after it."""
    day = published_at.astimezone(MARKET_TZ).date()
    return HarnessStory(story_id, symbol, day, published_at + LATENCY, published_at)


class Hold:
    """The gate harness playbook: buys at its start and holds until the fill model's exit."""

    name = "gate-hold"
    version = "1"

    def __init__(self, path_sessions: int, facts: TradeFacts) -> None:
        self._n = path_sessions
        self._facts = facts
        self._state = "DETECTED"

    @property
    def path_sessions(self) -> int:
        return self._n

    def state(self) -> str:
        return self._state

    def live(self) -> bool:
        return self._state in ("ENTERING", "ENTERED")

    def _go(self, to: str, reason: str) -> Transition:
        self._state = to
        return Transition(to, reason)

    def start(self, ctx: Ctx) -> list[Intent]:
        return [self._go("ENTERING", TRIGGERED), Submit("buy", facts=self._facts)]

    def on(self, ev: Input, ctx: Ctx) -> list[Intent]:
        if isinstance(ev, FillIn):
            if ev.side == "buy":
                return [self._go("ENTERED", "filled")]
            return [self._go("EXITED", ev.reason), Finish(ev.reason)]
        if isinstance(ev, OrderClosedIn) and ev.side == "buy":
            return [self._go("EXPIRED", ev.reason), Finish("entry_unfilled")]
        return []


@dataclass(frozen=True, slots=True)
class HoldFactory:
    """Builds :class:`Hold` playbooks; ``facts`` by story id (costs differ by rank)."""

    path_sessions: int
    facts: Mapping[str, TradeFacts]
    name: str = Hold.name
    version: str = Hold.version

    def __call__(self, story: StoryView) -> Hold:
        return Hold(self.path_sessions, self.facts[story.story_id])


# ── R1: the dropped sets ──────────────────────────────────────

# The loader's skips the reactor study has no counterpart for (module docstring). Not
# ``types.DATA_SKIPS``, the Stage A data-skip set: ``no_daily`` is a coverage skip there,
# and ``units_missing`` / ``spy_missing`` are data skips that R1 compares.
R1_SET_ASIDE: Final = ("adjust_defect", "bad_bars", "no_daily", "spy_thin")
BARS_CUT: Final = "bars_cut"  # the sanity rule cut 1 to 5 of the stock's bars on S
SPY_BARS_CUT: Final = "spy_bars_cut"  # it cut SPY's bars on S: the whole day, conservatively
R1_SET_ASIDE_CAP: Final = 0.01  # pre-stated: more than 1% of the headlines set aside fails R1
IMPLAUSIBLE: Final = "implausible"  # a trade outside the study's _PLAUSIBLE exit/entry ratio


def r1_set_aside(summary: RunSummary) -> dict[str, tuple[str, ...]]:
    """Each headline R1 sets aside, with every reason (:data:`R1_SET_ASIDE`,
    :data:`BARS_CUT`, :data:`SPY_BARS_CUT`), sorted by id."""
    out: dict[str, list[str]] = {}
    for reason in R1_SET_ASIDE:
        for sid in summary.skip_ids.get(reason, ()):
            out.setdefault(sid, []).append(reason)
    for sid in summary.bar_drop_ids:
        out.setdefault(sid, []).append(BARS_CUT)
    for sid in summary.spy_drop_ids:
        out.setdefault(sid, []).append(SPY_BARS_CUT)
    return {sid: tuple(out[sid]) for sid in sorted(out)}


def reactor_plausible(trade: TradeRecord) -> bool:
    """The study's filter on a trade: ``lo < exit / entry < hi`` on raw prices
    (``intraday._PLAUSIBLE``; R1's path is the one session S)."""
    lo, hi = STUDY_PLAUSIBLE
    return lo < trade.exit_px / trade.entry_px < hi


@dataclass(frozen=True, slots=True)
class R1Dropped:
    """R1's dropped-set check (module docstring, steps 1 to 3).

    ``passed`` needs the set-aside within :data:`R1_SET_ASIDE_CAP` and no
    headline dropped by one side only; ``kept`` are the headlines both sides
    kept, on which the gate compares returns.
    """

    headlines: int
    set_aside: dict[str, tuple[str, ...]]  # id -> every reason
    sim_only: dict[str, str]  # dropped by the simulator alone: id -> its reason
    study_only: tuple[str, ...]  # dropped by the study alone
    kept: tuple[str, ...]  # sorted

    @property
    def compared(self) -> int:
        """Headlines outside the set-aside."""
        return self.headlines - len(self.set_aside)

    @property
    def share(self) -> float:
        """The set-aside's share of the headlines."""
        return len(self.set_aside) / self.headlines if self.headlines else 0.0

    @property
    def within_cap(self) -> bool:
        return self.share <= R1_SET_ASIDE_CAP

    @property
    def identical(self) -> bool:
        return not self.sim_only and not self.study_only

    @property
    def passed(self) -> bool:
        return self.within_cap and self.identical

    def as_config(self) -> dict[str, object]:
        """What the gate row records of the check (counts, the cap, every mismatch)."""
        by_reason = Counter(r for reasons in self.set_aside.values() for r in reasons)
        return {
            "headlines": self.headlines,
            "set_aside": len(self.set_aside),
            "set_aside_by_reason": dict(sorted(by_reason.items())),
            "set_aside_share": self.share,
            "set_aside_cap": R1_SET_ASIDE_CAP,
            "set_aside_rules": [*R1_SET_ASIDE, BARS_CUT, SPY_BARS_CUT],
            "compared": self.compared,
            "kept": len(self.kept),
            "sim_only": dict(sorted(self.sim_only.items())),
            "study_only": list(self.study_only),
        }


def r1_dropped(
    headlines: Collection[str],
    summary: RunSummary,
    trades: Iterable[TradeRecord],
    study_dropped: Iterable[str],
) -> R1Dropped:
    """Compare the simulator's dropped headlines with the study's (module docstring).

    ``headlines`` are H's ids, ``summary`` the explicit-set run over them
    (``sim.run(..., expected=H)``), ``trades`` its trade records, and
    ``study_dropped`` the ids ``intraday.run`` returned no outcome for.
    Raises ``ValueError`` when the run was not over H or an id is not H's.
    """
    ids = set(headlines)
    if summary.expected != len(ids):
        raise ValueError(
            f"the run accounted for {summary.expected} headlines, not H's {len(ids)}: "
            f"run it with expected=H"
        )
    study = set(study_dropped)
    sim = dict(summary.dropped)
    for t in trades:
        if t.story_id not in sim and not reactor_plausible(t):
            sim[t.story_id] = IMPLAUSIBLE
    stray = sorted((study | set(sim)) - ids)
    if stray:
        raise ValueError(f"{len(stray)} dropped ids are not in H: {', '.join(stray[:5])}")
    aside = r1_set_aside(summary)
    return R1Dropped(
        headlines=len(ids),
        set_aside=aside,
        sim_only={
            sid: why for sid, why in sorted(sim.items()) if sid not in study and sid not in aside
        },
        study_only=tuple(sorted(study - set(sim) - set(aside))),
        kept=tuple(sorted(ids - set(sim) - study - set(aside))),
    )


# ── S1: pseudo paths from daily bars ──────────────────────────


@dataclass(frozen=True, slots=True)
class DailyEntry:
    """Where ``study.entry_point`` enters an observation published at ``published``."""

    published: datetime
    index: int  # in the source's calendar
    session: date
    at: Literal["open", "close"]
    decide_at: datetime  # the open, or one second after it for a close entry
    lookahead: bool  # a "close" entry published after an early close: excluded by S1


class DailyBarSource:
    """S1's pseudo paths, on the calendar and adjusted prices of ``study.load_bars``.

    It is also those runs' ``ContextView``: prices are already adjusted, so
    every A-factor is 1; there is no official close to fall back to (every
    exit is a pseudo-bar's close) and no screen is read (the gate fill
    admits without one).
    """

    def __init__(self, bars: study.Bars) -> None:
        self._bars = bars
        self.sessions: list[date] = list(bars.sessions)

    @classmethod
    async def load(cls, engine: AsyncEngine, symbols: Sequence[str]) -> DailyBarSource:
        return cls(await study.load_bars(engine, symbols))

    # ContextView
    def adj(self, symbol: str, day: date) -> float | None:
        return 1.0

    def screen_verdict(self, symbol: str, day: date) -> str:
        return "no_screen"

    def daily(self, symbol: str, day: date) -> None:
        return None

    def entry(self, published: datetime) -> DailyEntry | None:
        """``study.entry_point`` on this calendar; None where it has no entry."""
        point = study.entry_point(published, self.sessions)
        if point is None:
            return None
        i, at = point
        day = self.sessions[i]
        s = Session.of(day)
        if at == "open":
            return DailyEntry(published, i, day, "open", s.open, False)
        lookahead = published >= s.close  # only after an early close, before 16:00
        return DailyEntry(published, i, day, "close", s.open + CLOSE_ENTRY_AFTER_OPEN, lookahead)

    def exit_index(self, entry: DailyEntry, horizon: int) -> int:
        return entry.index + horizon - (1 if entry.at == "open" else 0)

    def path(
        self, story_id: str, symbol: str, entry: DailyEntry, horizon: int
    ) -> PathData | PathSkip:
        """The pseudo path from the entry session to the exit ``horizon`` sessions on.

        ``no_daily`` (``spy_missing`` for SPY) where ``study.outcome`` has no
        return: the exit is past the calendar, or an entry or exit price is
        missing or zero.
        """
        last = self.exit_index(entry, horizon)
        if horizon < 1 or last >= len(self.sessions):
            return PathSkip(story_id, "no_daily")
        days = self.sessions[entry.index : last + 1]
        needs: tuple[tuple[str, SkipReason], ...] = ((symbol, "no_daily"), (SPY, "spy_missing"))
        for name, reason in needs:
            first = (self._bars.open if entry.at == "open" else self._bars.close).get(name, {})
            if not first.get(entry.session) or not self._bars.close.get(name, {}).get(days[-1]):
                return PathSkip(story_id, reason)
        return PathData(
            story_id=story_id,
            symbol=symbol,
            sessions=tuple(Session.of(d) for d in days),
            bars=tuple(self.pseudo(symbol, d) for d in days),
        )

    def spy(self, days: Iterable[date]) -> SpyData:
        return SpyData({d: self.pseudo(SPY, d) for d in days})

    def pseudo(self, symbol: str, day: date) -> BarArrays:
        """``symbol``'s pseudo-bars on ``day``: 09:30 at the adjusted open, close - 1 min at
        the adjusted close (each one present only if that price is). Volume is unknown (NaN)."""
        s = Session.of(day)
        rows: list[tuple[int, float]] = []
        o = self._bars.open.get(symbol, {}).get(day)
        c = self._bars.close.get(symbol, {}).get(day)
        if o:
            rows.append((int(s.open.timestamp()), float(o)))
        if c:
            rows.append((int(s.close.timestamp()) - 60, float(c)))
        ts = np.asarray([t for t, _ in rows], dtype=np.int64)
        px = np.asarray([p for _, p in rows], dtype=np.float64)
        nan = np.full(len(rows), np.nan, dtype=np.float64)
        return BarArrays(ts, px, px.copy(), px.copy(), px.copy(), nan, px.copy())

    def simulate(
        self, entry: DailyEntry, *, story_id: str, symbol: str, horizon: int, facts: TradeFacts
    ) -> StoryOutcome | PathSkip:
        """One observation at one horizon through the simulator (its ``trade.r_net_abn``
        is ``study.outcome``'s return when ``facts.cost_bps`` is the study's cost)."""
        p = self.path(story_id, symbol, entry, horizon)
        if isinstance(p, PathSkip):
            return p
        story = HarnessStory(story_id, symbol, entry.session, entry.decide_at, entry.published)
        (out,) = simulate_symbol(
            symbol,
            [story],
            HoldFactory(len(p.sessions), {story_id: facts}),
            {story_id: p},
            self.spy(s.day for s in p.sessions),
            self,
            daily_config(),
        )
        return out


__all__ = [
    "BARS_CUT",
    "IMPLAUSIBLE",
    "R1_SET_ASIDE",
    "R1_SET_ASIDE_CAP",
    "SPY_BARS_CUT",
    "DailyBarFill",
    "DailyBarSource",
    "DailyEntry",
    "GateFill",
    "HarnessCard",
    "HarnessStory",
    "Hold",
    "HoldFactory",
    "LegacyReactorFill",
    "R1Dropped",
    "daily_config",
    "r1_dropped",
    "r1_set_aside",
    "reactor_config",
    "reactor_plausible",
    "reactor_story",
]
