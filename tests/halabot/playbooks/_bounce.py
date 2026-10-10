"""Stories on the real taxonomy, pre-event context, paths and markets for the bounce tests.

:class:`TStory` labels itself with ``taxonomy.resolve`` (the H1 family, the
unclear and structural vetoes, followers by spec §A.5(b)), so the playbook
is tested against the rules it will run on, not a stand-in.

**The base path** (``base_story``, ``base_bars``; MON 2016-03-07, a
downgrade public at 07:00, so the story starts at the open, "out": P0 =
100, SPY0 = 200; σ 0.018, so thr = 0.036):

* 09:30 c 96 (D = -0.04: triggered when visible, 09:31:05); flat 96;
* 09:45 l 94 (L = 94, t_L 09:45); flat 94.8 to 10:04;
* 10:05 c 95.4 > AVWAP 3431.9/36 = 95.33, and 95.4 - 94 = 1.4 <= 0.25 x 6:
  armed (20 min quiet) and entered when visible, 10:06:05; the buy works at
  10:06:08 and fills on the 10:07 bar at its VWAP 95.45. L* = 94, TGT = 97;
* 10:30 c 97.0 >= TGT: the target, decided 10:31:05, filled on the 10:32 bar
  at 97.1. SPY's legs (``SPY_ROWS``): 200.5 at 10:07, 201.0 at 10:32.

**The "in" path** (``IN_LEVELS``, ``IN_ROWS``, ``IN_SPY``): flat 100 before
the news, 10:49 closes 100.2 (P0), SPY flat 201 (SPY0 201, not its previous
close 200); with σ 0.01, thr is the 0.03 floor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np

from halabot.playbooks.bounce import BounceFactory, BounceParams
from halabot.playbooks.records import StoryOutcome
from halabot.playbooks.sim import StopAt, simulate_symbol
from halabot.playbooks.types import PathData, Session, SimConfig, SpyData
from halal_trader.data.minutes import BarArrays
from halal_trader.events.context import Eligibility, PreEvent
from halal_trader.events.earnings_parse import EarningsFacts
from halal_trader.events.taxonomy import FAMILY, REACTIVE, StoryCard, resolve
from halal_trader.market_hours import next_trading_day, previous_trading_day
from tests.halabot.playbooks._support import (
    MON,
    TUE,
    Context,
    Item,
    Row,
    Story,
    daily,
    epoch,
    et,
    minutes_of,
    path,
    session_bars,
)
from tests.halabot.playbooks._synth import WEEK, Market, random_session

NEWS_LAG = timedelta(seconds=600)
EPS = 1e-12
COST = 15.0  # study.cost_bps for rank 300-999
BETA = 1.2

# ── stories ──


@dataclass(frozen=True, slots=True)
class TItem:
    """A story item as ``taxonomy.ItemLike``: public at ``at``, usable 600 s later."""

    event_id: int
    at: datetime
    itype: str
    facts: tuple[EarningsFacts, ...] = ()
    supersedes: tuple[int, ...] = ()

    @property
    def available_at(self) -> datetime:
        return self.at + NEWS_LAG


@dataclass(slots=True)
class TStory:
    """A StoryView labelled by ``taxonomy.resolve``.

    ``parent`` says an earlier non-follower story exists, so rule (b) of
    spec §A.5 applies: the story follows while every non-noise item known
    is reactive.
    """

    symbol: str
    session: date
    items: list[TItem] = field(default_factory=list)
    parent: bool = False

    @property
    def story_id(self) -> str:
        return f"{self.symbol}:{self.session.isoformat()}"

    def follower_at(self, t: datetime) -> bool:
        if not self.parent:
            return False
        known = [
            i for i in self.items if i.available_at <= t and i.itype not in ("noise", "law_firm")
        ]
        return all(i.itype in REACTIVE for i in known)

    def card_at(self, t: datetime) -> StoryCard:
        return resolve(self.items, t, follower=self.follower_at(t))

    def _nsn_item(self, cutoff: datetime) -> TItem | None:
        for i in sorted(self.items, key=lambda i: (i.available_at, i.event_id)):
            if i.available_at > cutoff:
                return None
            if self.card_at(i.available_at).family == FAMILY:
                return i
        return None

    def nsn_at(self, cutoff: datetime) -> datetime | None:
        item = self._nsn_item(cutoff)
        return item.available_at if item is not None else None

    def at_news(self) -> datetime | None:
        item = self._nsn_item(Session.of(self.session).entry_cutoff)
        return item.at if item is not None else None

    def start_case(self) -> str:
        at = self.at_news()
        s = Session.of(self.session)
        return "in" if at is not None and s.open <= at < s.close else "out"

    def news_times(self) -> list[datetime]:
        return [i.available_at for i in self.items]

    def before(self, t: datetime) -> TStory:
        return type(self)(
            self.symbol, self.session, [i for i in self.items if i.available_at <= t], self.parent
        )


class AtlasStory(TStory):
    """The atlas's view (spec §F): detected at its first item, whatever its type.

    The simulator starts only stories that become NSN; the atlas wraps the
    others so that ``nsn_at`` and ``at_news`` come from the first item.
    """

    def nsn_at(self, cutoff: datetime) -> datetime | None:
        first = min((i.available_at for i in self.items), default=None)
        return first if first is not None and first <= cutoff else None

    def at_news(self) -> datetime | None:
        return min((i.at for i in self.items), default=None)


def tstory(
    symbol: str, session: date, *items: tuple[datetime, str], parent: bool = False
) -> TStory:
    """A story of ``(at, itype)`` items, event ids in the order given."""
    return TStory(symbol, session, titems(*items), parent)


def titems(*items: tuple[datetime, str]) -> list[TItem]:
    """``(at, itype)`` items, event ids in the order given."""
    return [TItem(n + 1, at, itype) for n, (at, itype) in enumerate(items)]


# ── context ──


def pre_event(
    day: date,
    *,
    prev_close: float = 100.0,
    spy_prev: float = 200.0,
    sigma: float = 0.018,
    beta: float = BETA,
    adv20_usd: float = 5e7,
) -> PreEvent:
    nan = math.nan
    return PreEvent(
        session=day,
        prev_session=previous_trading_day(day),
        prev_close_s=prev_close,
        spy_prev_close_s=spy_prev,
        sigma=sigma,
        sigma_n=60,
        beta=beta,
        atr_pct=0.02,
        adv20_usd=adv20_usd,
        ret5_vs_spy=0.0,
        ret20_vs_spy=0.0,
        hi20_s=nan,
        lo20_s=nan,
        hi252_s=nan,
        lo252_s=nan,
    )


def eligibility(*, reason: str = "ok", rank: int = 500, tech: bool = False) -> Eligibility:
    return Eligibility(
        eligible=reason == "ok",
        reason=reason,  # type: ignore[arg-type]
        universe="primary",
        screen_as_of=date(2015, 12, 31),
        verdict="halal" if reason == "ok" else None,
        cik=1,
        sector=None,
        tech=tech,
        liquidity_rank=rank,
        cost_bps=COST if rank >= 300 else 7.0,
    )


# ── bars ──


def bars(
    day: date,
    levels: Sequence[tuple[tuple[int, int], float]],
    rows: Mapping[tuple[int, int], Row] | None = None,
    *,
    skip: Sequence[tuple[int, int]] = (),
    first: tuple[int, int] = (9, 30),
    last: tuple[int, int] | None = None,
    volume: float = 1_000.0,
) -> BarArrays:
    """Flat bars at each level from its minute until the next level; ``rows`` override minutes."""
    out: dict[tuple[int, int], Row] = {}
    marks = sorted(levels)
    for m in minutes_of(day):
        price = math.nan
        for start, level in marks:
            if m >= start:
                price = level
        if not math.isnan(price):
            out[m] = (price, price, price, price, volume, price)
    out.update(rows or {})
    return session_bars(day, rows=out, skip=skip, first=first, last=last, volume=volume)


def spy(
    days: Sequence[date],
    rows: Mapping[date, Mapping[tuple[int, int], Row]] | None = None,
    *,
    price: float = 200.0,
) -> SpyData:
    """Flat SPY sessions with per-day row overrides."""
    rows = rows or {}
    return SpyData({d: bars(d, [((9, 30), price)], rows.get(d), volume=1e5) for d in days})


# ── the base path and the "in" path (module docstring) ──

A = "AAA"
BASE_LEVELS = [((9, 30), 96.0), ((9, 46), 94.8), ((10, 6), 95.4), ((10, 8), 96.0)]
BASE_ROWS: dict[tuple[int, int], Row] = {
    (9, 30): (97.0, 97.0, 96.0, 96.0, 1_000.0, 96.5),
    (9, 45): (96.0, 96.0, 94.0, 94.5, 1_000.0, 95.0),
    (10, 5): (94.8, 95.5, 94.7, 95.4, 1_000.0, 95.2),
    (10, 7): (95.4, 95.6, 95.3, 95.5, 1_000.0, 95.45),
}
TARGET_LEVELS = [((10, 31), 97.0)]
TARGET_ROWS: dict[tuple[int, int], Row] = {
    (10, 30): (96.5, 97.2, 96.4, 97.0, 1_000.0, 96.9),
    (10, 32): (97.0, 97.3, 96.9, 97.2, 2_000.0, 97.1),
}
SPY_ROWS: dict[tuple[int, int], Row] = {
    (10, 7): (200.0, 200.6, 199.9, 200.0, 1e5, 200.5),
    (10, 32): (200.0, 201.2, 199.8, 200.0, 1e5, 201.0),
}
IN_LEVELS = [
    ((9, 30), 100.0),
    ((10, 52), 96.6),
    ((10, 56), 96.3),
    ((11, 16), 96.8),
    ((11, 18), 97.5),
    ((11, 41), 98.2),
]
IN_ROWS: dict[tuple[int, int], Row] = {
    (10, 49): (100.0, 100.3, 99.9, 100.2, 1_000.0, 100.1),
    (10, 50): (100.2, 100.2, 97.0, 97.2, 1_000.0, 98.0),  # D = -0.0299: not yet
    (10, 51): (97.2, 97.3, 96.5, 96.6, 1_000.0, 96.9),  # D = -0.0359: triggered
    (10, 55): (96.6, 96.6, 96.0, 96.2, 1_000.0, 96.1),  # L = 96.0
    (11, 15): (96.3, 96.9, 96.25, 96.8, 1_000.0, 96.7),  # 20 min quiet, > AVWAP 96.43
    (11, 17): (96.8, 97.0, 96.7, 96.9, 1_000.0, 96.85),
    (11, 40): (97.5, 98.3, 97.4, 98.2, 1_000.0, 98.0),  # >= TGT 98.1
    (11, 42): (98.2, 98.4, 98.1, 98.3, 1_000.0, 98.25),
}
IN_SPY: dict[tuple[int, int], Row] = {(11, 42): (201.0, 201.4, 200.9, 201.0, 1e5, 201.3)}


def base_story(day: date = MON, *extra: tuple[datetime, str], parent: bool = False) -> TStory:
    """The downgrade public at 07:00 on ``day``, plus ``extra`` items."""
    return tstory(A, day, (et(day, 7, 0), "analyst_downgrade"), *extra, parent=parent)


def base_bars(
    *,
    target: bool = True,
    rows: Mapping[tuple[int, int], Row] | None = None,
    skip: Sequence[tuple[int, int]] = (),
    last: tuple[int, int] | None = None,
) -> BarArrays:
    """MON's base path (with or without the target bar), ``rows`` overriding minutes."""
    levels = BASE_LEVELS + (TARGET_LEVELS if target else [])
    merged = BASE_ROWS | (TARGET_ROWS if target else {}) | dict(rows or {})
    return bars(MON, levels, merged, skip=skip, last=last)


def late_reclaim(rows: Mapping[tuple[int, int], Row] | None = None) -> BarArrays:
    """MON's base path reclaiming only on the 14:45 bar: armed from 10:06:05, the entry is
    decided at 14:46:05 and fills on the 14:47 bar at 95.4 (FillIn 14:48:01); no target."""
    opening: dict[tuple[int, int], Row] = {
        (9, 30): BASE_ROWS[(9, 30)],
        (9, 45): BASE_ROWS[(9, 45)],
        (14, 45): (94.8, 95.5, 94.7, 95.4, 1_000.0, 95.2),
    }
    levels = [((9, 30), 96.0), ((9, 46), 94.8), ((14, 46), 95.4)]
    return bars(MON, levels, opening | dict(rows or {}))


def tue_story(*items: tuple[datetime, str]) -> TStory:
    """TUE's story: ``items`` (public on MON after 14:20, so usable after 14:30: S + 1's), then
    a downgrade at 07:00 TUE. A structural item keeps it from ever being NSN, so it never starts
    and only carries news to MON's playbook."""
    return tstory(A, TUE, *items, (et(TUE, 7, 0), "analyst_downgrade"))


# ── running ──


def run(
    stories: Sequence[TStory | Story],
    paths: Mapping[str, PathData],
    spy_data: SpyData,
    context: Mapping[str, tuple[PreEvent | None, Eligibility]],
    *,
    ctx: Context | None = None,
    params: BounceParams | None = None,
    cfg: SimConfig | None = None,
    keep: bool = True,
    stop_at: StopAt = "end",
    assume_full_hold: bool = False,
) -> list[StoryOutcome]:
    """Every story of one symbol through the bounce, in the simulator."""
    params = params or BounceParams()
    symbols = {s.symbol for s in stories}
    assert len(symbols) == 1
    return simulate_symbol(
        symbols.pop(),
        stories,
        BounceFactory(dict(context), params),
        paths,
        spy_data,
        ctx or Context(),
        cfg or SimConfig(),
        keep_transitions=keep,
        stop_at=stop_at,
        assume_full_hold=assume_full_hold,
    )


def transitions(o: StoryOutcome) -> list[tuple[datetime, str, str]]:
    """(at, to, reason) of each transition (the ``from`` label is the simulator's)."""
    return [(at, to, why) for at, _, to, why in o.transitions]


def r_expected(
    entry: float,
    exit_: float,
    spy_entry: float,
    spy_exit: float,
    *,
    a_entry: float = 1.0,
    a_exit: float = 1.0,
    cost: float = COST,
    beta: float = BETA,
) -> tuple[float, float, float, float]:
    """(r_gross, r_spy, r_net_abn, r_beta_adj) by the spec §G.6 formula."""
    r_gross = (exit_ * a_exit) / (entry * a_entry) - 1.0
    r_spy = spy_exit / spy_entry - 1.0
    return (
        r_gross,
        r_spy,
        r_gross - r_spy - 2 * cost / 1e4,
        r_gross - beta * r_spy - 2 * cost / 1e4,
    )


# ── a random market with dips, for the look-ahead and determinism tests ──


def _dip(bars_: BarArrays, t0: int, rng: np.random.Generator) -> BarArrays:
    """Multiply the bars from epoch second ``t0`` on by a fall-and-partial-recovery profile."""
    depth = float(rng.uniform(0.02, 0.10))
    fall = float(rng.uniform(2, 40)) * 60
    back = float(rng.uniform(0.0, 1.3))
    rec = float(rng.uniform(20, 240)) * 60
    m = bars_.ts.astype(np.float64) - t0
    f = np.where(
        m < 0,
        1.0,
        np.where(
            m < fall,
            1.0 - depth * m / fall,
            1.0 - depth + depth * back * np.minimum((m - fall) / rec, 1.0),
        ),
    )
    return BarArrays(
        bars_.ts, bars_.o * f, bars_.h * f, bars_.l * f, bars_.c * f, bars_.v, bars_.vw * f
    )


def _calm(b: BarArrays, *, level: float = 200.0, power: float = 0.25) -> BarArrays:
    """SPY's walk damped towards ``level`` (``level * (p/level)**power``): about 1% a day."""

    def f(x: np.ndarray) -> np.ndarray:
        return level * (x / level) ** power

    return BarArrays(b.ts, f(b.o), f(b.h), f(b.l), f(b.c), b.v, f(b.vw))


def bounce_market(
    seed: int, symbols: int, *, sessions: int
) -> tuple[Market, dict[str, tuple[PreEvent | None, Eligibility]]]:
    """``symbols`` names with a story on WEEK[0] (often one more later), each with a dip.

    News comes before the open or inside the session; the price falls from
    its public time (or the open) by 2-10% and recovers a random share of
    it. Some stories carry a structural or an unclear item later, some are
    ineligible, some names fail the screen on the next session.
    """
    rng = np.random.default_rng(seed)
    spy_days = {d: _calm(random_session(rng, d, 200.0, holes=False)) for d in WEEK}
    daily_points = {("SPY", d): daily(float(b.c[-1]), float(b.o[0])) for d, b in spy_days.items()}
    stories: list[Story] = []
    paths: dict[str, PathData] = {}
    context: dict[str, tuple[PreEvent | None, Eligibility]] = {}
    verdicts: dict[tuple[str, date], str] = {}
    for j in range(symbols):
        symbol = f"B{j:03d}"
        story_days = [WEEK[0]]
        if rng.random() < 0.6:
            story_days.append(WEEK[int(rng.integers(1, 3))])
        news: dict[date, datetime] = {}
        for s in story_days:
            if rng.random() < 0.5:
                news[s] = et(s, int(rng.integers(6, 9)), int(rng.integers(0, 60)))
            else:
                news[s] = et(s, int(rng.integers(10, 14)), int(rng.integers(0, 60)))
        p = float(rng.uniform(20, 200))
        prev_close: dict[date, float] = {}
        day_bars: dict[date, BarArrays] = {}
        for d in WEEK:
            prev_close[d] = p
            b = random_session(rng, d, p)
            if d in news and len(b):
                at = news[d] - NEWS_LAG
                t0 = max(int(at.timestamp()), epoch(d, 9, 30))
                b = _dip(b, t0, rng)
            day_bars[d] = b
            if len(b):
                p = float(b.c[-1])
                daily_points[(symbol, d)] = daily(p, float(b.o[0]))
        for s in story_days:
            items = [Item(news[s])]
            if rng.random() < 0.25:  # a structural item later in the session
                items.append(
                    Item(
                        et(s, int(rng.integers(10, 16)), int(rng.integers(0, 60))),
                        "offering",
                        structural=True,
                    )
                )
            if rng.random() < 0.2:
                items.append(
                    Item(et(s, int(rng.integers(9, 16)), int(rng.integers(0, 60))), "noise")
                )
            st = Story(symbol, s, items)
            stories.append(st)
            pdays = [s]
            while len(pdays) < sessions:
                pdays.append(next_trading_day(pdays[-1]))
            paths[st.story_id] = path(st.story_id, symbol, pdays, [day_bars[d] for d in pdays])
            spy_prev = (
                float(spy_days[previous_trading_day(s)].c[-1])
                if previous_trading_day(s) in spy_days
                else 200.0
            )
            pre = pre_event(
                s,
                prev_close=prev_close[s],
                spy_prev=spy_prev,
                sigma=float(rng.uniform(0.006, 0.02)),
                beta=float(rng.uniform(0.8, 1.5)),
            )
            reason = "rank" if rng.random() < 0.1 else "ok"
            if verdicts.get((symbol, s)) == "not_halal":  # the screen the eligibility reads
                reason = "not_halal"
            context[st.story_id] = (pre, eligibility(reason=reason, rank=int(rng.integers(0, 999))))
            if rng.random() < 0.15:
                verdicts[(symbol, next_trading_day(s))] = "not_halal"
    market = Market(
        stories, paths, SpyData(spy_days), Context(daily=daily_points, verdicts=verdicts), sessions
    )
    return market, context


__all__ = [
    "BASE_LEVELS",
    "BASE_ROWS",
    "COST",
    "EPS",
    "IN_LEVELS",
    "IN_ROWS",
    "IN_SPY",
    "NEWS_LAG",
    "SPY_ROWS",
    "TARGET_LEVELS",
    "TARGET_ROWS",
    "A",
    "AtlasStory",
    "TItem",
    "TStory",
    "bars",
    "base_bars",
    "base_story",
    "bounce_market",
    "eligibility",
    "late_reclaim",
    "pre_event",
    "r_expected",
    "run",
    "spy",
    "titems",
    "transitions",
    "tstory",
    "tue_story",
]
