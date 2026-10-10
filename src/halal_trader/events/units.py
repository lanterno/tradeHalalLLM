"""Plan H: the minute-bar units the news engine's Phase 0 gates and H1 read (spec §H).

A unit is one (symbol, session) of ``minute_bars`` (``data/minutes.py``),
fetched once and marked done once settled, empty ones included. Plan H is
seven parts, fetched in :data:`PARTS` order by :func:`fetch` through
``minutes.backfill`` (grouped by session):

* ``spy``: SPY on every session 2016-01-04..2024-12-31.
* ``gate_g1`` (look-ahead gate): :data:`G1_STORIES` stories with S in
  2016-01-04..2016-09-30 whose symbol has liquidity rank < 1000 and a
  previous close of at least $5 in S units, NSN_CORE stories (``nsn_at`` by
  S's entry cutoff) first, then the other stories whose ``type_close`` has a
  negative direction (``taxonomy.TYPES``), each group in seeded order
  (:func:`g1_stories`): ``path(S, 4, 2016-09-30)``. A path cut there holds
  fewer sessions; :func:`g1_path_sessions` gives the gate the sessions it
  may ask the loader for.
* ``gate_calib`` (G3 calibration): :data:`CALIB_PAIRS` seeded non-event
  (symbol, session) pairs, rank < 1000, sessions 2016-01-04..2016-08-31
  (:func:`calib_pairs`): the session and the session four later.
* ``gate_sue`` (G3): Σ_s, a seeded sample of :data:`SUE_SAMPLE` events of the
  SUE complement Σ_c (:func:`sue_complement`, :func:`sue_sample`): each
  event's entry session and its exit sessions at h = 5 and h = 20 on
  ``study.entry_point``'s clock.
* ``gate_reactor`` (G2): the pinned reactor headline set,
  ``intraday.selection(first_in_session(scored_before=2026-10-10T00:00Z))``:
  each headline's (symbol, New York day) on a session, and SPY's.
* ``train``: every story with S in 2016-10-03..2021-12-31 whose symbol is in
  H1's universe at S (:func:`in_universe`) and which has an item of a type
  outside :data:`TRAIN_SKIP_TYPES`: ``path(S, 4, 2021-12-31)``.
* ``validation``: every story with S in 2022-01-03..2024-12-31, in the
  universe at S, that is NSN_CORE by S's entry cutoff under the fastest news
  lag any run uses (:data:`FASTEST_NEWS_LAG`): ``path(S, 4, 2024-12-31)``.

A path is S and the sessions after it (:func:`path`): four of them, the three
a multi-day trade can hold and the spare its close fallback reads, cut at the
window's last session. A train or validation story holding an 8-K can also
react in S−1, since a corrected filing time can move it one session back
(:func:`story_units`): when it is in the universe at S its units add S−1,
and when it is in the universe at S−1 they add ``path(S−1, 4, cap)``.

**Selection inputs.** The persisted stories (``news_stories`` of
``stories.BUILDER_VERSION``: each item's type and kind, ``nsn_at``,
``type_close``), the screens and liquidity ranks (``monthly_bars`` through
``universe_at``), the SUE observations' times and the reactor's scores.
Daily bars enter only through ``context.PitContext``: its eligibility (the
previous close in S units and σ, from before the news) and, for the G1
price rule, the same previous close. ``A(S)``, a ratio of S's two closes
(the corporate actions effective at S's open), is the one S-dated input, as
context.py states. No return on or after S is read, so no outcome selects a
unit; the sets are fixed before anything is simulated.

**Nothing on or after 2025-01-01** except ``gate_reactor`` (whose bars, from
2025-12, are the gate's own), and **no gate unit outside its gate's dates**
(``loader.GATE_RANGES``): :meth:`UnitPlan.check` refuses either, so a run
stops before it fetches anything.

Decisions where the spec or the brief leave a choice, for the
pre-registration to cite:

* :meth:`UnitPlan.sha` is the full sha256 of the sorted
  ``SYMBOL:YYYY-MM-DD`` lines (``loader.unit_set_sha``), not its first 12
  hex: ``loader.register_gate_units(expected_sha=...)`` compares the full
  digest. The dry run prints the first 12.
* **The universe at S is PRIMARY or BROAD, at both news times** a story
  reacting in S can have (:func:`in_universe`). σ's window depends on
  whether the news came before S−1's close or after it, and the plan must
  hold the paths of every reading (the H1 runner's, at the NSN item, and the
  atlas's), so a story is in when either time admits it. PRIMARY is asked
  as well as BROAD because it is not a subset of BROAD: each universe keeps
  one share class per CIK among the names *it* admits, so a halal class
  can win in PRIMARY and lose in BROAD to a more liquid class that only an
  index veto keeps out of PRIMARY (spec §H assumes PRIMARY ⊂ BROAD).
* **filing_other makes no train story** (the brief adds it to spec §H's
  five types): an 8-K of routine items only (5.03, 5.07, 7.01, 8.01, 9.01,
  or none that ``taxonomy.ITEM_TYPES`` types) has direction "none". The
  path atlas (spec §F lists the five) filters its units with
  :data:`TRAIN_SKIP_TYPES` too, so every atlas story has train units.
* **S−1 for 8-K stories** only when S−1 is inside the part's window: a story
  moved before the window's first session leaves the window. The universe
  is judged at S−1 as at S; the rank prefilter admits a story liquid at
  either session.
* **The fastest news lag.** H1's 60 s news-lag sensitivity makes every item
  usable 540 s earlier than ``stories.NEWS_LAG`` does (``h1.relag``), and so
  can make a story NSN by the cutoff that is not at 600 s. Validation
  therefore selects ``nsn_at − (NEWS_LAG − 60 s) <= entry cutoff``; the 1,200 s
  sensitivity only makes stories NSN later, a subset.
* **gate_calib is non-event** (spec §E.3; the brief is silent): a pair whose
  session is the entry session of one of the symbol's SUE observations is
  left out of the population.
* **Σ_c** holds the observations published (New York date) 2016-01-04..
  2019-12-31 with an entry on ``study.entry_point``'s clock, rank < 1000 at
  the entry session, and in neither PRIMARY nor BROAD there at the
  publication time. An observation published on an early-close day between
  the early close and 16:00 is left out and counted (``early_close``):
  ``entry_point`` enters it at a close that came before the news (spec §E.3,
  S1), and no context can judge a story at a session that closed before its
  news. The clock walks ``market_hours``' sessions, which precondition D1
  equates with SPY's raw daily bars (the study's calendar).
* **Σ_c never touches H1's units.** An event whose units (entry or exit
  session) meet the ``train`` or ``validation`` part is dropped from Σ_c
  and counted (``h1_overlap``), before Σ_s is drawn: a September 2016 entry
  can exit in October 2016 on a name that is in the universe by then, and
  spec §E.3 says the gate never reads H1's own universe. Σ_c therefore needs
  both window parts (:func:`h1_windows`) before the gate's set is pinned.
* **Seeds.** Every seeded part draws from its own ``random.Random(seed)``
  over a sorted population: G1 shuffles the NSN group, then the others,
  with one generator; calibration and Σ_s each sample once.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks.loader import GATE_RANGES, unit_set_sha
from halabot.playbooks.types import Session
from halal_trader.core.heartbeat import DAILY_JOBS, RESEARCH
from halal_trader.data import minutes
from halal_trader.data.minutes import session_bounds
from halal_trader.events.context import MAX_RANK, MIN_PREV_CLOSE, TOP_N
from halal_trader.events.stories import BUILDER_VERSION, NEWS_LAG
from halal_trader.events.study import BENCHMARK, Observation, entry_point
from halal_trader.events.taxonomy import FILING_KINDS, TYPES
from halal_trader.market_hours import (
    MARKET_TZ,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
)

if TYPE_CHECKING:
    from halal_trader.events.intraday import Headline

logger = logging.getLogger(__name__)

Unit = tuple[str, date]
Part = Literal["spy", "gate_g1", "gate_calib", "gate_sue", "gate_reactor", "train", "validation"]
Window = Literal["train", "validation"]

SEED: Final = 20261010
SPY: Final = BENCHMARK
# The fetch order (spec §H, with the reactor's set before the windows).
PARTS: Final[tuple[Part, ...]] = (
    "spy",
    "gate_g1",
    "gate_calib",
    "gate_sue",
    "gate_reactor",
    "train",
    "validation",
)
# Each gate part's id in the loader (GATE_RANGES, register_gate_units).
GATE_OF_PART: Final[dict[str, str]] = {
    "gate_g1": "g1",
    "gate_calib": "calib",
    "gate_sue": "sue",
    "gate_reactor": "reactor",
}

# Three sessions (MD3) and the spare a close fallback reads.
PATH_SESSIONS: Final = 4
HOLD_SESSIONS: Final = PATH_SESSIONS - 1  # the most a trade holds (MD3); the spare is extra
SPY_RANGE: Final = (date(2016, 1, 4), date(2024, 12, 31))
TRAIN: Final = (date(2016, 10, 3), date(2021, 12, 31))
VALIDATION: Final = (date(2022, 1, 3), date(2024, 12, 31))
WINDOWS: Final[dict[Window, tuple[date, date]]] = {"train": TRAIN, "validation": VALIDATION}
LAST_UNLOCKED: Final = date(2024, 12, 31)  # nothing later, except the reactor's own set

G1_RANGE: Final = (date(2016, 1, 4), date(2016, 9, 30))
G1_STORIES: Final = 500
CALIB_RANGE: Final = (date(2016, 1, 4), date(2016, 8, 31))
CALIB_CAP: Final = date(2016, 9, 30)
CALIB_PAIRS: Final = 2_000
CALIB_HORIZON: Final = 5  # a 09:30 entry and an h=5 close exit: the session and four later
SUE_RANGE: Final = (date(2016, 1, 4), date(2019, 12, 31))  # publication, New York date
SUE_SAMPLE: Final = 4_000
SUE_HORIZONS: Final = (5, 20)
# The calendar entry_point walks for Σ_c: from before the first publication
# to past the last h=20 exit (2020-01-30).
_SUE_CALENDAR: Final = (date(2015, 12, 1), date(2020, 3, 31))
REACTOR_SCORED_BEFORE: Final = datetime(2026, 10, 10, tzinfo=UTC)

# Item types that do not make a train story: spec §H's five, plus filing_other.
# The path atlas (spec §F) filters its units with this same set, so every
# atlas story has train units.
TRAIN_SKIP_TYPES: Final[frozenset[str]] = frozenset(
    {"noise", "law_firm", "mover", "other", "analyst_other", "filing_other"}
)
# The universes a plan story may be in (in_universe); BROAD first, as it admits more.
UNIVERSES: Final[tuple[Literal["primary", "broad"], ...]] = ("broad", "primary")
# The fastest news lag a run reads stories with (h1.NEWS_LAGS' 60 s sensitivity; a
# test keeps them equal): validation holds the stories NSN by the cutoff under it.
FASTEST_NEWS_LAG: Final = timedelta(seconds=60)

# The dry run's request estimate (data/alpaca_market.minute_bars_many).
BARS_PER_UNIT: Final = 390
PAGE_BARS: Final = 10_000
SYMBOLS_PER_REQUEST: Final = 100
CHUNK_SESSIONS: Final = 100  # sessions per backfill call: the busy check runs between calls


# ── paths and the calendar ─────────────────────────────────────


def path(session: date, n: int, cap: date) -> list[date]:
    """``n`` sessions from ``session`` (included) by ``market_hours``, truncated at ``cap``."""
    if not is_trading_day(session):
        raise ValueError(f"{session} is not a trading session")
    if n < 1:
        raise ValueError(f"a path holds at least one session, not {n}")
    days: list[date] = []
    day = session
    while len(days) < n and day <= cap:
        days.append(day)
        day = next_trading_day(day)
    return days


def sessions_between(start: date, end: date) -> list[date]:
    """Every ``market_hours`` session in [start, end]."""
    out: list[date] = []
    day = start
    while day <= end:
        if is_trading_day(day):
            out.append(day)
        day += timedelta(days=1)
    return out


# ── the plan ───────────────────────────────────────────────────


class PlanError(ValueError):
    """A plan that may not be fetched (:meth:`UnitPlan.check`)."""


@dataclass(frozen=True, slots=True)
class UnitPlan:
    """The units of each part (keys from :data:`PARTS`)."""

    parts: Mapping[str, frozenset[Unit]]

    def all(self) -> frozenset[Unit]:
        """Every unit of every part, once."""
        return frozenset().union(*self.parts.values())

    def sha(self, part: str) -> str:
        """The part's pin: sha256 of its sorted ``SYMBOL:YYYY-MM-DD`` lines
        (``loader.unit_set_sha``, what ``register_gate_units`` compares)."""
        return unit_set_sha(self.parts[part])

    def fetch_order(self) -> list[str]:
        """The plan's parts in :data:`PARTS` order."""
        return [p for p in PARTS if p in self.parts]

    def check(self) -> None:
        """:class:`PlanError` on an unknown part, a unit after 2024 outside the
        reactor's set, or a gate part's unit outside its gate's dates
        (``loader.GATE_RANGES``, which ``register_gate_units`` enforces only
        after the fetch): the plan is refused before anything is fetched."""
        unknown = sorted(set(self.parts) - set(PARTS))
        if unknown:
            raise PlanError(f"unknown part(s): {', '.join(unknown)}")
        for part, units in self.parts.items():
            if part != "gate_reactor":
                late = sorted(u for u in units if u[1] > LAST_UNLOCKED)
                if late:
                    raise PlanError(
                        f"{part}: {len(late)} unit(s) after {LAST_UNLOCKED}, first "
                        f"{minutes.unit(*late[0])}"
                    )
            gate = GATE_OF_PART.get(part)
            if gate is not None:
                lo, hi = GATE_RANGES[gate]
                outside = sorted(u for u in units if not lo <= u[1] <= hi)
                if outside:
                    raise PlanError(
                        f"{part}: {len(outside)} unit(s) outside the {gate} gate's dates "
                        f"{lo}..{hi}, first {minutes.unit(*outside[0])}"
                    )


# ── liquidity ──────────────────────────────────────────────────


class LiquidityRanks:
    """0-based ranks in ``universe_at(month, top_n=3000)``, one query per month."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._months: dict[date, dict[str, int]] = {}
        self._names: dict[date, list[str]] = {}

    async def load(self, days: Iterable[date]) -> None:
        from halal_trader.data.universe import universe_at

        for month in sorted({d.replace(day=1) for d in days} - set(self._months)):
            names = await universe_at(self._engine, month, top_n=TOP_N)
            self._names[month] = names
            self._months[month] = {s: i for i, s in enumerate(names)}

    def rank(self, symbol: str, day: date) -> int | None:
        return self._months[day.replace(day=1)].get(symbol)

    def liquid(self, symbol: str, day: date) -> bool:
        """Rank < 1000 at ``day`` (the months before its month)."""
        rank = self.rank(symbol, day)
        return rank is not None and rank < MAX_RANK

    def top(self, day: date) -> list[str]:
        """The names ranked < 1000 at ``day``, most liquid first."""
        return self._names[day.replace(day=1)][:MAX_RANK]


# ── the context's questions ────────────────────────────────────


class _Eligible(Protocol):
    @property
    def eligible(self) -> bool: ...


class _Daily(Protocol):
    @property
    def close(self) -> float: ...
    @property
    def adj(self) -> float: ...


class ContextLike(Protocol):
    """What the plan asks ``context.PitContext``."""

    def eligibility(
        self,
        symbol: str,
        session: date,
        *,
        at_news: datetime,
        universe: Literal["primary", "broad"],
    ) -> _Eligible: ...

    def daily(self, symbol: str, day: date) -> _Daily | None: ...

    def adj(self, symbol: str, day: date) -> float | None: ...


async def _context(
    engine: AsyncEngine, symbols: Collection[str], lo: date, hi: date
) -> ContextLike:
    from halal_trader.events.context import PitContext

    return await PitContext.load(engine, symbols=symbols, start=lo, end=hi)


def news_times(session: date) -> tuple[datetime, datetime]:
    """One news time of each σ window a story reacting in ``session`` can have:
    just before S−1's close (σ through S−2) and S's open (σ through S−1)."""
    prev_close = session_bounds(previous_trading_day(session))[1]
    return prev_close - timedelta(seconds=1), session_bounds(session)[0]


def in_universe(ctx: ContextLike, symbol: str, session: date) -> bool:
    """In PRIMARY or BROAD at ``session`` for news at some time a story there can have.

    Both universes are asked because PRIMARY is not a subset of BROAD: each
    keeps one share class per CIK among the names it admits, so a halal
    class PRIMARY keeps can lose, in BROAD, to a more liquid class that only
    an index veto keeps out of PRIMARY (``share_class`` there).
    """
    return any(
        ctx.eligibility(symbol, session, at_news=at, universe=universe).eligible
        for universe in UNIVERSES
        for at in news_times(session)
    )


def prev_close_s(ctx: ContextLike, symbol: str, session: date) -> float | None:
    """``close_raw(S−1) · A(S−1)/A(S)``: S−1's close in S units, or None without the bars.

    The G1 price rule. It must agree with the price ``context.PitContext``
    judges eligibility on (``context._prev_close_s``), which has no public
    accessor; it is computed here through ``daily`` and ``adj``, and a test
    holds the two equal on the context's own bars. The one way they can
    differ: ``daily`` also wants S−1's raw open, high, low and volume, which
    ``daily_bars`` stores NOT NULL with the close.
    """
    prev = ctx.daily(symbol, previous_trading_day(session))
    a_s = ctx.adj(symbol, session)
    if prev is None or a_s is None:
        return None
    return prev.close * prev.adj / a_s


def _years(start: date, end: date) -> list[tuple[date, date]]:
    return [
        (max(start, date(y, 1, 1)), min(end, date(y, 12, 31)))
        for y in range(start.year, end.year + 1)
    ]


# ── stories ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class StoryRef:
    """What the plan reads of one persisted story."""

    story_id: str
    symbol: str
    session: date
    nsn: bool  # NSN_CORE by S's entry cutoff (``nsn_at`` <= Session.entry_cutoff)
    nsn_fast: bool  # the same under FASTEST_NEWS_LAG (see nsn_by_cutoff); nsn implies it
    type_close: str
    substantive: bool  # an item of a type outside TRAIN_SKIP_TYPES
    has_8k: bool  # an 8-K or 8-K/A item


def nsn_by_cutoff(nsn_at: datetime | None, session: date, *, lag: timedelta = NEWS_LAG) -> bool:
    """Whether a story whose persisted ``nsn_at`` (items usable ``stories.NEWS_LAG``
    after their time) is NSN_CORE by S's entry cutoff when items are usable
    ``lag`` after their time instead.

    Moving every item's ``available_at`` by the same amount (``h1.relag``)
    moves the first NSN time by it too: the cards depend only on which items
    are known, so the story is NSN by the cutoff iff
    ``nsn_at − (NEWS_LAG − lag) <= cutoff``. ``nsn_at`` was found by S's
    close, which is later than the cutoff plus any shift a run uses.
    """
    if nsn_at is None:
        return False
    return nsn_at - (NEWS_LAG - lag) <= Session.of(session).entry_cutoff


_STORIES_SQL: Final = text(
    """
    SELECT s.story_id, s.symbol, s.session, s.nsn_at, s.type_close,
           EXISTS (SELECT 1 FROM jsonb_array_elements(s.items) x
                   WHERE NOT (x->>'itype' = ANY(CAST(:skip AS text[])))) AS substantive,
           EXISTS (SELECT 1 FROM jsonb_array_elements(s.items) x
                   JOIN events e ON e.id = CAST(x->>'event_id' AS bigint)
                   WHERE e.kind = ANY(CAST(:filings AS text[]))) AS has_8k
    FROM news_stories s
    WHERE s.builder_version = :v AND s.session >= :a AND s.session <= :b
    ORDER BY s.session, s.story_id
    """
)


async def read_stories(engine: AsyncEngine, start: date, end: date) -> list[StoryRef]:
    """The persisted stories (this builder version) with S in [start, end]."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            _STORIES_SQL,
            {
                "v": BUILDER_VERSION,
                "a": start,
                "b": end,
                "skip": sorted(TRAIN_SKIP_TYPES),
                "filings": sorted(FILING_KINDS),
            },
        )
        return [
            StoryRef(
                story_id=str(r.story_id),
                symbol=str(r.symbol),
                session=r.session,
                nsn=nsn_by_cutoff(r.nsn_at, r.session),
                nsn_fast=nsn_by_cutoff(r.nsn_at, r.session, lag=FASTEST_NEWS_LAG),
                type_close=str(r.type_close),
                substantive=bool(r.substantive),
                has_8k=bool(r.has_8k),
            )
            for r in rows
        ]


def reaction_sessions(story: StoryRef, start: date) -> list[date]:
    """The sessions ``story`` can react in: S, then S−1 for a story with an 8-K
    when S−1 >= ``start`` (a corrected filing time moves it one session back)."""
    days = [story.session]
    if story.has_8k:
        prev = previous_trading_day(story.session)
        if prev >= start:
            days.append(prev)
    return days


def _anywhere(day: date) -> bool:
    return True


def _judge(ctx: ContextLike, ranks: LiquidityRanks, symbol: str) -> Callable[[date], bool]:
    """Whether ``symbol`` is ranked < 1000 and :func:`in_universe` at a session, memoised."""
    judged: dict[date, bool] = {}

    def eligible(day: date) -> bool:
        if day not in judged:
            judged[day] = ranks.liquid(symbol, day) and in_universe(ctx, symbol, day)
        return judged[day]

    return eligible


def story_units(
    story: StoryRef, start: date, cap: date, *, eligible: Callable[[date], bool] = _anywhere
) -> set[Unit]:
    """The units of ``story`` given the sessions it is in the universe at
    (``eligible``; every session by default).

    In at S: ``path(S, 4, cap)``, and S−1 too for a story with an 8-K. In at
    S−1 (a story with an 8-K): ``path(S−1, 4, cap)``. S−1 counts only when
    >= ``start`` (:func:`reaction_sessions`).
    """
    session, *before = reaction_sessions(story, start)
    days: set[date] = set()
    if eligible(session):
        days.update(path(session, PATH_SESSIONS, cap))
        days.update(before)
    for prev in before:
        if eligible(prev):
            days.update(path(prev, PATH_SESSIONS, cap))
    return {(story.symbol, d) for d in days}


async def window_units(
    engine: AsyncEngine,
    window: Window,
    *,
    ranks: LiquidityRanks | None = None,
    counts: Counter[str] | None = None,
) -> frozenset[Unit]:
    """The ``train`` or ``validation`` part: one year of stories and context at a time.

    Train takes the substantive stories, validation those NSN by the cutoff
    under :data:`FASTEST_NEWS_LAG`; a story is in at each of its reaction
    sessions where its symbol is ranked < 1000 and :func:`in_universe`.
    """
    start, end = WINDOWS[window]
    ranks = ranks if ranks is not None else LiquidityRanks(engine)
    c = counts if counts is not None else Counter()
    out: set[Unit] = set()
    for lo, hi in _years(start, end):
        refs = await read_stories(engine, lo, hi)
        c["stories"] += len(refs)
        keep = [r for r in refs if (r.substantive if window == "train" else r.nsn_fast)]
        days = {r.story_id: reaction_sessions(r, start) for r in keep}
        await ranks.load(d for r in keep for d in days[r.story_id])
        keep = [r for r in keep if any(ranks.liquid(r.symbol, d) for d in days[r.story_id])]
        if not keep:
            continue
        first = min(min(days[r.story_id]) for r in keep)  # an S−1 can fall in the year before
        ctx = await _context(engine, {r.symbol for r in keep}, min(lo, first), hi)
        for r in keep:
            eligible = _judge(ctx, ranks, r.symbol)
            got = story_units(r, start, end, eligible=eligible)
            if not got:
                continue
            c["selected"] += 1
            c["with_8k"] += r.has_8k
            c["before_only"] += not eligible(r.session)  # in only through S−1
            out |= got
        del ctx
        logger.info("plan h1 %s %s..%s: %d units so far", window, lo, hi, len(out))
    return frozenset(out)


async def h1_windows(
    engine: AsyncEngine,
    *,
    windows: Collection[Window] | None = None,
    ranks: LiquidityRanks | None = None,
    counts: Counter[str] | None = None,
) -> dict[Window, frozenset[Unit]]:
    """The ``train`` and ``validation`` parts (or only ``windows``); ``counts``
    gets ``<window>.<count>``.

    Σ_c drops every SUE event that meets the two (:func:`sue_complement`).
    """
    ranks = ranks if ranks is not None else LiquidityRanks(engine)
    c = counts if counts is not None else Counter()
    out: dict[Window, frozenset[Unit]] = {}
    for window in [w for w in WINDOWS if windows is None or w in windows]:
        window_counts: Counter[str] = Counter()
        out[window] = await window_units(engine, window, ranks=ranks, counts=window_counts)
        c.update({f"{window}.{k}": v for k, v in window_counts.items()})
    return out


# ── gate_g1 ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GateStory:
    """One of the look-ahead gate's stories."""

    story_id: str
    symbol: str
    session: date
    nsn: bool  # NSN_CORE by the entry cutoff: the bounce runs with the family check


def negative(type_close: str) -> bool:
    meta = TYPES.get(type_close)
    return meta is not None and meta.direction == "neg"


def order_g1(candidates: Iterable[StoryRef], *, seed: int = SEED) -> list[GateStory]:
    """NSN stories in seeded order, then the other negative ones, first :data:`G1_STORIES`."""
    pool = sorted(candidates, key=lambda r: r.story_id)
    nsn = [r for r in pool if r.nsn]
    others = [r for r in pool if not r.nsn and negative(r.type_close)]
    rng = random.Random(seed)
    rng.shuffle(nsn)
    rng.shuffle(others)
    return [GateStory(r.story_id, r.symbol, r.session, r.nsn) for r in (nsn + others)[:G1_STORIES]]


async def g1_stories(
    engine: AsyncEngine, *, seed: int = SEED, ranks: LiquidityRanks | None = None
) -> list[GateStory]:
    """The G1 gate's stories: S in 2016-01-04..2016-09-30, rank < 1000, a previous
    close of at least $5, NSN or a negative type; seeded order (:func:`order_g1`)."""
    lo, hi = G1_RANGE
    ranks = ranks if ranks is not None else LiquidityRanks(engine)
    refs = [r for r in await read_stories(engine, lo, hi) if r.nsn or negative(r.type_close)]
    await ranks.load(r.session for r in refs)
    refs = [r for r in refs if ranks.liquid(r.symbol, r.session)]
    if not refs:
        return []
    ctx = await _context(engine, {r.symbol for r in refs}, lo, hi)
    priced = []
    for r in refs:
        close = prev_close_s(ctx, r.symbol, r.session)
        if close is not None and close >= MIN_PREV_CLOSE:
            priced.append(r)
    return order_g1(priced, seed=seed)


def g1_units(stories: Iterable[GateStory]) -> frozenset[Unit]:
    return frozenset(
        (s.symbol, d) for s in stories for d in path(s.session, PATH_SESSIONS, G1_RANGE[1])
    )


class _OnSession(Protocol):
    @property
    def session(self) -> date: ...


def g1_path_sessions(story: _OnSession, hold: int = HOLD_SESSIONS) -> int:
    """The sessions the G1 gate may ask the loader for (``PathRequest.n_sessions``)
    for a story held up to ``hold`` sessions: ``hold``, cut where :func:`g1_units`
    cuts the path, at 2016-09-30.

    A story with S in 2016-09-28..30 has a shorter path, and the loader refuses
    (``WindowLocked``, the whole run) a path that leaves the gate's unit set.
    The spare session the loader adds itself, only where it is a unit.
    """
    if hold < 1:
        raise ValueError(f"a story holds at least one session, not {hold}")
    return min(hold, len(path(story.session, PATH_SESSIONS, G1_RANGE[1])))


# ── SUE: Σ_c, Σ_s and the calibration pairs ────────────────────


@dataclass(frozen=True, slots=True)
class SueEvent:
    """One SUE observation on ``study.entry_point``'s clock."""

    symbol: str
    published_at: datetime
    signal: float
    session: date  # the entry (reaction) session
    entry: Literal["open", "close"]
    exits: tuple[date, ...]  # the exit sessions at SUE_HORIZONS

    def units(self) -> set[Unit]:
        return {(self.symbol, self.session)} | {(self.symbol, d) for d in self.exits}


async def load_sue_observations(engine: AsyncEngine) -> list[Observation]:
    """Every SUE observation, as ``events study sue`` builds them (no date filter)."""
    from halal_trader.events.history import covered_companies
    from halal_trader.events.sue import sue_observations

    raw = await sue_observations(engine, await covered_companies(engine))
    return [Observation(o.symbol, o.announced_at, o.sue) for o in raw]


def sue_event(obs: Observation, sessions: Sequence[date]) -> SueEvent | str:
    """``obs`` on the study's clock, or why it has no event: ``no_entry`` (no
    session to enter at, or none for an exit) or ``early_close`` (a close entry
    on an early-close day for news published after that close)."""
    entry = entry_point(obs.published_at, sessions)
    if entry is None:
        return "no_entry"
    i, at = entry
    session = sessions[i]
    if at == "close" and obs.published_at >= session_bounds(session)[1]:
        return "early_close"
    exits = [i + h - (1 if at == "open" else 0) for h in SUE_HORIZONS]
    if exits[-1] >= len(sessions):
        return "no_entry"
    return SueEvent(
        symbol=obs.symbol,
        published_at=obs.published_at,
        signal=obs.signal,
        session=session,
        entry="open" if at == "open" else "close",
        exits=tuple(sessions[k] for k in exits),
    )


def _sue_key(e: SueEvent) -> tuple[datetime, str, float, date]:
    return (e.published_at, e.symbol, e.signal, e.session)


async def sue_complement(
    engine: AsyncEngine,
    *,
    h1_units: Collection[Unit],
    observations: Sequence[Observation] | None = None,
    ranks: LiquidityRanks | None = None,
    counts: Counter[str] | None = None,
) -> list[SueEvent]:
    """Σ_c: SUE observations published 2016-01-04..2019-12-31 whose symbol has rank
    < 1000 at the entry session and is in neither PRIMARY nor BROAD there at the
    publication time, less every event with a unit in ``h1_units`` (the train and
    validation parts, :func:`h1_windows`); sorted. ``counts`` gets each exclusion,
    ``h1_overlap`` the events dropped for meeting ``h1_units``."""
    h1 = h1_units if isinstance(h1_units, (set, frozenset)) else set(h1_units)
    obs = observations if observations is not None else await load_sue_observations(engine)
    ranks = ranks if ranks is not None else LiquidityRanks(engine)
    c = counts if counts is not None else Counter()
    sessions = sessions_between(*_SUE_CALENDAR)
    events: list[SueEvent] = []
    for o in obs:
        day = o.published_at.astimezone(MARKET_TZ).date()
        if not SUE_RANGE[0] <= day <= SUE_RANGE[1]:
            continue
        c["published"] += 1
        event = sue_event(o, sessions)
        if isinstance(event, str):
            c[event] += 1
            continue
        events.append(event)
    await ranks.load(e.session for e in events)
    liquid = [e for e in events if ranks.liquid(e.symbol, e.session)]
    c["rank"] += len(events) - len(liquid)
    by_year: dict[int, list[SueEvent]] = defaultdict(list)
    for e in liquid:
        by_year[e.session.year].append(e)
    out: list[SueEvent] = []
    for year in sorted(by_year):
        group = by_year[year]
        lo, hi = min(e.session for e in group), max(e.session for e in group)
        ctx = await _context(engine, {e.symbol for e in group}, lo, hi)
        for e in group:
            if any(
                ctx.eligibility(e.symbol, e.session, at_news=e.published_at, universe=u).eligible
                for u in UNIVERSES
            ):
                c["universe"] += 1
            elif not e.units().isdisjoint(h1):
                c["h1_overlap"] += 1
            else:
                out.append(e)
        del ctx
    c["complement"] = len(out)
    return sorted(out, key=_sue_key)


def sue_sample(
    complement: Sequence[SueEvent], *, seed: int = SEED, n: int = SUE_SAMPLE
) -> list[SueEvent]:
    """Σ_s: ``n`` events of Σ_c drawn with ``random.Random(seed)``, sorted (all when fewer)."""
    pool = sorted(complement, key=_sue_key)
    if len(pool) <= n:
        return pool
    return sorted(random.Random(seed).sample(pool, n), key=_sue_key)


def sue_units(events: Iterable[SueEvent]) -> frozenset[Unit]:
    return frozenset(u for e in events for u in e.units())


def event_sessions(observations: Iterable[Observation]) -> set[Unit]:
    """(symbol, entry session) of every SUE observation: the calibration's events."""
    sessions = sessions_between(*_SUE_CALENDAR)
    out: set[Unit] = set()
    for o in observations:
        entry = entry_point(o.published_at, sessions)
        if entry is not None:
            out.add((o.symbol, sessions[entry[0]]))
    return out


async def calib_pairs(
    engine: AsyncEngine,
    *,
    seed: int = SEED,
    observations: Sequence[Observation] | None = None,
    ranks: LiquidityRanks | None = None,
) -> list[Unit]:
    """:data:`CALIB_PAIRS` (symbol, session) pairs drawn with ``random.Random(seed)``
    from every session 2016-01-04..2016-08-31 and every name ranked < 1000 there,
    less the SUE entry sessions; sorted by session, then symbol."""
    obs = observations if observations is not None else await load_sue_observations(engine)
    events = event_sessions(obs)
    days = sessions_between(*CALIB_RANGE)
    ranks = ranks if ranks is not None else LiquidityRanks(engine)
    await ranks.load(days)
    population = sorted(
        ((s, d) for d in days for s in ranks.top(d) if (s, d) not in events),
        key=lambda u: (u[1], u[0]),
    )
    if len(population) <= CALIB_PAIRS:
        return population
    return sorted(random.Random(seed).sample(population, CALIB_PAIRS), key=lambda u: (u[1], u[0]))


def calib_units(pairs: Iterable[Unit]) -> frozenset[Unit]:
    """Each pair's session and the session :data:`CALIB_HORIZON` − 1 later."""
    out: set[Unit] = set()
    for symbol, day in pairs:
        days = path(day, CALIB_HORIZON, CALIB_CAP)
        out |= {(symbol, days[0]), (symbol, days[-1])}
    return frozenset(out)


# ── gate_reactor ───────────────────────────────────────────────


async def reactor_headlines(engine: AsyncEngine) -> list[Headline]:
    """The reactor gate's pinned headlines: ``selection(first_in_session(scored_before=...))``."""
    from halal_trader.events.intraday import first_in_session, selection

    return selection(await first_in_session(engine, scored_before=REACTOR_SCORED_BEFORE))


def reactor_units(headlines: Iterable[Headline]) -> frozenset[Unit]:
    """(symbol, New York day) of each headline on a session, and SPY's."""
    out: set[Unit] = set()
    for h in headlines:
        day = h.published_at.astimezone(MARKET_TZ).date()
        if is_trading_day(day):
            out |= {(h.symbol, day), (SPY, day)}
    return frozenset(out)


def spy_units() -> frozenset[Unit]:
    return frozenset((SPY, d) for d in sessions_between(*SPY_RANGE))


# ── the plan ───────────────────────────────────────────────────


async def h1_plan(
    engine: AsyncEngine,
    *,
    seed: int = SEED,
    parts: Collection[str] | None = None,
    counts: Counter[str] | None = None,
) -> UnitPlan:
    """Plan H's parts (all, or ``parts``); ``counts`` gets the selection counts.

    ``gate_sue`` needs the train and validation parts (Σ_c drops the events
    that meet them), so asking for it selects those too, counts included.
    """
    if parts is not None and (unknown := sorted(set(parts) - set(PARTS))):
        raise PlanError(f"unknown part(s): {', '.join(unknown)}")
    wanted = [p for p in PARTS if parts is None or p in parts]
    c = counts if counts is not None else Counter()
    ranks = LiquidityRanks(engine)
    out: dict[str, frozenset[Unit]] = {}
    observations: list[Observation] | None = None
    windows = await h1_windows(
        engine,
        windows=[w for w in WINDOWS if w in wanted or "gate_sue" in wanted],
        ranks=ranks,
        counts=c,
    )
    for part in wanted:
        if part == "spy":
            out[part] = spy_units()
        elif part == "gate_g1":
            chosen = await g1_stories(engine, seed=seed, ranks=ranks)
            c["g1.stories"] = len(chosen)
            c["g1.nsn"] = sum(s.nsn for s in chosen)
            out[part] = g1_units(chosen)
        elif part in ("gate_calib", "gate_sue"):
            if observations is None:
                observations = await load_sue_observations(engine)
            if part == "gate_calib":
                pairs = await calib_pairs(engine, seed=seed, observations=observations, ranks=ranks)
                c["calib.pairs"] = len(pairs)
                out[part] = calib_units(pairs)
            else:
                sue_counts: Counter[str] = Counter()
                complement = await sue_complement(
                    engine,
                    h1_units=windows["train"] | windows["validation"],
                    observations=observations,
                    ranks=ranks,
                    counts=sue_counts,
                )
                sample = sue_sample(complement, seed=seed)
                c.update({f"sue.{k}": v for k, v in sue_counts.items()})
                c["sue.sample"] = len(sample)
                out[part] = sue_units(sample)
        elif part == "gate_reactor":
            headlines = await reactor_headlines(engine)
            c["reactor.headlines"] = len(headlines)
            out[part] = reactor_units(headlines)
        else:
            out[part] = windows["train" if part == "train" else "validation"]
    plan = UnitPlan(out)
    plan.check()
    return plan


# ── the dry run's estimate ─────────────────────────────────────


def requests_for(units: Iterable[Unit]) -> int:
    """Requests to fetch ``units``: per session, :data:`SYMBOLS_PER_REQUEST` symbols a
    request, pages of :data:`PAGE_BARS` bars at :data:`BARS_PER_UNIT` bars a unit."""
    by_day: dict[date, int] = Counter(d for _, d in set(units))
    total = 0
    for n in by_day.values():
        full, rest = divmod(n, SYMBOLS_PER_REQUEST)
        chunks = [SYMBOLS_PER_REQUEST] * full + ([rest] if rest else [])
        total += sum(max(1, math.ceil(k * BARS_PER_UNIT / PAGE_BARS)) for k in chunks)
    return total


def hours_at(requests: int, rate: int) -> float:
    """Hours ``requests`` take at ``rate`` requests a minute."""
    return requests / rate / 60.0


@dataclass(frozen=True, slots=True)
class PartEstimate:
    part: str
    units: int
    new: int  # not in an earlier part
    done: int  # of the new ones, already fetched and settled
    to_fetch: int
    sessions: int  # sessions with a unit to fetch
    requests: int
    first: date | None
    last: date | None
    sha: str


def estimate(plan: UnitPlan, done: Collection[str]) -> list[PartEstimate]:
    """Per part, in fetch order: what is new, what is done, what a run would request."""
    seen: set[Unit] = set()
    out: list[PartEstimate] = []
    for part in plan.fetch_order():
        units = plan.parts[part]
        new = units - seen
        seen |= units
        fetch = {u for u in new if minutes.unit(*u) not in done}
        days = [d for _, d in units]
        out.append(
            PartEstimate(
                part=part,
                units=len(units),
                new=len(new),
                done=len(new) - len(fetch),
                to_fetch=len(fetch),
                sessions=len({d for _, d in fetch}),
                requests=requests_for(fetch),
                first=min(days) if days else None,
                last=max(days) if days else None,
                sha=plan.sha(part),
            )
        )
    return out


# ── fetching ───────────────────────────────────────────────────


def busy(now: datetime) -> str | None:
    """Why minute bars should not be fetched at ``now``, or None: US market hours
    and the research job's window (spec §H "When"), on trading days."""
    local = now.astimezone(MARKET_TZ)
    day = local.date()
    if not is_trading_day(day):
        return None
    open_, close = session_bounds(day)
    if open_ <= now < close:
        return f"US market hours ({open_:%H:%M}-{close:%H:%M} ET)"
    job = DAILY_JOBS[RESEARCH]
    start = datetime.combine(day, job.at, MARKET_TZ)
    if start <= now < start + job.grace:
        end = start + job.grace
        return f"the research job's window ({start:%H:%M}-{end:%H:%M} ET)"
    return None


@dataclass(slots=True)
class FetchReport:
    stored: dict[str, int] = field(default_factory=dict)  # bars stored per part
    fetched: dict[str, int] = field(default_factory=dict)  # units asked for per part
    stopped: str | None = None  # why the run stopped early


def _chunks(units: Iterable[Unit], sessions: int) -> list[list[Unit]]:
    """``units`` by session, oldest first, ``sessions`` sessions a chunk."""
    by_day: dict[date, list[Unit]] = defaultdict(list)
    for u in sorted(set(units), key=lambda u: (u[1], u[0])):
        by_day[u[1]].append(u)
    days = sorted(by_day)
    return [
        [u for d in days[i : i + sessions] for u in by_day[d]]
        for i in range(0, len(days), sessions)
    ]


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def fetch(
    engine: AsyncEngine,
    market: Any,
    plan: UnitPlan,
    *,
    stop: Callable[[datetime], str | None] | None = busy,
    clock: Callable[[], datetime] = _utc_now,
    chunk_sessions: int = CHUNK_SESSIONS,
    on_part: Callable[[str, int, int], None] | None = None,
) -> FetchReport:
    """Fetch every unit of the plan not yet done, part by part in fetch order,
    ``chunk_sessions`` sessions per ``minutes.backfill`` call.

    Before each call ``stop(now)`` may name a reason to stop (default
    :func:`busy`); the run then ends, and a later run resumes where it
    stopped (done units are never fetched again). ``on_part(part, units,
    bars)`` is told each finished part.
    """
    report = FetchReport()
    done = await minutes.done_units(engine)
    for part in plan.fetch_order():
        todo = [u for u in plan.parts[part] if minutes.unit(*u) not in done]
        report.fetched[part] = 0
        report.stored[part] = 0
        for chunk in _chunks(todo, chunk_sessions):
            if stop is not None and (why := stop(clock())) is not None:
                report.stopped = why
                return report
            report.stored[part] += await minutes.backfill(engine, market, chunk)
            report.fetched[part] += len(chunk)
            done.update(minutes.unit(*u) for u in chunk)
            logger.info(
                "plan h1 %s: %d/%d units, %d bars",
                part,
                report.fetched[part],
                len(todo),
                report.stored[part],
            )
        if on_part is not None:
            on_part(part, report.fetched[part], report.stored[part])
    return report
