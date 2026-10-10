"""The path atlas: how prices moved after each kind of story, on the train window (spec §F).

**Descriptive only.** The atlas reports counts, quantiles, shares and means
with their date-clustered (CR1) standard errors. It computes no p-value,
ranks no cell (tables are sorted by their keys), and sweeps no H1 constant.
A cell with fewer than :data:`MIN_N` stories or :data:`MIN_DATES` distinct
dates shows its counts only. Any idea taken from it becomes a new
registered trial, fitted on train and tested once on validation.

**When it may run** (:func:`h1_closed`, checked before anything else is
read). The newest ``research.news.h1`` preregistration must have, under its
config hash, a ``verdict`` row or a ``stage-a`` row recording
``fail: insufficient events``; otherwise :class:`AtlasLocked`. Minute bars
are read through the simulator's loader under that registration
(``WindowUnlock``), window ``train``, ending :data:`DATA_END`: no minute or
daily bar after 2021-12-31 is read. The last session is :data:`ATLAS_END`,
2021-12-23, the last whose five-session continuation ends by 2021-12-31.
It is never extended to validation or holdout.

**Unit.** Every PRIMARY story (``PitContext.eligibility``) with its reaction
session S in [:data:`ATLAS_START`, :data:`ATLAS_END`] and an item of a type
outside ``units.TRAIN_SKIP_TYPES`` (the filter of plan H's ``train`` part,
which fetched the paths). Candidates are read from the persisted
``news_stories`` rows (eligibility judged one year at a time, as
``stories.count_stories`` does), then the stories are rebuilt from the
event store with ``stories.build`` (``card_at`` needs the items) and checked
against their rows: a story whose rebuilt labels differ from its row is
counted (``meta["counts"]["persisted_mismatch"]``), not dropped.

**Keys.** Cells are keyed by ``card_at(detect_at).type`` (``type_detect``);
``type_close`` and ``family_ever`` are row columns.

**References** (S units, spec §G.3-G.4). The news time ``ref_at`` is the
``at`` of the item that made the story NSN_CORE when it is NSN by S's entry
cutoff, else the ``at`` of its first substantive item (``detect_at`` less the
news lag). "in" (``ref_at`` in S's session): P0 is the close of the last bar
with ``ts + 60 s <= ref_at`` (else ``prev_close_s``), SPY0 the same for SPY
(else ``spy_prev_close_s``), and the anchor the first bar with ``ts >=
floor_minute(ref_at)``; "out": P0 = ``prev_close_s``, SPY0 =
``spy_prev_close_s``, the anchor S's first bar. σ is ``PreEvent.sigma`` at
``ref_at``.

**Per-story measures** (:func:`path_measures`; abnormal to SPY at the same bars):

* ``D_t = (c_t/P0 - 1) - (spy_c_t/SPY0 - 1)`` on S's bars from the anchor,
  ``spy_c_t`` SPY's bar of the same start or its latest before (SPY0 before any);
* ``gap_sigma = D(anchor)/σ``; ``low_sigma = min D_t/σ`` over S;
* L = the lowest low of S from the anchor; ``t_low`` = minutes from the
  anchor to the latest bar attaining it;
* with a drop (P0 > L), ``retrace(p) = (p - L)/(P0 - L)``: ``retrace_close``
  at S's last close; ``retrace_max_s`` and ``retrace_max_s2`` the highest
  over the closes from L's bar to S's close, and to S+2's close (S units);
  ``fade``: a low below L by S+2 after the first close that retraced
  :data:`FADE_AFTER` (None when none did). Without a drop all are None;
* ``cont_h``, h in :data:`CONT_HORIZONS`: the all-adjusted close-to-close
  return from S's close to S+h's, minus SPY's (daily bars);
* the H1 machine (:class:`MachineRun`), for each negative-direction story
  (its detect card's ``direction``) and each story NSN by the cutoff:
  ``OverreactionBounce`` with the family check disabled
  (``require_family=False``), once for ID and once for MD3, through
  ``sim.run``. It starts at ``max(news detection, open)`` as H1 does
  (:class:`AtlasStory`). Each **lane** (:attr:`AtlasRow.lane`) is its own
  run: the stories NSN_CORE by S's cutoff (``NSN_CORE``: exactly H1's
  starters over the range, H1 starting every eligible NSN story from
  2016-10-03), and each other negative type alone. In a run every other
  story of the symbol, of any lane, reaches the playbook as news only. One
  playbook per symbol is live at a time, so a story can end
  ``blocked_open``, as in H1, but only behind a story of its own lane: a
  ``legal_adverse`` story held under MD3 never blocks an NSN one (in H1
  only NSN stories start), and the NSN lane's machine numbers are not
  filtered by another type's outcome. Each cell counts its blocked stories
  (``<variant>_blocked`` of ``<variant>_starters``); they are not in P(trigger)
  or r. With a later ``start`` than :data:`ATLAS_START`, a story before it
  starts nothing, so it blocks nothing.

**Tables** (:func:`cells_of`): ``base`` is type x ``low_sigma`` bucket
(:data:`BUCKETS`); ``type`` the type alone; the marginals cross the type
with one of: timing, rank, SPY 20-day realised-volatility tercile, SPY
above or below its 200-day SMA (both known at S: closes through S-1),
screen regime (2020-10-01), analyst-coverage regime (2018-01-01),
Technology. ``coverage`` counts every unit by type and path outcome
(measured, or the loader's skip). Cells hold the measured stories. A
cell's ``per_year`` is its n over its own time in years of 252 sessions
(:class:`Exposure`): the range's sessions, or, in a table keyed by a state
of the calendar (:data:`DATED`: SPY's states and the two regimes), the
range's sessions in that state.

**SPY's regimes** (:func:`spy_state`) come from a SPY-only context of the
whole atlas range, whatever ``start`` and ``end`` are asked: the tercile
edges are fixed on SPY's sessions in [:data:`ATLAS_START`,
:data:`DATA_END`] (spec §F: SPY 2016-10..2021-12), so a shorter run puts a
story in the same tercile as the full one. SPY's daily bars start on
2016-01-04, so its 200-session SMA first exists on 2016-10-18: stories of
2016-10-03..2016-10-17 have the trend ``n/a`` (their own cell; nothing is
backfilled).

**Deviations from spec §F** (stated for the integrator):

* ``filing_other`` is excluded with the spec's five types: units are
  filtered with plan H's ``units.TRAIN_SKIP_TYPES``, and plan H's ``train``
  part never fetches a story made only of those (every one would be a
  ``units_missing`` skip), so stories made only of ``filing_other`` items
  are not units.
* The machine also runs on a story NSN by the cutoff whose detect type is
  not negative (an 8-K's unparsed earnings, then a miss): such a story is
  an H1 event.
* "Retrace >= x by the close / by S+2" reads the highest retrace reached
  (``retrace_max_s`` / ``retrace_max_s2``); the denominators of those
  shares are the stories with a drop (P0 > L).
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks.bounce import BounceFactory, BounceParams
from halabot.playbooks.loader import (
    SPY,
    MinuteBarLoader,
    PathRequest,
    Window,
    WindowUnlock,
)
from halabot.playbooks.records import MemorySink, StoryOutcome
from halabot.playbooks.sim import run as simulate
from halabot.playbooks.types import PathData, PathSkip, Session, SimConfig, path_days
from halal_trader.data.minutes import BarArrays
from halal_trader.events import h1
from halal_trader.events import stories as builder
from halal_trader.events.aliases import load_aliases
from halal_trader.events.context import LOOKAHEAD_DAYS, Eligibility, PitContext, PreEvent
from halal_trader.events.stats import clustered_mean
from halal_trader.events.taxonomy import FAMILY, NOISE_TYPES, StoryCard
from halal_trader.events.units import TRAIN_SKIP_TYPES
from halal_trader.market_hours import MARKET_TZ, is_trading_day, previous_trading_day

if TYPE_CHECKING:
    from halal_trader.config import Settings
    from halal_trader.events.aliases import AliasMatcher
    from halal_trader.events.stories import Story

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────

ATLAS_START: Final = date(2016, 10, 3)  # the first PRIMARY session (screens start 2016-09-30)
ATLAS_END: Final = date(2021, 12, 23)  # its 5-session continuation ends on 2021-12-31
DATA_END: Final = date(2021, 12, 31)  # no bar after this is read
PATH_SESSIONS: Final = 3  # S .. S+2 for the retrace and fade measures
CONT_HORIZONS: Final = (1, 3, 5)
# low_sigma buckets: (-inf, -5], (-5, -3], (-3, -2], (-2, -1], (-1, inf)
LOW_EDGES: Final = (-5.0, -3.0, -2.0, -1.0)
BUCKETS: Final = ("(-inf,-5]", "(-5,-3]", "(-3,-2]", "(-2,-1]", "(-1,inf)")
QUANTILES: Final = (0.10, 0.25, 0.50, 0.75, 0.90)
RETRACE_LEVELS: Final = (0.25, 0.50, 1.00)
FADE_AFTER: Final = 0.25
MIN_N: Final = 30
MIN_DATES: Final = 20
RANK_SPLIT: Final = 300
SPY_VOL_SESSIONS: Final = 20
SPY_SMA_SESSIONS: Final = 200
SCREEN_BREAK: Final = date(2020, 10, 1)
ANALYST_BREAK: Final = date(2018, 1, 1)
SESSIONS_PER_YEAR: Final = 252
# The marginals keyed by a state of the calendar at S (a regime, SPY's state):
# their per-year rate divides by the time that state covers in the range.
DATED: Final = ("spy_vol", "spy_trend", "screen_regime", "analyst_regime")
EXIT_REASONS: Final = ("target", "stop", "abort", "compliance", "time_stop")
VARIANTS: Final = (("ID", 1), ("MD3", 3))
BLOCKED: Final = "blocked_open"  # the simulator's reason for a story started behind a live one
OUTPUT_NAME: Final = f"news_atlas-{builder.BUILDER_VERSION}.json"
BATCH_SYMBOLS: Final = 100

Timing = Literal["pre_open", "in_session", "evening_weekend"]

TABLES: Final = (
    "type",
    "base",
    "timing",
    "rank",
    "spy_vol",
    "spy_trend",
    "screen_regime",
    "analyst_regime",
    "sector",
    "coverage",
)
TITLES: Final = {
    "type": "type",
    "base": "type x low_sigma bucket",
    "timing": "type x news timing (pre-open, in-session, previous evening or weekend)",
    "rank": "type x liquidity rank (< 300, 300-999)",
    "spy_vol": "type x SPY 20-day realised-volatility tercile",
    "spy_trend": "type x SPY against its 200-day SMA",
    "screen_regime": f"type x screen regime (before / from {SCREEN_BREAK})",
    "analyst_regime": f"type x analyst-coverage regime (before / from {ANALYST_BREAK})",
    "sector": "type x sector (Technology or other)",
    "coverage": "type x path (measured, or the loader's skip): counts only",
}
# The order of each marginal's second key (unknown values sort last).
_ORDER: Final[dict[str, tuple[str, ...]]] = {
    "base": BUCKETS,
    "timing": ("pre_open", "in_session", "evening_weekend"),
    "rank": ("rank<300", "rank300-999"),
    "spy_vol": ("low", "mid", "high", "n/a"),
    "spy_trend": ("above", "below", "n/a"),
    "screen_regime": ("before", "from"),
    "analyst_regime": ("before", "from"),
    "sector": ("tech", "other"),
    "coverage": ("measured",),
}


class AtlasLocked(RuntimeError):
    """The atlas may not run: no H1 registration, or no verdict (nor Stage-A failure) for it."""


# ── types ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Registration:
    """The H1 registration the atlas reads under, and the row that closed it."""

    id: int
    config_hash: str
    closed_by: int
    closing: str  # "<kind>: <verdict>"


@dataclass(frozen=True, slots=True)
class PathMeasures:
    """One story's path from its references (module docstring); levels in S units."""

    p0: float
    spy0: float
    anchor_ts: int  # epoch seconds of the anchor bar's start
    gap_sigma: float
    low_sigma: float
    t_low: float  # minutes from the anchor to the bar that set L
    low: float  # L
    retrace_close: float | None
    retrace_max_s: float | None
    retrace_max_s2: float | None
    fade: bool | None


@dataclass(frozen=True, slots=True)
class MachineRun:
    """How ``OverreactionBounce`` (no family check) ended on one story, for one variant."""

    state: str
    reason: str
    skip: str | None
    triggered: bool
    armed: bool
    entered: bool
    exit_reason: str | None
    r: float | None  # r_net_abn of the trade
    flags: tuple[str, ...] = ()

    @property
    def ran(self) -> bool:
        """A playbook ran on the story: not skipped by the loader, not dismissed or blocked."""
        return self.skip is None and self.state != "DISMISSED"

    @property
    def blocked(self) -> bool:
        """Started while a playbook of its symbol (in its lane) was live: it never got one."""
        return self.state == "DISMISSED" and self.reason == BLOCKED

    @classmethod
    def of(cls, o: StoryOutcome) -> MachineRun:
        trade = o.trade
        return cls(
            state=o.terminal_state,
            reason=o.reason,
            skip=o.skip,
            triggered=o.triggered_at is not None,
            armed=o.armed_at is not None,
            entered=o.entered,
            exit_reason=trade.exit_reason if trade is not None else None,
            r=trade.r_net_abn if trade is not None else None,
            flags=tuple(trade.flags) if trade is not None else (),
        )


@dataclass(frozen=True, slots=True)
class AtlasRow:
    """One unit story: its labels, its context at S, its path and the H1 machine on it."""

    story_id: str
    symbol: str
    session: date
    type: str  # card_at(detect_at).type: the cells' key
    type_close: str
    family_ever: str | None
    direction: str  # the detect card's
    follower: bool  # at detect_at
    nsn: bool  # NSN_CORE by S's entry cutoff
    start_case: str
    detect_at: datetime
    ref_at: datetime
    timing: Timing
    rank: int
    tech: bool
    cost_bps: float
    spy_vol: str
    spy_trend: str
    sigma: float | None
    skip: str | None  # None: the path was measured
    measures: PathMeasures | None
    cont: tuple[float | None, ...]  # one per CONT_HORIZONS
    machine: bool  # the bounce ran on it (negative direction, or NSN)
    lane: str | None = None  # its machine run: "NSN_CORE" (H1's starters) or its type
    id_run: MachineRun | None = None
    md3_run: MachineRun | None = None

    @property
    def low_sigma(self) -> float | None:
        return self.measures.low_sigma if self.measures is not None else None

    def run(self, variant: str) -> MachineRun | None:
        return self.id_run if variant == "ID" else self.md3_run


@dataclass(frozen=True, slots=True)
class AtlasCell:
    table: str
    key: tuple[str, ...]
    n: int
    dates: int
    per_year: float
    stats: dict[str, float] | None  # None: counts only (n < 30 or fewer than 20 dates)


@dataclass(slots=True)
class Atlas:
    rows: list[AtlasRow]
    cells: list[AtlasCell]
    meta: dict[str, Any] = field(default_factory=dict)


# ── the stories as the machine sees them ──────────────────────


class AtlasStory:
    """A story with the news detection the atlas's machine starts from (a ``StoryView``).

    ``nsn_at(cutoff)`` answers ``detect`` (when it is no later than the
    cutoff), so the simulator starts the playbook at ``max(detect, open)``;
    ``at_news()`` is ``at``, the public time of that item, which sets the
    start case. For a story NSN by S's entry cutoff they are its own
    ``nsn_at`` and ``at_news``; otherwise its first substantive item's
    ``available_at`` and ``at``. With ``detect`` None the story never starts
    and only brings its items to a live playbook as news. The card is the
    story's own, read at any ``t``.
    """

    __slots__ = ("_at", "_detect", "_story")

    def __init__(self, story: Story, detect: datetime | None, at: datetime | None) -> None:
        self._story = story
        self._detect = detect
        self._at = at

    @classmethod
    def news_only(cls, story: Story) -> AtlasStory:
        return cls(story, None, None)

    @property
    def story_id(self) -> str:
        return self._story.story_id

    @property
    def symbol(self) -> str:
        return self._story.symbol

    @property
    def session(self) -> date:
        return self._story.session

    def card_at(self, t: datetime) -> StoryCard:
        return self._story.card_at(t)

    def nsn_at(self, cutoff: datetime) -> datetime | None:
        if self._detect is None or self._detect > cutoff:
            return None
        return self._detect

    def at_news(self) -> datetime | None:
        return self._at if self._detect is not None else None

    def start_case(self) -> str:
        at = self.at_news()
        if at is None:
            return "out"
        s = Session.of(self.session)
        return "in" if s.open <= at < s.close else "out"

    def news_times(self) -> list[datetime]:
        return self._story.news_times()


def detection(story: Story) -> tuple[datetime, datetime, bool] | None:
    """(detect, at, nsn) of a story: its NSN item's ``available_at`` and ``at`` when it is
    NSN_CORE by S's entry cutoff, else its first substantive item's; None without one."""
    cutoff = Session.of(story.session).entry_cutoff
    nsn = story.nsn_at(cutoff)
    if nsn is not None:
        at = story.at_news()
        if at is not None:
            return nsn, at, True
    first = next((i for i in story.items if i.itype not in NOISE_TYPES), None)
    if first is None:
        return None
    return first.available_at, first.at, False


def substantive(itypes: Iterable[str]) -> bool:
    """Whether a story has an item of a type outside ``units.TRAIN_SKIP_TYPES``."""
    return any(t not in TRAIN_SKIP_TYPES for t in itypes)


def timing_of(at: datetime, session: date) -> Timing:
    """``in_session`` in S's session; ``pre_open`` earlier on S's own day; else the
    previous evening, a weekend or a holiday (late news of the session before included)."""
    s = Session.of(session)
    if s.open <= at < s.close:
        return "in_session"
    if at < s.open and at.astimezone(MARKET_TZ).date() == session:
        return "pre_open"
    return "evening_weekend"


# ── the lock (spec §F: after H1's verdict) ─────────────────────


async def h1_closed(engine: AsyncEngine) -> Registration:
    """The newest H1 registration and the row that closed it; :class:`AtlasLocked` when
    there is none, or when it has neither a verdict nor a Stage-A ``fail: insufficient
    events`` row under its config hash (H1 may still be running: its bars stay locked)."""
    async with engine.connect() as conn:
        reg = (
            await conn.execute(
                text(
                    "SELECT id, config_hash FROM quant_trials "
                    "WHERE kind = 'preregistration' AND name = :n ORDER BY id DESC LIMIT 1"
                ),
                {"n": h1.NAME},
            )
        ).first()
        if reg is None:
            raise AtlasLocked(f"no {h1.NAME} preregistration: the atlas runs after H1's verdict")
        closing = (
            await conn.execute(
                text(
                    "SELECT id, kind, verdict FROM quant_trials WHERE config_hash = :h "
                    "AND ((kind = 'verdict' AND verdict IS NOT NULL) "
                    "OR (kind = 'stage-a' AND verdict LIKE :fail)) ORDER BY id LIMIT 1"
                ),
                {"h": reg.config_hash, "fail": f"{h1.STAGE_A_FAIL}%"},
            )
        ).first()
    if closing is None:
        raise AtlasLocked(
            f"{h1.NAME} registration {reg.id} has no verdict and no Stage-A "
            f"'{h1.STAGE_A_FAIL}' row yet: the atlas runs after H1's verdict"
        )
    return Registration(
        int(reg.id), str(reg.config_hash), int(closing.id), f"{closing.kind}: {closing.verdict}"
    )


def check_range(start: date, end: date) -> None:
    """``ValueError`` unless [start, end] lies inside the train range the atlas may read
    and ``end`` is a session (the paths and continuations are counted from it)."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    if start < ATLAS_START:
        raise ValueError(f"the atlas starts on {ATLAS_START} (the first PRIMARY session)")
    if end > ATLAS_END:
        raise ValueError(
            f"the atlas ends by {ATLAS_END}: its 5-session continuation ends on {DATA_END}; "
            "it is never extended to validation or holdout"
        )
    if not is_trading_day(end):
        raise ValueError(
            f"end {end} is not a trading session; the session before it is "
            f"{previous_trading_day(end)}"
        )


def context_end(end: date) -> date:
    """The ``end`` to load the context with (``check_range``'d ``end``).

    The context reads daily bars to its end plus ``LOOKAHEAD_DAYS``: two
    weeks past ``end`` reach S+5 of every unit, and the end is clamped so
    that no daily bar after :data:`DATA_END` is read (2021-12-23 loads to
    2021-12-24, so bars to 2021-12-31).
    """
    lookahead = timedelta(days=LOOKAHEAD_DAYS)
    out = max(end, min(end + lookahead, DATA_END - lookahead))
    if out + lookahead > DATA_END:
        raise AssertionError(f"a context to {out} would read daily bars after {DATA_END}")
    return out


# ── measures (pure) ────────────────────────────────────────────


def path_measures(
    bars: Sequence[BarArrays],
    scales: Sequence[float],
    spy: BarArrays,
    *,
    session: Session,
    ref_at: datetime,
    start_case: str,
    prev_close_s: float,
    spy_prev_close_s: float,
    sigma: float,
) -> PathMeasures | None:
    """The path measures of one story (module docstring); None without a bar of S
    from the anchor on.

    ``bars`` are the raw bars of S, S+1, S+2 (fewer is fine), ``scales``
    their ``A(d)/A(S)``; ``spy`` is SPY's bars of S.
    """
    s_bars = bars[0]
    ts = s_bars.ts
    at_s = math.floor(ref_at.timestamp())
    if start_case == "in":
        anchor_from = at_s // 60 * 60  # floor_minute(ref_at)
    else:
        anchor_from = int(session.open.timestamp())
    a = int(np.searchsorted(ts, anchor_from, side="left"))
    if a >= len(ts):
        return None
    if start_case == "in":
        limit = at_s - 60  # ts + 60 s <= ref_at
        j = int(np.searchsorted(ts[:a], limit, side="right")) - 1
        p0 = float(s_bars.c[j]) if j >= 0 else prev_close_s
        k = int(np.searchsorted(spy.ts, limit, side="right")) - 1
        spy0 = float(spy.c[k]) if k >= 0 else spy_prev_close_s
    else:
        p0, spy0 = prev_close_s, spy_prev_close_s

    c = s_bars.c[a:]
    js = np.searchsorted(spy.ts, ts[a:], side="right") - 1
    spy_c = np.where(js >= 0, spy.c[np.maximum(js, 0)] if len(spy) else spy0, spy0)
    d = (c / p0 - 1.0) - (spy_c / spy0 - 1.0)
    lows = s_bars.l[a:]
    low = float(lows.min())
    tl = a + len(lows) - 1 - int(np.argmin(lows[::-1]))  # the latest bar attaining L
    t_low = (int(ts[tl]) - int(ts[a])) / 60.0

    retrace_close = retrace_max_s = retrace_max_s2 = None
    fade: bool | None = None
    drop = p0 - low
    if drop > 0.0:
        closes = np.concatenate([b.c * s for b, s in zip(bars, scales)])
        all_lows = np.concatenate([b.l * s for b, s in zip(bars, scales)])
        r = (closes[tl:] - low) / drop
        retrace_close = float((float(c[-1]) - low) / drop)
        retrace_max_s = float(r[: len(ts) - tl].max())
        retrace_max_s2 = float(r.max())
        hits = np.nonzero(r >= FADE_AFTER)[0]
        if hits.size:
            fade = bool((all_lows[tl + int(hits[0]) + 1 :] < low).any())
    return PathMeasures(
        p0=p0,
        spy0=spy0,
        anchor_ts=int(ts[a]),
        gap_sigma=float(d[0]) / sigma,
        low_sigma=float(d.min()) / sigma,
        t_low=t_low,
        low=low,
        retrace_close=retrace_close,
        retrace_max_s=retrace_max_s,
        retrace_max_s2=retrace_max_s2,
        fade=fade,
    )


def bucket(low_sigma: float) -> str:
    """``low_sigma``'s bucket: (-inf,-5], (-5,-3], (-3,-2], (-2,-1], (-1,inf)."""
    for edge, label in zip(LOW_EDGES, BUCKETS):
        if low_sigma <= edge:
            return label
    return BUCKETS[-1]


@dataclass(frozen=True, slots=True)
class SpyRegimes:
    """SPY's state known at each session S (from closes through S-1)."""

    vol: Mapping[date, float]  # sd of the 20 daily returns through S-1
    above: Mapping[date, bool]  # close(S-1) above the mean of the 200 closes through S-1
    edges: tuple[float, float]  # the volatility terciles' edges

    def vol_tercile(self, day: date) -> str:
        v = self.vol.get(day)
        if v is None or not math.isfinite(self.edges[0]):
            return "n/a"
        return "low" if v < self.edges[0] else "mid" if v < self.edges[1] else "high"

    def trend(self, day: date) -> str:
        above = self.above.get(day)
        return "n/a" if above is None else "above" if above else "below"


def spy_regimes(
    sessions: Sequence[date],
    closes: Sequence[float | None],
    *,
    edges_from: date,
    edges_to: date,
) -> SpyRegimes:
    """SPY's regimes from its all-adjusted ``closes`` on ``sessions`` (sorted).

    The volatility terciles' edges are the 1/3 and 2/3 quantiles of the
    sessions' volatility in [edges_from, edges_to].
    """
    c = np.asarray([math.nan if x is None else x for x in closes], dtype=np.float64)
    vol: dict[date, float] = {}
    above: dict[date, bool] = {}
    for j, day in enumerate(sessions):
        if j >= SPY_VOL_SESSIONS + 1:
            w = c[j - SPY_VOL_SESSIONS - 1 : j]
            if np.isfinite(w).all():
                vol[day] = float(np.std(w[1:] / w[:-1] - 1.0, ddof=1))
        if j >= SPY_SMA_SESSIONS:
            w = c[j - SPY_SMA_SESSIONS : j]
            if np.isfinite(w).all():
                above[day] = bool(w[-1] > w.mean())
    inside = [v for d, v in vol.items() if edges_from <= d <= edges_to]
    if inside:
        lo, hi = np.quantile(np.asarray(inside), [1.0 / 3.0, 2.0 / 3.0])
        edges = (float(lo), float(hi))
    else:
        edges = (math.nan, math.nan)
    return SpyRegimes(vol, above, edges)


def continuation(ctx: PitContext, symbol: str, session: date) -> tuple[float | None, ...]:
    """``cont_h`` for each h of :data:`CONT_HORIZONS`: the all-adjusted close-to-close
    return from S to S+h, minus SPY's; None where a daily bar is missing."""
    days = path_days(session, max(CONT_HORIZONS) + 1)

    def close_all(sym: str, day: date) -> float | None:
        p = ctx.daily(sym, day)
        return p.close * p.adj if p is not None else None

    base, spy_base = close_all(symbol, session), close_all(SPY, session)
    out: list[float | None] = []
    for h in CONT_HORIZONS:
        x, m = close_all(symbol, days[h]), close_all(SPY, days[h])
        if base is None or spy_base is None or x is None or m is None:
            out.append(None)
        else:
            out.append((x / base - 1.0) - (m / spy_base - 1.0))
    return tuple(out)


# ── cells (pure) ───────────────────────────────────────────────


def regime(day: date, start: date) -> str:
    """``before`` a regime that starts on ``start``, else ``from``."""
    return "before" if day < start else "from"


def keys_of(row: AtlasRow) -> dict[str, tuple[str, ...]]:
    """Every table's key for a measured row."""
    t = row.type
    low = row.low_sigma
    return {
        "type": (t,),
        "base": (t, bucket(low) if low is not None else "n/a"),
        "timing": (t, row.timing),
        "rank": (t, "rank<300" if row.rank < RANK_SPLIT else "rank300-999"),
        "spy_vol": (t, row.spy_vol),
        "spy_trend": (t, row.spy_trend),
        "screen_regime": (t, regime(row.session, SCREEN_BREAK)),
        "analyst_regime": (t, regime(row.session, ANALYST_BREAK)),
        "sector": (t, "tech" if row.tech else "other"),
    }


@dataclass(frozen=True, slots=True)
class Exposure:
    """How long each cell's stories could arise in, in years of :data:`SESSIONS_PER_YEAR`
    sessions: the whole range, or for a :data:`DATED` table the sessions in its state."""

    total: float
    states: Mapping[tuple[str, str], float] = field(default_factory=dict)  # (table, state)

    def years(self, table: str, key: tuple[str, ...]) -> float:
        if table in DATED:
            return self.states.get((table, key[-1]), 0.0)
        return self.total


def exposure(sessions: Sequence[date], regimes: SpyRegimes) -> Exposure:
    """The range's :class:`Exposure`: its ``sessions`` counted whole, and by the state
    each :data:`DATED` table gives the session (the key a story of it would take)."""
    states: Counter[tuple[str, str]] = Counter()
    for day in sessions:
        states[("spy_vol", regimes.vol_tercile(day))] += 1
        states[("spy_trend", regimes.trend(day))] += 1
        states[("screen_regime", regime(day, SCREEN_BREAK))] += 1
        states[("analyst_regime", regime(day, ANALYST_BREAK))] += 1
    return Exposure(
        len(sessions) / SESSIONS_PER_YEAR,
        {k: n / SESSIONS_PER_YEAR for k, n in sorted(states.items())},
    )


def _finite(values: Iterable[float | None]) -> tuple[NDArray[np.float64], int]:
    """The finite values, and how many others there were (NaN or infinite; None is no value)."""
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    keep = np.isfinite(a)
    return a[keep], int(a.size - keep.sum())


def _share(flags: Sequence[bool]) -> float:
    return sum(flags) / len(flags) if flags else math.nan


def _mean_se(values: Sequence[float], clusters: Sequence[date]) -> tuple[float, float, int]:
    """The mean of the finite ``values`` and its CR1 standard error clustered by
    ``clusters`` (NaN when undefined), and how many values were not finite (left out:
    ``stats.clustered_mean`` refuses them, and one must not end a run)."""
    kept = [(v, c) for v, c in zip(values, clusters) if math.isfinite(v)]
    dropped = len(values) - len(kept)
    if not kept:
        return math.nan, math.nan, dropped
    xs = [v for v, _ in kept]
    cm = clustered_mean(xs, [c for _, c in kept])
    if cm is None:
        return float(np.mean(xs)), math.nan, dropped
    return float(cm.mean), float(cm.se), dropped


def cell_stats(rows: Sequence[AtlasRow]) -> dict[str, float]:
    """Every statistic of one cell's measured rows (spec §F).

    A value that is not finite (NaN or infinite) enters no statistic;
    ``nonfinite`` counts them.
    """
    out: dict[str, float] = {}
    nonfinite = 0
    ms = [r.measures for r in rows if r.measures is not None]
    series: dict[str, list[float | None]] = {
        "low_sigma": [m.low_sigma for m in ms],
        "t_low": [m.t_low for m in ms],
        "retrace_close": [m.retrace_close for m in ms],
        "retrace_s2": [m.retrace_max_s2 for m in ms],
    }
    for name, values in series.items():
        a, bad = _finite(values)
        nonfinite += bad
        qs = [float(x) for x in np.quantile(a, QUANTILES)] if a.size else [math.nan] * 5
        for q, v in zip(QUANTILES, qs):
            out[f"{name}_q{round(q * 100):02d}"] = v
    pairs_s = [
        (m.retrace_max_s, m.retrace_max_s2)
        for m in ms
        if m.retrace_max_s is not None and m.retrace_max_s2 is not None
    ]
    drops = [(a, b) for a, b in pairs_s if math.isfinite(a) and math.isfinite(b)]
    nonfinite += len(pairs_s) - len(drops)
    out["drops"] = float(len(drops))
    for x in RETRACE_LEVELS:
        out[f"p_retrace_{x:.2f}_close"] = _share([by_close >= x for by_close, _ in drops])
        out[f"p_retrace_{x:.2f}_s2"] = _share([by_s2 >= x for _, by_s2 in drops])
    fades = [m.fade for m in ms if m.fade is not None]
    out["fade_n"] = float(len(fades))
    out["p_fade"] = _share(fades)
    for i, h in enumerate(CONT_HORIZONS):
        pairs = [(c, r.session) for r in rows if (c := r.cont[i]) is not None]
        mean, se, bad = _mean_se([c for c, _ in pairs], [d for _, d in pairs])
        nonfinite += bad
        out[f"cont_{h}_n"] = float(len(pairs) - bad)
        out[f"cont_{h}_mean"] = mean
        out[f"cont_{h}_se"] = se
    for variant, _ in VARIANTS:
        machine_out, bad = _machine_stats(rows, variant)
        out.update(machine_out)
        nonfinite += bad
    out["nonfinite"] = float(nonfinite)
    return out


def _machine_stats(rows: Sequence[AtlasRow], variant: str) -> tuple[dict[str, float], int]:
    """The stories the machine started (``starters``), how many were ``blocked``
    behind a live story of their lane and their share; over the ones that ran:
    P(trigger), P(entry | trigger), exit-reason shares and mean r (CR1 SE by S);
    and how many trades' r was not finite (in the shares, not in the means)."""
    v = variant.lower()
    starters = [m for r in rows if (m := r.run(variant)) is not None]
    blocked = sum(m.blocked for m in starters)
    runs = [(r, m) for r in rows if (m := r.run(variant)) is not None and m.ran]
    out: dict[str, float] = {
        f"{v}_starters": float(len(starters)),
        f"{v}_blocked": float(blocked),
        f"{v}_blocked_share": blocked / len(starters) if starters else math.nan,
        f"{v}_runs": float(len(runs)),
    }
    if not runs:
        return out, 0
    triggered = [m for _, m in runs if m.triggered]
    out[f"{v}_p_trigger"] = len(triggered) / len(runs)
    out[f"{v}_p_entry_given_trigger"] = _share([m.entered for m in triggered])
    trades = [
        (r.session, m.exit_reason, ret)
        for r, m in runs
        if m.exit_reason is not None and (ret := m.r) is not None
    ]
    out[f"{v}_trades"] = float(len(trades))
    if not trades:
        return out, 0
    finite = [(d, why, ret) for d, why, ret in trades if math.isfinite(ret)]
    out[f"{v}_r_mean"], out[f"{v}_r_se"], _ = _mean_se(
        [ret for _, _, ret in finite], [d for d, _, _ in finite]
    )
    for reason in EXIT_REASONS:
        out[f"{v}_share_{reason}"] = sum(why == reason for _, why, _ in trades) / len(trades)
        some = [(d, ret) for d, why, ret in finite if why == reason]
        if some:
            out[f"{v}_r_mean_{reason}"], out[f"{v}_r_se_{reason}"], _ = _mean_se(
                [ret for _, ret in some], [d for d, _ in some]
            )
    return out, len(trades) - len(finite)


def _sort_key(table: str, key: tuple[str, ...]) -> tuple[Any, ...]:
    order = _ORDER.get(table, ())
    rest = tuple((order.index(k) if k in order else len(order), k) for k in key[1:])
    return (key[0], *rest)


def cells_of(rows: Sequence[AtlasRow], *, exposure: Exposure) -> list[AtlasCell]:
    """Every table's cells, sorted by key (never by a statistic).

    Measured rows (a path) fill the tables; ``coverage`` counts every row
    by type and path outcome. A cell under :data:`MIN_N` stories or
    :data:`MIN_DATES` dates carries counts only. ``per_year`` is n over the
    cell's own time (:meth:`Exposure.years`); NaN when that is none.
    """
    groups: dict[tuple[str, tuple[str, ...]], list[AtlasRow]] = defaultdict(list)
    for row in rows:
        groups[("coverage", (row.type, row.skip or "measured"))].append(row)
        if row.measures is None:
            continue
        for table, key in keys_of(row).items():
            groups[(table, key)].append(row)
    cells: list[AtlasCell] = []
    for (table, key), members in groups.items():
        n = len(members)
        dates = len({r.session for r in members})
        enough = table != "coverage" and n >= MIN_N and dates >= MIN_DATES
        years = exposure.years(table, key)
        cells.append(
            AtlasCell(
                table=table,
                key=key,
                n=n,
                dates=dates,
                per_year=n / years if years > 0 else math.nan,
                stats=cell_stats(members) if enough else None,
            )
        )
    cells.sort(key=lambda c: (TABLES.index(c.table), _sort_key(c.table, c.key)))
    return cells


# ── printing and the file ──────────────────────────────────────


def _f(x: float | None, spec: str) -> str:
    if x is None or not math.isfinite(x):
        return "-"
    return format(x, spec)


def _mse(stats: Mapping[str, float], mean: str, se: str) -> str:
    m, s = stats.get(mean), stats.get(se)
    if m is None or not math.isfinite(m):
        return "-"
    return f"{m:+.2%}" + (f" ({s:.2%})" if s is not None and math.isfinite(s) else "")


_HEAD: Final = (
    f"{'key':<44}{'n':>6}{'dates':>6}{'/yr':>7}"
    f"{'lowσ50':>8}{'tlow50':>8}{'retC50':>8}{'retS2_50':>9}"
    f"{'P.5C':>6}{'P.5S2':>6}{'Pfade':>6}{'cont5 (se)':>18}"
    f"{'ID Ptr':>7}{'Pen|tr':>7}{'ID r (se)':>18}{'MD3blk':>7}{'MD3 r (se)':>18}"
)


def _line(cell: AtlasCell) -> str:
    key = " / ".join(cell.key)
    head = f"{key[:43]:<44}{cell.n:>6}{cell.dates:>6}{_f(cell.per_year, '.1f'):>7}"
    s = cell.stats
    if s is None:
        return head + "  counts only"
    return (
        head
        + f"{_f(s.get('low_sigma_q50'), '.2f'):>8}{_f(s.get('t_low_q50'), '.0f'):>8}"
        + f"{_f(s.get('retrace_close_q50'), '.2f'):>8}{_f(s.get('retrace_s2_q50'), '.2f'):>9}"
        + f"{_f(s.get('p_retrace_0.50_close'), '.2f'):>6}{_f(s.get('p_retrace_0.50_s2'), '.2f'):>6}"
        + f"{_f(s.get('p_fade'), '.2f'):>6}{_mse(s, 'cont_5_mean', 'cont_5_se'):>18}"
        + f"{_f(s.get('id_p_trigger'), '.2f'):>7}{_f(s.get('id_p_entry_given_trigger'), '.2f'):>7}"
        + f"{_mse(s, 'id_r_mean', 'id_r_se'):>18}{_f(s.get('md3_blocked_share'), '.2f'):>7}"
        + f"{_mse(s, 'md3_r_mean', 'md3_r_se'):>18}"
    )


def tables(cells: Sequence[AtlasCell]) -> list[str]:
    """The tables as text, table by table, cells in key order.

    A selection of each cell's statistics; the file holds them all. No
    p-value is computed and no cell is ranked.
    """
    lines: list[str] = []
    for table in TABLES:
        mine = [c for c in cells if c.table == table]
        if not mine:
            continue
        lines += ["", f"== {TITLES[table]} ==", _HEAD]
        lines += [_line(c) for c in mine]
    lines += [
        "",
        f"Cells under {MIN_N} stories or {MIN_DATES} dates show counts only. Descriptive: "
        "no p-values, no ranking; an idea taken from here is a new registered trial.",
    ]
    return lines


def _plain(x: Any) -> Any:
    """JSON-ready: dates as ISO strings, non-finite floats as null, tuples as lists."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, datetime | date):
        return x.isoformat()
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_plain(v) for v in x]
    return x


def to_json(atlas: Atlas) -> dict[str, Any]:
    """The file's content: ``meta``, every cell, every row (``cont`` keyed by horizon)."""
    rows = []
    for r in atlas.rows:
        d = asdict(r)
        d["cont"] = dict(zip((f"cont_{h}" for h in CONT_HORIZONS), r.cont))
        rows.append(d)
    out: dict[str, Any] = _plain(
        {"meta": atlas.meta, "cells": [asdict(c) for c in atlas.cells], "rows": rows}
    )
    return out


def output_path(settings: Settings) -> Path:
    """``data/research/news_atlas-stories-v1.json`` under the settings' data dir (git-ignored)."""
    return settings.resolve_data_dir() / "research" / OUTPUT_NAME


def write_atlas(atlas: Atlas, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(atlas), sort_keys=True, allow_nan=False, indent=1))
    return path


# ── reading and running ────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Persisted:
    story_id: str
    symbol: str
    session: date
    labels: tuple[Any, ...]  # the _LABELS columns of its row


# The labels a rebuilt story must share with its news_stories row.
_LABELS: Final = ("type_detect", "type_close", "family_ever", "detect_at", "nsn_at", "at_news")


# The persisted labels and whether an item's type is outside TRAIN_SKIP_TYPES
# (as units.read_stories asks it): the items themselves are not read.
_CANDIDATES_SQL: Final = text(
    """
    SELECT s.story_id, s.symbol, s.session, s.type_detect, s.type_close, s.family_ever,
           s.detect_at, s.nsn_at, s.at_news,
           EXISTS (SELECT 1 FROM jsonb_array_elements(s.items) x
                   WHERE NOT (x->>'itype' = ANY(CAST(:skip AS text[])))) AS substantive
    FROM news_stories s
    WHERE s.builder_version = :v AND s.session >= :a AND s.session <= :b
    ORDER BY s.symbol, s.session
    """
)


async def candidates(
    engine: AsyncEngine, start: date, end: date, counts: Counter[str]
) -> dict[str, _Persisted]:
    """The PRIMARY candidates among the persisted stories with S in [start, end] that have
    a substantive item, by story id; the context is loaded one year at a time."""
    chosen: dict[str, _Persisted] = {}
    for year in range(start.year, end.year + 1):
        lo, hi = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    _CANDIDATES_SQL,
                    {
                        "v": builder.BUILDER_VERSION,
                        "a": lo,
                        "b": hi,
                        "skip": sorted(TRAIN_SKIP_TYPES),
                    },
                )
            ).all()
        counts["stories"] += len(rows)
        kept = [r for r in rows if r.substantive]
        counts["substantive"] += len(kept)
        if not kept:
            continue
        ctx = await PitContext.load(engine, symbols={r.symbol for r in kept}, start=lo, end=hi)
        for r in kept:
            at = builder.eligibility_at(r.nsn_at, r.at_news, r.detect_at, r.session)
            elig = ctx.eligibility(r.symbol, r.session, at_news=at)
            counts[f"eligibility:{elig.reason}"] += 1
            if elig.eligible:
                chosen[r.story_id] = _Persisted(
                    r.story_id, r.symbol, r.session, tuple(getattr(r, k) for k in _LABELS)
                )
        del ctx
    counts["candidates"] = len(chosen)
    return chosen


async def rebuild(
    engine: AsyncEngine,
    symbols: Sequence[str],
    aliases: Mapping[str, AliasMatcher],
    *,
    lo: date,
    hi: date,
) -> dict[str, list[Story]]:
    """The symbols' stories with S in [lo, hi], built as ``stories.build_range`` builds
    them: from every item since ``stories.HISTORY_FROM``, one symbol at a time."""
    out: dict[str, list[Story]] = {}
    raws = builder.load_items(engine, start=builder.HISTORY_FROM, end=hi, symbols=symbols)
    async for rows in builder._per_symbol(raws):
        built = builder.build(rows, aliases)
        out[rows[0].symbol] = [s for s in built if lo <= s.session <= hi]
    return out


@dataclass(frozen=True, slots=True)
class _Unit:
    story: Story
    detect: datetime
    at: datetime
    nsn: bool
    card: StoryCard  # at detect_at
    detect_at: datetime
    type_close: str
    family_ever: str | None
    elig: Eligibility
    pre: PreEvent | None

    @property
    def machine(self) -> bool:
        return self.nsn or self.card.direction == "neg"

    @property
    def lane(self) -> str | None:
        """The machine run it starts in: H1's (``NSN_CORE``) when NSN by the cutoff,
        else its detect type's; None when the machine does not run on it."""
        if not self.machine:
            return None
        return FAMILY if self.nsn else self.card.type

    @property
    def view(self) -> AtlasStory:
        return AtlasStory(self.story, self.detect, self.at)


def _units(
    built: Mapping[str, Sequence[Story]],
    chosen: Mapping[str, _Persisted],
    ctx: PitContext,
    *,
    start: date,
    end: date,
    counts: Counter[str],
) -> list[_Unit]:
    out: list[_Unit] = []
    for symbol in sorted(built):
        for story in built[symbol]:
            persisted = chosen.get(story.story_id)
            if persisted is None or not start <= story.session <= end:
                continue
            counts["rebuilt"] += 1
            row = builder.story_row(story)
            if tuple(row[k] for k in _LABELS) != persisted.labels:
                counts["persisted_mismatch"] += 1
                logger.warning("atlas: %s differs from its news_stories row", story.story_id)
            found = detection(story)
            if found is None or not substantive(i.itype for i in story.items):
                counts["dropped:not_substantive"] += 1
                continue
            detect, at, nsn = found
            elig = ctx.eligibility(symbol, story.session, at_news=at)
            if not elig.eligible:
                counts[f"dropped:{elig.reason}"] += 1
                continue
            detect_at = row["detect_at"]
            out.append(
                _Unit(
                    story=story,
                    detect=detect,
                    at=at,
                    nsn=nsn,
                    card=story.card_at(detect_at),
                    detect_at=detect_at,
                    type_close=row["type_close"],
                    family_ever=row["family_ever"],
                    elig=elig,
                    pre=ctx.pre_event(symbol, story.session, at),
                )
            )
    return out


def _row(
    u: _Unit, ctx: PitContext, regimes: SpyRegimes, skip: str | None, m: PathMeasures | None
) -> AtlasRow:
    s = u.story
    return AtlasRow(
        story_id=s.story_id,
        symbol=s.symbol,
        session=s.session,
        type=u.card.type,
        type_close=u.type_close,
        family_ever=u.family_ever,
        direction=u.card.direction,
        follower=u.card.follower,
        nsn=u.nsn,
        start_case=u.view.start_case(),
        detect_at=u.detect_at,
        ref_at=u.at,
        timing=timing_of(u.at, s.session),
        rank=u.elig.liquidity_rank if u.elig.liquidity_rank is not None else -1,
        tech=u.elig.tech,
        cost_bps=u.elig.cost_bps,
        spy_vol=regimes.vol_tercile(s.session),
        spy_trend=regimes.trend(s.session),
        sigma=u.pre.sigma if u.pre is not None else None,
        skip=skip,
        measures=m,
        cont=continuation(ctx, s.symbol, s.session),
        machine=u.machine,
        lane=u.lane,
    )


async def measure(
    engine: AsyncEngine,
    units: Sequence[_Unit],
    ctx: PitContext,
    regimes: SpyRegimes,
    unlock: WindowUnlock,
    counts: Counter[str],
) -> list[AtlasRow]:
    """Every unit's row: its S..S+2 path through the loader (behind the window guard)."""
    rows: list[AtlasRow] = []
    by_id = {u.story.story_id: u for u in units}
    requests = []
    for u in units:
        if u.pre is None:
            rows.append(_row(u, ctx, regimes, "no_pre_event", None))
            continue
        requests.append(
            PathRequest(u.story.story_id, u.story.symbol, u.story.session, PATH_SESSIONS)
        )
    if not requests:
        return rows
    loader = MinuteBarLoader(engine, window=Window.TRAIN, window_end=DATA_END, unlock=unlock)
    await loader.prepare()
    loader.check(requests)
    spy = await loader.spy()
    async for batch in loader.batches(requests, ctx):
        for item in batch:
            u = by_id[item.story_id]
            if isinstance(item, PathSkip):
                counts[f"skip:{item.reason}"] += 1
                rows.append(_row(u, ctx, regimes, item.reason, None))
                continue
            rows.append(_measured(u, item, spy.get(item.sessions[0].day), ctx, regimes, counts))
    return rows


def _measured(
    u: _Unit,
    path: PathData,
    spy: BarArrays | None,
    ctx: PitContext,
    regimes: SpyRegimes,
    counts: Counter[str],
) -> AtlasRow:
    assert u.pre is not None
    s = path.sessions[0]
    a_s = ctx.adj(u.story.symbol, s.day)
    if spy is None or a_s is None:  # the loader skips both; kept as a guard
        counts["skip:spy_missing" if spy is None else "skip:no_daily"] += 1
        return _row(u, ctx, regimes, "spy_missing" if spy is None else "no_daily", None)
    scales = [(ctx.adj(u.story.symbol, x.day) or a_s) / a_s for x in path.sessions]
    m = path_measures(
        path.bars,
        scales,
        spy,
        session=s,
        ref_at=u.at,
        start_case=u.view.start_case(),
        prev_close_s=u.pre.prev_close_s,
        spy_prev_close_s=u.pre.spy_prev_close_s,
        sigma=u.pre.sigma,
    )
    if m is None:
        counts["skip:no_anchor_bar"] += 1
        return _row(u, ctx, regimes, "no_anchor_bar", None)
    counts["measured"] += 1
    return _row(u, ctx, regimes, None, m)


def lanes_of(units: Sequence[_Unit]) -> dict[str, dict[str, _Unit]]:
    """The machine units by lane (:attr:`_Unit.lane`), then story id; lanes sorted.

    Every unit starts in its lane, eligible as all units are, as H1 starts an
    eligible NSN story whatever its pre-event state (without one, the
    playbook dismisses it at its start and it blocks nothing).
    """
    lanes: dict[str, dict[str, _Unit]] = defaultdict(dict)
    for u in units:
        if u.lane is not None:
            lanes[u.lane][u.story.story_id] = u
    return {lane: lanes[lane] for lane in sorted(lanes)}


async def machine(
    engine: AsyncEngine,
    units: Sequence[_Unit],
    built: Mapping[str, Sequence[Story]],
    ctx: PitContext,
    unlock: WindowUnlock,
    *,
    hold_sessions: int,
    workers: int,
    path_end: date,
) -> tuple[dict[str, MachineRun], dict[str, dict[str, Any]]]:
    """``OverreactionBounce`` without the family check on every machine unit: one
    ``sim.run`` per lane (:func:`lanes_of`), in which the lane's units start and every
    other story of their symbols (S up to ``path_end``) brings its news only.

    Returns each unit's run and each lane's run summary.
    """
    runs: dict[str, MachineRun] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for lane, starters in lanes_of(units).items():
        symbols = {u.story.symbol for u in starters.values()}
        views: list[AtlasStory] = []
        for symbol in sorted(symbols):
            for story in built.get(symbol, ()):
                if story.session > path_end:
                    continue
                u = starters.get(story.story_id)
                views.append(u.view if u is not None else AtlasStory.news_only(story))
        factory = BounceFactory(
            context={sid: (u.pre, u.elig) for sid, u in starters.items()},
            params=BounceParams(hold_sessions=hold_sessions, require_family=False),
        )
        sink = MemorySink()
        summary = await simulate(
            engine,
            views,
            factory,
            context=ctx,
            window=Window.TRAIN,
            window_end=DATA_END,
            cfg=SimConfig(),
            unlock=unlock,
            sink=sink,
            workers=workers,
        )
        runs.update({o.story_id: MachineRun.of(o) for o in sink.outcomes if o.story_id in starters})
        summaries[lane] = summary.as_dict()
    return runs, summaries


async def spy_state(engine: AsyncEngine) -> SpyRegimes:
    """SPY's regimes on every session of the atlas range, from a context of SPY alone
    loaded for [:data:`ATLAS_START`, :data:`ATLAS_END`] (daily bars to
    :data:`DATA_END`, never after); the tercile edges from [ATLAS_START, DATA_END]."""
    ctx = await PitContext.load(engine, symbols=(), start=ATLAS_START, end=context_end(ATLAS_END))
    sessions = [d for d in ctx.sessions if d <= DATA_END]

    def close_all(day: date) -> float | None:
        p = ctx.daily(SPY, day)
        return p.close * p.adj if p is not None else None

    return spy_regimes(
        sessions, [close_all(d) for d in sessions], edges_from=ATLAS_START, edges_to=DATA_END
    )


def _merge(summaries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """The sim runs' counts (``RunSummary.as_dict``), summed over symbol batches."""
    out: Counter[str] = Counter()
    for s in summaries:
        for k in ("stories", "started", "outcomes", "entries", "trades"):
            out[k] += int(s.get(k, 0))
        for group in ("terminal", "skips"):
            for k, v in dict(s.get(group, {})).items():
                out[f"{group}:{k}"] += int(v)
    return dict(sorted(out.items()))


async def run_atlas(
    engine: AsyncEngine,
    *,
    start: date = ATLAS_START,
    end: date = ATLAS_END,
    workers: int = 1,
    batch_symbols: int = BATCH_SYMBOLS,
) -> Atlas:
    """Every unit's row and every table's cells, with what the run read (``meta``).

    Refuses (:class:`AtlasLocked`) before H1's verdict, and (``ValueError``,
    before anything is read) a range outside [:data:`ATLAS_START`,
    :data:`ATLAS_END`] or an ``end`` that is not a session. Symbols go in
    batches of ``batch_symbols``: stories rebuilt, paths measured, then the
    machine run for ID and MD3.
    """
    check_range(start, end)
    reg = await h1_closed(engine)
    unlock = WindowUnlock(prereg_id=reg.id, config_hash=reg.config_hash)
    counts: Counter[str] = Counter()
    chosen = await candidates(engine, start, end, counts)
    symbols = sorted({p.symbol for p in chosen.values()})
    meta: dict[str, Any] = {
        "builder_version": builder.BUILDER_VERSION,
        "pins": await builder.pins(engine),
        "start": start,
        "end": end,
        "data_end": DATA_END,
        "registration": asdict(reg),
        "unit_excluded": sorted(TRAIN_SKIP_TYPES),
        "rules": {
            "min_n": MIN_N,
            "min_dates": MIN_DATES,
            "buckets": list(BUCKETS),
            "quantiles": list(QUANTILES),
            "retrace_levels": list(RETRACE_LEVELS),
            "fade_after": FADE_AFTER,
            "cont_horizons": list(CONT_HORIZONS),
            "bounce": BounceParams(require_family=False).as_config(),
        },
    }
    rows: list[AtlasRow] = []
    summaries: dict[str, dict[str, list[dict[str, Any]]]] = {
        v: defaultdict(list) for v, _ in VARIANTS
    }
    regimes = await spy_state(engine)
    meta["spy_vol_edges"] = list(regimes.edges)
    if symbols:
        ctx = await PitContext.load(engine, symbols=symbols, start=start, end=context_end(end))
        aliases = await load_aliases(engine)
        path_end = path_days(end, PATH_SESSIONS)[-1]
        items_end = min(DATA_END, end + timedelta(days=14))
        for i in range(0, len(symbols), batch_symbols):
            batch = symbols[i : i + batch_symbols]
            built = await rebuild(engine, batch, aliases, lo=start, hi=items_end)
            units = _units(built, chosen, ctx, start=start, end=end, counts=counts)
            measured = await measure(engine, units, ctx, regimes, unlock, counts)
            runs: dict[str, dict[str, MachineRun]] = {}
            for variant, hold in VARIANTS:
                found, summary = await machine(
                    engine,
                    units,
                    built,
                    ctx,
                    unlock,
                    hold_sessions=hold,
                    workers=workers,
                    path_end=path_end,
                )
                runs[variant] = found
                for lane, lane_summary in summary.items():
                    summaries[variant][lane].append(lane_summary)
            rows += [
                replace(r, id_run=runs["ID"].get(r.story_id), md3_run=runs["MD3"].get(r.story_id))
                for r in measured
            ]
            logger.info("atlas: %d of %d symbols, %d rows", i + len(batch), len(symbols), len(rows))
    rows.sort(key=lambda r: (r.session, r.story_id))
    counts["missing_rebuilt"] = len(chosen) - counts["rebuilt"]
    meta["counts"] = dict(sorted(counts.items()))
    meta["machine"] = {
        v: _merge([x for lane in by_lane.values() for x in lane])
        for v, by_lane in summaries.items()
    }
    meta["machine_lanes"] = {
        v: {lane: _merge(by_lane[lane]) for lane in sorted(by_lane)}
        for v, by_lane in summaries.items()
    }
    exposed = exposure(h1.sessions_between(start, end), regimes)
    meta["years"] = exposed.total
    meta["state_years"] = {f"{t}:{k}": y for (t, k), y in exposed.states.items()}
    cells = cells_of(rows, exposure=exposed)
    return Atlas(rows=rows, cells=cells, meta=meta)


async def build_atlas(
    engine: AsyncEngine, *, start: date = ATLAS_START, end: date = ATLAS_END
) -> list[AtlasCell]:
    """The atlas's cells (spec §F's signature); :func:`run_atlas` keeps the rows too."""
    return (await run_atlas(engine, start=start, end=end)).cells


__all__ = [
    "ATLAS_END",
    "ATLAS_START",
    "BUCKETS",
    "CONT_HORIZONS",
    "DATA_END",
    "MIN_DATES",
    "MIN_N",
    "OUTPUT_NAME",
    "TABLES",
    "DATED",
    "Atlas",
    "AtlasCell",
    "AtlasLocked",
    "AtlasRow",
    "AtlasStory",
    "Exposure",
    "MachineRun",
    "PathMeasures",
    "Registration",
    "SpyRegimes",
    "bucket",
    "build_atlas",
    "candidates",
    "cell_stats",
    "cells_of",
    "check_range",
    "context_end",
    "continuation",
    "detection",
    "exposure",
    "h1_closed",
    "keys_of",
    "lanes_of",
    "output_path",
    "path_measures",
    "regime",
    "rebuild",
    "run_atlas",
    "spy_regimes",
    "spy_state",
    "substantive",
    "tables",
    "timing_of",
    "to_json",
    "write_atlas",
]
