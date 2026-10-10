"""The Phase 0 gate runner (spec §E.1-E.3): the simulator earns trust before H1 registers.

Every gate writes one ledger row per run: ``quant_trials`` ``name=``
:data:`GATE_NAME`, ``kind='gate'``, ``config={"sim": SimConfig().as_config(),
"gate": <id>, "seed": 20261010, ...}``, ``verdict`` "pass" or "fail", the
numbers in ``metrics`` and the rule in ``criterion``. No Sharpe, so no row
counts as a trial; a rerun after a bug fix is a new row, and the H1 runner
reads the latest row of each id (:data:`GATE_IDS`). Every gate is
deterministic (seed 20261010), so a rerun on the same data writes the same
numbers.

**Unit sets.** A gate that reads minute bars pins its unit set first,
``loader.register_gate_units(engine, gate, units, expected_sha=UnitPlan.sha(part))``
with the units module's own selection (``units.g1_stories``,
``units.calib_pairs``, ``units.sue_complement`` + ``units.sue_sample``,
``units.reactor_headlines``), never a rebuilt one, and **refuses to run**
(:class:`GateRefused`, no row) while any of those units, or SPY's on their
days, is not a done unit of the minute backfill. Bars are read only through
the loader's window guard (the gate unlock admits exactly the pinned set).

**G1, mechanics and look-ahead** (``lookahead``: :func:`run_lookahead`):

* ``g1-lookahead``: the 500 ``gate_g1`` stories (S in 2016-01-04..2016-09-30)
  under the bounce, with the family check for the NSN stories and without it
  for the other negative ones, in both H1 cells (ID and MD3; MD3 paths cut at
  2016-09-30, ``units.g1_path_sessions`` when the units module has it). The
  universe is liquidity only (:class:`LiquidityOnly`, :func:`liquidity_eligibility`):
  no screen exists before 2016-10, so rank < 1000, a previous close of $5 and
  σ decide, and the simulator's screen check reads "halal". For 5 seeded
  decision times T per story the symbol is replayed with everything unknown
  at T changed (``halabot.playbooks.lookahead``, its fill-bar exemption
  included): every intent and transition at or before T, and the set of
  stories started by T, must be identical, and every fill bar must start at
  or after its order's active time.
* ``g1-synthetic``: the same on 1,000 synthetic paths (:func:`synthetic_world`).
* ``g1-determinism``: 1 and 6 workers give the same sha256 over the records.

**G2, the reactor replication** (``reactor``: :func:`run_reactor`; the bars
are in the holdout, 2025-12..2026-10, read as the gate's pinned set only):

* ``r0``: the legacy ``intraday.run`` on the pinned headlines, with a market
  that raises on any fetch, reproduces n = 7,922, -0.30% (2 dp), t -12.1
  (1 dp), controls -0.28%, score <= -0.4 -0.40%;
* ``r1``: the simulator (HARNESS feed, 60 s news lag, ``LegacyReactorFill``)
  on the explicit headline set matches every headline to 1e-10 with an
  identical dropped set (``legacy.r1_dropped``: its set-aside rules and their
  1% cap);
* ``r2``: realistic fills (SIP_RT, ORDER_LAG, the D.5 market rule, flatten at
  close - 5 min, 60 s news lag): the strong group's mean inside the legacy
  95% CI clustered by New York date; the clustered t and a per-headline
  decomposition (entry rule, exit rule) are reported.

**G3, the SUE replication** (``sue``: :func:`run_sue`):

* ``s0``: ``events study sue --start 2016 --end 2019 --by bucket``, recomputed
  (the 8-K times are being corrected from EDGAR headers, so the values it
  computes become the reference) with its deltas against the numbers the
  plan recorded (IC +0.04..0.09 at 5-20 d, mid-cap 20 d D10-D1 +1.98%); then
  on Σ_c in daily mode, D10-D1 and IC at 5 and 20 days with 95% bootstrap
  CIs over entry-session clusters (B = 2,000);
* ``s1``: ``legacy.DailyBarSource`` through the simulator equals
  ``study.evaluate`` on Σ_c to 1e-10 at h in {1, 5, 20, 60}, with identical
  observation sets and decile rows (early-close publications excluded and
  counted);
* ``s1-calib``: p99 of |minute - daily| abnormal return on ``gate_calib``
  (09:30 entry, h = 5 close exit), recorded;
* ``s2``: minute mode on the study's clock on Σ_s, paired d = minute - daily:
  the 90% clustered CI within ±0.10% (TOST at α 0.05), median |d| <= 0.10%,
  p99 |d| <= max(1.00%, 1.5 p99_cal), at h = 5 and h = 20;
* ``s3``: realistic fills on Σ_s with the exchange's own functions on the
  entry and exit sessions' bars (entry by the D.5 market rule at the first
  bar with ts >= max(published + 600 s, 09:30); exit by the flatten rule):
  D10-D1 and IC inside the daily-mode 95% CI recomputed on the same events,
  with the same sign.

Decisions the spec leaves open, stated for the pre-registration:

* G1 runs both cells (ID and MD3), five T per story per cell, drawn uniformly
  over [S's open - 30 min, the close of the story's last path session], and
  noises the official prices of the sessions from the probed story's S on.
  The determinism check partitions symbols as ``sim.run`` does
  (``sim.simulate_many``), in this process.
* R1 and R2 run through ``sim.run`` on the explicit set; R2 uses a gate-only
  copy of the D.5 market rule (:class:`GateMarketFill`) so that, like the
  study, it admits every headline (no screen, no entry window), and
  :class:`FlattenHold` sells at the flatten. R2 compares the headlines R1 did
  not set aside, with the study's plausibility filter.
* SUE clusters are the entry session (the study's entry day). D10-D1 uses the
  study's deciles (``study.summarise``). The calibration, S2 and S3 need their
  returns on at least :data:`MIN_COVERAGE` of their pairs or events, a floor
  set here so a comparison on a minority cannot pass. A first bar after 09:30
  is still the minute entry of an open entry (counted ``late_first_bar``).
* S0 passes when the table and the Σ_c statistics compute; its deltas against
  the recorded numbers are reported, not judged (the data vintage changed).
* Refusals are per gate (:class:`GateRun`): ``g1-synthetic`` reads no stored
  bar and always runs; S0 and S1 read daily bars only; S2 is refused unless
  this run's ``s1-calib`` passed (its p99 is S2's bound). ``events sim-gate
  all`` runs every group whatever another refused.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks import bounce, legacy
from halabot.playbooks.bounce import BounceParams, EligibilityLike, OverreactionBounce, PreEventLike
from halabot.playbooks.clock import ORDER_LAG, span_us, to_us
from halabot.playbooks.exchange import (
    FallbackFill,
    MarketFill,
    first_eligible,
    market_fill,
    spy_fill,
    unfilled_exit,
)
from halabot.playbooks.interfaces import CardView, ContextView, DailyPointLike, StoryView
from halabot.playbooks.loader import (
    Gate,
    MinuteBarLoader,
    PathRequest,
    Window,
    WindowGuard,
    WindowUnlock,
    check_calendar,
    register_gate_units,
    sane,
)
from halabot.playbooks.lookahead import Probe, check_symbol, fill_bar_violations
from halabot.playbooks.playbook import TRIGGERED, Ctx
from halabot.playbooks.records import MemorySink, TradeRecord, outcomes_sha256
from halabot.playbooks.sim import RunSummary, run, simulate_many, simulate_symbol
from halabot.playbooks.types import (
    BarSeries,
    Cancel,
    FillIn,
    Finish,
    Input,
    Intent,
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
from halal_trader.core import events
from halal_trader.core.signal_eval import information_coefficient
from halal_trader.data import minutes
from halal_trader.data.minutes import BarArrays
from halal_trader.events import stats, study, units
from halal_trader.events.context import MAX_RANK, MIN_PREV_CLOSE, PreEvent
from halal_trader.events.intraday import STRONG, Headline, Outcome
from halal_trader.events.stories import (
    HISTORY_FROM,
    NEWS_LAG,
    RawItem,
    Story,
    StoryItem,
    build,
    load_items,
)
from halal_trader.events.study import BENCHMARK, Observation
from halal_trader.events.taxonomy import NOISE_TYPES, TYPES
from halal_trader.events.units import GateStory, SueEvent, Unit, UnitPlan
from halal_trader.market_hours import MARKET_TZ, is_trading_day, next_trading_day

if TYPE_CHECKING:
    from halal_trader.events.context import PitContext

logger = logging.getLogger(__name__)

GATE_NAME: Final = "research.news.sim-gate"
GATE_KIND: Final = "gate"
SEED: Final = 20261010
SPY: Final = BENCHMARK

G1_IDS: Final = ("g1-lookahead", "g1-synthetic", "g1-determinism")
REACTOR_IDS: Final = ("r0", "r1", "r2")
SUE_IDS: Final = ("s0", "s1", "s1-calib", "s2", "s3")
GATE_IDS: Final = (*G1_IDS, *REACTOR_IDS, *SUE_IDS)
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "lookahead": G1_IDS,
    "reactor": REACTOR_IDS,
    "sue": SUE_IDS,
}
Group = Literal["lookahead", "reactor", "sue"]

# ── G1 ──
G1_CELLS: Final = (1, 3)  # H1's ID and MD3 holds
G1_PER_STORY: Final = 5  # decision times T per story and cell
G1_SYNTHETIC: Final = 1_000
G1_T_BEFORE_OPEN: Final = timedelta(minutes=30)
DETERMINISM_WORKERS: Final = 6
MISMATCHES_SHOWN: Final = 20

# ── G2 ──
R0_EXPECTED: Final[dict[str, float]] = {
    "headlines": 30_614,  # first_in_session(scored_before=2026-10-10T00:00Z)
    "strong_headlines": 7_932,  # of the pinned set, score >= 0.4
    "strong_n": 7_922,  # score >= 0.4, same day
    "strong_mean_pct": -0.30,  # 2 dp
    "strong_t": -12.1,  # 1 dp, iid
    "control_mean_pct": -0.28,
    "negative_mean_pct": -0.40,
}
STRONG_LABEL: Final = f"score >= {STRONG}"
CONTROL_LABEL: Final = "neutral (control)"
NEGATIVE_LABEL: Final = f"score <= -{STRONG}"
R1_TOL: Final = 1e-10
REACTOR_END: Final = date(2026, 10, 9)  # the reactor gate's last session (loader.GATE_RANGES)
CI_LEVEL: Final = 0.95

# ── G3 ──
S0_YEARS: Final = (2016, 2019)
S0_RECORDED: Final[dict[str, Any]] = {
    "ic_range_5_20d": (0.04, 0.09),  # docs/MODERNIZATION_PLAN.md, 2026-10-09
    "mid_20d_d10_d1": 0.0198,
}
MID: Final = "mid (300-1000)"
SUE_H: Final = units.SUE_HORIZONS  # (5, 20)
S1_TOL: Final = 1e-10
S1_T_REL: Final = 1e-8
TOST_MARGIN: Final = 0.001  # ±0.10%
TOST_ALPHA: Final = 0.05
MEDIAN_MAX: Final = 0.001
P99_FLOOR: Final = 0.01
P99_CAL_X: Final = 1.5
MIN_COVERAGE: Final = 0.90
S3_NEWS_LAG: Final = timedelta(seconds=600)
SUE_END: Final = units.SUE_RANGE[1]

CRITERIA: Final[dict[str, str]] = {
    "g1-lookahead": (
        "500 gate_g1 stories, cells ID and MD3, 5 seeded T each: every intent and transition "
        "at or before T and the stories started by T identical after perturbing what is unknown "
        "at T; no fill bar before its order's active time"
    ),
    "g1-synthetic": "the same look-ahead invariance on 1,000 synthetic paths",
    "g1-determinism": "1 and 6 workers give the same sha256 over the sorted records, each cell",
    "r0": (
        "intraday.run on the pinned set, no fetch: score >= 0.4 same day n = 7,922, mean -0.30% "
        "(2 dp), t -12.1 (1 dp); controls -0.28%; score <= -0.4 -0.40%; 30,614 headlines, "
        "7,932 at score >= 0.4"
    ),
    "r1": (
        "the simulator (HARNESS, 60 s lag, LegacyReactorFill) on the explicit set: identical "
        "dropped sets after the set-aside (<= 1% of H) and |dr| <= 1e-10 on every headline kept"
    ),
    "r2": (
        "SIP_RT, ORDER_LAG, D.5 market rule, flatten at close - 5 min: the strong group's mean "
        "inside the legacy 95% CI clustered by New York date"
    ),
    "s0": (
        "events study sue 2016-2019 by bucket recomputed (the reference) and Σ_c daily-mode "
        "D10-D1 and IC at 5 and 20 days with 95% date-cluster bootstrap CIs (B = 2,000)"
    ),
    "s1": (
        "DailyBarSource through the simulator equals study.evaluate on Σ_c within 1e-10 at "
        "h in {1, 5, 20, 60}, identical observations and decile rows"
    ),
    "s1-calib": "p99 of |minute - daily| on gate_calib (09:30 entry, h = 5 close exit), recorded",
    "s2": (
        "Σ_s minute mode on the study's clock, h = 5 and 20: TOST 90% clustered CI within "
        "±0.10%, median |d| <= 0.10%, p99 |d| <= max(1.00%, 1.5 p99_cal)"
    ),
    "s3": (
        "Σ_s realistic fills: D10-D1 and IC at 5 and 20 days inside the daily-mode 95% CI "
        "recomputed on the same events, same sign"
    ),
}


# ── results and the ledger ────────────────────────────────────


class GateRefused(RuntimeError):
    """A gate that may not run yet (units not done, stale stories): no row is written."""


@dataclass(frozen=True, slots=True)
class GateResult:
    """One gate's run: its verdict, numbers and what its config pins beyond the constants."""

    gate: str
    passed: bool
    metrics: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)
    window: str = ""

    @property
    def verdict(self) -> str:
        return "pass" if self.passed else "fail"


@dataclass(slots=True)
class GateRun:
    """What a run of gates did: the rows it wrote, and the gates it refused (id -> why)."""

    results: list[GateResult] = field(default_factory=list)
    refused: dict[str, str] = field(default_factory=dict)

    def refuse(self, gates: Iterable[str], why: str) -> None:
        for g in gates:
            self.refused[g] = why

    def extend(self, other: GateRun) -> None:
        self.results += other.results
        self.refused.update(other.refused)

    @property
    def passed(self) -> bool:
        """Every gate ran and passed."""
        return not self.refused and all(r.passed for r in self.results)


def gate_config(gate: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The row's config: the simulator's constants, the gate id, the seed, then ``extra``."""
    if gate not in GATE_IDS:
        raise ValueError(f"unknown gate {gate!r}")
    return {"sim": SimConfig().as_config(), "gate": gate, "seed": SEED, **dict(extra or {})}


def to_json(value: Any) -> Any:
    """``value`` as plain JSON: NaN and infinities None, dates ISO, tuples and sets lists."""
    if isinstance(value, bool | str) or value is None:
        return value
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, set | frozenset):
        return [to_json(v) for v in sorted(value)]
    if isinstance(value, list | tuple):
        return [to_json(v) for v in value]
    return str(value)


async def record_gate(engine: AsyncEngine, result: GateResult) -> int:
    """Write ``result`` as its gate row; returns the row id."""
    from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl

    row = await QuantTrialRepoImpl(engine).record_trial(
        name=GATE_NAME,
        kind=GATE_KIND,
        config=to_json(gate_config(result.gate, result.config)),
        window=result.window,
        metrics=to_json(result.metrics),
        criterion=CRITERIA[result.gate],
        verdict=result.verdict,
    )
    logger.info(
        "sim gate %s: %s (quant_trials %d)",
        result.gate,
        result.verdict,
        row,
        extra={
            "event": events.SIM_GATE_RUN,
            "gate": result.gate,
            "verdict": result.verdict,
            "trial_id": row,
        },
    )
    return row


# ── unit sets ─────────────────────────────────────────────────


def with_spy(unit_set: Iterable[Unit]) -> frozenset[Unit]:
    """A unit set and SPY's units on its days."""
    out = set(unit_set)
    return frozenset(out | {(SPY, d) for _, d in out})


async def pin_units(engine: AsyncEngine, gate: Gate, part: str, unit_set: Iterable[Unit]) -> str:
    """Pin a gate's unit set (the units module's selection for ``part``); returns its sha256."""
    plan = UnitPlan({part: frozenset(unit_set)})
    return await register_gate_units(engine, gate, plan.parts[part], expected_sha=plan.sha(part))


async def require_done(engine: AsyncEngine, gates: Sequence[str], unit_set: Iterable[Unit]) -> None:
    """:class:`GateRefused` unless every unit, and SPY's on its days, is a done unit."""
    done = await minutes.done_units(engine)
    wanted = with_spy(unit_set)
    missing = sorted(u for u in wanted if minutes.unit(*u) not in done)
    if missing:
        shown = ", ".join(minutes.unit(*u) for u in missing[:5])
        logger.warning(
            "sim gate %s refused: %d of %d units not done",
            "/".join(gates),
            len(missing),
            len(wanted),
            extra={
                "event": events.SIM_GATE_REFUSED,
                "gates": list(gates),
                "missing": len(missing),
            },
        )
        raise GateRefused(
            f"{'/'.join(gates)}: {len(missing)} of {len(wanted)} units are not done "
            f"(first: {shown}); run `halal-trader data minutes --plan h1`"
        )


async def read_gate_bars(
    engine: AsyncEngine,
    gate: Gate,
    unit_set: frozenset[Unit],
    sha: str,
    window_end: date,
) -> tuple[dict[Unit, BarArrays], int]:
    """Every unit of the pinned set and SPY's on its days, read behind the window guard and
    cut by the loader's sanity rule; returns the bars and the rows the rule dropped."""
    guard = WindowGuard(
        window=Window.GATE,
        window_end=window_end,
        unlock=WindowUnlock(gate=gate, units=unit_set, units_sha=sha),
    )
    await guard.verify(engine)
    wanted = sorted(with_spy(unit_set))
    for symbol, day in wanted:
        guard.check(symbol, day)
    await check_calendar(engine, min(d for _, d in wanted), max(d for _, d in wanted))
    raw = await minutes.read_windows(engine, wanted)
    out: dict[Unit, BarArrays] = {}
    dropped = 0
    for u, arrays in raw.items():
        kept, n = sane(arrays)
        out[u] = kept
        dropped += n
    return out, dropped


def ny_day(t: datetime) -> date:
    return t.astimezone(MARKET_TZ).date()


# ── G1: stories, eligibility and context as the gate runs them ──

Mode = Literal["nsn", "other"]


@dataclass(frozen=True, slots=True)
class G1Story:
    """A story as G1 runs it (a ``StoryView`` that can forget its items).

    ``mode`` "nsn": the story's own detection (``nsn_at``, ``at_news``), run
    with the family check; "other": a negative story that is not NSN by its
    cutoff, detected at its first substantive item (the atlas's rule), run
    without it; None: the story never starts and only brings its items.
    """

    story: Story
    mode: Mode | None

    @property
    def story_id(self) -> str:
        return self.story.story_id

    @property
    def symbol(self) -> str:
        return self.story.symbol

    @property
    def session(self) -> date:
        return self.story.session

    def card_at(self, t: datetime) -> CardView:
        return self.story.card_at(t)

    def _detect(self) -> tuple[datetime, datetime] | None:
        if self.mode is None:
            return None
        cutoff = Session.of(self.session).entry_cutoff
        if self.mode == "nsn":
            nsn, at = self.story.nsn_at(cutoff), self.story.at_news()
            return (nsn, at) if nsn is not None and at is not None else None
        first = next((i for i in self.story.items if i.itype not in NOISE_TYPES), None)
        if first is None or first.available_at > cutoff:
            return None
        return first.available_at, first.at

    def nsn_at(self, cutoff: datetime) -> datetime | None:
        hit = self._detect()
        return hit[0] if hit is not None and hit[0] <= cutoff else None

    def at_news(self) -> datetime | None:
        hit = self._detect()
        return hit[1] if hit is not None else None

    def start_case(self) -> str:
        at = self.at_news()
        if at is None:
            return "out"
        s = Session.of(self.session)
        return "in" if s.open <= at < s.close else "out"

    def news_times(self) -> list[datetime]:
        return self.story.news_times()

    def before(self, t: datetime) -> G1Story:
        kept = [i for i in self.story.items if i.available_at <= t]
        return G1Story(replace(self.story, items=kept), self.mode)


@dataclass(frozen=True, slots=True)
class LiquidityEligibility:
    """G1's universe (liquidity only): the fields the bounce reads of an eligibility."""

    eligible: bool
    reason: str
    liquidity_rank: int | None
    cost_bps: float
    tech: bool = False


class PitLike(Protocol):
    """What :func:`liquidity_eligibility` asks ``context.PitContext``."""

    def eligibility(
        self,
        symbol: str,
        session: date,
        *,
        at_news: datetime,
        universe: Literal["primary", "broad"] = ...,
    ) -> Any: ...

    def pre_event(self, symbol: str, session: date, at_news: datetime) -> PreEvent | None: ...


def liquidity_eligibility(
    ctx: PitLike, symbol: str, session: date, at_news: datetime
) -> tuple[PreEvent | None, LiquidityEligibility]:
    """Rank < 1000, a previous close of $5 in S units and σ, all at ``at_news``; no screen.

    Before 2016-10 no screen exists, so the context's own eligibility stops at
    ``no_screen``; only its liquidity rank (and the study's cost for it) is
    read, and the pre-event state gives the price and σ.
    """
    try:
        rank = ctx.eligibility(symbol, session, at_news=at_news, universe="broad").liquidity_rank
        pre = ctx.pre_event(symbol, session, at_news)
    except ValueError:
        return None, LiquidityEligibility(False, "no_pre_event", None, study.cost_bps(None))
    cost = study.cost_bps(rank)
    if rank is None or rank >= MAX_RANK:
        reason = "rank"
    elif pre is None:
        reason = "no_pre_event"
    elif pre.prev_close_s < MIN_PREV_CLOSE:
        reason = "price"
    else:
        reason = "ok"
    return pre, LiquidityEligibility(reason == "ok", reason, rank, cost)


@dataclass(frozen=True, slots=True)
class LiquidityOnly:
    """A ``ContextView`` whose screen admits every name (G1's liquidity-only universe);
    A-factors, daily bars and sessions are ``base``'s."""

    base: ContextView

    @property
    def sessions(self) -> Sequence[date]:
        return self.base.sessions

    def adj(self, symbol: str, day: date) -> float | None:
        return self.base.adj(symbol, day)

    def screen_verdict(self, symbol: str, day: date) -> str:
        return "halal"

    def daily(self, symbol: str, day: date) -> DailyPointLike | None:
        return self.base.daily(symbol, day)


@dataclass(frozen=True, slots=True)
class G1Factory:
    """The bounce for each G1 story: its own hold and family check, its own context."""

    context: Mapping[str, tuple[PreEventLike | None, EligibilityLike]]
    params: Mapping[str, BounceParams]
    hold: int
    name: str = bounce.NAME
    version: str = bounce.VERSION

    @property
    def path_sessions(self) -> int:
        return self.hold

    def __call__(self, story: StoryView) -> OverreactionBounce:
        pre, elig = self.context[story.story_id]
        return OverreactionBounce(story, pre, elig, self.params[story.story_id])


@dataclass(slots=True)
class LookaheadWorld:
    """Everything one look-ahead run needs, per cell hold."""

    stories: list[G1Story]  # every story of the symbols; the gate's have a mode
    paths: dict[int, dict[str, PathData | PathSkip]]  # hold -> story id -> path
    spy: SpyData
    ctx: ContextView
    context: dict[str, tuple[PreEventLike | None, EligibilityLike]]
    holds: dict[int, dict[str, int]]  # hold -> story id -> sessions (cut at the window)

    def gate_stories(self) -> list[G1Story]:
        return sorted((s for s in self.stories if s.mode is not None), key=lambda s: s.story_id)

    def factory(self, hold: int) -> G1Factory:
        params = {
            s.story_id: BounceParams(
                hold_sessions=self.holds[hold][s.story_id], require_family=s.mode == "nsn"
            )
            for s in self.gate_stories()
        }
        return G1Factory(self.context, params, hold)

    def by_symbol(self) -> dict[str, list[G1Story]]:
        out: dict[str, list[G1Story]] = defaultdict(list)
        for s in sorted(self.stories, key=lambda s: (s.session, s.story_id)):
            out[s.symbol].append(s)
        return dict(out)


def _probes(
    story: G1Story, path: PathData | PathSkip | None, rng: np.random.Generator, n: int
) -> list[Probe]:
    """``n`` decision times over [S's open - 30 min, the close of the story's last path day]."""
    last = path.sessions[-1].day if isinstance(path, PathData) else story.session
    lo = Session.of(story.session).open - G1_T_BEFORE_OPEN
    span = (Session.of(last).close - lo).total_seconds()
    return [
        Probe(lo + timedelta(seconds=int(rng.uniform(0.0, span))), since=story.session)
        for _ in range(n)
    ]


def lookahead(
    world: LookaheadWorld,
    *,
    seed: int = SEED,
    per_story: int = G1_PER_STORY,
    cfg: SimConfig | None = None,
    require_trades: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """The look-ahead invariance of every gate story of ``world`` in each cell.

    Passes when every replay agrees with its base run, at least one
    comparison is made per cell, and no fill bar starts before its order's
    active time; ``require_trades`` also needs trades with exits (a synthetic
    market built to exercise the machine).
    """
    cfg = cfg or SimConfig()
    by_symbol = world.by_symbol()
    cells: dict[str, Any] = {}
    passed = True
    for hold in sorted(world.holds):
        factory = world.factory(hold)
        paths = world.paths[hold]
        times = np.random.default_rng([seed, hold, 0])
        noise = np.random.default_rng([seed, hold, 1])
        checked = probes = 0
        mismatches: list[tuple[str, str, str]] = []
        violations: list[str] = []
        terminal: Counter[str] = Counter()
        entries = trades = started = 0
        for symbol in sorted(by_symbol):
            stories = by_symbol[symbol]
            gate = sorted((s for s in stories if s.mode is not None), key=lambda s: s.story_id)
            if not gate:
                continue
            base = simulate_symbol(
                symbol, stories, factory, paths, world.spy, world.ctx, cfg, keep_transitions=True
            )
            todo = [p for s in gate for p in _probes(s, paths.get(s.story_id), times, per_story)]
            check = check_symbol(
                symbol, stories, factory, paths, world.spy, world.ctx, cfg, todo, noise, base=base
            )
            checked += check.checked
            probes += check.probes
            mismatches += [(m.story_id, m.t.isoformat(), m.what) for m in check.mismatches]
            violations += fill_bar_violations(base)
            for o in base:
                terminal[f"{o.terminal_state}/{o.reason}"] += 1
                started += o.start_at is not None
                entries += o.entered
                trades += o.trade is not None
        exits = sum(n for k, n in terminal.items() if k.startswith("EXITED/"))
        ok = not mismatches and not violations and checked > 0
        if require_trades:
            ok = ok and trades > 0 and exits > 0
        passed = passed and ok
        cells[f"hold{hold}"] = {
            "passed": ok,
            "stories": len(world.gate_stories()),
            "started": started,
            "entries": entries,
            "trades": trades,
            "probes": probes,
            "checked": checked,
            "mismatches": len(mismatches),
            "mismatches_shown": mismatches[:MISMATCHES_SHOWN],
            "fill_bar_violations": violations[:MISMATCHES_SHOWN],
            "terminal": dict(sorted(terminal.items())),
        }
    return passed, {"cells": cells, "per_story": per_story}


def determinism(
    world: LookaheadWorld, *, workers: int = DETERMINISM_WORKERS, cfg: SimConfig | None = None
) -> tuple[bool, dict[str, Any]]:
    """1 and ``workers`` workers give the same sha256 over the records, in every cell."""
    cfg = cfg or SimConfig()
    shas: dict[str, Any] = {}
    passed = True
    for hold in sorted(world.holds):
        factory = world.factory(hold)
        one, many = (
            outcomes_sha256(
                simulate_many(
                    world.stories,
                    factory,
                    world.paths[hold],
                    world.spy,
                    world.ctx,
                    cfg,
                    workers=w,
                )
            )
            for w in (1, workers)
        )
        passed = passed and one == many
        shas[f"hold{hold}"] = {"workers_1": one, f"workers_{workers}": many}
    return passed, {"workers": workers, "sha256": shas}


# ── G1: a synthetic market ────────────────────────────────────

SYNTH_DAYS: Final = 12  # sessions from SYNTH_FIRST: stories on the first eight
SYNTH_FIRST: Final = date(2016, 3, 1)
SYNTH_STORY_DAYS: Final = 8


def _sessions_from(first: date, n: int) -> list[date]:
    out = [first if is_trading_day(first) else next_trading_day(first)]
    while len(out) < n:
        out.append(next_trading_day(out[-1]))
    return out


def _ny(day: date, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm, ss), MARKET_TZ).astimezone(UTC)


def _walk(
    rng: np.random.Generator, day: date, p0: float, *, holes: bool, sigma: float = 0.002
) -> BarArrays:
    """A random walk over ``day``'s minutes: some missing, a halt-like gap, null VWAPs."""
    s = Session.of(day)
    n = int((s.close - s.open).total_seconds() // 60)
    ts = np.asarray([int(s.open.timestamp()) + 60 * k for k in range(n)], dtype=np.int64)
    keep = np.ones(n, dtype=bool)
    if holes:
        keep &= rng.random(n) > 0.05
        if rng.random() < 0.5:
            at = int(rng.integers(10, n - 20))
            keep[at : at + int(rng.integers(6, 13))] = False
        if rng.random() < 0.2:
            keep[: int(rng.integers(5, 12))] = False
    c = p0 * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))
    o = np.concatenate([[p0], c[:-1]])
    spread = np.abs(rng.normal(0.0, 0.001, n)) + 0.0002
    h = np.maximum(o, c) * (1 + spread)
    low = np.minimum(o, c) * (1 - spread)
    vw = (o + c + h + low) / 4
    vw[rng.random(n) < 0.1] = np.nan
    v = rng.integers(100, 5_000, n).astype(np.float64)
    return BarArrays(ts[keep], o[keep], h[keep], low[keep], c[keep], v[keep], vw[keep])


def _dip(bars: BarArrays, t0: int, rng: np.random.Generator) -> BarArrays:
    """The bars from epoch second ``t0`` on times a fall-and-partial-recovery profile."""
    depth = float(rng.uniform(0.02, 0.10))
    fall = float(rng.uniform(2, 40)) * 60
    back = float(rng.uniform(0.0, 1.3))
    rec = float(rng.uniform(20, 240)) * 60
    m = bars.ts.astype(np.float64) - t0
    f = np.where(
        m < 0,
        1.0,
        np.where(
            m < fall,
            1.0 - depth * m / fall,
            1.0 - depth + depth * back * np.minimum((m - fall) / rec, 1.0),
        ),
    )
    return BarArrays(bars.ts, bars.o * f, bars.h * f, bars.l * f, bars.c * f, bars.v, bars.vw * f)


@dataclass(frozen=True, slots=True)
class SynthDaily:
    open: float
    high: float
    low: float
    close: float
    volume: float


def _daily_of(b: BarArrays, volume: float) -> SynthDaily:
    return SynthDaily(float(b.o[0]), float(b.h.max()), float(b.l.min()), float(b.c[-1]), volume)


@dataclass(slots=True)
class SynthContext:
    """The synthetic market's ``ContextView``: A-factors (1 unless set), screens (halal
    unless set) and official daily bars."""

    sessions: list[date]
    adjs: dict[tuple[str, date], float] = field(default_factory=dict)
    verdicts: dict[tuple[str, date], str] = field(default_factory=dict)
    dailies: dict[tuple[str, date], SynthDaily] = field(default_factory=dict)

    def adj(self, symbol: str, day: date) -> float | None:
        return self.adjs.get((symbol, day), 1.0)

    def screen_verdict(self, symbol: str, day: date) -> str:
        return self.verdicts.get((symbol, day), "halal")

    def daily(self, symbol: str, day: date) -> SynthDaily | None:
        return self.dailies.get((symbol, day))


def _item(eid: int, symbol: str, at: datetime, itype: str) -> StoryItem:
    raw = RawItem(eid, f"synth-{eid}", "news", symbol, at, at, "", 1)
    return StoryItem(raw, at, at + NEWS_LAG, itype, frozenset())


def _pre(day: date, prev: float, spy_prev: float, rng: np.random.Generator) -> PreEvent:
    nan = math.nan
    return PreEvent(
        session=day,
        prev_session=day - timedelta(days=1),
        prev_close_s=prev,
        spy_prev_close_s=spy_prev,
        sigma=float(rng.uniform(0.006, 0.02)),
        sigma_n=60,
        beta=float(rng.uniform(0.8, 1.5)),
        atr_pct=0.02,
        adv20_usd=float(rng.uniform(2e7, 2e8)),
        ret5_vs_spy=0.0,
        ret20_vs_spy=0.0,
        hi20_s=nan,
        lo20_s=nan,
        hi252_s=nan,
        lo252_s=nan,
    )


def synthetic_world(
    seed: int = SEED, *, n_paths: int = G1_SYNTHETIC, cells: Sequence[int] = G1_CELLS
) -> LookaheadWorld:
    """``n_paths`` synthetic stories with a dip at their news, for G1's synthetic run.

    Each name trades every session of a 12-session calendar from 2016-03-01
    and has one to three stories on its first eight. A story's first item is
    an analyst downgrade (NSN, run with the family check) or a price-target
    cut (another negative type, run without it), public before the open or
    in the session; some stories carry a structural or an unclear item later,
    or noise. The price dips from the news and recovers a random share; a
    tenth of the names are ineligible, some fail the screen on S+1
    (compliance exits) and some have a 2:1 split inside their path.
    """
    rng = np.random.default_rng(seed)
    days = _sessions_from(SYNTH_FIRST, SYNTH_DAYS)
    spy: dict[date, BarArrays] = {}
    ctx = SynthContext(sessions=days)
    p_spy = 200.0
    for d in days:
        b = _walk(rng, d, p_spy, holes=False, sigma=0.0004)
        spy[d] = b
        ctx.dailies[(SPY, d)] = _daily_of(b, 1e9)
        p_spy = float(b.c[-1])
    stories: list[G1Story] = []
    context: dict[str, tuple[PreEventLike | None, EligibilityLike]] = {}
    planned: list[tuple[str, date, str, datetime, list[tuple[datetime, str]]]] = []
    eid = 0
    j = 0
    while len(planned) < n_paths:
        symbol = f"Z{j:04d}"
        j += 1
        k = int(rng.integers(1, 4))
        story_days = sorted(rng.choice(np.arange(SYNTH_STORY_DAYS), size=k, replace=False).tolist())
        for i in story_days[: n_paths - len(planned)]:
            day = days[int(i)]
            if rng.random() < 0.5:
                at = _ny(day, int(rng.integers(6, 9)), int(rng.integers(0, 60)))
            else:
                at = _ny(day, int(rng.integers(10, 14)), int(rng.integers(0, 60)))
            first = "analyst_downgrade" if rng.random() < 0.7 else "analyst_pt_cut"
            extra: list[tuple[datetime, str]] = []
            if rng.random() < 0.25:
                kind = "dilution" if rng.random() < 0.6 else "legal_adverse"
                extra.append((_ny(day, int(rng.integers(10, 15)), int(rng.integers(0, 60))), kind))
            if rng.random() < 0.2:
                extra.append(
                    (_ny(day, int(rng.integers(9, 15)), int(rng.integers(0, 60))), "noise")
                )
            planned.append((symbol, day, first, at, extra))
    by_symbol: dict[str, list[tuple[date, str, datetime, list[tuple[datetime, str]]]]] = (
        defaultdict(list)
    )
    for symbol, day, first, at, extra in planned:
        by_symbol[symbol].append((day, first, at, extra))
    bars_of: dict[tuple[str, date], BarArrays] = {}
    for symbol in sorted(by_symbol):
        news = {day: at for day, _, at, _ in by_symbol[symbol]}
        split = days[int(rng.integers(1, SYNTH_DAYS))] if rng.random() < 0.1 else None
        p = float(rng.uniform(20, 200))
        prev: dict[date, float] = {}
        for d in days:
            if d == split:
                p /= 2.0
            prev[d] = p
            b = _walk(rng, d, p, holes=True)
            if d in news and len(b):
                b = _dip(b, max(int(news[d].timestamp()), int(Session.of(d).open.timestamp())), rng)
            bars_of[(symbol, d)] = b
            if len(b):
                p = float(b.c[-1])
                ctx.dailies[(symbol, d)] = _daily_of(b, 1e6)
            if split is not None and d < split:
                ctx.adjs[(symbol, d)] = 0.5
        for day, first, at, extra in sorted(by_symbol[symbol], key=lambda x: x[0]):
            items = [(at, first), *extra]
            made: list[StoryItem] = []
            for t, itype in sorted(items, key=lambda x: x[0]):
                eid += 1
                made.append(_item(eid, symbol, t, itype))
            st = Story(f"{symbol}:{day.isoformat()}", symbol, day, made)
            nsn = st.nsn_at(Session.of(day).entry_cutoff) is not None
            negative = TYPES.get(st.type_close(), None)
            mode: Mode | None = (
                "nsn" if nsn else ("other" if negative and negative.direction == "neg" else None)
            )
            g = G1Story(st, mode)
            stories.append(g)
            spy_prev = float(spy[days[days.index(day) - 1]].c[-1]) if day != days[0] else 200.0
            pre = _pre(day, prev[day], spy_prev, rng)
            rank = int(rng.integers(0, 999))
            ok = rng.random() >= 0.1
            context[st.story_id] = (
                pre,
                LiquidityEligibility(ok, "ok" if ok else "rank", rank, study.cost_bps(rank)),
            )
            if rng.random() < 0.1:
                ctx.verdicts[(symbol, next_trading_day(day))] = "not_halal"
    paths: dict[int, dict[str, PathData | PathSkip]] = {}
    holds: dict[int, dict[str, int]] = {}
    for hold in cells:
        paths[hold] = {}
        holds[hold] = {}
        for g in stories:
            i = days.index(g.session)
            pdays = days[i : i + hold]
            spare = days[i + hold] if i + hold < len(days) else None
            paths[hold][g.story_id] = PathData(
                story_id=g.story_id,
                symbol=g.symbol,
                sessions=tuple(Session.of(d) for d in pdays),
                bars=tuple(bars_of[(g.symbol, d)] for d in pdays),
                spare=Session.of(spare) if spare is not None else None,
                spare_bars=bars_of[(g.symbol, spare)] if spare is not None else None,
            )
            holds[hold][g.story_id] = hold
    return LookaheadWorld(stories, paths, SpyData(spy), ctx, context, holds)


# ── G1: the real stories ──────────────────────────────────────


async def rebuild_stories(
    engine: AsyncEngine, symbols: Sequence[str], *, end: date, batch: int = 100
) -> list[Story]:
    """Every story of ``symbols`` built from the history start to ``end`` (as
    ``events stories build`` does: ``stories.load_items`` then ``stories.build``)."""
    from halal_trader.events.aliases import load_aliases

    aliases = await load_aliases(engine)
    out: list[Story] = []
    for i in range(0, len(symbols), batch):
        chunk = symbols[i : i + batch]
        raws: list[RawItem] = []
        async for raw in load_items(engine, start=HISTORY_FROM, end=end, symbols=chunk):
            if raws and raw.symbol != raws[0].symbol:
                out += build(raws, aliases)
                raws = []
            raws.append(raw)
        if raws:
            out += build(raws, aliases)
    return out


def g1_sessions(story: GateStory, hold: int) -> int:
    """A gate story's path length in a cell: the hold, cut at 2016-09-30 (the units
    module's ``g1_path_sessions`` when it has one, else its own ``path``)."""
    cap = getattr(units, "g1_path_sessions", None)
    if cap is not None:
        return min(hold, int(cap(story)))
    return min(hold, len(units.path(story.session, units.PATH_SESSIONS, units.G1_RANGE[1])))


async def g1_world(
    engine: AsyncEngine,
    chosen: Sequence[GateStory],
    unit_set: frozenset[Unit],
    sha: str,
    *,
    cells: Sequence[int] = G1_CELLS,
) -> LookaheadWorld:
    """The 500 gate stories rebuilt from the event store, their context and their paths.

    :class:`GateRefused` when a rebuilt story is missing or no longer agrees
    with the stored selection (NSN by its cutoff): the stories are stale.
    """
    from halal_trader.events.context import PitContext

    lo, hi = units.G1_RANGE
    symbols = sorted({g.symbol for g in chosen})
    built = {s.story_id: s for s in await rebuild_stories(engine, symbols, end=hi)}
    stale = []
    for g in chosen:
        s = built.get(g.story_id)
        if s is None or (s.nsn_at(Session.of(s.session).entry_cutoff) is not None) != g.nsn:
            stale.append(g.story_id)
    if stale:
        raise GateRefused(
            f"g1: {len(stale)} gate stories differ when rebuilt (first: {', '.join(stale[:5])}); "
            "rebuild them with `halal-trader events stories build`"
        )
    modes: dict[str, Mode] = {g.story_id: "nsn" if g.nsn else "other" for g in chosen}
    stories = [
        G1Story(s, modes.get(sid))
        for sid, s in sorted(built.items())
        if sid in modes or lo <= s.session <= hi
    ]
    pit = await PitContext.load(engine, symbols=symbols, start=lo, end=hi)
    ctx = LiquidityOnly(pit)
    context: dict[str, tuple[PreEventLike | None, EligibilityLike]] = {}
    for st in stories:
        if st.mode is None:
            continue
        at = st.at_news()
        if at is None:
            context[st.story_id] = (None, LiquidityEligibility(False, "no_at_news", None, 0.0))
        else:
            context[st.story_id] = liquidity_eligibility(pit, st.symbol, st.session, at)
    unlock = WindowUnlock(gate="g1", units=unit_set, units_sha=sha)
    spy = SpyData()
    paths: dict[int, dict[str, PathData | PathSkip]] = {}
    holds: dict[int, dict[str, int]] = {}
    for hold in cells:
        holds[hold] = {g.story_id: g1_sessions(g, hold) for g in chosen}
        loader = MinuteBarLoader(engine, window=Window.GATE, window_end=hi, unlock=unlock)
        await loader.prepare()
        requests = [
            PathRequest(g.story_id, g.symbol, g.session, holds[hold][g.story_id]) for g in chosen
        ]
        loader.check(requests)
        paths[hold] = {}
        async for item in loader.paths(requests, ctx):
            paths[hold][item.story_id] = item
        spy.days.update((await loader.spy()).days)
    return LookaheadWorld(stories, paths, spy, ctx, context, holds)


async def run_lookahead(
    engine: AsyncEngine,
    *,
    workers: int = DETERMINISM_WORKERS,
    synthetic_paths: int = G1_SYNTHETIC,
    write: bool = True,
) -> GateRun:
    """G1: ``g1-synthetic``, then ``g1-lookahead`` and ``g1-determinism`` (module docstring).

    The synthetic run needs no stored bar, so it runs (and is written) even
    when the real stories' units are not done; the other two are then refused.
    """
    out = GateRun()

    async def done(r: GateResult) -> None:
        out.results.append(r)
        if write:
            await record_gate(engine, r)

    synth = synthetic_world(n_paths=synthetic_paths)
    ok, metrics = lookahead(synth, require_trades=True)
    metrics["paths"] = synthetic_paths
    await done(GateResult("g1-synthetic", ok, metrics, {"paths": synthetic_paths}))
    chosen = await units.g1_stories(engine)
    if len(chosen) < units.G1_STORIES:
        logger.warning("g1: the selection holds %d stories, not %d", len(chosen), units.G1_STORIES)
    unit_set = frozenset(units.g1_units(chosen))
    try:
        if not unit_set:
            raise GateRefused("g1: the selection is empty (no gate_g1 story)")
        sha = await pin_units(engine, "g1", "gate_g1", unit_set)
        await require_done(engine, ["g1-lookahead", "g1-determinism"], unit_set)
        world = await g1_world(engine, chosen, unit_set, sha)
    except GateRefused as e:
        out.refuse(["g1-lookahead", "g1-determinism"], str(e))
        return out
    window = f"{units.G1_RANGE[0]}..{units.G1_RANGE[1]}"
    pinned = {"units_sha": sha, "units": len(unit_set)}
    ok, metrics = lookahead(world)
    metrics["selection"] = {"stories": len(chosen), "nsn": sum(g.nsn for g in chosen)}
    await done(GateResult("g1-lookahead", ok, metrics, pinned, window))
    ok_real, real = determinism(world, workers=workers)
    ok_synth, fake = determinism(synth, workers=workers)
    await done(
        GateResult(
            "g1-determinism",
            ok_real and ok_synth,
            {"g1": real, "synthetic": fake},
            {**pinned, "workers": workers, "paths": synthetic_paths},
            window,
        )
    )
    return out


# ── G2: the reactor ───────────────────────────────────────────


def headline_id(h: Headline) -> str:
    """A pinned headline's id: its symbol and New York day (one headline per pair)."""
    return f"{h.symbol}:{ny_day(h.published_at).isoformat()}"


def explicit_set(headlines: Sequence[Headline]) -> dict[str, tuple[str, date]]:
    """``sim.run``'s explicit set: headline id -> (symbol, New York day)."""
    out: dict[str, tuple[str, date]] = {}
    for h in headlines:
        sid = headline_id(h)
        if sid in out:
            raise ValueError(f"two pinned headlines share {sid}")
        out[sid] = (h.symbol, ny_day(h.published_at))
    return out


class FetchAttempted(RuntimeError):
    """R0's market was asked for bars: a unit of the pinned set was not stored."""


class NoFetchMarket:
    """R0's market: any fetch raises (every bar must already be stored and done)."""

    async def minute_bars_many(self, symbols: Sequence[str], **_: Any) -> dict[str, Any]:
        raise FetchAttempted(f"R0 asked the market for {', '.join(symbols[:5])}")


def _bucket(outcomes: Sequence[Outcome], label: str) -> Any:
    from halal_trader.events.intraday import summarise

    return next(
        (b for b in summarise(list(outcomes)) if b.label == label and b.horizon == "same_day"),
        None,
    )


def r0_result(total: int, headlines: Sequence[Headline], outcomes: Sequence[Outcome]) -> GateResult:
    """R0's verdict from the study's outcomes on the pinned set (module docstring)."""
    strong = _bucket(outcomes, STRONG_LABEL)
    control = _bucket(outcomes, CONTROL_LABEL)
    negative = _bucket(outcomes, NEGATIVE_LABEL)

    def pct(b: Any) -> float | None:
        return round(b.mean * 100, 2) if b is not None else None

    got: dict[str, float | None] = {
        "headlines": total,
        "strong_headlines": sum(h.score >= STRONG for h in headlines),
        "strong_n": strong.n if strong is not None else None,
        "strong_mean_pct": pct(strong),
        "strong_t": round(strong.t, 1) if strong is not None else None,
        "control_mean_pct": pct(control),
        "negative_mean_pct": pct(negative),
    }
    checks = {k: {"got": got[k], "want": v, "ok": got[k] == v} for k, v in R0_EXPECTED.items()}
    raw = {
        b_label: {"n": b.n, "mean": b.mean, "t": b.t}
        for b_label, b in (
            (STRONG_LABEL, strong),
            (CONTROL_LABEL, control),
            (NEGATIVE_LABEL, negative),
        )
        if b is not None
    }
    return GateResult(
        "r0",
        all(c["ok"] for c in checks.values()),
        {"checks": checks, "same_day": raw, "outcomes": len(outcomes), "pinned": len(headlines)},
    )


def study_returns(outcomes: Iterable[Outcome]) -> dict[str, float]:
    """The study's same-day return of each headline it kept, by headline id."""
    return {headline_id(o.headline): o.same_day for o in outcomes}


def r1_result(
    headlines: Sequence[Headline],
    summary: RunSummary,
    trades: Sequence[TradeRecord],
    study_r: Mapping[str, float],
) -> GateResult:
    """R1's verdict: identical dropped sets (``legacy.r1_dropped``) and |dr| <= 1e-10.

    ``study_r`` is the study's same-day return of each headline it kept.
    """
    ids = list(explicit_set(headlines))
    dropped = legacy.r1_dropped(ids, summary, trades, set(ids) - set(study_r))
    by_id = {t.story_id: t for t in trades}
    diffs = {sid: abs(by_id[sid].r_net_abn - study_r[sid]) for sid in dropped.kept}
    over = sorted((sid for sid, d in diffs.items() if not d <= R1_TOL), key=lambda s: -diffs[s])
    worst = max(diffs.values(), default=0.0)
    passed = dropped.passed and not over and bool(dropped.kept)
    return GateResult(
        "r1",
        passed,
        {
            "dropped": dropped.as_config(),
            "compared": len(diffs),
            "max_abs_diff": worst,
            "over_tolerance": len(over),
            "over_shown": {sid: diffs[sid] for sid in over[:MISMATCHES_SHOWN]},
            "tolerance": R1_TOL,
            "run": run_counts(summary),
        },
        {"run_sim": legacy.reactor_config().as_config(), "news_lag_s": 60},
    )


# A run summary's per-story lists (R1's dropped check reports what matters of them) and
# its run id (new each run): left out of a gate row, so a rerun writes the same numbers.
_NOT_RECORDED: Final = frozenset(
    {"run_id", "skip_ids", "bar_drop_ids", "spy_drop_ids", "spare_drop_ids", "dropped"}
)


def run_counts(summary: RunSummary) -> dict[str, object]:
    """A simulator run's counts, as a gate row records them."""
    return {k: v for k, v in summary.as_dict().items() if k not in _NOT_RECORDED}


class GateMarketFill(MarketFill):
    """R2's fills: the D.5 market rule, as a gate-only model.

    Gate-only, so that the simulator admits every headline the study took (no
    screen, no entry window) and leaves the exit to :class:`FlattenHold`; the
    bars, prices and fallbacks are the market rule's.
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        return "gate-market"

    @property
    def gate_only(self) -> bool:
        return True


GATE_MARKET_FILL: Final = GateMarketFill()
ENTRY_TAG: Final = "entry"


def realistic_config() -> SimConfig:
    """R2's simulator: SIP_RT, ORDER_LAG and the D.5 market rule (gate-only admission)."""
    return SimConfig(fill=GATE_MARKET_FILL)


class FlattenHold:
    """R2's playbook: buys at its start and sells at the deadline session's flatten
    (close - 5 min), as the simulator's own flatten would; a buy still working then
    is cancelled."""

    name = "gate-flatten-hold"
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
        return self._state in ("ENTERING", "ENTERED", "EXITING")

    def _go(self, to: str, reason: str) -> Transition:
        self._state = to
        return Transition(to, reason)

    def start(self, ctx: Ctx) -> list[Intent]:
        return [self._go("ENTERING", TRIGGERED), Submit("buy", tag=ENTRY_TAG, facts=self._facts)]

    def on(self, ev: Input, ctx: Ctx) -> list[Intent]:
        if isinstance(ev, FillIn):
            if ev.side == "buy":
                return [self._go("ENTERED", "filled")]
            return [self._go("EXITED", ev.reason), Finish(ev.reason)]
        if isinstance(ev, OrderClosedIn) and ev.side == "buy":
            return [self._go("EXPIRED", "entry_unfilled"), Finish("entry_unfilled")]
        if isinstance(ev, SessionIn) and ev.kind == "flatten" and ev.k == self._n - 1:
            if self._state == "ENTERED":
                return [self._go("EXITING", "time_stop"), Submit("sell", reason="time_stop")]
            if self._state == "ENTERING":
                return [Cancel(tag=ENTRY_TAG)]
        return []


@dataclass(frozen=True, slots=True)
class FlattenHoldFactory:
    """Builds :class:`FlattenHold` playbooks; ``facts`` by story id."""

    facts: Mapping[str, TradeFacts]
    path_sessions: int = 1
    name: str = FlattenHold.name
    version: str = FlattenHold.version

    def __call__(self, story: StoryView) -> FlattenHold:
        return FlattenHold(self.path_sessions, self.facts[story.story_id])


def _r(entry: float, exit_: float, spy_entry: float, spy_exit: float, cost_bps: float) -> float:
    return (exit_ / entry - 1.0) - (spy_exit / spy_entry - 1.0) - 2.0 * cost_bps / 1e4


def decomposition(legacy_trade: TradeRecord, real: TradeRecord) -> tuple[float, float]:
    """(entry rule Δ, exit rule Δ) of one headline, on raw prices (one session: A cancels).

    entry Δ = r(realistic entry, legacy exit) - r(legacy); exit Δ = r(realistic)
    - r(realistic entry, legacy exit). They sum to r(realistic) - r(legacy).
    """
    c = legacy_trade.cost_bps
    old = _r(
        legacy_trade.entry_px,
        legacy_trade.exit_px,
        legacy_trade.spy_entry_px,
        legacy_trade.spy_exit_px,
        c,
    )
    mixed = _r(real.entry_px, legacy_trade.exit_px, real.spy_entry_px, legacy_trade.spy_exit_px, c)
    new = _r(real.entry_px, real.exit_px, real.spy_entry_px, real.spy_exit_px, c)
    return mixed - old, new - mixed


def r2_result(
    headlines: Sequence[Headline],
    study_r: Mapping[str, float],
    r1_trades: Sequence[TradeRecord],
    r1_aside: Iterable[str],
    summary: RunSummary,
    trades: Sequence[TradeRecord],
) -> GateResult:
    """R2's verdict: the strong group's realistic mean inside the legacy 95% CI clustered by
    New York date, with the clustered t of both and each headline's decomposition."""
    score = {headline_id(h): h.score for h in headlines}
    day = {headline_id(h): ny_day(h.published_at) for h in headlines}
    strong = {sid for sid, s in score.items() if s >= STRONG}
    old_ids = sorted(sid for sid in study_r if sid in strong)
    old = stats.clustered_mean([study_r[s] for s in old_ids], [day[s] for s in old_ids])
    aside = set(r1_aside)
    real = {
        t.story_id: t
        for t in trades
        if t.story_id in strong
        and t.story_id not in aside
        and math.isfinite(t.r_net_abn)
        and legacy.reactor_plausible(t)
    }
    new_ids = sorted(real)
    new = stats.clustered_mean([real[s].r_net_abn for s in new_ids], [day[s] for s in new_ids])
    legacy_by_id = {t.story_id: t for t in r1_trades}
    parts = {
        sid: decomposition(legacy_by_id[sid], real[sid])
        for sid in new_ids
        if sid in legacy_by_id and sid in study_r
    }
    metrics: dict[str, Any] = {
        "legacy": _clustered(old),
        "realistic": _clustered(new),
        "excluded": {
            "set_aside": len(strong & aside),
            "no_trade": len(strong - aside - {t.story_id for t in trades}),
        },
    }
    passed = False
    if old is not None and new is not None:
        lo, hi = old.ci(CI_LEVEL)
        passed = lo <= new.mean <= hi
        metrics["legacy_ci95"] = [lo, hi]
    if parts:
        e = np.asarray([p[0] for p in parts.values()])
        x = np.asarray([p[1] for p in parts.values()])
        metrics["decomposition"] = {
            "n": len(parts),
            "entry_mean": float(e.mean()),
            "exit_mean": float(x.mean()),
            "entry_median": float(np.median(e)),
            "exit_median": float(np.median(x)),
            "per_headline": {sid: [p[0], p[1]] for sid, p in sorted(parts.items())},
        }
    return GateResult(
        "r2",
        passed,
        metrics,
        {"run_sim": realistic_config().as_config(), "news_lag_s": 60, "flatten_min": 5},
    )


def _clustered(m: stats.ClusteredMean | None) -> dict[str, Any] | None:
    if m is None:
        return None
    return {"n": m.n, "clusters": m.clusters, "mean": m.mean, "se": m.se, "t": m.t, "df": m.df}


async def _reactor_inputs(
    engine: AsyncEngine, headlines: Sequence[Headline]
) -> tuple[PitContext, dict[str, TradeFacts]]:
    from halal_trader.events.context import PitContext

    days = sorted({ny_day(h.published_at) for h in headlines})
    sessions = [d for d in days if is_trading_day(d)]
    ranks = units.LiquidityRanks(engine)
    await ranks.load(days)
    facts = {}
    for h in headlines:
        rank = ranks.rank(h.symbol, ny_day(h.published_at))
        facts[headline_id(h)] = TradeFacts(
            family_type="reactor",
            cell="gate",
            variant="legacy",
            cost_bps=study.cost_bps(rank),
            rank=rank if rank is not None else -1,
            tech=False,
        )
    ctx = await PitContext.load(
        engine, symbols=sorted({h.symbol for h in headlines}), start=sessions[0], end=sessions[-1]
    )
    return ctx, facts


async def run_reactor(engine: AsyncEngine, *, workers: int = 1, write: bool = True) -> GateRun:
    """G2: ``r0``, ``r1`` and ``r2`` on the pinned headline set (module docstring); all
    three are refused while a unit of the set (SPY's included) is not done."""
    from halal_trader.events import intraday

    out = GateRun()

    async def done(r: GateResult) -> None:
        out.results.append(r)
        if write:
            await record_gate(engine, r)

    headlines = await units.reactor_headlines(engine)
    unit_set = frozenset(units.reactor_units(headlines))
    try:
        if not unit_set:
            raise GateRefused("reactor: the pinned headline set has no session")
        sha = await pin_units(engine, "reactor", "gate_reactor", unit_set)
        await require_done(engine, list(REACTOR_IDS), unit_set)
    except GateRefused as e:
        out.refuse(REACTOR_IDS, str(e))
        return out
    pinned = {"units_sha": sha, "units": len(unit_set), "headlines": len(headlines)}
    window = f"{min(d for _, d in unit_set)}..{max(d for _, d in unit_set)}"
    total = len(await intraday.first_in_session(engine, scored_before=units.REACTOR_SCORED_BEFORE))
    try:
        outcomes = await intraday.run(engine, NoFetchMarket(), list(headlines))
    except FetchAttempted as e:
        await done(GateResult("r0", False, {"error": str(e)}, pinned, window))
        outcomes = []
    else:
        r0 = r0_result(total, headlines, outcomes)
        await done(replace(r0, config=pinned, window=window))
    study_r = study_returns(outcomes)
    ctx, facts = await _reactor_inputs(engine, headlines)
    expected = explicit_set(headlines)
    stories = [legacy.reactor_story(headline_id(h), h.symbol, h.published_at) for h in headlines]
    unlock = WindowUnlock(gate="reactor", units=unit_set, units_sha=sha)
    one = MemorySink()
    summary = await run(
        engine,
        stories,
        legacy.HoldFactory(1, facts),
        context=ctx,
        window=Window.GATE,
        window_end=REACTOR_END,
        cfg=legacy.reactor_config(),
        unlock=unlock,
        sink=one,
        workers=workers,
        expected=expected,
    )
    r1_trades = [o.trade for o in one.outcomes if o.trade is not None]
    r1 = r1_result(headlines, summary, r1_trades, study_r)
    await done(replace(r1, config={**pinned, **r1.config}, window=window))
    two = MemorySink()
    summary2 = await run(
        engine,
        stories,
        FlattenHoldFactory(facts),
        context=ctx,
        window=Window.GATE,
        window_end=REACTOR_END,
        cfg=realistic_config(),
        unlock=unlock,
        sink=two,
        workers=workers,
        expected=expected,
    )
    r2 = r2_result(
        headlines,
        study_r,
        r1_trades,
        legacy.r1_set_aside(summary),
        summary2,
        [o.trade for o in two.outcomes if o.trade is not None],
    )
    r2.metrics["run"] = run_counts(summary2)
    await done(replace(r2, config={**pinned, **r2.config}, window=window))
    return out


# ── G3: SUE ───────────────────────────────────────────────────

FloatArray = NDArray[np.float64]


def decile_spread(signal: FloatArray, rets: FloatArray) -> float:
    """D10 - D1 of ``rets`` by deciles of ``signal`` as ``study.summarise`` cuts them."""
    n = len(signal)
    if n < 10:
        return math.nan
    ranks = signal.argsort().argsort()
    deciles = np.minimum(ranks * 10 // n, 9)
    top, bot = rets[deciles == 9], rets[deciles == 0]
    if not len(top) or not len(bot):
        return math.nan
    return float(top.mean() - bot.mean())


def rank_ic(signal: FloatArray, rets: FloatArray) -> float:
    return information_coefficient(signal, rets)


@dataclass(frozen=True, slots=True)
class Spread:
    """D10 - D1 and the rank IC of one horizon, with 95% cluster-bootstrap intervals."""

    n: int
    clusters: int
    d10_d1: float
    d10_d1_ci: tuple[float, float]
    ic: float
    ic_ci: tuple[float, float]
    dropped: int  # resamples on which a statistic was undefined

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "clusters": self.clusters,
            "d10_d1": self.d10_d1,
            "d10_d1_ci95": list(self.d10_d1_ci),
            "ic": self.ic,
            "ic_ci95": list(self.ic_ci),
            "dropped_draws": self.dropped,
        }


def spread(
    signal: Sequence[float] | FloatArray,
    rets: Sequence[float] | FloatArray,
    clusters: Sequence[Any],
    *,
    b: int = stats.BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> Spread:
    """D10 - D1 and IC with their 95% intervals over ``b`` resamples of whole clusters."""
    s = np.asarray(signal, dtype=np.float64)
    r = np.asarray(rets, dtype=np.float64)

    def d(idx: Sequence[int]) -> float:
        i = np.asarray(idx, dtype=np.int64)
        return decile_spread(s[i], r[i])

    def c(idx: Sequence[int]) -> float:
        i = np.asarray(idx, dtype=np.int64)
        return rank_ic(s[i], r[i])

    d_ci = stats.cluster_bootstrap_ci(d, clusters, b=b, seed=seed, level=CI_LEVEL)
    c_ci = stats.cluster_bootstrap_ci(c, clusters, b=b, seed=seed, level=CI_LEVEL)
    return Spread(
        n=len(s),
        clusters=len(set(clusters)),
        d10_d1=decile_spread(s, r),
        d10_d1_ci=(d_ci.lo, d_ci.hi),
        ic=rank_ic(s, r),
        ic_ci=(c_ci.lo, c_ci.hi),
        dropped=d_ci.dropped + c_ci.dropped,
    )


def _key(o: Observation) -> tuple[str, datetime, float]:
    return (o.symbol, o.published_at, o.signal)


def observations_of(events_: Iterable[SueEvent]) -> list[Observation]:
    return [Observation(e.symbol, e.published_at, e.signal) for e in events_]


def reference_table(result: study.StudyResult) -> dict[str, Any]:
    """S0's table: per bucket, n, IC and D10 - D1 at every horizon (ICs to 3 dp as recorded)."""
    out: dict[str, Any] = {}
    for group in sorted(result.n):
        rows = {(r.horizon, r.decile): r for r in result.rows if r.group == group}
        horizons = sorted({h for g, h in result.ic if g == group})
        out[group] = {
            "n": result.n[group],
            "ic": {str(h): round(result.ic[(group, h)], 3) for h in horizons},
            "d10_d1": {
                str(h): rows[(h, 10)].mean - rows[(h, 1)].mean
                for h in horizons
                if (h, 10) in rows and (h, 1) in rows
            },
        }
    return out


def reference_deltas(table: Mapping[str, Any]) -> dict[str, Any]:
    """The table against the plan's recorded numbers (reported, not judged)."""
    lo, hi = S0_RECORDED["ic_range_5_20d"]
    ics: dict[str, Any] = {}
    for group, row in table.items():
        for h in ("5", "20"):
            ic = row["ic"].get(h)
            if ic is None:
                continue
            ics[f"{group} {h}d"] = {
                "ic": ic,
                "in_range": lo <= ic <= hi,
                "outside_by": 0.0 if lo <= ic <= hi else (lo - ic if ic < lo else ic - hi),
            }
    mid = table.get(MID, {}).get("d10_d1", {}).get("20")
    want = S0_RECORDED["mid_20d_d10_d1"]
    return {
        "recorded": {"ic_range_5_20d": [lo, hi], "mid_20d_d10_d1": want},
        "ic": ics,
        "mid_20d_d10_d1": {
            "got": mid,
            "delta": mid - want if mid is not None else None,
            "same_sign": mid is not None and (mid > 0) == (want > 0),
        },
    }


def daily_spreads(
    complement: Sequence[SueEvent],
    evaluated: Sequence[tuple[Observation, int, float]],
    *,
    b: int = stats.BOOTSTRAP_DRAWS,
) -> dict[int, Spread]:
    """Σ_c in daily mode (``study.evaluate``'s returns): D10 - D1 and IC at 5 and 20 days,
    clustered by entry session."""
    session = {_key(o): e.session for e, o in zip(complement, observations_of(complement))}
    out: dict[int, Spread] = {}
    for h in SUE_H:
        rows = [(o.signal, r, session[_key(o)]) for o, hh, r in evaluated if hh == h]
        out[h] = spread([x[0] for x in rows], [x[1] for x in rows], [x[2] for x in rows], b=b)
    return out


def s0_result(
    table: Mapping[str, Any], sigma_c: Mapping[int, Spread] | None, error: str | None = None
) -> GateResult:
    """S0 passes when the reference table and the Σ_c statistics compute."""
    complete = (
        bool(table)
        and all({"5", "20"} <= set(row["ic"]) for row in table.values())
        and "20" in table.get(MID, {}).get("d10_d1", {})
    )
    finite = sigma_c is not None and all(
        math.isfinite(s.d10_d1) and math.isfinite(s.ic) for s in sigma_c.values()
    )
    metrics: dict[str, Any] = {
        "table": dict(table),
        "deltas": reference_deltas(table) if table else None,
        "sigma_c_daily": {str(h): s.as_dict() for h, s in (sigma_c or {}).items()},
        "reference": "recomputed (the 8-K times were corrected from EDGAR headers)",
    }
    if error is not None:
        metrics["error"] = error
    return GateResult("s0", complete and finite and error is None, metrics, {"years": S0_YEARS})


def _plausible(trade: TradeRecord) -> bool:
    lo, hi = study._PLAUSIBLE  # read, not copied: the study's own filter
    return lo < trade.exit_px / trade.entry_px < hi


def daily_mode(
    src: legacy.DailyBarSource,
    observations: Sequence[Observation],
    rank_of: Mapping[tuple[str, datetime], int | None],
    horizons: Sequence[int] = study.HORIZONS,
) -> tuple[list[tuple[Observation, int, float]], int, Counter[str]]:
    """``DailyBarSource`` through the simulator on ``observations``, in ``study.evaluate``'s
    order and tags; returns the outcomes, the look-ahead entries left out, and skips."""
    out: list[tuple[Observation, int, float]] = []
    lookahead = 0
    skips: Counter[str] = Counter()
    for n, obs in enumerate(observations):
        entry = src.entry(obs.published_at)
        if entry is None:
            continue
        if entry.lookahead:
            lookahead += 1
            continue
        rank = rank_of[(obs.symbol, obs.published_at)]
        tagged = Observation(
            obs.symbol,
            obs.published_at,
            obs.signal,
            {**obs.tags, "year": obs.published_at.year, "bucket": study.bucket_of(rank)},
        )
        facts = TradeFacts(
            "sue",
            "gate",
            "daily",
            cost_bps=study.cost_bps(rank),
            rank=rank if rank is not None else -1,
            tech=False,
        )
        for h in horizons:
            got = src.simulate(
                entry, story_id=f"s1:{n}:{h}", symbol=obs.symbol, horizon=h, facts=facts
            )
            if isinstance(got, PathSkip):
                skips[got.reason] += 1
                continue
            t = got.trade
            if t is None:
                skips[got.reason or got.terminal_state] += 1
                continue
            if not _plausible(t):
                skips["implausible"] += 1
                continue
            out.append((tagged, h, t.r_net_abn))
    return out, lookahead, skips


def s1_result(
    want: Sequence[tuple[Observation, int, float]],
    got: Sequence[tuple[Observation, int, float]],
    *,
    lookahead: int,
    early_close: int,
    excluded: Iterable[tuple[str, datetime, float]] = (),
    skips: Mapping[str, int] | None = None,
) -> GateResult:
    """S1: the same (observation, horizon) multiset, |dr| <= 1e-10 on each, identical decile
    rows (n, and means to 1e-10) for all events and by bucket."""
    gone = set(excluded)
    want = [w for w in want if _key(w[0]) not in gone]
    got = [g for g in got if _key(g[0]) not in gone]
    a: dict[tuple[str, datetime, float, int], list[float]] = defaultdict(list)
    b: dict[tuple[str, datetime, float, int], list[float]] = defaultdict(list)
    for o, h, r in want:
        a[(*_key(o), h)].append(r)
    for o, h, r in got:
        b[(*_key(o), h)].append(r)
    only_study = sorted(k for k in a if len(a[k]) != len(b.get(k, [])))
    only_sim = sorted(k for k in b if k not in a)
    worst = 0.0
    over = 0
    for k, rs in a.items():
        if k not in b or len(b[k]) != len(rs):
            continue
        for x, y in zip(sorted(rs), sorted(b[k])):
            d = abs(x - y)
            worst = max(worst, d)
            over += not d <= S1_TOL
    rows_ok = True
    row_diffs: dict[str, Any] = {}
    for by in (None, "bucket"):
        sa, sb = study.summarise(want, by=by), study.summarise(got, by=by)
        ra = {(r.group, r.horizon, r.decile): r for r in sa.rows}
        rb = {(r.group, r.horizon, r.decile): r for r in sb.rows}
        same = set(ra) == set(rb) and all(
            ra[k].n == rb[k].n
            and abs(ra[k].mean - rb[k].mean) <= S1_TOL
            and abs(ra[k].t - rb[k].t) <= S1_T_REL * max(1.0, abs(ra[k].t))
            for k in ra
            if k in rb
        )
        rows_ok = rows_ok and same and sa.n == sb.n
        row_diffs[by or "all"] = {
            "rows": len(ra),
            "identical": same and sa.n == sb.n,
            "n_study": dict(sa.n),
            "n_sim": dict(sb.n),
        }
    passed = not only_study and not only_sim and over == 0 and rows_ok and bool(want)
    return GateResult(
        "s1",
        passed,
        {
            "outcomes_study": len(want),
            "outcomes_sim": len(got),
            "only_study": len(only_study),
            "only_sim": len(only_sim),
            "only_shown": [list(k) for k in (only_study + only_sim)[:MISMATCHES_SHOWN]],
            "max_abs_diff": worst,
            "over_tolerance": over,
            "tolerance": S1_TOL,
            "deciles": row_diffs,
            "early_close_excluded": early_close + lookahead,
            "sim_skips": dict(sorted((skips or {}).items())),
        },
        {"horizons": list(study.HORIZONS), "run_sim": legacy.daily_config().as_config()},
    )


@dataclass(frozen=True, slots=True)
class MinuteReturns:
    """What the minute-mode computations read: bars, A-factors, official closes, costs."""

    bars: Mapping[Unit, BarArrays]
    adj: Any  # (symbol, day) -> A or None: ContextView.adj
    daily: Any  # (symbol, day) -> DailyPointLike or None: ContextView.daily


def _a(view: MinuteReturns, symbol: str, day: date) -> float | None:
    a = view.adj(symbol, day)
    return float(a) if a is not None else None


def study_clock_return(
    view: MinuteReturns,
    symbol: str,
    entry_day: date,
    at: Literal["open", "close"],
    exit_day: date,
    cost_bps: float,
) -> tuple[float | None, str, bool]:
    """Minute mode on the study's clock: entry at the open of the session's first bar (an
    open entry) or the close of its last bar, exit at the exit session's last close; SPY
    the same; A-ratios. Returns (net abnormal return or None, why not, late first bar)."""
    legs: list[float] = []
    late = False
    for name in (symbol, SPY):
        e, x = view.bars.get((name, entry_day)), view.bars.get((name, exit_day))
        if e is None or x is None or not len(e) or not len(x):
            return None, "no_bars" if name == symbol else "no_spy", late
        a_e, a_x = _a(view, name, entry_day), _a(view, name, exit_day)
        if a_e is None or a_x is None:
            return None, "no_adj", late
        if at == "open":
            p0 = float(e.o[0])
            late = late or int(e.ts[0]) > int(Session.of(entry_day).open.timestamp())
        else:
            p0 = float(e.c[-1])
        legs.append((float(x.c[-1]) * a_x) / (p0 * a_e) - 1.0)
    return legs[0] - legs[1] - 2.0 * cost_bps / 1e4, "", late


def _series(arrays: BarArrays) -> BarSeries:
    return BarSeries.build([arrays], [1.0], 0)


def realistic_return(
    view: MinuteReturns,
    symbol: str,
    published: datetime,
    entry_day: date,
    exit_day: date,
    cost_bps: float,
) -> tuple[float | None, str]:
    """S3: the D.5 market rule at the first bar with ts >= max(published + 600 s, 09:30) of
    the entry session; the flatten rule (a market sell at close - 5 min + ORDER_LAG, the
    official close as its fallback, else the last trade) on the exit session; SPY at the
    same bars. Returns (net abnormal return or None, why not)."""
    gap_us = span_us(SimConfig().gap)
    lag_us = span_us(ORDER_LAG)
    s_in, s_out = Session.of(entry_day), Session.of(exit_day)
    stock_in = view.bars.get((symbol, entry_day))
    stock_out = view.bars.get((symbol, exit_day))
    spy_in = view.bars.get((SPY, entry_day))
    spy_out = view.bars.get((SPY, exit_day))
    if stock_in is None or stock_out is None or spy_in is None or spy_out is None:
        return None, "no_bars"
    b_in, b_out = _series(stock_in), _series(stock_out)
    q_in, q_out = _series(spy_in), _series(spy_out)
    open_in = to_us(s_in.open)
    active = max(to_us(published + S3_NEWS_LAG), open_in)
    i = first_eligible(b_in, active, 0)
    if i is None:
        return None, "entry_unfilled"
    entry, flags = market_fill(b_in, i, active, gap_us=gap_us, session_open_us=open_in)
    spy_entry, _ = spy_fill(q_in, int(b_in.ts[i]), 0, use_open="gap_fill" in flags)
    sell = to_us(s_out.flatten) + lag_us
    j = first_eligible(b_out, sell, 0)
    if j is not None:
        exit_, xflags = market_fill(
            b_out, j, sell, gap_us=gap_us, session_open_us=to_us(s_out.open)
        )
        spy_exit, _ = spy_fill(q_out, int(b_out.ts[j]), 0, use_open="gap_fill" in xflags)
    else:
        dp = view.daily(symbol, exit_day)
        spy_dp = view.daily(SPY, exit_day)
        if dp is None and not len(b_out):
            return None, "no_exit"
        spy_close = (
            float(spy_dp.close)
            if spy_dp is not None
            else (float(q_out.c[-1]) if len(q_out) else math.nan)
        )
        fb: FallbackFill = unfilled_exit(
            official_close=float(dp.close) if dp is not None else None,
            spy_close=spy_close,
            spare=None,
            spy_spare=None,
            spare_open_us=None,
            lag_us=lag_us,
            gap_us=gap_us,
            last_close=float(b_out.c[-1]) if len(b_out) else math.nan,
            spy_last=(float(q_out.c[-1]) if len(q_out) else math.nan, ()),
        )
        exit_, spy_exit = fb.price, fb.spy_price
    a_e, a_x = _a(view, symbol, entry_day), _a(view, symbol, exit_day)
    b_e, b_x = _a(view, SPY, entry_day), _a(view, SPY, exit_day)
    if a_e is None or a_x is None or b_e is None or b_x is None:
        return None, "no_adj"
    r = (exit_ * a_x) / (entry * a_e) - 1.0
    q = (spy_exit * b_x) / (spy_entry * b_e) - 1.0
    out = r - q - 2.0 * cost_bps / 1e4
    return (out, "") if math.isfinite(out) else (None, "no_spy")


def calibration(
    view: MinuteReturns, bars: study.Bars, pairs: Sequence[Unit]
) -> tuple[list[float], Counter[str]]:
    """d = minute - daily abnormal return of each calibration pair: a 09:30 entry (the open
    of the session's first bar; the adjusted daily open) and an h = 5 close exit."""
    out: list[float] = []
    counts: Counter[str] = Counter()
    for symbol, day in pairs:
        exit_day = units.path(day, units.CALIB_HORIZON, units.CALIB_CAP)[-1]
        minute, why, late = study_clock_return(view, symbol, day, "open", exit_day, 0.0)
        counts["late_first_bar"] += late
        i = bars.sessions.index(day) if day in bars.sessions else None
        daily = (
            study.outcome(bars, symbol, (i, "open"), units.CALIB_HORIZON, 0.0)
            if i is not None
            else None
        )
        if minute is None:
            counts[why] += 1
            continue
        if daily is None:
            counts["no_daily"] += 1
            continue
        out.append(minute - daily)
    return out, counts


def calib_result(diffs: Sequence[float], pairs: int, counts: Mapping[str, int]) -> GateResult:
    """s1-calib: p99 |d| recorded; passes when it computes on at least MIN_COVERAGE of pairs."""
    coverage = len(diffs) / pairs if pairs else 0.0
    p99 = float(np.quantile(np.abs(diffs), 0.99)) if diffs else math.nan
    return GateResult(
        "s1-calib",
        bool(diffs) and coverage >= MIN_COVERAGE and math.isfinite(p99),
        {
            "pairs": pairs,
            "n": len(diffs),
            "coverage": coverage,
            "p99_cal": p99,
            "median_abs": float(np.median(np.abs(diffs))) if diffs else None,
            "counts": dict(sorted(counts.items())),
        },
        {"horizon": units.CALIB_HORIZON, "min_coverage": MIN_COVERAGE},
    )


@dataclass(frozen=True, slots=True)
class Paired:
    """One Σ_s event at one horizon: its signal, cluster, daily and minute-mode returns."""

    key: tuple[str, datetime, float]
    h: int
    signal: float
    cluster: date
    daily: float
    minute: float


def s2_result(
    pairs: Sequence[Paired], events_: int, p99_cal: float, counts: Mapping[str, Any]
) -> GateResult:
    """S2: at each horizon the paired d passes TOST (±0.10%, α 0.05, clustered), median |d|
    <= 0.10% and p99 |d| <= max(1.00%, 1.5 p99_cal), on at least MIN_COVERAGE of Σ_s."""
    limit = max(P99_FLOOR, P99_CAL_X * p99_cal)
    per: dict[str, Any] = {}
    passed = True
    for h in SUE_H:
        rows = [p for p in pairs if p.h == h]
        d = [p.minute - p.daily for p in rows]
        cl = [p.cluster for p in rows]
        coverage = len(rows) / events_ if events_ else 0.0
        if len(set(cl)) < 2:
            per[str(h)] = {"n": len(rows), "coverage": coverage, "passed": False}
            passed = False
            continue
        abs_d = np.abs(np.asarray(d))
        median, p99 = float(np.median(abs_d)), float(np.quantile(abs_d, 0.99))
        tost = stats.tost(d, cl, margin=TOST_MARGIN, alpha=TOST_ALPHA)
        fit = stats.clustered_mean(d, cl)
        ok = tost and median <= MEDIAN_MAX and p99 <= limit and coverage >= MIN_COVERAGE
        passed = passed and ok
        per[str(h)] = {
            "n": len(rows),
            "clusters": len(set(cl)),
            "coverage": coverage,
            "mean_d": float(np.mean(d)),
            "ci90": list(fit.ci(1 - 2 * TOST_ALPHA)) if fit is not None else None,
            "tost": tost,
            "median_abs_d": median,
            "p99_abs_d": p99,
            "p99_limit": limit,
            "passed": ok,
        }
    return GateResult(
        "s2",
        passed,
        {"horizons": per, "events": events_, "p99_cal": p99_cal, "counts": dict(counts)},
        {
            "margin": TOST_MARGIN,
            "alpha": TOST_ALPHA,
            "median_max": MEDIAN_MAX,
            "p99_floor": P99_FLOOR,
            "p99_cal_x": P99_CAL_X,
            "min_coverage": MIN_COVERAGE,
        },
    )


def s3_result(
    pairs: Sequence[Paired],
    events_: int,
    counts: Mapping[str, Any],
    *,
    b: int = stats.BOOTSTRAP_DRAWS,
) -> GateResult:
    """S3: at h = 5 and 20, the realistic D10 - D1 and IC (``Paired.minute`` holds the
    realistic return here) inside the daily-mode 95% CI on the same events, same sign."""
    per: dict[str, Any] = {}
    passed = True
    for h in SUE_H:
        rows = [p for p in pairs if p.h == h]
        coverage = len(rows) / events_ if events_ else 0.0
        if len({p.cluster for p in rows}) < 2 or len(rows) < 10:
            per[str(h)] = {"n": len(rows), "coverage": coverage, "passed": False}
            passed = False
            continue
        sig = np.asarray([p.signal for p in rows])
        daily = spread(sig, [p.daily for p in rows], [p.cluster for p in rows], b=b)
        real = np.asarray([p.minute for p in rows])
        real_d, real_ic = decile_spread(sig, real), rank_ic(sig, real)
        d_in = daily.d10_d1_ci[0] <= real_d <= daily.d10_d1_ci[1]
        ic_in = daily.ic_ci[0] <= real_ic <= daily.ic_ci[1]
        d_sign = np.sign(real_d) == np.sign(daily.d10_d1)
        ic_sign = np.sign(real_ic) == np.sign(daily.ic)
        ok = bool(d_in and ic_in and d_sign and ic_sign and coverage >= MIN_COVERAGE)
        passed = passed and ok
        per[str(h)] = {
            "coverage": coverage,
            "daily": daily.as_dict(),
            "realistic": {"d10_d1": real_d, "ic": real_ic},
            "inside": {"d10_d1": bool(d_in), "ic": bool(ic_in)},
            "same_sign": {"d10_d1": bool(d_sign), "ic": bool(ic_sign)},
            "passed": ok,
        }
    return GateResult(
        "s3",
        passed,
        {"horizons": per, "events": events_, "counts": dict(counts)},
        {"news_lag_s": int(S3_NEWS_LAG.total_seconds()), "min_coverage": MIN_COVERAGE},
    )


async def _ranks(
    engine: AsyncEngine, obs: Sequence[Observation]
) -> dict[tuple[str, datetime], int | None]:
    """Each observation's rank in its publication month's universe (``study.evaluate``'s)."""
    ranks = units.LiquidityRanks(engine)
    await ranks.load(ny_day(o.published_at) for o in obs)
    return {(o.symbol, o.published_at): ranks.rank(o.symbol, ny_day(o.published_at)) for o in obs}


async def run_sue(
    engine: AsyncEngine, *, write: bool = True, b: int = stats.BOOTSTRAP_DRAWS
) -> GateRun:
    """G3: ``s0``, ``s1``, ``s1-calib``, ``s2`` and ``s3`` (module docstring).

    S0 and S1 read daily bars only and always run. ``s1-calib`` is refused
    while a calibration unit is not done; S2 and S3 while a ``gate_sue`` unit
    is not done; S2 also needs this run's ``s1-calib`` to have passed (its
    p99_cal is S2's bound).
    """
    observations = await units.load_sue_observations(engine)
    counts: Counter[str] = Counter()
    complement = await units.sue_complement(engine, observations=observations, counts=counts)
    obs_c = observations_of(complement)
    out = GateRun()

    async def done(r: GateResult) -> None:
        out.results.append(r)
        if write:
            await record_gate(engine, r)

    # S0: the recorded table, recomputed, then Σ_c in daily mode.
    lo_y, hi_y = S0_YEARS
    years = [o for o in observations if lo_y <= o.published_at.year <= hi_y]
    table = reference_table(study.summarise(await study.evaluate(engine, years), by="bucket"))
    evaluated = await study.evaluate(engine, obs_c)
    try:
        sigma_c: dict[int, Spread] | None = daily_spreads(complement, evaluated, b=b)
        error = None
    except ValueError as exc:
        sigma_c, error = None, str(exc)
    s0 = s0_result(table, sigma_c, error)
    s0.metrics["sigma_c"] = {"events": len(complement), "counts": dict(sorted(counts.items()))}
    await done(s0)

    # S1: DailyBarSource through the simulator, against study.evaluate on Σ_c.
    src = await legacy.DailyBarSource.load(engine, sorted({o.symbol for o in obs_c}))
    rank_of = await _ranks(engine, obs_c)
    got, lookahead_n, skips = daily_mode(src, obs_c, rank_of)
    excluded = [
        _key(o) for o in obs_c if (ent := src.entry(o.published_at)) is not None and ent.lookahead
    ]
    await done(
        s1_result(
            evaluated,
            got,
            lookahead=lookahead_n,
            early_close=counts.get("early_close", 0),
            excluded=excluded,
            skips=skips,
        )
    )

    # s1-calib: minute minus daily on the calibration pairs.
    p99_cal: float | None = None
    pairs = await units.calib_pairs(engine, observations=observations)
    try:
        cal = await _calibrate(engine, pairs)
    except GateRefused as e:
        out.refuse(["s1-calib"], str(e))
    else:
        await done(cal)
        p99_cal = float(cal.metrics["p99_cal"]) if cal.passed else None

    # S2 and S3 on Σ_s.
    sample = units.sue_sample(complement)
    try:
        clock, real, c2, c3, pinned = await _sue_minutes(engine, sample)
    except GateRefused as e:
        out.refuse(["s2", "s3"], str(e))
        return out
    if p99_cal is None:
        out.refuse(["s2"], "s2 needs this run's s1-calib to pass (p99_cal is its bound)")
    else:
        s2 = s2_result(clock, len(sample), p99_cal, c2)
        await done(replace(s2, config={**s2.config, **pinned}))
    s3 = s3_result(real, len(sample), c3, b=b)
    await done(replace(s3, config={**s3.config, **pinned}))
    return out


async def _calibrate(engine: AsyncEngine, pairs: Sequence[Unit]) -> GateResult:
    """``s1-calib`` on the calibration pairs (pinned, done-checked, read behind the guard)."""
    from halal_trader.events.context import PitContext

    calib = frozenset(units.calib_units(pairs))
    if not calib:
        raise GateRefused("s1-calib: no calibration pair")
    sha = await pin_units(engine, "calib", "gate_calib", calib)
    await require_done(engine, ["s1-calib"], calib)
    bars, cut = await read_gate_bars(engine, "calib", calib, sha, units.CALIB_CAP)
    names = sorted({s for s, _ in pairs})
    pit = await PitContext.load(engine, symbols=names, start=units.G1_RANGE[0], end=units.CALIB_CAP)
    view = MinuteReturns(bars, pit.adj, pit.daily)
    diffs, counts = calibration(view, await study.load_bars(engine, names), pairs)
    counts["bars_cut"] += cut
    cal = calib_result(diffs, len(pairs), counts)
    return replace(cal, config={**cal.config, "units_sha": sha, "units": len(calib)})


async def _sue_minutes(
    engine: AsyncEngine, sample: Sequence[SueEvent]
) -> tuple[list[Paired], list[Paired], Counter[str], Counter[str], dict[str, Any]]:
    """Σ_s paired with the study's daily returns: on the study's clock (S2) and with
    realistic fills (S3); the counts of what each left out; the pin."""
    from halal_trader.events.context import PitContext

    sue_units = frozenset(units.sue_units(sample))
    if not sue_units:
        raise GateRefused("s2/s3: Σ_s is empty")
    sha = await pin_units(engine, "sue", "gate_sue", sue_units)
    await require_done(engine, ["s2", "s3"], sue_units)
    window_end = max(d for _, d in sue_units)
    bars, cut = await read_gate_bars(engine, "sue", sue_units, sha, window_end)
    names = sorted({e.symbol for e in sample})
    pit = await PitContext.load(engine, symbols=names, start=units.SUE_RANGE[0], end=window_end)
    daily_bars = await study.load_bars(engine, names)
    view = MinuteReturns(bars, pit.adj, pit.daily)
    obs_s = observations_of(sample)
    rank_s = await _ranks(engine, obs_s)
    clock: list[Paired] = []
    real: list[Paired] = []
    c2: Counter[str] = Counter(bars_cut=cut)
    c3: Counter[str] = Counter(bars_cut=cut)
    for e, o in zip(sample, obs_s):
        cost = study.cost_bps(rank_s[(o.symbol, o.published_at)])
        point = study.entry_point(o.published_at, daily_bars.sessions)
        if point is None or daily_bars.sessions[point[0]] != e.session or point[1] != e.entry:
            c2["clock_mismatch"] += 1
            c3["clock_mismatch"] += 1
            continue
        for h, exit_day in zip(SUE_H, e.exits):
            daily = study.outcome(daily_bars, e.symbol, point, h, cost)
            if daily is None:
                c2["no_daily"] += 1
                c3["no_daily"] += 1
                continue
            minute, why, late = study_clock_return(
                view, e.symbol, e.session, e.entry, exit_day, cost
            )
            c2["late_first_bar"] += late
            if minute is None:
                c2[why] += 1
            else:
                clock.append(Paired(_key(o), h, o.signal, e.session, daily, minute))
            fill, why3 = realistic_return(view, e.symbol, e.published_at, e.session, exit_day, cost)
            if fill is None:
                c3[why3] += 1
            else:
                real.append(Paired(_key(o), h, o.signal, e.session, daily, fill))
    pinned = {"units_sha": sha, "units": len(sue_units), "events": len(sample)}
    return clock, real, c2, c3, pinned


# ── every gate ────────────────────────────────────────────────


async def run_gates(
    engine: AsyncEngine,
    group: Group | Literal["all"],
    *,
    workers: int = DETERMINISM_WORKERS,
    synthetic_paths: int = G1_SYNTHETIC,
) -> GateRun:
    """Run one gate group (or all three: G1, G2, G3) and write its rows; a group that is
    refused leaves the others to run."""
    out = GateRun()
    if group in ("lookahead", "all"):
        out.extend(await run_lookahead(engine, workers=workers, synthetic_paths=synthetic_paths))
    if group in ("reactor", "all"):
        out.extend(await run_reactor(engine))
    if group in ("sue", "all"):
        out.extend(await run_sue(engine))
    return out


def describe(result: GateResult) -> str:
    """One line of a gate's numbers for the console."""
    m = result.metrics
    g = result.gate
    if g in ("g1-lookahead", "g1-synthetic"):
        cells = m.get("cells", {})
        return "; ".join(
            f"{k}: {c['checked']} checked, {c['mismatches']} mismatches, {c['trades']} trades"
            for k, c in cells.items()
        )
    if g == "g1-determinism":
        return ", ".join(
            f"{src} {k} {'equal' if len(set(v.values())) == 1 else 'DIFFER'}"
            for src in ("g1", "synthetic")
            for k, v in m.get(src, {}).get("sha256", {}).items()
        )
    if g == "r0":
        if "error" in m:
            return str(m["error"])
        return ", ".join(
            f"{k} {c['got']} (want {c['want']})" for k, c in m.get("checks", {}).items()
        )
    if g == "r1":
        d = m.get("dropped", {})
        return (
            f"{m.get('compared', 0)} compared, max |dr| {m.get('max_abs_diff', 0.0):.1e}, "
            f"set aside {d.get('set_aside', 0)} ({d.get('set_aside_share', 0.0):.2%}), "
            f"sim only {len(d.get('sim_only', {}))}, study only {len(d.get('study_only', []))}"
        )
    if g == "r2":
        old, new, ci = m.get("legacy"), m.get("realistic"), m.get("legacy_ci95")
        if not old or not new or not ci:
            return "not computable"
        return (
            f"realistic {new['mean']:+.3%} (clustered t {new['t']:+.1f}, n {new['n']}) vs legacy "
            f"{old['mean']:+.3%} [{ci[0]:+.3%}, {ci[1]:+.3%}] (clustered t {old['t']:+.1f})"
        )
    if g == "s0":
        mid = (m.get("deltas") or {}).get("mid_20d_d10_d1", {})
        spreads = m.get("sigma_c_daily", {})
        parts = [f"mid 20d D10-D1 {mid.get('got')} (delta {mid.get('delta')})"]
        parts += [
            f"Σ_c {h}d D10-D1 {s['d10_d1']:+.3%} IC {s['ic']:+.3f}" for h, s in spreads.items()
        ]
        return "; ".join(parts)
    if g == "s1":
        return (
            f"{m.get('outcomes_sim', 0)} outcomes, max |dr| {m.get('max_abs_diff', 0.0):.1e}, "
            f"study only {m.get('only_study', 0)}, sim only {m.get('only_sim', 0)}"
        )
    if g == "s1-calib":
        return f"p99_cal {m.get('p99_cal')} on {m.get('n')} of {m.get('pairs')} pairs"
    if g in ("s2", "s3"):
        return "; ".join(
            f"{h}d {'ok' if v.get('passed') else 'FAIL'} (n {v.get('n', '?')})"
            for h, v in m.get("horizons", {}).items()
        )
    return ""


__all__ = [
    "CRITERIA",
    "GateRun",
    "describe",
    "DETERMINISM_WORKERS",
    "G1_IDS",
    "GATE_IDS",
    "GATE_KIND",
    "GATE_MARKET_FILL",
    "GATE_NAME",
    "GROUPS",
    "R0_EXPECTED",
    "REACTOR_IDS",
    "SEED",
    "SUE_IDS",
    "FetchAttempted",
    "FlattenHold",
    "FlattenHoldFactory",
    "G1Factory",
    "G1Story",
    "GateMarketFill",
    "GateRefused",
    "GateResult",
    "LiquidityEligibility",
    "LiquidityOnly",
    "LookaheadWorld",
    "MinuteReturns",
    "NoFetchMarket",
    "Paired",
    "Spread",
    "calib_result",
    "calibration",
    "daily_mode",
    "daily_spreads",
    "decile_spread",
    "decomposition",
    "determinism",
    "explicit_set",
    "g1_sessions",
    "g1_world",
    "gate_config",
    "headline_id",
    "liquidity_eligibility",
    "lookahead",
    "pin_units",
    "r0_result",
    "r1_result",
    "r2_result",
    "read_gate_bars",
    "realistic_config",
    "realistic_return",
    "rebuild_stories",
    "record_gate",
    "reference_deltas",
    "reference_table",
    "require_done",
    "run_gates",
    "run_lookahead",
    "run_reactor",
    "run_sue",
    "s0_result",
    "s1_result",
    "s2_result",
    "s3_result",
    "spread",
    "study_clock_return",
    "study_returns",
    "synthetic_world",
    "to_json",
    "with_spy",
]
