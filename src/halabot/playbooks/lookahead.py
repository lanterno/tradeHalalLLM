"""The look-ahead harness (spec §E.1): nothing a playbook does by T may depend on what comes after.

For a decision time T, a replay changes everything that is unknown at T:

* every bar visible after T (``ts + 60 s + feed lag > T``) is scaled by
  U(0.5, 1.5), one factor per bar for o/h/l/c/vw and another for its volume
  (:func:`perturb`), the stock's path sessions, its spare session and SPY's;
* every story item with ``available_at > T`` is deleted (each story's
  ``before(T)``, :class:`Replayable`);
* the official open and close of every session from ``since`` on are
  replaced with noise, A-factors kept (:class:`NoisyContext`).

Replaying the symbol must give the same intents and transitions at or
before T (:func:`upto`) and the same set of stories started by T
(:func:`started_by`). :func:`check_symbol` does all of it for one symbol and
a list of probes.

**The fill-bar exemption** (:func:`fill_bars_known_by`). A fill's price is
known when the fill is acknowledged, the filling bar's end + 1 s, which can
be before the bar itself reaches a delayed feed. A bar that filled an order
acknowledged by T is therefore known at T and is not perturbed: that is not
look-ahead. **Fill bars start at or after their order's active time**
(:func:`fill_bar_violations`; the exchange asserts it as it fills).

The Phase 0 gate G1 (``halal_trader/events/sim_gate.py``) runs this on 500
real stories and 1,000 synthetic paths; the simulator's own tests run it on
random markets and hand-built paths.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Protocol

import numpy as np

from halabot.playbooks.clock import FILL_ACK, FeedProfile
from halabot.playbooks.interfaces import ContextView, StoryView
from halabot.playbooks.playbook import PlaybookFactory
from halabot.playbooks.records import StoryOutcome
from halabot.playbooks.sim import MakePlaybook, simulate_symbol
from halabot.playbooks.types import PathData, PathSkip, SimConfig, SpyData
from halal_trader.data.minutes import BarArrays

BAR_SECONDS = 60
SCALE = (0.5, 1.5)  # U(0.5, 1.5): every bar visible after T
NOISE = (1.0, 500.0)  # the official open and close of sessions from ``since`` on


class Replayable(StoryView, Protocol):
    """A story that can forget its items: ``before(t)`` keeps those with ``available_at <= t``."""

    def before(self, t: datetime) -> Replayable: ...


def visible_cut(t: datetime, feed: FeedProfile) -> int:
    """Bars starting after this epoch second become visible after ``t`` (``ts + 60 + lag > t``)."""
    return int(t.timestamp()) - BAR_SECONDS - int(feed.bar_lag.total_seconds())


def perturb(
    bars: BarArrays, after: int, keep: Collection[int], rng: np.random.Generator
) -> BarArrays:
    """``bars`` with every bar starting after epoch second ``after`` scaled, except ``keep``.

    o/h/l/c/vw share one U(0.5, 1.5) factor per bar, the volume has its own;
    two draws of ``len(bars)`` are taken whatever is hit, so the generator
    advances the same way on every replay.
    """
    hit = np.asarray([int(t) > after and int(t) not in keep for t in bars.ts], dtype=bool)
    f = np.where(hit, rng.uniform(*SCALE, len(bars)), 1.0)
    g = np.where(hit, rng.uniform(*SCALE, len(bars)), 1.0)
    return BarArrays(
        bars.ts, bars.o * f, bars.h * f, bars.l * f, bars.c * f, bars.v * g, bars.vw * f
    )


def perturb_path(
    path: PathData, after: int, keep: Collection[int], rng: np.random.Generator
) -> PathData:
    """A path with its sessions' bars, and its spare session's, perturbed after ``after``."""
    spare = path.spare_bars
    return replace(
        path,
        bars=tuple(perturb(b, after, keep, rng) for b in path.bars),
        spare_bars=perturb(spare, after, keep, rng) if spare is not None else None,
    )


def fill_bars_known_by(outcomes: Iterable[StoryOutcome], t: datetime) -> set[int]:
    """Start (epoch s) of every bar that filled an order acknowledged by ``t``: exempt."""
    ack = timedelta(seconds=BAR_SECONDS) + FILL_ACK
    keep: set[int] = set()
    for o in outcomes:
        bars = [o.entry_bar_ts]
        if o.trade is not None:
            bars.append(o.trade.exit_bar_ts)
        for b in bars:
            if b is not None and b + ack <= t:
                keep.add(int(b.timestamp()))
    return keep


def fill_bar_violations(outcomes: Iterable[StoryOutcome]) -> list[str]:
    """Story ids of trades whose entry or exit bar starts before its order was active."""
    bad: list[str] = []
    for o in outcomes:
        t = o.trade
        if t is None:
            continue
        early_entry = t.entry_bar_ts < t.entry_active_at
        early_exit = t.exit_bar_ts is not None and t.exit_bar_ts < t.exit_active_at
        if early_entry or early_exit:
            bad.append(o.story_id)
    return bad


@dataclass(frozen=True, slots=True)
class NoisyDaily:
    """An official daily bar replaced with noise (the fields the simulator reads)."""

    open: float
    high: float
    low: float
    close: float
    volume: float


class NoisyContext:
    """``base`` with the official open and close of every session from ``since`` on
    (every session when None) replaced with U(1, 500) noise; A-factors, screens and
    sessions are ``base``'s. Each (symbol, day) is drawn once, on first use."""

    def __init__(
        self, base: ContextView, rng: np.random.Generator, *, since: date | None = None
    ) -> None:
        self._base = base
        self._rng = rng
        self._since = since
        self._cache: dict[tuple[str, date], NoisyDaily | None] = {}

    @property
    def sessions(self) -> Sequence[date]:
        return self._base.sessions

    def adj(self, symbol: str, day: date) -> float | None:
        return self._base.adj(symbol, day)

    def screen_verdict(self, symbol: str, day: date) -> str:
        return self._base.screen_verdict(symbol, day)

    def daily(self, symbol: str, day: date) -> NoisyDaily | None:
        key = (symbol, day)
        if key in self._cache:
            return self._cache[key]
        dp = self._base.daily(symbol, day)
        out: NoisyDaily | None = None
        if dp is not None:
            volume = float(getattr(dp, "volume", 0.0))
            if self._since is not None and day < self._since:
                o, c = float(dp.open), float(dp.close)  # known by then: kept
            else:
                o, c = (float(x) for x in self._rng.uniform(*NOISE, 2))
            out = NoisyDaily(o, max(o, c), min(o, c), c, volume)
        self._cache[key] = out
        return out


def upto(outcome: StoryOutcome, t: datetime) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The outcome's intents and transitions at or before ``t``, each as its exact ``repr``
    (floats round-trip, and a NaN equals a NaN)."""
    return (
        tuple(repr(x) for x in outcome.intents if x[0] <= t),
        tuple(repr(x) for x in outcome.transitions if x[0] <= t),
    )


def started_by(outcomes: Iterable[StoryOutcome], t: datetime) -> set[str]:
    """Stories whose playbook started at or before ``t``."""
    return {o.story_id for o in outcomes if o.start_at is not None and o.start_at <= t}


@dataclass(frozen=True, slots=True)
class Probe:
    """One decision time to check, and the first session whose official prices are noised."""

    t: datetime
    since: date | None = None


@dataclass(frozen=True, slots=True)
class Mismatch:
    """A replay that differed from the base run at or before T."""

    story_id: str
    t: datetime
    what: str  # "started" | "intents" | "transitions"


@dataclass(slots=True)
class SymbolCheck:
    """One symbol's look-ahead check: comparisons made, mismatches, the base run."""

    symbol: str
    checked: int = 0
    probes: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    base: list[StoryOutcome] = field(default_factory=list)


def check_symbol(
    symbol: str,
    stories: Sequence[Replayable],
    make_playbook: PlaybookFactory | MakePlaybook,
    paths: Mapping[str, PathData | PathSkip],
    spy: SpyData,
    ctx: ContextView,
    cfg: SimConfig,
    probes: Iterable[Probe],
    rng: np.random.Generator,
    *,
    base: Sequence[StoryOutcome] | None = None,
) -> SymbolCheck:
    """Replay ``symbol`` at each probe with everything unknown at its T changed (module
    docstring) and compare with the base run (``base``, or simulated here).

    ``stories`` are every story of the symbol (those that never start bring
    their items); only the symbol's own paths, and SPY on their sessions,
    are perturbed. A comparison is one story started by T; a mismatch is a
    different set of started stories, or a started story's intents or
    transitions differing at or before T.
    """
    if base is None:
        base = simulate_symbol(
            symbol, stories, make_playbook, paths, spy, ctx, cfg, keep_transitions=True
        )
    out = SymbolCheck(symbol, base=list(base))
    ids = {s.story_id for s in stories}
    own = {sid: p for sid, p in sorted(paths.items()) if sid in ids}
    days = sorted(
        {s.day for p in own.values() if isinstance(p, PathData) for s in p.sessions}
        | {p.spare.day for p in own.values() if isinstance(p, PathData) and p.spare is not None}
    )
    spy_days = {d: spy.days[d] for d in days if d in spy.days}
    for probe in sorted(probes, key=lambda p: (p.t, p.since or date.min)):
        t = probe.t
        cut = visible_cut(t, cfg.feed)
        keep = fill_bars_known_by(base, t)
        moved: dict[str, PathData | PathSkip] = {
            sid: perturb_path(p, cut, keep, rng) if isinstance(p, PathData) else p
            for sid, p in own.items()
        }
        spy_moved = SpyData({d: perturb(b, cut, set(), rng) for d, b in spy_days.items()})
        again = simulate_symbol(
            symbol,
            [s.before(t) for s in stories],
            make_playbook,
            moved,
            spy_moved,
            NoisyContext(ctx, rng, since=probe.since),
            cfg,
            keep_transitions=True,
        )
        out.probes += 1
        started = started_by(base, t)
        if started != started_by(again, t):
            for sid in sorted(started ^ started_by(again, t)):
                out.mismatches.append(Mismatch(sid, t, "started"))
            continue
        by_id = {o.story_id: o for o in again}
        for o in base:
            if o.story_id not in started:
                continue
            intents, transitions = upto(o, t)
            intents_again, transitions_again = upto(by_id[o.story_id], t)
            if intents != intents_again:
                out.mismatches.append(Mismatch(o.story_id, t, "intents"))
            elif transitions != transitions_again:
                out.mismatches.append(Mismatch(o.story_id, t, "transitions"))
            out.checked += 1
    return out


__all__ = [
    "NOISE",
    "SCALE",
    "Mismatch",
    "NoisyContext",
    "NoisyDaily",
    "Probe",
    "Replayable",
    "SymbolCheck",
    "check_symbol",
    "fill_bar_violations",
    "fill_bars_known_by",
    "perturb",
    "perturb_path",
    "started_by",
    "upto",
    "visible_cut",
]
