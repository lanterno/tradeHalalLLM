"""Paths for the simulator: minute bars per story, behind the window guard.

**Window guard** (:class:`WindowGuard`). Every session read is checked:

* no session after ``window_end``, so a path never spills into the next
  window (and ``window_end`` itself cannot pass its window's last day);
* sessions in [2016-10-01, 2024-12-31] need ``prereg_id``: a
  ``quant_trials`` row of kind ``preregistration`` whose ``config_hash``
  equals ``unlock.config_hash``;
* sessions from 2025-01-01 need ``holdout=True``, and the latest
  ``kind='verdict'`` row under that same hash and the preregistration's
  name must say ``pass`` with no ``stage-a``, ``window`` or ``amendment``
  row of that trial after it;
* a ``gate=`` unlock admits only its **pinned** unit set, whatever the
  date. ``unlock.units`` must hash (``unit_set_sha``) to ``unlock.units_sha``;
  every unit must lie in the gate's date range (:data:`GATE_RANGES`: g1 and
  calib 2016-01-04..2016-09-30, sue 2016-01-04..2020-01-30, reactor
  2025-12-01..2026-10-09); and the ledger must hold the pin, a
  ``quant_trials`` row ``kind='gate-units'``, ``name=`` :data:`GATE_UNITS_NAME`,
  ``config={"gate": <id>, "units_sha": <sha>}``, which the gate code writes
  with :func:`register_gate_units` before it runs, and it must be the gate's
  current pin. SPY is admitted on the days of those units.

**One pin per gate, re-pinned only with a reason.** :func:`register_gate_units`
refuses a set whose sha differs from the gate's pin (re-registering the same
set is a no-op), and refuses a set that does not hash to ``expected_sha``
when the caller passes one (the plan's ``UnitPlan.sha("gate_<id>")``, so a
gate's set is the one plan H fetched). An upstream data fix (the aliases,
the stories) legitimately changes a selection; then ``supersede=<reason>``
re-pins: a new ``gate-units`` row, ``config={"gate", "units_sha",
"supersedes": <the pin it replaces>, "reason"}``. The old pin stays on the
ledger; the **latest** pin is the gate's (:func:`gate_pin`), so the old set
never opens again. Every change of set must be such a re-pin of the row
before it (:func:`current_pin`): two registrations racing with different
sets, or two re-pins of the same pin, leave a change nobody sanctioned, and
then no set opens (fail closed) until a re-pin with a reason settles it.
:meth:`WindowGuard.verify` opens a gate only for its current pin.

The sue range ends at :func:`sue_last_exit`, the last h=20 exit session of an
observation published by 2019-12-31 (spec §H: ``gate_sue`` holds the entry
session and the h=5 and h=20 exit sessions).

:meth:`MinuteBarLoader.prepare` verifies the unlock against the ledger and
:meth:`MinuteBarLoader.check` every day of every requested path, so
``sim.run`` refuses a run before it writes anything.

**Coverage.** A path is simulated only if every (symbol, d) and (SPY, d) of
it is a done unit of the minute backfill (``minutes.done_units``; zero bars
allowed). Then, per path, in order:

* ``spy_missing``: a SPY session with no bar, or no A-factor;
* ``spy_thin``: a SPY session with bars, but fewer than 300 (150 on an early
  close). Both are data skips (``types.DATA_SKIPS``); ``spy_thin`` is its own
  reason because a study that needs only one SPY bar after its decision
  computes those headlines (the reactor study, R1 in ``legacy.py``);
* ``halted_all_day``: no bar on S and no daily bar on S (not a data skip);
* ``units_missing``: a path session with no bar while a daily bar exists;
* ``no_daily``: no A-factor for S, or for a path session that has bars;
* ``adjust_defect``: ``compliance.runner.corporate_actions`` reports a stale
  adjusted series, or an adjustment it cannot read as a split, for the
  symbol or SPY between the first sigma session and the last path session;
* ``bad_bars``: more than 5 bars dropped by the sanity rule (a non-positive
  price, ``h < max(o, c)`` or ``l > min(o, c)``) on the path sessions.

A loaded path keeps the count of bars the rule dropped from its sessions
(``PathData.dropped``) and, separately, from its spare session
(``spare_dropped``, which never counts toward ``bad_bars``: the spare is not
a path session). SPY's sessions are cut by the same rule when they load, and
:attr:`MinuteBarLoader.spy_dropped` maps each session that lost bars to the
cut bars' timestamps (so a reader can tell which decisions they precede). The
counts (``dropped_bars``, ``spare_dropped_bars``, ``spy_dropped_bars``) are
in :attr:`MinuteBarLoader.counts`.

**Loading.** Requests are cut into month-local batches of 250 paths, by
reaction session; each batch is one ``minutes.read_windows`` call (and one
``corporate_actions`` call), and the next batch loads while the current one
is simulated. At start the loader asserts that ``market_hours`` sessions
over the window equal SPY's raw daily-bar sessions.

**Known dependency (the spare session).** An exit that finds no market on
the deadline session fills on the next session (``no_market``) only when
that session's units are loaded and done. Plan H (``events/units.py``)
fetches the path sessions S .. S+n-1 and not S+n, so until ``units.py``
adds the spare session, such exits fall through to ``unresolved``, which
counts against the T6 limit (0.5%). That fix belongs to ``units.py``, not
here: the loader already uses the spare whenever it is done.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from collections import Counter
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks.interfaces import ContextView
from halabot.playbooks.types import PathData, PathSkip, Session, SpyData, path_days
from halal_trader.core import events
from halal_trader.data import minutes
from halal_trader.data.minutes import BarArrays
from halal_trader.market_hours import (
    EARLY_CLOSE_DATES,
    is_trading_day,
    next_trading_day,
    trading_days_back,
)

if TYPE_CHECKING:
    from halal_trader.compliance.runner import CorporateActions

logger = logging.getLogger(__name__)

SPY: Final = "SPY"
LOCK_START: Final = date(2016, 10, 1)  # from here a registered preregistration is needed
HOLDOUT_START: Final = date(2025, 1, 1)  # from here a passing verdict under the same hash
SPY_MIN_BARS: Final = 300
SPY_MIN_BARS_EARLY: Final = 150
MAX_DROPPED: Final = 5  # more bars than this dropped in one path: bad_bars
DEFECT_SESSIONS_BEFORE: Final = 62  # the defect check starts 62 sessions before S


class Window(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "holdout"
    GATE = "gate"


# First session of each window (the calendar check), and the last day a window may reach.
WINDOW_START: Final = {
    Window.TRAIN: date(2016, 10, 3),
    Window.VALIDATION: date(2022, 1, 3),
    Window.HOLDOUT: date(2025, 1, 2),
    Window.GATE: date(2016, 1, 4),
}
WINDOW_LAST: Final = {Window.TRAIN: date(2021, 12, 31), Window.VALIDATION: date(2024, 12, 31)}

Gate = Literal["g1", "calib", "reactor", "sue"]

# Σ_c (spec §E.3): SUE observations published 2016-01-04..2019-12-31; gate_sue (spec §H)
# holds their entry sessions and their exit sessions at h = 5 and h = 20.
SUE_FIRST: Final = date(2016, 1, 4)
SUE_LAST_PUBLISHED: Final = date(2019, 12, 31)
SUE_MAX_HORIZON: Final = 20


def sue_last_exit() -> date:
    """The last session ``gate_sue`` can hold: the h=20 exit of the last observation.

    An observation published on 2019-12-31 enters at the latest at the open
    of the next session (``study.entry_point``), and exits on the 20th
    session counted from it (``exit_i = i + h - 1``); a close entry on
    2019-12-31 exits on the same session (``i + h``). By ``market_hours``:
    2020-01-30.
    """
    return path_days(next_trading_day(SUE_LAST_PUBLISHED), SUE_MAX_HORIZON)[-1]


# The sessions each Phase 0 gate may read (spec §E, §H): every unit of its set lies inside.
GATE_RANGES: Final[dict[str, tuple[date, date]]] = {
    "g1": (date(2016, 1, 4), date(2016, 9, 30)),
    "calib": (date(2016, 1, 4), date(2016, 9, 30)),
    "sue": (SUE_FIRST, sue_last_exit()),
    "reactor": (date(2025, 12, 1), date(2026, 10, 9)),
}
GATE_UNITS_NAME: Final = "research.news.sim-gate.units"
GATE_UNITS_KIND: Final = "gate-units"


@dataclass(frozen=True, slots=True)
class WindowUnlock:
    """What lets a run read locked sessions.

    ``prereg_id``: a ``quant_trials`` row of kind ``preregistration`` whose
    ``config_hash`` is ``config_hash`` (``config_hash(PREREG)``). ``holdout``
    additionally needs the latest ``verdict`` row under that hash to pass,
    with nothing newer that could change it. ``gate``
    with ``units`` and ``units_sha`` admits exactly that unit set, once it is
    pinned in the ledger (:func:`register_gate_units`) and inside the gate's
    dates (:data:`GATE_RANGES`).
    """

    prereg_id: int | None = None
    config_hash: str | None = None
    gate: Gate | None = None
    units: frozenset[tuple[str, date]] | None = None
    units_sha: str | None = None
    holdout: bool = False


class WindowLocked(RuntimeError):
    """A read the window guard does not allow."""


class CalendarMismatch(RuntimeError):
    """``market_hours`` and SPY's raw daily bars disagree on the sessions."""


def unit_set_sha(units: Iterable[tuple[str, date]]) -> str:
    """sha256 of the sorted ``SYMBOL:YYYY-MM-DD`` lines of a unit set."""
    lines = sorted(minutes.unit(symbol, day) for symbol, day in units)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def check_gate_units(gate: str, units: Iterable[tuple[str, date]]) -> frozenset[tuple[str, date]]:
    """``units`` as a set; :class:`WindowLocked` unless all lie in ``gate``'s dates."""
    span = GATE_RANGES.get(gate)
    if span is None:
        raise WindowLocked(f"unknown gate {gate!r}")
    out = frozenset(units)
    if not out:
        raise WindowLocked(f"gate {gate!r} has an empty unit set")
    lo, hi = span
    outside = sorted(u for u in out if not lo <= u[1] <= hi)
    if outside:
        shown = ", ".join(minutes.unit(s, d) for s, d in outside[:5])
        raise WindowLocked(
            f"gate {gate!r} reads {lo}..{hi} only; {len(outside)} units outside: {shown}"
        )
    return out


@dataclass(frozen=True, slots=True)
class GatePin:
    """One ``gate-units`` row: the set it pins and, for a re-pin, the pin it replaces and why."""

    row_id: int
    sha: str
    supersedes: str | None = None
    reason: str | None = None


async def gate_pin_rows(engine: AsyncEngine, gate: str) -> list[GatePin]:
    """Every ``gate-units`` row of ``gate``, oldest first."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT id, config->>'units_sha' AS sha, config->>'supersedes' AS supersedes, "
                "config->>'reason' AS reason FROM quant_trials "
                "WHERE kind = :k AND name = :n AND config->>'gate' = :g ORDER BY id"
            ),
            {"k": GATE_UNITS_KIND, "n": GATE_UNITS_NAME, "g": gate},
        )
        return [GatePin(int(r.id), str(r.sha), r.supersedes, r.reason) for r in rows]


async def gate_pins(engine: AsyncEngine, gate: str) -> set[str]:
    """Every unit-set sha the ledger has pinned for ``gate``, superseded ones included."""
    return {p.sha for p in await gate_pin_rows(engine, gate)}


def current_pin(gate: str, rows: Sequence[GatePin]) -> str | None:
    """The gate's pin: the latest row's set, or None when nothing is pinned.

    Every change of set along the rows (oldest first) must be a re-pin that
    supersedes the row just before it; :class:`WindowLocked` otherwise (two
    registrations raced with different sets, two re-pins replaced the same
    pin, or a row was written by hand): no set opens. A later re-pin of the
    row before it settles the gate again.
    """
    ordered = sorted(rows, key=lambda p: p.row_id)
    unsanctioned: GatePin | None = None
    for before, pin in zip(ordered, ordered[1:]):
        if pin.supersedes is not None:  # a re-pin is sanctioned only over the row before it
            unsanctioned = None if pin.supersedes == before.sha else pin
        elif pin.sha != before.sha:
            unsanctioned = pin
    if unsanctioned is not None:
        raise WindowLocked(
            f"gate {gate!r}: ledger row {unsanctioned.row_id} pins {unsanctioned.sha[:12]} "
            "without superseding the pin before it (a race, or a row written by hand); "
            "no unit set opens until a re-pin with a reason"
        )
    return ordered[-1].sha if ordered else None


async def gate_pin(engine: AsyncEngine, gate: str) -> str | None:
    """``gate``'s current pin (:func:`current_pin`) as the ledger holds it."""
    return current_pin(gate, await gate_pin_rows(engine, gate))


async def register_gate_units(
    engine: AsyncEngine,
    gate: Gate,
    units: Iterable[tuple[str, date]],
    *,
    expected_sha: str | None = None,
    supersede: str | None = None,
) -> str:
    """Pin ``gate``'s unit set in the ledger before the gate runs; returns its sha256.

    Writes one ``quant_trials`` row (``kind='gate-units'``, no Sharpe, so it
    is never counted as a trial) unless the set is already the gate's pin.
    Refuses (:class:`WindowLocked`) a unit outside the gate's dates, a set
    that does not hash to ``expected_sha`` when one is given (the units plan
    passes ``UnitPlan.sha("gate_<id>")``), and a set other than the gate's
    pin, unless ``supersede`` gives the reason for a re-pin (an upstream data
    fix changed the selection): the row then records ``supersedes`` (the
    pin it replaces, the latest row's) and ``reason``, and the new set is
    the gate's pin from then on. A re-pin also settles a gate whose rows
    conflict (:func:`current_pin`).
    """
    from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl

    if supersede is not None and not supersede.strip():
        raise WindowLocked(f"gate {gate!r}: a re-pin needs a reason")
    unit_set = check_gate_units(gate, units)
    sha = unit_set_sha(unit_set)
    if expected_sha is not None and sha != expected_sha:
        raise WindowLocked(
            f"gate {gate!r}: the unit set hashes to {sha[:12]}, "
            f"not the expected {expected_sha[:12]}"
        )
    rows = await gate_pin_rows(engine, gate)
    try:
        pinned = current_pin(gate, rows)
    except WindowLocked:
        if supersede is None:
            raise
        pinned = None  # a conflict: only a re-pin of the latest row settles it
    if pinned == sha:
        return sha
    config: dict[str, str] = {"gate": gate, "units_sha": sha}
    if rows:
        old = rows[-1].sha
        if supersede is None:
            raise WindowLocked(
                f"gate {gate!r} is already pinned to {old[:12]}; one pin per gate, not "
                f"{sha[:12]} (a re-pin needs a reason)"
            )
        config |= {"supersedes": old, "reason": supersede.strip()}
    lo, hi = GATE_RANGES[gate]
    row_id = await QuantTrialRepoImpl(engine).record_trial(
        name=GATE_UNITS_NAME,
        kind=GATE_UNITS_KIND,
        config=config,
        window=f"{lo}..{hi}",
        metrics={"units": len(unit_set)},
    )
    if "supersedes" in config:
        logger.warning(
            "gate %s re-pinned: %s supersedes %s (%s)",
            gate,
            sha[:12],
            config["supersedes"][:12],
            config["reason"],
            extra={
                "event": events.GATE_UNITS_REPINNED,
                "gate": gate,
                "units_sha": sha,
                "supersedes": config["supersedes"],
                "reason": config["reason"],
                "trial_id": row_id,
            },
        )
    return sha


class WindowGuard:
    """Refuses every session a run may not read (see the module docstring)."""

    def __init__(self, *, window: Window, window_end: date, unlock: WindowUnlock) -> None:
        last = WINDOW_LAST.get(window)
        if last is not None and window_end > last:
            raise WindowLocked(f"{window} ends by {last}, not {window_end}")
        self.window = window
        self.window_end = window_end
        self.unlock = unlock
        self._prereg_ok = False
        self._holdout_ok = False
        self._gate_ok = False
        self._gate_days: frozenset[date] = frozenset()
        if unlock.gate is not None:
            if unlock.units is None or unlock.units_sha is None:
                raise WindowLocked(f"gate {unlock.gate!r} needs its unit set and its sha256")
            units = check_gate_units(unlock.gate, unlock.units)
            if unit_set_sha(units) != unlock.units_sha:
                raise WindowLocked(f"gate {unlock.gate!r}: the unit set does not match its sha256")
            self._gate_days = frozenset(day for _, day in units)

    async def verify(self, engine: AsyncEngine) -> None:
        """Check the unlock's ledger rows; raises :class:`WindowLocked` on any mismatch."""
        u = self.unlock
        if (u.prereg_id is not None or u.holdout) and not u.config_hash:
            raise WindowLocked("an unlock needs the preregistered config_hash")
        if u.gate is not None:
            assert u.units_sha is not None  # checked at construction
            rows = await gate_pin_rows(engine, u.gate)
            pin = current_pin(u.gate, rows)  # WindowLocked on a change nobody sanctioned
            if pin != u.units_sha:
                if any(p.sha == u.units_sha for p in rows):
                    assert pin is not None
                    raise WindowLocked(
                        f"gate {u.gate!r}: unit set {u.units_sha[:12]} was superseded; "
                        f"the gate's pin is {pin[:12]}"
                    )
                raise WindowLocked(
                    f"gate {u.gate!r}: unit set {u.units_sha[:12]} is not pinned in the ledger "
                    f"(register_gate_units)"
                )
            self._gate_ok = True
        async with engine.connect() as conn:
            if u.prereg_id is not None:
                row = (
                    await conn.execute(
                        text("SELECT kind, config_hash FROM quant_trials WHERE id = :id"),
                        {"id": u.prereg_id},
                    )
                ).first()
                if row is None or row.kind != "preregistration":
                    raise WindowLocked(f"quant_trials {u.prereg_id} is not a preregistration")
                if row.config_hash != u.config_hash:
                    raise WindowLocked(
                        f"preregistration {u.prereg_id} has hash {row.config_hash}, "
                        f"not {u.config_hash}"
                    )
                self._prereg_ok = True
            if u.holdout:
                # The latest verdict under the hash, of the trial registered under it,
                # must pass, and nothing that can change it may have come after it.
                latest = (
                    await conn.execute(
                        text(
                            "SELECT id, name, verdict FROM quant_trials WHERE kind = 'verdict' "
                            "AND config_hash = :h AND name IN (SELECT name FROM quant_trials "
                            "WHERE kind = 'preregistration' AND config_hash = :h) "
                            "ORDER BY id DESC LIMIT 1"
                        ),
                        {"h": u.config_hash},
                    )
                ).first()
                if latest is None or latest.verdict != "pass":
                    raise WindowLocked(
                        "the holdout opens only after a passing verdict "
                        f"(the latest is {latest.verdict if latest else 'none'})"
                    )
                newer = await conn.scalar(
                    text(
                        "SELECT count(*) FROM quant_trials WHERE name = :n AND config_hash = :h "
                        "AND kind IN ('stage-a', 'window', 'amendment') AND id > :i"
                    ),
                    {"n": latest.name, "h": u.config_hash, "i": latest.id},
                )
                if newer:
                    raise WindowLocked(
                        f"the holdout opens only after a passing verdict: {newer} stage, "
                        f"window or amendment row(s) came after verdict {latest.id}"
                    )
                self._holdout_ok = True

    def reason(self, symbol: str, day: date) -> str | None:
        """Why (symbol, day) may not be read, or None."""
        if day > self.window_end:
            return f"{day} is after the window's end {self.window_end}"
        if self.unlock.gate is not None:
            if not self._gate_ok:
                return f"gate {self.unlock.gate!r}'s unit set is not verified as pinned"
            units = self.unlock.units or frozenset()
            if (symbol, day) in units or (symbol == SPY and day in self._gate_days):
                return None
            return f"{symbol} {day} is outside gate {self.unlock.gate!r}'s unit set"
        if day >= HOLDOUT_START:
            return None if self._holdout_ok else f"{day} is in the holdout"
        if day >= LOCK_START:
            return None if self._prereg_ok else f"{day} needs a registered preregistration"
        return None

    def allows(self, symbol: str, day: date) -> bool:
        return self.reason(symbol, day) is None

    def check(self, symbol: str, day: date) -> None:
        if (why := self.reason(symbol, day)) is not None:
            raise WindowLocked(why)


async def check_calendar(engine: AsyncEngine, start: date, end: date) -> None:
    """Raise :class:`CalendarMismatch` unless market_hours' sessions are SPY's raw daily ones."""
    if start > end:
        return
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT day FROM daily_bars WHERE symbol = :s AND adjustment = 'raw' "
                "AND day BETWEEN :a AND :b"
            ),
            {"s": SPY, "a": start, "b": end},
        )
        bars = {r.day for r in rows}
    ours: set[date] = set()
    d = start
    while d <= end:
        if is_trading_day(d):
            ours.add(d)
        d += timedelta(days=1)
    if ours != bars:
        only_ours = [d.isoformat() for d in sorted(ours - bars)[:5]]
        only_bars = [d.isoformat() for d in sorted(bars - ours)[:5]]
        raise CalendarMismatch(
            f"sessions {start}..{end}: market_hours only {only_ours}, SPY bars only {only_bars}"
        )


def sane(bars: BarArrays) -> tuple[BarArrays, int]:
    """Drop rows with a non-positive price, ``h < max(o, c)`` or ``l > min(o, c)``.

    A non-positive VWAP is read as missing (NaN), not dropped. Returns the
    kept bars and the number dropped; rows are never repaired.
    """
    o, h, low, c = bars.o, bars.h, bars.l, bars.c
    ok = (o > 0) & (h > 0) & (low > 0) & (c > 0) & (h >= np.maximum(o, c))
    ok &= low <= np.minimum(o, c)
    vw = np.where(bars.vw > 0, bars.vw, np.nan)
    dropped = int(len(ok) - int(ok.sum()))
    if dropped == 0 and not np.any(bars.vw <= 0):
        return bars, 0
    return (
        BarArrays(bars.ts[ok], o[ok], h[ok], low[ok], c[ok], bars.v[ok], vw[ok].astype(np.float64)),
        dropped,
    )


@dataclass(frozen=True, slots=True)
class PathRequest:
    story_id: str
    symbol: str
    session: date
    n_sessions: int


def batches_of(requests: Sequence[PathRequest], size: int) -> list[list[PathRequest]]:
    """Requests by (session, story_id), cut into batches of ``size`` inside one month."""
    out: list[list[PathRequest]] = []
    month: tuple[int, int] | None = None
    for r in sorted(requests, key=lambda r: (r.session, r.story_id)):
        key = (r.session.year, r.session.month)
        if not out or key != month or len(out[-1]) >= size:
            out.append([])
            month = key
        out[-1].append(r)
    return out


@dataclass(slots=True)
class _Plan:
    req: PathRequest
    days: list[date]
    spare: date | None
    since: date  # adjust_defect span: (since, days[-1]]


class MinuteBarLoader:
    """Loads paths in month-local batches, prefetching one batch ahead."""

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        window: Window,
        window_end: date,
        unlock: WindowUnlock,
        batch_paths: int = 250,
    ) -> None:
        self._engine = engine
        self.guard = WindowGuard(window=window, window_end=window_end, unlock=unlock)
        self._batch_paths = batch_paths
        self._spy = SpyData()
        self._done: set[str] | None = None
        self.counts: Counter[str] = Counter()  # skips by reason, dropped bars
        # SPY sessions the sanity rule cut bars from -> the cut bars' ts (epoch seconds, sorted)
        self.spy_dropped: dict[date, tuple[int, ...]] = {}

    async def prepare(self) -> None:
        """Verify the unlock, check the calendar and read the done units (once)."""
        if self._done is not None:
            return
        await self.guard.verify(self._engine)
        await check_calendar(self._engine, WINDOW_START[self.guard.window], self.guard.window_end)
        self._done = await minutes.done_units(self._engine)

    def check(self, requests: Sequence[PathRequest]) -> None:
        """Raise :class:`WindowLocked` unless every day of every path (and SPY's) may be read.

        Call it after :meth:`prepare` and before writing anything: a path
        that crosses ``window_end``, or leaves a gate's unit set, is refused
        up front rather than when its batch loads.
        """
        for req in requests:
            for d in path_days(req.session, req.n_sessions):
                self.guard.check(req.symbol, d)
                self.guard.check(SPY, d)

    async def spy(self) -> SpyData:
        """SPY's bars, filled batch by batch as paths load."""
        await self.prepare()
        return self._spy

    async def paths(
        self, requests: Sequence[PathRequest], context: ContextView
    ) -> AsyncIterator[PathData | PathSkip]:
        async for batch in self.batches(requests, context):
            for item in batch:
                yield item

    async def batches(
        self, requests: Sequence[PathRequest], context: ContextView
    ) -> AsyncIterator[list[PathData | PathSkip]]:
        """One list per batch, in session order; the next batch loads meanwhile."""
        await self.prepare()
        groups = batches_of(requests, self._batch_paths)
        if not groups:
            return
        pending = asyncio.ensure_future(self._load(groups[0], context))
        try:
            for n in range(len(groups)):
                result = await pending
                if n + 1 < len(groups):
                    pending = asyncio.ensure_future(self._load(groups[n + 1], context))
                yield result
        finally:
            if not pending.done():
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending

    def _plan(self, req: PathRequest) -> _Plan:
        days = path_days(req.session, req.n_sessions)
        for d in days:
            self.guard.check(req.symbol, d)
            self.guard.check(SPY, d)
        spare: date | None = next_trading_day(days[-1])
        assert self._done is not None
        if not (
            spare is not None
            and self.guard.allows(req.symbol, spare)
            and self.guard.allows(SPY, spare)
            and minutes.unit(req.symbol, spare) in self._done
            and minutes.unit(SPY, spare) in self._done
        ):
            spare = None
        since = trading_days_back(req.session, DEFECT_SESSIONS_BEFORE + 1)[0]
        return _Plan(req, days, spare, since)

    async def _load(
        self, batch: list[PathRequest], context: ContextView
    ) -> list[PathData | PathSkip]:
        assert self._done is not None
        done = self._done
        out: dict[str, PathData | PathSkip] = {}
        plans: list[_Plan] = []
        for req in batch:
            plan = self._plan(req)
            if any(minutes.unit(req.symbol, d) not in done for d in plan.days):
                out[req.story_id] = PathSkip(req.story_id, "units_missing")
            elif any(minutes.unit(SPY, d) not in done for d in plan.days):
                out[req.story_id] = PathSkip(req.story_id, "spy_missing")
            else:
                plans.append(plan)

        defects = await self._defects(plans)
        units: set[tuple[str, date]] = set()
        for p in plans:
            for d in [*p.days, *([p.spare] if p.spare else [])]:
                units.add((p.req.symbol, d))
                if d not in self._spy.days:
                    units.add((SPY, d))
        bars = await minutes.read_windows(self._engine, sorted(units))
        end = minutes.session_bounds(self.guard.window_end)[1].timestamp()
        for (symbol, day), arrays in bars.items():
            if len(arrays) and int(arrays.ts[-1]) >= end:  # the guard, asserted on the data
                raise WindowLocked(f"{symbol} {day}: a bar after the window's end")
            if symbol == SPY and day not in self._spy.days:
                kept, n = sane(arrays)
                self.counts["spy_dropped_bars"] += n
                if n:  # ts is unique per (symbol, minute): the rows not kept are the cut ones
                    cut = arrays.ts[~np.isin(arrays.ts, kept.ts)]
                    self.spy_dropped[day] = tuple(int(t) for t in cut)
                self._spy.days[day] = kept

        for p in plans:
            out[p.req.story_id] = self._path(p, bars, defects, context)
        result = [out[r.story_id] for r in batch]
        for item in result:
            if isinstance(item, PathSkip):
                self.counts[f"skip:{item.reason}"] += 1
            else:
                self.counts["paths"] += 1
                self.counts["dropped_bars"] += item.dropped
                self.counts["spare_dropped_bars"] += item.spare_dropped
        return result

    def _path(
        self,
        p: _Plan,
        bars: dict[tuple[str, date], BarArrays],
        defects: set[str],
        context: ContextView,
    ) -> PathData | PathSkip:
        sid, symbol, s = p.req.story_id, p.req.symbol, p.req.session
        for d in p.days:
            spy = self._spy.days.get(d)
            need = SPY_MIN_BARS_EARLY if d in EARLY_CLOSE_DATES else SPY_MIN_BARS
            if spy is None or len(spy) == 0 or context.adj(SPY, d) is None:
                return PathSkip(sid, "spy_missing")
            if len(spy) < need:
                return PathSkip(sid, "spy_thin")
        kept: list[BarArrays] = []
        dropped = 0
        for d in p.days:
            arrays, n = sane(bars.get((symbol, d), BarArrays.empty()))
            kept.append(arrays)
            dropped += n
        if len(kept[0]) == 0 and context.daily(symbol, s) is None:
            return PathSkip(sid, "halted_all_day")
        for d, arrays in zip(p.days, kept):
            if len(arrays) == 0 and context.daily(symbol, d) is not None:
                return PathSkip(sid, "units_missing")
        if context.adj(symbol, s) is None or any(
            len(arrays) and context.adj(symbol, d) is None for d, arrays in zip(p.days, kept)
        ):
            return PathSkip(sid, "no_daily")
        if sid in defects:
            return PathSkip(sid, "adjust_defect")
        if dropped > MAX_DROPPED:
            return PathSkip(sid, "bad_bars")
        spare = spare_bars = None
        spare_dropped = 0
        if p.spare is not None:
            spare_bars, spare_dropped = sane(bars.get((symbol, p.spare), BarArrays.empty()))
            spare = Session.of(p.spare)
        return PathData(
            story_id=sid,
            symbol=symbol,
            sessions=tuple(Session.of(d) for d in p.days),
            bars=tuple(kept),
            spare=spare,
            spare_bars=spare_bars,
            dropped=dropped,
            spare_dropped=spare_dropped,
        )

    async def _defects(self, plans: Sequence[_Plan]) -> set[str]:
        """Story ids whose symbol or SPY has a stale or unreadable adjustment in the span."""
        if not plans:
            return set()
        from halal_trader.compliance.runner import corporate_actions

        symbols = sorted({p.req.symbol for p in plans} | {SPY})
        since = min(p.since for p in plans)
        through = max(p.days[-1] for p in plans)
        found = await corporate_actions(self._engine, symbols, since, through)
        requeried: dict[tuple[str, date, date], CorporateActions] = {}
        out: set[str] = set()
        for p in plans:
            for symbol in (p.req.symbol, SPY):
                actions = found
                stale = found.stale.get(symbol)
                if stale is not None and stale <= p.since:
                    # ``stale`` keeps only the first date; ask again over this span.
                    key = (symbol, p.since, p.days[-1])
                    if key not in requeried:
                        requeried[key] = await corporate_actions(
                            self._engine, [symbol], p.since, p.days[-1]
                        )
                    actions = requeried[key]
                    stale = actions.stale.get(symbol)
                if stale is not None and p.since < stale <= p.days[-1]:
                    out.add(p.req.story_id)
                if any(
                    ratio is None and p.since < day <= p.days[-1]
                    for day, ratio in actions.splits.get(symbol, [])
                ):
                    out.add(p.req.story_id)
        return out


__all__ = [
    "GATE_RANGES",
    "GATE_UNITS_KIND",
    "GATE_UNITS_NAME",
    "HOLDOUT_START",
    "LOCK_START",
    "SPY",
    "SUE_FIRST",
    "SUE_LAST_PUBLISHED",
    "SUE_MAX_HORIZON",
    "WINDOW_LAST",
    "WINDOW_START",
    "CalendarMismatch",
    "Gate",
    "GatePin",
    "MinuteBarLoader",
    "PathRequest",
    "SpyData",
    "Window",
    "WindowGuard",
    "WindowLocked",
    "WindowUnlock",
    "batches_of",
    "check_calendar",
    "check_gate_units",
    "current_pin",
    "gate_pin",
    "gate_pin_rows",
    "gate_pins",
    "register_gate_units",
    "sane",
    "sue_last_exit",
    "unit_set_sha",
]
