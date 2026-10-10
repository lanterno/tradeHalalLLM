"""What a simulation run produces: trade records, story outcomes, daily legs and the book.

Every price is session-S units where it is a level (``p0``, ``low_star``,
``target``) and raw where it is a fill (``entry_px``, ``exit_px``), with
the A-factors next to it, so

``r_gross = (exit_px * adj_exit) / (entry_px * adj_entry) - 1``

and the same for SPY with its own factors. ``r_net_abn`` subtracts SPY's
return over the same bars and both sides' costs; nothing is winsorised or
dropped.

**Daily legs** (one per held session) feed the calendar-time series: on
each held session the trade runs from ``a`` (the entry fill, or the
previous official close) to ``z`` (the exit fill, or that session's
official close), and SPY over the same bars or closes.

**Daily book** (:func:`daily_book`): NAV_{d-1}/8 per trade, first come first
served by ``entry_decided_at``; a trade is rejected when eight positions
are open, or when trades closed earlier that day lost 2% of NAV_{d-1} or
more that day (open positions' marks are not counted). Open positions are
marked to market: a trade's value compounds leg by leg, ``v_d = v_{d-1} *
(1 + leg_d.stock)``, and day d earns ``v_{d-1} * leg_d.stock``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import insert, update
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks.types import Intent, Session

# ── records ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Leg:
    """One held session of a trade, in session-S units.

    ``cost`` is the fraction charged that day (one side on the entry day,
    one on the exit day).
    """

    day: date
    a: float
    z: float
    spy_a: float
    spy_z: float
    cost: float = 0.0

    @property
    def stock(self) -> float:
        return self.z / self.a - 1.0 - self.cost

    @property
    def spy(self) -> float:
        return self.spy_z / self.spy_a - 1.0

    @property
    def abn(self) -> float:
        """``z/a - 1 - (Z/A - 1) - cost``."""
        return self.stock - self.spy


@dataclass(frozen=True, slots=True)
class TradeRecord:
    run_id: str
    story_id: str
    symbol: str
    family_type: str
    cell: str
    variant: str
    feed: str
    session: date
    exit_session: date
    sessions_held: int
    start_case: str
    at_news: datetime | None
    nsn_at: datetime | None
    anchor_ts: datetime | None
    entry_decided_at: datetime
    entry_active_at: datetime
    entry_bar_ts: datetime
    exit_decided_at: datetime
    exit_active_at: datetime
    exit_bar_ts: datetime | None  # None for close_fallback and unresolved exits
    p0: float  # levels: session-S units
    spy0: float
    sigma: float
    thr: float
    low_star: float
    target: float
    entry_px: float  # fills: raw prices, with their A-factors
    exit_px: float
    adj_entry: float
    adj_exit: float
    spy_entry_px: float
    spy_exit_px: float
    spy_adj_entry: float
    spy_adj_exit: float
    cost_bps: float  # one way, the mean of the two sides charged
    rank: int
    tech: bool
    beta: float
    r_gross: float
    r_spy: float
    r_net_abn: float
    r_beta_adj: float
    exit_reason: str  # target | stop | abort | compliance | time_stop
    mae: float
    mfe: float
    hold_minutes: int
    participation: float
    flags: tuple[str, ...]  # gap_fill, late_open, close_fallback, no_market, unresolved, spy_proxy

    @property
    def exit_time(self) -> datetime:
        """When the exit filled: the end of its bar, or the exit session's close."""
        if self.exit_bar_ts is not None:
            return self.exit_bar_ts + timedelta(seconds=60)
        return Session.of(self.exit_session).close


@dataclass(frozen=True, slots=True)
class StoryOutcome:
    """How one story ended in a run.

    ``terminal_state`` is the playbook's last state, ``"DISMISSED"`` for a
    story blocked by a live playbook on its symbol (reason
    ``blocked_open``), or ``"SKIPPED"`` for a story without a path
    (``skip`` names the loader's reason). ``trade`` is None for a run that
    stops at the entry (Stage A): ``entry_decided_at`` and
    ``entry_bar_ts`` then say whether and when it entered.
    """

    story_id: str
    symbol: str
    session: date
    terminal_state: str
    reason: str
    skip: str | None
    triggered_at: datetime | None
    armed_at: datetime | None
    trade: TradeRecord | None
    transitions: tuple[tuple[datetime, str, str, str], ...] = ()  # (at, from, to, reason)
    start_at: datetime | None = None
    nsn_at: datetime | None = None
    entry_decided_at: datetime | None = None
    entry_bar_ts: datetime | None = None
    legs: tuple[Leg, ...] = ()
    intents: tuple[tuple[datetime, Intent], ...] = ()  # only with keep_transitions

    @property
    def entered(self) -> bool:
        return self.entry_bar_ts is not None


def _json_default(o: object) -> object:
    if isinstance(o, datetime | date):
        return o.isoformat()
    if isinstance(o, frozenset | set):
        return sorted(o)  # type: ignore[type-var]
    return str(o)


def canonical(outcome: StoryOutcome) -> str:
    """A deterministic JSON form of an outcome (floats by exact ``repr``)."""
    return json.dumps(dataclasses.asdict(outcome), sort_keys=True, default=_json_default)


def outcomes_sha256(outcomes: Sequence[StoryOutcome]) -> str:
    """sha256 over the canonical records sorted by story id: equal for any worker count."""
    digest = hashlib.sha256()
    for line in sorted(canonical(o) for o in outcomes):
        digest.update(line.encode())
        digest.update(b"\n")
    return digest.hexdigest()


# ── sinks ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RunInfo:
    """What the driver knows about a run when it starts."""

    window: str
    window_end: date
    feed: str
    stop_at: str
    playbook: str
    playbook_version: str
    sim: Mapping[str, object] = field(default_factory=dict)


class OutcomeSink(Protocol):
    async def begin(self, info: RunInfo) -> str:
        """Open the run; returns its id."""
        ...

    async def write(self, outcomes: Sequence[StoryOutcome]) -> None: ...
    async def finish(self, summary: Mapping[str, object]) -> None: ...


class MemorySink:
    """Keeps every outcome in memory (tests, Stage A counting, the atlas)."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.info: RunInfo | None = None
        self.outcomes: list[StoryOutcome] = []
        self.summary: Mapping[str, object] | None = None

    async def begin(self, info: RunInfo) -> str:
        self.info = info
        return self.run_id

    async def write(self, outcomes: Sequence[StoryOutcome]) -> None:
        self.outcomes.extend(outcomes)

    async def finish(self, summary: Mapping[str, object]) -> None:
        self.summary = summary


_TRADE_FIELDS = tuple(f.name for f in dataclasses.fields(TradeRecord))


def _num(x: float) -> float | None:
    """NaN and infinities stored as NULL."""
    return x if math.isfinite(x) else None


def story_row(run_id: str, o: StoryOutcome) -> dict[str, Any]:
    return {
        "run_id": uuid.UUID(run_id),
        "story_id": o.story_id,
        "symbol": o.symbol,
        "session": o.session,
        "start_at": o.start_at,
        "nsn_at": o.nsn_at,
        "terminal_state": o.terminal_state,
        "reason": o.reason,
        "skip": o.skip,
        "triggered_at": o.triggered_at,
        "armed_at": o.armed_at,
        "entry_decided_at": o.entry_decided_at,
        "entry_bar_ts": o.entry_bar_ts,
    }


def trade_row(run_id: str, t: TradeRecord, legs: Sequence[Leg]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name in _TRADE_FIELDS:
        value = getattr(t, name)
        if isinstance(value, float):
            value = _num(value)
        row[name] = value
    row["run_id"] = uuid.UUID(run_id)
    row["flags"] = list(t.flags)
    row["legs"] = [
        {
            "day": leg.day.isoformat(),
            "a": _num(leg.a),
            "z": _num(leg.z),
            "spy_a": _num(leg.spy_a),
            "spy_z": _num(leg.spy_z),
            "cost": _num(leg.cost),
        }
        for leg in legs
    ]
    return row


class PgOutcomeSink:
    """Writes ``hb_playbook_run``, ``hb_playbook_story`` and ``hb_playbook_trade``.

    ``config`` is the trial configuration the run belongs to; ``config_hash``
    is its ledger hash (``quant_trials.config_hash``). The simulator's
    constants are stored next to it under ``config["sim"]``. Only runs with
    ``mode`` shadow, paper or live are meant for backups; ``sim`` rows can be
    recomputed.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        mode: str = "sim",
        cell: str = "",
        config: Mapping[str, Any] | None = None,
        prereg_trial_id: int | None = None,
        code_sha: str | None = None,
        run_id: str | None = None,
    ) -> None:
        if mode not in ("sim", "shadow", "paper", "live"):
            raise ValueError(f"unknown mode {mode!r}")
        self._engine = engine
        self._mode = mode
        self._cell = cell
        self._config = dict(config or {})
        self._prereg = prereg_trial_id
        self._code_sha = code_sha
        self.run_id = run_id or str(uuid.uuid4())

    async def begin(self, info: RunInfo) -> str:
        from halabot.platform.db import playbook_run
        from halal_trader.db.repos.quant_trials import config_hash

        async with self._engine.begin() as conn:
            await conn.execute(
                insert(playbook_run).values(
                    run_id=uuid.UUID(self.run_id),
                    created_at=datetime.now(UTC),
                    mode=self._mode,
                    playbook=info.playbook,
                    playbook_version=info.playbook_version,
                    cell=self._cell,
                    feed=info.feed,
                    window=info.window,
                    window_end=info.window_end,
                    stop_at=info.stop_at,
                    config={**self._config, "sim": dict(info.sim)},
                    config_hash=config_hash(self._config),
                    prereg_trial_id=self._prereg,
                    code_sha=self._code_sha,
                )
            )
        return self.run_id

    async def write(self, outcomes: Sequence[StoryOutcome]) -> None:
        from halabot.platform.db import playbook_story, playbook_trade

        if not outcomes:
            return
        stories = [story_row(self.run_id, o) for o in outcomes]
        trades = [trade_row(self.run_id, o.trade, o.legs) for o in outcomes if o.trade is not None]
        async with self._engine.begin() as conn:
            await conn.execute(insert(playbook_story), stories)
            if trades:
                await conn.execute(insert(playbook_trade), trades)

    async def finish(self, summary: Mapping[str, object]) -> None:
        from halabot.platform.db import playbook_run

        async with self._engine.begin() as conn:
            await conn.execute(
                update(playbook_run)
                .where(playbook_run.c.run_id == uuid.UUID(self.run_id))
                .values(summary=json.loads(json.dumps(summary, default=_json_default)))
            )


# ── the daily book ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DailyBook:
    days: list[date]
    returns: NDArray[np.float64]  # value-weighted stock legs, net of costs, / NAV_{d-1}
    benchmark: NDArray[np.float64]  # exposure-matched SPY legs
    exposure: NDArray[np.float64]  # sum of the open positions' values at d-1 / NAV_{d-1}
    skipped_full: int
    skipped_loss_limit: int


def _entry_day(t: TradeRecord, legs: Sequence[Leg]) -> date:
    return legs[0].day if legs else t.session


@dataclass(slots=True)
class _Held:
    """An accepted trade and its marked value (the stake, compounded by each leg so far)."""

    trade: TradeRecord
    legs: dict[date, Leg]
    value: float


def daily_book(
    trades: Sequence[TradeRecord],
    legs: Mapping[str, Sequence[Leg]],
    sessions: Sequence[date],
    *,
    slots: int = 8,
    daily_loss_limit: float = 0.02,
) -> DailyBook:
    """The calendar-time book of ``trades`` over ``sessions`` (the window's sessions).

    Days run from the window's first session to the last accepted exit;
    days without a position are 0. Each accepted trade starts at
    NAV_{d-1}/``slots`` and is marked to market by its legs. The loss limit
    counts what trades closed earlier on day d lost **on d** (their earlier
    days are already in NAV_{d-1}).
    """
    days_all = sorted(sessions)
    empty = np.zeros(0, dtype=np.float64)
    if not trades or not days_all:
        return DailyBook([], empty, empty, empty, 0, 0)
    order = sorted(trades, key=lambda t: (t.entry_decided_at, t.story_id))
    by_day: dict[date, list[TradeRecord]] = {}
    for t in order:
        by_day.setdefault(_entry_day(t, legs.get(t.story_id, ())), []).append(t)

    horizon = max(t.exit_session for t in trades)
    nav = 1.0
    book: list[_Held] = []  # accepted trades not exited before the current day
    skipped_full = skipped_loss = 0
    rets: list[float] = []
    bench: list[float] = []
    expo: list[float] = []
    last_exit: date | None = None
    out_days: list[date] = []
    for d in days_all:
        if d > horizon:
            break
        nav_prev = nav
        book = [h for h in book if h.trade.exit_session >= d]
        for t in by_day.get(d, []):
            decided = t.entry_decided_at
            open_now = sum(
                1 for h in book if h.trade.entry_decided_at <= decided < h.trade.exit_time
            )
            realised = sum(
                h.value * h.legs[d].stock  # its value at d-1 times its move on d
                for h in book
                if h.trade.exit_session == d and h.trade.exit_time <= decided and d in h.legs
            )
            if open_now >= slots:
                skipped_full += 1
            elif realised <= -daily_loss_limit * nav_prev:
                skipped_loss += 1
            else:
                own = {leg.day: leg for leg in legs.get(t.story_id, ())}
                book.append(_Held(t, own, nav_prev / slots))
                last_exit = max(last_exit or t.exit_session, t.exit_session)
        r = b = e = 0.0
        for h in book:
            leg = h.legs.get(d)
            if leg is None:
                continue
            r += h.value * leg.stock
            b += h.value * leg.spy
            e += h.value
            h.value *= 1.0 + leg.stock  # marked to market for the next day
        out_days.append(d)
        rets.append(r / nav_prev)
        bench.append(b / nav_prev)
        expo.append(e / nav_prev)
        nav = nav_prev * (1.0 + r / nav_prev)
    if last_exit is not None:
        keep = [i for i, d in enumerate(out_days) if d <= last_exit]
        out_days = [out_days[i] for i in keep]
        rets = [rets[i] for i in keep]
        bench = [bench[i] for i in keep]
        expo = [expo[i] for i in keep]
    else:
        out_days, rets, bench, expo = [], [], [], []
    return DailyBook(
        out_days,
        np.asarray(rets, dtype=np.float64),
        np.asarray(bench, dtype=np.float64),
        np.asarray(expo, dtype=np.float64),
        skipped_full,
        skipped_loss,
    )


__all__ = [
    "DailyBook",
    "Leg",
    "MemorySink",
    "OutcomeSink",
    "PgOutcomeSink",
    "RunInfo",
    "StoryOutcome",
    "TradeRecord",
    "canonical",
    "daily_book",
    "outcomes_sha256",
    "story_row",
    "trade_row",
]
