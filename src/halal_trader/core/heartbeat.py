"""Cross-process liveness: components upsert a heartbeat row, readers age it.

The stock bot, the shadow engine and the dashboard run in separate
containers, so in-process state (the old RuntimeView) cannot say whether the
bot is alive. Each long-running component calls :func:`beat`; the web and the
home stack's health probe call :func:`read_beats` and judge each row with
:func:`assess`: an age limit (:data:`STALE_AFTER`) for the loops that beat
continuously, the trading calendar (:data:`DAILY_JOBS`) for the once-a-day
jobs, which are stale when the last run they owed is missing -- not when some
fixed number of hours has passed.

:func:`beat` never raises: a heartbeat that can take down the thing it
reports on would be worse than none. A failed write shows up the way it
should -- as a stale heartbeat.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.market_hours import MARKET_TZ, effective_close_time, is_trading_day

logger = logging.getLogger(__name__)

# Component names. One process may own several.
STOCK_PROCESS = "stock.process"  # the bot's run loop is turning
STOCK_CYCLE = "stock.cycle"  # a trading cycle finished (only during market hours)
STOCK_MONITOR = "stock.monitor"  # the SL/TP monitor completed a tick
STOCK_LEDGER = "stock.ledger"  # the after-close broker-ledger sync succeeded
STOCK_EOD = "stock.eod"  # the end-of-day routine finished
RECOMMENDATION = "recommendation.daily"  # the pre-market advisory pick was made
MARKET_SNAPSHOT = "market.snapshot"  # the session's per-minute account/benchmark values
RESEARCH = "research.daily"  # the evening run: bars, weekly screen, forward books
CORE_TRADE = "core.trade"  # the core portfolio's 15:40 run finished (traded, held or refused)
WEEKLY_DIGEST = "digest.weekly"  # Friday's Telegram summary went out
SHADOW_PROCESS = "shadow.process"  # the halabot shadow engine's event loop is turning
WATCHDOG = "web.watchdog"  # the web's watchdog ran; its detail is what it has alerted
# Not a liveness row: the "already alerted" ledger of claim_once().
ALERT_MARKS = "alerts.sent"

# How old a beat may get before the component counts as dead. Generous
# multiples of each component's cadence, so a slow tick is not an alarm.
STALE_AFTER: dict[str, timedelta] = {
    STOCK_PROCESS: timedelta(minutes=3),  # beats every 60 s
    SHADOW_PROCESS: timedelta(minutes=3),  # beats every 60 s
    STOCK_MONITOR: timedelta(minutes=5),  # ticks every 30 s
    STOCK_CYCLE: timedelta(minutes=45),  # every 15 min, market hours only
    MARKET_SNAPSHOT: timedelta(minutes=5),  # every minute, session hours only
    WATCHDOG: timedelta(minutes=15),  # every 5 min
    # Coarse floors for Beat.is_stale; assess() judges these by the calendar.
    STOCK_LEDGER: timedelta(days=4),
    RESEARCH: timedelta(days=4),
}

# Loops that beat all the time: a missing row is as bad as a stale one.
CONTINUOUS = (STOCK_PROCESS, SHADOW_PROCESS, STOCK_MONITOR)


@dataclass(frozen=True, slots=True)
class DailyJob:
    """A job the bot runs once per trading day, at a fixed US/Eastern time."""

    at: time
    grace: timedelta  # how long after ``at`` its beat may take to land
    early_close_at: time | None = None  # its time on a 13:00-close day, if different
    weekday: int | None = None  # only on this weekday (0 = Monday); None = every trading day


# Mirrors the bot's cron (trading/scheduler.py). A job's beat is stale when
# the newest run it owed -- on a trading day, past its time plus grace --
# left no beat at or after its scheduled time.
DAILY_JOBS: dict[str, DailyJob] = {
    RECOMMENDATION: DailyJob(time(9, 5), timedelta(hours=1)),
    STOCK_EOD: DailyJob(time(15, 50), timedelta(minutes=30), early_close_at=time(12, 50)),
    STOCK_LEDGER: DailyJob(time(16, 30), timedelta(hours=1)),
    RESEARCH: DailyJob(time(20, 30), timedelta(hours=3)),
    CORE_TRADE: DailyJob(time(15, 40), timedelta(minutes=45), early_close_at=time(12, 40)),
    WEEKLY_DIGEST: DailyJob(time(17, 15), timedelta(hours=1), weekday=4),
}

_UPSERT = text(
    """
    INSERT INTO heartbeats (component, beat_at, detail)
    VALUES (:component, :beat_at, CAST(:detail AS JSONB))
    ON CONFLICT (component) DO UPDATE
        SET beat_at = EXCLUDED.beat_at, detail = EXCLUDED.detail
    """
)


@dataclass(frozen=True, slots=True)
class Beat:
    component: str
    beat_at: datetime
    detail: dict[str, Any] | None

    def age(self, now: datetime | None = None) -> timedelta:
        return (now or datetime.now(UTC)) - self.beat_at

    def is_stale(self, now: datetime | None = None) -> bool:
        limit = STALE_AFTER.get(self.component)
        return limit is not None and self.age(now) > limit


async def beat(
    engine: AsyncEngine | None,
    component: str,
    detail: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> None:
    """Record that ``component`` is alive. Never raises."""
    if engine is None:
        return
    try:
        async with engine.begin() as conn:
            await conn.execute(
                _UPSERT,
                {
                    "component": component,
                    "beat_at": now or datetime.now(UTC),
                    "detail": json.dumps(detail) if detail is not None else None,
                },
            )
    except Exception as exc:  # noqa: BLE001 -- liveness must not break the bot
        logger.warning("heartbeat %s not recorded: %r", component, exc)


async def beat_forever(engine: AsyncEngine, component: str, *, interval_s: float = 60.0) -> None:
    """Beat ``component`` every ``interval_s`` until cancelled.

    For a process whose main loop is not this package's (the shadow engine):
    run it as a task on that process's event loop, and the beat stops when
    the loop does -- a hung or dead process shows up as a stale row.
    """
    while True:
        await beat(engine, component)
        await asyncio.sleep(interval_s)


async def read_beats(engine: AsyncEngine) -> dict[str, Beat]:
    """Every component's latest beat, keyed by component name."""
    async with engine.connect() as conn:
        rows = await conn.execute(text("SELECT component, beat_at, detail FROM heartbeats"))
        return {r.component: Beat(r.component, r.beat_at, r.detail) for r in rows}


def core_on(beats: dict[str, Beat]) -> bool:
    """Does the bot run the core? Its process beat says so (``{"core": ...}``):
    the core's keys never reach the web. True until the bot has said, so a
    missed core run is never excused by a guess."""
    b = beats.get(STOCK_PROCESS)
    return bool((b.detail or {}).get("core", True)) if b is not None else True


async def core_running(engine: AsyncEngine) -> bool:
    """:func:`core_on`, read from the database."""
    return core_on(await read_beats(engine))


async def cycle_risk(engine: AsyncEngine) -> tuple[dict[str, Any] | None, datetime | None]:
    """The risk snapshot the last trading cycle published, and when."""
    beats = await read_beats(engine)
    cycle = beats.get(STOCK_CYCLE)
    if cycle is None:
        return None, None
    risk = (cycle.detail or {}).get("risk")
    return (risk if isinstance(risk, dict) else None), cycle.beat_at


def bot_liveness(
    beats: dict[str, Beat], *, now: datetime, cycles_due: bool
) -> tuple[bool, str | None]:
    """Is the stock bot alive and working? ``(alive, reason-if-not)``.

    Alive means the process heartbeat is fresh -- and, when trading cycles
    are due (market open long enough for one to have run), that a cycle has
    completed recently. The second part catches a cycle that hangs: the
    process keeps beating while no trading happens, which a cancelling
    deadline would "fix" at the risk of an order placed but never recorded.
    """
    process = beats.get(STOCK_PROCESS)
    if process is None:
        return False, "no process heartbeat on record"
    if process.is_stale(now):
        return False, f"process heartbeat stale ({process.age(now)} old)"
    if cycles_due:
        cycle = beats.get(STOCK_CYCLE)
        if cycle is None or cycle.is_stale(now):
            return False, "no trading cycle completed in the last 45 min of market hours"
    return True, None


# ── judging every row ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Status:
    """One component's verdict.

    ``status`` is one of:

    * ``ok`` -- fresh, or not due;
    * ``stale`` -- too old, or a daily run it owed is missing;
    * ``missing`` -- a continuously-beating component with no row at all;
    * ``disabled`` -- not configured (the core's daily trade without its
      own Alpaca keys): never stale, whatever its age;
    * ``unknown`` -- nothing to judge by (a daily job that never ran yet).
    """

    status: str
    reason: str | None = None

    @property
    def failing(self) -> bool:
        return self.status in ("stale", "missing")


def scheduled_at(job: DailyJob, day: date) -> datetime:
    """When ``job`` runs on ``day`` (a trading day), as an aware datetime."""
    early = job.early_close_at is not None and effective_close_time(day).hour < 16
    at = job.early_close_at if early and job.early_close_at is not None else job.at
    return datetime.combine(day, at, MARKET_TZ)


def last_due(job: DailyJob, now: datetime) -> datetime | None:
    """The newest scheduled run of ``job`` whose beat should have landed by ``now``."""
    day = now.astimezone(MARKET_TZ).date()
    for _ in range(21):  # a few days' closure, or three weeks for a weekly job
        if is_trading_day(day) and (job.weekday is None or day.weekday() == job.weekday):
            at = scheduled_at(job, day)
            if at + job.grace <= now:
                return at
        day -= timedelta(days=1)
    return None


def cycles_due_at(now: datetime) -> bool:
    """Should trading cycles be completing at ``now`` (session open since 10:00 ET)?"""
    et = now.astimezone(MARKET_TZ)
    return is_trading_day(et.date()) and time(10, 0) <= et.time() < effective_close_time(et.date())


def snapshots_due_at(now: datetime) -> bool:
    """Should the per-minute market snapshot be beating at ``now``?

    Its cron runs 09:00-16:59 ET; allow a few minutes after it starts.
    """
    et = now.astimezone(MARKET_TZ)
    return is_trading_day(et.date()) and time(9, 5) <= et.time() < time(17, 0)


def _by_age(b: Beat | None, component: str, now: datetime, *, missing: str) -> Status:
    if b is None:
        return Status(missing, "no heartbeat on record")
    limit = STALE_AFTER[component]
    if b.age(now) > limit:
        minutes = int(b.age(now).total_seconds() // 60)
        return Status("stale", f"last beat {minutes} min ago (limit {limit})")
    return Status("ok")


def assess(
    beats: dict[str, Beat],
    *,
    now: datetime,
    cycles_due: bool,
) -> dict[str, Status]:
    """A verdict for every watched component, and for every other row on record.

    ``cycles_due`` comes from the caller (the web pins it to its own view of
    the market clock); :func:`cycles_due_at` computes it from ``now``.
    """
    out: dict[str, Status] = {}
    core_enabled = core_on(beats)
    for component in CONTINUOUS:
        out[component] = _by_age(beats.get(component), component, now, missing="missing")

    if cycles_due:
        out[STOCK_CYCLE] = _by_age(beats.get(STOCK_CYCLE), STOCK_CYCLE, now, missing="stale")
    else:
        out[STOCK_CYCLE] = Status("ok")

    if snapshots_due_at(now):
        out[MARKET_SNAPSHOT] = _by_age(
            beats.get(MARKET_SNAPSHOT), MARKET_SNAPSHOT, now, missing="missing"
        )
    else:
        out[MARKET_SNAPSHOT] = Status("ok")

    for component, job in DAILY_JOBS.items():
        if component == CORE_TRADE and not core_enabled:
            out[component] = Status("disabled", "the core's Alpaca keys are not set")
            continue
        b = beats.get(component)
        due = last_due(job, now)
        if b is None:
            out[component] = Status("unknown", "has not run yet")
        elif due is not None and b.beat_at < due:
            out[component] = Status(
                "stale", f"missed the {due:%a %d %b %H:%M} ET run (last {b.beat_at:%d %b %H:%M}Z)"
            )
        else:
            out[component] = Status("ok")

    for name, b in beats.items():
        if name not in out and name != ALERT_MARKS:
            out[name] = Status("stale", "too old") if b.is_stale(now) else Status("ok")
    return out


def describe(
    beats: dict[str, Beat], statuses: dict[str, Status], *, now: datetime
) -> dict[str, dict[str, Any]]:
    """The API's view of every heartbeat row (``/api/health``'s ``bot``).

    Each entry: ``beat_at``, ``age_seconds``, ``detail``, ``status`` (see
    :class:`Status`), ``reason`` (why, when not ``ok``) and ``stale``
    (``status`` is ``stale`` or ``missing``; never true for ``disabled``).
    """
    out: dict[str, dict[str, Any]] = {}
    # A watched component with no row yet still gets an entry: a job that has
    # never run must show as "has not run yet", not vanish from the page.
    for name, st in statuses.items():
        if name not in beats:
            out[name] = {
                "beat_at": None,
                "age_seconds": None,
                "stale": st.failing,
                "status": st.status,
                "reason": st.reason,
                "detail": None,
            }
    for name, b in beats.items():
        if name == ALERT_MARKS:
            continue
        st = statuses.get(name, Status("ok"))
        out[name] = {
            "beat_at": b.beat_at.isoformat(),
            "age_seconds": round(b.age(now).total_seconds(), 1),
            "stale": st.failing,
            "status": st.status,
            "reason": st.reason,
            "detail": b.detail,
        }
    return out


# ── once-only alerts, across processes and restarts ──────────────────

_CLAIM = text(
    """
    INSERT INTO heartbeats (component, beat_at, detail)
    VALUES (:row, now(), jsonb_build_object(CAST(:k AS TEXT), now()))
    ON CONFLICT (component) DO UPDATE
        SET beat_at = EXCLUDED.beat_at,
            detail = coalesce(heartbeats.detail, '{}'::jsonb) || EXCLUDED.detail
        WHERE (coalesce(heartbeats.detail, '{}'::jsonb) -> CAST(:k AS TEXT)) IS NULL
    RETURNING 1
    """
)
_PRUNE = text(
    """
    UPDATE heartbeats SET detail = (
        SELECT coalesce(jsonb_object_agg(key, value), '{}'::jsonb)
        FROM jsonb_each(detail)
        WHERE (value #>> '{}')::timestamptz > now() - make_interval(days => :days)
    )
    WHERE component = :row
    """
)


async def claim_once(engine: AsyncEngine, key: str, *, keep_days: int = 40) -> bool:
    """True the first time ``key`` is claimed -- by any process, ever; False after.

    For "alert once a day"-style dedup that has to survive a restart and be
    shared by the bot and the shadow (they bill the same LLM pool). One
    statement, so two processes racing for the same key cannot both win.
    Keys older than ``keep_days`` are dropped. Fails OPEN: if the database
    cannot answer, the caller alerts -- a duplicate beats a missed alert.
    """
    try:
        async with engine.begin() as conn:
            won = (await conn.execute(_CLAIM, {"row": ALERT_MARKS, "k": key})).first()
            if won is not None:
                await conn.execute(_PRUNE, {"row": ALERT_MARKS, "days": keep_days})
        return won is not None
    except Exception as exc:  # noqa: BLE001 -- dedup must not swallow an alert
        logger.warning("alert dedup unavailable (%r); alerting anyway", exc)
        return True
