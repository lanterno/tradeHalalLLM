"""Cross-process liveness: components upsert a heartbeat row, readers age it.

The stock bot, the shadow engine and the dashboard run in separate
containers, so in-process state (the old RuntimeView) cannot say whether the
bot is alive. Each long-running component calls :func:`beat`; the web and the
home stack's health probe call :func:`read_beats` and judge staleness with
:data:`STALE_AFTER`.

:func:`beat` never raises: a heartbeat that can take down the thing it
reports on would be worse than none. A failed write shows up the way it
should -- as a stale heartbeat.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# Component names. One process may own several.
STOCK_PROCESS = "stock.process"  # the bot's run loop is turning
STOCK_CYCLE = "stock.cycle"  # a trading cycle finished (only during market hours)
STOCK_MONITOR = "stock.monitor"  # the SL/TP monitor completed a tick
STOCK_LEDGER = "stock.ledger"  # the after-close broker-ledger sync succeeded
RESEARCH = "research.daily"  # the evening run: bars, weekly screen, forward books

# How old a beat may get before the component counts as dead. Generous
# multiples of each component's cadence, so a slow tick is not an alarm.
STALE_AFTER: dict[str, timedelta] = {
    STOCK_PROCESS: timedelta(minutes=3),  # beats every 60 s
    STOCK_MONITOR: timedelta(minutes=5),  # ticks every 30 s while it has work
    STOCK_CYCLE: timedelta(minutes=45),  # every 15 min, market hours only
    STOCK_LEDGER: timedelta(days=4),  # once a trading day; spans a long weekend
    RESEARCH: timedelta(days=4),
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
    import json

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


async def read_beats(engine: AsyncEngine) -> dict[str, Beat]:
    """Every component's latest beat, keyed by component name."""
    async with engine.connect() as conn:
        rows = await conn.execute(text("SELECT component, beat_at, detail FROM heartbeats"))
        return {r.component: Beat(r.component, r.beat_at, r.detail) for r in rows}


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
