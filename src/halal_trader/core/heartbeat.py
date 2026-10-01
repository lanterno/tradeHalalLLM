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

# How old a beat may get before the component counts as dead. Generous
# multiples of each component's cadence, so a slow tick is not an alarm.
STALE_AFTER: dict[str, timedelta] = {
    STOCK_PROCESS: timedelta(minutes=3),  # beats every 60 s
    STOCK_MONITOR: timedelta(minutes=5),  # ticks every 30 s while it has work
    STOCK_CYCLE: timedelta(minutes=45),  # every 15 min, market hours only
    STOCK_LEDGER: timedelta(days=4),  # once a trading day; spans a long weekend
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
