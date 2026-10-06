"""Per-asset bar high-water marks, so a restart doesn't re-emit history.

The bar source asks Alpaca for the last ``days`` of bars on every poll and
used to dedup them only in memory. A restart forgot that set, so the first
poll published the whole window again (about 1,000 bars) as live
``observation.bar`` events: 32,630 bar events for 2,728 distinct bars between
2026-10-01 and 10-06. The event log already says which bars were published, so
the newest ``bar_ts`` per asset there is the mark the source resumes from.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.platform.clock import parse_iso
from halabot.platform.db import event_log
from halabot.platform.events import EventType


class BarWatermark(Protocol):
    async def load(self) -> dict[str, datetime]:
        """Latest published bar time per asset."""
        ...


class InMemoryBarWatermark:
    """A fixed set of marks (tests)."""

    def __init__(self, marks: dict[str, datetime] | None = None) -> None:
        self._marks = dict(marks or {})

    async def load(self) -> dict[str, datetime]:
        return dict(self._marks)


class PgBarWatermark:
    """Reads the marks from ``hb_event_log``'s ``observation.bar`` events.

    Looks back ``lookback_days`` of event time (the source's own fetch window
    plus a margin is enough: an older bar can't come back from the fetch).
    The ``bar_ts`` strings are parsed here rather than cast in SQL, so one
    malformed payload costs its own row instead of the whole query.
    """

    def __init__(self, engine: AsyncEngine, *, lookback_days: float) -> None:
        self._engine = engine
        self._lookback_s = lookback_days * 86400.0

    async def load(self) -> dict[str, datetime]:
        t = event_log
        cutoff = sa.text("now() - make_interval(secs => :s)").bindparams(s=self._lookback_s)
        stmt = (
            sa.select(t.c.asset, t.c.payload["bar_ts"].astext)
            .distinct()
            .where(t.c.type == str(EventType.OBSERVATION_BAR), t.c.ts >= cutoff)
        )
        marks: dict[str, datetime] = {}
        async with self._engine.connect() as conn:
            for asset, raw in await conn.execute(stmt):
                ts = parse_iso(raw)
                if asset is None or ts is None:
                    continue
                if asset not in marks or ts > marks[asset]:
                    marks[asset] = ts
        return marks
