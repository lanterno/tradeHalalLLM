"""Writing events and their scores (the event-driven research plan, Phase 0).

The news reactor records every headline it sees and every score it
computes, so the live classifier's judgements can be measured against
what prices did next. They are the only LLM evidence the model cannot
have seen in training: they postdate it.

Recording is best effort. A database hiccup is logged and dropped; it
never delays or blocks the reactor.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Score:
    scorer: str
    score: float
    tag: str | None = None
    rationale: str | None = None


@dataclass(frozen=True, slots=True)
class EventRecord:
    source: str
    source_id: str
    kind: str
    symbol: str
    published_at: datetime
    seen_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    score: Score | None = None


class EventRecorder:
    def __init__(self, engine: AsyncEngine, *, raise_errors: bool = False) -> None:
        self._engine = engine
        # Backfills raise (a failed day must not be marked done); the live
        # reactor never does.
        self._raise = raise_errors

    async def record(self, records: Sequence[EventRecord]) -> int:
        """Store events (first sighting wins) and their scores; returns events written."""
        if not records:
            return 0
        written = 0
        try:
            async with self._engine.begin() as conn:
                for r in records:
                    row = (
                        await conn.execute(
                            text(
                                "INSERT INTO events (source, source_id, kind, symbol, "
                                "published_at, seen_at, payload) VALUES (:src, :sid, :k, :sym, "
                                ":pub, :seen, CAST(:p AS JSONB)) ON CONFLICT ON CONSTRAINT "
                                "uq_events_source_item DO UPDATE SET source = EXCLUDED.source "
                                "RETURNING id, (xmax = 0) AS inserted"
                            ),
                            {
                                "src": r.source,
                                "sid": r.source_id,
                                "k": r.kind,
                                "sym": r.symbol,
                                "pub": r.published_at,
                                "seen": r.seen_at,
                                "p": json.dumps(r.payload),
                            },
                        )
                    ).one()
                    written += int(bool(row.inserted))
                    if r.score is not None:
                        await conn.execute(
                            text(
                                "INSERT INTO event_scores (event_id, scorer, score, tag, "
                                "rationale, scored_at) VALUES (:e, :s, :v, :t, :r, :at) "
                                "ON CONFLICT ON CONSTRAINT uq_event_scores_event_scorer DO NOTHING"
                            ),
                            {
                                "e": row.id,
                                "s": r.score.scorer,
                                "v": r.score.score,
                                "t": r.score.tag,
                                "r": (r.score.rationale or "")[:500] or None,
                                "at": datetime.now(UTC),
                            },
                        )
        except Exception as exc:  # noqa: BLE001 -- recording must never block the reactor
            if self._raise:
                raise
            logger.warning("event store: %d record(s) dropped: %r", len(records), exc)
            return 0
        return written
