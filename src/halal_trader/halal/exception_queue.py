"""Sharia exception queue.

When a screener returns "doubtful" or a new instrument lacks a ruling,
we don't want to block trading on every one — but we also can't just
guess. The right answer is a queue: pending entries land here with the
LLM's preliminary fiqh reasoning, the operator approves or rejects via
the dashboard, and decisions are logged for future learning.

Design choices:

* DB-backed (``sharia_exceptions`` table) — operator-driven low rate,
  but ops surfaces (CLI / dashboard / bot screener) all consume the
  same source of truth without file-locking races.
* Status FSM: ``pending`` → ``approved`` | ``rejected`` | ``deferred``.
  Decided entries are kept (not deleted) so the screener can learn from
  past rulings.
* Idempotent on ``(instrument, kind)`` — a re-screening of the same
  symbol updates the existing entry rather than spamming the queue.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel import select

from halal_trader.db.models import ShariaExceptionRow

logger = logging.getLogger(__name__)


ExceptionStatus = Literal["pending", "approved", "rejected", "deferred"]


@dataclass
class ExceptionEntry:
    """One pending Sharia ruling."""

    entry_id: str  # stable: "<instrument>:<kind>"
    instrument: str
    kind: str
    reasoning: str
    status: ExceptionStatus = "pending"
    created_at: str = ""  # ISO timestamp
    decided_at: str | None = None
    decided_by: str = ""
    operator_note: str = ""


def _row_to_entry(row: ShariaExceptionRow) -> ExceptionEntry:
    return ExceptionEntry(
        entry_id=row.entry_id,
        instrument=row.instrument,
        kind=row.kind,
        reasoning=row.reasoning,
        status=row.status,  # type: ignore[arg-type]
        created_at=row.created_at.isoformat() if row.created_at else "",
        decided_at=row.decided_at.isoformat() if row.decided_at else None,
        decided_by=row.decided_by,
        operator_note=row.operator_note,
    )


@dataclass
class ExceptionQueue:
    """Postgres-backed queue of Sharia exception entries."""

    engine: AsyncEngine

    @property
    def _sm(self) -> "async_sessionmaker[Any]":
        return async_sessionmaker(self.engine, expire_on_commit=False)

    @staticmethod
    def _key(instrument: str, kind: str) -> str:
        return f"{instrument.upper()}:{kind}"

    async def add(self, *, instrument: str, kind: str, reasoning: str) -> ExceptionEntry:
        """Add a new pending entry (idempotent on instrument+kind).

        - Pending row of the same key: update reasoning, keep created_at.
        - Already-decided row of the same key: replace with a fresh
          pending row (operator can re-approve / re-reject after a
          re-screening).
        """
        key = self._key(instrument, kind)
        async with self._sm() as s:
            existing = await s.get(ShariaExceptionRow, key)
            if existing is not None and existing.status == "pending":
                existing.reasoning = reasoning
                s.add(existing)
                await s.commit()
                return _row_to_entry(existing)
            if existing is not None:
                # decided previously — overwrite with a fresh pending entry
                existing.reasoning = reasoning
                existing.status = "pending"
                existing.created_at = datetime.now(UTC)
                existing.decided_at = None
                existing.decided_by = ""
                existing.operator_note = ""
                s.add(existing)
                await s.commit()
                return _row_to_entry(existing)
            row = ShariaExceptionRow(
                entry_id=key,
                instrument=instrument.upper(),
                kind=kind,
                reasoning=reasoning,
            )
            s.add(row)
            await s.commit()
            await s.refresh(row)
            return _row_to_entry(row)

    async def decide(
        self,
        entry_id: str,
        *,
        status: ExceptionStatus,
        decided_by: str = "",
        operator_note: str = "",
    ) -> bool:
        """Apply an operator decision; returns False if entry was unknown."""
        if status not in ("approved", "rejected", "deferred"):
            raise ValueError(f"invalid decision status: {status!r}")
        async with self._sm() as s:
            row = await s.get(ShariaExceptionRow, entry_id)
            if row is None:
                return False
            row.status = status
            row.decided_at = datetime.now(UTC)
            row.decided_by = decided_by
            row.operator_note = operator_note
            s.add(row)
            await s.commit()
            return True

    async def all(self) -> list[ExceptionEntry]:
        async with self._sm() as s:
            result = await s.execute(select(ShariaExceptionRow))
            rows = result.scalars().all()
        return [_row_to_entry(r) for r in rows]

    async def by_status(self, status: ExceptionStatus) -> list[ExceptionEntry]:
        async with self._sm() as s:
            result = await s.execute(
                select(ShariaExceptionRow).where(ShariaExceptionRow.status == status)
            )
            rows = result.scalars().all()
        return [_row_to_entry(r) for r in rows]

    async def pending(self) -> list[ExceptionEntry]:
        return await self.by_status("pending")

    async def is_approved(self, instrument: str, kind: str) -> bool:
        """Quick gate: returns True only if an approval exists for this pair."""
        async with self._sm() as s:
            row = await s.get(ShariaExceptionRow, self._key(instrument, kind))
            return row is not None and row.status == "approved"


def render_summary(entries: Iterable[ExceptionEntry]) -> str:
    """Operator-friendly summary table."""
    entries = list(entries)
    if not entries:
        return "Sharia exception queue: empty"
    lines = ["Sharia exception queue:"]
    for e in entries:
        marker = {"pending": "?", "approved": "✓", "rejected": "✗", "deferred": "…"}[e.status]
        lines.append(f"  {marker} [{e.status}] {e.instrument} ({e.kind}) — {e.reasoning[:60]}")
    return "\n".join(lines)
