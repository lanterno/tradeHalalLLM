"""Sharia exception queue.

When a screener returns "doubtful" or a new instrument lacks a ruling,
we don't want to block trading on every one — but we also can't just
guess. The right answer is a queue: pending entries land here with the
LLM's preliminary fiqh reasoning, the operator approves or rejects via
the dashboard, and decisions are logged for future learning.

Design choices:

* JSON sidecar persistence for now — low write rate, operator-driven.
  Promote to a Postgres table when concurrent multi-bot review lands.
* Status FSM: ``pending`` → ``approved`` | ``rejected`` | ``deferred``.
  Decided entries are kept (not deleted) so the screener can learn from
  past rulings.
* Idempotent on ``(instrument, kind)`` — a re-screening of the same
  symbol updates the existing entry rather than spamming the queue.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


ExceptionStatus = Literal["pending", "approved", "rejected", "deferred"]


@dataclass
class ExceptionEntry:
    """One pending Sharia ruling."""

    entry_id: str  # stable: "<instrument>:<kind>"
    instrument: str
    kind: str  # "new_token" | "ambiguous_derivative" | "doubtful_screen" | ...
    reasoning: str  # LLM's preliminary case
    status: ExceptionStatus = "pending"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    decided_at: str | None = None
    decided_by: str = ""
    operator_note: str = ""


@dataclass
class ExceptionQueue:
    """Append-only sidecar of Sharia exception entries."""

    path: Path
    entries: dict[str, ExceptionEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("exception queue unreadable: %s — starting fresh", exc)
            return
        for k, v in raw.get("entries", {}).items():
            try:
                self.entries[k] = ExceptionEntry(**v)
            except Exception:  # noqa: BLE001
                continue

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(
                {"entries": {k: asdict(v) for k, v in self.entries.items()}},
                indent=2,
                sort_keys=True,
            )
        )

    @staticmethod
    def _key(instrument: str, kind: str) -> str:
        return f"{instrument.upper()}:{kind}"

    def add(self, *, instrument: str, kind: str, reasoning: str) -> ExceptionEntry:
        """Add a new pending entry (idempotent on instrument+kind)."""
        key = self._key(instrument, kind)
        existing = self.entries.get(key)
        if existing is not None and existing.status == "pending":
            # Refresh reasoning text but keep created_at and status
            existing.reasoning = reasoning
            self._save()
            return existing
        entry = ExceptionEntry(
            entry_id=key,
            instrument=instrument.upper(),
            kind=kind,
            reasoning=reasoning,
        )
        self.entries[key] = entry
        self._save()
        return entry

    def decide(
        self,
        entry_id: str,
        *,
        status: ExceptionStatus,
        decided_by: str = "",
        operator_note: str = "",
    ) -> bool:
        """Apply an operator decision; returns False if entry was unknown."""
        entry = self.entries.get(entry_id)
        if entry is None:
            return False
        if status not in ("approved", "rejected", "deferred"):
            raise ValueError(f"invalid decision status: {status!r}")
        entry.status = status
        entry.decided_at = datetime.now(UTC).isoformat()
        entry.decided_by = decided_by
        entry.operator_note = operator_note
        self._save()
        return True

    def pending(self) -> list[ExceptionEntry]:
        return [e for e in self.entries.values() if e.status == "pending"]

    def by_status(self, status: ExceptionStatus) -> list[ExceptionEntry]:
        return [e for e in self.entries.values() if e.status == status]

    def all(self) -> list[ExceptionEntry]:
        return list(self.entries.values())

    def is_approved(self, instrument: str, kind: str) -> bool:
        """Quick gate: returns True only if an approval exists for this pair."""
        e = self.entries.get(self._key(instrument, kind))
        return e is not None and e.status == "approved"


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
