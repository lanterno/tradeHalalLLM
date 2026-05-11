"""Committee transcript audit log — Round-5 Wave 8.E.

Every committee debate produces a structured transcript: which agent
spoke, in what order, with what stance + confidence + rationale, and
what the verdict was. Without persistence, the operator cannot review
why a trade was made — auditability is non-negotiable for a regulated
halal platform.

This module is the **append-only transcript log + searcher**. It
composes with `core/merkle_audit.py` (Wave 19.H) so a chain of
transcripts can be Merkle-anchored for tamper-evidence.

Pinned semantics:

- **Append-only.** No `update`/`delete` API; superseding transcripts
  are appended with a `supersedes_id` reference. This makes the
  store a true audit log, not a cache.
- **Immutable Transcript dataclass** — once written, the entry's
  hash never changes.
- **Hash-chain** — each entry's `prev_hash` points to the previous
  entry's hash; tamper detection is one walk over the chain.
- **Searchable** by ticker / role / stance / date range / verdict.
- **Pure-Python deterministic.** `InMemoryTranscriptStore` for tests;
  the production replacement uses Postgres + Merkle anchoring (one
  layer up).
- **No-secret-leak pin** on render — rationale text is *truncated*,
  not redacted (unlike the contradiction-detector renderer which
  drops it entirely).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from halal_trader.core.llm_committee import AgentRole, AgentVote, Stance


def _hash_payload(payload: dict[str, object]) -> str:
    """Stable sha256-hex over the canonical JSON of a dict."""
    j = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode()).hexdigest()


@dataclass(frozen=True)
class TranscriptEntry:
    """One immutable row in the transcript log."""

    transcript_id: str
    ticker: str
    decision_at: datetime
    debate_round: int
    final_stance: Stance
    final_confidence: float
    veto_invoked: bool
    votes: tuple[AgentVote, ...]
    prev_hash: str
    """Hash of the previous transcript in the chain (empty for the
    first entry)."""
    supersedes_id: str = ""
    """If non-empty, this transcript supersedes a prior one (e.g. an
    operator override produced a new verdict)."""

    def __post_init__(self) -> None:
        if not self.transcript_id or not self.transcript_id.strip():
            raise ValueError("transcript_id must be non-empty")
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")
        if self.debate_round < 1:
            raise ValueError("debate_round must be ≥ 1")
        if not 0.0 <= self.final_confidence <= 1.0:
            raise ValueError("final_confidence must be in [0, 1]")
        if not self.votes:
            raise ValueError("transcript must have at least one vote")

    def payload_for_hash(self) -> dict[str, object]:
        """Canonical dict for hashing. `prev_hash` is included so the
        chain is tamper-evident."""
        return {
            "transcript_id": self.transcript_id,
            "ticker": self.ticker,
            "decision_at": self.decision_at.isoformat(),
            "debate_round": self.debate_round,
            "final_stance": self.final_stance.value,
            "final_confidence": self.final_confidence,
            "veto_invoked": self.veto_invoked,
            "votes": [
                {
                    "role": v.role.value,
                    "stance": v.stance.value,
                    "confidence": v.confidence,
                    "rationale_hash": hashlib.sha256(v.rationale.encode()).hexdigest()[:16],
                }
                for v in self.votes
            ],
            "prev_hash": self.prev_hash,
            "supersedes_id": self.supersedes_id,
        }

    def entry_hash(self) -> str:
        return _hash_payload(self.payload_for_hash())


class TranscriptStore(Protocol):
    """Protocol for the persistent transcript log."""

    def append(self, entry: TranscriptEntry) -> None: ...

    def all(self) -> tuple[TranscriptEntry, ...]: ...

    def by_id(self, transcript_id: str) -> TranscriptEntry | None: ...

    def search(
        self,
        *,
        ticker: str | None = None,
        role: AgentRole | None = None,
        stance: Stance | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> tuple[TranscriptEntry, ...]: ...

    def latest_hash(self) -> str: ...


class InMemoryTranscriptStore:
    """List-backed transcript store with the hash-chain enforced."""

    def __init__(self) -> None:
        self._entries: list[TranscriptEntry] = []

    def append(self, entry: TranscriptEntry) -> None:
        # Pin the chain: entry.prev_hash must equal the previous entry's hash.
        expected_prev = self.latest_hash()
        if entry.prev_hash != expected_prev:
            raise ValueError(
                f"prev_hash mismatch: expected {expected_prev!r}, got {entry.prev_hash!r}"
            )
        # Pin: transcript_id must be unique.
        if any(e.transcript_id == entry.transcript_id for e in self._entries):
            raise ValueError(f"transcript_id {entry.transcript_id} already present")
        # If supersedes_id is set, the prior must exist + not have been superseded.
        if entry.supersedes_id:
            prior = self.by_id(entry.supersedes_id)
            if prior is None:
                raise ValueError(f"supersedes_id {entry.supersedes_id} does not exist")
            if any(e.supersedes_id == entry.supersedes_id for e in self._entries):
                raise ValueError(f"transcript {entry.supersedes_id} already superseded")
        self._entries.append(entry)

    def all(self) -> tuple[TranscriptEntry, ...]:
        return tuple(self._entries)

    def by_id(self, transcript_id: str) -> TranscriptEntry | None:
        for e in self._entries:
            if e.transcript_id == transcript_id:
                return e
        return None

    def search(
        self,
        *,
        ticker: str | None = None,
        role: AgentRole | None = None,
        stance: Stance | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> tuple[TranscriptEntry, ...]:
        out: list[TranscriptEntry] = []
        for e in self._entries:
            if ticker is not None and e.ticker != ticker:
                continue
            if stance is not None and e.final_stance is not stance:
                continue
            if date_from is not None and e.decision_at < date_from:
                continue
            if date_to is not None and e.decision_at > date_to:
                continue
            if role is not None:
                if not any(v.role is role for v in e.votes):
                    continue
            out.append(e)
        return tuple(out)

    def latest_hash(self) -> str:
        if not self._entries:
            return ""
        return self._entries[-1].entry_hash()


def verify_chain(entries: Sequence[TranscriptEntry]) -> bool:
    """True iff each entry's `prev_hash` matches the previous entry's hash.

    Pure-functional verifier — operators can run this against the
    persisted log to detect tampering without touching the store.
    """
    prev = ""
    for e in entries:
        if e.prev_hash != prev:
            return False
        prev = e.entry_hash()
    return True


def supersede(
    store: TranscriptStore,
    *,
    new_id: str,
    superseded_id: str,
    decision_at: datetime,
    debate_round: int,
    final_stance: Stance,
    final_confidence: float,
    veto_invoked: bool,
    votes: tuple[AgentVote, ...],
) -> TranscriptEntry:
    """Append a transcript that supersedes a prior one (operator override).

    Convenience wrapper that pulls the prior's ticker + the latest_hash
    so callers don't have to thread either explicitly.
    """
    prior = store.by_id(superseded_id)
    if prior is None:
        raise ValueError(f"superseded_id {superseded_id} not found")
    new_entry = TranscriptEntry(
        transcript_id=new_id,
        ticker=prior.ticker,
        decision_at=decision_at,
        debate_round=debate_round,
        final_stance=final_stance,
        final_confidence=final_confidence,
        veto_invoked=veto_invoked,
        votes=votes,
        prev_hash=store.latest_hash(),
        supersedes_id=superseded_id,
    )
    store.append(new_entry)
    return new_entry


def _truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[:n] + "…"


def render_entry(entry: TranscriptEntry, *, rationale_chars: int = 80) -> str:
    """Operator-readable summary of a single transcript entry.

    Rationale text is truncated (not redacted) — operators need a
    glimpse to recall what the agent argued. The full text lives in
    the store and is retrievable by `transcript_id`.
    """
    super_str = f" supersedes={entry.supersedes_id}" if entry.supersedes_id else ""
    veto_str = " [VETO]" if entry.veto_invoked else ""
    head = (
        f"📜 {entry.transcript_id} {entry.ticker} "
        f"{entry.decision_at.isoformat()} "
        f"r{entry.debate_round} → {entry.final_stance.value} "
        f"c={entry.final_confidence:.2f}{veto_str}{super_str}"
    )
    lines = [head]
    for v in entry.votes:
        rat = _truncate(v.rationale, rationale_chars)
        lines.append(
            f"  • {v.role.value}: {v.stance.value} c={v.confidence:.2f}"
            + (f" — {rat}" if rat else "")
        )
    return "\n".join(lines)
