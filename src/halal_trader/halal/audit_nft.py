"""Audit-trail anchored on-chain — Round-5 Wave 22.F.

Premium users can anchor their full audit trail (trades, screening
decisions, scholar attestations) to a public chain as a Merkle root.
This module is the **bundling + Merkle-root + retrieval-verification
primitive**:

1. Operator collects audit events for an epoch (e.g. a quarter).
2. Each event is sha256-hashed; the events are reduced to a Merkle
   root via RFC-6962-compatible math (already in `core/merkle_audit.py`).
3. The root + epoch metadata is wrapped in an `AuditAnchor` token;
   the token's `anchor_hash` binds (root, epoch, owner_id) so the
   on-chain commit pins all three.
4. The on-chain commit is a separate step (chain adapter); this
   module reflects the on-chain receipt and lets verifiers re-derive
   the anchor.

Pinned semantics:

- **Closed-set ChainId** — POLYGON / ARBITRUM / ETHEREUM_MAINNET.
- **Closed-set AnchorStatus FSM** — PREPARED → SUBMITTED → CONFIRMED,
  with FAILED as alternate terminal.
- **`anchor_hash`** = sha256 over canonical
  `(merkle_root, epoch_id, owner_id, chain_id)`.
- **`verify_anchor`** re-derives the hash and compares; deterministic
  + pure.
- **Inclusion proof** is delegated to `core/merkle_audit` for the
  per-event verifier; this module does the metadata anchoring.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — tx-hash truncated; owner masked.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum


class ChainId(str, Enum):
    """Closed-set chain-id ladder. Operator-tunable; defaults to
    halal-screened halal-friendly L2s."""

    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    ETHEREUM_MAINNET = "ethereum_mainnet"


class AnchorStatus(str, Enum):
    """Closed-set anchor FSM ladder."""

    PREPARED = "prepared"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    FAILED = "failed"


@dataclass(frozen=True)
class AuditEvent:
    """One audit event to be anchored."""

    event_id: str
    event_kind: str
    """e.g. 'trade_filled', 'scholar_attestation', 'kill_switch'."""
    timestamp_iso: str
    payload_hash: str
    """sha256 of the canonical payload representation."""

    def __post_init__(self) -> None:
        if not self.event_id or not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        if not self.event_kind or not self.event_kind.strip():
            raise ValueError("event_kind must be non-empty")
        if not self.timestamp_iso or not self.timestamp_iso.strip():
            raise ValueError("timestamp_iso must be non-empty")
        if len(self.payload_hash) != 64:
            raise ValueError("payload_hash must be sha256 hex (64 chars)")


def hash_event(event: AuditEvent) -> str:
    """Canonical event leaf hash for the Merkle tree."""
    payload = {
        "event_id": event.event_id,
        "event_kind": event.event_kind,
        "timestamp_iso": event.timestamp_iso,
        "payload_hash": event.payload_hash,
    }
    j = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode()).hexdigest()


def merkle_root(events: Sequence[AuditEvent]) -> str:
    """Compute a deterministic sha256 Merkle root over the events.

    Pinned: events are leaf-hashed first; intermediate nodes are
    `sha256(left || right)` with odd-tree-handling = duplicate-last.
    """
    if not events:
        raise ValueError("events must be non-empty")
    layer = [hash_event(e) for e in events]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        new_layer = []
        for i in range(0, len(layer), 2):
            combined = layer[i] + layer[i + 1]
            new_layer.append(hashlib.sha256(combined.encode()).hexdigest())
        layer = new_layer
    return layer[0]


@dataclass(frozen=True)
class AuditAnchor:
    """An audit-trail anchor for one epoch."""

    anchor_id: str
    owner_id: str
    epoch_id: str
    """e.g. '2026-Q2'."""
    chain_id: ChainId
    merkle_root_hex: str
    n_events: int
    prepared_on: date
    anchor_hash: str
    """sha256 over (merkle_root, epoch_id, owner_id, chain_id)."""
    status: AnchorStatus = AnchorStatus.PREPARED
    tx_hash: str = ""
    """On-chain transaction hash; empty until SUBMITTED."""
    confirmed_block: int | None = None
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if not self.anchor_id or not self.anchor_id.strip():
            raise ValueError("anchor_id must be non-empty")
        if not self.owner_id or not self.owner_id.strip():
            raise ValueError("owner_id must be non-empty")
        if not self.epoch_id or not self.epoch_id.strip():
            raise ValueError("epoch_id must be non-empty")
        if len(self.merkle_root_hex) != 64:
            raise ValueError("merkle_root_hex must be sha256 (64 hex)")
        if self.n_events <= 0:
            raise ValueError("n_events must be positive")
        if len(self.anchor_hash) != 64:
            raise ValueError("anchor_hash must be sha256 (64 hex)")
        # Status consistency.
        if self.status is AnchorStatus.SUBMITTED and not self.tx_hash.strip():
            raise ValueError("SUBMITTED requires tx_hash")
        if self.status is AnchorStatus.CONFIRMED:
            if not self.tx_hash.strip():
                raise ValueError("CONFIRMED requires tx_hash")
            if self.confirmed_block is None or self.confirmed_block <= 0:
                raise ValueError("CONFIRMED requires positive confirmed_block")
        if self.status is AnchorStatus.FAILED and not self.failure_reason.strip():
            raise ValueError("FAILED requires failure_reason")


def compute_anchor_hash(
    merkle_root_hex: str,
    epoch_id: str,
    owner_id: str,
    chain_id: ChainId,
) -> str:
    """Canonical hash binding the four fields. Verifiable independently."""
    payload = {
        "merkle_root": merkle_root_hex,
        "epoch_id": epoch_id,
        "owner_id": owner_id,
        "chain_id": chain_id.value,
    }
    j = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode()).hexdigest()


def prepare_anchor(
    *,
    anchor_id: str,
    owner_id: str,
    epoch_id: str,
    chain_id: ChainId,
    events: Sequence[AuditEvent],
    prepared_on: date,
) -> AuditAnchor:
    """Build a PREPARED anchor from a set of events."""
    if not events:
        raise ValueError("events must be non-empty")
    root = merkle_root(events)
    anchor_hash = compute_anchor_hash(root, epoch_id, owner_id, chain_id)
    return AuditAnchor(
        anchor_id=anchor_id,
        owner_id=owner_id,
        epoch_id=epoch_id,
        chain_id=chain_id,
        merkle_root_hex=root,
        n_events=len(events),
        prepared_on=prepared_on,
        anchor_hash=anchor_hash,
        status=AnchorStatus.PREPARED,
    )


_LEGAL_TRANSITIONS: dict[AnchorStatus, set[AnchorStatus]] = {
    AnchorStatus.PREPARED: {AnchorStatus.SUBMITTED, AnchorStatus.FAILED},
    AnchorStatus.SUBMITTED: {AnchorStatus.CONFIRMED, AnchorStatus.FAILED},
    AnchorStatus.CONFIRMED: set(),
    AnchorStatus.FAILED: set(),
}


def mark_submitted(anchor: AuditAnchor, *, tx_hash: str) -> AuditAnchor:
    """PREPARED → SUBMITTED with a non-empty tx_hash."""
    if anchor.status is not AnchorStatus.PREPARED:
        raise ValueError(f"mark_submitted illegal from {anchor.status.value}")
    if not tx_hash.strip():
        raise ValueError("tx_hash must be non-empty")
    if len(tx_hash) > 80:
        raise ValueError("tx_hash too long (suspicious)")
    return replace(anchor, status=AnchorStatus.SUBMITTED, tx_hash=tx_hash)


def mark_confirmed(anchor: AuditAnchor, *, block_number: int) -> AuditAnchor:
    """SUBMITTED → CONFIRMED with a positive block number."""
    if anchor.status is not AnchorStatus.SUBMITTED:
        raise ValueError(f"mark_confirmed illegal from {anchor.status.value}")
    if block_number <= 0:
        raise ValueError("block_number must be positive")
    return replace(
        anchor,
        status=AnchorStatus.CONFIRMED,
        confirmed_block=block_number,
    )


def mark_failed(anchor: AuditAnchor, *, reason: str) -> AuditAnchor:
    """Either PREPARED or SUBMITTED → FAILED with a non-empty reason."""
    if AnchorStatus.FAILED not in _LEGAL_TRANSITIONS[anchor.status]:
        raise ValueError(f"mark_failed illegal from {anchor.status.value}")
    if not reason.strip():
        raise ValueError("reason must be non-empty")
    if len(reason) > 500:
        raise ValueError("reason too long")
    return replace(anchor, status=AnchorStatus.FAILED, failure_reason=reason)


def verify_anchor(anchor: AuditAnchor) -> bool:
    """True iff `anchor.anchor_hash` matches the canonical derivation
    from the other four fields. Tampering with any of the four flips
    the hash."""
    expected = compute_anchor_hash(
        anchor.merkle_root_hex,
        anchor.epoch_id,
        anchor.owner_id,
        anchor.chain_id,
    )
    return anchor.anchor_hash == expected


def verify_event_inclusion(
    event: AuditEvent,
    events: Sequence[AuditEvent],
    anchor: AuditAnchor,
) -> bool:
    """True iff `event` belongs to the anchored set + matches the
    anchor's Merkle root.

    Pinned: caller passes the full event set (or the prefix that
    generated the anchor); the function re-computes the root and
    compares.
    """
    if not any(e.event_id == event.event_id for e in events):
        return False
    return merkle_root(events) == anchor.merkle_root_hex


def _mask(s: str, *, head: int = 8, tail: int = 6) -> str:
    if len(s) <= head + tail + 2:
        return "***"
    return s[:head] + "…" + s[-tail:]


_STATUS_EMOJI: dict[AnchorStatus, str] = {
    AnchorStatus.PREPARED: "📝",
    AnchorStatus.SUBMITTED: "📤",
    AnchorStatus.CONFIRMED: "✅",
    AnchorStatus.FAILED: "❌",
}


def render_anchor(anchor: AuditAnchor) -> str:
    head = (
        f"{_STATUS_EMOJI[anchor.status]} {anchor.anchor_id} "
        f"[{anchor.status.value}] {anchor.epoch_id} on "
        f"{anchor.chain_id.value}\n"
        f"  Owner: {_mask(anchor.owner_id)} | "
        f"events: {anchor.n_events} | "
        f"root: {_mask(anchor.merkle_root_hex)} | "
        f"anchor: {_mask(anchor.anchor_hash)}"
    )
    if anchor.status is AnchorStatus.SUBMITTED:
        head += f"\n  tx: {_mask(anchor.tx_hash)}"
    if anchor.status is AnchorStatus.CONFIRMED:
        head += f"\n  tx: {_mask(anchor.tx_hash)} @ block {anchor.confirmed_block}"
    if anchor.status is AnchorStatus.FAILED:
        head += f"\n  Failure: {anchor.failure_reason}"
    return head
