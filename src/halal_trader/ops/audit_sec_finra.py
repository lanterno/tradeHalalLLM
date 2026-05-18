"""SEC + FINRA audit package generator — Round-5 Wave 19.A.

Bundles the artefacts an SEC / FINRA examination requests:

- Trade blotter with timestamps + venues + executions
- Order routing logs
- Risk policy + limits in force during the audit period
- Cycle replay snapshots for sample dates (Wave 5.6 already exists)
- Merkle root anchoring the bundle (Wave 19.H)
- Operator certifications

This module is the **bundle composer + manifest builder**. The
actual file packaging (zip / tar / signed S3 upload) lives one layer
up; here we exercise the structure-and-completeness checks in
isolation so the packager sees a stable contract.

Pinned semantics:

- **Closed-set ArtefactKind ladder** — 8 artefact types pinned.
- **Bundle completeness check** — REQUIRED artefacts must be present.
- **Merkle root** is computed by hashing the manifest ordered
  representation (Wave 19.H primitive composes here).
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import Enum

from halal_trader.core.merkle_audit import MerkleTree


class ArtefactKind(str, Enum):
    """Closed-set audit artefact kinds."""

    TRADE_BLOTTER = "trade_blotter"
    ORDER_ROUTING_LOG = "order_routing_log"
    RISK_POLICY = "risk_policy"
    REPLAY_SNAPSHOTS = "replay_snapshots"
    KILL_SWITCH_HISTORY = "kill_switch_history"
    OPERATOR_CERTIFICATION = "operator_certification"
    SIGNED_RECEIPT_CHAIN = "signed_receipt_chain"
    SUPERVISORY_RECORDS = "supervisory_records"


# REQUIRED artefacts for a SEC/FINRA audit. Operators audited by
# specific entities can require more; this is the floor.
REQUIRED_ARTEFACTS: frozenset[ArtefactKind] = frozenset(
    {
        ArtefactKind.TRADE_BLOTTER,
        ArtefactKind.ORDER_ROUTING_LOG,
        ArtefactKind.RISK_POLICY,
        ArtefactKind.OPERATOR_CERTIFICATION,
    }
)


@dataclass(frozen=True)
class Artefact:
    """A single artefact entry in the audit bundle."""

    kind: ArtefactKind
    title: str
    payload_digest: str  # SHA-256 hex of the artefact payload
    period_start: date
    period_end: date
    record_count: int

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if len(self.payload_digest) != 64:
            raise ValueError("payload_digest must be 64-hex-char SHA-256")
        if self.period_end < self.period_start:
            raise ValueError("period_end before period_start")
        if self.record_count < 0:
            raise ValueError("record_count must be non-negative")


@dataclass(frozen=True)
class AuditBundle:
    """The full audit bundle."""

    operator_handle: str
    audit_period_start: date
    audit_period_end: date
    artefacts: tuple[Artefact, ...]
    merkle_root: str
    is_complete: bool
    missing_required: frozenset[ArtefactKind]

    def __post_init__(self) -> None:
        if not self.operator_handle or not self.operator_handle.strip():
            raise ValueError("operator_handle must be non-empty")
        if "@" in self.operator_handle:
            raise ValueError("operator_handle must be a handle, not an email")
        if self.audit_period_end < self.audit_period_start:
            raise ValueError("period_end before period_start")
        if self.is_complete and self.missing_required:
            raise ValueError("is_complete=True but missing_required non-empty")
        if (not self.is_complete) and not self.missing_required:
            raise ValueError("is_complete=False but missing_required empty")
        if len(self.merkle_root) != 64:
            raise ValueError("merkle_root must be 64-hex-char SHA-256")


def _artefact_canonical(artefact: Artefact) -> bytes:
    """Canonical byte form of an artefact for Merkle anchoring."""
    payload = {
        "kind": artefact.kind.value,
        "title": artefact.title,
        "payload_digest": artefact.payload_digest,
        "period_start": artefact.period_start.isoformat(),
        "period_end": artefact.period_end.isoformat(),
        "record_count": artefact.record_count,
    }
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def build_bundle(
    *,
    operator_handle: str,
    audit_period_start: date,
    audit_period_end: date,
    artefacts: Iterable[Artefact],
) -> AuditBundle:
    """Compose the audit bundle + compute completeness + Merkle root."""
    artefacts_t = tuple(artefacts)

    # Each artefact's period must lie within the audit window.
    for a in artefacts_t:
        if a.period_start < audit_period_start or a.period_end > audit_period_end:
            raise ValueError(f"artefact {a.kind.value} period outside audit window")

    present = {a.kind for a in artefacts_t}
    missing = REQUIRED_ARTEFACTS - present
    is_complete = not missing

    tree = MerkleTree()
    for a in sorted(artefacts_t, key=lambda x: (x.kind.value, x.period_start)):
        tree = tree.add_leaf(_artefact_canonical(a))

    return AuditBundle(
        operator_handle=operator_handle,
        audit_period_start=audit_period_start,
        audit_period_end=audit_period_end,
        artefacts=artefacts_t,
        merkle_root=tree.root(),
        is_complete=is_complete,
        missing_required=frozenset(missing),
    )


def hash_payload(payload: bytes) -> str:
    """Compute SHA-256 hex digest of an artefact's bytes."""
    return hashlib.sha256(payload).hexdigest()


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
    "SSN",
    "TaxID",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_bundle(bundle: AuditBundle) -> str:
    emoji = "✅" if bundle.is_complete else "⚠️"
    head = (
        f"{emoji} SEC/FINRA audit bundle: {bundle.operator_handle} "
        f"({bundle.audit_period_start.isoformat()}→{bundle.audit_period_end.isoformat()})"
    )
    lines = [
        head,
        f"  artefacts: {len(bundle.artefacts)}",
        f"  merkle root: {bundle.merkle_root[:16]}…",
    ]
    if bundle.missing_required:
        for k in sorted(bundle.missing_required, key=lambda x: x.value):
            lines.append(f"  ⚠ missing required: {k.value}")
    for a in bundle.artefacts:
        lines.append(
            f"  • {a.kind.value:24s} {a.title} "
            f"(records={a.record_count}, digest={a.payload_digest[:8]}…)"
        )
    return _scrub("\n".join(lines))
