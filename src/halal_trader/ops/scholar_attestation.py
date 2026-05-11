"""Scholar e-signature on compliance attestations — Round-5 Wave 19.I.

Each quarter the platform's **scholar of record** signs a compliance
attestation confirming the screened universe + structural products
remain Shariah-compliant for the reporting period. This module is the
**attestation FSM + envelope tracker + signature anchor**.

Pinned semantics:

- **Closed-set ReportingPeriod** — Q1 / Q2 / Q3 / Q4 / ANNUAL.
- **Closed-set EnvelopeStatus FSM** — DRAFT → SENT → COUNTERSIGNED →
  COMPLETED, with VOIDED as alternate terminal.
- **Each (scholar, period, year) tuple uniquely identifies an
  attestation** — one envelope per period per scholar.
- **Attestation hash anchors `(scholar, year, period, content_hash)`**
  for tamper-evidence; verifiable without retrieving the doc.
- **Pure-Python deterministic.**
- **No-secret-leak pin** — scholar IDs masked; doc-store URIs masked.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import Enum


class ReportingPeriod(str, Enum):
    """Closed-set reporting-period ladder."""

    Q1 = "q1"
    Q2 = "q2"
    Q3 = "q3"
    Q4 = "q4"
    ANNUAL = "annual"


class EnvelopeStatus(str, Enum):
    """Closed-set envelope FSM ladder."""

    DRAFT = "draft"
    SENT = "sent"
    """Sent to the scholar for review."""
    COUNTERSIGNED = "countersigned"
    """Scholar signed; platform countersign pending."""
    COMPLETED = "completed"
    """Both signed; envelope is final."""
    VOIDED = "voided"


@dataclass(frozen=True)
class Attestation:
    """The text + scope of one attestation."""

    attestation_id: str
    scholar_id: str
    year: int
    period: ReportingPeriod
    content: str
    """Human-readable attestation text. Operator generates from
    template; this layer holds it as opaque bytes for hashing."""
    universe_size: int
    """Number of halal-screened tickers in scope for the period."""
    structured_products_count: int
    """How many structured-product templates (Wa'd / Salam / etc.)
    the scholar attests to."""
    drafted_on: date

    def __post_init__(self) -> None:
        if not self.attestation_id or not self.attestation_id.strip():
            raise ValueError("attestation_id must be non-empty")
        if not self.scholar_id or not self.scholar_id.strip():
            raise ValueError("scholar_id must be non-empty")
        if not 2020 <= self.year <= 2100:
            raise ValueError("year outside reasonable bounds")
        if not self.content or not self.content.strip():
            raise ValueError("content must be non-empty")
        if len(self.content) > 50_000:
            raise ValueError("content must be ≤ 50000 chars")
        if self.universe_size < 0:
            raise ValueError("universe_size must be non-negative")
        if self.structured_products_count < 0:
            raise ValueError("structured_products_count must be non-negative")

    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()


def anchor_hash(attestation: Attestation) -> str:
    """Canonical hash of (scholar, year, period, content_hash).

    Used as the immutable identifier the envelope binds to.
    """
    payload = {
        "scholar_id": attestation.scholar_id,
        "year": attestation.year,
        "period": attestation.period.value,
        "content_hash": attestation.content_hash(),
    }
    j = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode()).hexdigest()


@dataclass(frozen=True)
class SignatureEvent:
    """One signature on the envelope."""

    signer_id: str
    signer_role: str
    """e.g. 'scholar', 'platform_officer'."""
    signed_at: datetime
    method: str = "docusign"
    """e.g. 'docusign', 'hsm', 'manual'."""

    def __post_init__(self) -> None:
        if not self.signer_id or not self.signer_id.strip():
            raise ValueError("signer_id must be non-empty")
        if not self.signer_role or not self.signer_role.strip():
            raise ValueError("signer_role must be non-empty")
        if not self.method or not self.method.strip():
            raise ValueError("method must be non-empty")


@dataclass(frozen=True)
class Envelope:
    """The DocuSign-style envelope holding the attestation + signatures."""

    envelope_id: str
    attestation: Attestation
    anchor: str
    """Hash of the attestation; bound at envelope creation."""
    doc_store_uri: str
    """Pointer to the rendered PDF; masked in render output."""
    status: EnvelopeStatus = EnvelopeStatus.DRAFT
    sent_at: datetime | None = None
    signatures: tuple[SignatureEvent, ...] = ()
    void_reason: str = ""

    def __post_init__(self) -> None:
        if not self.envelope_id or not self.envelope_id.strip():
            raise ValueError("envelope_id must be non-empty")
        if not self.doc_store_uri or not self.doc_store_uri.strip():
            raise ValueError("doc_store_uri must be non-empty")
        if self.anchor != anchor_hash(self.attestation):
            raise ValueError("anchor does not match attestation hash")
        if self.status is EnvelopeStatus.SENT and self.sent_at is None:
            raise ValueError("SENT requires sent_at")
        if self.status is EnvelopeStatus.COUNTERSIGNED and not any(
            s.signer_role == "scholar" for s in self.signatures
        ):
            raise ValueError("COUNTERSIGNED requires a scholar signature")
        if self.status is EnvelopeStatus.COMPLETED and not any(
            s.signer_role == "platform_officer" for s in self.signatures
        ):
            raise ValueError("COMPLETED requires a platform_officer signature")
        if self.status is EnvelopeStatus.VOIDED and not self.void_reason.strip():
            raise ValueError("VOIDED requires void_reason")


_LEGAL_TRANSITIONS: dict[EnvelopeStatus, set[EnvelopeStatus]] = {
    EnvelopeStatus.DRAFT: {EnvelopeStatus.SENT, EnvelopeStatus.VOIDED},
    EnvelopeStatus.SENT: {
        EnvelopeStatus.COUNTERSIGNED,
        EnvelopeStatus.VOIDED,
    },
    EnvelopeStatus.COUNTERSIGNED: {
        EnvelopeStatus.COMPLETED,
        EnvelopeStatus.VOIDED,
    },
    EnvelopeStatus.COMPLETED: set(),
    EnvelopeStatus.VOIDED: set(),
}


def send_envelope(envelope: Envelope, *, at: datetime) -> Envelope:
    """DRAFT → SENT."""
    if envelope.status is not EnvelopeStatus.DRAFT:
        raise ValueError(f"send_envelope requires DRAFT, not {envelope.status.value}")
    return replace(envelope, status=EnvelopeStatus.SENT, sent_at=at)


def countersign(envelope: Envelope, *, scholar_signature: SignatureEvent) -> Envelope:
    """SENT → COUNTERSIGNED."""
    if envelope.status is not EnvelopeStatus.SENT:
        raise ValueError(f"countersign requires SENT, not {envelope.status.value}")
    if scholar_signature.signer_role != "scholar":
        raise ValueError("countersign expects scholar signer_role")
    if scholar_signature.signer_id != envelope.attestation.scholar_id:
        raise ValueError("scholar_signature.signer_id does not match attestation.scholar_id")
    return replace(
        envelope,
        status=EnvelopeStatus.COUNTERSIGNED,
        signatures=(*envelope.signatures, scholar_signature),
    )


def complete(envelope: Envelope, *, platform_signature: SignatureEvent) -> Envelope:
    """COUNTERSIGNED → COMPLETED."""
    if envelope.status is not EnvelopeStatus.COUNTERSIGNED:
        raise ValueError(f"complete requires COUNTERSIGNED, not {envelope.status.value}")
    if platform_signature.signer_role != "platform_officer":
        raise ValueError("complete expects platform_officer signer_role")
    return replace(
        envelope,
        status=EnvelopeStatus.COMPLETED,
        signatures=(*envelope.signatures, platform_signature),
    )


def void(envelope: Envelope, *, reason: str) -> Envelope:
    """Transition to VOIDED with a non-empty reason."""
    if envelope.status in (EnvelopeStatus.COMPLETED, EnvelopeStatus.VOIDED):
        raise ValueError(f"void illegal from {envelope.status.value}")
    if not reason.strip():
        raise ValueError("void requires a non-empty reason")
    return replace(envelope, status=EnvelopeStatus.VOIDED, void_reason=reason)


def verify_envelope(envelope: Envelope) -> bool:
    """True iff the envelope's anchor matches the attestation hash."""
    return envelope.anchor == anchor_hash(envelope.attestation)


def _mask(s: str) -> str:
    if len(s) <= 8:
        return "***"
    return s[:4] + "…" + s[-4:]


_STATUS_EMOJI: dict[EnvelopeStatus, str] = {
    EnvelopeStatus.DRAFT: "📝",
    EnvelopeStatus.SENT: "📤",
    EnvelopeStatus.COUNTERSIGNED: "✍️",
    EnvelopeStatus.COMPLETED: "✅",
    EnvelopeStatus.VOIDED: "🚫",
}


def render_envelope(envelope: Envelope) -> str:
    a = envelope.attestation
    head = (
        f"{_STATUS_EMOJI[envelope.status]} {envelope.envelope_id} "
        f"[{envelope.status.value}] {a.year}-{a.period.value}\n"
        f"  Scholar: {_mask(a.scholar_id)} | "
        f"universe={a.universe_size} | "
        f"products={a.structured_products_count}\n"
        f"  Anchor: {envelope.anchor[:16]}… | "
        f"doc: {_mask(envelope.doc_store_uri)}"
    )
    if envelope.signatures:
        head += "\n  Signatures:"
        for s in envelope.signatures:
            head += (
                f"\n    • {s.signer_role} ({_mask(s.signer_id)}) "
                f"via {s.method} at {s.signed_at.isoformat()}"
            )
    if envelope.status is EnvelopeStatus.VOIDED:
        head += f"\n  Voided: {envelope.void_reason}"
    return head
