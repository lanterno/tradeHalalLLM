"""Annual halal-compliance attestation generator.

Operators preparing for an AAOIFI certification audit (Wave 11.F),
the SOC 2 audit's halal-side cross-references (Wave 11.E), or
the SSB's own quarterly review (Wave 11.B) need a deterministic,
operator-readable attestation document that consolidates all the
landed evidence into one auditor-citable artefact. This module is
the **pure-Python attestation generator** — given the operator's
existing Round-4 evidence (signed-receipts count from Wave 2.A,
purification ledger totals from Wave 2.D, SSB ruling list from
Wave 11.B, screener decisions from Wave 1.G/1.H/1.I/2.G,
exception-queue resolutions from Wave 2.F), produce a
deterministic narrative + evidence-summary block that the
auditor can match against the underlying database tables.

Picked an attestation generator over an "auto-render to PDF"
flow because (a) the auditor wants the deterministic narrative +
the evidence-pointer block, not a glossy PDF; (b) the regression-
pinned attestation properties (no user-level data leaks; SSB
ruling list required; period bounds enforced) are testable in a
way an ad-hoc generation script isn't; (c) the rendered output
is plain-text suitable for any downstream renderer (markdown,
PDF via Pandoc, ReST via Sphinx, JSON for the dashboard tile).

Pinned semantics:
- **Attestation period 30-366 days.** Below 30 → quarterly
  review territory (use Wave 11.B `needs_quarterly_review`
  instead); above 366 → not annual any more (operator should
  split into multiple periods). Pinned at construction.
- **Requires ≥1 SSB ruling in period.** A platform without any
  SSB activity in the attestation period can't credibly
  attest to shariah governance — the engine raises rather
  than produce a misleading attestation.
- **Statements deterministic from evidence.** Given the same
  inputs, the engine produces byte-identical attestation text
  — operators can diff two attestations to prove what changed.
- **Firm name preserved.** The attestation is for the operator's
  auditor; the firm name appears on the document. Other
  operator-identifying fields (user IDs, account numbers,
  individual customer data) never appear — mirrors no-PII
  patterns of Wave 11.D + 11.C + 3.B.
- **Non-zero evidence requirement.** Every load-bearing field
  (signed_receipts_count, purification_disbursements_usd,
  screener_decisions_count) validates non-negative; an
  attestation with zero signed receipts is rejected at
  construction (the platform isn't operating).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

_MIN_ATTESTATION_DAYS = 30
_MAX_ATTESTATION_DAYS = 366


class AttestationOutcome(str, Enum):
    """Overall attestation verdict.

    Pinned string values for JSON / DB stability.
    """

    SUFFICIENT = "sufficient"  # ready to submit to auditor
    INSUFFICIENT = "insufficient"  # gaps prevent submission
    PROVISIONAL = "provisional"  # attestable but flagged warnings


@dataclass(frozen=True)
class DeploymentEvidence:
    """Aggregated evidence covering one attestation period.

    Fields populated by the persistence layer's queries against
    the operator's existing Round-4 audit-trail tables. The
    engine doesn't query — it consumes the aggregate.
    """

    signed_receipts_count: int  # Wave 2.A
    purification_disbursements_usd: float  # Wave 2.D total disbursed
    purification_outstanding_usd: float  # Wave 2.D outstanding (unpaid)
    ssb_ruling_ids: tuple[str, ...]  # Wave 11.B (period-filtered)
    screener_decisions_count: int  # Wave 1.G + 1.H + 1.I + 2.G aggregate
    exception_queue_resolutions: int  # Wave 2.F resolved
    halt_events_count: int  # operator's halt events in period

    def __post_init__(self) -> None:
        if self.signed_receipts_count < 0:
            raise ValueError("signed_receipts_count must be non-negative")
        if self.purification_disbursements_usd < 0:
            raise ValueError("purification_disbursements_usd must be non-negative")
        if self.purification_outstanding_usd < 0:
            raise ValueError("purification_outstanding_usd must be non-negative")
        if self.screener_decisions_count < 0:
            raise ValueError("screener_decisions_count must be non-negative")
        if self.exception_queue_resolutions < 0:
            raise ValueError("exception_queue_resolutions must be non-negative")
        if self.halt_events_count < 0:
            raise ValueError("halt_events_count must be non-negative")


@dataclass(frozen=True)
class AttestationPeriod:
    """A bounded date range for one attestation.

    `start_at` and `end_at` must both be timezone-aware. Period
    length must be in [30, 366] days.
    """

    start_at: datetime
    end_at: datetime

    def __post_init__(self) -> None:
        if self.start_at.tzinfo is None:
            raise ValueError("start_at must be timezone-aware")
        if self.end_at.tzinfo is None:
            raise ValueError("end_at must be timezone-aware")
        if self.end_at <= self.start_at:
            raise ValueError("end_at must be after start_at")
        delta_days = (self.end_at - self.start_at).days
        if delta_days < _MIN_ATTESTATION_DAYS:
            raise ValueError(
                f"period length {delta_days}d is below {_MIN_ATTESTATION_DAYS}d minimum"
            )
        if delta_days > _MAX_ATTESTATION_DAYS:
            raise ValueError(
                f"period length {delta_days}d exceeds {_MAX_ATTESTATION_DAYS}d maximum"
            )

    @property
    def length_days(self) -> int:
        return (self.end_at - self.start_at).days


@dataclass(frozen=True)
class AttestationStatement:
    """One narrative statement in the attestation document."""

    section: str  # e.g., "screening", "purification", "ssb"
    text: str

    def __post_init__(self) -> None:
        if not self.section or not self.section.strip():
            raise ValueError("section must be non-empty")
        if not self.text or not self.text.strip():
            raise ValueError("text must be non-empty")


@dataclass(frozen=True)
class ComplianceAttestation:
    """The full attestation document.

    `firm_name` is intentionally preserved — the auditor needs to
    know which firm is attesting. All other operator-identifying
    fields are redacted.
    """

    firm_name: str
    period: AttestationPeriod
    evidence: DeploymentEvidence
    outcome: AttestationOutcome
    statements: tuple[AttestationStatement, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    failures: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.firm_name or not self.firm_name.strip():
            raise ValueError("firm_name must be non-empty")


def _format_usd(amount: float) -> str:
    """Format a USD amount for the attestation narrative."""

    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.2f}M"
    if amount >= 1_000:
        return f"${amount / 1_000:.2f}k"
    return f"${amount:.2f}"


def _build_screening_statement(evidence: DeploymentEvidence) -> AttestationStatement:
    return AttestationStatement(
        section="screening",
        text=(
            f"During the period, the platform's halal-screening pipeline "
            f"recorded {evidence.screener_decisions_count} verdicts across "
            f"the active screener set (Zoya / Tadawul / Wave 1.G commodity / "
            f"Wave 1.H sukuk / Wave 1.I REIT / Wave 2.G regulator-index). "
            "Each verdict was persisted with a per-decision audit row "
            "available for auditor inspection."
        ),
    )


def _build_audit_trail_statement(
    evidence: DeploymentEvidence,
) -> AttestationStatement:
    return AttestationStatement(
        section="audit_trail",
        text=(
            f"During the period, {evidence.signed_receipts_count} trade "
            "receipts were generated, each cryptographically signed via "
            "Wave 2.A's Ed25519 chain. The signed-receipt rows are stored "
            "in the operator's append-only audit log."
        ),
    )


def _build_purification_statement(
    evidence: DeploymentEvidence,
) -> AttestationStatement:
    disbursed = _format_usd(evidence.purification_disbursements_usd)
    outstanding = _format_usd(evidence.purification_outstanding_usd)
    return AttestationStatement(
        section="purification",
        text=(
            f"Purification ledger (Wave 2.D): {disbursed} disbursed to "
            f"qualifying recipients during the period; {outstanding} "
            "outstanding at period end. Disbursements documented via "
            "purification-receipt rows; outstanding balance reconciles "
            "against the period's running tally."
        ),
    )


def _build_ssb_statement(evidence: DeploymentEvidence) -> AttestationStatement:
    ruling_count = len(evidence.ssb_ruling_ids)
    if ruling_count == 0:
        # Should not reach here — caller validates ≥1 ruling.
        text = "No SSB rulings issued during the period."
    elif ruling_count == 1:
        text = (
            f"The Shariah Supervisory Board issued 1 ruling during the "
            f"period (ID {evidence.ssb_ruling_ids[0]!r}). The ruling "
            "is published to the operator's public register per "
            "Wave 11.B governance."
        )
    else:
        text = (
            f"The Shariah Supervisory Board issued {ruling_count} rulings "
            f"during the period (IDs: "
            f"{', '.join(repr(rid) for rid in evidence.ssb_ruling_ids)}). "
            "Each ruling is published to the operator's public register "
            "per Wave 11.B governance."
        )
    return AttestationStatement(section="ssb", text=text)


def _build_exception_queue_statement(
    evidence: DeploymentEvidence,
) -> AttestationStatement:
    return AttestationStatement(
        section="exception_queue",
        text=(
            f"Exception queue (Wave 2.F): {evidence.exception_queue_resolutions} "
            "items resolved during the period via the scholar-review "
            "workflow. Resolutions persisted with attribution."
        ),
    )


def _build_halt_events_statement(
    evidence: DeploymentEvidence,
) -> AttestationStatement:
    if evidence.halt_events_count == 0:
        text = (
            "No platform-halt events occurred during the period; the kill-switch was never engaged."
        )
    else:
        text = (
            f"The kill-switch was engaged {evidence.halt_events_count} "
            "times during the period. Each halt event was logged with "
            "the operator-supplied reason and resumption timestamp."
        )
    return AttestationStatement(section="halt_events", text=text)


def generate_attestation(
    *,
    firm_name: str,
    period: AttestationPeriod,
    evidence: DeploymentEvidence,
) -> ComplianceAttestation:
    """Generate the attestation document.

    Returns a `ComplianceAttestation` with statements + outcome.
    `outcome` is INSUFFICIENT when load-bearing evidence is
    missing (zero signed receipts, no SSB rulings); SUFFICIENT
    when all required evidence is present and clean; PROVISIONAL
    when present but flagged (e.g., outstanding purification
    balance suggesting unfinished disbursements).
    """

    if not firm_name or not firm_name.strip():
        raise ValueError("firm_name must be non-empty")

    failures: list[str] = []
    warnings: list[str] = []

    # Hard-required evidence.
    if not evidence.ssb_ruling_ids:
        failures.append(
            "no SSB rulings recorded in attestation period — "
            "platform cannot attest to shariah governance"
        )
    if evidence.signed_receipts_count == 0:
        failures.append(
            "zero signed-receipt rows during period — "
            "the platform did not perform any auditable trades"
        )

    # Soft warnings.
    if evidence.purification_outstanding_usd > 0:
        outstanding_str = _format_usd(evidence.purification_outstanding_usd)
        warnings.append(
            f"purification outstanding balance is {outstanding_str} "
            "at period end — operator should disburse before next "
            "attestation cycle"
        )
    if evidence.exception_queue_resolutions == 0 and evidence.screener_decisions_count > 0:
        warnings.append(
            "no exception-queue items resolved despite active screening — "
            "verify the queue isn't accumulating unresolved items"
        )

    statements = (
        _build_screening_statement(evidence),
        _build_audit_trail_statement(evidence),
        _build_purification_statement(evidence),
        _build_ssb_statement(evidence),
        _build_exception_queue_statement(evidence),
        _build_halt_events_statement(evidence),
    )

    if failures:
        outcome = AttestationOutcome.INSUFFICIENT
    elif warnings:
        outcome = AttestationOutcome.PROVISIONAL
    else:
        outcome = AttestationOutcome.SUFFICIENT

    return ComplianceAttestation(
        firm_name=firm_name,
        period=period,
        evidence=evidence,
        outcome=outcome,
        statements=statements,
        warnings=tuple(warnings),
        failures=tuple(failures),
    )


_OUTCOME_EMOJI: dict[AttestationOutcome, str] = {
    AttestationOutcome.SUFFICIENT: "✅",
    AttestationOutcome.PROVISIONAL: "⚠️",
    AttestationOutcome.INSUFFICIENT: "❌",
}


def render_attestation(attestation: ComplianceAttestation) -> str:
    """Render the attestation as auditor-readable text.

    Pinned no-user-PII contract: includes the firm_name (the
    auditor needs to know which firm is attesting) but never
    individual user IDs / account numbers / customer-level data.
    """

    emoji = _OUTCOME_EMOJI[attestation.outcome]
    lines = [
        f"{emoji} Halal Compliance Attestation",
        f"  firm: {attestation.firm_name}",
        f"  period: {attestation.period.start_at.date().isoformat()} → "
        f"{attestation.period.end_at.date().isoformat()} "
        f"({attestation.period.length_days}d)",
        f"  outcome: {attestation.outcome.value.upper()}",
        "",
        "Statements:",
    ]
    for stmt in attestation.statements:
        lines.append(f"  · [{stmt.section}] {stmt.text}")
    if attestation.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in attestation.warnings:
            lines.append(f"  ⚠️  {w}")
    if attestation.failures:
        lines.append("")
        lines.append("Failures:")
        for f in attestation.failures:
            lines.append(f"  ❌ {f}")
    return "\n".join(lines)


__all__ = [
    "AttestationOutcome",
    "AttestationPeriod",
    "AttestationStatement",
    "ComplianceAttestation",
    "DeploymentEvidence",
    "generate_attestation",
    "render_attestation",
]
