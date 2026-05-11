"""SOC2 Type II evidence collector — Round-5 Wave 19.E.

SOC2 Type II audits require *continuous* evidence collection across
five trust-service categories (security, availability, processing
integrity, confidentiality, privacy). This module is the **evidence
registry + completeness scorer**:

1. Operators register evidence artefacts (access-log snapshots, change-
   management records, incident-response postmortems, etc.) against
   the platform's control catalogue.
2. Per-control completeness is computed against the *audit window*
   (typically 12 months); controls below the minimum coverage threshold
   are surfaced for follow-up.
3. A bundle is emitted ready for the SOC2 auditor — list of controls,
   per-control evidence count + gaps.

This module is pure-Python; the deployment layer owns persistence and
the actual artefact files (S3 / encrypted blob store).

Pinned semantics:

- **Closed-set TrustServiceCategory ladder** — SECURITY / AVAILABILITY /
  PROCESSING_INTEGRITY / CONFIDENTIALITY / PRIVACY.
- **Closed-set EvidenceKind ladder** — 8 categories matching the standard
  SOC2 evidence map.
- **Closed-set BundleStatus FSM** — DRAFT → SUBMITTED → AUDITED, with
  REJECTED as a terminal state requiring a new bundle.
- **Audit window is inclusive on both ends.**
- **Completeness threshold default 0.80** — a control needs ≥80% of
  expected periodic evidence to be considered "covered".
- **Pure-Python deterministic.**
- **No-secret-leak pin** — evidence artefact URIs are masked in render.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum


class TrustServiceCategory(str, Enum):
    """Closed-set SOC2 trust-service category ladder."""

    SECURITY = "security"
    AVAILABILITY = "availability"
    PROCESSING_INTEGRITY = "processing_integrity"
    CONFIDENTIALITY = "confidentiality"
    PRIVACY = "privacy"


class EvidenceKind(str, Enum):
    """Closed-set evidence-kind ladder."""

    ACCESS_LOG = "access_log"
    CHANGE_RECORD = "change_record"
    INCIDENT_REPORT = "incident_report"
    BACKUP_VERIFICATION = "backup_verification"
    VULN_SCAN = "vuln_scan"
    POLICY_REVIEW = "policy_review"
    TRAINING_COMPLETION = "training_completion"
    VENDOR_REVIEW = "vendor_review"


class BundleStatus(str, Enum):
    """Closed-set bundle status ladder."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    AUDITED = "audited"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Control:
    """One SOC2 control."""

    control_id: str
    category: TrustServiceCategory
    title: str
    expected_kinds: tuple[EvidenceKind, ...]
    """Kinds of evidence expected for this control."""
    expected_periodic_count: int
    """How many artefacts of these kinds the auditor expects across
    the audit window. E.g. 12 for monthly evidence."""

    def __post_init__(self) -> None:
        if not self.control_id or not self.control_id.strip():
            raise ValueError("control_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.expected_kinds:
            raise ValueError("expected_kinds must be non-empty")
        if len(set(self.expected_kinds)) != len(self.expected_kinds):
            raise ValueError("expected_kinds must be unique")
        if self.expected_periodic_count <= 0:
            raise ValueError("expected_periodic_count must be positive")


@dataclass(frozen=True)
class EvidenceArtefact:
    """An evidence artefact submitted against a control."""

    artefact_id: str
    control_id: str
    kind: EvidenceKind
    collected_on: date
    uri: str
    """Pointer to the artefact (S3 URI, doc-store id). Masked in render."""
    integrity_hash: str
    """sha256 of the artefact contents — auditor can re-verify."""

    def __post_init__(self) -> None:
        if not self.artefact_id or not self.artefact_id.strip():
            raise ValueError("artefact_id must be non-empty")
        if not self.control_id or not self.control_id.strip():
            raise ValueError("control_id must be non-empty")
        if not self.uri or not self.uri.strip():
            raise ValueError("uri must be non-empty")
        if not self.integrity_hash or not self.integrity_hash.strip():
            raise ValueError("integrity_hash must be non-empty")
        if len(self.integrity_hash) != 64:
            raise ValueError("integrity_hash must be sha256-length (64 hex)")


def integrity_of(content: bytes) -> str:
    """Helper for callers that compute the artefact hash themselves."""
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class AuditWindow:
    """Inclusive period over which evidence is collected."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("end must be ≥ start")

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end

    def days(self) -> int:
        return (self.end - self.start).days + 1


@dataclass(frozen=True)
class ControlCoverage:
    """Output of `compute_coverage` per control."""

    control_id: str
    category: TrustServiceCategory
    expected_count: int
    actual_count: int
    coverage_ratio: float
    """In [0, 1]; actual / expected, capped at 1.0."""
    missing_kinds: tuple[EvidenceKind, ...]
    is_complete: bool


def compute_coverage(
    control: Control,
    artefacts: Sequence[EvidenceArtefact],
    *,
    window: AuditWindow,
    completeness_threshold: float = 0.80,
) -> ControlCoverage:
    """Compute coverage for a single control over the audit window."""
    if not 0.0 < completeness_threshold <= 1.0:
        raise ValueError("completeness_threshold must be in (0, 1]")
    relevant = [
        a
        for a in artefacts
        if a.control_id == control.control_id and window.contains(a.collected_on)
    ]
    actual = len(relevant)
    ratio = min(1.0, actual / control.expected_periodic_count)
    present_kinds = {a.kind for a in relevant}
    missing = tuple(k for k in control.expected_kinds if k not in present_kinds)
    is_complete = ratio >= completeness_threshold and not missing
    return ControlCoverage(
        control_id=control.control_id,
        category=control.category,
        expected_count=control.expected_periodic_count,
        actual_count=actual,
        coverage_ratio=ratio,
        missing_kinds=missing,
        is_complete=is_complete,
    )


def coverage_for_catalog(
    catalog: Sequence[Control],
    artefacts: Sequence[EvidenceArtefact],
    *,
    window: AuditWindow,
    completeness_threshold: float = 0.80,
) -> tuple[ControlCoverage, ...]:
    """Compute coverage for every control in the catalog."""
    if not catalog:
        raise ValueError("catalog must be non-empty")
    # Per-id uniqueness.
    ids = [c.control_id for c in catalog]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate control_id in catalog")
    return tuple(
        compute_coverage(
            c,
            artefacts,
            window=window,
            completeness_threshold=completeness_threshold,
        )
        for c in catalog
    )


@dataclass(frozen=True)
class EvidenceBundle:
    """A SOC2 evidence bundle in some lifecycle state."""

    bundle_id: str
    window: AuditWindow
    catalog: tuple[Control, ...]
    artefacts: tuple[EvidenceArtefact, ...]
    status: BundleStatus = BundleStatus.DRAFT
    submitted_on: date | None = None
    audited_on: date | None = None
    auditor_notes: str = ""

    def __post_init__(self) -> None:
        if not self.bundle_id or not self.bundle_id.strip():
            raise ValueError("bundle_id must be non-empty")
        if not self.catalog:
            raise ValueError("catalog must be non-empty")
        if self.status is BundleStatus.SUBMITTED and self.submitted_on is None:
            raise ValueError("SUBMITTED requires submitted_on")
        if self.status is BundleStatus.AUDITED and self.audited_on is None:
            raise ValueError("AUDITED requires audited_on")
        if self.submitted_on is not None and not self.window.start <= self.submitted_on:
            raise ValueError("submitted_on must be ≥ window.start")
        if (
            self.audited_on is not None
            and self.submitted_on is not None
            and self.audited_on < self.submitted_on
        ):
            raise ValueError("audited_on must be ≥ submitted_on")
        # Artefacts must reference catalog control_ids.
        ids = {c.control_id for c in self.catalog}
        for a in self.artefacts:
            if a.control_id not in ids:
                raise ValueError(
                    f"artefact {a.artefact_id} references unknown control {a.control_id}"
                )
        # Artefact IDs unique.
        a_ids = [a.artefact_id for a in self.artefacts]
        if len(set(a_ids)) != len(a_ids):
            raise ValueError("duplicate artefact_id")


_LEGAL_TRANSITIONS: dict[BundleStatus, set[BundleStatus]] = {
    BundleStatus.DRAFT: {BundleStatus.SUBMITTED},
    BundleStatus.SUBMITTED: {BundleStatus.AUDITED, BundleStatus.REJECTED},
    BundleStatus.AUDITED: set(),
    BundleStatus.REJECTED: set(),
}


def transition_bundle(
    bundle: EvidenceBundle,
    *,
    new_status: BundleStatus,
    at: date,
    auditor_notes: str = "",
) -> EvidenceBundle:
    """Move the bundle through the FSM.

    Pinned legal moves:
        DRAFT → SUBMITTED
        SUBMITTED → AUDITED
        SUBMITTED → REJECTED
    """
    if new_status not in _LEGAL_TRANSITIONS[bundle.status]:
        raise ValueError(f"illegal transition {bundle.status.value} → {new_status.value}")
    submitted_on = bundle.submitted_on
    audited_on = bundle.audited_on
    if new_status is BundleStatus.SUBMITTED:
        submitted_on = at
    if new_status is BundleStatus.AUDITED:
        audited_on = at
    return replace(
        bundle,
        status=new_status,
        submitted_on=submitted_on,
        audited_on=audited_on,
        auditor_notes=auditor_notes or bundle.auditor_notes,
    )


def add_artefact(bundle: EvidenceBundle, artefact: EvidenceArtefact) -> EvidenceBundle:
    """Append a new artefact; only legal in DRAFT."""
    if bundle.status is not BundleStatus.DRAFT:
        raise ValueError("artefacts can only be added in DRAFT")
    return replace(bundle, artefacts=(*bundle.artefacts, artefact))


@dataclass(frozen=True)
class BundleSummary:
    """Output of `summarise_bundle`."""

    bundle_id: str
    status: BundleStatus
    n_controls: int
    n_complete: int
    n_incomplete: int
    coverage_avg: float
    incomplete_control_ids: tuple[str, ...]


def summarise_bundle(
    bundle: EvidenceBundle,
    *,
    completeness_threshold: float = 0.80,
) -> BundleSummary:
    """One-shot summary across all controls."""
    coverages = coverage_for_catalog(
        bundle.catalog,
        bundle.artefacts,
        window=bundle.window,
        completeness_threshold=completeness_threshold,
    )
    n_complete = sum(1 for c in coverages if c.is_complete)
    avg = sum(c.coverage_ratio for c in coverages) / len(coverages) if coverages else 0.0
    incomplete = tuple(c.control_id for c in coverages if not c.is_complete)
    return BundleSummary(
        bundle_id=bundle.bundle_id,
        status=bundle.status,
        n_controls=len(coverages),
        n_complete=n_complete,
        n_incomplete=len(incomplete),
        coverage_avg=avg,
        incomplete_control_ids=incomplete,
    )


def _mask_uri(uri: str) -> str:
    if len(uri) <= 16:
        return "***"
    return uri[:8] + "…" + uri[-4:]


_STATUS_EMOJI: dict[BundleStatus, str] = {
    BundleStatus.DRAFT: "📝",
    BundleStatus.SUBMITTED: "📤",
    BundleStatus.AUDITED: "✅",
    BundleStatus.REJECTED: "❌",
}


def render_coverage(coverage: ControlCoverage) -> str:
    marker = "✅" if coverage.is_complete else "🟡"
    missing_str = (
        f" missing={','.join(k.value for k in coverage.missing_kinds)}"
        if coverage.missing_kinds
        else ""
    )
    return (
        f"{marker} {coverage.control_id} [{coverage.category.value}]: "
        f"{coverage.actual_count}/{coverage.expected_count} "
        f"({coverage.coverage_ratio * 100:.0f}%){missing_str}"
    )


def render_summary(summary: BundleSummary) -> str:
    return (
        f"{_STATUS_EMOJI[summary.status]} {summary.bundle_id} "
        f"[{summary.status.value}]: "
        f"{summary.n_complete}/{summary.n_controls} complete "
        f"(avg coverage {summary.coverage_avg * 100:.0f}%)"
    )
