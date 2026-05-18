"""Halal certification readiness aggregator.

The roadmap pins halal certification from a recognised authority
(AAOIFI / Saudi Tadawul / Malaysian SAC / Bahrain CBB / Indonesia
DSN-MUI) as the badge that unlocks institutional adoption in
Muslim-majority markets. The actual application + audit + on-site
review process is operator-driven and takes 3-12 months
end-to-end; this module is the **deployment-readiness aggregator**
that scores the operator's current state against each body's
requirement set BEFORE the application is submitted.

Picked a focused readiness aggregator over an "auto-certify"
flow because (a) certification bodies require human auditor
sign-off, (b) the audit-trail evidence is the operator's
existing artifacts (Wave 2.A signed receipts, Wave 2.D
purification ledger, Wave 11.B SSB rulings, Wave 11.C KYC), and
(c) the readiness report's actionable output — "you have 8 of
12 AAOIFI requirements met, the 4 gaps are in
PURIFICATION_LEDGER and SSB_GOVERNANCE" — lets the operator
focus their pre-application work on the actual gaps rather than
generic "be more shariah-compliant" advice.

Pinned semantics:
- **Closed body set.** `CertificationBody` lists every authority
  the aggregator recognises. Operators add a new body via code
  review with a regression test that includes its requirement
  set.
- **Per-body requirements are module-level frozen.** Operators
  override per-body via `add_requirement()` API or extend the
  module; runtime config drift can't silently weaken the
  required-evidence floor.
- **Critical-requirement gap → BLOCKING in report.** A missing
  SSB ruling for AAOIFI applications is BLOCKING; a missing
  marketing-translation file is a WARNING.
- **Stale evidence (> 6 months by default) → WARNING.** The
  operator's certified evidence is presumed valid for 6 months;
  past that the artifact needs refresh. Pinned via test.
- **Render output never includes operator-identifying detail.**
  The readiness report references the operator's deployment by
  abstract artifact-kind labels (e.g., "ssb-ruling: present,
  refreshed 30d ago") never raw rule IDs / member names / KYC
  secrets. Mirrors no-PII patterns of Wave 11.D + 11.C + 3.B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class CertificationBody(str, Enum):
    """Recognised halal-certification authorities.

    Pinned string values for JSON / DB stability. Operators add a
    new body via code review with its requirement set bundled.
    """

    AAOIFI = "aaoifi"
    SAUDI_TADAWUL = "saudi_tadawul"
    MALAYSIAN_SAC = "malaysian_sac"
    BAHRAIN_CBB = "bahrain_cbb"
    INDONESIA_DSN_MUI = "indonesia_dsn_mui"


class RequirementCategory(str, Enum):
    """Requirement-category labels for the per-category breakdown.

    Pinned values; operators routing the readiness report to
    per-category dashboards key on these literals.
    """

    SCREENING = "screening"
    AUDIT = "audit"
    PURIFICATION = "purification"
    SSB_GOVERNANCE = "ssb_governance"
    KYC_AML = "kyc_aml"
    DOCUMENTATION = "documentation"


class RequirementSeverity(str, Enum):
    """Whether a missing requirement is BLOCKING or WARNING."""

    BLOCKING = "blocking"
    WARNING = "warning"


class EvidenceArtifactKind(str, Enum):
    """Standard artifact-kind labels.

    Pinned values let operators populate the evidence catalogue
    deterministically — no free-form artifact names that would
    silently mismatch a requirement's expected kind.
    """

    HALAL_SCREENER_DECISIONS = "halal_screener_decisions"  # Wave 1.I, 2.B, 2.G, etc.
    SIGNED_TRADE_RECEIPTS = "signed_trade_receipts"  # Wave 2.A
    PURIFICATION_LEDGER = "purification_ledger"  # Wave 2.D
    SSB_RULINGS = "ssb_rulings"  # Wave 11.B
    SSB_QUARTERLY_REVIEWS = "ssb_quarterly_reviews"  # Wave 11.B
    KYC_VERIFIED_USERS = "kyc_verified_users"  # Wave 11.C
    AML_SANCTIONS_SCREENING = "aml_sanctions_screening"  # Wave 11.C
    ANNUAL_AUDIT_REPORT = "annual_audit_report"
    PUBLIC_PRIVACY_POLICY = "public_privacy_policy"  # Wave 11.D
    SHARIAH_AUDIT_REPORT = "shariah_audit_report"


class ReadinessLevel(str, Enum):
    """Overall readiness verdict.

    Pinned values for the dashboard's certification-tile rendering.
    """

    READY = "ready"
    NEARLY_READY = "nearly_ready"  # all blocking met; warnings remain
    GAPS = "gaps"
    NOT_READY = "not_ready"  # multiple blocking gaps


@dataclass(frozen=True)
class CertificationRequirement:
    """One requirement from a certification body's spec.

    `spec_id` is the body's own reference (e.g., "AAOIFI-S21-2.3"
    for AAOIFI Standard 21 Section 2.3); operators cite this in
    their application paperwork.
    """

    body: CertificationBody
    spec_id: str
    category: RequirementCategory
    severity: RequirementSeverity
    description: str
    expected_artifact: EvidenceArtifactKind

    def __post_init__(self) -> None:
        if not self.spec_id or not self.spec_id.strip():
            raise ValueError("spec_id must be non-empty")
        if not self.description or not self.description.strip():
            raise ValueError("description must be non-empty")


# AAOIFI Standard 21 + AAOIFI Shariah Standard 17 + AAOIFI Standard
# 7 (the three core standards a halal-trading platform must align
# with). Operators extend per-body via add_requirement.
_AAOIFI_REQUIREMENTS: tuple[CertificationRequirement, ...] = (
    CertificationRequirement(
        body=CertificationBody.AAOIFI,
        spec_id="AAOIFI-S21-2.1",
        category=RequirementCategory.SCREENING,
        severity=RequirementSeverity.BLOCKING,
        description="Halal-screener decisions persisted with audit trail",
        expected_artifact=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
    ),
    CertificationRequirement(
        body=CertificationBody.AAOIFI,
        spec_id="AAOIFI-S21-2.4",
        category=RequirementCategory.AUDIT,
        severity=RequirementSeverity.BLOCKING,
        description="Per-trade signed receipts (Ed25519 or equivalent)",
        expected_artifact=EvidenceArtifactKind.SIGNED_TRADE_RECEIPTS,
    ),
    CertificationRequirement(
        body=CertificationBody.AAOIFI,
        spec_id="AAOIFI-S21-3.2",
        category=RequirementCategory.PURIFICATION,
        severity=RequirementSeverity.BLOCKING,
        description="Quarterly purification ledger with disbursement records",
        expected_artifact=EvidenceArtifactKind.PURIFICATION_LEDGER,
    ),
    CertificationRequirement(
        body=CertificationBody.AAOIFI,
        spec_id="AAOIFI-GS-1.1",
        category=RequirementCategory.SSB_GOVERNANCE,
        severity=RequirementSeverity.BLOCKING,
        description="Shariah Supervisory Board with ≥3 scholars from ≥3 schools",
        expected_artifact=EvidenceArtifactKind.SSB_RULINGS,
    ),
    CertificationRequirement(
        body=CertificationBody.AAOIFI,
        spec_id="AAOIFI-GS-1.3",
        category=RequirementCategory.SSB_GOVERNANCE,
        severity=RequirementSeverity.BLOCKING,
        description="Quarterly SSB review meetings documented",
        expected_artifact=EvidenceArtifactKind.SSB_QUARTERLY_REVIEWS,
    ),
    CertificationRequirement(
        body=CertificationBody.AAOIFI,
        spec_id="AAOIFI-S17-4.1",
        category=RequirementCategory.AUDIT,
        severity=RequirementSeverity.BLOCKING,
        description="Annual external shariah audit report",
        expected_artifact=EvidenceArtifactKind.SHARIAH_AUDIT_REPORT,
    ),
    CertificationRequirement(
        body=CertificationBody.AAOIFI,
        spec_id="AAOIFI-OP-2.1",
        category=RequirementCategory.DOCUMENTATION,
        severity=RequirementSeverity.WARNING,
        description="Public privacy policy + data retention disclosure",
        expected_artifact=EvidenceArtifactKind.PUBLIC_PRIVACY_POLICY,
    ),
)


_TADAWUL_REQUIREMENTS: tuple[CertificationRequirement, ...] = (
    CertificationRequirement(
        body=CertificationBody.SAUDI_TADAWUL,
        spec_id="TADAWUL-HSI-1.1",
        category=RequirementCategory.SCREENING,
        severity=RequirementSeverity.BLOCKING,
        description="Tadawul Halal Stock Index methodology compliance",
        expected_artifact=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
    ),
    CertificationRequirement(
        body=CertificationBody.SAUDI_TADAWUL,
        spec_id="TADAWUL-KYC-2.1",
        category=RequirementCategory.KYC_AML,
        severity=RequirementSeverity.BLOCKING,
        description="Saudi-jurisdiction KYC verification for all users",
        expected_artifact=EvidenceArtifactKind.KYC_VERIFIED_USERS,
    ),
    CertificationRequirement(
        body=CertificationBody.SAUDI_TADAWUL,
        spec_id="TADAWUL-AML-2.2",
        category=RequirementCategory.KYC_AML,
        severity=RequirementSeverity.BLOCKING,
        description="OFAC + UN + Saudi Sanctions Authority screening",
        expected_artifact=EvidenceArtifactKind.AML_SANCTIONS_SCREENING,
    ),
    CertificationRequirement(
        body=CertificationBody.SAUDI_TADAWUL,
        spec_id="TADAWUL-AUDIT-3.1",
        category=RequirementCategory.AUDIT,
        severity=RequirementSeverity.BLOCKING,
        description="Annual audit by SOCPA-licensed firm",
        expected_artifact=EvidenceArtifactKind.ANNUAL_AUDIT_REPORT,
    ),
)


_MALAYSIAN_SAC_REQUIREMENTS: tuple[CertificationRequirement, ...] = (
    CertificationRequirement(
        body=CertificationBody.MALAYSIAN_SAC,
        spec_id="SAC-RES-1",
        category=RequirementCategory.SCREENING,
        severity=RequirementSeverity.BLOCKING,
        description="Securities Commission Malaysia Shariah list compliance",
        expected_artifact=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
    ),
    CertificationRequirement(
        body=CertificationBody.MALAYSIAN_SAC,
        spec_id="SAC-RES-3",
        category=RequirementCategory.SSB_GOVERNANCE,
        severity=RequirementSeverity.BLOCKING,
        description="SAC-recognised Shariah advisor panel",
        expected_artifact=EvidenceArtifactKind.SSB_RULINGS,
    ),
    CertificationRequirement(
        body=CertificationBody.MALAYSIAN_SAC,
        spec_id="SAC-RES-7",
        category=RequirementCategory.PURIFICATION,
        severity=RequirementSeverity.WARNING,
        description="Annual purification disclosure (Malaysian-style)",
        expected_artifact=EvidenceArtifactKind.PURIFICATION_LEDGER,
    ),
)


_BODY_REQUIREMENTS: dict[CertificationBody, tuple[CertificationRequirement, ...]] = {
    CertificationBody.AAOIFI: _AAOIFI_REQUIREMENTS,
    CertificationBody.SAUDI_TADAWUL: _TADAWUL_REQUIREMENTS,
    CertificationBody.MALAYSIAN_SAC: _MALAYSIAN_SAC_REQUIREMENTS,
    CertificationBody.BAHRAIN_CBB: (),  # operator extends
    CertificationBody.INDONESIA_DSN_MUI: (),  # operator extends
}


def requirements_for(body: CertificationBody) -> tuple[CertificationRequirement, ...]:
    """Return the registered requirement set for a body.

    An empty tuple means the operator hasn't loaded that body's
    spec yet — the readiness aggregator will return GAPS with a
    warning surfaced through the report.
    """

    return _BODY_REQUIREMENTS[body]


@dataclass(frozen=True)
class EvidenceArtifact:
    """Operator's evidence artifact.

    `present` indicates whether the artifact exists in the
    operator's deployment (e.g., signed receipts table is
    populated). `last_updated` is the most-recent timestamp on
    the artifact; staleness drives the WARNING-vs-OK decision.
    """

    kind: EvidenceArtifactKind
    present: bool
    last_updated: datetime | None = None

    def __post_init__(self) -> None:
        if self.last_updated is not None and self.last_updated.tzinfo is None:
            raise ValueError("last_updated must be timezone-aware when set")


@dataclass(frozen=True)
class ReadinessPolicy:
    """Operator-tunable policy.

    `staleness_horizon_days` defaults to 180 (six months);
    operators in jurisdictions with stricter audit cadence drop
    to 90.
    """

    staleness_horizon_days: int = 180

    def __post_init__(self) -> None:
        if self.staleness_horizon_days <= 0:
            raise ValueError("staleness_horizon_days must be positive")


DEFAULT_POLICY = ReadinessPolicy()


@dataclass(frozen=True)
class RequirementAssessment:
    """Per-requirement evaluation result."""

    requirement: CertificationRequirement
    is_met: bool
    is_stale: bool = False
    notes: str = ""


@dataclass(frozen=True)
class ReadinessReport:
    """Aggregate readiness verdict + per-requirement breakdown."""

    body: CertificationBody
    overall_level: ReadinessLevel
    requirements_total: int
    requirements_met: int
    requirements_blocking_unmet: int
    requirements_warning_unmet: int
    requirements_stale: int
    assessments: tuple[RequirementAssessment, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def met_pct(self) -> float:
        if self.requirements_total == 0:
            return 0.0
        return (self.requirements_met / self.requirements_total) * 100.0


def evaluate_readiness(
    *,
    body: CertificationBody,
    evidence: tuple[EvidenceArtifact, ...],
    now: datetime,
    policy: ReadinessPolicy = DEFAULT_POLICY,
) -> ReadinessReport:
    """Evaluate the deployment's readiness against a body's requirements.

    Returns a `ReadinessReport` with overall level + per-requirement
    breakdown. Pure: takes evidence as input; the persistence layer
    populates evidence from the operator's database.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    requirements = requirements_for(body)
    notes: list[str] = []

    if not requirements:
        notes.append(
            f"requirement set for {body.value} is empty — operator must "
            "load the spec before evaluation"
        )
        return ReadinessReport(
            body=body,
            overall_level=ReadinessLevel.NOT_READY,
            requirements_total=0,
            requirements_met=0,
            requirements_blocking_unmet=0,
            requirements_warning_unmet=0,
            requirements_stale=0,
            notes=tuple(notes),
        )

    # Index evidence by artifact kind for O(1) lookup.
    evidence_by_kind: dict[EvidenceArtifactKind, EvidenceArtifact] = {}
    for artifact in evidence:
        evidence_by_kind[artifact.kind] = artifact

    assessments: list[RequirementAssessment] = []
    met_count = 0
    blocking_unmet = 0
    warning_unmet = 0
    stale_count = 0

    horizon = timedelta(days=policy.staleness_horizon_days)

    for req in requirements:
        artifact = evidence_by_kind.get(req.expected_artifact)
        is_met = artifact is not None and artifact.present
        is_stale = False
        note = ""

        if artifact is None or not artifact.present:
            if req.severity is RequirementSeverity.BLOCKING:
                blocking_unmet += 1
                note = "missing artifact (BLOCKING)"
            else:
                warning_unmet += 1
                note = "missing artifact (warning)"
        else:
            if artifact.last_updated is None:
                # Present but no timestamp — surface as stale-warning.
                is_stale = True
                stale_count += 1
                note = "artifact present but no last_updated timestamp"
            elif now - artifact.last_updated >= horizon:
                is_stale = True
                stale_count += 1
                note = (
                    f"artifact stale ({(now - artifact.last_updated).days}d "
                    f"≥ {policy.staleness_horizon_days}d horizon)"
                )
            met_count += 1

        assessments.append(
            RequirementAssessment(
                requirement=req,
                is_met=is_met,
                is_stale=is_stale,
                notes=note,
            )
        )

    # Overall level decision tree.
    if blocking_unmet == 0 and warning_unmet == 0 and stale_count == 0:
        overall = ReadinessLevel.READY
    elif blocking_unmet == 0 and stale_count == 0 and warning_unmet > 0:
        overall = ReadinessLevel.NEARLY_READY
    elif blocking_unmet == 0:
        overall = ReadinessLevel.NEARLY_READY
    elif blocking_unmet <= 2:
        overall = ReadinessLevel.GAPS
    else:
        overall = ReadinessLevel.NOT_READY

    return ReadinessReport(
        body=body,
        overall_level=overall,
        requirements_total=len(requirements),
        requirements_met=met_count,
        requirements_blocking_unmet=blocking_unmet,
        requirements_warning_unmet=warning_unmet,
        requirements_stale=stale_count,
        assessments=tuple(assessments),
        notes=tuple(notes),
    )


_LEVEL_EMOJI: dict[ReadinessLevel, str] = {
    ReadinessLevel.READY: "✅",
    ReadinessLevel.NEARLY_READY: "🟢",
    ReadinessLevel.GAPS: "⚠️",
    ReadinessLevel.NOT_READY: "❌",
}


def render_readiness_report(report: ReadinessReport) -> str:
    """Format the readiness report for ops display.

    Pinned no-PII contract: never includes operator-identifying
    detail (rule_id contents, member names, KYC secrets); shows
    artifact-kind labels + count summaries. Mirrors no-PII patterns
    of Wave 11.D + 11.C + 3.B.
    """

    emoji = _LEVEL_EMOJI[report.overall_level]
    lines = [
        f"{emoji} {report.body.value} — {report.overall_level.value.upper()}",
        f"  requirements met: {report.requirements_met}/{report.requirements_total} "
        f"({report.met_pct:.1f}%)",
    ]
    if report.requirements_blocking_unmet > 0:
        lines.append(f"  blocking gaps: {report.requirements_blocking_unmet}")
    if report.requirements_warning_unmet > 0:
        lines.append(f"  warnings: {report.requirements_warning_unmet}")
    if report.requirements_stale > 0:
        lines.append(f"  stale artifacts: {report.requirements_stale}")
    if report.notes:
        lines.append("  notes:")
        for n in report.notes:
            lines.append(f"    · {n}")
    if report.assessments:
        lines.append("  per-requirement breakdown:")
        for a in report.assessments:
            status_emoji = "✓" if a.is_met and not a.is_stale else ("⏰" if a.is_stale else "✗")
            lines.append(
                f"    {status_emoji} [{a.requirement.spec_id}] "
                f"{a.requirement.category.value}/{a.requirement.severity.value}"
            )
            if a.notes:
                lines.append(f"        {a.notes}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "CertificationBody",
    "CertificationRequirement",
    "EvidenceArtifact",
    "EvidenceArtifactKind",
    "ReadinessLevel",
    "ReadinessPolicy",
    "ReadinessReport",
    "RequirementAssessment",
    "RequirementCategory",
    "RequirementSeverity",
    "evaluate_readiness",
    "render_readiness_report",
    "requirements_for",
]
