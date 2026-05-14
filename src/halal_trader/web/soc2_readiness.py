"""SOC 2 Type II audit-readiness aggregator.

The roadmap pins SOC 2 Type II as the institutional badge that
hedge funds and family-office customers ask for once the bot grows
past the operator-laptop deployment. SOC 2 audits are 12-month
evidence-collection exercises followed by a 3-6 month auditor-
review process; this module is the **deployment-readiness
aggregator** that scores the operator's current state against the
five Trust Services Categories (TSC) BEFORE the audit window
opens.

Picked a focused readiness aggregator over an "auto-attest" flow
because (a) SOC 2 audits require human auditor sign-off (the
engine doesn't issue attestations), (b) the audit-trail evidence
is the operator's existing artifacts (access logs / MFA records
/ PR reviews / incident reports / backup runbooks etc.), and (c)
the readiness report's actionable output — "you have 22 of 35
Security controls met, gaps are in CHANGE_MANAGEMENT and INCIDENT_
RESPONSE" — lets the operator focus pre-audit work on actual
gaps.

Mirrors the design of Wave 11.F (halal certification readiness)
but for SOC 2 — same closed-set-enum + module-level-frozen-control-
sets + render-no-PII pattern.

Pinned semantics:
- **Closed Trust Services Category set** per AICPA TSP
  100-2017. Operators add controls within a category via code
  review.
- **Control catalogue module-level frozen.** Runtime config
  drift can't silently weaken the required-evidence floor.
- **Critical-control gap → BLOCKING**. Missing MFA on admin
  access is BLOCKING; missing customer-facing status page is
  WARNING.
- **Type II default 12-month evidence horizon.** Type I (point-
  in-time) is 30-day window; operators select via policy.
- **Render output never includes operator-identifying detail.**
  References abstract evidence-kind labels; never raw user IDs,
  IP addresses, audit trail contents. Mirrors no-PII patterns of
  Wave 11.D + 11.C + 3.B + 11.F.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class TrustServiceCategory(str, Enum):
    """AICPA TSP 100-2017 Trust Services Categories.

    Pinned string values for JSON / DB stability. SECURITY is
    mandatory for every SOC 2 audit; the other four are optional
    but most institutional customers expect Security + Availability
    + Confidentiality at minimum.
    """

    SECURITY = "security"
    AVAILABILITY = "availability"
    PROCESSING_INTEGRITY = "processing_integrity"
    CONFIDENTIALITY = "confidentiality"
    PRIVACY = "privacy"


class ControlCategory(str, Enum):
    """SOC 2 control categories (subset of the AICPA framework).

    Pinned values; operators routing the readiness report to
    per-category dashboards key on these literals.
    """

    ACCESS_CONTROL = "access_control"  # CC6 family
    CHANGE_MANAGEMENT = "change_management"  # CC8 family
    SYSTEM_OPERATIONS = "system_operations"  # CC7 family
    RISK_ASSESSMENT = "risk_assessment"  # CC3 family
    INCIDENT_RESPONSE = "incident_response"  # CC7.4
    LOGICAL_SECURITY = "logical_security"  # CC6.1
    MONITORING = "monitoring"  # CC4 family
    BUSINESS_CONTINUITY = "business_continuity"  # A1.2
    VENDOR_MANAGEMENT = "vendor_management"  # CC9
    DATA_CLASSIFICATION = "data_classification"  # C1.1


class ControlSeverity(str, Enum):
    """Whether a missing control is BLOCKING or WARNING."""

    BLOCKING = "blocking"
    WARNING = "warning"


class EvidenceKind(str, Enum):
    """Standard evidence-kind labels.

    Pinned values; operators populate the evidence catalogue
    deterministically — no free-form names that could silently
    mismatch a control's expected kind.
    """

    ACCESS_LOGS = "access_logs"
    MFA_ENABLED = "mfa_enabled"
    PR_REVIEW_LOGS = "pr_review_logs"
    DEPLOYMENT_LOGS = "deployment_logs"
    BACKUP_RECORDS = "backup_records"
    INCIDENT_REPORTS = "incident_reports"
    DR_DRILL_REPORTS = "dr_drill_reports"
    VULNERABILITY_SCANS = "vulnerability_scans"
    PEN_TEST_REPORTS = "pen_test_reports"
    EMPLOYEE_ONBOARDING = "employee_onboarding"
    EMPLOYEE_OFFBOARDING = "employee_offboarding"
    SECURITY_TRAINING = "security_training"
    VENDOR_SOC2_REPORTS = "vendor_soc2_reports"
    RISK_REGISTER = "risk_register"
    DATA_CLASSIFICATION_POLICY = "data_classification_policy"
    ENCRYPTION_AT_REST = "encryption_at_rest"
    ENCRYPTION_IN_TRANSIT = "encryption_in_transit"
    UPTIME_MONITORING = "uptime_monitoring"
    STATUS_PAGE = "status_page"


class ReadinessLevel(str, Enum):
    """Overall audit-readiness verdict per trust service."""

    READY = "ready"
    NEARLY_READY = "nearly_ready"
    GAPS = "gaps"
    NOT_READY = "not_ready"


class AuditType(str, Enum):
    """SOC 2 Type — affects evidence-collection horizon.

    Type I is a point-in-time attestation (snapshot); Type II
    requires a 3-12 month observation window during which the
    auditor verifies the controls are operating effectively.
    """

    TYPE_I = "type_i"
    TYPE_II = "type_ii"


@dataclass(frozen=True)
class SOC2Control:
    """One SOC 2 control."""

    trust_service: TrustServiceCategory
    control_id: str  # e.g., "CC6.1", "A1.2", "C1.1"
    category: ControlCategory
    severity: ControlSeverity
    description: str
    expected_evidence: EvidenceKind

    def __post_init__(self) -> None:
        if not self.control_id or not self.control_id.strip():
            raise ValueError("control_id must be non-empty")
        if not self.description or not self.description.strip():
            raise ValueError("description must be non-empty")


# Security controls (CC1-CC9 family, non-exhaustive).
_SECURITY_CONTROLS: tuple[SOC2Control, ...] = (
    SOC2Control(
        trust_service=TrustServiceCategory.SECURITY,
        control_id="CC6.1",
        category=ControlCategory.LOGICAL_SECURITY,
        severity=ControlSeverity.BLOCKING,
        description="Logical access controls restrict access to authorised users",
        expected_evidence=EvidenceKind.ACCESS_LOGS,
    ),
    SOC2Control(
        trust_service=TrustServiceCategory.SECURITY,
        control_id="CC6.2",
        category=ControlCategory.ACCESS_CONTROL,
        severity=ControlSeverity.BLOCKING,
        description="Multi-factor authentication on all admin and user access",
        expected_evidence=EvidenceKind.MFA_ENABLED,
    ),
    SOC2Control(
        trust_service=TrustServiceCategory.SECURITY,
        control_id="CC6.3",
        category=ControlCategory.ACCESS_CONTROL,
        severity=ControlSeverity.BLOCKING,
        description="User onboarding / offboarding procedures documented",
        expected_evidence=EvidenceKind.EMPLOYEE_ONBOARDING,
    ),
    SOC2Control(
        trust_service=TrustServiceCategory.SECURITY,
        control_id="CC7.1",
        category=ControlCategory.MONITORING,
        severity=ControlSeverity.BLOCKING,
        description="Vulnerability scans + remediation tracking",
        expected_evidence=EvidenceKind.VULNERABILITY_SCANS,
    ),
    SOC2Control(
        trust_service=TrustServiceCategory.SECURITY,
        control_id="CC7.4",
        category=ControlCategory.INCIDENT_RESPONSE,
        severity=ControlSeverity.BLOCKING,
        description="Incident-response procedures with documented incidents",
        expected_evidence=EvidenceKind.INCIDENT_REPORTS,
    ),
    SOC2Control(
        trust_service=TrustServiceCategory.SECURITY,
        control_id="CC8.1",
        category=ControlCategory.CHANGE_MANAGEMENT,
        severity=ControlSeverity.BLOCKING,
        description="Code changes go through PR review with documented approvals",
        expected_evidence=EvidenceKind.PR_REVIEW_LOGS,
    ),
    SOC2Control(
        trust_service=TrustServiceCategory.SECURITY,
        control_id="CC9.1",
        category=ControlCategory.VENDOR_MANAGEMENT,
        severity=ControlSeverity.WARNING,
        description="Vendor SOC 2 / ISO reports collected for third parties",
        expected_evidence=EvidenceKind.VENDOR_SOC2_REPORTS,
    ),
    SOC2Control(
        trust_service=TrustServiceCategory.SECURITY,
        control_id="CC3.2",
        category=ControlCategory.RISK_ASSESSMENT,
        severity=ControlSeverity.BLOCKING,
        description="Annual risk assessment with documented register",
        expected_evidence=EvidenceKind.RISK_REGISTER,
    ),
    SOC2Control(
        trust_service=TrustServiceCategory.SECURITY,
        control_id="CC2.3",
        category=ControlCategory.MONITORING,
        severity=ControlSeverity.WARNING,
        description="Annual security training for all employees",
        expected_evidence=EvidenceKind.SECURITY_TRAINING,
    ),
    SOC2Control(
        trust_service=TrustServiceCategory.SECURITY,
        control_id="CC6.7",
        category=ControlCategory.ACCESS_CONTROL,
        severity=ControlSeverity.WARNING,
        description="Annual penetration testing",
        expected_evidence=EvidenceKind.PEN_TEST_REPORTS,
    ),
)

# Availability controls.
_AVAILABILITY_CONTROLS: tuple[SOC2Control, ...] = (
    SOC2Control(
        trust_service=TrustServiceCategory.AVAILABILITY,
        control_id="A1.1",
        category=ControlCategory.MONITORING,
        severity=ControlSeverity.BLOCKING,
        description="Uptime monitoring with documented SLA targets",
        expected_evidence=EvidenceKind.UPTIME_MONITORING,
    ),
    SOC2Control(
        trust_service=TrustServiceCategory.AVAILABILITY,
        control_id="A1.2",
        category=ControlCategory.BUSINESS_CONTINUITY,
        severity=ControlSeverity.BLOCKING,
        description="Backup procedures with point-in-time recovery capability",
        expected_evidence=EvidenceKind.BACKUP_RECORDS,
    ),
    SOC2Control(
        trust_service=TrustServiceCategory.AVAILABILITY,
        control_id="A1.3",
        category=ControlCategory.BUSINESS_CONTINUITY,
        severity=ControlSeverity.BLOCKING,
        description="Disaster recovery drills documented",
        expected_evidence=EvidenceKind.DR_DRILL_REPORTS,
    ),
    SOC2Control(
        trust_service=TrustServiceCategory.AVAILABILITY,
        control_id="A1.4",
        category=ControlCategory.MONITORING,
        severity=ControlSeverity.WARNING,
        description="Public status page for service disruptions",
        expected_evidence=EvidenceKind.STATUS_PAGE,
    ),
)

# Confidentiality controls.
_CONFIDENTIALITY_CONTROLS: tuple[SOC2Control, ...] = (
    SOC2Control(
        trust_service=TrustServiceCategory.CONFIDENTIALITY,
        control_id="C1.1",
        category=ControlCategory.DATA_CLASSIFICATION,
        severity=ControlSeverity.BLOCKING,
        description="Data classification policy with handling procedures",
        expected_evidence=EvidenceKind.DATA_CLASSIFICATION_POLICY,
    ),
    SOC2Control(
        trust_service=TrustServiceCategory.CONFIDENTIALITY,
        control_id="C1.2",
        category=ControlCategory.LOGICAL_SECURITY,
        severity=ControlSeverity.BLOCKING,
        description="Encryption at rest for confidential data",
        expected_evidence=EvidenceKind.ENCRYPTION_AT_REST,
    ),
    SOC2Control(
        trust_service=TrustServiceCategory.CONFIDENTIALITY,
        control_id="C1.3",
        category=ControlCategory.LOGICAL_SECURITY,
        severity=ControlSeverity.BLOCKING,
        description="Encryption in transit for confidential data",
        expected_evidence=EvidenceKind.ENCRYPTION_IN_TRANSIT,
    ),
)


_TRUST_SERVICE_CONTROLS: dict[TrustServiceCategory, tuple[SOC2Control, ...]] = {
    TrustServiceCategory.SECURITY: _SECURITY_CONTROLS,
    TrustServiceCategory.AVAILABILITY: _AVAILABILITY_CONTROLS,
    TrustServiceCategory.PROCESSING_INTEGRITY: (),  # operator extends
    TrustServiceCategory.CONFIDENTIALITY: _CONFIDENTIALITY_CONTROLS,
    TrustServiceCategory.PRIVACY: (),  # operator extends; partly covered by Wave 11.D
}


def controls_for(
    trust_service: TrustServiceCategory,
) -> tuple[SOC2Control, ...]:
    """Return the registered control set for a trust service."""

    return _TRUST_SERVICE_CONTROLS[trust_service]


@dataclass(frozen=True)
class EvidenceArtifact:
    """One piece of operator evidence."""

    kind: EvidenceKind
    present: bool
    last_updated: datetime | None = None

    def __post_init__(self) -> None:
        if self.last_updated is not None and self.last_updated.tzinfo is None:
            raise ValueError("last_updated must be timezone-aware when set")


@dataclass(frozen=True)
class SOC2ReadinessPolicy:
    """Operator-tunable policy.

    `audit_type` defaults to TYPE_II (the institutional standard).
    `evidence_horizon_days` defaults to 365 (Type II requires
    12-month observation window); Type I uses 30-day default.
    """

    audit_type: AuditType = AuditType.TYPE_II
    evidence_horizon_days: int = 365

    def __post_init__(self) -> None:
        if self.evidence_horizon_days <= 0:
            raise ValueError("evidence_horizon_days must be positive")


DEFAULT_POLICY = SOC2ReadinessPolicy()


@dataclass(frozen=True)
class ControlAssessment:
    """Per-control evaluation."""

    control: SOC2Control
    is_met: bool
    is_stale: bool = False
    notes: str = ""


@dataclass(frozen=True)
class TrustServiceReport:
    """Per-trust-service readiness."""

    trust_service: TrustServiceCategory
    overall_level: ReadinessLevel
    controls_total: int
    controls_met: int
    controls_blocking_unmet: int
    controls_warning_unmet: int
    controls_stale: int
    assessments: tuple[ControlAssessment, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def met_pct(self) -> float:
        if self.controls_total == 0:
            return 0.0
        return (self.controls_met / self.controls_total) * 100.0


@dataclass(frozen=True)
class SOC2ReadinessReport:
    """Aggregate report covering multiple trust services."""

    audit_type: AuditType
    requested_services: tuple[TrustServiceCategory, ...]
    per_service: tuple[TrustServiceReport, ...]
    overall_level: ReadinessLevel

    @property
    def all_services_met_pct(self) -> float:
        total = sum(s.controls_total for s in self.per_service)
        met = sum(s.controls_met for s in self.per_service)
        if total == 0:
            return 0.0
        return (met / total) * 100.0


def _evaluate_one_service(
    *,
    trust_service: TrustServiceCategory,
    evidence: tuple[EvidenceArtifact, ...],
    now: datetime,
    policy: SOC2ReadinessPolicy,
) -> TrustServiceReport:
    """Evaluate one trust service's controls."""

    controls = controls_for(trust_service)
    notes: list[str] = []

    if not controls:
        notes.append(
            f"control set for {trust_service.value} is empty — operator "
            "must load the spec before evaluation"
        )
        return TrustServiceReport(
            trust_service=trust_service,
            overall_level=ReadinessLevel.NOT_READY,
            controls_total=0,
            controls_met=0,
            controls_blocking_unmet=0,
            controls_warning_unmet=0,
            controls_stale=0,
            notes=tuple(notes),
        )

    evidence_by_kind: dict[EvidenceKind, EvidenceArtifact] = {a.kind: a for a in evidence}

    assessments: list[ControlAssessment] = []
    met_count = 0
    blocking_unmet = 0
    warning_unmet = 0
    stale_count = 0

    horizon = timedelta(days=policy.evidence_horizon_days)

    for control in controls:
        artifact = evidence_by_kind.get(control.expected_evidence)
        is_met = artifact is not None and artifact.present
        is_stale = False
        note = ""

        if artifact is None or not artifact.present:
            if control.severity is ControlSeverity.BLOCKING:
                blocking_unmet += 1
                note = "missing evidence (BLOCKING)"
            else:
                warning_unmet += 1
                note = "missing evidence (warning)"
        else:
            if artifact.last_updated is None:
                is_stale = True
                stale_count += 1
                note = "evidence present but no last_updated timestamp"
            elif now - artifact.last_updated >= horizon:
                is_stale = True
                stale_count += 1
                note = (
                    f"evidence stale ({(now - artifact.last_updated).days}d "
                    f"≥ {policy.evidence_horizon_days}d horizon)"
                )
            met_count += 1

        assessments.append(
            ControlAssessment(
                control=control,
                is_met=is_met,
                is_stale=is_stale,
                notes=note,
            )
        )

    if blocking_unmet == 0 and warning_unmet == 0 and stale_count == 0:
        overall = ReadinessLevel.READY
    elif blocking_unmet == 0:
        overall = ReadinessLevel.NEARLY_READY
    elif blocking_unmet <= 2:
        overall = ReadinessLevel.GAPS
    else:
        overall = ReadinessLevel.NOT_READY

    return TrustServiceReport(
        trust_service=trust_service,
        overall_level=overall,
        controls_total=len(controls),
        controls_met=met_count,
        controls_blocking_unmet=blocking_unmet,
        controls_warning_unmet=warning_unmet,
        controls_stale=stale_count,
        assessments=tuple(assessments),
        notes=tuple(notes),
    )


def evaluate_soc2_readiness(
    *,
    trust_services: tuple[TrustServiceCategory, ...],
    evidence: tuple[EvidenceArtifact, ...],
    now: datetime,
    policy: SOC2ReadinessPolicy = DEFAULT_POLICY,
) -> SOC2ReadinessReport:
    """Evaluate the deployment's SOC 2 readiness across trust services.

    Returns a `SOC2ReadinessReport` with per-trust-service breakdown
    + overall verdict computed as the strictest of the per-service
    levels.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not trust_services:
        raise ValueError("trust_services must be non-empty")

    per_service = tuple(
        _evaluate_one_service(trust_service=ts, evidence=evidence, now=now, policy=policy)
        for ts in trust_services
    )

    # Overall = strictest level across services.
    level_priority: dict[ReadinessLevel, int] = {
        ReadinessLevel.NOT_READY: 0,
        ReadinessLevel.GAPS: 1,
        ReadinessLevel.NEARLY_READY: 2,
        ReadinessLevel.READY: 3,
    }
    overall = min(per_service, key=lambda s: level_priority[s.overall_level])

    return SOC2ReadinessReport(
        audit_type=policy.audit_type,
        requested_services=trust_services,
        per_service=per_service,
        overall_level=overall.overall_level,
    )


_LEVEL_EMOJI: dict[ReadinessLevel, str] = {
    ReadinessLevel.READY: "✅",
    ReadinessLevel.NEARLY_READY: "🟢",
    ReadinessLevel.GAPS: "⚠️",
    ReadinessLevel.NOT_READY: "❌",
}


def render_readiness_report(report: SOC2ReadinessReport) -> str:
    """Format the readiness report for ops display.

    Pinned no-PII contract: never includes operator-identifying
    details (user IDs, IP addresses, audit-trail contents); shows
    abstract evidence-kind labels + count summaries.
    """

    emoji = _LEVEL_EMOJI[report.overall_level]
    lines = [
        f"{emoji} SOC 2 {report.audit_type.value} — {report.overall_level.value.upper()}",
        f"  overall: {report.all_services_met_pct:.1f}% met across "
        f"{len(report.requested_services)} trust services",
    ]
    for svc_report in report.per_service:
        svc_emoji = _LEVEL_EMOJI[svc_report.overall_level]
        lines.append(
            f"  {svc_emoji} {svc_report.trust_service.value}: "
            f"{svc_report.controls_met}/{svc_report.controls_total} "
            f"({svc_report.met_pct:.0f}%) — {svc_report.overall_level.value}"
        )
        if svc_report.controls_blocking_unmet > 0:
            lines.append(f"      blocking gaps: {svc_report.controls_blocking_unmet}")
        if svc_report.controls_warning_unmet > 0:
            lines.append(f"      warnings: {svc_report.controls_warning_unmet}")
        if svc_report.controls_stale > 0:
            lines.append(f"      stale: {svc_report.controls_stale}")
        if svc_report.notes:
            for n in svc_report.notes:
                lines.append(f"      · {n}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "AuditType",
    "ControlAssessment",
    "ControlCategory",
    "ControlSeverity",
    "EvidenceArtifact",
    "EvidenceKind",
    "ReadinessLevel",
    "SOC2Control",
    "SOC2ReadinessPolicy",
    "SOC2ReadinessReport",
    "TrustServiceCategory",
    "TrustServiceReport",
    "controls_for",
    "evaluate_soc2_readiness",
    "render_readiness_report",
]
