"""Investment-advisor registration readiness aggregator.

The roadmap pins Wave 11.A: "Once we serve real-money users,
we're an investment advisor. File the relevant registrations.
~6 months lead time." This module is the **deployment-readiness
aggregator** that scores the operator's current state against
the registration requirements of each major regulator BEFORE
the application paperwork is filed.

Mirrors the design of Wave 11.E (SOC 2 readiness) and Wave 11.F
(halal certification readiness): closed authority set, module-
level frozen requirement sets, per-authority control catalogue,
12-month staleness horizon for annual filings, no-PII render
contract. Operators run the aggregator, see the gaps, close
them, then file the paperwork with confidence.

Picked a focused readiness aggregator over an "auto-file" flow
because (a) registration paperwork requires human signatures + a
notarised Form ADV / FCA SUP submission (the engine doesn't sign
attestations), (b) the audit-trail evidence is the operator's
existing artifacts (firm registration documents + executive
backgrounds + compliance program + recordkeeping policies + AML
program), and (c) the readiness report's actionable output —
"you have 8 of 12 SEC RIA requirements met, the 4 gaps are
SERIES_65_PASSED + SURETY_BOND + ANNUAL_FORM_ADV_AMENDMENT" —
lets the operator focus pre-filing work on the actual gaps.

Pinned semantics:
- **Closed regulator-authority set.** `RegulatorAuthority` lists
  every regulator the aggregator recognises. Operators add a
  new authority via code review.
- **Per-authority requirements module-level frozen.** Runtime
  config drift can't silently weaken the required-evidence
  floor.
- **Critical-requirement gap → BLOCKING.** Missing Form ADV
  Part 1A is BLOCKING; missing optional client-meeting log is
  WARNING.
- **12-month staleness horizon (default).** SEC RIAs file
  annual Form ADV amendments; FCA firms file annual Gabriel /
  RegData returns. Past 12 months → stale + WARNING.
- **Render output never includes operator-identifying detail.**
  References abstract evidence-kind labels; never raw firm
  name / CRD number / executive PII / IARD codes. Mirrors no-PII
  patterns of Wave 11.D + 11.C + 3.B + 11.E + 11.F.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class RegulatorAuthority(str, Enum):
    """Recognised investment-advisor registration authorities.

    Pinned BCP-style string values for JSON / DB stability.
    Operators add a new authority via code review.
    """

    SEC_RIA = "sec_ria"  # US Investment Advisers Act of 1940
    FCA_UK = "fca_uk"  # UK Financial Conduct Authority
    SAUDI_CMA = "saudi_cma"  # Saudi Capital Market Authority
    UAE_VARA = "uae_vara"  # UAE Virtual Assets Regulatory Authority
    SINGAPORE_MAS = "singapore_mas"  # MAS Capital Markets Services licence
    AUSTRALIA_ASIC = "australia_asic"  # ASIC AFSL


class RegistrationCategory(str, Enum):
    """Categories of registration requirements.

    Pinned values — operators routing the readiness report to
    per-category dashboards key on these literals.
    """

    FORM_FILING = "form_filing"
    BACKGROUND_CHECKS = "background_checks"
    EXAMS_LICENSURE = "exams_licensure"
    SURETY_BOND = "surety_bond"
    AML = "aml"
    COMPLIANCE_PROGRAM = "compliance_program"
    RECORDKEEPING = "recordkeeping"
    CLIENT_MONEY = "client_money"
    DISPUTE_RESOLUTION = "dispute_resolution"


class RequirementSeverity(str, Enum):
    """BLOCKING (filing can't proceed) vs WARNING (flagged but proceeds)."""

    BLOCKING = "blocking"
    WARNING = "warning"


class EvidenceKind(str, Enum):
    """Standard evidence-kind labels for registration filings.

    Pinned values; operators populate the evidence catalogue
    deterministically.
    """

    # Form filings
    FORM_ADV_PART_1A = "form_adv_part_1a"
    FORM_ADV_PART_2A = "form_adv_part_2a"
    FORM_ADV_PART_2B = "form_adv_part_2b"
    ANNUAL_FORM_ADV_AMENDMENT = "annual_form_adv_amendment"
    FCA_SUP_FORM = "fca_sup_form"
    FCA_GABRIEL_RETURN = "fca_gabriel_return"

    # Identifiers / registration numbers
    CRD_NUMBER = "crd_number"
    IARD_REGISTRATION = "iard_registration"
    FCA_FIRM_REFERENCE_NUMBER = "fca_firm_reference_number"
    SAUDI_CMA_LICENCE = "saudi_cma_licence"
    SINGAPORE_CMS_LICENCE = "singapore_cms_licence"
    AUSTRALIA_AFSL = "australia_afsl"

    # Personal qualifications
    SERIES_65_PASSED = "series_65_passed"
    SERIES_66_PASSED = "series_66_passed"
    SMCR_CERTIFIED_PERSONS = "smcr_certified_persons"
    EXEC_BACKGROUND_CHECK = "exec_background_check"

    # Compliance program
    SURETY_BOND = "surety_bond"
    AML_PROGRAM = "aml_program"
    COMPLIANCE_MANUAL = "compliance_manual"
    RECORDKEEPING_PROCEDURES = "recordkeeping_procedures"

    # UK / Singapore-specific
    CLIENT_MONEY_RULES_DOCUMENTED = "client_money_rules_documented"
    FOS_MEMBERSHIP = "fos_membership"
    DISPUTE_RESOLUTION_PROCEDURES = "dispute_resolution_procedures"


class ReadinessLevel(str, Enum):
    """Overall readiness verdict per regulator."""

    READY = "ready"
    NEARLY_READY = "nearly_ready"
    GAPS = "gaps"
    NOT_READY = "not_ready"


@dataclass(frozen=True)
class RegistrationRequirement:
    """One registration requirement."""

    authority: RegulatorAuthority
    requirement_id: str  # e.g., "SEC-RIA-1.1", "FCA-CASS-7.2"
    category: RegistrationCategory
    severity: RequirementSeverity
    description: str
    expected_evidence: EvidenceKind

    def __post_init__(self) -> None:
        if not self.requirement_id or not self.requirement_id.strip():
            raise ValueError("requirement_id must be non-empty")
        if not self.description or not self.description.strip():
            raise ValueError("description must be non-empty")


# US SEC RIA requirements — non-exhaustive but covers the
# load-bearing items operators need before filing Form ADV.
_SEC_RIA_REQUIREMENTS: tuple[RegistrationRequirement, ...] = (
    RegistrationRequirement(
        authority=RegulatorAuthority.SEC_RIA,
        requirement_id="SEC-RIA-1.1",
        category=RegistrationCategory.FORM_FILING,
        severity=RequirementSeverity.BLOCKING,
        description="Form ADV Part 1A submitted via IARD",
        expected_evidence=EvidenceKind.FORM_ADV_PART_1A,
    ),
    RegistrationRequirement(
        authority=RegulatorAuthority.SEC_RIA,
        requirement_id="SEC-RIA-1.2",
        category=RegistrationCategory.FORM_FILING,
        severity=RequirementSeverity.BLOCKING,
        description="Form ADV Part 2A brochure prepared",
        expected_evidence=EvidenceKind.FORM_ADV_PART_2A,
    ),
    RegistrationRequirement(
        authority=RegulatorAuthority.SEC_RIA,
        requirement_id="SEC-RIA-1.3",
        category=RegistrationCategory.FORM_FILING,
        severity=RequirementSeverity.BLOCKING,
        description="Form ADV Part 2B brochure supplement for advisory persons",
        expected_evidence=EvidenceKind.FORM_ADV_PART_2B,
    ),
    RegistrationRequirement(
        authority=RegulatorAuthority.SEC_RIA,
        requirement_id="SEC-RIA-2.1",
        category=RegistrationCategory.EXAMS_LICENSURE,
        severity=RequirementSeverity.BLOCKING,
        description="Series 65 (Uniform Investment Adviser Law Exam) passed",
        expected_evidence=EvidenceKind.SERIES_65_PASSED,
    ),
    RegistrationRequirement(
        authority=RegulatorAuthority.SEC_RIA,
        requirement_id="SEC-RIA-3.1",
        category=RegistrationCategory.BACKGROUND_CHECKS,
        severity=RequirementSeverity.BLOCKING,
        description="Executive background check (FINRA U4) completed",
        expected_evidence=EvidenceKind.EXEC_BACKGROUND_CHECK,
    ),
    RegistrationRequirement(
        authority=RegulatorAuthority.SEC_RIA,
        requirement_id="SEC-RIA-4.1",
        category=RegistrationCategory.SURETY_BOND,
        severity=RequirementSeverity.BLOCKING,
        description="Surety bond posted (state-specific minimum)",
        expected_evidence=EvidenceKind.SURETY_BOND,
    ),
    RegistrationRequirement(
        authority=RegulatorAuthority.SEC_RIA,
        requirement_id="SEC-RIA-5.1",
        category=RegistrationCategory.AML,
        severity=RequirementSeverity.BLOCKING,
        description="AML program documented + designated AML compliance officer",
        expected_evidence=EvidenceKind.AML_PROGRAM,
    ),
    RegistrationRequirement(
        authority=RegulatorAuthority.SEC_RIA,
        requirement_id="SEC-RIA-6.1",
        category=RegistrationCategory.COMPLIANCE_PROGRAM,
        severity=RequirementSeverity.BLOCKING,
        description="Written compliance manual under Rule 206(4)-7",
        expected_evidence=EvidenceKind.COMPLIANCE_MANUAL,
    ),
    RegistrationRequirement(
        authority=RegulatorAuthority.SEC_RIA,
        requirement_id="SEC-RIA-7.1",
        category=RegistrationCategory.RECORDKEEPING,
        severity=RequirementSeverity.BLOCKING,
        description="Recordkeeping procedures under Rule 204-2",
        expected_evidence=EvidenceKind.RECORDKEEPING_PROCEDURES,
    ),
    RegistrationRequirement(
        authority=RegulatorAuthority.SEC_RIA,
        requirement_id="SEC-RIA-8.1",
        category=RegistrationCategory.FORM_FILING,
        severity=RequirementSeverity.WARNING,
        description="Annual Form ADV amendment filed within 90 days of fiscal year end",
        expected_evidence=EvidenceKind.ANNUAL_FORM_ADV_AMENDMENT,
    ),
)


# UK FCA requirements — non-exhaustive but covers the load-bearing
# items operators need before filing under SUP / SYSC / CASS / DISP.
_FCA_REQUIREMENTS: tuple[RegistrationRequirement, ...] = (
    RegistrationRequirement(
        authority=RegulatorAuthority.FCA_UK,
        requirement_id="FCA-SUP-1.1",
        category=RegistrationCategory.FORM_FILING,
        severity=RequirementSeverity.BLOCKING,
        description="FCA SUP authorisation form submitted",
        expected_evidence=EvidenceKind.FCA_SUP_FORM,
    ),
    RegistrationRequirement(
        authority=RegulatorAuthority.FCA_UK,
        requirement_id="FCA-SUP-2.1",
        category=RegistrationCategory.BACKGROUND_CHECKS,
        severity=RequirementSeverity.BLOCKING,
        description="SMCR-certified senior managers identified + DBS check",
        expected_evidence=EvidenceKind.SMCR_CERTIFIED_PERSONS,
    ),
    RegistrationRequirement(
        authority=RegulatorAuthority.FCA_UK,
        requirement_id="FCA-CASS-7.1",
        category=RegistrationCategory.CLIENT_MONEY,
        severity=RequirementSeverity.BLOCKING,
        description="Client money rules documented (CASS 7)",
        expected_evidence=EvidenceKind.CLIENT_MONEY_RULES_DOCUMENTED,
    ),
    RegistrationRequirement(
        authority=RegulatorAuthority.FCA_UK,
        requirement_id="FCA-DISP-1.1",
        category=RegistrationCategory.DISPUTE_RESOLUTION,
        severity=RequirementSeverity.BLOCKING,
        description="FOS (Financial Ombudsman Service) membership",
        expected_evidence=EvidenceKind.FOS_MEMBERSHIP,
    ),
    RegistrationRequirement(
        authority=RegulatorAuthority.FCA_UK,
        requirement_id="FCA-SYSC-6.3",
        category=RegistrationCategory.AML,
        severity=RequirementSeverity.BLOCKING,
        description="MLR 2017 compliance program + nominated MLRO",
        expected_evidence=EvidenceKind.AML_PROGRAM,
    ),
    RegistrationRequirement(
        authority=RegulatorAuthority.FCA_UK,
        requirement_id="FCA-SUP-9.1",
        category=RegistrationCategory.FORM_FILING,
        severity=RequirementSeverity.WARNING,
        description="Annual Gabriel / RegData return submitted",
        expected_evidence=EvidenceKind.FCA_GABRIEL_RETURN,
    ),
)


_AUTHORITY_REQUIREMENTS: dict[RegulatorAuthority, tuple[RegistrationRequirement, ...]] = {
    RegulatorAuthority.SEC_RIA: _SEC_RIA_REQUIREMENTS,
    RegulatorAuthority.FCA_UK: _FCA_REQUIREMENTS,
    RegulatorAuthority.SAUDI_CMA: (),  # operator extends
    RegulatorAuthority.UAE_VARA: (),  # operator extends
    RegulatorAuthority.SINGAPORE_MAS: (),  # operator extends
    RegulatorAuthority.AUSTRALIA_ASIC: (),  # operator extends
}


def requirements_for(
    authority: RegulatorAuthority,
) -> tuple[RegistrationRequirement, ...]:
    """Return the registered requirement set for an authority."""

    return _AUTHORITY_REQUIREMENTS[authority]


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
class RegistrationPolicy:
    """Operator-tunable policy.

    `staleness_horizon_days` defaults to 365 (the standard annual
    filing cadence for SEC RIAs + FCA firms); operators in
    multi-jurisdiction setups may run different policies per
    authority.
    """

    staleness_horizon_days: int = 365

    def __post_init__(self) -> None:
        if self.staleness_horizon_days <= 0:
            raise ValueError("staleness_horizon_days must be positive")


DEFAULT_POLICY = RegistrationPolicy()


@dataclass(frozen=True)
class RequirementAssessment:
    """Per-requirement evaluation."""

    requirement: RegistrationRequirement
    is_met: bool
    is_stale: bool = False
    notes: str = ""


@dataclass(frozen=True)
class RegistrationReadinessReport:
    """Aggregate readiness verdict + per-requirement breakdown."""

    authority: RegulatorAuthority
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


def evaluate_registration_readiness(
    *,
    authority: RegulatorAuthority,
    evidence: tuple[EvidenceArtifact, ...],
    now: datetime,
    policy: RegistrationPolicy = DEFAULT_POLICY,
) -> RegistrationReadinessReport:
    """Evaluate the operator's registration readiness."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    requirements = requirements_for(authority)
    notes: list[str] = []

    if not requirements:
        notes.append(
            f"requirement set for {authority.value} is empty — operator must "
            "load the spec before evaluation"
        )
        return RegistrationReadinessReport(
            authority=authority,
            overall_level=ReadinessLevel.NOT_READY,
            requirements_total=0,
            requirements_met=0,
            requirements_blocking_unmet=0,
            requirements_warning_unmet=0,
            requirements_stale=0,
            notes=tuple(notes),
        )

    evidence_by_kind: dict[EvidenceKind, EvidenceArtifact] = {a.kind: a for a in evidence}

    assessments: list[RequirementAssessment] = []
    met_count = 0
    blocking_unmet = 0
    warning_unmet = 0
    stale_count = 0
    horizon = timedelta(days=policy.staleness_horizon_days)

    for req in requirements:
        artifact = evidence_by_kind.get(req.expected_evidence)
        is_met = artifact is not None and artifact.present
        is_stale = False
        note = ""

        if artifact is None or not artifact.present:
            if req.severity is RequirementSeverity.BLOCKING:
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

    if blocking_unmet == 0 and warning_unmet == 0 and stale_count == 0:
        overall = ReadinessLevel.READY
    elif blocking_unmet == 0:
        overall = ReadinessLevel.NEARLY_READY
    elif blocking_unmet <= 2:
        overall = ReadinessLevel.GAPS
    else:
        overall = ReadinessLevel.NOT_READY

    return RegistrationReadinessReport(
        authority=authority,
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


def render_readiness_report(report: RegistrationReadinessReport) -> str:
    """Format the readiness report for ops display.

    Pinned no-PII contract: never includes operator-identifying
    detail (firm name, CRD number, executive PII, IARD codes);
    references abstract evidence-kind labels + count summaries.
    """

    emoji = _LEVEL_EMOJI[report.overall_level]
    lines = [
        f"{emoji} {report.authority.value} — {report.overall_level.value.upper()}",
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
            status = "✓" if a.is_met and not a.is_stale else ("⏰" if a.is_stale else "✗")
            lines.append(
                f"    {status} [{a.requirement.requirement_id}] "
                f"{a.requirement.category.value}/{a.requirement.severity.value}"
            )
            if a.notes:
                lines.append(f"        {a.notes}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "EvidenceArtifact",
    "EvidenceKind",
    "ReadinessLevel",
    "RegistrationCategory",
    "RegistrationPolicy",
    "RegistrationReadinessReport",
    "RegistrationRequirement",
    "RegulatorAuthority",
    "RequirementAssessment",
    "RequirementSeverity",
    "evaluate_registration_readiness",
    "render_readiness_report",
    "requirements_for",
]
