"""Tests for the investment-advisor registration readiness aggregator."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.web.advisor_registration_readiness import (
    DEFAULT_POLICY,
    EvidenceArtifact,
    EvidenceKind,
    ReadinessLevel,
    RegistrationCategory,
    RegistrationPolicy,
    RegistrationRequirement,
    RegulatorAuthority,
    RequirementSeverity,
    evaluate_registration_readiness,
    render_readiness_report,
    requirements_for,
)

_NOW = datetime(2026, 5, 1, tzinfo=UTC)


def _artifact(
    *,
    kind: EvidenceKind = EvidenceKind.FORM_ADV_PART_1A,
    present: bool = True,
    last_updated: datetime | None = None,
) -> EvidenceArtifact:
    return EvidenceArtifact(
        kind=kind,
        present=present,
        last_updated=last_updated if last_updated is not None else _NOW - timedelta(days=30),
    )


def _full_sec_ria_evidence() -> tuple[EvidenceArtifact, ...]:
    """All SEC RIA evidence artifacts present + fresh."""

    return (
        _artifact(kind=EvidenceKind.FORM_ADV_PART_1A),
        _artifact(kind=EvidenceKind.FORM_ADV_PART_2A),
        _artifact(kind=EvidenceKind.FORM_ADV_PART_2B),
        _artifact(kind=EvidenceKind.SERIES_65_PASSED),
        _artifact(kind=EvidenceKind.EXEC_BACKGROUND_CHECK),
        _artifact(kind=EvidenceKind.SURETY_BOND),
        _artifact(kind=EvidenceKind.AML_PROGRAM),
        _artifact(kind=EvidenceKind.COMPLIANCE_MANUAL),
        _artifact(kind=EvidenceKind.RECORDKEEPING_PROCEDURES),
        _artifact(kind=EvidenceKind.ANNUAL_FORM_ADV_AMENDMENT),
    )


def _full_fca_evidence() -> tuple[EvidenceArtifact, ...]:
    """All FCA evidence artifacts present + fresh."""

    return (
        _artifact(kind=EvidenceKind.FCA_SUP_FORM),
        _artifact(kind=EvidenceKind.SMCR_CERTIFIED_PERSONS),
        _artifact(kind=EvidenceKind.CLIENT_MONEY_RULES_DOCUMENTED),
        _artifact(kind=EvidenceKind.FOS_MEMBERSHIP),
        _artifact(kind=EvidenceKind.AML_PROGRAM),
        _artifact(kind=EvidenceKind.FCA_GABRIEL_RETURN),
    )


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_default_policy() -> None:
    assert DEFAULT_POLICY.staleness_horizon_days == 365


def test_policy_rejects_zero_horizon() -> None:
    with pytest.raises(ValueError, match="staleness_horizon_days"):
        RegistrationPolicy(staleness_horizon_days=0)


def test_policy_rejects_negative_horizon() -> None:
    with pytest.raises(ValueError, match="staleness_horizon_days"):
        RegistrationPolicy(staleness_horizon_days=-1)


# ---------------------------------------------------------------------------
# Requirement set queries
# ---------------------------------------------------------------------------


def test_sec_ria_requirements_present() -> None:
    requirements = requirements_for(RegulatorAuthority.SEC_RIA)
    assert len(requirements) > 0
    ids = {r.requirement_id for r in requirements}
    assert "SEC-RIA-1.1" in ids
    assert "SEC-RIA-2.1" in ids  # Series 65
    assert "SEC-RIA-4.1" in ids  # Surety bond


def test_fca_requirements_include_smcr_and_cass() -> None:
    requirements = requirements_for(RegulatorAuthority.FCA_UK)
    ids = {r.requirement_id for r in requirements}
    assert "FCA-SUP-1.1" in ids
    assert "FCA-SUP-2.1" in ids  # SMCR
    assert "FCA-CASS-7.1" in ids  # client money
    assert "FCA-DISP-1.1" in ids  # FOS


def test_saudi_cma_empty_by_default() -> None:
    """Pin: authorities without bundled spec are empty for operator extension."""

    assert requirements_for(RegulatorAuthority.SAUDI_CMA) == ()


def test_uae_vara_empty_by_default() -> None:
    assert requirements_for(RegulatorAuthority.UAE_VARA) == ()


def test_singapore_mas_empty_by_default() -> None:
    assert requirements_for(RegulatorAuthority.SINGAPORE_MAS) == ()


def test_australia_asic_empty_by_default() -> None:
    assert requirements_for(RegulatorAuthority.AUSTRALIA_ASIC) == ()


# ---------------------------------------------------------------------------
# Requirement validation
# ---------------------------------------------------------------------------


def test_requirement_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="requirement_id"):
        RegistrationRequirement(
            authority=RegulatorAuthority.SEC_RIA,
            requirement_id="",
            category=RegistrationCategory.FORM_FILING,
            severity=RequirementSeverity.BLOCKING,
            description="x",
            expected_evidence=EvidenceKind.FORM_ADV_PART_1A,
        )


def test_requirement_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="description"):
        RegistrationRequirement(
            authority=RegulatorAuthority.SEC_RIA,
            requirement_id="X",
            category=RegistrationCategory.FORM_FILING,
            severity=RequirementSeverity.BLOCKING,
            description="",
            expected_evidence=EvidenceKind.FORM_ADV_PART_1A,
        )


# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------


def test_artifact_rejects_naive_last_updated() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        EvidenceArtifact(
            kind=EvidenceKind.FORM_ADV_PART_1A,
            present=True,
            last_updated=datetime(2026, 5, 1),
        )


def test_artifact_accepts_none_last_updated() -> None:
    a = EvidenceArtifact(kind=EvidenceKind.FORM_ADV_PART_1A, present=True, last_updated=None)
    assert a.last_updated is None


# ---------------------------------------------------------------------------
# READY — full evidence
# ---------------------------------------------------------------------------


def test_sec_ria_full_evidence_is_ready() -> None:
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA,
        evidence=_full_sec_ria_evidence(),
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.READY


def test_fca_full_evidence_is_ready() -> None:
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.FCA_UK,
        evidence=_full_fca_evidence(),
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.READY


# ---------------------------------------------------------------------------
# NOT_READY / GAPS / NEARLY_READY
# ---------------------------------------------------------------------------


def test_no_evidence_is_not_ready() -> None:
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA,
        evidence=(),
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.NOT_READY


def test_one_blocking_gap_is_gaps() -> None:
    """Drop SURETY_BOND → GAPS with 1 blocking unmet."""

    evidence = tuple(a for a in _full_sec_ria_evidence() if a.kind is not EvidenceKind.SURETY_BOND)
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA, evidence=evidence, now=_NOW
    )
    assert report.overall_level is ReadinessLevel.GAPS
    assert report.requirements_blocking_unmet == 1


def test_three_blocking_gaps_is_not_ready() -> None:
    """Drop three blocking → NOT_READY."""

    evidence = tuple(
        a
        for a in _full_sec_ria_evidence()
        if a.kind
        not in (
            EvidenceKind.SERIES_65_PASSED,
            EvidenceKind.SURETY_BOND,
            EvidenceKind.AML_PROGRAM,
        )
    )
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA, evidence=evidence, now=_NOW
    )
    assert report.overall_level is ReadinessLevel.NOT_READY


def test_only_warning_unmet_is_nearly_ready() -> None:
    """Drop ANNUAL_FORM_ADV_AMENDMENT (warning) → NEARLY_READY."""

    evidence = tuple(
        a for a in _full_sec_ria_evidence() if a.kind is not EvidenceKind.ANNUAL_FORM_ADV_AMENDMENT
    )
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA, evidence=evidence, now=_NOW
    )
    assert report.overall_level is ReadinessLevel.NEARLY_READY
    assert report.requirements_blocking_unmet == 0
    assert report.requirements_warning_unmet == 1


# ---------------------------------------------------------------------------
# Stale evidence — annual filing horizon
# ---------------------------------------------------------------------------


def test_stale_evidence_drops_to_nearly_ready() -> None:
    """Pin: 400d-old artifact > 365d default → stale."""

    fresh_others = tuple(
        a for a in _full_sec_ria_evidence() if a.kind is not EvidenceKind.FORM_ADV_PART_1A
    )
    stale_one = (
        EvidenceArtifact(
            kind=EvidenceKind.FORM_ADV_PART_1A,
            present=True,
            last_updated=_NOW - timedelta(days=400),
        ),
    )
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA,
        evidence=fresh_others + stale_one,
        now=_NOW,
    )
    assert report.requirements_stale == 1
    assert report.overall_level is ReadinessLevel.NEARLY_READY


def test_artifact_at_exactly_365_days_is_stale() -> None:
    """Pin: at exactly the horizon, the artifact is stale (boundary inclusive)."""

    artifact = (
        EvidenceArtifact(
            kind=EvidenceKind.FORM_ADV_PART_1A,
            present=True,
            last_updated=_NOW - timedelta(days=365),
        ),
    )
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA, evidence=artifact, now=_NOW
    )
    stale = [a for a in report.assessments if a.is_stale]
    assert len(stale) == 1


def test_no_last_updated_is_stale() -> None:
    """Pin: present but no timestamp → stale."""

    artifact = (
        EvidenceArtifact(
            kind=EvidenceKind.FORM_ADV_PART_1A,
            present=True,
            last_updated=None,
        ),
    )
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA, evidence=artifact, now=_NOW
    )
    stale = [a for a in report.assessments if a.is_stale]
    assert len(stale) >= 1


def test_strict_180_day_policy_catches_300_day_artifact() -> None:
    strict = RegistrationPolicy(staleness_horizon_days=180)
    artifact = (
        EvidenceArtifact(
            kind=EvidenceKind.FORM_ADV_PART_1A,
            present=True,
            last_updated=_NOW - timedelta(days=300),
        ),
    )
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA,
        evidence=artifact,
        now=_NOW,
        policy=strict,
    )
    assert report.requirements_stale == 1


# ---------------------------------------------------------------------------
# Empty authority requirement set
# ---------------------------------------------------------------------------


def test_empty_authority_returns_not_ready_with_note() -> None:
    """Pin: an authority with no loaded spec → NOT_READY with note."""

    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SAUDI_CMA, evidence=(), now=_NOW
    )
    assert report.overall_level is ReadinessLevel.NOT_READY
    assert any("requirement set" in n for n in report.notes)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_evaluate_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate_registration_readiness(
            authority=RegulatorAuthority.SEC_RIA,
            evidence=(),
            now=datetime(2026, 5, 1),
        )


# ---------------------------------------------------------------------------
# Per-requirement assessments carry through
# ---------------------------------------------------------------------------


def test_assessments_count_matches_requirements() -> None:
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA,
        evidence=_full_sec_ria_evidence(),
        now=_NOW,
    )
    assert len(report.assessments) == report.requirements_total


def test_unmet_assessment_carries_note() -> None:
    """Pin: unmet assessment includes 'missing evidence (BLOCKING)' note."""

    evidence = tuple(a for a in _full_sec_ria_evidence() if a.kind is not EvidenceKind.SURETY_BOND)
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA, evidence=evidence, now=_NOW
    )
    unmet = [a for a in report.assessments if not a.is_met]
    assert len(unmet) == 1
    assert "missing" in unmet[0].notes
    assert "BLOCKING" in unmet[0].notes


def test_stale_assessment_carries_note() -> None:
    artifact = (
        EvidenceArtifact(
            kind=EvidenceKind.FORM_ADV_PART_1A,
            present=True,
            last_updated=_NOW - timedelta(days=400),
        ),
    )
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA, evidence=artifact, now=_NOW
    )
    stale = [a for a in report.assessments if a.is_stale]
    assert any("stale" in a.notes for a in stale)


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_requirement_is_frozen() -> None:
    r = RegistrationRequirement(
        authority=RegulatorAuthority.SEC_RIA,
        requirement_id="X",
        category=RegistrationCategory.FORM_FILING,
        severity=RequirementSeverity.BLOCKING,
        description="x",
        expected_evidence=EvidenceKind.FORM_ADV_PART_1A,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.requirement_id = "Y"  # type: ignore[misc]


def test_artifact_is_frozen() -> None:
    a = _artifact()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.present = False  # type: ignore[misc]


def test_report_is_frozen() -> None:
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA, evidence=(), now=_NOW
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.overall_level = ReadinessLevel.READY  # type: ignore[misc]


def test_policy_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_POLICY.staleness_horizon_days = 90  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned
# ---------------------------------------------------------------------------


def test_authority_string_values() -> None:
    assert RegulatorAuthority.SEC_RIA.value == "sec_ria"
    assert RegulatorAuthority.FCA_UK.value == "fca_uk"
    assert RegulatorAuthority.SAUDI_CMA.value == "saudi_cma"
    assert RegulatorAuthority.UAE_VARA.value == "uae_vara"
    assert RegulatorAuthority.SINGAPORE_MAS.value == "singapore_mas"
    assert RegulatorAuthority.AUSTRALIA_ASIC.value == "australia_asic"


def test_category_string_values() -> None:
    assert RegistrationCategory.FORM_FILING.value == "form_filing"
    assert RegistrationCategory.BACKGROUND_CHECKS.value == "background_checks"
    assert RegistrationCategory.EXAMS_LICENSURE.value == "exams_licensure"
    assert RegistrationCategory.AML.value == "aml"
    assert RegistrationCategory.CLIENT_MONEY.value == "client_money"


def test_severity_string_values() -> None:
    assert RequirementSeverity.BLOCKING.value == "blocking"
    assert RequirementSeverity.WARNING.value == "warning"


def test_evidence_kind_string_values() -> None:
    assert EvidenceKind.FORM_ADV_PART_1A.value == "form_adv_part_1a"
    assert EvidenceKind.SERIES_65_PASSED.value == "series_65_passed"
    assert EvidenceKind.FCA_SUP_FORM.value == "fca_sup_form"
    assert EvidenceKind.FOS_MEMBERSHIP.value == "fos_membership"


def test_level_string_values() -> None:
    assert ReadinessLevel.READY.value == "ready"
    assert ReadinessLevel.NEARLY_READY.value == "nearly_ready"
    assert ReadinessLevel.GAPS.value == "gaps"
    assert ReadinessLevel.NOT_READY.value == "not_ready"


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


def test_render_ready_report() -> None:
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA,
        evidence=_full_sec_ria_evidence(),
        now=_NOW,
    )
    text = render_readiness_report(report)
    assert "✅" in text
    assert "READY" in text
    assert "sec_ria" in text


def test_render_not_ready_report() -> None:
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA, evidence=(), now=_NOW
    )
    text = render_readiness_report(report)
    assert "❌" in text
    assert "NOT_READY" in text


def test_render_includes_per_requirement_breakdown() -> None:
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA,
        evidence=_full_sec_ria_evidence(),
        now=_NOW,
    )
    text = render_readiness_report(report)
    assert "SEC-RIA-1.1" in text
    assert "per-requirement breakdown" in text


def test_render_does_not_include_operator_pii() -> None:
    """Pin: render never includes firm name / CRD number / executive PII."""

    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA,
        evidence=_full_sec_ria_evidence(),
        now=_NOW,
    )
    text = render_readiness_report(report)
    assert "firm_name" not in text
    assert "crd_number" not in text or "CRD_NUMBER" in text  # the artifact-kind label is allowed
    # but no actual CRD number values
    assert "12345" not in text


# ---------------------------------------------------------------------------
# End-to-end realistic scenarios
# ---------------------------------------------------------------------------


def test_typical_pre_filing_sec_ria_with_warning() -> None:
    """Operator preparing first SEC RIA filing has all blocking met but
    no annual amendment yet (warning) → NEARLY_READY."""

    evidence = (
        _artifact(kind=EvidenceKind.FORM_ADV_PART_1A),
        _artifact(kind=EvidenceKind.FORM_ADV_PART_2A),
        _artifact(kind=EvidenceKind.FORM_ADV_PART_2B),
        _artifact(kind=EvidenceKind.SERIES_65_PASSED),
        _artifact(kind=EvidenceKind.EXEC_BACKGROUND_CHECK),
        _artifact(kind=EvidenceKind.SURETY_BOND),
        _artifact(kind=EvidenceKind.AML_PROGRAM),
        _artifact(kind=EvidenceKind.COMPLIANCE_MANUAL),
        _artifact(kind=EvidenceKind.RECORDKEEPING_PROCEDURES),
        # ANNUAL_FORM_ADV_AMENDMENT missing — warning severity
    )
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA, evidence=evidence, now=_NOW
    )
    assert report.overall_level is ReadinessLevel.NEARLY_READY
    assert report.requirements_blocking_unmet == 0
    assert report.requirements_warning_unmet == 1


def test_uk_fca_filing_with_smcr_gap() -> None:
    """FCA filing missing SMCR-certified persons → GAPS (1 blocking)."""

    evidence = tuple(
        a for a in _full_fca_evidence() if a.kind is not EvidenceKind.SMCR_CERTIFIED_PERSONS
    )
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.FCA_UK, evidence=evidence, now=_NOW
    )
    assert report.overall_level is ReadinessLevel.GAPS
    assert report.requirements_blocking_unmet == 1


def test_assessment_has_correct_count() -> None:
    """Pin: assessment count equals the requirements_total."""

    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA,
        evidence=_full_sec_ria_evidence(),
        now=_NOW,
    )
    assert len(report.assessments) == report.requirements_total


def test_met_pct_aggregates() -> None:
    report = evaluate_registration_readiness(
        authority=RegulatorAuthority.SEC_RIA,
        evidence=_full_sec_ria_evidence(),
        now=_NOW,
    )
    assert report.met_pct == 100.0
