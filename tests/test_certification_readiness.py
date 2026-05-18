"""Tests for the halal certification readiness aggregator."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.halal.certification_readiness import (
    DEFAULT_POLICY,
    CertificationBody,
    CertificationRequirement,
    EvidenceArtifact,
    EvidenceArtifactKind,
    ReadinessLevel,
    ReadinessPolicy,
    RequirementCategory,
    RequirementSeverity,
    evaluate_readiness,
    render_readiness_report,
    requirements_for,
)

_NOW = datetime(2026, 5, 1, tzinfo=UTC)


def _artifact(
    *,
    kind: EvidenceArtifactKind = EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
    present: bool = True,
    last_updated: datetime | None = None,
) -> EvidenceArtifact:
    return EvidenceArtifact(
        kind=kind,
        present=present,
        last_updated=last_updated if last_updated is not None else _NOW - timedelta(days=30),
    )


def _aaoifi_complete_evidence() -> tuple[EvidenceArtifact, ...]:
    """All AAOIFI artifacts present + fresh."""

    return (
        _artifact(kind=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS),
        _artifact(kind=EvidenceArtifactKind.SIGNED_TRADE_RECEIPTS),
        _artifact(kind=EvidenceArtifactKind.PURIFICATION_LEDGER),
        _artifact(kind=EvidenceArtifactKind.SSB_RULINGS),
        _artifact(kind=EvidenceArtifactKind.SSB_QUARTERLY_REVIEWS),
        _artifact(kind=EvidenceArtifactKind.SHARIAH_AUDIT_REPORT),
        _artifact(kind=EvidenceArtifactKind.PUBLIC_PRIVACY_POLICY),
    )


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_default_policy_staleness_horizon() -> None:
    assert DEFAULT_POLICY.staleness_horizon_days == 180


def test_policy_rejects_zero_staleness() -> None:
    with pytest.raises(ValueError, match="staleness_horizon_days"):
        ReadinessPolicy(staleness_horizon_days=0)


def test_policy_rejects_negative_staleness() -> None:
    with pytest.raises(ValueError, match="staleness_horizon_days"):
        ReadinessPolicy(staleness_horizon_days=-1)


# ---------------------------------------------------------------------------
# Requirement set queries
# ---------------------------------------------------------------------------


def test_aaoifi_requirements_present() -> None:
    requirements = requirements_for(CertificationBody.AAOIFI)
    assert len(requirements) > 0
    spec_ids = {r.spec_id for r in requirements}
    assert "AAOIFI-S21-2.1" in spec_ids
    assert "AAOIFI-GS-1.1" in spec_ids


def test_tadawul_requirements_include_kyc() -> None:
    """Pin: Tadawul requires Saudi-jurisdiction KYC verification."""

    requirements = requirements_for(CertificationBody.SAUDI_TADAWUL)
    spec_ids = {r.spec_id for r in requirements}
    assert "TADAWUL-KYC-2.1" in spec_ids
    assert "TADAWUL-AML-2.2" in spec_ids


def test_malaysian_sac_requirements_include_screening() -> None:
    requirements = requirements_for(CertificationBody.MALAYSIAN_SAC)
    spec_ids = {r.spec_id for r in requirements}
    assert "SAC-RES-1" in spec_ids


def test_bahrain_cbb_requirements_empty_by_default() -> None:
    """Pin: bodies without bundled spec are empty; operator extends."""

    requirements = requirements_for(CertificationBody.BAHRAIN_CBB)
    assert requirements == ()


def test_indonesia_dsn_mui_requirements_empty_by_default() -> None:
    requirements = requirements_for(CertificationBody.INDONESIA_DSN_MUI)
    assert requirements == ()


# ---------------------------------------------------------------------------
# Requirement validation
# ---------------------------------------------------------------------------


def test_requirement_rejects_empty_spec_id() -> None:
    with pytest.raises(ValueError, match="spec_id"):
        CertificationRequirement(
            body=CertificationBody.AAOIFI,
            spec_id="",
            category=RequirementCategory.SCREENING,
            severity=RequirementSeverity.BLOCKING,
            description="x",
            expected_artifact=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
        )


def test_requirement_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="description"):
        CertificationRequirement(
            body=CertificationBody.AAOIFI,
            spec_id="x",
            category=RequirementCategory.SCREENING,
            severity=RequirementSeverity.BLOCKING,
            description="",
            expected_artifact=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
        )


# ---------------------------------------------------------------------------
# EvidenceArtifact validation
# ---------------------------------------------------------------------------


def test_artifact_rejects_naive_last_updated() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        EvidenceArtifact(
            kind=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
            present=True,
            last_updated=datetime(2026, 5, 1),
        )


def test_artifact_accepts_none_last_updated() -> None:
    a = EvidenceArtifact(
        kind=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
        present=True,
        last_updated=None,
    )
    assert a.last_updated is None


# ---------------------------------------------------------------------------
# READY — all requirements met, all fresh
# ---------------------------------------------------------------------------


def test_aaoifi_complete_evidence_is_ready() -> None:
    """Best-case: all AAOIFI artifacts present + recent → READY."""

    report = evaluate_readiness(
        body=CertificationBody.AAOIFI,
        evidence=_aaoifi_complete_evidence(),
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.READY
    assert report.requirements_met == report.requirements_total
    assert report.requirements_blocking_unmet == 0
    assert report.requirements_warning_unmet == 0
    assert report.requirements_stale == 0


def test_aaoifi_complete_met_pct_is_100() -> None:
    report = evaluate_readiness(
        body=CertificationBody.AAOIFI,
        evidence=_aaoifi_complete_evidence(),
        now=_NOW,
    )
    assert report.met_pct == 100.0


# ---------------------------------------------------------------------------
# NOT_READY — multiple blocking gaps
# ---------------------------------------------------------------------------


def test_no_evidence_is_not_ready() -> None:
    """Pin: empty evidence → NOT_READY with all blocking unmet."""

    report = evaluate_readiness(
        body=CertificationBody.AAOIFI,
        evidence=(),
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.NOT_READY
    assert report.requirements_met == 0
    assert report.requirements_blocking_unmet > 0


def test_three_blocking_gaps_is_not_ready() -> None:
    """Three+ blocking unmet → NOT_READY (the GAPS upper threshold is 2)."""

    # Provide only 3 of 6 blocking artifacts
    evidence = (
        _artifact(kind=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS),
        _artifact(kind=EvidenceArtifactKind.SIGNED_TRADE_RECEIPTS),
        _artifact(kind=EvidenceArtifactKind.PURIFICATION_LEDGER),
        _artifact(kind=EvidenceArtifactKind.PUBLIC_PRIVACY_POLICY),
    )
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=evidence, now=_NOW)
    assert report.overall_level is ReadinessLevel.NOT_READY


# ---------------------------------------------------------------------------
# GAPS — one or two blocking gaps
# ---------------------------------------------------------------------------


def test_one_blocking_gap_is_gaps() -> None:
    """Missing one blocking artifact (SHARIAH_AUDIT_REPORT) → GAPS."""

    evidence = tuple(
        a
        for a in _aaoifi_complete_evidence()
        if a.kind is not EvidenceArtifactKind.SHARIAH_AUDIT_REPORT
    )
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=evidence, now=_NOW)
    assert report.overall_level is ReadinessLevel.GAPS
    assert report.requirements_blocking_unmet == 1


def test_two_blocking_gaps_is_gaps() -> None:
    """Pin: 2 blocking unmet → GAPS (boundary)."""

    evidence = tuple(
        a
        for a in _aaoifi_complete_evidence()
        if a.kind
        not in (
            EvidenceArtifactKind.SHARIAH_AUDIT_REPORT,
            EvidenceArtifactKind.PURIFICATION_LEDGER,
        )
    )
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=evidence, now=_NOW)
    assert report.overall_level is ReadinessLevel.GAPS
    assert report.requirements_blocking_unmet == 2


# ---------------------------------------------------------------------------
# NEARLY_READY — only warnings unmet
# ---------------------------------------------------------------------------


def test_only_warnings_unmet_is_nearly_ready() -> None:
    """Pin: all blocking met but a WARNING unmet → NEARLY_READY."""

    # Drop the privacy policy (which is a WARNING-severity requirement)
    evidence = tuple(
        a
        for a in _aaoifi_complete_evidence()
        if a.kind is not EvidenceArtifactKind.PUBLIC_PRIVACY_POLICY
    )
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=evidence, now=_NOW)
    assert report.overall_level is ReadinessLevel.NEARLY_READY
    assert report.requirements_blocking_unmet == 0
    assert report.requirements_warning_unmet == 1


# ---------------------------------------------------------------------------
# Stale artifacts → NEARLY_READY (warning-equivalent)
# ---------------------------------------------------------------------------


def test_stale_artifact_drops_to_nearly_ready() -> None:
    """Pin: artifact > 180d old triggers stale warning."""

    stale = (
        EvidenceArtifact(
            kind=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
            present=True,
            last_updated=_NOW - timedelta(days=200),
        ),
        _artifact(kind=EvidenceArtifactKind.SIGNED_TRADE_RECEIPTS),
        _artifact(kind=EvidenceArtifactKind.PURIFICATION_LEDGER),
        _artifact(kind=EvidenceArtifactKind.SSB_RULINGS),
        _artifact(kind=EvidenceArtifactKind.SSB_QUARTERLY_REVIEWS),
        _artifact(kind=EvidenceArtifactKind.SHARIAH_AUDIT_REPORT),
        _artifact(kind=EvidenceArtifactKind.PUBLIC_PRIVACY_POLICY),
    )
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=stale, now=_NOW)
    assert report.requirements_stale == 1
    # Still nearly_ready (no missing blocking)
    assert report.overall_level is ReadinessLevel.NEARLY_READY


def test_artifact_at_exactly_180_days_is_stale() -> None:
    """Pin: at exactly the staleness horizon, the artifact is stale."""

    stale = (
        EvidenceArtifact(
            kind=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
            present=True,
            last_updated=_NOW - timedelta(days=180),
        ),
    )
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=stale, now=_NOW)
    assert report.requirements_stale == 1


def test_artifact_just_inside_horizon_is_fresh() -> None:
    """Pin: < 180d is fresh."""

    fresh = (
        EvidenceArtifact(
            kind=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
            present=True,
            last_updated=_NOW - timedelta(days=179),
        ),
        # Add the rest as fresh too for full evidence
        _artifact(kind=EvidenceArtifactKind.SIGNED_TRADE_RECEIPTS),
        _artifact(kind=EvidenceArtifactKind.PURIFICATION_LEDGER),
        _artifact(kind=EvidenceArtifactKind.SSB_RULINGS),
        _artifact(kind=EvidenceArtifactKind.SSB_QUARTERLY_REVIEWS),
        _artifact(kind=EvidenceArtifactKind.SHARIAH_AUDIT_REPORT),
        _artifact(kind=EvidenceArtifactKind.PUBLIC_PRIVACY_POLICY),
    )
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=fresh, now=_NOW)
    assert report.requirements_stale == 0
    assert report.overall_level is ReadinessLevel.READY


def test_no_last_updated_is_stale() -> None:
    """Pin: present but no timestamp → treated as stale (warning)."""

    no_timestamp = (
        EvidenceArtifact(
            kind=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
            present=True,
            last_updated=None,
        ),
    )
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=no_timestamp, now=_NOW)
    assert report.requirements_stale == 1


def test_strict_staleness_policy_flips_verdict() -> None:
    """Pin: 90-day staleness horizon catches a 100-day-old artifact."""

    strict = ReadinessPolicy(staleness_horizon_days=90)
    artifact = (
        EvidenceArtifact(
            kind=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
            present=True,
            last_updated=_NOW - timedelta(days=100),
        ),
    )
    report = evaluate_readiness(
        body=CertificationBody.AAOIFI,
        evidence=artifact,
        now=_NOW,
        policy=strict,
    )
    assert report.requirements_stale == 1


# ---------------------------------------------------------------------------
# Empty body requirements → NOT_READY
# ---------------------------------------------------------------------------


def test_empty_body_requirements_returns_not_ready() -> None:
    """Pin: a body with no loaded spec → NOT_READY with note."""

    report = evaluate_readiness(
        body=CertificationBody.BAHRAIN_CBB,
        evidence=(),
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.NOT_READY
    assert report.requirements_total == 0
    assert any("requirement set" in n for n in report.notes)


# ---------------------------------------------------------------------------
# evaluate_readiness validation
# ---------------------------------------------------------------------------


def test_evaluate_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate_readiness(
            body=CertificationBody.AAOIFI,
            evidence=(),
            now=datetime(2026, 5, 1),
        )


# ---------------------------------------------------------------------------
# Per-requirement assessments carry through
# ---------------------------------------------------------------------------


def test_assessments_count_matches_requirements() -> None:
    report = evaluate_readiness(
        body=CertificationBody.AAOIFI,
        evidence=_aaoifi_complete_evidence(),
        now=_NOW,
    )
    assert len(report.assessments) == report.requirements_total


def test_unmet_assessment_carries_note() -> None:
    """Pin: unmet assessment includes the 'missing artifact' note."""

    # Drop one blocking artifact
    evidence = tuple(
        a
        for a in _aaoifi_complete_evidence()
        if a.kind is not EvidenceArtifactKind.SHARIAH_AUDIT_REPORT
    )
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=evidence, now=_NOW)
    unmet = [a for a in report.assessments if not a.is_met]
    assert len(unmet) == 1
    assert "missing" in unmet[0].notes
    assert "BLOCKING" in unmet[0].notes


def test_stale_assessment_carries_note() -> None:
    stale = (
        EvidenceArtifact(
            kind=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
            present=True,
            last_updated=_NOW - timedelta(days=300),
        ),
    )
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=stale, now=_NOW)
    stale_assessments = [a for a in report.assessments if a.is_stale]
    assert len(stale_assessments) >= 1
    assert "stale" in stale_assessments[0].notes


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_requirement_is_frozen() -> None:
    r = CertificationRequirement(
        body=CertificationBody.AAOIFI,
        spec_id="X",
        category=RequirementCategory.SCREENING,
        severity=RequirementSeverity.BLOCKING,
        description="x",
        expected_artifact=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.spec_id = "Y"  # type: ignore[misc]


def test_artifact_is_frozen() -> None:
    a = _artifact()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.present = False  # type: ignore[misc]


def test_report_is_frozen() -> None:
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=(), now=_NOW)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.overall_level = ReadinessLevel.READY  # type: ignore[misc]


def test_policy_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_POLICY.staleness_horizon_days = 30  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB stability
# ---------------------------------------------------------------------------


def test_body_string_values() -> None:
    assert CertificationBody.AAOIFI.value == "aaoifi"
    assert CertificationBody.SAUDI_TADAWUL.value == "saudi_tadawul"
    assert CertificationBody.MALAYSIAN_SAC.value == "malaysian_sac"
    assert CertificationBody.BAHRAIN_CBB.value == "bahrain_cbb"
    assert CertificationBody.INDONESIA_DSN_MUI.value == "indonesia_dsn_mui"


def test_category_string_values() -> None:
    assert RequirementCategory.SCREENING.value == "screening"
    assert RequirementCategory.AUDIT.value == "audit"
    assert RequirementCategory.PURIFICATION.value == "purification"
    assert RequirementCategory.SSB_GOVERNANCE.value == "ssb_governance"
    assert RequirementCategory.KYC_AML.value == "kyc_aml"
    assert RequirementCategory.DOCUMENTATION.value == "documentation"


def test_severity_string_values() -> None:
    assert RequirementSeverity.BLOCKING.value == "blocking"
    assert RequirementSeverity.WARNING.value == "warning"


def test_artifact_kind_string_values() -> None:
    assert EvidenceArtifactKind.HALAL_SCREENER_DECISIONS.value == "halal_screener_decisions"
    assert EvidenceArtifactKind.SIGNED_TRADE_RECEIPTS.value == "signed_trade_receipts"
    assert EvidenceArtifactKind.PURIFICATION_LEDGER.value == "purification_ledger"
    assert EvidenceArtifactKind.SSB_RULINGS.value == "ssb_rulings"
    assert EvidenceArtifactKind.KYC_VERIFIED_USERS.value == "kyc_verified_users"


def test_level_string_values() -> None:
    assert ReadinessLevel.READY.value == "ready"
    assert ReadinessLevel.NEARLY_READY.value == "nearly_ready"
    assert ReadinessLevel.GAPS.value == "gaps"
    assert ReadinessLevel.NOT_READY.value == "not_ready"


# ---------------------------------------------------------------------------
# Render output — pinned no-PII contract
# ---------------------------------------------------------------------------


def test_render_ready_report() -> None:
    report = evaluate_readiness(
        body=CertificationBody.AAOIFI,
        evidence=_aaoifi_complete_evidence(),
        now=_NOW,
    )
    text = render_readiness_report(report)
    assert "✅" in text
    assert "READY" in text
    assert "aaoifi" in text
    assert "100.0%" in text


def test_render_not_ready_report() -> None:
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=(), now=_NOW)
    text = render_readiness_report(report)
    assert "❌" in text
    assert "NOT_READY" in text
    assert "blocking gaps" in text


def test_render_gaps_report() -> None:
    evidence = tuple(
        a
        for a in _aaoifi_complete_evidence()
        if a.kind is not EvidenceArtifactKind.SHARIAH_AUDIT_REPORT
    )
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=evidence, now=_NOW)
    text = render_readiness_report(report)
    assert "⚠️" in text
    assert "GAPS" in text


def test_render_nearly_ready_report() -> None:
    evidence = tuple(
        a
        for a in _aaoifi_complete_evidence()
        if a.kind is not EvidenceArtifactKind.PUBLIC_PRIVACY_POLICY
    )
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=evidence, now=_NOW)
    text = render_readiness_report(report)
    assert "🟢" in text
    assert "NEARLY_READY" in text


def test_render_includes_per_requirement_breakdown() -> None:
    report = evaluate_readiness(
        body=CertificationBody.AAOIFI,
        evidence=_aaoifi_complete_evidence(),
        now=_NOW,
    )
    text = render_readiness_report(report)
    assert "per-requirement breakdown" in text
    assert "AAOIFI-S21-2.1" in text


def test_render_does_not_include_operator_pii() -> None:
    """Pin: render never includes operator PII / member names / KYC secrets."""

    report = evaluate_readiness(
        body=CertificationBody.AAOIFI,
        evidence=_aaoifi_complete_evidence(),
        now=_NOW,
    )
    text = render_readiness_report(report)
    # No member names, KYC secrets, IDs in render
    assert "user_id" not in text
    assert "ssn" not in text.lower()
    assert "passport" not in text.lower()


# ---------------------------------------------------------------------------
# End-to-end realistic scenarios
# ---------------------------------------------------------------------------


def test_typical_pre_application_readiness_flow() -> None:
    """An operator preparing for AAOIFI application:
    has screening + signing + purification + SSB rulings — but
    missing the annual external shariah audit report and the
    quarterly review documentation.
    """

    evidence = (
        _artifact(kind=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS),
        _artifact(kind=EvidenceArtifactKind.SIGNED_TRADE_RECEIPTS),
        _artifact(kind=EvidenceArtifactKind.PURIFICATION_LEDGER),
        _artifact(kind=EvidenceArtifactKind.SSB_RULINGS),
        _artifact(kind=EvidenceArtifactKind.PUBLIC_PRIVACY_POLICY),
    )
    report = evaluate_readiness(body=CertificationBody.AAOIFI, evidence=evidence, now=_NOW)
    # 2 blocking gaps (SSB_QUARTERLY_REVIEWS + SHARIAH_AUDIT_REPORT) → GAPS
    assert report.requirements_blocking_unmet == 2
    assert report.overall_level is ReadinessLevel.GAPS


def test_tadawul_application_readiness() -> None:
    """Tadawul-readiness: the Tadawul requirements include KYC + AML."""

    evidence = (
        _artifact(kind=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS),
        _artifact(kind=EvidenceArtifactKind.KYC_VERIFIED_USERS),
        _artifact(kind=EvidenceArtifactKind.AML_SANCTIONS_SCREENING),
        _artifact(kind=EvidenceArtifactKind.ANNUAL_AUDIT_REPORT),
    )
    report = evaluate_readiness(body=CertificationBody.SAUDI_TADAWUL, evidence=evidence, now=_NOW)
    assert report.overall_level is ReadinessLevel.READY


def test_malaysian_sac_readiness_with_warning_only() -> None:
    """SAC-readiness with the warning-severity purification gap."""

    evidence = (
        _artifact(kind=EvidenceArtifactKind.HALAL_SCREENER_DECISIONS),
        _artifact(kind=EvidenceArtifactKind.SSB_RULINGS),
        # Skip PURIFICATION_LEDGER which is a warning-severity req
    )
    report = evaluate_readiness(body=CertificationBody.MALAYSIAN_SAC, evidence=evidence, now=_NOW)
    # blocking met, only warning unmet → NEARLY_READY
    assert report.overall_level is ReadinessLevel.NEARLY_READY
    assert report.requirements_blocking_unmet == 0
    assert report.requirements_warning_unmet == 1
