"""Tests for the SOC 2 readiness aggregator."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.web.soc2_readiness import (
    DEFAULT_POLICY,
    AuditType,
    ControlCategory,
    ControlSeverity,
    EvidenceArtifact,
    EvidenceKind,
    ReadinessLevel,
    SOC2Control,
    SOC2ReadinessPolicy,
    TrustServiceCategory,
    controls_for,
    evaluate_soc2_readiness,
    render_readiness_report,
)

_NOW = datetime(2026, 5, 1, tzinfo=UTC)


def _artifact(
    *,
    kind: EvidenceKind = EvidenceKind.ACCESS_LOGS,
    present: bool = True,
    last_updated: datetime | None = None,
) -> EvidenceArtifact:
    return EvidenceArtifact(
        kind=kind,
        present=present,
        last_updated=last_updated if last_updated is not None else _NOW - timedelta(days=30),
    )


def _full_security_evidence() -> tuple[EvidenceArtifact, ...]:
    """All Security-trust-service evidence artifacts present + fresh."""

    return (
        _artifact(kind=EvidenceKind.ACCESS_LOGS),
        _artifact(kind=EvidenceKind.MFA_ENABLED),
        _artifact(kind=EvidenceKind.EMPLOYEE_ONBOARDING),
        _artifact(kind=EvidenceKind.VULNERABILITY_SCANS),
        _artifact(kind=EvidenceKind.INCIDENT_REPORTS),
        _artifact(kind=EvidenceKind.PR_REVIEW_LOGS),
        _artifact(kind=EvidenceKind.VENDOR_SOC2_REPORTS),
        _artifact(kind=EvidenceKind.RISK_REGISTER),
        _artifact(kind=EvidenceKind.SECURITY_TRAINING),
        _artifact(kind=EvidenceKind.PEN_TEST_REPORTS),
    )


def _full_availability_evidence() -> tuple[EvidenceArtifact, ...]:
    return (
        _artifact(kind=EvidenceKind.UPTIME_MONITORING),
        _artifact(kind=EvidenceKind.BACKUP_RECORDS),
        _artifact(kind=EvidenceKind.DR_DRILL_REPORTS),
        _artifact(kind=EvidenceKind.STATUS_PAGE),
    )


def _full_confidentiality_evidence() -> tuple[EvidenceArtifact, ...]:
    return (
        _artifact(kind=EvidenceKind.DATA_CLASSIFICATION_POLICY),
        _artifact(kind=EvidenceKind.ENCRYPTION_AT_REST),
        _artifact(kind=EvidenceKind.ENCRYPTION_IN_TRANSIT),
    )


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_default_policy_type_ii_horizon() -> None:
    p = DEFAULT_POLICY
    assert p.audit_type is AuditType.TYPE_II
    assert p.evidence_horizon_days == 365


def test_policy_rejects_zero_horizon() -> None:
    with pytest.raises(ValueError, match="evidence_horizon_days"):
        SOC2ReadinessPolicy(evidence_horizon_days=0)


def test_policy_rejects_negative_horizon() -> None:
    with pytest.raises(ValueError, match="evidence_horizon_days"):
        SOC2ReadinessPolicy(evidence_horizon_days=-1)


def test_policy_accepts_type_i_with_short_horizon() -> None:
    p = SOC2ReadinessPolicy(audit_type=AuditType.TYPE_I, evidence_horizon_days=30)
    assert p.audit_type is AuditType.TYPE_I


# ---------------------------------------------------------------------------
# Control set queries
# ---------------------------------------------------------------------------


def test_security_controls_present() -> None:
    controls = controls_for(TrustServiceCategory.SECURITY)
    assert len(controls) > 0
    ids = {c.control_id for c in controls}
    assert "CC6.1" in ids
    assert "CC6.2" in ids
    assert "CC8.1" in ids


def test_availability_controls_include_backups() -> None:
    controls = controls_for(TrustServiceCategory.AVAILABILITY)
    ids = {c.control_id for c in controls}
    assert "A1.1" in ids
    assert "A1.2" in ids


def test_confidentiality_controls_include_encryption() -> None:
    controls = controls_for(TrustServiceCategory.CONFIDENTIALITY)
    ids = {c.control_id for c in controls}
    assert "C1.2" in ids  # encryption at rest
    assert "C1.3" in ids  # encryption in transit


def test_processing_integrity_empty_by_default() -> None:
    assert controls_for(TrustServiceCategory.PROCESSING_INTEGRITY) == ()


def test_privacy_empty_by_default() -> None:
    """Pin: Privacy is partly covered by Wave 11.D; SOC 2 spec is operator-extension."""

    assert controls_for(TrustServiceCategory.PRIVACY) == ()


# ---------------------------------------------------------------------------
# Control validation
# ---------------------------------------------------------------------------


def test_control_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="control_id"):
        SOC2Control(
            trust_service=TrustServiceCategory.SECURITY,
            control_id="",
            category=ControlCategory.ACCESS_CONTROL,
            severity=ControlSeverity.BLOCKING,
            description="x",
            expected_evidence=EvidenceKind.ACCESS_LOGS,
        )


def test_control_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="description"):
        SOC2Control(
            trust_service=TrustServiceCategory.SECURITY,
            control_id="X",
            category=ControlCategory.ACCESS_CONTROL,
            severity=ControlSeverity.BLOCKING,
            description="",
            expected_evidence=EvidenceKind.ACCESS_LOGS,
        )


# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------


def test_artifact_rejects_naive_last_updated() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        EvidenceArtifact(
            kind=EvidenceKind.ACCESS_LOGS,
            present=True,
            last_updated=datetime(2026, 5, 1),
        )


def test_artifact_accepts_none_last_updated() -> None:
    a = EvidenceArtifact(kind=EvidenceKind.ACCESS_LOGS, present=True, last_updated=None)
    assert a.last_updated is None


# ---------------------------------------------------------------------------
# READY — full evidence
# ---------------------------------------------------------------------------


def test_security_full_evidence_is_ready() -> None:
    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=_full_security_evidence(),
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.READY


def test_security_plus_availability_full_is_ready() -> None:
    report = evaluate_soc2_readiness(
        trust_services=(
            TrustServiceCategory.SECURITY,
            TrustServiceCategory.AVAILABILITY,
        ),
        evidence=_full_security_evidence() + _full_availability_evidence(),
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.READY


def test_three_services_full_is_ready() -> None:
    report = evaluate_soc2_readiness(
        trust_services=(
            TrustServiceCategory.SECURITY,
            TrustServiceCategory.AVAILABILITY,
            TrustServiceCategory.CONFIDENTIALITY,
        ),
        evidence=(
            _full_security_evidence()
            + _full_availability_evidence()
            + _full_confidentiality_evidence()
        ),
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.READY


# ---------------------------------------------------------------------------
# NOT_READY / GAPS / NEARLY_READY
# ---------------------------------------------------------------------------


def test_no_evidence_is_not_ready() -> None:
    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=(),
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.NOT_READY


def test_one_blocking_gap_is_gaps() -> None:
    """Drop one blocking artifact (PR_REVIEW_LOGS) → GAPS."""

    evidence = tuple(
        a for a in _full_security_evidence() if a.kind is not EvidenceKind.PR_REVIEW_LOGS
    )
    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=evidence,
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.GAPS


def test_three_blocking_gaps_is_not_ready() -> None:
    """Drop three blocking artifacts → NOT_READY."""

    evidence = tuple(
        a
        for a in _full_security_evidence()
        if a.kind
        not in (
            EvidenceKind.PR_REVIEW_LOGS,
            EvidenceKind.RISK_REGISTER,
            EvidenceKind.VULNERABILITY_SCANS,
        )
    )
    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=evidence,
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.NOT_READY


def test_only_warnings_unmet_is_nearly_ready() -> None:
    """Drop the warning-severity artifacts → NEARLY_READY."""

    evidence = tuple(
        a
        for a in _full_security_evidence()
        if a.kind
        not in (
            EvidenceKind.VENDOR_SOC2_REPORTS,
            EvidenceKind.SECURITY_TRAINING,
            EvidenceKind.PEN_TEST_REPORTS,
        )
    )
    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=evidence,
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.NEARLY_READY


# ---------------------------------------------------------------------------
# Stale evidence
# ---------------------------------------------------------------------------


def test_stale_evidence_drops_to_nearly_ready() -> None:
    """Pin: evidence > 365d (Type II horizon) → stale."""

    stale = (
        EvidenceArtifact(
            kind=EvidenceKind.ACCESS_LOGS,
            present=True,
            last_updated=_NOW - timedelta(days=400),
        ),
    ) + tuple(a for a in _full_security_evidence() if a.kind is not EvidenceKind.ACCESS_LOGS)
    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=stale,
        now=_NOW,
    )
    assert report.per_service[0].controls_stale == 1
    assert report.overall_level is ReadinessLevel.NEARLY_READY


def test_artifact_at_exactly_365_days_is_stale() -> None:
    """Pin: at exactly the horizon, the artifact is stale."""

    stale = (
        EvidenceArtifact(
            kind=EvidenceKind.ACCESS_LOGS,
            present=True,
            last_updated=_NOW - timedelta(days=365),
        ),
    )
    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=stale,
        now=_NOW,
    )
    # stale flag set on this control
    stale_assessments = [a for a in report.per_service[0].assessments if a.is_stale]
    assert len(stale_assessments) >= 1


def test_type_i_short_horizon_catches_recent_artifact() -> None:
    """Type I 30-day horizon catches a 60-day-old artifact."""

    type_i = SOC2ReadinessPolicy(audit_type=AuditType.TYPE_I, evidence_horizon_days=30)
    artifact = (
        EvidenceArtifact(
            kind=EvidenceKind.ACCESS_LOGS,
            present=True,
            last_updated=_NOW - timedelta(days=60),
        ),
    )
    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=artifact,
        now=_NOW,
        policy=type_i,
    )
    assert report.per_service[0].controls_stale == 1


def test_no_last_updated_is_stale() -> None:
    """Pin: present but no timestamp → stale."""

    no_ts = (
        EvidenceArtifact(
            kind=EvidenceKind.ACCESS_LOGS,
            present=True,
            last_updated=None,
        ),
    )
    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=no_ts,
        now=_NOW,
    )
    stale = [a for a in report.per_service[0].assessments if a.is_stale]
    assert len(stale) >= 1


# ---------------------------------------------------------------------------
# Empty service requirement set
# ---------------------------------------------------------------------------


def test_processing_integrity_returns_not_ready_with_note() -> None:
    """Pin: a service with no loaded controls → NOT_READY with note."""

    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.PROCESSING_INTEGRITY,),
        evidence=(),
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.NOT_READY
    pi_report = report.per_service[0]
    assert any("control set" in n for n in pi_report.notes)


# ---------------------------------------------------------------------------
# Multi-service overall verdict
# ---------------------------------------------------------------------------


def test_one_ready_one_not_ready_overall_is_not_ready() -> None:
    """Pin: overall = strictest (worst) per-service level."""

    report = evaluate_soc2_readiness(
        trust_services=(
            TrustServiceCategory.SECURITY,
            TrustServiceCategory.AVAILABILITY,
        ),
        evidence=_full_security_evidence(),  # availability missing
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.NOT_READY


def test_one_ready_one_gaps_overall_is_gaps() -> None:
    """Pin: GAPS in any service drops overall to GAPS."""

    # Drop one blocking from availability
    avail_partial = tuple(
        a for a in _full_availability_evidence() if a.kind is not EvidenceKind.BACKUP_RECORDS
    )
    report = evaluate_soc2_readiness(
        trust_services=(
            TrustServiceCategory.SECURITY,
            TrustServiceCategory.AVAILABILITY,
        ),
        evidence=_full_security_evidence() + avail_partial,
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.GAPS


def test_overall_met_pct_aggregates_across_services() -> None:
    report = evaluate_soc2_readiness(
        trust_services=(
            TrustServiceCategory.SECURITY,
            TrustServiceCategory.AVAILABILITY,
        ),
        evidence=_full_security_evidence() + _full_availability_evidence(),
        now=_NOW,
    )
    assert report.all_services_met_pct == 100.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_evaluate_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate_soc2_readiness(
            trust_services=(TrustServiceCategory.SECURITY,),
            evidence=(),
            now=datetime(2026, 5, 1),
        )


def test_evaluate_rejects_empty_trust_services() -> None:
    with pytest.raises(ValueError, match="trust_services must be non-empty"):
        evaluate_soc2_readiness(
            trust_services=(),
            evidence=(),
            now=_NOW,
        )


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_control_is_frozen() -> None:
    c = SOC2Control(
        trust_service=TrustServiceCategory.SECURITY,
        control_id="X",
        category=ControlCategory.ACCESS_CONTROL,
        severity=ControlSeverity.BLOCKING,
        description="x",
        expected_evidence=EvidenceKind.ACCESS_LOGS,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.control_id = "Y"  # type: ignore[misc]


def test_artifact_is_frozen() -> None:
    a = _artifact()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.present = False  # type: ignore[misc]


def test_report_is_frozen() -> None:
    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=(),
        now=_NOW,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.overall_level = ReadinessLevel.READY  # type: ignore[misc]


def test_policy_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_POLICY.evidence_horizon_days = 30  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned
# ---------------------------------------------------------------------------


def test_trust_service_string_values() -> None:
    assert TrustServiceCategory.SECURITY.value == "security"
    assert TrustServiceCategory.AVAILABILITY.value == "availability"
    assert TrustServiceCategory.PROCESSING_INTEGRITY.value == "processing_integrity"
    assert TrustServiceCategory.CONFIDENTIALITY.value == "confidentiality"
    assert TrustServiceCategory.PRIVACY.value == "privacy"


def test_audit_type_string_values() -> None:
    assert AuditType.TYPE_I.value == "type_i"
    assert AuditType.TYPE_II.value == "type_ii"


def test_severity_string_values() -> None:
    assert ControlSeverity.BLOCKING.value == "blocking"
    assert ControlSeverity.WARNING.value == "warning"


def test_evidence_kind_string_values() -> None:
    assert EvidenceKind.ACCESS_LOGS.value == "access_logs"
    assert EvidenceKind.MFA_ENABLED.value == "mfa_enabled"
    assert EvidenceKind.BACKUP_RECORDS.value == "backup_records"
    assert EvidenceKind.INCIDENT_REPORTS.value == "incident_reports"
    assert EvidenceKind.ENCRYPTION_AT_REST.value == "encryption_at_rest"


def test_level_string_values() -> None:
    assert ReadinessLevel.READY.value == "ready"
    assert ReadinessLevel.NEARLY_READY.value == "nearly_ready"
    assert ReadinessLevel.GAPS.value == "gaps"
    assert ReadinessLevel.NOT_READY.value == "not_ready"


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


def test_render_ready_report() -> None:
    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=_full_security_evidence(),
        now=_NOW,
    )
    text = render_readiness_report(report)
    assert "✅" in text
    assert "READY" in text
    assert "type_ii" in text


def test_render_not_ready_report() -> None:
    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=(),
        now=_NOW,
    )
    text = render_readiness_report(report)
    assert "❌" in text
    assert "NOT_READY" in text


def test_render_includes_per_service_breakdown() -> None:
    report = evaluate_soc2_readiness(
        trust_services=(
            TrustServiceCategory.SECURITY,
            TrustServiceCategory.AVAILABILITY,
        ),
        evidence=_full_security_evidence() + _full_availability_evidence(),
        now=_NOW,
    )
    text = render_readiness_report(report)
    assert "security" in text
    assert "availability" in text


def test_render_does_not_include_user_pii() -> None:
    """Pin: render never includes user IDs / IP addresses / audit-trail contents."""

    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=_full_security_evidence(),
        now=_NOW,
    )
    text = render_readiness_report(report)
    assert "user_id" not in text
    assert "ip_address" not in text
    assert "10.0.0" not in text
    assert "192.168" not in text


# ---------------------------------------------------------------------------
# End-to-end realistic scenarios
# ---------------------------------------------------------------------------


def test_typical_pre_audit_security_only() -> None:
    """An operator preparing for SOC 2 Type II Security-only audit:
    has access logs + MFA + employee onboarding + vuln scans + incident
    reports + PR reviews + risk register but missing pen test + security
    training (warnings) and vendor SOC 2 (warning).
    """

    evidence = (
        _artifact(kind=EvidenceKind.ACCESS_LOGS),
        _artifact(kind=EvidenceKind.MFA_ENABLED),
        _artifact(kind=EvidenceKind.EMPLOYEE_ONBOARDING),
        _artifact(kind=EvidenceKind.VULNERABILITY_SCANS),
        _artifact(kind=EvidenceKind.INCIDENT_REPORTS),
        _artifact(kind=EvidenceKind.PR_REVIEW_LOGS),
        _artifact(kind=EvidenceKind.RISK_REGISTER),
    )
    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=evidence,
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.NEARLY_READY


def test_full_three_service_audit_readiness() -> None:
    """Operator wants Security + Availability + Confidentiality (the
    common institutional SOC 2 set) — has everything → READY."""

    report = evaluate_soc2_readiness(
        trust_services=(
            TrustServiceCategory.SECURITY,
            TrustServiceCategory.AVAILABILITY,
            TrustServiceCategory.CONFIDENTIALITY,
        ),
        evidence=(
            _full_security_evidence()
            + _full_availability_evidence()
            + _full_confidentiality_evidence()
        ),
        now=_NOW,
    )
    assert report.overall_level is ReadinessLevel.READY
    assert len(report.per_service) == 3


def test_partial_service_evidence_realistic_flow() -> None:
    """Security + Confidentiality fully evidenced but Availability has 1 blocking gap."""

    avail_partial = tuple(
        a for a in _full_availability_evidence() if a.kind is not EvidenceKind.DR_DRILL_REPORTS
    )
    report = evaluate_soc2_readiness(
        trust_services=(
            TrustServiceCategory.SECURITY,
            TrustServiceCategory.AVAILABILITY,
            TrustServiceCategory.CONFIDENTIALITY,
        ),
        evidence=(_full_security_evidence() + avail_partial + _full_confidentiality_evidence()),
        now=_NOW,
    )
    # Availability has 1 blocking gap → GAPS for that service → overall GAPS
    assert report.overall_level is ReadinessLevel.GAPS


def test_assessment_count_matches_controls() -> None:
    report = evaluate_soc2_readiness(
        trust_services=(TrustServiceCategory.SECURITY,),
        evidence=_full_security_evidence(),
        now=_NOW,
    )
    sec_report = report.per_service[0]
    assert len(sec_report.assessments) == sec_report.controls_total
