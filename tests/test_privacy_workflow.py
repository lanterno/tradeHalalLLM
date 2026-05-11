"""Tests for ops/privacy_workflow.py — Round-5 Wave 19.G."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.ops.privacy_workflow import (
    DataCategory,
    DataSubjectRequest,
    ErasureScope,
    Jurisdiction,
    RequestKind,
    RequestStatus,
    RetentionPolicy,
    erasure_scope,
    portability_export,
    render_erasure_scope,
    render_request,
    sla_days,
    transition,
)


def _request(
    request_id: str = "R1",
    subject_id: str = "alice",
    jurisdiction: Jurisdiction = Jurisdiction.GDPR_EU,
    kind: RequestKind = RequestKind.ERASURE,
    filed_at: date = date(2026, 5, 1),
    status: RequestStatus = RequestStatus.FILED,
    verified_at: date | None = None,
    completed_at: date | None = None,
    rejection_reason: str = "",
) -> DataSubjectRequest:
    return DataSubjectRequest(
        request_id=request_id,
        subject_id=subject_id,
        jurisdiction=jurisdiction,
        kind=kind,
        filed_at=filed_at,
        status=status,
        verified_at=verified_at,
        completed_at=completed_at,
        rejection_reason=rejection_reason,
    )


# --- sla_days helper --------------------------------------------


def test_sla_gdpr_erasure_30():
    assert sla_days(Jurisdiction.GDPR_EU, RequestKind.ERASURE) == 30


def test_sla_ccpa_opt_out_15():
    """Pin: CCPA opt-out is 15 days."""
    assert sla_days(Jurisdiction.CCPA_CA, RequestKind.OPT_OUT) == 15


def test_sla_dpdp_grievance_7():
    """Pin: DPDP grievance is 7 days."""
    assert sla_days(Jurisdiction.DPDP_IN, RequestKind.GRIEVANCE) == 7


def test_sla_unrecognised_returns_none():
    """CCPA does not have a RECTIFICATION right."""
    assert sla_days(Jurisdiction.CCPA_CA, RequestKind.RECTIFICATION) is None


# --- DataSubjectRequest validation ------------------------------


def test_request_valid():
    r = _request()
    assert r.sla_days() == 30
    assert r.deadline() == date(2026, 5, 31)


def test_request_unsupported_kind_rejected():
    """Pin: CCPA + RECTIFICATION → no SLA → reject."""
    with pytest.raises(ValueError):
        _request(jurisdiction=Jurisdiction.CCPA_CA, kind=RequestKind.RECTIFICATION)


def test_request_empty_id_rejected():
    with pytest.raises(ValueError):
        _request(request_id="")


def test_request_verified_before_filed_rejected():
    with pytest.raises(ValueError):
        _request(verified_at=date(2026, 4, 1))


def test_request_completed_before_filed_rejected():
    with pytest.raises(ValueError):
        _request(
            status=RequestStatus.COMPLETED,
            completed_at=date(2026, 4, 1),
        )


def test_request_rejected_requires_reason():
    with pytest.raises(ValueError):
        _request(
            status=RequestStatus.REJECTED,
            completed_at=date(2026, 5, 5),
            rejection_reason="",
        )


def test_request_completed_without_date_rejected():
    with pytest.raises(ValueError):
        _request(status=RequestStatus.COMPLETED, completed_at=None)


def test_request_immutable():
    r = _request()
    with pytest.raises(AttributeError):
        r.kind = RequestKind.ACCESS  # type: ignore[misc]


# --- is_overdue --------------------------------------------------


def test_is_overdue_false_before_deadline():
    r = _request()
    assert not r.is_overdue(date(2026, 5, 25))


def test_is_overdue_true_after_deadline():
    r = _request()
    assert r.is_overdue(date(2026, 6, 5))


def test_is_overdue_false_when_completed():
    r = _request(
        status=RequestStatus.COMPLETED,
        completed_at=date(2026, 5, 10),
    )
    assert not r.is_overdue(date(2026, 6, 5))


def test_is_overdue_false_when_rejected():
    r = _request(
        status=RequestStatus.REJECTED,
        completed_at=date(2026, 5, 10),
        rejection_reason="invalid id",
    )
    assert not r.is_overdue(date(2026, 6, 5))


# --- transition FSM ----------------------------------------------


def test_transition_filed_to_verifying():
    r = _request()
    r2 = transition(r, new_status=RequestStatus.VERIFYING, at=date(2026, 5, 5))
    assert r2.status is RequestStatus.VERIFYING
    assert r2.verified_at == date(2026, 5, 5)


def test_transition_verifying_to_in_progress():
    r = transition(_request(), new_status=RequestStatus.VERIFYING, at=date(2026, 5, 5))
    r2 = transition(r, new_status=RequestStatus.IN_PROGRESS, at=date(2026, 5, 6))
    assert r2.status is RequestStatus.IN_PROGRESS


def test_transition_in_progress_to_completed():
    r = transition(_request(), new_status=RequestStatus.VERIFYING, at=date(2026, 5, 5))
    r = transition(r, new_status=RequestStatus.IN_PROGRESS, at=date(2026, 5, 6))
    r2 = transition(r, new_status=RequestStatus.COMPLETED, at=date(2026, 5, 20))
    assert r2.status is RequestStatus.COMPLETED
    assert r2.completed_at == date(2026, 5, 20)


def test_transition_rejected_at_any_point():
    """Each early state can transition to REJECTED."""
    r = _request()
    r2 = transition(
        r,
        new_status=RequestStatus.REJECTED,
        at=date(2026, 5, 5),
        rejection_reason="invalid identity",
    )
    assert r2.status is RequestStatus.REJECTED
    assert r2.rejection_reason == "invalid identity"


def test_transition_reject_requires_reason():
    r = _request()
    with pytest.raises(ValueError):
        transition(r, new_status=RequestStatus.REJECTED, at=date(2026, 5, 5))


def test_transition_skip_state_rejected():
    """FILED → COMPLETED directly is illegal."""
    r = _request()
    with pytest.raises(ValueError):
        transition(r, new_status=RequestStatus.COMPLETED, at=date(2026, 5, 5))


def test_transition_completed_terminal():
    r = transition(_request(), new_status=RequestStatus.VERIFYING, at=date(2026, 5, 5))
    r = transition(r, new_status=RequestStatus.IN_PROGRESS, at=date(2026, 5, 6))
    r = transition(r, new_status=RequestStatus.COMPLETED, at=date(2026, 5, 10))
    with pytest.raises(ValueError):
        transition(r, new_status=RequestStatus.IN_PROGRESS, at=date(2026, 5, 11))


# --- erasure_scope ----------------------------------------------


def test_erasure_default_audit_trail_retained():
    """Pin: AUDIT_TRAIL is never deletable by default."""
    r = _request(kind=RequestKind.ERASURE)
    scope = erasure_scope(
        r,
        held_categories=[
            DataCategory.ACCOUNT_PROFILE,
            DataCategory.AUDIT_TRAIL,
        ],
    )
    assert DataCategory.AUDIT_TRAIL in scope.retained_categories
    assert DataCategory.ACCOUNT_PROFILE in scope.deletable_categories


def test_erasure_kyc_retained():
    """Pin: KYC docs retained for AML compliance."""
    r = _request(kind=RequestKind.ERASURE)
    scope = erasure_scope(r, held_categories=[DataCategory.KYC_DOCUMENTS])
    assert DataCategory.KYC_DOCUMENTS in scope.retained_categories


def test_erasure_payment_retained():
    r = _request(kind=RequestKind.ERASURE)
    scope = erasure_scope(r, held_categories=[DataCategory.PAYMENT_RECORDS])
    assert DataCategory.PAYMENT_RECORDS in scope.retained_categories


def test_erasure_trade_history_deletable():
    r = _request(kind=RequestKind.ERASURE)
    scope = erasure_scope(r, held_categories=[DataCategory.TRADE_HISTORY])
    assert DataCategory.TRADE_HISTORY in scope.deletable_categories


def test_erasure_with_policy_override():
    pol = RetentionPolicy(
        overrides={
            (DataCategory.ACCOUNT_PROFILE, Jurisdiction.GDPR_EU): False,
        }
    )
    r = _request(kind=RequestKind.ERASURE)
    scope = erasure_scope(r, held_categories=[DataCategory.ACCOUNT_PROFILE], policy=pol)
    assert DataCategory.ACCOUNT_PROFILE in scope.retained_categories


def test_erasure_only_valid_for_erasure_kind():
    r = _request(kind=RequestKind.ACCESS)
    with pytest.raises(ValueError):
        erasure_scope(r, held_categories=[])


# --- portability_export --------------------------------------


def test_portability_excludes_audit_trail():
    """Pin: AUDIT_TRAIL never exported."""
    r = _request(kind=RequestKind.PORTABILITY)
    export = portability_export(
        r,
        held_categories=[
            DataCategory.ACCOUNT_PROFILE,
            DataCategory.AUDIT_TRAIL,
        ],
    )
    assert DataCategory.AUDIT_TRAIL not in export.categories
    assert DataCategory.ACCOUNT_PROFILE in export.categories


def test_portability_access_also_works():
    r = _request(kind=RequestKind.ACCESS)
    export = portability_export(r, held_categories=[DataCategory.ACCOUNT_PROFILE])
    assert export.categories == (DataCategory.ACCOUNT_PROFILE,)


def test_portability_other_kinds_rejected():
    r = _request(kind=RequestKind.ERASURE)
    with pytest.raises(ValueError):
        portability_export(r, held_categories=[])


def test_portability_invalid_format_rejected():
    r = _request(kind=RequestKind.ACCESS)
    with pytest.raises(ValueError):
        portability_export(r, held_categories=[DataCategory.ACCOUNT_PROFILE], format="xml")


# --- Render ------------------------------------------------


def test_render_request_no_secret_leak():
    r = _request(subject_id="alice@example.com")
    out = render_request(r)
    assert "alice@example.com" not in out


def test_render_request_overdue_flag():
    r = _request()
    out = render_request(r, as_of=date(2026, 6, 5))
    assert "OVERDUE" in out


def test_render_request_rejected_shows_reason():
    r = _request(
        status=RequestStatus.REJECTED,
        completed_at=date(2026, 5, 5),
        rejection_reason="failed identity check",
    )
    out = render_request(r)
    assert "Rejected" in out
    assert "failed identity check" in out


def test_render_erasure_scope_format():
    scope = ErasureScope(
        deletable_categories=(DataCategory.ACCOUNT_PROFILE,),
        retained_categories=(DataCategory.AUDIT_TRAIL,),
    )
    out = render_erasure_scope(scope)
    assert "delete=" in out
    assert "retain=" in out
