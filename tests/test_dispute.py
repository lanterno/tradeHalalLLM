"""Tests for marketplace/dispute.py — Round-5 Wave 21.H."""

from __future__ import annotations

from datetime import datetime

import pytest

from halal_trader.marketplace.dispute import (
    Dispute,
    DisputeReason,
    DisputeStatus,
    ResolutionOutcome,
    compute_refund,
    is_terminal,
    render_dispute,
    transition,
)


def _dispute(
    dispute_id: str = "D1",
    subscription_id: str = "SUB1",
    subscriber_id: str = "alice",
    author_id: str = "bob",
    fee: float = 100.0,
    service_days_provided: int = 15,
    total_service_days: int = 30,
    reason: DisputeReason = DisputeReason.MISREPRESENTATION,
    rationale: str = "The strategy claimed Sharpe 2.0 but live realised 0.3.",
    filed_at: datetime = datetime(2026, 5, 11, 10, 0),
    status: DisputeStatus = DisputeStatus.FILED,
    resolution: ResolutionOutcome | None = None,
    refund: float = 0.0,
    resolved_at: datetime | None = None,
) -> Dispute:
    return Dispute(
        dispute_id=dispute_id,
        subscription_id=subscription_id,
        subscriber_id=subscriber_id,
        author_id=author_id,
        fee_paid_usd=fee,
        service_days_provided=service_days_provided,
        total_service_days=total_service_days,
        reason=reason,
        rationale=rationale,
        filed_at=filed_at,
        status=status,
        resolution=resolution,
        refund_amount_usd=refund,
        resolved_at=resolved_at,
    )


# --- Dispute validation -------------------------------------------


def test_dispute_valid():
    d = _dispute()
    assert d.fee_paid_usd == 100.0


def test_dispute_empty_id_rejected():
    with pytest.raises(ValueError):
        _dispute(dispute_id="")


def test_dispute_self_dealing_rejected():
    with pytest.raises(ValueError):
        _dispute(subscriber_id="alice", author_id="alice")


def test_dispute_negative_fee_rejected():
    with pytest.raises(ValueError):
        _dispute(fee=-1.0)


def test_dispute_service_days_above_total_rejected():
    with pytest.raises(ValueError):
        _dispute(service_days_provided=40, total_service_days=30)


def test_dispute_zero_total_days_rejected():
    with pytest.raises(ValueError):
        _dispute(total_service_days=0)


def test_dispute_empty_rationale_rejected():
    with pytest.raises(ValueError):
        _dispute(rationale=" ")


def test_dispute_long_rationale_rejected():
    with pytest.raises(ValueError):
        _dispute(rationale="x" * 2500)


def test_dispute_refund_above_fee_rejected():
    with pytest.raises(ValueError):
        _dispute(refund=200.0)


def test_dispute_resolved_without_resolution_rejected():
    with pytest.raises(ValueError):
        _dispute(
            status=DisputeStatus.RESOLVED,
            resolution=None,
            resolved_at=datetime(2026, 5, 12),
        )


def test_dispute_resolution_set_on_non_resolved_rejected():
    with pytest.raises(ValueError):
        _dispute(
            status=DisputeStatus.UNDER_REVIEW,
            resolution=ResolutionOutcome.FULL_REFUND,
        )


def test_dispute_resolved_without_resolved_at_rejected():
    with pytest.raises(ValueError):
        _dispute(
            status=DisputeStatus.RESOLVED,
            resolution=ResolutionOutcome.FULL_REFUND,
            resolved_at=None,
        )


def test_dispute_resolved_at_before_filed_rejected():
    with pytest.raises(ValueError):
        _dispute(
            filed_at=datetime(2026, 5, 11),
            resolved_at=datetime(2026, 5, 10),
            status=DisputeStatus.RESOLVED,
            resolution=ResolutionOutcome.FULL_REFUND,
        )


def test_dispute_immutable():
    d = _dispute()
    with pytest.raises(AttributeError):
        d.fee_paid_usd = 0  # type: ignore[misc]


# --- transition — legal moves ----------------------------------


def test_transition_filed_to_under_review():
    d = _dispute()
    d2 = transition(d, new_status=DisputeStatus.UNDER_REVIEW, at=datetime(2026, 5, 12, 9, 0))
    assert d2.status is DisputeStatus.UNDER_REVIEW


def test_transition_filed_to_withdrawn():
    d = _dispute()
    d2 = transition(d, new_status=DisputeStatus.WITHDRAWN, at=datetime(2026, 5, 12))
    assert d2.status is DisputeStatus.WITHDRAWN


def test_transition_under_review_to_resolved_full_refund():
    d = transition(_dispute(), new_status=DisputeStatus.UNDER_REVIEW, at=datetime(2026, 5, 12))
    d2 = transition(
        d,
        new_status=DisputeStatus.RESOLVED,
        at=datetime(2026, 5, 13),
        resolution=ResolutionOutcome.FULL_REFUND,
    )
    assert d2.status is DisputeStatus.RESOLVED
    assert d2.resolution is ResolutionOutcome.FULL_REFUND
    assert d2.refund_amount_usd == 100.0


def test_transition_resolved_requires_resolution():
    d = transition(_dispute(), new_status=DisputeStatus.UNDER_REVIEW, at=datetime(2026, 5, 12))
    with pytest.raises(ValueError):
        transition(d, new_status=DisputeStatus.RESOLVED, at=datetime(2026, 5, 13))


def test_transition_resolved_is_terminal():
    d = transition(_dispute(), new_status=DisputeStatus.UNDER_REVIEW, at=datetime(2026, 5, 12))
    d = transition(
        d,
        new_status=DisputeStatus.RESOLVED,
        at=datetime(2026, 5, 13),
        resolution=ResolutionOutcome.REJECT,
    )
    with pytest.raises(ValueError):
        transition(d, new_status=DisputeStatus.FILED, at=datetime(2026, 5, 14))


def test_transition_withdrawn_is_terminal():
    d = transition(_dispute(), new_status=DisputeStatus.WITHDRAWN, at=datetime(2026, 5, 12))
    with pytest.raises(ValueError):
        transition(d, new_status=DisputeStatus.FILED, at=datetime(2026, 5, 13))


def test_transition_under_review_to_withdrawn():
    d = transition(_dispute(), new_status=DisputeStatus.UNDER_REVIEW, at=datetime(2026, 5, 12))
    d2 = transition(d, new_status=DisputeStatus.WITHDRAWN, at=datetime(2026, 5, 13))
    assert d2.status is DisputeStatus.WITHDRAWN


def test_transition_skip_under_review_rejected():
    """FILED → RESOLVED directly is illegal."""
    d = _dispute()
    with pytest.raises(ValueError):
        transition(
            d,
            new_status=DisputeStatus.RESOLVED,
            at=datetime(2026, 5, 12),
            resolution=ResolutionOutcome.REJECT,
        )


# --- compute_refund ------------------------------------------


def test_refund_full():
    d = _dispute(fee=100.0)
    assert compute_refund(d, outcome=ResolutionOutcome.FULL_REFUND) == 100.0


def test_refund_partial_pro_rata():
    """Pin: PARTIAL = fee × (1 - days_provided/total_days)."""
    d = _dispute(fee=100.0, service_days_provided=10, total_service_days=30)
    # Refund = 100 × (1 - 10/30) = 100 × 2/3 ≈ 66.67.
    assert compute_refund(d, outcome=ResolutionOutcome.PARTIAL_REFUND) == pytest.approx(
        66.666_666, abs=1e-3
    )


def test_refund_reject_zero():
    d = _dispute()
    assert compute_refund(d, outcome=ResolutionOutcome.REJECT) == 0.0


def test_refund_warn_author_zero():
    d = _dispute()
    assert compute_refund(d, outcome=ResolutionOutcome.WARN_AUTHOR) == 0.0


def test_refund_override_respected():
    d = _dispute(fee=100.0)
    assert (
        compute_refund(d, outcome=ResolutionOutcome.FULL_REFUND, override_refund_usd=42.0) == 42.0
    )


def test_refund_override_negative_rejected():
    d = _dispute()
    with pytest.raises(ValueError):
        compute_refund(d, outcome=ResolutionOutcome.FULL_REFUND, override_refund_usd=-1.0)


def test_refund_override_above_fee_rejected():
    d = _dispute(fee=100.0)
    with pytest.raises(ValueError):
        compute_refund(d, outcome=ResolutionOutcome.FULL_REFUND, override_refund_usd=200.0)


def test_refund_partial_at_full_service_zero():
    """If service was fully delivered, partial refund = 0."""
    d = _dispute(service_days_provided=30, total_service_days=30, fee=100.0)
    assert compute_refund(d, outcome=ResolutionOutcome.PARTIAL_REFUND) == 0.0


def test_refund_partial_at_zero_service_full():
    """If no service was provided, partial refund = full fee."""
    d = _dispute(service_days_provided=0, total_service_days=30, fee=100.0)
    assert compute_refund(d, outcome=ResolutionOutcome.PARTIAL_REFUND) == 100.0


# --- is_terminal --------------------------------------------


def test_is_terminal_resolved_and_withdrawn():
    assert is_terminal(DisputeStatus.RESOLVED)
    assert is_terminal(DisputeStatus.WITHDRAWN)
    assert not is_terminal(DisputeStatus.FILED)
    assert not is_terminal(DisputeStatus.UNDER_REVIEW)


# --- Resolution integration tests ---------------------------


def test_resolve_with_partial_refund_pinned():
    d = transition(
        _dispute(fee=100.0, service_days_provided=10, total_service_days=30),
        new_status=DisputeStatus.UNDER_REVIEW,
        at=datetime(2026, 5, 12),
    )
    d2 = transition(
        d,
        new_status=DisputeStatus.RESOLVED,
        at=datetime(2026, 5, 13),
        resolution=ResolutionOutcome.PARTIAL_REFUND,
    )
    assert d2.refund_amount_usd == pytest.approx(66.666, abs=1e-3)


def test_resolve_with_reject_pays_zero():
    d = transition(
        _dispute(),
        new_status=DisputeStatus.UNDER_REVIEW,
        at=datetime(2026, 5, 12),
    )
    d2 = transition(
        d,
        new_status=DisputeStatus.RESOLVED,
        at=datetime(2026, 5, 13),
        resolution=ResolutionOutcome.REJECT,
    )
    assert d2.refund_amount_usd == 0.0


def test_resolve_with_operator_override():
    d = transition(
        _dispute(fee=100.0),
        new_status=DisputeStatus.UNDER_REVIEW,
        at=datetime(2026, 5, 12),
    )
    d2 = transition(
        d,
        new_status=DisputeStatus.RESOLVED,
        at=datetime(2026, 5, 13),
        resolution=ResolutionOutcome.FULL_REFUND,
        refund_amount_usd=50.0,
        operator_notes="goodwill discount applied",
    )
    assert d2.refund_amount_usd == 50.0
    assert "goodwill" in d2.operator_notes


# --- Render -----------------------------------------------


def test_render_status_emoji():
    d = _dispute()
    out = render_dispute(d)
    assert "📨" in out


def test_render_no_secret_leak():
    d = _dispute(
        subscriber_id="alice@example.com",
        author_id="bob@example.com",
    )
    out = render_dispute(d)
    assert "alice@example.com" not in out
    assert "bob@example.com" not in out


def test_render_includes_resolution_when_set():
    d = transition(_dispute(), new_status=DisputeStatus.UNDER_REVIEW, at=datetime(2026, 5, 12))
    d = transition(
        d,
        new_status=DisputeStatus.RESOLVED,
        at=datetime(2026, 5, 13),
        resolution=ResolutionOutcome.PARTIAL_REFUND,
    )
    out = render_dispute(d)
    assert "Resolution" in out
    assert "💵" in out


def test_render_omits_resolution_when_unset():
    d = _dispute()
    out = render_dispute(d)
    assert "Resolution" not in out
