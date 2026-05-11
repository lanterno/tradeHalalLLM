"""Tests for marketplace/peer_review.py — Round-5 Wave 21.C."""

from __future__ import annotations

from datetime import datetime

import pytest

from halal_trader.marketplace.peer_review import (
    ConflictOfInterestError,
    Review,
    ReviewDimension,
    ReviewerTier,
    ReviewVerdict,
    evaluate_panel,
    render_panel,
    render_review,
)


def _review(
    review_id: str = "R1",
    strategy_id: str = "S1",
    reviewer_id: str = "bob",
    tier: ReviewerTier = ReviewerTier.SENIOR,
    dimension: ReviewDimension = ReviewDimension.HALAL_COMPLIANCE,
    verdict: ReviewVerdict = ReviewVerdict.APPROVE,
    reason: str = "Looks compliant; no riba mechanisms detected.",
    submitted_at: datetime = datetime(2026, 5, 11, 10, 0),
) -> Review:
    return Review(
        review_id=review_id,
        strategy_id=strategy_id,
        reviewer_id=reviewer_id,
        reviewer_tier=tier,
        dimension=dimension,
        verdict=verdict,
        reason=reason,
        submitted_at=submitted_at,
    )


# --- Review validation -------------------------------------------


def test_review_valid():
    r = _review()
    assert r.vote_weight() == 1.0  # SENIOR


def test_review_empty_id_rejected():
    with pytest.raises(ValueError):
        _review(review_id="")


def test_review_empty_reason_rejected():
    with pytest.raises(ValueError):
        _review(reason=" ")


def test_review_long_reason_rejected():
    with pytest.raises(ValueError):
        _review(reason="x" * 2500)


def test_review_vote_weight_per_tier():
    assert _review(tier=ReviewerTier.JUNIOR).vote_weight() == 0.5
    assert _review(tier=ReviewerTier.SENIOR).vote_weight() == 1.0
    assert _review(tier=ReviewerTier.SCHOLAR).vote_weight() == 1.5


def test_review_immutable():
    r = _review()
    with pytest.raises(AttributeError):
        r.verdict = ReviewVerdict.REJECT  # type: ignore[misc]


# --- evaluate_panel — happy path --------------------------------


def _full_panel_approve() -> list[Review]:
    return [
        _review(
            review_id=f"R{i}",
            reviewer_id=f"reviewer{i}",
            dimension=dim,
            verdict=ReviewVerdict.APPROVE,
        )
        for i, dim in enumerate(
            [
                ReviewDimension.HALAL_COMPLIANCE,
                ReviewDimension.HALAL_COMPLIANCE,
                ReviewDimension.STATISTICAL_VALIDITY,
                ReviewDimension.STATISTICAL_VALIDITY,
            ]
        )
    ]


def test_evaluate_all_approve():
    reviews = _full_panel_approve()
    result = evaluate_panel("S1", "alice", reviews)
    assert result.overall_verdict is ReviewVerdict.APPROVE
    for dv in result.per_dimension:
        assert dv.verdict is ReviewVerdict.APPROVE


def test_evaluate_review_count():
    reviews = _full_panel_approve()
    result = evaluate_panel("S1", "alice", reviews)
    assert result.review_count == 4


# --- evaluate_panel — REJECT --------------------------------------


def test_one_reject_propagates_to_dimension():
    """Pin: a single weighted REJECT ≥ 1.0 → dimension REJECT."""
    reviews = [
        _review(
            review_id="R1",
            reviewer_id="bob",
            tier=ReviewerTier.SENIOR,
            dimension=ReviewDimension.HALAL_COMPLIANCE,
            verdict=ReviewVerdict.REJECT,
        ),
        _review(
            review_id="R2",
            reviewer_id="charlie",
            dimension=ReviewDimension.HALAL_COMPLIANCE,
        ),
        _review(
            review_id="R3",
            reviewer_id="dave",
            dimension=ReviewDimension.STATISTICAL_VALIDITY,
        ),
        _review(
            review_id="R4",
            reviewer_id="eve",
            dimension=ReviewDimension.STATISTICAL_VALIDITY,
        ),
    ]
    result = evaluate_panel("S1", "alice", reviews)
    by_dim = {dv.dimension: dv for dv in result.per_dimension}
    assert by_dim[ReviewDimension.HALAL_COMPLIANCE].verdict is ReviewVerdict.REJECT
    assert result.overall_verdict is ReviewVerdict.REJECT


def test_junior_reject_alone_not_enough():
    """Pin: JUNIOR weight = 0.5 alone < threshold 1.0 → not enough to reject."""
    reviews = [
        _review(
            review_id="R1",
            reviewer_id="bob",
            tier=ReviewerTier.JUNIOR,
            dimension=ReviewDimension.HALAL_COMPLIANCE,
            verdict=ReviewVerdict.REJECT,
        ),
        _review(
            review_id="R2",
            reviewer_id="charlie",
            dimension=ReviewDimension.HALAL_COMPLIANCE,
        ),
        _review(
            review_id="R3",
            reviewer_id="dave",
            dimension=ReviewDimension.STATISTICAL_VALIDITY,
        ),
        _review(
            review_id="R4",
            reviewer_id="eve",
            dimension=ReviewDimension.STATISTICAL_VALIDITY,
        ),
    ]
    result = evaluate_panel("S1", "alice", reviews)
    by_dim = {dv.dimension: dv for dv in result.per_dimension}
    # JUNIOR reject (weight 0.5) + SENIOR approve (1.0) → concern, not reject.
    assert by_dim[ReviewDimension.HALAL_COMPLIANCE].verdict is ReviewVerdict.CONCERN


def test_two_juniors_reject_meets_threshold():
    reviews = [
        _review(
            review_id="R1",
            reviewer_id="bob",
            tier=ReviewerTier.JUNIOR,
            dimension=ReviewDimension.HALAL_COMPLIANCE,
            verdict=ReviewVerdict.REJECT,
        ),
        _review(
            review_id="R2",
            reviewer_id="charlie",
            tier=ReviewerTier.JUNIOR,
            dimension=ReviewDimension.HALAL_COMPLIANCE,
            verdict=ReviewVerdict.REJECT,
        ),
        _review(
            review_id="R3",
            reviewer_id="dave",
            dimension=ReviewDimension.STATISTICAL_VALIDITY,
        ),
        _review(
            review_id="R4",
            reviewer_id="eve",
            dimension=ReviewDimension.STATISTICAL_VALIDITY,
        ),
    ]
    result = evaluate_panel("S1", "alice", reviews)
    by_dim = {dv.dimension: dv for dv in result.per_dimension}
    # 0.5 + 0.5 = 1.0 → exactly at threshold → REJECT.
    assert by_dim[ReviewDimension.HALAL_COMPLIANCE].verdict is ReviewVerdict.REJECT


def test_scholar_reject_alone_enough():
    """Scholar weight 1.5 alone > threshold 1.0."""
    reviews = [
        _review(
            review_id="R1",
            reviewer_id="bob",
            tier=ReviewerTier.SCHOLAR,
            dimension=ReviewDimension.HALAL_COMPLIANCE,
            verdict=ReviewVerdict.REJECT,
        ),
        _review(
            review_id="R2",
            reviewer_id="charlie",
            dimension=ReviewDimension.HALAL_COMPLIANCE,
        ),
        _review(
            review_id="R3",
            reviewer_id="dave",
            dimension=ReviewDimension.STATISTICAL_VALIDITY,
        ),
        _review(
            review_id="R4",
            reviewer_id="eve",
            dimension=ReviewDimension.STATISTICAL_VALIDITY,
        ),
    ]
    result = evaluate_panel("S1", "alice", reviews)
    by_dim = {dv.dimension: dv for dv in result.per_dimension}
    assert by_dim[ReviewDimension.HALAL_COMPLIANCE].verdict is ReviewVerdict.REJECT


# --- evaluate_panel — CONCERN ----------------------------------


def test_concern_without_reject_yields_concern():
    reviews = [
        _review(
            review_id="R1",
            reviewer_id="bob",
            dimension=ReviewDimension.HALAL_COMPLIANCE,
            verdict=ReviewVerdict.CONCERN,
        ),
        _review(
            review_id="R2",
            reviewer_id="charlie",
            dimension=ReviewDimension.HALAL_COMPLIANCE,
        ),
        _review(
            review_id="R3",
            reviewer_id="dave",
            dimension=ReviewDimension.STATISTICAL_VALIDITY,
        ),
        _review(
            review_id="R4",
            reviewer_id="eve",
            dimension=ReviewDimension.STATISTICAL_VALIDITY,
        ),
    ]
    result = evaluate_panel("S1", "alice", reviews)
    assert result.overall_verdict is ReviewVerdict.CONCERN


def test_too_few_reviews_yields_concern():
    """Pin: below min reviewer count, dimension is CONCERN."""
    reviews = [
        _review(
            review_id="R1",
            reviewer_id="bob",
            dimension=ReviewDimension.HALAL_COMPLIANCE,
        ),
        # No STATISTICAL_VALIDITY reviews.
    ]
    result = evaluate_panel("S1", "alice", reviews)
    by_dim = {dv.dimension: dv for dv in result.per_dimension}
    assert by_dim[ReviewDimension.HALAL_COMPLIANCE].verdict is ReviewVerdict.CONCERN
    assert by_dim[ReviewDimension.STATISTICAL_VALIDITY].verdict is ReviewVerdict.CONCERN


# --- Conflict of interest -------------------------------------


def test_author_review_rejected():
    reviews = [
        _review(reviewer_id="alice"),  # author of S1
    ]
    with pytest.raises(ConflictOfInterestError):
        evaluate_panel("S1", "alice", reviews)


# --- Strategy mismatch ----------------------------------------


def test_strategy_id_mismatch_rejected():
    reviews = [_review(strategy_id="S2")]
    with pytest.raises(ValueError):
        evaluate_panel("S1", "alice", reviews)


# --- Duplicate review per dimension ---------------------------


def test_duplicate_same_dimension_rejected():
    reviews = [
        _review(
            review_id="R1",
            reviewer_id="bob",
            dimension=ReviewDimension.HALAL_COMPLIANCE,
        ),
        _review(
            review_id="R2",
            reviewer_id="bob",
            dimension=ReviewDimension.HALAL_COMPLIANCE,
            verdict=ReviewVerdict.REJECT,
        ),
    ]
    with pytest.raises(ValueError):
        evaluate_panel("S1", "alice", reviews)


def test_same_reviewer_different_dimensions_allowed():
    reviews = [
        _review(
            review_id="R1",
            reviewer_id="bob",
            dimension=ReviewDimension.HALAL_COMPLIANCE,
        ),
        _review(
            review_id="R2",
            reviewer_id="bob",
            dimension=ReviewDimension.STATISTICAL_VALIDITY,
        ),
        _review(
            review_id="R3",
            reviewer_id="charlie",
            dimension=ReviewDimension.HALAL_COMPLIANCE,
        ),
        _review(
            review_id="R4",
            reviewer_id="dave",
            dimension=ReviewDimension.STATISTICAL_VALIDITY,
        ),
    ]
    result = evaluate_panel("S1", "alice", reviews)
    assert result.review_count == 4


# --- Policy params -------------------------------------------


def test_invalid_min_reviews_rejected():
    with pytest.raises(ValueError):
        evaluate_panel("S1", "alice", [], min_reviews_per_dimension=0)


def test_invalid_reject_threshold_rejected():
    with pytest.raises(ValueError):
        evaluate_panel("S1", "alice", [], reject_weight_threshold=0)


# --- Render --------------------------------------------------


def test_render_review_truncates_long_reason():
    r = _review(reason="x" * 200)
    out = render_review(r, reason_chars=50)
    assert "…" in out


def test_render_review_no_secret_leak():
    r = _review(reviewer_id="alice@example.com")
    out = render_review(r)
    assert "alice@example.com" not in out


def test_render_panel_overall_emoji():
    reviews = _full_panel_approve()
    result = evaluate_panel("S1", "alice", reviews)
    out = render_panel(result)
    assert "✅" in out
    assert "APPROVE" in out


def test_render_panel_per_dimension_visible():
    reviews = _full_panel_approve()
    result = evaluate_panel("S1", "alice", reviews)
    out = render_panel(result)
    assert "halal_compliance" in out
    assert "statistical_validity" in out
