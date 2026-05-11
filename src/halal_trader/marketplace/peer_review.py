"""Strategy peer review — Round-5 Wave 21.C.

After a strategy clears the publishing gate, peers can review it on
two independent dimensions:

1. **Halal compliance** — does the strategy structurally avoid riba /
   gharar / maysir? Reviewers vote APPROVE / CONCERN / REJECT and
   pin a written reason.
2. **Statistical validity** — is the backtest + paper-trading record
   coherent? (Sample size, overfitting risk, look-ahead bias, etc.)

A `ReviewPanel` collects N reviews across both dimensions; consensus
rules decide whether the strategy clears peer review.

Pinned semantics:

- **Closed-set ReviewDimension** — HALAL_COMPLIANCE / STATISTICAL_VALIDITY.
- **Closed-set ReviewVerdict** — APPROVE / CONCERN / REJECT.
- **Closed-set ReviewerTier** — JUNIOR / SENIOR / SCHOLAR. Each tier
  has a vote weight; SCHOLAR carries 1.5×, SENIOR 1.0×, JUNIOR 0.5×.
- **Conflict-of-interest pin** — the strategy's author cannot review
  their own strategy.
- **Consensus rule** — for each dimension: any REJECT (weighted >= 1
  unit) → panel rejects; any CONCERN without REJECTs → panel
  CONCERN; else APPROVE. Both dimensions must APPROVE for an
  `OVERALL_APPROVE`.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — reviewer IDs masked; review-text
  truncated.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ReviewDimension(str, Enum):
    """Closed-set dimension ladder."""

    HALAL_COMPLIANCE = "halal_compliance"
    STATISTICAL_VALIDITY = "statistical_validity"


class ReviewVerdict(str, Enum):
    """Closed-set verdict ladder."""

    APPROVE = "approve"
    CONCERN = "concern"
    REJECT = "reject"


class ReviewerTier(str, Enum):
    """Closed-set reviewer tier ladder."""

    JUNIOR = "junior"
    SENIOR = "senior"
    SCHOLAR = "scholar"


_TIER_WEIGHT: dict[ReviewerTier, float] = {
    ReviewerTier.JUNIOR: 0.5,
    ReviewerTier.SENIOR: 1.0,
    ReviewerTier.SCHOLAR: 1.5,
}


@dataclass(frozen=True)
class Review:
    """One reviewer's vote on one dimension."""

    review_id: str
    strategy_id: str
    reviewer_id: str
    reviewer_tier: ReviewerTier
    dimension: ReviewDimension
    verdict: ReviewVerdict
    reason: str
    submitted_at: datetime

    def __post_init__(self) -> None:
        if not self.review_id or not self.review_id.strip():
            raise ValueError("review_id must be non-empty")
        if not self.strategy_id or not self.strategy_id.strip():
            raise ValueError("strategy_id must be non-empty")
        if not self.reviewer_id or not self.reviewer_id.strip():
            raise ValueError("reviewer_id must be non-empty")
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if len(self.reason) > 2000:
            raise ValueError("reason must be ≤ 2000 chars")

    def vote_weight(self) -> float:
        return _TIER_WEIGHT[self.reviewer_tier]


@dataclass(frozen=True)
class DimensionVerdict:
    """Aggregate verdict for one dimension."""

    dimension: ReviewDimension
    verdict: ReviewVerdict
    n_reviews: int
    approve_weight: float
    concern_weight: float
    reject_weight: float


@dataclass(frozen=True)
class PanelResult:
    """Output of `evaluate_panel`."""

    strategy_id: str
    per_dimension: tuple[DimensionVerdict, ...]
    overall_verdict: ReviewVerdict
    review_count: int


class ConflictOfInterestError(ValueError):
    """The author submitted a review on their own strategy."""


def _aggregate_dimension(
    dimension: ReviewDimension,
    reviews: Sequence[Review],
    *,
    reject_weight_threshold: float = 1.0,
) -> DimensionVerdict:
    """Aggregate reviews for one dimension into a single verdict."""
    approve_w = sum(r.vote_weight() for r in reviews if r.verdict is ReviewVerdict.APPROVE)
    concern_w = sum(r.vote_weight() for r in reviews if r.verdict is ReviewVerdict.CONCERN)
    reject_w = sum(r.vote_weight() for r in reviews if r.verdict is ReviewVerdict.REJECT)
    if reject_w >= reject_weight_threshold - 1e-12:
        verdict = ReviewVerdict.REJECT
    elif concern_w > 0 or reject_w > 0:
        # Any below-threshold reject is still a signal — escalate to
        # CONCERN so operators see the dissent without auto-rejecting.
        verdict = ReviewVerdict.CONCERN
    else:
        verdict = ReviewVerdict.APPROVE
    return DimensionVerdict(
        dimension=dimension,
        verdict=verdict,
        n_reviews=len(reviews),
        approve_weight=approve_w,
        concern_weight=concern_w,
        reject_weight=reject_w,
    )


def evaluate_panel(
    strategy_id: str,
    author_id: str,
    reviews: Iterable[Review],
    *,
    min_reviews_per_dimension: int = 2,
    reject_weight_threshold: float = 1.0,
) -> PanelResult:
    """Aggregate a panel of reviews.

    Pinned:
    - The strategy's author cannot review their own strategy.
    - Each dimension requires `min_reviews_per_dimension` reviews;
      below that, the dimension's verdict is CONCERN.
    - Overall verdict is the worst across dimensions.
    """
    if min_reviews_per_dimension <= 0:
        raise ValueError("min_reviews_per_dimension must be positive")
    if reject_weight_threshold <= 0:
        raise ValueError("reject_weight_threshold must be positive")
    review_list = list(reviews)
    for r in review_list:
        if r.strategy_id != strategy_id:
            raise ValueError(f"review {r.review_id} strategy_id mismatch")
        if r.reviewer_id == author_id:
            raise ConflictOfInterestError(f"reviewer {r.reviewer_id} cannot review own strategy")
    # Dedup: each reviewer can submit at most one review per dimension.
    seen: set[tuple[str, ReviewDimension]] = set()
    for r in review_list:
        key = (r.reviewer_id, r.dimension)
        if key in seen:
            raise ValueError(
                f"reviewer {r.reviewer_id} submitted multiple reviews "
                f"for dimension {r.dimension.value}"
            )
        seen.add(key)
    per_dim: list[DimensionVerdict] = []
    overall = ReviewVerdict.APPROVE
    for dim in ReviewDimension:
        dim_reviews = [r for r in review_list if r.dimension is dim]
        if len(dim_reviews) < min_reviews_per_dimension:
            dv = DimensionVerdict(
                dimension=dim,
                verdict=ReviewVerdict.CONCERN,
                n_reviews=len(dim_reviews),
                approve_weight=sum(
                    r.vote_weight() for r in dim_reviews if r.verdict is ReviewVerdict.APPROVE
                ),
                concern_weight=sum(
                    r.vote_weight() for r in dim_reviews if r.verdict is ReviewVerdict.CONCERN
                ),
                reject_weight=sum(
                    r.vote_weight() for r in dim_reviews if r.verdict is ReviewVerdict.REJECT
                ),
            )
        else:
            dv = _aggregate_dimension(
                dim,
                dim_reviews,
                reject_weight_threshold=reject_weight_threshold,
            )
        per_dim.append(dv)
        # Update overall.
        if dv.verdict is ReviewVerdict.REJECT:
            overall = ReviewVerdict.REJECT
        elif dv.verdict is ReviewVerdict.CONCERN and overall is not ReviewVerdict.REJECT:
            overall = ReviewVerdict.CONCERN
    return PanelResult(
        strategy_id=strategy_id,
        per_dimension=tuple(per_dim),
        overall_verdict=overall,
        review_count=len(review_list),
    )


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_VERDICT_EMOJI: dict[ReviewVerdict, str] = {
    ReviewVerdict.APPROVE: "✅",
    ReviewVerdict.CONCERN: "🟡",
    ReviewVerdict.REJECT: "❌",
}


def render_review(review: Review, *, reason_chars: int = 80) -> str:
    """Operator-readable review summary; reason truncated."""
    if len(review.reason) <= reason_chars:
        truncated = review.reason
    else:
        truncated = review.reason[:reason_chars] + "…"
    return (
        f"{_VERDICT_EMOJI[review.verdict]} [{review.review_id}] "
        f"{_mask(review.reviewer_id)} ({review.reviewer_tier.value}) "
        f"{review.dimension.value}: {truncated}"
    )


def render_panel(result: PanelResult) -> str:
    head = (
        f"{_VERDICT_EMOJI[result.overall_verdict]} Panel for "
        f"{result.strategy_id}: {result.overall_verdict.value.upper()} "
        f"({result.review_count} review(s))"
    )
    lines = [head]
    for dv in result.per_dimension:
        lines.append(
            f"  • {dv.dimension.value}: "
            f"{_VERDICT_EMOJI[dv.verdict]} {dv.verdict.value} "
            f"(approve={dv.approve_weight:.1f}, concern={dv.concern_weight:.1f}, "
            f"reject={dv.reject_weight:.1f}, n={dv.n_reviews})"
        )
    return "\n".join(lines)
