"""Tests for the private helpers in :mod:`core.llm.ensemble`.

`test_ensemble.py` covers `aggregate_plans` end-to-end. This file
pins the small primitives underneath:

* `_consensus_decision` — median quantity + confidence across
  agreeing variants (the operator picks "the middle of three" when
  the LLMs disagree by size).
* `_build_consensus_plan` — pydantic `model_copy` path + plain-object
  mutate fallback, with the `| ensemble consensus` risk-notes annotation.
* `_multiplier_from_agreement` — the ramp from 0.5 at quorum to 1.0
  at unanimity, with the optional `skip_at` skip threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

from halal_trader.core.llm.ensemble import (
    _build_consensus_plan,
    _consensus_decision,
    _multiplier_from_agreement,
)

# ── _consensus_decision ────────────────────────────────────


@dataclass
class _Decision:
    """Plain-object decision (no `model_copy`)."""

    symbol: str = "BTCUSDT"
    quantity: float = 0.0
    confidence: float = 0.0


class _PydanticLike:
    """A decision shape that mimics pydantic's `model_copy` API so
    `_consensus_decision`'s actual median path runs (the non-pydantic
    fallback returns the first variant unchanged)."""

    def __init__(self, *, quantity: float = 0.0, confidence: float = 0.0) -> None:
        self.quantity = quantity
        self.confidence = confidence

    def model_copy(self, *, update):
        new = _PydanticLike(quantity=self.quantity, confidence=self.confidence)
        for k, v in update.items():
            setattr(new, k, v)
        return new


def test_consensus_picks_median_quantity_of_three():
    """Three quantities → middle one wins. Pin so a refactor to mean
    doesn't silently shift sizing on disagreeing variants."""
    decisions = [
        _PydanticLike(quantity=0.1),
        _PydanticLike(quantity=0.5),
        _PydanticLike(quantity=1.0),
    ]
    out = _consensus_decision(decisions)
    assert out.quantity == 0.5


def test_consensus_picks_median_quantity_of_five():
    """Five quantities, sorted: [0.1, 0.2, 0.3, 0.4, 0.5] → index 2 = 0.3."""
    decisions = [_PydanticLike(quantity=q) for q in [0.5, 0.1, 0.3, 0.2, 0.4]]
    out = _consensus_decision(decisions)
    assert out.quantity == 0.3


def test_consensus_picks_median_quantity_of_two():
    """Even count: `len // 2` = 1 → upper-of-the-two (Python integer
    division). Pin so the implementation choice is explicit; a
    refactor to mean of the two would change downstream sizing."""
    decisions = [_PydanticLike(quantity=0.1), _PydanticLike(quantity=0.5)]
    out = _consensus_decision(decisions)
    assert out.quantity == 0.5  # upper, not (0.1+0.5)/2


def test_consensus_picks_median_confidence_independently():
    """Quantity + confidence are sorted independently — they're not
    paired. Pin so a refactor doesn't accidentally couple them."""
    decisions = [
        _PydanticLike(quantity=0.1, confidence=0.9),
        _PydanticLike(quantity=0.5, confidence=0.3),
        _PydanticLike(quantity=1.0, confidence=0.6),
    ]
    out = _consensus_decision(decisions)
    assert out.quantity == 0.5  # median qty
    assert out.confidence == 0.6  # median conf — different decision's value


def test_consensus_returns_first_variant_unchanged_for_non_pydantic():
    """For a plain-object decision (no `model_copy`), the helper
    *cannot* update the immutable shape — falls through to
    `return base` (the first variant unchanged). Pin this fallback
    so a refactor doesn't accidentally start mutating plain objects."""
    a = _Decision(quantity=0.1, confidence=0.5)
    b = _Decision(quantity=0.5, confidence=0.7)
    c = _Decision(quantity=1.0, confidence=0.9)
    out = _consensus_decision([a, b, c])
    # Without `model_copy`, falls through to `return base` — same
    # identity AND quantity/confidence unchanged from `a`.
    assert out is a
    assert out.quantity == 0.1  # NOT median 0.5
    assert out.confidence == 0.5  # NOT median 0.7


def test_consensus_works_with_pydantic_like_model_copy():
    """A pydantic model_copy creates a new object with updated fields
    (immutable shape) — pin the contract."""
    a = _PydanticLike(quantity=0.1, confidence=0.5)
    b = _PydanticLike(quantity=0.5, confidence=0.7)
    c = _PydanticLike(quantity=1.0, confidence=0.9)
    out = _consensus_decision([a, b, c])
    # `model_copy` returns a *new* object — not `a`.
    assert out is not a
    assert out.quantity == 0.5
    assert out.confidence == 0.7


# ── _build_consensus_plan ──────────────────────────────────


def test_build_consensus_plan_pydantic_path():
    """When the base plan has `model_copy`, the helper produces a
    fresh plan with the new `decisions` list AND appends
    "ensemble consensus" to risk_notes."""

    class _PlanLike:
        def __init__(self, decisions, risk_notes=""):
            self.decisions = decisions
            self.risk_notes = risk_notes

        def model_copy(self, *, update):
            new = _PlanLike(decisions=list(self.decisions), risk_notes=self.risk_notes)
            for k, v in update.items():
                setattr(new, k, v)
            return new

    base = _PlanLike(decisions=[_Decision()], risk_notes="watch the open")
    new_decisions = [_Decision(symbol="ETH"), _Decision(symbol="SOL")]
    out = _build_consensus_plan(base, new_decisions)

    assert out is not base  # new object
    assert out.decisions == new_decisions
    assert "ensemble consensus" in out.risk_notes
    assert "watch the open" in out.risk_notes  # original notes preserved


def test_build_consensus_plan_empty_risk_notes_no_leading_pipe():
    """If risk_notes was empty, the consensus annotation isn't
    prefixed by `' | '` (would render as ` | ensemble consensus`)."""

    class _PlanLike:
        def __init__(self, decisions, risk_notes=""):
            self.decisions = decisions
            self.risk_notes = risk_notes

        def model_copy(self, *, update):
            new = _PlanLike(decisions=list(self.decisions), risk_notes=self.risk_notes)
            for k, v in update.items():
                setattr(new, k, v)
            return new

    base = _PlanLike(decisions=[], risk_notes="")
    out = _build_consensus_plan(base, [_Decision()])
    # Pin the lstrip behavior — the helper does `.strip(" |")`.
    assert not out.risk_notes.startswith("|")
    assert "ensemble consensus" in out.risk_notes


def test_build_consensus_plan_plain_object_mutates():
    """For a non-pydantic plan, the helper mutates `decisions` and
    returns the same object."""
    base = _Decision()  # using as a placeholder "plan" with `decisions` attr — won't work
    # Simulate a plain object with a `decisions` attribute we can write.

    class _PlainPlan:
        def __init__(self):
            self.decisions = []

    base = _PlainPlan()
    new_decisions = [_Decision()]
    out = _build_consensus_plan(base, new_decisions)
    # Mutated in place + same identity.
    assert out is base
    assert base.decisions == new_decisions


def test_build_consensus_plan_swallows_attr_error_on_unmutable_plan():
    """If the plain-object plan's `decisions` attribute can't be
    written (e.g. read-only property), the helper swallows the
    AttributeError and returns the original — ensemble must never
    crash the cycle."""

    class _ReadOnlyPlan:
        @property
        def decisions(self):
            return []

    base = _ReadOnlyPlan()
    out = _build_consensus_plan(base, [_Decision()])
    # No crash; original returned.
    assert out is base


# ── _multiplier_from_agreement ─────────────────────────────


def test_multiplier_unanimous_returns_one():
    """100% agreement → full size."""
    out = _multiplier_from_agreement(1.0, quorum=2, n_variants=3)
    assert out == 1.0


def test_multiplier_at_quorum_threshold_returns_floor():
    """Exactly at the quorum threshold → 0.5 (the floor). Pin so
    a refactor doesn't shift this constant."""
    # 2 / 3 ≈ 0.667 — at this agreement, multiplier = 0.5.
    out = _multiplier_from_agreement(2 / 3, quorum=2, n_variants=3)
    assert out == 0.5


def test_multiplier_below_quorum_returns_floor():
    """Below the quorum threshold → still returns the floor (0.5).
    The skip_at threshold is separate — without it set, even very
    low agreement keeps 0.5 sizing."""
    out = _multiplier_from_agreement(0.1, quorum=2, n_variants=3)
    assert out == 0.5


def test_multiplier_skip_at_zeroes_below_threshold():
    """When `skip_at=0.4`, agreement < 0.4 → 0.0 (skip the cycle).
    Above the skip threshold, the normal ramp kicks in."""
    assert _multiplier_from_agreement(0.3, quorum=2, n_variants=3, skip_at=0.4) == 0.0
    # At threshold → not skipped (the check is `<`, exclusive).
    assert _multiplier_from_agreement(0.4, quorum=2, n_variants=3, skip_at=0.4) == 0.5


def test_multiplier_one_variant_returns_one():
    """Single-variant ensemble → no scaling (no one to disagree with).
    Defensive fallback — doesn't divide by zero or anything weird."""
    out = _multiplier_from_agreement(1.0, quorum=1, n_variants=1)
    assert out == 1.0


def test_multiplier_zero_variants_returns_one():
    """Zero variants → defensive 1.0 (the n_variants <= 1 guard
    catches this)."""
    out = _multiplier_from_agreement(1.0, quorum=0, n_variants=0)
    assert out == 1.0


def test_multiplier_interpolates_linearly_between_quorum_and_unanimity():
    """Halfway between quorum (2/3 → 0.5x) and unanimity (1.0 → 1.0x)
    is at (2/3 + 1.0) / 2 = 0.833... → multiplier ≈ 0.75."""
    halfway = (2 / 3 + 1.0) / 2  # 0.8333...
    out = _multiplier_from_agreement(halfway, quorum=2, n_variants=3)
    assert abs(out - 0.75) < 0.01


def test_multiplier_above_unanimity_capped_at_one():
    """Defensive: agreement > 1.0 (numerical drift) → 1.0 cap."""
    out = _multiplier_from_agreement(1.5, quorum=2, n_variants=3)
    assert out == 1.0


def test_multiplier_quorum_equals_n_variants_returns_floor_at_unanimity():
    """When quorum == n_variants, ANY agreement (including 1.0) hits
    the `agreement <= quorum_share` branch first → returns the floor
    (0.5). The post-quorum ramp never runs because there's no span
    above the quorum threshold. Pin this surprising semantic."""
    out = _multiplier_from_agreement(1.0, quorum=3, n_variants=3)
    assert out == 0.5


def test_multiplier_skip_at_zero_never_skips():
    """skip_at=0 is the same as no skip threshold — the `< 0` check
    never matches a non-negative agreement."""
    out = _multiplier_from_agreement(0.0, quorum=2, n_variants=3, skip_at=0.0)
    # 0.0 is NOT < 0.0, so falls through to the normal floor.
    assert out == 0.5
