"""Tests for `halal/consensus.py` (multi-source halal-screening
consensus aggregator).

Covers each of the three resolution policies (STRICT / MAJORITY /
WEIGHTED), the conservatism tiebreak, the empty-input contract,
the dissenter tracking, and the string-coercion entry point.
"""

from __future__ import annotations

import pytest

from halal_trader.halal.consensus import (
    ConsensusDecision,
    ConsensusPolicy,
    Decision,
    ScreeningOpinion,
    consensus,
)


def _op(source: str, decision: str, weight: float = 1.0) -> ScreeningOpinion:
    return ScreeningOpinion(source=source, decision=decision, weight=weight)


# ── empty-input contract ─────────────────────────────────


def test_empty_opinions_resolves_to_doubtful():
    """No-opinions = unattested = refuse to trade. The conservative
    default protects the caller from "fail open" surprises."""
    result = consensus([])
    assert result.decision == Decision.DOUBTFUL
    assert result.policy == ConsensusPolicy.STRICT
    assert result.opinions == []
    assert "no opinions" in result.reason.lower()


# ── STRICT ────────────────────────────────────────────────


def test_strict_unanimous_halal_is_halal():
    result = consensus(
        [_op("zoya", "halal"), _op("musaffa", "halal"), _op("idealratings", "halal")]
    )
    assert result.decision == Decision.HALAL
    assert result.dissenters == []


def test_strict_any_not_halal_rejects_even_if_others_halal():
    """Pin the conservative default — a single not_halal vote
    overrides any number of halal votes."""
    result = consensus(
        [
            _op("zoya", "halal"),
            _op("musaffa", "not_halal"),  # one rejection
            _op("idealratings", "halal"),
        ]
    )
    assert result.decision == Decision.NOT_HALAL
    # Dissenters in STRICT = sources that didn't agree with the winner.
    assert "zoya" in result.dissenters
    assert "musaffa" not in result.dissenters
    assert "musaffa" in result.reason


def test_strict_any_doubtful_when_no_not_halal_yields_doubtful():
    result = consensus(
        [
            _op("zoya", "halal"),
            _op("musaffa", "doubtful"),
            _op("idealratings", "halal"),
        ]
    )
    assert result.decision == Decision.DOUBTFUL
    assert "musaffa" in result.reason


def test_strict_not_halal_takes_precedence_over_doubtful():
    """When *both* doubtful and not_halal are present, not_halal
    wins — the precedence ladder must not wobble."""
    result = consensus(
        [
            _op("zoya", "doubtful"),
            _op("musaffa", "not_halal"),
        ]
    )
    assert result.decision == Decision.NOT_HALAL


# ── MAJORITY ─────────────────────────────────────────────


def test_majority_picks_most_common_decision():
    result = consensus(
        [
            _op("a", "halal"),
            _op("b", "halal"),
            _op("c", "doubtful"),
        ],
        policy=ConsensusPolicy.MAJORITY,
    )
    assert result.decision == Decision.HALAL
    assert result.dissenters == ["c"]


def test_majority_tiebreak_picks_more_conservative():
    """When two decisions tie on count, the more conservative one
    wins. Pin so a refactor of the rank lookup can't flip the
    direction silently."""
    result = consensus(
        [
            _op("a", "halal"),
            _op("b", "halal"),
            _op("c", "not_halal"),
            _op("d", "not_halal"),
        ],
        policy=ConsensusPolicy.MAJORITY,
    )
    assert result.decision == Decision.NOT_HALAL
    assert "tie" in result.reason.lower()


def test_majority_three_way_tie_resolves_to_not_halal():
    """1-1-1 across all three states → tie at top → most
    conservative wins."""
    result = consensus(
        [
            _op("a", "halal"),
            _op("b", "doubtful"),
            _op("c", "not_halal"),
        ],
        policy=ConsensusPolicy.MAJORITY,
    )
    assert result.decision == Decision.NOT_HALAL


# ── WEIGHTED ─────────────────────────────────────────────


def test_weighted_winner_uses_weights_not_counts():
    """A high-weight halal vote can outweigh two lower-weight
    doubtful votes."""
    result = consensus(
        [
            _op("a", "halal", weight=5.0),
            _op("b", "doubtful", weight=1.0),
            _op("c", "doubtful", weight=1.0),
        ],
        policy=ConsensusPolicy.WEIGHTED,
    )
    assert result.decision == Decision.HALAL


def test_weighted_tie_breaks_to_more_conservative():
    result = consensus(
        [
            _op("a", "halal", weight=2.0),
            _op("b", "doubtful", weight=2.0),
        ],
        policy=ConsensusPolicy.WEIGHTED,
    )
    assert result.decision == Decision.DOUBTFUL


def test_weighted_clamps_negative_weights_to_zero():
    """A buggy config that sets a negative weight must not silently
    invert the contribution. Pin the clamp."""
    result = consensus(
        [
            _op("a", "not_halal", weight=-99.0),  # clamps to 0 → zero contribution
            _op("b", "halal", weight=1.0),
        ],
        policy=ConsensusPolicy.WEIGHTED,
    )
    assert result.decision == Decision.HALAL


def test_weighted_zero_total_defaults_to_doubtful():
    """If every weight is zero, the aggregator can't tell — must
    default to doubtful, not halal."""
    result = consensus(
        [
            _op("a", "halal", weight=0.0),
            _op("b", "halal", weight=0.0),
        ],
        policy=ConsensusPolicy.WEIGHTED,
    )
    assert result.decision == Decision.DOUBTFUL


# ── dissenter tracking ───────────────────────────────────


def test_dissenters_lists_sources_that_did_not_agree():
    result = consensus([_op("a", "halal"), _op("b", "halal"), _op("c", "not_halal")])
    # STRICT yields not_halal (one rejector). Dissenters = sources
    # that disagree with not_halal = a, b.
    assert sorted(result.dissenters) == ["a", "b"]


def test_no_dissenters_on_unanimous_decision():
    result = consensus(
        [_op("a", "halal"), _op("b", "halal")],
    )
    assert result.dissenters == []


# ── string coercion ──────────────────────────────────────


def test_decision_can_be_passed_as_string():
    """A JSON-loaded opinion (lossy on the enum) should still feed
    in. Pin the coercion path."""
    result = consensus([_op("a", "halal"), _op("b", "halal")])
    assert result.decision == Decision.HALAL


def test_unknown_decision_string_raises():
    """Pin so a typo'd 'haalal' surfaces immediately rather than
    silently degrading."""
    with pytest.raises(ValueError, match="unknown halal decision"):
        consensus([_op("a", "totally_not_a_decision")])


def test_decision_can_be_passed_as_enum():
    op = ScreeningOpinion(source="a", decision=Decision.HALAL)
    result = consensus([op, op])
    assert result.decision == Decision.HALAL


# ── output structure ─────────────────────────────────────


def test_consensus_decision_is_immutable():
    """Pin frozen dataclass so a downstream consumer can safely
    cache the result."""
    result = consensus([_op("a", "halal")])
    assert isinstance(result, ConsensusDecision)
    with pytest.raises(Exception):  # FrozenInstanceError
        result.decision = Decision.NOT_HALAL  # type: ignore[misc]


def test_consensus_preserves_input_order():
    """The opinions list in the result must reflect the input order
    so audit replay is deterministic."""
    a = _op("a", "halal")
    b = _op("b", "halal")
    c = _op("c", "halal")
    result = consensus([b, a, c])
    assert [o.source for o in result.opinions] == ["b", "a", "c"]


def test_consensus_carries_policy_used():
    result = consensus(
        [_op("a", "halal"), _op("b", "halal")],
        policy=ConsensusPolicy.MAJORITY,
    )
    assert result.policy == ConsensusPolicy.MAJORITY


# ── reason string ────────────────────────────────────────


def test_strict_reason_names_rejecting_sources():
    result = consensus([_op("zoya", "halal"), _op("musaffa", "not_halal"), _op("ir", "not_halal")])
    assert "musaffa" in result.reason
    assert "ir" in result.reason


def test_majority_reason_includes_score():
    result = consensus(
        [_op("a", "halal"), _op("b", "halal"), _op("c", "doubtful")],
        policy=ConsensusPolicy.MAJORITY,
    )
    # reason mentions the score
    assert "2" in result.reason


def test_weighted_reason_mentions_weighted_label():
    result = consensus(
        [_op("a", "halal", weight=2.0), _op("b", "halal", weight=1.0)],
        policy=ConsensusPolicy.WEIGHTED,
    )
    assert "weighted" in result.reason.lower()
