"""Tests for halal/esg_alignment.py — Round-5 Wave 11.H."""

from __future__ import annotations

import pytest

from halal_trader.halal.esg_alignment import (
    AlignmentPolicy,
    AlignmentTier,
    Pillar,
    PillarScore,
    render_report,
    score_alignment,
)

# --- Validation -----------------------------


def test_pillar_string_values():
    assert Pillar.ENVIRONMENTAL.value == "environmental"
    assert Pillar.SOCIAL.value == "social"
    assert Pillar.GOVERNANCE.value == "governance"
    assert Pillar.HALAL.value == "halal"


def test_tier_string_values():
    assert AlignmentTier.LEADER.value == "leader"
    assert AlignmentTier.ALIGNED.value == "aligned"
    assert AlignmentTier.NEUTRAL.value == "neutral"
    assert AlignmentTier.MISALIGNED.value == "misaligned"


def test_default_policy_weights_sum_to_one():
    p = AlignmentPolicy()
    assert sum(p.weights.values()) == pytest.approx(1.0)


def test_policy_weights_unsorted_thresholds_rejected():
    with pytest.raises(ValueError):
        AlignmentPolicy(neutral_threshold=0.8, aligned_threshold=0.5)


def test_policy_weights_dont_sum_to_one_rejected():
    with pytest.raises(ValueError):
        AlignmentPolicy(weights={Pillar.HALAL: 0.5, Pillar.ENVIRONMENTAL: 0.3})


def test_pillar_score_outside_unit_rejected():
    with pytest.raises(ValueError):
        PillarScore(pillar=Pillar.HALAL, score=1.5)


# --- Score alignment ----------------------


def test_empty_scores_neutral():
    r = score_alignment("ABC", [])
    assert r.tier is AlignmentTier.NEUTRAL


def test_unique_pillar_required():
    with pytest.raises(ValueError):
        score_alignment(
            "ABC",
            [
                PillarScore(pillar=Pillar.HALAL, score=0.9),
                PillarScore(pillar=Pillar.HALAL, score=0.5),
            ],
        )


def test_high_composite_leader_tier():
    scores = [
        PillarScore(pillar=Pillar.ENVIRONMENTAL, score=0.90),
        PillarScore(pillar=Pillar.SOCIAL, score=0.90),
        PillarScore(pillar=Pillar.GOVERNANCE, score=0.90),
        PillarScore(pillar=Pillar.HALAL, score=0.90),
    ]
    r = score_alignment("ABC", scores)
    assert r.tier is AlignmentTier.LEADER


def test_aligned_tier():
    scores = [
        PillarScore(pillar=Pillar.ENVIRONMENTAL, score=0.70),
        PillarScore(pillar=Pillar.SOCIAL, score=0.70),
        PillarScore(pillar=Pillar.GOVERNANCE, score=0.70),
        PillarScore(pillar=Pillar.HALAL, score=0.70),
    ]
    r = score_alignment("ABC", scores)
    assert r.tier is AlignmentTier.ALIGNED


def test_neutral_tier():
    scores = [
        PillarScore(pillar=Pillar.HALAL, score=0.50),
    ]
    r = score_alignment("ABC", scores)
    assert r.tier is AlignmentTier.NEUTRAL


def test_misaligned_tier():
    scores = [
        PillarScore(pillar=Pillar.HALAL, score=0.30),
    ]
    r = score_alignment("ABC", scores)
    assert r.tier is AlignmentTier.MISALIGNED


def test_partial_pillars_normalised():
    """When fewer than 4 pillars are scored, composite normalises to weights present."""
    scores = [PillarScore(pillar=Pillar.HALAL, score=0.90)]
    r = score_alignment("ABC", scores)
    assert r.composite_score == pytest.approx(0.90)


def test_composite_weighted_average():
    """Default: env 25%, social 25%, governance 20%, halal 30%."""
    scores = [
        PillarScore(pillar=Pillar.ENVIRONMENTAL, score=0.4),
        PillarScore(pillar=Pillar.SOCIAL, score=0.6),
        PillarScore(pillar=Pillar.GOVERNANCE, score=0.5),
        PillarScore(pillar=Pillar.HALAL, score=0.9),
    ]
    r = score_alignment("ABC", scores)
    expected = 0.4 * 0.25 + 0.6 * 0.25 + 0.5 * 0.20 + 0.9 * 0.30
    assert r.composite_score == pytest.approx(expected)


def test_empty_issuer_rejected():
    with pytest.raises(ValueError):
        score_alignment("", [])


# --- Render -------------------------------


def test_render_includes_pillars():
    scores = [
        PillarScore(pillar=Pillar.ENVIRONMENTAL, score=0.7, rationale="low carbon"),
        PillarScore(pillar=Pillar.HALAL, score=0.95),
    ]
    r = score_alignment("ABC", scores)
    out = render_report(r)
    assert "ABC" in out
    assert "environmental" in out
    assert "halal" in out
    assert "low carbon" in out


def test_render_leader_emoji():
    scores = [
        PillarScore(pillar=Pillar.ENVIRONMENTAL, score=0.95),
        PillarScore(pillar=Pillar.SOCIAL, score=0.90),
        PillarScore(pillar=Pillar.GOVERNANCE, score=0.90),
        PillarScore(pillar=Pillar.HALAL, score=0.95),
    ]
    r = score_alignment("ABC", scores)
    assert "🌟" in render_report(r)


def test_render_misaligned_emoji():
    scores = [PillarScore(pillar=Pillar.HALAL, score=0.10)]
    r = score_alignment("ABC", scores)
    assert "⚠️" in render_report(r)


def test_render_no_secret_leak():
    scores = [PillarScore(pillar=Pillar.HALAL, score=0.5)]
    r = score_alignment("ABC", scores)
    out = render_report(r)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ---------------------------


def test_e2e_clean_solar_company_leader():
    scores = [
        PillarScore(pillar=Pillar.ENVIRONMENTAL, score=0.95, rationale="100% renewable"),
        PillarScore(pillar=Pillar.SOCIAL, score=0.85, rationale="strong labour rights"),
        PillarScore(pillar=Pillar.GOVERNANCE, score=0.85),
        PillarScore(pillar=Pillar.HALAL, score=0.90, rationale="passes Std 21"),
    ]
    r = score_alignment("SunCorp", scores)
    assert r.tier is AlignmentTier.LEADER


def test_replay_consistency():
    scores = [PillarScore(pillar=Pillar.HALAL, score=0.7)]
    a = score_alignment("ABC", scores)
    b = score_alignment("ABC", scores)
    assert a == b
