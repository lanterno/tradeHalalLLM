"""Tests for `halal/scholar_profiles.py`.

Covers the registry lookup contract (case-insensitive, KeyError on
typos), the threshold evaluator (per-ratio caps, missing-input
skipping, defaults match AAOIFI 30/5/33), the provider-weight
re-mapping helper, and the `consensus_with_profile` integration
path that ties it back to the consensus aggregator.
"""

from __future__ import annotations

import pytest

from halal_trader.halal.consensus import (
    ConsensusPolicy,
    Decision,
    ScreeningOpinion,
)
from halal_trader.halal.scholar_profiles import (
    AAOIFI_DEFAULT,
    DELORENZO_DJIM,
    TAQI_USMANI,
    ScholarProfile,
    ScreeningThresholds,
    apply_profile_weights,
    consensus_with_profile,
    evaluate_thresholds,
    get_profile,
    list_profiles,
    profile_to_dict,
    register_profile,
)

# ── threshold defaults ───────────────────────────────────


def test_default_thresholds_match_aaoifi_30_5_33():
    """The published AAOIFI triplet must be the default; pin so a
    copy-paste error in a new profile doesn't mutate the global
    default."""
    t = ScreeningThresholds()
    assert t.debt_to_marketcap_max == 0.30
    assert t.non_permissible_income_max == 0.05
    assert t.cash_and_receivables_max == 0.33


def test_aaoifi_default_profile_uses_aaoifi_thresholds():
    assert AAOIFI_DEFAULT.thresholds == ScreeningThresholds()
    assert AAOIFI_DEFAULT.default_policy == ConsensusPolicy.STRICT


def test_taqi_usmani_profile_is_stricter_than_default():
    """Pin the relative ordering — Taqi Usmani profile must remain
    stricter than the AAOIFI default."""
    assert (
        TAQI_USMANI.thresholds.debt_to_marketcap_max
        < AAOIFI_DEFAULT.thresholds.debt_to_marketcap_max
    )
    assert (
        TAQI_USMANI.thresholds.non_permissible_income_max
        < AAOIFI_DEFAULT.thresholds.non_permissible_income_max
    )


def test_delorenzo_djim_profile_uses_majority_policy():
    """DeLorenzo profile defaults to MAJORITY, not STRICT — pin
    the methodology divergence so a refactor can't reset every
    profile to the same default."""
    assert DELORENZO_DJIM.default_policy == ConsensusPolicy.MAJORITY


# ── evaluate_thresholds ──────────────────────────────────


def test_evaluate_thresholds_passes_when_all_under_caps():
    ok, violations = evaluate_thresholds(
        profile=AAOIFI_DEFAULT,
        debt_to_marketcap=0.10,
        non_permissible_income=0.01,
        cash_and_receivables=0.20,
    )
    assert ok is True
    assert violations == []


def test_evaluate_thresholds_flags_debt_violation():
    ok, violations = evaluate_thresholds(
        profile=AAOIFI_DEFAULT,
        debt_to_marketcap=0.45,
        non_permissible_income=0.01,
    )
    assert ok is False
    assert len(violations) == 1
    assert "debt" in violations[0]


def test_evaluate_thresholds_flags_multiple_violations_at_once():
    ok, violations = evaluate_thresholds(
        profile=AAOIFI_DEFAULT,
        debt_to_marketcap=0.50,
        non_permissible_income=0.10,
        cash_and_receivables=0.50,
    )
    assert ok is False
    assert len(violations) == 3


def test_evaluate_thresholds_skips_missing_inputs_not_silently_passes():
    """Pin: a None value is "not measured", not "passes". A partial
    filing must not silently approve."""
    ok, violations = evaluate_thresholds(
        profile=AAOIFI_DEFAULT,
        debt_to_marketcap=0.50,  # over cap
        non_permissible_income=None,  # not measured
        cash_and_receivables=None,
    )
    assert ok is False
    assert len(violations) == 1


def test_evaluate_thresholds_with_taqi_usmani_rejects_aaoifi_passing_case():
    """A 28% debt ratio passes AAOIFI's 30% cap but fails Taqi
    Usmani's 25%. Pin the profile differentiation."""
    ok_aaoifi, _ = evaluate_thresholds(profile=AAOIFI_DEFAULT, debt_to_marketcap=0.28)
    ok_taqi, _ = evaluate_thresholds(profile=TAQI_USMANI, debt_to_marketcap=0.28)
    assert ok_aaoifi is True
    assert ok_taqi is False


# ── apply_profile_weights ────────────────────────────────


def _op(source: str, decision: str, weight: float = 1.0) -> ScreeningOpinion:
    return ScreeningOpinion(source=source, decision=decision, weight=weight)


def test_apply_profile_weights_returns_unchanged_when_profile_has_no_weights():
    """If the profile carries no provider_weights, the helper must
    return the input weights unchanged."""
    out = apply_profile_weights(
        AAOIFI_DEFAULT, [_op("zoya", "halal", 1.0), _op("musaffa", "halal", 2.0)]
    )
    assert [op.weight for op in out] == [1.0, 2.0]


def test_apply_profile_weights_overrides_listed_sources():
    """Provider listed in the profile gets the profile's weight; an
    unlisted provider keeps its existing weight."""
    out = apply_profile_weights(
        TAQI_USMANI,  # weights: musaffa=1.5, idealratings=1.5, zoya=1.0
        [
            _op("zoya", "halal", weight=10.0),  # overridden to 1.0
            _op("musaffa", "halal", weight=10.0),  # overridden to 1.5
            _op("custom", "halal", weight=10.0),  # not in profile → keeps 10.0
        ],
    )
    weights = {op.source: op.weight for op in out}
    assert weights["zoya"] == 1.0
    assert weights["musaffa"] == 1.5
    assert weights["custom"] == 10.0


def test_apply_profile_weights_does_not_mutate_input():
    inputs = [_op("zoya", "halal", weight=10.0)]
    apply_profile_weights(TAQI_USMANI, inputs)
    assert inputs[0].weight == 10.0  # unchanged


def test_apply_profile_weights_preserves_decision_and_criteria():
    """The re-weight helper must touch only the weight — decision
    and criteria flow through unchanged."""
    op = ScreeningOpinion(
        source="zoya",
        decision="doubtful",
        weight=2.0,
        criteria={"debt_pct": 0.4},
    )
    [out] = apply_profile_weights(TAQI_USMANI, [op])
    assert out.decision == "doubtful"
    assert out.criteria == {"debt_pct": 0.4}


# ── consensus_with_profile ───────────────────────────────


def test_consensus_with_profile_uses_profile_default_policy():
    """When `policy=None`, the helper picks the profile's
    `default_policy`. Pin so dashboard / CLI callers don't have to
    re-pick the policy themselves."""
    # AAOIFI_DEFAULT → STRICT, so a single not_halal must reject.
    result = consensus_with_profile(
        AAOIFI_DEFAULT,
        [_op("zoya", "halal"), _op("musaffa", "not_halal")],
    )
    assert result.policy == ConsensusPolicy.STRICT
    assert result.decision == Decision.NOT_HALAL


def test_consensus_with_profile_lets_caller_override_policy():
    """The dashboard's "show me the conservative read" toggle must
    be able to override the profile's chosen policy."""
    result = consensus_with_profile(
        DELORENZO_DJIM,  # default: MAJORITY
        [_op("a", "halal"), _op("b", "halal"), _op("c", "not_halal")],
        policy=ConsensusPolicy.STRICT,
    )
    assert result.policy == ConsensusPolicy.STRICT
    assert result.decision == Decision.NOT_HALAL


def test_consensus_with_profile_applies_weights_only_for_weighted_policy():
    """STRICT / MAJORITY ignore weights, so re-weighting is wasteful
    work. Pin: the helper only re-weights under WEIGHTED."""
    # Use TAQI_USMANI weights (zoya=1.0, musaffa=1.5). Run with
    # STRICT — weights don't matter, but the helper must not
    # accidentally re-weight under STRICT.
    result = consensus_with_profile(
        TAQI_USMANI,
        [_op("zoya", "halal", 99.0), _op("musaffa", "halal", 99.0)],
        policy=ConsensusPolicy.STRICT,
    )
    # STRICT: unanimous halal → halal regardless of weights
    assert result.decision == Decision.HALAL
    # The opinions in the result should still carry the input weights
    # because STRICT didn't re-weight them.
    weights = [op.weight for op in result.opinions]
    assert weights == [99.0, 99.0]


def test_consensus_with_profile_under_weighted_re_weights_opinions():
    """Pin: under WEIGHTED, the profile's per-source weights take
    over. A high-trust source can outvote two low-trust sources."""
    result = consensus_with_profile(
        TAQI_USMANI,  # musaffa=1.5, zoya=1.0
        [
            _op("zoya", "halal"),  # weight → 1.0
            _op("zoya", "halal"),  # weight → 1.0
            _op("musaffa", "doubtful"),  # weight → 1.5
            _op("musaffa", "doubtful"),  # weight → 1.5
        ],
        policy=ConsensusPolicy.WEIGHTED,
    )
    # halal sums to 2.0; doubtful to 3.0 → doubtful wins.
    assert result.decision == Decision.DOUBTFUL


# ── registry ─────────────────────────────────────────────


def test_get_profile_is_case_insensitive():
    """Operators paste names from various sources; case shouldn't
    matter."""
    assert get_profile("AAOIFI_default") is AAOIFI_DEFAULT
    assert get_profile("aaoifi_default") is AAOIFI_DEFAULT


def test_get_profile_raises_with_available_list_on_typo():
    """Pin: a typo'd profile name must surface immediately with
    the candidate list, not degrade silently to a default."""
    with pytest.raises(KeyError, match="unknown scholar profile"):
        get_profile("haalal_default")


def test_list_profiles_returns_all_three_builtins():
    profiles = list_profiles()
    names = {p.name for p in profiles}
    assert "aaoifi_default" in names
    assert "taqi_usmani" in names
    assert "delorenzo_djim" in names


def test_register_profile_makes_new_profile_findable():
    custom = ScholarProfile(
        name="custom_test_profile",
        description="Test-only registration target.",
    )
    register_profile(custom)
    try:
        assert get_profile("custom_test_profile") is custom
    finally:
        # Hand-cleanup — re-importing the module would also work but
        # would lose the existing builtins for other tests.
        from halal_trader.halal import scholar_profiles

        del scholar_profiles._PROFILES["custom_test_profile"]


def test_register_profile_rejects_empty_name():
    """A nameless profile is unfindable; pin the early failure."""
    with pytest.raises(ValueError, match="must not be empty"):
        register_profile(ScholarProfile(name="", description=""))


# ── profile_to_dict ──────────────────────────────────────


def test_profile_to_dict_round_trips_thresholds():
    """The audit-trail dict must capture every threshold so a
    later replay can reconstruct the decision exactly."""
    out = profile_to_dict(TAQI_USMANI)
    assert out["thresholds"]["debt_to_marketcap_max"] == 0.25
    assert out["thresholds"]["non_permissible_income_max"] == 0.03
    assert out["thresholds"]["cash_and_receivables_max"] == 0.33


def test_profile_to_dict_uses_string_for_policy():
    """Pin: the policy field must be the JSON-friendly string, not
    the enum (which doesn't survive json.dumps cleanly)."""
    out = profile_to_dict(AAOIFI_DEFAULT)
    assert out["default_policy"] == "strict"
    assert isinstance(out["default_policy"], str)


def test_profile_to_dict_includes_provider_weights_as_plain_dict():
    out = profile_to_dict(TAQI_USMANI)
    assert isinstance(out["provider_weights"], dict)
    assert out["provider_weights"]["musaffa"] == 1.5


def test_profile_to_dict_carries_rulings_doc_pointer():
    """The pointer to the markdown ruling doc lets the dashboard
    link out to the long-form rationale."""
    out = profile_to_dict(AAOIFI_DEFAULT)
    assert out["rulings_doc"] is not None
    assert "halal_jurisprudence" in out["rulings_doc"]
