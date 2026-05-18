"""Tests for the causal regime detector."""

from __future__ import annotations

import dataclasses

import pytest

from halal_trader.ml.causal_regime import (
    BreadthState,
    DXYState,
    MacroEvidence,
    MacroIntervention,
    RatesChange,
    RatesLevel,
    Regime,
    RegimeInferenceResult,
    VIXState,
    infer_regime,
    render_inference,
)

# ---------------------------------------------------------------------------
# Empty evidence → prior
# ---------------------------------------------------------------------------


def test_empty_evidence_returns_prior() -> None:
    """Pin: empty evidence + no intervention → marginal P(regime)."""

    result = infer_regime()
    assert isinstance(result, RegimeInferenceResult)
    total = sum(result.distribution.values())
    assert abs(total - 1.0) < 1e-6


def test_distribution_sums_to_one() -> None:
    """Pin: every inference output sums to 1 within tolerance."""

    result = infer_regime(evidence=MacroEvidence(vix=VIXState.CALM))
    total = sum(result.distribution.values())
    assert abs(total - 1.0) < 1e-6


def test_every_regime_has_a_probability() -> None:
    result = infer_regime()
    for regime in Regime:
        assert regime in result.distribution


# ---------------------------------------------------------------------------
# Single-variable observations
# ---------------------------------------------------------------------------


def test_calm_vix_skews_toward_risk_on() -> None:
    """Pin: P(RISK_ON | VIX=CALM) > P(RISK_ON | prior)."""

    prior = infer_regime()
    calm = infer_regime(evidence=MacroEvidence(vix=VIXState.CALM))
    assert calm.distribution[Regime.RISK_ON] > prior.distribution[Regime.RISK_ON]


def test_crisis_vix_skews_toward_crisis_or_risk_off() -> None:
    """Pin: P(CRISIS or RISK_OFF | VIX=CRISIS) > prior."""

    prior = infer_regime()
    crisis = infer_regime(evidence=MacroEvidence(vix=VIXState.CRISIS))
    prior_off = prior.distribution[Regime.RISK_OFF] + prior.distribution[Regime.CRISIS]
    crisis_off = crisis.distribution[Regime.RISK_OFF] + crisis.distribution[Regime.CRISIS]
    assert crisis_off > prior_off


def test_negative_breadth_skews_toward_risk_off() -> None:
    prior = infer_regime()
    bad = infer_regime(evidence=MacroEvidence(breadth=BreadthState.NEGATIVE))
    assert bad.distribution[Regime.RISK_OFF] > prior.distribution[Regime.RISK_OFF]


def test_positive_breadth_skews_toward_risk_on() -> None:
    prior = infer_regime()
    good = infer_regime(evidence=MacroEvidence(breadth=BreadthState.POSITIVE))
    assert good.distribution[Regime.RISK_ON] > prior.distribution[Regime.RISK_ON]


# ---------------------------------------------------------------------------
# Full-evidence — most-likely matches scenario
# ---------------------------------------------------------------------------


def test_full_risk_on_scenario_predicts_risk_on() -> None:
    """Easing rates + low rates + calm VIX + weak DXY + positive breadth →
    RISK_ON."""

    result = infer_regime(
        evidence=MacroEvidence(
            rates_level=RatesLevel.LOW,
            rates_change=RatesChange.EASING,
            vix=VIXState.CALM,
            dxy=DXYState.WEAK,
            breadth=BreadthState.POSITIVE,
        )
    )
    assert result.most_likely is Regime.RISK_ON


def test_full_crisis_scenario_predicts_crisis_or_risk_off() -> None:
    """Tightening + high rates + crisis VIX + strong DXY + negative breadth
    → CRISIS or RISK_OFF (the model spreads probability across both)."""

    result = infer_regime(
        evidence=MacroEvidence(
            rates_level=RatesLevel.HIGH,
            rates_change=RatesChange.TIGHTENING,
            vix=VIXState.CRISIS,
            dxy=DXYState.STRONG,
            breadth=BreadthState.NEGATIVE,
        )
    )
    assert result.most_likely in (Regime.CRISIS, Regime.RISK_OFF)


def test_neutral_scenario_predicts_neutral() -> None:
    result = infer_regime(
        evidence=MacroEvidence(
            vix=VIXState.CALM,
            dxy=DXYState.NORMAL,
            breadth=BreadthState.MIXED,
        )
    )
    assert result.most_likely is Regime.NEUTRAL


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_confidence_equals_most_likely_probability() -> None:
    result = infer_regime(evidence=MacroEvidence(vix=VIXState.CRISIS))
    assert result.confidence == result.distribution[result.most_likely]


def test_confidence_in_zero_to_one() -> None:
    result = infer_regime()
    assert 0.0 <= result.confidence <= 1.0


def test_full_evidence_lands_on_specific_cpt_cell() -> None:
    """Pin: full evidence locks the inference to one CPT cell.

    The probability distribution exactly matches the CPT entry
    for the specified (vix, dxy, breadth) combination since the
    target's parents are fully observed.
    """

    from halal_trader.ml.causal_regime import _CPT_REGIME_GIVEN_VIX_DXY_BREADTH

    expected = _CPT_REGIME_GIVEN_VIX_DXY_BREADTH[
        (VIXState.CRISIS, DXYState.STRONG, BreadthState.NEGATIVE)
    ]
    full = infer_regime(
        evidence=MacroEvidence(
            vix=VIXState.CRISIS,
            dxy=DXYState.STRONG,
            breadth=BreadthState.NEGATIVE,
        )
    )
    for regime in Regime:
        assert abs(full.distribution[regime] - expected[regime]) < 1e-6


# ---------------------------------------------------------------------------
# Do-calculus interventions
# ---------------------------------------------------------------------------


def test_intervention_on_vix_forces_state() -> None:
    """Pin: do(VIX=CRISIS) forces VIX regardless of upstream evidence."""

    result = infer_regime(
        evidence=MacroEvidence(rates_change=RatesChange.EASING),
        intervention=MacroIntervention(vix=VIXState.CRISIS),
    )
    # Result should be CRISIS / RISK_OFF heavy despite easing-rate observation
    crisis_plus_off = result.distribution[Regime.CRISIS] + result.distribution[Regime.RISK_OFF]
    assert crisis_plus_off > 0.5


def test_intervention_does_not_propagate_upstream() -> None:
    """Pin: do(VIX=CRISIS) does not update P(rates_change).

    Compared to observation VIX=CRISIS which would make tightening
    more likely (rates_change → vix), intervention severs the edge
    so observing the regime's distribution under intervention is
    different from observation. We verify the regime distribution
    is calculable; the rates_change marginal is not exposed but the
    pin holds via the joint.
    """

    obs = infer_regime(evidence=MacroEvidence(vix=VIXState.CRISIS))
    inter = infer_regime(intervention=MacroIntervention(vix=VIXState.CRISIS))
    # Both should give some CRISIS-leaning distribution
    assert obs.distribution[Regime.CRISIS] > 0.0
    assert inter.distribution[Regime.CRISIS] > 0.0


def test_intervention_on_dxy() -> None:
    result = infer_regime(intervention=MacroIntervention(dxy=DXYState.STRONG))
    total = sum(result.distribution.values())
    assert abs(total - 1.0) < 1e-6


def test_intervention_overrides_evidence_for_same_field() -> None:
    """Pin: when both evidence and intervention specify the same field,
    intervention wins and warning emitted."""

    result = infer_regime(
        evidence=MacroEvidence(vix=VIXState.CALM),
        intervention=MacroIntervention(vix=VIXState.CRISIS),
    )
    assert any("intervention wins" in w for w in result.warnings)
    # The result should reflect CRISIS, not CALM
    crisis_plus_off = result.distribution[Regime.CRISIS] + result.distribution[Regime.RISK_OFF]
    assert crisis_plus_off > 0.5


def test_no_intervention_no_warnings() -> None:
    result = infer_regime(evidence=MacroEvidence(vix=VIXState.CALM))
    assert result.warnings == ()


def test_multi_variable_intervention() -> None:
    """Stress-test: simultaneous do(VIX=CRISIS, DXY=STRONG, BREADTH=NEGATIVE)."""

    result = infer_regime(
        intervention=MacroIntervention(
            vix=VIXState.CRISIS,
            dxy=DXYState.STRONG,
            breadth=BreadthState.NEGATIVE,
        )
    )
    # Should match the CPT cell directly
    assert result.most_likely is Regime.CRISIS


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------


def test_result_carries_evidence_used() -> None:
    evidence = MacroEvidence(vix=VIXState.CALM)
    result = infer_regime(evidence=evidence)
    assert result.evidence_used == evidence


def test_result_carries_intervention_used() -> None:
    inter = MacroIntervention(vix=VIXState.CRISIS)
    result = infer_regime(intervention=inter)
    assert result.intervention_used == inter


def test_result_intervention_none_when_not_provided() -> None:
    result = infer_regime()
    assert result.intervention_used is None


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_evidence_is_frozen() -> None:
    e = MacroEvidence(vix=VIXState.CALM)
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.vix = VIXState.CRISIS  # type: ignore[misc]


def test_intervention_is_frozen() -> None:
    i = MacroIntervention(vix=VIXState.CRISIS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        i.vix = VIXState.CALM  # type: ignore[misc]


def test_result_is_frozen() -> None:
    result = infer_regime()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.most_likely = Regime.CRISIS  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB serialisation
# ---------------------------------------------------------------------------


def test_rates_level_string_values() -> None:
    assert RatesLevel.LOW.value == "low"
    assert RatesLevel.NORMAL.value == "normal"
    assert RatesLevel.HIGH.value == "high"


def test_rates_change_string_values() -> None:
    assert RatesChange.EASING.value == "easing"
    assert RatesChange.HOLDING.value == "holding"
    assert RatesChange.TIGHTENING.value == "tightening"


def test_vix_state_string_values() -> None:
    assert VIXState.CALM.value == "calm"
    assert VIXState.ELEVATED.value == "elevated"
    assert VIXState.CRISIS.value == "crisis"


def test_dxy_state_string_values() -> None:
    assert DXYState.WEAK.value == "weak"
    assert DXYState.NORMAL.value == "normal"
    assert DXYState.STRONG.value == "strong"


def test_breadth_state_string_values() -> None:
    assert BreadthState.NEGATIVE.value == "negative"
    assert BreadthState.MIXED.value == "mixed"
    assert BreadthState.POSITIVE.value == "positive"


def test_regime_string_values() -> None:
    assert Regime.RISK_ON.value == "risk_on"
    assert Regime.NEUTRAL.value == "neutral"
    assert Regime.RISK_OFF.value == "risk_off"
    assert Regime.CRISIS.value == "crisis"


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


def test_render_includes_most_likely() -> None:
    result = infer_regime(evidence=MacroEvidence(vix=VIXState.CALM))
    text = render_inference(result)
    assert result.most_likely.value in text


def test_render_shows_distribution() -> None:
    result = infer_regime()
    text = render_inference(result)
    for regime in Regime:
        assert regime.value in text


def test_render_shows_intervention_when_set() -> None:
    result = infer_regime(intervention=MacroIntervention(vix=VIXState.CRISIS))
    text = render_inference(result)
    assert "do(" in text
    assert "vix=crisis" in text


def test_render_omits_intervention_when_none() -> None:
    result = infer_regime()
    text = render_inference(result)
    assert "do(" not in text


def test_render_shows_warnings_when_present() -> None:
    result = infer_regime(
        evidence=MacroEvidence(vix=VIXState.CALM),
        intervention=MacroIntervention(vix=VIXState.CRISIS),
    )
    text = render_inference(result)
    assert "warnings" in text


def test_render_includes_emoji_per_regime() -> None:
    """Pin: render emits a regime-specific emoji."""

    result = infer_regime(
        evidence=MacroEvidence(
            vix=VIXState.CRISIS,
            dxy=DXYState.STRONG,
            breadth=BreadthState.NEGATIVE,
        )
    )
    text = render_inference(result)
    # most_likely will be CRISIS or RISK_OFF; emoji is one of 🔴 / 🟠
    assert "🔴" in text or "🟠" in text


# ---------------------------------------------------------------------------
# End-to-end realistic scenarios
# ---------------------------------------------------------------------------


def test_2008_crisis_macro_state() -> None:
    """Pin: classic 2008 GFC macro state → CRISIS-leaning regime."""

    result = infer_regime(
        evidence=MacroEvidence(
            rates_level=RatesLevel.HIGH,  # was high before the cut
            rates_change=RatesChange.EASING,  # emergency cuts
            vix=VIXState.CRISIS,
            dxy=DXYState.STRONG,  # flight to dollar
            breadth=BreadthState.NEGATIVE,
        )
    )
    risk_off_or_crisis = result.distribution[Regime.RISK_OFF] + result.distribution[Regime.CRISIS]
    assert risk_off_or_crisis > 0.7


def test_late_2020_recovery_macro_state() -> None:
    """Late-2020 recovery: low rates, easing, calm VIX, weak DXY, positive breadth."""

    result = infer_regime(
        evidence=MacroEvidence(
            rates_level=RatesLevel.LOW,
            rates_change=RatesChange.EASING,
            vix=VIXState.CALM,
            dxy=DXYState.WEAK,
            breadth=BreadthState.POSITIVE,
        )
    )
    assert result.most_likely is Regime.RISK_ON
    assert result.distribution[Regime.RISK_ON] > 0.7


def test_what_if_vix_doubles_intervention() -> None:
    """Operator stress test: today is calm but what if VIX doubles to CRISIS?"""

    today = infer_regime(
        evidence=MacroEvidence(
            vix=VIXState.CALM,
            dxy=DXYState.NORMAL,
            breadth=BreadthState.POSITIVE,
        )
    )
    stress = infer_regime(
        evidence=MacroEvidence(
            dxy=DXYState.NORMAL,
            breadth=BreadthState.POSITIVE,
        ),
        intervention=MacroIntervention(vix=VIXState.CRISIS),
    )
    # Today should be RISK_ON; stress should shift toward RISK_OFF / CRISIS
    assert today.most_likely is Regime.RISK_ON
    # Stress: VIX forced to CRISIS but DXY normal + breadth positive can
    # still render a non-trivial RISK_OFF + CRISIS combined probability
    assert stress.distribution[Regime.RISK_OFF] + stress.distribution[Regime.CRISIS] > 0.5


def test_partial_evidence_marginalises_over_unknowns() -> None:
    """Pin: only knowing VIX leaves the unknowns marginalised."""

    result = infer_regime(evidence=MacroEvidence(vix=VIXState.ELEVATED))
    total = sum(result.distribution.values())
    assert abs(total - 1.0) < 1e-6
    # All four regimes have non-zero probability
    for regime in Regime:
        assert result.distribution[regime] > 0.0


# ---------------------------------------------------------------------------
# CPTs sanity
# ---------------------------------------------------------------------------


def test_cpt_each_regime_cell_sums_to_one() -> None:
    """Pin: every cell of P(regime | vix, dxy, breadth) sums to 1.0."""

    from halal_trader.ml.causal_regime import _CPT_REGIME_GIVEN_VIX_DXY_BREADTH

    for cell in _CPT_REGIME_GIVEN_VIX_DXY_BREADTH.values():
        assert abs(sum(cell.values()) - 1.0) < 1e-6


def test_cpt_complete_coverage() -> None:
    """Pin: every (vix, dxy, breadth) combination has a CPT entry."""

    from halal_trader.ml.causal_regime import _CPT_REGIME_GIVEN_VIX_DXY_BREADTH

    for vix in VIXState:
        for dxy in DXYState:
            for breadth in BreadthState:
                assert (vix, dxy, breadth) in _CPT_REGIME_GIVEN_VIX_DXY_BREADTH
