"""Tests for the gharar (excessive uncertainty) detector."""

from __future__ import annotations

import pytest

from halal_trader.halal.gharar_detector import (
    GhararAssessment,
    GhararInputs,
    GhararLevel,
    GhararPolicy,
    GhararSignal,
    assess_batch,
    assess_gharar,
    filter_blocked,
    is_tradable,
    render_assessment,
)


def _clean_inputs(**overrides) -> GhararInputs:
    """Construct a deliberately-clean instrument (no signals fire)."""
    base = {
        "instrument_id": "AAPL",
        "underlying_disclosed": True,
        "counterparty_disclosed": True,
        "asset_backing_transparent": True,
        "payoff_contingent_on_event": False,
        "delivery_date_specified": True,
        "derivative_layers": 0,
        "has_dual_class_unequal_rights": False,
        "fees_fully_disclosed": True,
    }
    base.update(overrides)
    return GhararInputs(**base)


# --- Enum string-value pins ---------------------------------------------------


def test_level_string_values():
    assert GhararLevel.NONE.value == "none"
    assert GhararLevel.MINOR.value == "minor"
    assert GhararLevel.MODERATE.value == "moderate"
    assert GhararLevel.SEVERE.value == "severe"


def test_signal_string_values():
    assert GhararSignal.UNDISCLOSED_UNDERLYING.value == "undisclosed_underlying"
    assert GhararSignal.COUNTERPARTY_UNDISCLOSED.value == "counterparty_undisclosed"
    assert GhararSignal.ASSET_BACKING_OPAQUE.value == "asset_backing_opaque"
    assert GhararSignal.CONTINGENT_PAYOFF.value == "contingent_payoff"
    assert GhararSignal.FUTURE_DELIVERY_INDETERMINATE.value == "future_delivery_indeterminate"
    assert GhararSignal.NESTED_DERIVATIVE_LAYERS.value == "nested_derivative_layers"
    assert GhararSignal.DUAL_CLASS_UNEQUAL_RIGHTS.value == "dual_class_unequal_rights"
    assert GhararSignal.OPAQUE_FEE_STRUCTURE.value == "opaque_fee_structure"


# --- Policy validation --------------------------------------------------------


def test_default_policy_pins():
    p = GhararPolicy()
    assert p.nested_layers_threshold == 2
    assert p.minor_score_threshold == 1
    assert p.moderate_score_threshold == 3
    assert p.severe_score_threshold == 5


def test_nested_layers_threshold_below_2_rejected():
    """Pin: at least 2 nested layers required to fire signal."""
    with pytest.raises(ValueError, match="nested_layers_threshold"):
        GhararPolicy(nested_layers_threshold=1)


def test_score_threshold_ordering_pin():
    """Pin: minor < moderate < severe score thresholds."""
    with pytest.raises(ValueError, match="minor < moderate < severe"):
        GhararPolicy(
            minor_score_threshold=3,
            moderate_score_threshold=2,
            severe_score_threshold=5,
        )


def test_zero_minor_threshold_rejected():
    with pytest.raises(ValueError, match="minor_score_threshold"):
        GhararPolicy(
            minor_score_threshold=0,
            moderate_score_threshold=3,
            severe_score_threshold=5,
        )


def test_policy_immutable():
    p = GhararPolicy()
    with pytest.raises(Exception):
        p.minor_score_threshold = 99  # type: ignore[misc]


# --- GhararInputs validation -------------------------------------------------


def test_empty_instrument_id_rejected():
    with pytest.raises(ValueError, match="instrument_id"):
        _clean_inputs(instrument_id="")


def test_whitespace_instrument_id_rejected():
    with pytest.raises(ValueError, match="instrument_id"):
        _clean_inputs(instrument_id="   ")


def test_negative_derivative_layers_rejected():
    with pytest.raises(ValueError, match="derivative_layers"):
        _clean_inputs(derivative_layers=-1)


def test_inputs_immutable():
    i = _clean_inputs()
    with pytest.raises(Exception):
        i.derivative_layers = 99  # type: ignore[misc]


# --- Signal detection: each signal in isolation ------------------------------


def test_clean_instrument_no_signals():
    a = assess_gharar(_clean_inputs())
    assert a.signals == frozenset()
    assert a.level is GhararLevel.NONE
    assert a.score == 0


def test_undisclosed_underlying_fires():
    a = assess_gharar(_clean_inputs(underlying_disclosed=False))
    assert GhararSignal.UNDISCLOSED_UNDERLYING in a.signals


def test_counterparty_undisclosed_fires():
    a = assess_gharar(_clean_inputs(counterparty_disclosed=False))
    assert GhararSignal.COUNTERPARTY_UNDISCLOSED in a.signals


def test_asset_backing_opaque_fires():
    a = assess_gharar(_clean_inputs(asset_backing_transparent=False))
    assert GhararSignal.ASSET_BACKING_OPAQUE in a.signals


def test_contingent_payoff_fires():
    a = assess_gharar(_clean_inputs(payoff_contingent_on_event=True))
    assert GhararSignal.CONTINGENT_PAYOFF in a.signals


def test_future_delivery_indeterminate_fires():
    a = assess_gharar(_clean_inputs(delivery_date_specified=False))
    assert GhararSignal.FUTURE_DELIVERY_INDETERMINATE in a.signals


def test_nested_derivative_layers_fires_at_threshold():
    """Pin: nested-layers threshold is inclusive (>=)."""
    a = assess_gharar(_clean_inputs(derivative_layers=2))
    assert GhararSignal.NESTED_DERIVATIVE_LAYERS in a.signals


def test_nested_derivative_layers_does_not_fire_below_threshold():
    a = assess_gharar(_clean_inputs(derivative_layers=1))
    assert GhararSignal.NESTED_DERIVATIVE_LAYERS not in a.signals


def test_dual_class_unequal_rights_fires():
    a = assess_gharar(_clean_inputs(has_dual_class_unequal_rights=True))
    assert GhararSignal.DUAL_CLASS_UNEQUAL_RIGHTS in a.signals


def test_opaque_fee_structure_fires():
    a = assess_gharar(_clean_inputs(fees_fully_disclosed=False))
    assert GhararSignal.OPAQUE_FEE_STRUCTURE in a.signals


# --- Score → Level mapping ---------------------------------------------------


def test_minor_at_score_one():
    """OPAQUE_FEE_STRUCTURE alone (weight 1) → score 1 → MINOR."""
    a = assess_gharar(_clean_inputs(fees_fully_disclosed=False))
    assert a.score == 1
    assert a.level is GhararLevel.MINOR


def test_minor_at_score_two():
    """ASSET_BACKING_OPAQUE alone (weight 2) → score 2 → MINOR (below 3)."""
    a = assess_gharar(_clean_inputs(asset_backing_transparent=False))
    assert a.score == 2
    assert a.level is GhararLevel.MINOR


def test_moderate_at_score_three():
    """Pin: score >= moderate_score_threshold (default 3) → MODERATE.
    Single weight-3 signal (UNDISCLOSED_UNDERLYING) → MODERATE."""
    a = assess_gharar(_clean_inputs(underlying_disclosed=False))
    assert a.score == 3
    assert a.level is GhararLevel.MODERATE


def test_severe_at_score_five():
    """Pin: score >= severe_score_threshold (default 5) → SEVERE.
    UNDISCLOSED_UNDERLYING (3) + ASSET_BACKING_OPAQUE (2) = 5 → SEVERE."""
    a = assess_gharar(
        _clean_inputs(
            underlying_disclosed=False,
            asset_backing_transparent=False,
        )
    )
    assert a.score == 5
    assert a.level is GhararLevel.SEVERE


def test_severe_at_six_two_weight_three():
    """Two weight-3 signals → score 6 → SEVERE."""
    a = assess_gharar(
        _clean_inputs(
            underlying_disclosed=False,
            counterparty_disclosed=False,
        )
    )
    assert a.score == 6
    assert a.level is GhararLevel.SEVERE


# --- Assessment validation ---------------------------------------------------


def test_assessment_immutable():
    a = assess_gharar(_clean_inputs())
    with pytest.raises(Exception):
        a.score = 99  # type: ignore[misc]


def test_none_level_with_signals_rejected():
    """Pin: NONE level → empty signals required."""
    with pytest.raises(ValueError, match="NONE level must have empty signals"):
        GhararAssessment(
            instrument_id="X",
            signals=frozenset({GhararSignal.OPAQUE_FEE_STRUCTURE}),
            level=GhararLevel.NONE,
            score=0,
        )


def test_non_none_level_without_signals_rejected():
    """Pin: non-NONE level → at least one signal required."""
    with pytest.raises(ValueError, match="non-NONE level requires at least one signal"):
        GhararAssessment(
            instrument_id="X",
            signals=frozenset(),
            level=GhararLevel.MINOR,
            score=1,
        )


def test_negative_score_rejected():
    with pytest.raises(ValueError, match="score"):
        GhararAssessment(
            instrument_id="X",
            signals=frozenset({GhararSignal.OPAQUE_FEE_STRUCTURE}),
            level=GhararLevel.MINOR,
            score=-1,
        )


def test_assessment_empty_id_rejected():
    with pytest.raises(ValueError, match="instrument_id"):
        GhararAssessment(
            instrument_id="",
            signals=frozenset(),
            level=GhararLevel.NONE,
            score=0,
        )


# --- is_tradable + filter_blocked --------------------------------------------


def test_tradable_for_none_minor_moderate():
    """Pin: NONE + MINOR + MODERATE are tradable (only SEVERE blocked)."""
    cases = [
        (GhararLevel.NONE, 0, frozenset()),
        (GhararLevel.MINOR, 1, frozenset({GhararSignal.OPAQUE_FEE_STRUCTURE})),
        (GhararLevel.MODERATE, 3, frozenset({GhararSignal.UNDISCLOSED_UNDERLYING})),
    ]
    for level, score, signals in cases:
        a = GhararAssessment(instrument_id="X", signals=signals, level=level, score=score)
        assert is_tradable(a) is True


def test_not_tradable_for_severe():
    """Pin: SEVERE is the only blocked level."""
    a = GhararAssessment(
        instrument_id="X",
        signals=frozenset({GhararSignal.UNDISCLOSED_UNDERLYING}),
        level=GhararLevel.SEVERE,
        score=10,
    )
    assert is_tradable(a) is False


def test_filter_blocked_returns_only_severe():
    clean = assess_gharar(_clean_inputs())
    severe = assess_gharar(
        _clean_inputs(
            underlying_disclosed=False,
            counterparty_disclosed=False,
        )
    )
    minor = assess_gharar(_clean_inputs(fees_fully_disclosed=False))
    blocked = filter_blocked([clean, severe, minor])
    assert len(blocked) == 1
    assert blocked[0].level is GhararLevel.SEVERE


# --- assess_batch ------------------------------------------------------------


def test_batch_returns_sorted_by_id():
    inputs = [
        _clean_inputs(instrument_id="ZZZ"),
        _clean_inputs(instrument_id="AAA"),
        _clean_inputs(instrument_id="MMM"),
    ]
    results = assess_batch(inputs)
    assert [a.instrument_id for a in results] == ["AAA", "MMM", "ZZZ"]


def test_batch_empty():
    assert assess_batch([]) == ()


# --- Render -------------------------------------------------------------------


def test_render_clean_shows_check():
    a = assess_gharar(_clean_inputs())
    out = render_assessment(a)
    assert "✅" in out
    assert "AAPL" in out
    assert "none" in out


def test_render_severe_shows_red():
    a = assess_gharar(
        _clean_inputs(
            underlying_disclosed=False,
            counterparty_disclosed=False,
        )
    )
    out = render_assessment(a)
    assert "🔴" in out
    assert "severe" in out


def test_render_includes_signal_labels():
    a = assess_gharar(_clean_inputs(underlying_disclosed=False))
    out = render_assessment(a)
    assert "undisclosed underlying" in out


def test_render_signals_sorted():
    """Pin: signal labels rendered alphabetically."""
    a = assess_gharar(
        _clean_inputs(
            underlying_disclosed=False,
            counterparty_disclosed=False,
        )
    )
    out = render_assessment(a)
    # "counterparty undisclosed" < "undisclosed underlying" alphabetically
    assert out.index("counterparty") < out.index("undisclosed underlying")


def test_render_no_secret_leak():
    """Pin: render output never includes prospectus / disclosure text."""
    a = assess_gharar(_clean_inputs())
    out = render_assessment(a)
    forbidden = [
        "prospectus",
        "Annex A",
        "Schedule II",
        "Authorization",
        "Bearer",
        "/api/",
        "private_key",
    ]
    for word in forbidden:
        assert word not in out


# --- E2E flows ----------------------------------------------------------------


def test_e2e_opaque_structured_note_severe():
    """A structured note with undisclosed underlying + opaque counterparty
    + 3 nested derivative wrappers + opaque fees → SEVERE."""
    inputs = GhararInputs(
        instrument_id="STRUCTNOTE_42",
        underlying_disclosed=False,  # weight 3
        counterparty_disclosed=False,  # weight 3
        derivative_layers=3,  # weight 2
        fees_fully_disclosed=False,  # weight 1
    )
    a = assess_gharar(inputs)
    # 3+3+2+1 = 9 → SEVERE
    assert a.score == 9
    assert a.level is GhararLevel.SEVERE
    assert is_tradable(a) is False
    expected = {
        GhararSignal.UNDISCLOSED_UNDERLYING,
        GhararSignal.COUNTERPARTY_UNDISCLOSED,
        GhararSignal.NESTED_DERIVATIVE_LAYERS,
        GhararSignal.OPAQUE_FEE_STRUCTURE,
    }
    assert a.signals == expected


def test_e2e_plain_equity_passes_clean():
    """Plain common stock — every flag clean → NONE."""
    inputs = GhararInputs(instrument_id="MSFT")
    a = assess_gharar(inputs)
    assert a.level is GhararLevel.NONE
    assert is_tradable(a) is True


def test_e2e_dual_class_governance_minor_only():
    """Google-style dual-class with unequal rights — alone, just MINOR."""
    inputs = GhararInputs(
        instrument_id="GOOG",
        has_dual_class_unequal_rights=True,
    )
    a = assess_gharar(inputs)
    assert a.score == 1
    assert a.level is GhararLevel.MINOR
    assert is_tradable(a) is True  # MODERATE is the cutoff; MINOR passes


def test_e2e_salam_forward_with_indeterminate_delivery_moderate():
    """A Salam contract with no specified delivery date is itself void
    per Standard 7 — fires FUTURE_DELIVERY_INDETERMINATE (weight 2)
    + COUNTERPARTY_UNDISCLOSED (weight 3) for an OTC counterparty
    we couldn't identify → score 5 → SEVERE."""
    inputs = GhararInputs(
        instrument_id="SALAM_42",
        delivery_date_specified=False,  # weight 2
        counterparty_disclosed=False,  # weight 3
    )
    a = assess_gharar(inputs)
    assert a.score == 5
    assert a.level is GhararLevel.SEVERE
    assert is_tradable(a) is False


def test_e2e_replay_consistency():
    """Pin: same inputs → equal assessment."""
    inputs = _clean_inputs(
        underlying_disclosed=False,
        derivative_layers=2,
    )
    a1 = assess_gharar(inputs)
    a2 = assess_gharar(inputs)
    assert a1 == a2


def test_e2e_custom_policy_strictness():
    """Operator-strict: MINOR threshold lowered to detect even tiny gharar.
    Default minor=1 already does that; here we test a stricter SEVERE
    threshold of 3 — single weight-3 signal becomes SEVERE."""
    strict = GhararPolicy(
        minor_score_threshold=1,
        moderate_score_threshold=2,
        severe_score_threshold=3,
    )
    a = assess_gharar(_clean_inputs(underlying_disclosed=False), policy=strict)
    # score=3 with severe_threshold=3 → SEVERE
    assert a.level is GhararLevel.SEVERE
    assert is_tradable(a) is False
