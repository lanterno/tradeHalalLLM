"""Tests for halal/aaoifi_standard_30.py — Round-5 Wave 1.C."""

from __future__ import annotations

import pytest

from halal_trader.halal.aaoifi_standard_30 import (
    CLAUSES,
    StandardClause,
    TawarruqAssessment,
    TawarruqInputs,
    TawarruqStructure,
    TawarruqViolation,
    clause_by_id,
    clauses_for_violation,
    render_assessment,
    render_clause,
    render_coverage_matrix,
    screen_tawarruq,
)


def test_violation_string_values():
    assert TawarruqViolation.BAY_AL_INAH.value == "bay_al_inah"
    assert TawarruqViolation.NO_CONSTRUCTIVE_POSSESSION.value == "no_constructive_possession"
    assert TawarruqViolation.PRE_ARRANGED_BUYBACK.value == "pre_arranged_buyback"
    assert TawarruqViolation.RATE_TRACKING_PERPETUAL.value == "rate_tracking_perpetual"
    assert TawarruqViolation.SAME_COUNTERPARTY_LOOP.value == "same_counterparty_loop"


def test_structure_string_values():
    assert TawarruqStructure.ORGANISED_TAWARRUQ.value == "organised_tawarruq"
    assert TawarruqStructure.REVERSE_MURABAHA.value == "reverse_murabaha"
    assert TawarruqStructure.DIRECT_TAWARRUQ.value == "direct_tawarruq"
    assert TawarruqStructure.AGENT_BASED_TAWARRUQ.value == "agent_based_tawarruq"
    assert TawarruqStructure.COMMODITY_SALE.value == "commodity_sale"
    assert TawarruqStructure.INAH.value == "inah"


def test_clauses_sorted():
    keys = [tuple(int(s) for s in c.clause_id.split(".")) for c in CLAUSES]
    assert keys == sorted(keys)


def test_clauses_unique_ids():
    ids = [c.clause_id for c in CLAUSES]
    assert len(ids) == len(set(ids))


def test_every_violation_has_clause():
    covered = {c.violation for c in CLAUSES}
    assert covered == set(TawarruqViolation)


def test_clause_by_id_known():
    c = clause_by_id("4.3")
    assert c is not None
    assert c.violation is TawarruqViolation.BAY_AL_INAH


def test_clause_by_id_unknown():
    assert clause_by_id("99.99") is None


def test_clauses_for_violation():
    cs = clauses_for_violation(TawarruqViolation.BAY_AL_INAH)
    assert len(cs) >= 1


def test_standard_clause_validation_bad_id():
    with pytest.raises(ValueError):
        StandardClause(
            clause_id="bad",
            title="x",
            violation=TawarruqViolation.BAY_AL_INAH,
            summary="x",
        )


def test_standard_clause_validation_empty_title():
    with pytest.raises(ValueError):
        StandardClause(
            clause_id="1.1",
            title="",
            violation=TawarruqViolation.BAY_AL_INAH,
            summary="x",
        )


def test_standard_clause_validation_empty_summary():
    with pytest.raises(ValueError):
        StandardClause(
            clause_id="1.1",
            title="x",
            violation=TawarruqViolation.BAY_AL_INAH,
            summary=" ",
        )


def test_inputs_negative_markup_rejected():
    with pytest.raises(ValueError):
        TawarruqInputs(structure=TawarruqStructure.ORGANISED_TAWARRUQ, markup_bps=-1.0)


def test_inputs_defaults_clean():
    a = screen_tawarruq(TawarruqInputs(structure=TawarruqStructure.ORGANISED_TAWARRUQ))
    assert a.is_compliant


def test_inah_always_blocked_even_clean_flags():
    a = screen_tawarruq(TawarruqInputs(structure=TawarruqStructure.INAH))
    assert not a.is_compliant
    assert TawarruqViolation.BAY_AL_INAH in a.violations


def test_same_counterparty_buyback_flagged():
    a = screen_tawarruq(
        TawarruqInputs(
            structure=TawarruqStructure.ORGANISED_TAWARRUQ, same_counterparty_buyback=True
        )
    )
    assert TawarruqViolation.BAY_AL_INAH in a.violations


def test_no_possession_flagged():
    a = screen_tawarruq(
        TawarruqInputs(
            structure=TawarruqStructure.ORGANISED_TAWARRUQ, constructive_possession_taken=False
        )
    )
    assert TawarruqViolation.NO_CONSTRUCTIVE_POSSESSION in a.violations


def test_pre_arranged_buyback_flagged():
    a = screen_tawarruq(
        TawarruqInputs(
            structure=TawarruqStructure.ORGANISED_TAWARRUQ, pre_arranged_third_party=True
        )
    )
    assert TawarruqViolation.PRE_ARRANGED_BUYBACK in a.violations


def test_dependent_third_party_flagged():
    a = screen_tawarruq(
        TawarruqInputs(
            structure=TawarruqStructure.ORGANISED_TAWARRUQ, independent_third_party=False
        )
    )
    assert TawarruqViolation.SAME_COUNTERPARTY_LOOP in a.violations


def test_perpetual_rate_tracking_flagged():
    a = screen_tawarruq(
        TawarruqInputs(structure=TawarruqStructure.ORGANISED_TAWARRUQ, perpetual_rate_tracking=True)
    )
    assert TawarruqViolation.RATE_TRACKING_PERPETUAL in a.violations


def test_multiple_violations_combined():
    a = screen_tawarruq(
        TawarruqInputs(
            structure=TawarruqStructure.INAH,
            same_counterparty_buyback=True,
            constructive_possession_taken=False,
            pre_arranged_third_party=True,
            independent_third_party=False,
            perpetual_rate_tracking=True,
        )
    )
    assert not a.is_compliant
    # All five distinct violation kinds
    assert {
        TawarruqViolation.BAY_AL_INAH,
        TawarruqViolation.NO_CONSTRUCTIVE_POSSESSION,
        TawarruqViolation.PRE_ARRANGED_BUYBACK,
        TawarruqViolation.SAME_COUNTERPARTY_LOOP,
        TawarruqViolation.RATE_TRACKING_PERPETUAL,
    } <= a.violations


def test_assessment_structural_pin_compliant_with_violations():
    with pytest.raises(ValueError):
        TawarruqAssessment(
            structure=TawarruqStructure.ORGANISED_TAWARRUQ,
            violations=frozenset({TawarruqViolation.BAY_AL_INAH}),
            is_compliant=True,
        )


def test_assessment_structural_pin_non_compliant_without_violations():
    with pytest.raises(ValueError):
        TawarruqAssessment(
            structure=TawarruqStructure.ORGANISED_TAWARRUQ,
            violations=frozenset(),
            is_compliant=False,
        )


def test_render_clean():
    a = screen_tawarruq(TawarruqInputs(structure=TawarruqStructure.ORGANISED_TAWARRUQ))
    out = render_assessment(a)
    assert "✅" in out
    assert "organised_tawarruq" in out


def test_render_violations():
    a = screen_tawarruq(TawarruqInputs(structure=TawarruqStructure.INAH))
    out = render_assessment(a)
    assert "❌" in out
    assert "bay_al_inah" in out
    assert "§4.3" in out  # citation


def test_render_clause_format():
    c = clause_by_id("4.3")
    assert c is not None
    out = render_clause(c)
    assert "§4.3" in out
    assert "bay_al_inah" in out


def test_render_no_secret_leak():
    out = render_coverage_matrix()
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


def test_render_coverage_matrix_default_engages_all():
    out = render_coverage_matrix()
    assert f"{len(CLAUSES)}/{len(CLAUSES)}" in out


def test_render_coverage_matrix_partial():
    out = render_coverage_matrix({TawarruqViolation.BAY_AL_INAH})
    assert f"/{len(CLAUSES)}" in out


def test_e2e_organised_tawarruq_clean():
    a = screen_tawarruq(
        TawarruqInputs(
            structure=TawarruqStructure.ORGANISED_TAWARRUQ,
            same_counterparty_buyback=False,
            pre_arranged_third_party=False,
            constructive_possession_taken=True,
            independent_third_party=True,
            perpetual_rate_tracking=False,
            markup_bps=350.0,
        )
    )
    assert a.is_compliant


def test_e2e_loop_through_subsidiary_blocked():
    """Real-world failure: tri-party loop where counterparty is wholly-owned."""
    a = screen_tawarruq(
        TawarruqInputs(
            structure=TawarruqStructure.ORGANISED_TAWARRUQ, independent_third_party=False
        )
    )
    assert not a.is_compliant
    assert TawarruqViolation.SAME_COUNTERPARTY_LOOP in a.violations


def test_replay_consistency():
    inp = TawarruqInputs(structure=TawarruqStructure.ORGANISED_TAWARRUQ)
    assert screen_tawarruq(inp) == screen_tawarruq(inp)
