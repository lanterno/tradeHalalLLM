"""Tests for halal/aaoifi_standard_17.py — Round-5 Wave 1.B."""

from __future__ import annotations

import pytest

from halal_trader.halal.aaoifi_standard_17 import (
    CLAUSES,
    TRADABLE_IN_SECONDARY,
    SukukAssessment,
    SukukClause,
    SukukIssuanceInputs,
    SukukRule,
    SukukType,
    clause_by_id,
    clauses_for_rule,
    is_tradable_in_secondary,
    render_assessment,
    render_clause,
    render_coverage_matrix,
    screen_sukuk,
)


def test_sukuk_rule_string_values():
    assert SukukRule.UNDERLYING_ASSET_HALAL.value == "underlying_asset_halal"
    assert SukukRule.OWNERSHIP_TRANSFER_REAL.value == "ownership_transfer_real"
    assert SukukRule.NO_GUARANTEED_PRINCIPAL.value == "no_guaranteed_principal"
    assert SukukRule.PROFIT_FROM_REAL_ACTIVITY.value == "profit_from_real_activity"
    assert SukukRule.LOSS_SHARING_PROPORTIONAL.value == "loss_sharing_proportional"
    assert SukukRule.NO_INTEREST_RATE_LINK.value == "no_interest_rate_link"
    assert SukukRule.TANGIBLE_ASSET_RATIO.value == "tangible_asset_ratio"
    assert SukukRule.SECONDARY_MARKET_TRADABILITY.value == "secondary_market_tradability"
    assert SukukRule.PURPOSE_LISTED_HALAL.value == "purpose_listed_halal"
    assert SukukRule.SHARIA_BOARD_OPINION.value == "sharia_board_opinion"
    assert SukukRule.REDEMPTION_AT_FAIR_VALUE.value == "redemption_at_fair_value"
    assert SukukRule.PROCEEDS_USAGE_DISCLOSED.value == "proceeds_usage_disclosed"


def test_sukuk_type_string_values():
    assert SukukType.IJARA.value == "ijara"
    assert SukukType.MUDARABAH.value == "mudarabah"
    assert SukukType.MUSHARAKAH.value == "musharakah"
    assert SukukType.MURABAHA.value == "murabaha"
    assert SukukType.SALAM.value == "salam"
    assert SukukType.ISTISNA.value == "istisna"
    assert SukukType.WAKALAH.value == "wakalah"


def test_seven_sukuk_types_pinned():
    assert {t.value for t in SukukType} == {
        "ijara",
        "mudarabah",
        "musharakah",
        "murabaha",
        "salam",
        "istisna",
        "wakalah",
    }


def test_tradable_in_secondary_set_pin():
    assert TRADABLE_IN_SECONDARY == frozenset(
        {
            SukukType.IJARA,
            SukukType.MUDARABAH,
            SukukType.MUSHARAKAH,
            SukukType.WAKALAH,
            SukukType.ISTISNA,
        }
    )


def test_murabaha_not_tradable():
    assert is_tradable_in_secondary(SukukType.MURABAHA) is False


def test_salam_not_tradable():
    assert is_tradable_in_secondary(SukukType.SALAM) is False


def test_ijara_tradable():
    assert is_tradable_in_secondary(SukukType.IJARA) is True


def test_clauses_sorted():
    keys = [tuple(int(s) for s in c.clause_id.split(".")) for c in CLAUSES]
    assert keys == sorted(keys)


def test_clauses_unique_ids():
    ids = [c.clause_id for c in CLAUSES]
    assert len(ids) == len(set(ids))


def test_clause_by_id():
    c = clause_by_id("3.1")
    assert c is not None
    assert c.rule is SukukRule.NO_GUARANTEED_PRINCIPAL


def test_clause_by_id_unknown():
    assert clause_by_id("99.99") is None


def test_clauses_for_rule_returns_matching():
    matches = clauses_for_rule(SukukRule.OWNERSHIP_TRANSFER_REAL)
    assert len(matches) >= 1


def test_sukuk_clause_validation_bad_id():
    with pytest.raises(ValueError):
        SukukClause(
            clause_id="bad",
            title="x",
            rule=SukukRule.UNDERLYING_ASSET_HALAL,
            summary="x",
        )


def test_sukuk_clause_validation_empty_title():
    with pytest.raises(ValueError):
        SukukClause(
            clause_id="9.9",
            title=" ",
            rule=SukukRule.UNDERLYING_ASSET_HALAL,
            summary="x",
        )


def test_sukuk_clause_validation_empty_summary():
    with pytest.raises(ValueError):
        SukukClause(
            clause_id="9.9",
            title="x",
            rule=SukukRule.UNDERLYING_ASSET_HALAL,
            summary="",
        )


def _good_inputs(**overrides) -> SukukIssuanceInputs:
    base = {
        "issuer": "GovOfMalaysia",
        "sukuk_type": SukukType.IJARA,
        "underlying_purpose": "highway construction",
        "tangible_asset_ratio": 0.85,
        "proceeds_usage_disclosed": True,
        "sharia_board_opinion_published": True,
        "purpose_is_halal": True,
        "interest_rate_linked_payouts": False,
        "principal_guaranteed_by_issuer": False,
        "redemption_is_fair_value": True,
    }
    base.update(overrides)
    return SukukIssuanceInputs(**base)


def test_inputs_validation_empty_issuer():
    with pytest.raises(ValueError):
        _good_inputs(issuer="")


def test_inputs_validation_empty_purpose():
    with pytest.raises(ValueError):
        _good_inputs(underlying_purpose="  ")


def test_inputs_validation_tangibility_below_zero():
    with pytest.raises(ValueError):
        _good_inputs(tangible_asset_ratio=-0.1)


def test_inputs_validation_tangibility_above_one():
    with pytest.raises(ValueError):
        _good_inputs(tangible_asset_ratio=1.1)


def test_screen_clean_ijara_passes():
    a = screen_sukuk(_good_inputs())
    assert a.is_compliant
    assert a.violated_rules == frozenset()
    assert a.secondary_tradable is True


def test_screen_haram_purpose_blocked():
    a = screen_sukuk(_good_inputs(purpose_is_halal=False))
    assert not a.is_compliant
    assert SukukRule.UNDERLYING_ASSET_HALAL in a.violated_rules


def test_screen_no_disclosure_blocked():
    a = screen_sukuk(_good_inputs(proceeds_usage_disclosed=False))
    assert SukukRule.PROCEEDS_USAGE_DISCLOSED in a.violated_rules


def test_screen_principal_guarantee_blocked():
    a = screen_sukuk(_good_inputs(principal_guaranteed_by_issuer=True))
    assert SukukRule.NO_GUARANTEED_PRINCIPAL in a.violated_rules


def test_screen_interest_rate_link_blocked():
    a = screen_sukuk(_good_inputs(interest_rate_linked_payouts=True))
    assert SukukRule.NO_INTEREST_RATE_LINK in a.violated_rules


def test_screen_no_sharia_board_blocked():
    a = screen_sukuk(_good_inputs(sharia_board_opinion_published=False))
    assert SukukRule.SHARIA_BOARD_OPINION in a.violated_rules


def test_screen_face_value_redemption_blocked():
    a = screen_sukuk(_good_inputs(redemption_is_fair_value=False))
    assert SukukRule.REDEMPTION_AT_FAIR_VALUE in a.violated_rules


def test_screen_low_tangibility_not_secondary_tradable():
    a = screen_sukuk(_good_inputs(tangible_asset_ratio=0.40))
    # Still compliant if all other rules pass
    assert a.is_compliant
    assert a.secondary_tradable is False


def test_screen_at_threshold_tangibility_secondary_tradable():
    a = screen_sukuk(_good_inputs(tangible_asset_ratio=0.51))
    assert a.secondary_tradable is True


def test_screen_just_under_threshold_not_tradable():
    a = screen_sukuk(_good_inputs(tangible_asset_ratio=0.50))
    assert a.secondary_tradable is False


def test_screen_murabaha_never_secondary_tradable():
    a = screen_sukuk(_good_inputs(sukuk_type=SukukType.MURABAHA))
    assert a.secondary_tradable is False


def test_screen_salam_never_secondary_tradable():
    a = screen_sukuk(_good_inputs(sukuk_type=SukukType.SALAM))
    assert a.secondary_tradable is False


def test_assessment_structural_pin_compliant_with_violations():
    with pytest.raises(ValueError):
        SukukAssessment(
            issuer="x",
            sukuk_type=SukukType.IJARA,
            violated_rules=frozenset({SukukRule.UNDERLYING_ASSET_HALAL}),
            is_compliant=True,
            secondary_tradable=False,
        )


def test_assessment_structural_pin_non_compliant_without_violations():
    with pytest.raises(ValueError):
        SukukAssessment(
            issuer="x",
            sukuk_type=SukukType.IJARA,
            violated_rules=frozenset(),
            is_compliant=False,
            secondary_tradable=False,
        )


def test_render_assessment_clean():
    a = screen_sukuk(_good_inputs())
    out = render_assessment(a)
    assert "✅" in out
    assert "ijara" in out
    assert "tradable on secondary" in out


def test_render_assessment_violations():
    a = screen_sukuk(_good_inputs(purpose_is_halal=False, proceeds_usage_disclosed=False))
    out = render_assessment(a)
    assert "❌" in out
    assert "underlying_asset_halal" in out
    assert "proceeds_usage_disclosed" in out


def test_render_no_secret_leak():
    out = render_coverage_matrix()
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


def test_render_clause_format():
    c = clause_by_id("3.1")
    assert c is not None
    out = render_clause(c)
    assert "§3.1" in out
    assert "no_guaranteed_principal" in out


def test_render_coverage_matrix_default_engages_all():
    out = render_coverage_matrix()
    # Header reflects 100% engagement when None passed (defaults to all rules)
    assert f"{len(CLAUSES)}/{len(CLAUSES)}" in out


def test_render_coverage_matrix_partial():
    out = render_coverage_matrix({SukukRule.UNDERLYING_ASSET_HALAL})
    assert f"/{len(CLAUSES)}" in out


def test_e2e_classic_haram_brewery_blocked():
    a = screen_sukuk(
        _good_inputs(
            issuer="BreweryCo",
            underlying_purpose="brewery construction",
            purpose_is_halal=False,
        )
    )
    assert not a.is_compliant
    assert SukukRule.UNDERLYING_ASSET_HALAL in a.violated_rules


def test_e2e_pure_murabaha_compliant_but_primary_only():
    """Murabaha sukuk can be compliant under Standard 17 but not secondary-tradable."""
    a = screen_sukuk(_good_inputs(sukuk_type=SukukType.MURABAHA))
    assert a.is_compliant
    assert a.secondary_tradable is False


def test_replay_consistency():
    a = screen_sukuk(_good_inputs())
    b = screen_sukuk(_good_inputs())
    assert a == b
