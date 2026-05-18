"""Tests for halal/salam_istisna.py — Round-5 Wave 1.L."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.salam_istisna import (
    FUNGIBLE_ASSET_CLASSES,
    AssetClass,
    ContractInputs,
    ContractIssue,
    ContractKind,
    StructuringPolicy,
    StructuringResult,
    render_contract,
    structure_contract,
)


def _salam_inputs(**overrides) -> ContractInputs:
    base = {
        "contract_id": "SALAM-001",
        "kind": ContractKind.SALAM,
        "asset_class": AssetClass.AGRICULTURAL_COMMODITY,
        "description": "Hard red winter wheat, US grade No. 2, 12% protein.",
        "quantity": 5000.0,
        "quantity_unit": "bushels",
        "delivery_location": "Kansas City, MO, USA",
        "contracted_price": 30000.0,
        "prepayment_amount": 30000.0,
        "contract_date": date(2026, 5, 1),
        "delivery_date": date(2026, 11, 1),
    }
    base.update(overrides)
    return ContractInputs(**base)


def _istisna_inputs(**overrides) -> ContractInputs:
    base = {
        "contract_id": "ISTISNA-001",
        "kind": ContractKind.ISTISNA,
        "asset_class": AssetClass.MANUFACTURED_GOOD,
        "description": "Custom industrial CNC mill model X-7 with halal-spec fluids.",
        "quantity": 1.0,
        "quantity_unit": "unit",
        "delivery_location": "Riyadh, KSA",
        "contracted_price": 500000.0,
        "prepayment_amount": 100000.0,  # partial prepayment OK
        "contract_date": date(2026, 5, 1),
        "delivery_date": date(2027, 5, 1),
    }
    base.update(overrides)
    return ContractInputs(**base)


# --- Enum string-value pins --------------------------------------------------


def test_contract_kind_string_values():
    assert ContractKind.SALAM.value == "salam"
    assert ContractKind.ISTISNA.value == "istisna"


def test_asset_class_string_values():
    assert AssetClass.AGRICULTURAL_COMMODITY.value == "agricultural_commodity"
    assert AssetClass.PRECIOUS_METAL.value == "precious_metal"
    assert AssetClass.INDUSTRIAL_METAL.value == "industrial_metal"
    assert AssetClass.CURRENCY.value == "currency"
    assert AssetClass.ENERGY_FUNGIBLE.value == "energy_fungible"
    assert AssetClass.MANUFACTURED_GOOD.value == "manufactured_good"
    assert AssetClass.CONSTRUCTED_PROPERTY.value == "constructed_property"


def test_contract_issue_string_values():
    assert ContractIssue.NON_FUNGIBLE_FOR_SALAM.value == "non_fungible_for_salam"
    assert ContractIssue.INCOMPLETE_PREPAYMENT_FOR_SALAM.value == "incomplete_prepayment_for_salam"
    assert ContractIssue.DELIVERY_NOT_IN_FUTURE.value == "delivery_not_in_future"
    assert ContractIssue.DELIVERY_TOO_FAR.value == "delivery_too_far"
    assert ContractIssue.QUANTITY_NON_POSITIVE.value == "quantity_non_positive"
    assert ContractIssue.PRICE_NON_POSITIVE.value == "price_non_positive"
    assert ContractIssue.DESCRIPTION_TOO_VAGUE.value == "description_too_vague"
    assert ContractIssue.DELIVERY_LOCATION_MISSING.value == "delivery_location_missing"


def test_fungible_set_pin():
    assert FUNGIBLE_ASSET_CLASSES == frozenset(
        {
            AssetClass.AGRICULTURAL_COMMODITY,
            AssetClass.PRECIOUS_METAL,
            AssetClass.INDUSTRIAL_METAL,
            AssetClass.CURRENCY,
            AssetClass.ENERGY_FUNGIBLE,
        }
    )
    assert AssetClass.MANUFACTURED_GOOD not in FUNGIBLE_ASSET_CLASSES
    assert AssetClass.CONSTRUCTED_PROPERTY not in FUNGIBLE_ASSET_CLASSES


# --- Policy validation ------------------------------------------------------


def test_default_policy_loads():
    p = StructuringPolicy()
    assert p.salam_max_term_days == 365


def test_policy_zero_salam_term_rejected():
    with pytest.raises(ValueError):
        StructuringPolicy(salam_max_term_days=0)


def test_policy_salam_geq_istisna_rejected():
    with pytest.raises(ValueError):
        StructuringPolicy(salam_max_term_days=2000, istisna_max_term_days=2000)


def test_policy_zero_min_description_rejected():
    with pytest.raises(ValueError):
        StructuringPolicy(min_description_chars=0)


def test_policy_high_tolerance_rejected():
    with pytest.raises(ValueError):
        StructuringPolicy(salam_prepayment_tolerance=0.5)


def test_policy_immutable():
    p = StructuringPolicy()
    with pytest.raises(AttributeError):
        p.salam_max_term_days = 10  # type: ignore[misc]


# --- Inputs validation ------------------------------------------------------


def test_inputs_empty_contract_id_rejected():
    with pytest.raises(ValueError):
        _salam_inputs(contract_id="")


def test_inputs_empty_quantity_unit_rejected():
    with pytest.raises(ValueError):
        _salam_inputs(quantity_unit=" ")


# --- Salam validation -------------------------------------------------------


def test_clean_salam_passes():
    r = structure_contract(_salam_inputs())
    assert r.is_valid
    assert r.issues == frozenset()


def test_salam_with_non_fungible_asset_blocked():
    r = structure_contract(_salam_inputs(asset_class=AssetClass.MANUFACTURED_GOOD))
    assert ContractIssue.NON_FUNGIBLE_FOR_SALAM in r.issues
    assert not r.is_valid


def test_salam_with_partial_prepayment_blocked():
    r = structure_contract(_salam_inputs(prepayment_amount=15000.0, contracted_price=30000.0))
    assert ContractIssue.INCOMPLETE_PREPAYMENT_FOR_SALAM in r.issues


def test_salam_at_tolerance_passes():
    """Within 0.1% tolerance the prepayment is treated as complete."""
    r = structure_contract(_salam_inputs(prepayment_amount=29980.0, contracted_price=30000.0))
    assert ContractIssue.INCOMPLETE_PREPAYMENT_FOR_SALAM not in r.issues


def test_salam_just_under_tolerance_blocked():
    r = structure_contract(_salam_inputs(prepayment_amount=29900.0, contracted_price=30000.0))
    assert ContractIssue.INCOMPLETE_PREPAYMENT_FOR_SALAM in r.issues


def test_salam_term_at_max_passes():
    r = structure_contract(
        _salam_inputs(
            contract_date=date(2026, 1, 1),
            delivery_date=date(2026, 1, 1) + (date(2027, 1, 1) - date(2026, 1, 1)),
        )
    )
    # 365 days exactly — within max
    assert ContractIssue.DELIVERY_TOO_FAR not in r.issues


def test_salam_term_too_long_blocked():
    r = structure_contract(
        _salam_inputs(
            contract_date=date(2026, 1, 1),
            delivery_date=date(2027, 6, 1),
        )
    )
    assert ContractIssue.DELIVERY_TOO_FAR in r.issues


def test_salam_delivery_in_past_blocked():
    r = structure_contract(
        _salam_inputs(contract_date=date(2026, 5, 1), delivery_date=date(2026, 4, 1))
    )
    assert ContractIssue.DELIVERY_NOT_IN_FUTURE in r.issues


def test_salam_same_day_delivery_blocked():
    r = structure_contract(
        _salam_inputs(contract_date=date(2026, 5, 1), delivery_date=date(2026, 5, 1))
    )
    assert ContractIssue.DELIVERY_NOT_IN_FUTURE in r.issues


def test_salam_zero_quantity_blocked():
    r = structure_contract(_salam_inputs(quantity=0.0))
    assert ContractIssue.QUANTITY_NON_POSITIVE in r.issues


def test_salam_negative_price_blocked():
    r = structure_contract(_salam_inputs(contracted_price=-1.0))
    assert ContractIssue.PRICE_NON_POSITIVE in r.issues


def test_salam_short_description_blocked():
    r = structure_contract(_salam_inputs(description="wheat"))
    assert ContractIssue.DESCRIPTION_TOO_VAGUE in r.issues


def test_salam_missing_location_blocked():
    r = structure_contract(_salam_inputs(delivery_location=" "))
    assert ContractIssue.DELIVERY_LOCATION_MISSING in r.issues


# --- Istisna validation -----------------------------------------------------


def test_clean_istisna_passes_with_partial_prepayment():
    r = structure_contract(_istisna_inputs())
    assert r.is_valid


def test_istisna_no_prepayment_required():
    r = structure_contract(_istisna_inputs(prepayment_amount=0.0))
    assert ContractIssue.INCOMPLETE_PREPAYMENT_FOR_SALAM not in r.issues
    assert r.is_valid


def test_istisna_allows_non_fungible():
    r = structure_contract(_istisna_inputs(asset_class=AssetClass.CONSTRUCTED_PROPERTY))
    assert ContractIssue.NON_FUNGIBLE_FOR_SALAM not in r.issues
    assert r.is_valid


def test_istisna_long_term_within_5_years_passes():
    r = structure_contract(
        _istisna_inputs(
            contract_date=date(2026, 1, 1),
            delivery_date=date(2030, 12, 31),
        )
    )
    assert ContractIssue.DELIVERY_TOO_FAR not in r.issues


def test_istisna_term_over_5_years_blocked():
    r = structure_contract(
        _istisna_inputs(
            contract_date=date(2026, 1, 1),
            delivery_date=date(2032, 1, 1),
        )
    )
    assert ContractIssue.DELIVERY_TOO_FAR in r.issues


# --- Result invariants ------------------------------------------------------


def test_result_invalid_pin_with_no_issues_rejected():
    with pytest.raises(ValueError):
        StructuringResult(contract_id="x", issues=frozenset(), is_valid=False)


def test_result_valid_pin_with_issues_rejected():
    with pytest.raises(ValueError):
        StructuringResult(
            contract_id="x",
            issues=frozenset({ContractIssue.QUANTITY_NON_POSITIVE}),
            is_valid=True,
        )


# --- Render -----------------------------------------------------------------


def test_render_clean_salam():
    inp = _salam_inputs()
    r = structure_contract(inp)
    out = render_contract(inp, r)
    assert "✅" in out
    assert "salam" in out
    assert "agricultural_commodity" in out
    assert "Kansas City" in out


def test_render_invalid_lists_issues():
    inp = _salam_inputs(asset_class=AssetClass.MANUFACTURED_GOOD)
    r = structure_contract(inp)
    out = render_contract(inp, r)
    assert "❌" in out
    assert "non_fungible_for_salam" in out


def test_render_no_secret_leak():
    inp = _salam_inputs()
    r = structure_contract(inp)
    out = render_contract(inp, r)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E --------------------------------------------------------------------


def test_e2e_salam_short_wheat_view():
    """Operator with bearish wheat view sells forward 5000 bu via Salam."""
    inp = _salam_inputs()
    r = structure_contract(inp)
    assert r.is_valid


def test_e2e_istisna_constructed_property_view():
    """Operator with bearish view on construction sector commissions Istisna."""
    inp = _istisna_inputs(asset_class=AssetClass.CONSTRUCTED_PROPERTY)
    r = structure_contract(inp)
    assert r.is_valid


def test_replay_consistency():
    inp = _salam_inputs()
    assert structure_contract(inp) == structure_contract(inp)
