"""Tests for halal/climate_finance.py — Round-5 Wave 5.E."""

from __future__ import annotations

import pytest

from halal_trader.halal.climate_finance import (
    ClimateInputs,
    ClimateInstrument,
    ClimateIssue,
    ClimatePolicy,
    render_assessment,
    screen_instrument,
)


def _inputs(**overrides) -> ClimateInputs:
    base = {
        "instrument_id": "CARBON-001",
        "instrument": ClimateInstrument.CARBON_CREDIT_VERRA,
        "has_audited_provenance": True,
        "speculative_offset_proportion": 0.10,
        "has_interest_tranches": False,
        "underlying_includes_haram": False,
        "has_shariah_opinion": True,
    }
    base.update(overrides)
    return ClimateInputs(**base)


# --- Validation -----------------------


def test_instrument_string_values():
    assert ClimateInstrument.GREEN_SUKUK.value == "green_sukuk"
    assert ClimateInstrument.CARBON_CREDIT_VERRA.value == "carbon_credit_verra"
    assert ClimateInstrument.CARBON_CREDIT_GOLD_STANDARD.value == "carbon_credit_gold_standard"


def test_issue_string_values():
    assert ClimateIssue.UNVERIFIED_PROVENANCE.value == "unverified_provenance"
    assert ClimateIssue.SPECULATIVE_OFFSET.value == "speculative_offset"
    assert ClimateIssue.INTEREST_BEARING_TRANCHES.value == "interest_bearing_tranches"


def test_inputs_empty_id_rejected():
    with pytest.raises(ValueError):
        _inputs(instrument_id="")


def test_inputs_negative_speculative_rejected():
    with pytest.raises(ValueError):
        _inputs(speculative_offset_proportion=-0.1)


def test_inputs_above_one_speculative_rejected():
    with pytest.raises(ValueError):
        _inputs(speculative_offset_proportion=1.5)


# --- Screen ------------------------


def test_clean_verra_credit_passes():
    a = screen_instrument(_inputs())
    assert a.is_compliant


def test_unverified_provenance_blocked():
    a = screen_instrument(_inputs(has_audited_provenance=False))
    assert ClimateIssue.UNVERIFIED_PROVENANCE in a.issues


def test_speculative_offset_blocked_above_threshold():
    a = screen_instrument(_inputs(speculative_offset_proportion=0.50))
    assert ClimateIssue.SPECULATIVE_OFFSET in a.issues


def test_speculative_at_threshold_passes():
    a = screen_instrument(_inputs(speculative_offset_proportion=0.30))
    assert ClimateIssue.SPECULATIVE_OFFSET not in a.issues


def test_interest_tranches_blocked():
    a = screen_instrument(_inputs(has_interest_tranches=True))
    assert ClimateIssue.INTEREST_BEARING_TRANCHES in a.issues


def test_haram_underlying_blocked():
    a = screen_instrument(_inputs(underlying_includes_haram=True))
    assert ClimateIssue.UNDERLYING_INCLUDES_HARAM in a.issues


def test_carbon_credits_rejected_when_policy_disables():
    pol = ClimatePolicy(accept_carbon_credits=False)
    a = screen_instrument(_inputs(), policy=pol)
    assert not a.is_compliant


def test_renewable_equity_passes():
    a = screen_instrument(
        _inputs(
            instrument=ClimateInstrument.RENEWABLE_PROJECT_EQUITY,
            has_shariah_opinion=False,  # not required for non-sukuk
        )
    )
    assert a.is_compliant


def test_green_sukuk_no_shariah_opinion_blocked():
    a = screen_instrument(
        _inputs(instrument=ClimateInstrument.GREEN_SUKUK, has_shariah_opinion=False)
    )
    assert ClimateIssue.NO_SHARIAH_OPINION in a.issues


def test_green_sukuk_with_shariah_opinion_passes():
    a = screen_instrument(_inputs(instrument=ClimateInstrument.GREEN_SUKUK))
    assert a.is_compliant


def test_other_carbon_unverified_flagged_gharar():
    a = screen_instrument(
        _inputs(
            instrument=ClimateInstrument.CARBON_CREDIT_OTHER,
            has_audited_provenance=False,
        )
    )
    assert ClimateIssue.GHARAR_FUTURE_DELIVERY in a.issues


# --- Render --------------------------


def test_render_clean():
    a = screen_instrument(_inputs())
    out = render_assessment(a)
    assert "✅" in out
    assert "CARBON-001" in out


def test_render_violations():
    a = screen_instrument(_inputs(has_interest_tranches=True))
    out = render_assessment(a)
    assert "❌" in out
    assert "interest_bearing_tranches" in out


def test_render_no_secret_leak():
    a = screen_instrument(_inputs())
    out = render_assessment(a)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ----------------------------


def test_e2e_clean_green_sukuk_renewable_passes():
    """Green sukuk financing solar farm with full Shariah opinion → clean."""
    a = screen_instrument(
        _inputs(
            instrument_id="GS-2026-01",
            instrument=ClimateInstrument.GREEN_SUKUK,
            has_audited_provenance=True,
            speculative_offset_proportion=0.0,
            has_interest_tranches=False,
            underlying_includes_haram=False,
            has_shariah_opinion=True,
        )
    )
    assert a.is_compliant


def test_replay_consistency():
    a = screen_instrument(_inputs())
    b = screen_instrument(_inputs())
    assert a == b
