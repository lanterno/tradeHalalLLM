"""Tests for halal/vault_custodian.py — Round-5 Wave 5.A."""

from __future__ import annotations

import pytest

from halal_trader.halal.vault_custodian import (
    CustodianPolicy,
    CustodianTier,
    VaultInputs,
    VaultIssue,
    render_assessment,
    screen_custodian,
)


def _inputs(**overrides) -> VaultInputs:
    base = {
        "custodian_name": "BullionVault",
        "jurisdiction": "Switzerland",
        "metal": "gold",
        "has_independent_audit": True,
        "metal_lent_for_interest": False,
        "fully_physical_backed": True,
        "holdings_segregated_per_client": True,
        "permits_constructive_possession": True,
        "interest_bearing_cash_buffer": False,
        "aaoifi_certified": False,
    }
    base.update(overrides)
    return VaultInputs(**base)


# --- Validation ---------------------------------


def test_tier_string_values():
    assert CustodianTier.TIER_1.value == "tier_1"
    assert CustodianTier.TIER_2.value == "tier_2"
    assert CustodianTier.TIER_3.value == "tier_3"
    assert CustodianTier.REJECTED.value == "rejected"


def test_issue_string_values():
    assert VaultIssue.NO_AUDIT.value == "no_audit"
    assert VaultIssue.METAL_LENT_OUT.value == "metal_lent_out"
    assert VaultIssue.PAPER_BACKED_NOT_PHYSICAL.value == "paper_backed_not_physical"
    assert VaultIssue.NOT_SEGREGATED.value == "not_segregated"
    assert VaultIssue.NO_CONSTRUCTIVE_POSSESSION.value == "no_constructive_possession"
    assert VaultIssue.INTEREST_BEARING_CASH_BUFFER.value == "interest_bearing_cash_buffer"


def test_inputs_empty_name_rejected():
    with pytest.raises(ValueError):
        _inputs(custodian_name="")


def test_inputs_empty_jurisdiction_rejected():
    with pytest.raises(ValueError):
        _inputs(jurisdiction="")


def test_inputs_empty_metal_rejected():
    with pytest.raises(ValueError):
        _inputs(metal="")


# --- Tier laddering ---------------------------


def test_clean_custodian_tier_1():
    a = screen_custodian(_inputs())
    assert a.tier is CustodianTier.TIER_1


def test_one_minor_issue_tier_2():
    a = screen_custodian(_inputs(has_independent_audit=False))
    assert a.tier is CustodianTier.TIER_2


def test_metal_lent_out_immediately_rejected():
    """Metal-lending is the most severe issue — directly REJECTED."""
    a = screen_custodian(_inputs(metal_lent_for_interest=True))
    assert a.tier is CustodianTier.REJECTED


def test_paper_backed_directly_drops_to_tier_3_or_lower():
    a = screen_custodian(_inputs(fully_physical_backed=False))
    assert a.tier in (CustodianTier.TIER_3, CustodianTier.REJECTED)


def test_not_segregated_demoted():
    a = screen_custodian(_inputs(holdings_segregated_per_client=False))
    assert a.tier is CustodianTier.TIER_2


def test_multiple_issues_drop_through():
    a = screen_custodian(
        _inputs(
            has_independent_audit=False,
            holdings_segregated_per_client=False,
            permits_constructive_possession=False,
        )
    )
    assert a.tier is CustodianTier.TIER_3


def test_too_many_issues_rejected():
    a = screen_custodian(
        _inputs(
            has_independent_audit=False,
            metal_lent_for_interest=True,
            fully_physical_backed=False,
            holdings_segregated_per_client=False,
            permits_constructive_possession=False,
            interest_bearing_cash_buffer=True,
        )
    )
    assert a.tier is CustodianTier.REJECTED


def test_aaoifi_certification_required_for_tier_1_when_policy_set():
    pol = CustodianPolicy(require_aaoifi_certification=True)
    a = screen_custodian(_inputs(aaoifi_certified=False), policy=pol)
    # Without AAOIFI cert, can't be TIER_1 even if otherwise clean
    assert a.tier is not CustodianTier.TIER_1


def test_aaoifi_certified_with_strict_policy_tier_1():
    pol = CustodianPolicy(require_aaoifi_certification=True)
    a = screen_custodian(_inputs(aaoifi_certified=True), policy=pol)
    assert a.tier is CustodianTier.TIER_1


# --- Render -------------------------------


def test_render_tier_1_gold_emoji():
    a = screen_custodian(_inputs())
    out = render_assessment(a)
    assert "🥇" in out


def test_render_rejected():
    a = screen_custodian(_inputs(metal_lent_for_interest=True))
    out = render_assessment(a)
    assert "❌" in out


def test_render_no_secret_leak():
    a = screen_custodian(_inputs())
    out = render_assessment(a)
    for token in (
        "@",
        "zoom.us",
        "meet.google",
        "private_email",
        "+1-",
        "Authorization",
        "vault_address",
        "serial_number",
    ):
        assert token not in out


# --- E2E -----------------------------


def test_e2e_clean_swiss_vault_passes_tier_1():
    a = screen_custodian(
        VaultInputs(
            custodian_name="BullionVault",
            jurisdiction="Switzerland",
            metal="gold",
            has_independent_audit=True,
            metal_lent_for_interest=False,
            fully_physical_backed=True,
            holdings_segregated_per_client=True,
            permits_constructive_possession=True,
            interest_bearing_cash_buffer=False,
            aaoifi_certified=True,
        )
    )
    assert a.tier is CustodianTier.TIER_1


def test_e2e_paper_gold_with_lending_rejected():
    a = screen_custodian(
        _inputs(metal_lent_for_interest=True, fully_physical_backed=False)
    )
    assert a.tier is CustodianTier.REJECTED


def test_replay_consistency():
    a = screen_custodian(_inputs())
    b = screen_custodian(_inputs())
    assert a == b
