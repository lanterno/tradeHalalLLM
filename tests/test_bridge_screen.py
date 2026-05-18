"""Tests for halal/bridge_screen.py — Round-5 Wave 22.G."""

from __future__ import annotations

import pytest

from halal_trader.halal.bridge_screen import (
    BridgeAssessment,
    BridgeInputs,
    BridgeIssue,
    BridgeMechanism,
    BridgePolicy,
    render_assessment,
    screen_bridge,
)


def _inputs(**overrides) -> BridgeInputs:
    base = {
        "bridge_name": "halal-bridge-v1",
        "source_chain": "ethereum",
        "target_chain": "polygon",
        "asset_symbol": "PAXG",
        "asset_is_halal": True,
        "mechanism": BridgeMechanism.LOCK_AND_MINT,
        "custodian_is_halal": True,
        "has_proof_of_reserves": True,
        "is_audited": True,
        "interest_bearing_collateral": False,
        "fractional_reserve_in_use": False,
    }
    base.update(overrides)
    return BridgeInputs(**base)


# --- Validation ---------------------------------


def test_mechanism_string_values():
    assert BridgeMechanism.LOCK_AND_MINT.value == "lock_and_mint"
    assert BridgeMechanism.BURN_AND_MINT.value == "burn_and_mint"
    assert BridgeMechanism.LIQUIDITY_POOL.value == "liquidity_pool"
    assert BridgeMechanism.ATOMIC_SWAP.value == "atomic_swap"


def test_issue_string_values():
    assert BridgeIssue.UNDERLYING_ASSET_NOT_HALAL.value == "underlying_asset_not_halal"
    assert BridgeIssue.INTEREST_BEARING_COLLATERAL.value == "interest_bearing_collateral"
    assert BridgeIssue.FRACTIONAL_RESERVE_RISK.value == "fractional_reserve_risk"
    assert BridgeIssue.NO_PROOF_OF_RESERVES.value == "no_proof_of_reserves"
    assert BridgeIssue.CUSTODIAN_NOT_HALAL.value == "custodian_not_halal"
    assert BridgeIssue.UNAUDITED_BRIDGE_CONTRACT.value == "unaudited_bridge_contract"


def test_inputs_empty_bridge_rejected():
    with pytest.raises(ValueError):
        _inputs(bridge_name="")


def test_inputs_same_chain_rejected():
    with pytest.raises(ValueError):
        _inputs(source_chain="ethereum", target_chain="ethereum")


def test_inputs_empty_asset_rejected():
    with pytest.raises(ValueError):
        _inputs(asset_symbol=" ")


def test_assessment_invariant_compliant_with_issues():
    with pytest.raises(ValueError):
        BridgeAssessment(
            bridge_name="b",
            asset_symbol="A",
            issues=frozenset({BridgeIssue.NO_PROOF_OF_RESERVES}),
            is_compliant=True,
        )


def test_assessment_invariant_noncompliant_without_issues():
    with pytest.raises(ValueError):
        BridgeAssessment(
            bridge_name="b",
            asset_symbol="A",
            issues=frozenset(),
            is_compliant=False,
        )


# --- Screening ----------------------------------


def test_clean_lock_and_mint_passes():
    a = screen_bridge(_inputs())
    assert a.is_compliant


def test_haram_asset_blocked():
    a = screen_bridge(_inputs(asset_is_halal=False))
    assert BridgeIssue.UNDERLYING_ASSET_NOT_HALAL in a.issues


def test_interest_bearing_collateral_blocked():
    a = screen_bridge(_inputs(interest_bearing_collateral=True))
    assert BridgeIssue.INTEREST_BEARING_COLLATERAL in a.issues


def test_fractional_reserve_blocked():
    a = screen_bridge(_inputs(fractional_reserve_in_use=True))
    assert BridgeIssue.FRACTIONAL_RESERVE_RISK in a.issues


def test_liquidity_pool_blocked_with_strict_policy():
    pol = BridgePolicy(block_liquidity_pool_models=True)
    a = screen_bridge(_inputs(mechanism=BridgeMechanism.LIQUIDITY_POOL), policy=pol)
    assert BridgeIssue.FRACTIONAL_RESERVE_RISK in a.issues


def test_liquidity_pool_default_policy_passes():
    """Default policy doesn't block liquidity-pool model alone."""
    a = screen_bridge(_inputs(mechanism=BridgeMechanism.LIQUIDITY_POOL))
    assert a.is_compliant


def test_no_proof_of_reserves_blocked():
    a = screen_bridge(_inputs(has_proof_of_reserves=False))
    assert BridgeIssue.NO_PROOF_OF_RESERVES in a.issues


def test_no_proof_relaxed_policy_passes():
    pol = BridgePolicy(require_proof_of_reserves=False)
    a = screen_bridge(_inputs(has_proof_of_reserves=False), policy=pol)
    assert BridgeIssue.NO_PROOF_OF_RESERVES not in a.issues


def test_non_halal_custodian_blocked():
    a = screen_bridge(_inputs(custodian_is_halal=False))
    assert BridgeIssue.CUSTODIAN_NOT_HALAL in a.issues


def test_unaudited_blocked():
    a = screen_bridge(_inputs(is_audited=False))
    assert BridgeIssue.UNAUDITED_BRIDGE_CONTRACT in a.issues


def test_unaudited_relaxed_policy_passes():
    pol = BridgePolicy(require_audit=False)
    a = screen_bridge(_inputs(is_audited=False), policy=pol)
    assert BridgeIssue.UNAUDITED_BRIDGE_CONTRACT not in a.issues


def test_atomic_swap_passes():
    a = screen_bridge(_inputs(mechanism=BridgeMechanism.ATOMIC_SWAP))
    assert a.is_compliant


def test_burn_and_mint_passes():
    a = screen_bridge(_inputs(mechanism=BridgeMechanism.BURN_AND_MINT))
    assert a.is_compliant


# --- Render ---------------------------------


def test_render_clean():
    a = screen_bridge(_inputs())
    out = render_assessment(a)
    assert "✅" in out
    assert "halal-bridge-v1" in out


def test_render_invalid_lists_issues():
    a = screen_bridge(_inputs(asset_is_halal=False))
    out = render_assessment(a)
    assert "❌" in out
    assert "underlying_asset_not_halal" in out


def test_render_no_secret_leak():
    a = screen_bridge(_inputs())
    out = render_assessment(a)
    for token in (
        "@",
        "zoom.us",
        "meet.google",
        "private_email",
        "+1-",
        "Authorization",
        "wallet_address",
        "private_key",
        "0x",
    ):
        assert token not in out


# --- E2E -------------------------------------


def test_e2e_paxg_eth_to_polygon_clean():
    a = screen_bridge(
        _inputs(
            asset_symbol="PAXG",
            mechanism=BridgeMechanism.LOCK_AND_MINT,
        )
    )
    assert a.is_compliant


def test_e2e_haram_token_with_unverified_bridge_blocked():
    a = screen_bridge(
        _inputs(
            asset_symbol="HARAM",
            asset_is_halal=False,
            is_audited=False,
            has_proof_of_reserves=False,
        )
    )
    assert not a.is_compliant
    assert BridgeIssue.UNDERLYING_ASSET_NOT_HALAL in a.issues
    assert BridgeIssue.UNAUDITED_BRIDGE_CONTRACT in a.issues
    assert BridgeIssue.NO_PROOF_OF_RESERVES in a.issues


def test_replay_consistency():
    a = screen_bridge(_inputs())
    b = screen_bridge(_inputs())
    assert a == b
