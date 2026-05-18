"""Tests for halal/sukuk_onchain.py — Round-5 Wave 22.D."""

from __future__ import annotations

import pytest

from halal_trader.halal.aaoifi_standard_17 import SukukIssuanceInputs, SukukType
from halal_trader.halal.sukuk_onchain import (
    OnChainIssue,
    OnChainScreenPolicy,
    OnChainSukukInputs,
    render_assessment,
    screen_onchain,
)


def _structural(**overrides) -> SukukIssuanceInputs:
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


def _inputs(**overrides) -> OnChainSukukInputs:
    base = {
        "token_symbol": "MYIJ",
        "chain": "ethereum",
        "structural": _structural(),
        "contract_verified_on_explorer": True,
        "is_upgradeable_proxy": False,
        "has_third_party_audit": True,
        "shariah_board_signature_on_chain": True,
        "relies_on_external_oracle": False,
        "secondary_via_amm": False,
        "fallback_asset_is_halal": True,
    }
    base.update(overrides)
    return OnChainSukukInputs(**base)


# --- Validation -----------------------------------


def test_issue_string_values():
    assert OnChainIssue.UNVERIFIED_CONTRACT.value == "unverified_contract"
    assert OnChainIssue.UPGRADEABLE_PROXY_RISK.value == "upgradeable_proxy_risk"
    assert OnChainIssue.NO_AUDIT.value == "no_audit"
    assert OnChainIssue.SHARIAH_BOARD_OFFCHAIN_ONLY.value == "shariah_board_offchain_only"
    assert OnChainIssue.ORACLE_DEPENDENCY.value == "oracle_dependency"
    assert OnChainIssue.SECONDARY_TRADING_VIA_AMM.value == "secondary_trading_via_amm"
    assert OnChainIssue.NON_HALAL_FALLBACK_ASSET.value == "non_halal_fallback_asset"


def test_default_policy():
    p = OnChainScreenPolicy()
    assert p.require_audit is True


def test_inputs_empty_token_rejected():
    with pytest.raises(ValueError):
        _inputs(token_symbol="")


def test_inputs_empty_chain_rejected():
    with pytest.raises(ValueError):
        _inputs(chain=" ")


# --- Screening ----------------------------------


def test_clean_onchain_passes():
    a = screen_onchain(_inputs())
    assert a.is_compliant
    assert a.structural_compliant


def test_unverified_contract_blocked():
    a = screen_onchain(_inputs(contract_verified_on_explorer=False))
    assert OnChainIssue.UNVERIFIED_CONTRACT in a.on_chain_issues
    assert not a.is_compliant


def test_upgradeable_proxy_blocked():
    a = screen_onchain(_inputs(is_upgradeable_proxy=True))
    assert OnChainIssue.UPGRADEABLE_PROXY_RISK in a.on_chain_issues


def test_upgradeable_relaxed_passes():
    pol = OnChainScreenPolicy(require_upgradeable_pause=False)
    a = screen_onchain(_inputs(is_upgradeable_proxy=True), policy=pol)
    assert OnChainIssue.UPGRADEABLE_PROXY_RISK not in a.on_chain_issues


def test_no_audit_blocked():
    a = screen_onchain(_inputs(has_third_party_audit=False))
    assert OnChainIssue.NO_AUDIT in a.on_chain_issues


def test_no_audit_relaxed_passes():
    pol = OnChainScreenPolicy(require_audit=False)
    a = screen_onchain(_inputs(has_third_party_audit=False), policy=pol)
    assert OnChainIssue.NO_AUDIT not in a.on_chain_issues


def test_offchain_shariah_signature_flagged():
    a = screen_onchain(_inputs(shariah_board_signature_on_chain=False))
    assert OnChainIssue.SHARIAH_BOARD_OFFCHAIN_ONLY in a.on_chain_issues


def test_oracle_dependency_only_flagged_when_policy_enabled():
    """Default policy doesn't block on oracle alone."""
    a = screen_onchain(_inputs(relies_on_external_oracle=True))
    assert OnChainIssue.ORACLE_DEPENDENCY not in a.on_chain_issues


def test_oracle_dependency_blocked_with_strict_policy():
    pol = OnChainScreenPolicy(block_on_oracle_dependency=True)
    a = screen_onchain(_inputs(relies_on_external_oracle=True), policy=pol)
    assert OnChainIssue.ORACLE_DEPENDENCY in a.on_chain_issues


def test_amm_secondary_trading_for_murabaha_blocked():
    a = screen_onchain(
        _inputs(
            structural=_structural(sukuk_type=SukukType.MURABAHA),
            secondary_via_amm=True,
        )
    )
    assert OnChainIssue.SECONDARY_TRADING_VIA_AMM in a.on_chain_issues


def test_amm_for_ijara_no_special_flag():
    """Ijara is tradable on secondary; AMM is fine."""
    a = screen_onchain(
        _inputs(
            structural=_structural(sukuk_type=SukukType.IJARA),
            secondary_via_amm=True,
        )
    )
    assert OnChainIssue.SECONDARY_TRADING_VIA_AMM not in a.on_chain_issues


def test_non_halal_fallback_blocked():
    a = screen_onchain(_inputs(fallback_asset_is_halal=False))
    assert OnChainIssue.NON_HALAL_FALLBACK_ASSET in a.on_chain_issues


def test_structural_failure_propagates():
    """If the structural Standard-17 check fails, overall fails."""
    a = screen_onchain(_inputs(structural=_structural(purpose_is_halal=False)))
    assert not a.structural_compliant
    assert not a.is_compliant


# --- Render ----------------------------------


def test_render_clean():
    a = screen_onchain(_inputs())
    out = render_assessment(a)
    assert "✅" in out
    assert "MYIJ" in out


def test_render_invalid_lists_issues():
    a = screen_onchain(_inputs(contract_verified_on_explorer=False))
    out = render_assessment(a)
    assert "❌" in out
    assert "unverified_contract" in out


def test_render_no_secret_leak():
    a = screen_onchain(_inputs())
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


# --- E2E -----------------------------------


def test_e2e_clean_ijara_token_passes():
    """Clean Ijara-backed sukuk token on Ethereum with full audit + on-chain SSB sig."""
    a = screen_onchain(_inputs())
    assert a.is_compliant


def test_e2e_unverified_amm_murabaha_blocked():
    a = screen_onchain(
        _inputs(
            structural=_structural(sukuk_type=SukukType.MURABAHA),
            contract_verified_on_explorer=False,
            secondary_via_amm=True,
        )
    )
    assert not a.is_compliant
    assert OnChainIssue.UNVERIFIED_CONTRACT in a.on_chain_issues
    assert OnChainIssue.SECONDARY_TRADING_VIA_AMM in a.on_chain_issues


def test_replay_consistency():
    a = screen_onchain(_inputs())
    b = screen_onchain(_inputs())
    assert a == b
