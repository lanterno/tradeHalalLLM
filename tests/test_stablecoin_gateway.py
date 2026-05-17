"""Tests for halal/stablecoin_gateway.py — Round-5 Wave 22.A."""

from __future__ import annotations

import pytest

from halal_trader.halal.stablecoin_gateway import (
    BackingType,
    GatewayDecision,
    StablecoinInputs,
    StablecoinPolicy,
    render_assessment,
    screen,
)


def _inputs(**overrides) -> StablecoinInputs:
    base = {
        "coin_symbol": "PAXG",
        "backing_type": BackingType.GOLD,
        "issuer": "Paxos",
        "has_third_party_attestation": True,
        "reserves_segregated": True,
        "issuer_earns_riba_on_cash": False,
    }
    base.update(overrides)
    return StablecoinInputs(**base)


# --- Validation -----------------------------------------------


def test_backing_type_string_values():
    assert BackingType.GOLD.value == "gold"
    assert BackingType.USD_TBILL.value == "usd_tbill"
    assert BackingType.ALGORITHMIC.value == "algorithmic"
    assert BackingType.SALAM_BASED.value == "salam_based"


def test_decision_string_values():
    assert GatewayDecision.APPROVED.value == "approved"
    assert GatewayDecision.TRANSACTIONAL_ONLY.value == "transactional_only"
    assert GatewayDecision.BLOCKED.value == "blocked"


def test_default_policy():
    p = StablecoinPolicy()
    assert p.allow_transactional_use is True
    assert p.max_transactional_hold_hours == 24.0


def test_policy_zero_hold_rejected():
    with pytest.raises(ValueError):
        StablecoinPolicy(max_transactional_hold_hours=0.0)


def test_inputs_empty_symbol_rejected():
    with pytest.raises(ValueError):
        _inputs(coin_symbol="")


def test_inputs_empty_issuer_rejected():
    with pytest.raises(ValueError):
        _inputs(issuer=" ")


# --- Screening ---------------------------------------------


def test_clean_gold_approved():
    a = screen(_inputs())
    assert a.decision is GatewayDecision.APPROVED


def test_clean_silver_approved():
    a = screen(_inputs(backing_type=BackingType.SILVER, coin_symbol="XAGT"))
    assert a.decision is GatewayDecision.APPROVED


def test_salam_based_approved():
    a = screen(_inputs(backing_type=BackingType.SALAM_BASED, coin_symbol="HALAL"))
    assert a.decision is GatewayDecision.APPROVED


def test_usd_tbill_default_transactional():
    a = screen(_inputs(backing_type=BackingType.USD_TBILL, coin_symbol="USDC"))
    assert a.decision is GatewayDecision.TRANSACTIONAL_ONLY


def test_usd_tbill_strict_policy_blocks():
    a = screen(
        _inputs(backing_type=BackingType.USD_TBILL, coin_symbol="USDC"),
        policy=StablecoinPolicy(allow_transactional_use=False),
    )
    assert a.decision is GatewayDecision.BLOCKED


def test_algorithmic_blocked():
    a = screen(_inputs(backing_type=BackingType.ALGORITHMIC, coin_symbol="UST"))
    assert a.decision is GatewayDecision.BLOCKED


def test_crypto_collateral_blocked():
    a = screen(_inputs(backing_type=BackingType.CRYPTO_COLLATERAL, coin_symbol="DAI"))
    assert a.decision is GatewayDecision.BLOCKED


def test_no_attestation_blocked():
    a = screen(_inputs(has_third_party_attestation=False))
    assert a.decision is GatewayDecision.BLOCKED


def test_no_attestation_relaxed_passes():
    a = screen(
        _inputs(has_third_party_attestation=False),
        policy=StablecoinPolicy(require_attestation=False),
    )
    assert a.decision is GatewayDecision.APPROVED


def test_commingled_reserves_blocked():
    a = screen(_inputs(reserves_segregated=False))
    assert a.decision is GatewayDecision.BLOCKED


def test_commingled_relaxed_passes():
    a = screen(
        _inputs(reserves_segregated=False),
        policy=StablecoinPolicy(require_segregated_reserves=False),
    )
    assert a.decision is GatewayDecision.APPROVED


def test_riba_on_cash_blocked():
    a = screen(_inputs(issuer_earns_riba_on_cash=True))
    assert a.decision is GatewayDecision.BLOCKED


def test_reasons_populated():
    a = screen(_inputs())
    assert len(a.reasons) >= 1


def test_reasons_for_blocked_describe_issue():
    a = screen(_inputs(backing_type=BackingType.ALGORITHMIC))
    assert any("gharar" in r.lower() for r in a.reasons)


# --- Render ----------------------------------------------


def test_render_approved():
    a = screen(_inputs())
    out = render_assessment(a)
    assert "✅" in out
    assert "PAXG" in out


def test_render_transactional_yellow():
    a = screen(_inputs(backing_type=BackingType.USD_TBILL, coin_symbol="USDC"))
    out = render_assessment(a)
    assert "🟡" in out


def test_render_blocked_red():
    a = screen(_inputs(backing_type=BackingType.ALGORITHMIC, coin_symbol="UST"))
    out = render_assessment(a)
    assert "❌" in out


def test_render_no_secret_leak():
    a = screen(_inputs())
    out = render_assessment(a)
    for token in (
        "@",
        "zoom.us",
        "meet.google",
        "private_email",
        "+1-",
        "Authorization",
        "wallet_address",
    ):
        assert token not in out


# --- E2E ------------------------------------------------


def test_e2e_paxg_passes_clean():
    """Real-world: Paxos PAXG (gold-backed, attested, segregated) → APPROVED."""
    a = screen(
        StablecoinInputs(
            coin_symbol="PAXG",
            backing_type=BackingType.GOLD,
            issuer="Paxos",
            has_third_party_attestation=True,
            reserves_segregated=True,
            issuer_earns_riba_on_cash=False,
        )
    )
    assert a.decision is GatewayDecision.APPROVED


def test_e2e_usdt_transactional_with_purification():
    """Tether (USD-T-bill) → transactional-only, riba purification required."""
    a = screen(
        StablecoinInputs(
            coin_symbol="USDT",
            backing_type=BackingType.USD_TBILL,
            issuer="Tether",
            has_third_party_attestation=True,
            reserves_segregated=True,
            issuer_earns_riba_on_cash=True,
        )
    )
    assert a.decision is GatewayDecision.TRANSACTIONAL_ONLY


def test_replay_consistency():
    inp = _inputs()
    a = screen(inp)
    b = screen(inp)
    assert a == b
