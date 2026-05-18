"""Tests for halal/wakalah.py — Round-5 Wave 7.C."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.wakalah import (
    FeeStructure,
    WakalahContract,
    WakalahStatus,
    advance_status,
    calculate_fee,
    render_contract,
    render_settlement,
    settle,
)


def _contract(**overrides) -> WakalahContract:
    base = {
        "contract_id": "W-001",
        "principal_handle": "principal-1",
        "agent_handle": "agent-1",
        "capital_amount": 100000.0,
        "currency": "USD",
        "fee_structure": FeeStructure.FIXED_PCT_OF_CAPITAL,
        "fee_value": 0.01,  # 1% of capital
        "start_date": date(2026, 1, 1),
        "expected_end_date": date(2026, 12, 31),
        "status": WakalahStatus.DRAFT,
    }
    base.update(overrides)
    return WakalahContract(**base)


# --- Validation ----------------------------------


def test_status_string_values():
    assert WakalahStatus.DRAFT.value == "draft"
    assert WakalahStatus.ACTIVE.value == "active"
    assert WakalahStatus.SETTLING.value == "settling"
    assert WakalahStatus.CLOSED.value == "closed"


def test_fee_structure_string_values():
    assert FeeStructure.FIXED_AMOUNT.value == "fixed_amount"
    assert FeeStructure.FIXED_PCT_OF_CAPITAL.value == "fixed_pct_of_capital"


def test_empty_id_rejected():
    with pytest.raises(ValueError):
        _contract(contract_id="")


def test_email_handle_rejected():
    with pytest.raises(ValueError):
        _contract(principal_handle="ops@example.com")


def test_same_party_rejected():
    with pytest.raises(ValueError):
        _contract(principal_handle="x", agent_handle="x")


def test_zero_capital_rejected():
    with pytest.raises(ValueError):
        _contract(capital_amount=0)


def test_negative_fee_rejected():
    with pytest.raises(ValueError):
        _contract(fee_value=-1)


def test_pct_above_one_rejected():
    with pytest.raises(ValueError):
        _contract(
            fee_structure=FeeStructure.FIXED_PCT_OF_CAPITAL,
            fee_value=1.5,
        )


def test_end_before_start_rejected():
    with pytest.raises(ValueError):
        _contract(start_date=date(2026, 12, 31), expected_end_date=date(2026, 1, 1))


# --- Fee calculation ------------------------------


def test_fee_fixed_amount():
    c = _contract(fee_structure=FeeStructure.FIXED_AMOUNT, fee_value=500.0)
    assert calculate_fee(c) == 500.0


def test_fee_pct_of_capital():
    c = _contract(
        fee_structure=FeeStructure.FIXED_PCT_OF_CAPITAL, fee_value=0.02, capital_amount=50000
    )
    assert calculate_fee(c) == 1000.0


# --- Lifecycle -------------------------------------


def test_advance_draft_to_active():
    c = _contract()
    new = advance_status(c, WakalahStatus.ACTIVE)
    assert new.status is WakalahStatus.ACTIVE


def test_advance_invalid_transition_rejected():
    c = _contract(status=WakalahStatus.DRAFT)
    with pytest.raises(ValueError):
        advance_status(c, WakalahStatus.CLOSED)


# --- Settlement -----------------------------------


def test_settle_only_active_or_settling():
    c = _contract(status=WakalahStatus.DRAFT)
    with pytest.raises(ValueError):
        settle(c, final_capital_value=110000)


def test_settle_profit_principal_gets_net():
    c = _contract(status=WakalahStatus.ACTIVE)
    # Capital $100k, +$10k profit → final $110k. Agent fee 1% = $1000.
    s = settle(c, final_capital_value=110000)
    assert s.agent_fee == 1000
    assert s.principal_net == 109000


def test_settle_loss_agent_still_takes_fee():
    """Standard wakalah: agent gets fee even on loss (unless negligent)."""
    c = _contract(status=WakalahStatus.ACTIVE)
    s = settle(c, final_capital_value=90000)  # -$10k loss
    assert s.is_loss
    assert s.agent_fee == 1000  # still takes fee
    assert s.principal_net == 89000


def test_settle_negligent_agent_no_fee():
    c = _contract(status=WakalahStatus.ACTIVE)
    s = settle(c, final_capital_value=90000, agent_negligent=True)
    assert s.agent_fee == 0
    assert s.principal_net == 90000


def test_settle_fee_exceeds_capital_capped():
    """Defensive: fee can't exceed total final value."""
    c = _contract(
        status=WakalahStatus.ACTIVE,
        fee_structure=FeeStructure.FIXED_AMOUNT,
        fee_value=200000.0,  # huge fee
    )
    s = settle(c, final_capital_value=50000)  # less than fee
    assert s.agent_fee == 50000
    assert s.principal_net == 0


def test_settle_negative_final_value_rejected():
    c = _contract(status=WakalahStatus.ACTIVE)
    with pytest.raises(ValueError):
        settle(c, final_capital_value=-1.0)


# --- Render ----------------------------------


def test_render_contract_pct_fee():
    c = _contract(fee_value=0.015)
    out = render_contract(c)
    assert "Wakalah" in out
    assert "1.50%" in out


def test_render_contract_fixed_fee():
    c = _contract(fee_structure=FeeStructure.FIXED_AMOUNT, fee_value=500.0)
    out = render_contract(c)
    assert "fixed" in out


def test_render_settlement_profit():
    c = _contract(status=WakalahStatus.ACTIVE)
    s = settle(c, final_capital_value=110000)
    out = render_settlement(s)
    assert "profit" in out


def test_render_settlement_negligent():
    c = _contract(status=WakalahStatus.ACTIVE)
    s = settle(c, final_capital_value=90000, agent_negligent=True)
    out = render_settlement(s)
    assert "NEGLIGENT" in out


# --- E2E -----------------------------


def test_e2e_full_lifecycle_with_fee():
    c = _contract()
    c = advance_status(c, WakalahStatus.ACTIVE)
    s = settle(c, final_capital_value=120000)
    # Agent gets fixed 1% of capital = $1000; principal gets rest = $119000
    assert s.agent_fee == 1000
    assert s.principal_net == 119000


def test_replay_consistency():
    c = _contract(status=WakalahStatus.ACTIVE)
    a = settle(c, final_capital_value=110000)
    b = settle(c, final_capital_value=110000)
    assert a == b
