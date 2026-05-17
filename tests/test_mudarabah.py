"""Tests for halal/mudarabah.py — Round-5 Wave 7.A."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.mudarabah import (
    MudarabahContract,
    MudarabahStatus,
    Settlement,
    advance_status,
    render_contract,
    render_settlement,
    settle,
)


def _contract(**overrides) -> MudarabahContract:
    base = {
        "contract_id": "M-001",
        "rabb_handle": "investor-1",
        "mudarib_handle": "manager-1",
        "capital_amount": 100000.0,
        "currency": "USD",
        "rabb_profit_share": 0.70,
        "mudarib_profit_share": 0.30,
        "start_date": date(2026, 1, 1),
        "expected_end_date": date(2026, 12, 31),
        "status": MudarabahStatus.DRAFT,
    }
    base.update(overrides)
    return MudarabahContract(**base)


# --- Validation -----------------------------------


def test_status_string_values():
    assert MudarabahStatus.DRAFT.value == "draft"
    assert MudarabahStatus.ACTIVE.value == "active"
    assert MudarabahStatus.SETTLING.value == "settling"
    assert MudarabahStatus.CLOSED.value == "closed"


def test_empty_id_rejected():
    with pytest.raises(ValueError):
        _contract(contract_id="")


def test_email_handle_rejected():
    with pytest.raises(ValueError):
        _contract(rabb_handle="ops@example.com")


def test_same_party_rejected():
    with pytest.raises(ValueError):
        _contract(rabb_handle="op-1", mudarib_handle="op-1")


def test_zero_capital_rejected():
    with pytest.raises(ValueError):
        _contract(capital_amount=0)


def test_unbalanced_split_rejected():
    with pytest.raises(ValueError):
        _contract(rabb_profit_share=0.50, mudarib_profit_share=0.40)


def test_zero_share_rejected():
    with pytest.raises(ValueError):
        _contract(rabb_profit_share=0.0, mudarib_profit_share=1.0)


def test_full_share_rejected():
    with pytest.raises(ValueError):
        _contract(rabb_profit_share=1.0, mudarib_profit_share=0.0)


def test_end_before_start_rejected():
    with pytest.raises(ValueError):
        _contract(start_date=date(2026, 12, 31), expected_end_date=date(2026, 1, 1))


# --- Lifecycle ------------------------------------


def test_advance_draft_to_active():
    c = _contract(status=MudarabahStatus.DRAFT)
    new = advance_status(c, MudarabahStatus.ACTIVE)
    assert new.status is MudarabahStatus.ACTIVE


def test_advance_active_to_settling():
    c = _contract(status=MudarabahStatus.ACTIVE)
    new = advance_status(c, MudarabahStatus.SETTLING)
    assert new.status is MudarabahStatus.SETTLING


def test_advance_settling_to_closed():
    c = _contract(status=MudarabahStatus.SETTLING)
    new = advance_status(c, MudarabahStatus.CLOSED)
    assert new.status is MudarabahStatus.CLOSED


def test_invalid_transition_rejected():
    c = _contract(status=MudarabahStatus.DRAFT)
    with pytest.raises(ValueError):
        advance_status(c, MudarabahStatus.CLOSED)


def test_closed_cannot_advance():
    c = _contract(status=MudarabahStatus.CLOSED)
    with pytest.raises(ValueError):
        advance_status(c, MudarabahStatus.SETTLING)


# --- Settlement -----------------------------


def test_settle_only_active_or_settling():
    c = _contract(status=MudarabahStatus.DRAFT)
    with pytest.raises(ValueError):
        settle(c, final_capital_value=110000.0)


def test_settle_profit_distributed_per_ratio():
    c = _contract(status=MudarabahStatus.ACTIVE)
    s = settle(c, final_capital_value=110000.0)  # +$10k profit
    assert s.rabb_share == pytest.approx(7000.0)
    assert s.mudarib_share == pytest.approx(3000.0)
    assert not s.is_loss


def test_settle_loss_to_rabb_only():
    c = _contract(status=MudarabahStatus.ACTIVE)
    s = settle(c, final_capital_value=90000.0)  # -$10k loss
    assert s.is_loss
    assert s.rabb_share == -10000.0
    assert s.mudarib_share == 0.0


def test_settle_negligent_manager_bears_loss():
    c = _contract(status=MudarabahStatus.ACTIVE)
    s = settle(c, final_capital_value=90000.0, manager_negligent=True)
    assert s.is_loss
    assert s.rabb_share == 0.0
    assert s.mudarib_share == -10000.0


def test_settle_break_even_zero_distributions():
    c = _contract(status=MudarabahStatus.ACTIVE)
    s = settle(c, final_capital_value=100000.0)
    assert s.profit_or_loss == 0
    assert s.rabb_share == 0
    assert s.mudarib_share == 0


def test_settle_negative_final_value_rejected():
    c = _contract(status=MudarabahStatus.ACTIVE)
    with pytest.raises(ValueError):
        settle(c, final_capital_value=-1.0)


def test_settlement_negative_final_value_rejected():
    with pytest.raises(ValueError):
        Settlement(
            contract_id="x",
            final_capital_value=-1.0,
            profit_or_loss=-1.0,
            rabb_share=0,
            mudarib_share=0,
            is_loss=True,
            manager_negligent=False,
        )


# --- Render --------------------------------


def test_render_contract():
    c = _contract()
    out = render_contract(c)
    assert "M-001" in out
    assert "70/30" in out


def test_render_profit_settlement():
    c = _contract(status=MudarabahStatus.ACTIVE)
    s = settle(c, final_capital_value=110000.0)
    out = render_settlement(s)
    assert "profit" in out


def test_render_loss_settlement():
    c = _contract(status=MudarabahStatus.ACTIVE)
    s = settle(c, final_capital_value=90000.0)
    out = render_settlement(s)
    assert "loss" in out


def test_render_negligent_marker():
    c = _contract(status=MudarabahStatus.ACTIVE)
    s = settle(c, final_capital_value=90000.0, manager_negligent=True)
    out = render_settlement(s)
    assert "NEGLIGENT" in out


# --- E2E -----------------------------


def test_e2e_full_lifecycle_with_profit():
    c = _contract()
    c = advance_status(c, MudarabahStatus.ACTIVE)
    s = settle(c, final_capital_value=120000.0)
    assert s.profit_or_loss == 20000
    assert s.rabb_share == pytest.approx(14000)
    c = advance_status(c, MudarabahStatus.SETTLING)
    c = advance_status(c, MudarabahStatus.CLOSED)
    assert c.status is MudarabahStatus.CLOSED


def test_replay_consistency():
    c = _contract(status=MudarabahStatus.ACTIVE)
    a = settle(c, final_capital_value=110000)
    b = settle(c, final_capital_value=110000)
    assert a == b
