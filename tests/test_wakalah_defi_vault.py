"""Tests for halal/wakalah_defi_vault.py — Round-5 Wave 22.B."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from halal_trader.halal.wakalah_defi_vault import (
    ShareEntry,
    VaultStatus,
    WakalahPolicy,
    WakalahVault,
    accrue_fee,
    deposit,
    mark_to_market,
    new_vault,
    pay_fee_to_manager,
    render_vault,
    transition_status,
    withdraw,
)


def _vault(
    annual_fee: float = 0.015,
    inception: date = date(2026, 5, 1),
) -> WakalahVault:
    return new_vault(
        vault_id="V1",
        manager_id="alice-mgr",
        inception_on=inception,
        policy=WakalahPolicy(annual_fee_pct=annual_fee),
    )


# --- WakalahPolicy validation -----------------------


def test_policy_default():
    p = WakalahPolicy()
    assert p.annual_fee_pct == 0.015


def test_policy_excessive_fee_rejected():
    """Pin: ≥3%/yr reads as performance carry, not Wakalah."""
    with pytest.raises(ValueError):
        WakalahPolicy(annual_fee_pct=0.05)


def test_policy_negative_min_rejected():
    with pytest.raises(ValueError):
        WakalahPolicy(min_deposit_usd=-1.0)


def test_policy_negative_max_aum_rejected():
    with pytest.raises(ValueError):
        WakalahPolicy(max_aum_usd=-1.0)


# --- new_vault + invariants ---------------------------


def test_new_vault_clean():
    v = _vault()
    assert v.status is VaultStatus.OPEN
    assert v.aum_usd == 0.0
    assert v.total_shares == 0.0
    assert v.nav_per_share() == 1.0


def test_vault_empty_id_rejected():
    with pytest.raises(ValueError):
        new_vault(
            vault_id="",
            manager_id="alice",
            inception_on=date(2026, 5, 1),
        )


def test_vault_manager_as_holder_rejected():
    """Direct construction with manager as a holder must be rejected."""
    with pytest.raises(ValueError):
        WakalahVault(
            vault_id="V1",
            manager_id="alice",
            policy=WakalahPolicy(),
            inception_on=date(2026, 5, 1),
            aum_usd=1000.0,
            total_shares=1000.0,
            accrued_fee_usd=0.0,
            last_accrual_on=date(2026, 5, 1),
            holders=(ShareEntry(depositor_id="alice", shares=1000.0),),
        )


def test_vault_holders_reconcile_to_total_shares():
    with pytest.raises(ValueError):
        WakalahVault(
            vault_id="V1",
            manager_id="alice",
            policy=WakalahPolicy(),
            inception_on=date(2026, 5, 1),
            aum_usd=1000.0,
            total_shares=1000.0,
            accrued_fee_usd=0.0,
            last_accrual_on=date(2026, 5, 1),
            holders=(ShareEntry(depositor_id="bob", shares=500.0),),
        )


def test_vault_duplicate_holders_rejected():
    with pytest.raises(ValueError):
        WakalahVault(
            vault_id="V1",
            manager_id="alice",
            policy=WakalahPolicy(),
            inception_on=date(2026, 5, 1),
            aum_usd=2000.0,
            total_shares=2000.0,
            accrued_fee_usd=0.0,
            last_accrual_on=date(2026, 5, 1),
            holders=(
                ShareEntry(depositor_id="bob", shares=1000.0),
                ShareEntry(depositor_id="bob", shares=1000.0),
            ),
        )


# --- deposit -----------------------------------------


def test_deposit_first_pins_nav_one():
    v = _vault()
    v2 = deposit(v, depositor_id="bob", amount_usd=1000.0)
    assert v2.aum_usd == 1000.0
    assert v2.total_shares == 1000.0
    assert v2.nav_per_share() == 1.0


def test_deposit_second_at_current_nav():
    v = deposit(_vault(), depositor_id="bob", amount_usd=1000.0)
    # MTM to $2000 → NAV $2.
    v = mark_to_market(v, new_aum_usd=2000.0)
    assert v.nav_per_share() == 2.0
    v = deposit(v, depositor_id="charlie", amount_usd=1000.0)
    # Charlie's shares = 1000 / 2 = 500.
    charlie = next(h for h in v.holders if h.depositor_id == "charlie")
    assert charlie.shares == pytest.approx(500.0)


def test_deposit_below_min_rejected():
    v = _vault()
    with pytest.raises(ValueError):
        deposit(v, depositor_id="bob", amount_usd=10.0)


def test_deposit_above_max_aum_rejected():
    v = WakalahVault(
        vault_id="V1",
        manager_id="alice",
        policy=WakalahPolicy(max_aum_usd=10_000.0),
        inception_on=date(2026, 5, 1),
        aum_usd=0.0,
        total_shares=0.0,
        accrued_fee_usd=0.0,
        last_accrual_on=date(2026, 5, 1),
    )
    with pytest.raises(ValueError):
        deposit(v, depositor_id="bob", amount_usd=20_000.0)


def test_deposit_manager_rejected():
    v = _vault()
    with pytest.raises(ValueError):
        deposit(v, depositor_id="alice-mgr", amount_usd=1000.0)


def test_deposit_paused_rejected():
    v = _vault()
    v = transition_status(v, new_status=VaultStatus.PAUSED)
    with pytest.raises(ValueError):
        deposit(v, depositor_id="bob", amount_usd=1000.0)


def test_deposit_closed_rejected():
    v = _vault()
    v = transition_status(v, new_status=VaultStatus.CLOSED)
    with pytest.raises(ValueError):
        deposit(v, depositor_id="bob", amount_usd=1000.0)


# --- withdraw -----------------------------------------


def test_withdraw_basic():
    v = deposit(_vault(), depositor_id="bob", amount_usd=1000.0)
    v = withdraw(v, depositor_id="bob", amount_usd=400.0)
    assert v.aum_usd == 600.0
    bob = next(h for h in v.holders if h.depositor_id == "bob")
    assert bob.shares == 600.0


def test_withdraw_full_drops_holder():
    v = deposit(_vault(), depositor_id="bob", amount_usd=1000.0)
    v = withdraw(v, depositor_id="bob", amount_usd=1000.0)
    assert not any(h.depositor_id == "bob" for h in v.holders)


def test_withdraw_above_balance_rejected():
    v = deposit(_vault(), depositor_id="bob", amount_usd=1000.0)
    with pytest.raises(ValueError):
        withdraw(v, depositor_id="bob", amount_usd=2000.0)


def test_withdraw_unknown_holder_rejected():
    v = deposit(_vault(), depositor_id="bob", amount_usd=1000.0)
    with pytest.raises(ValueError):
        withdraw(v, depositor_id="charlie", amount_usd=100.0)


def test_withdraw_paused_allowed():
    """Pin: PAUSED allows withdrawals (emergency drain)."""
    v = deposit(_vault(), depositor_id="bob", amount_usd=1000.0)
    v = transition_status(v, new_status=VaultStatus.PAUSED)
    v2 = withdraw(v, depositor_id="bob", amount_usd=500.0)
    assert v2.aum_usd == 500.0


def test_withdraw_closed_rejected():
    v = deposit(_vault(), depositor_id="bob", amount_usd=1000.0)
    v = transition_status(v, new_status=VaultStatus.CLOSED)
    with pytest.raises(ValueError):
        withdraw(v, depositor_id="bob", amount_usd=500.0)


# --- mark_to_market ----------------------------------


def test_mtm_updates_aum_only():
    v = deposit(_vault(), depositor_id="bob", amount_usd=1000.0)
    v = mark_to_market(v, new_aum_usd=1200.0)
    assert v.aum_usd == 1200.0
    assert v.total_shares == 1000.0
    assert v.nav_per_share() == 1.2


def test_mtm_below_accrued_fee_rejected():
    v = deposit(_vault(), depositor_id="bob", amount_usd=1000.0)
    # Accrue full year of 1.5% fee → $15.
    v = accrue_fee(v, on_date=date(2027, 5, 1))
    # MTM to less than accrued fee.
    with pytest.raises(ValueError):
        mark_to_market(v, new_aum_usd=10.0)


def test_mtm_negative_rejected():
    v = deposit(_vault(), depositor_id="bob", amount_usd=1000.0)
    with pytest.raises(ValueError):
        mark_to_market(v, new_aum_usd=-1.0)


# --- accrue_fee --------------------------------------


def test_accrue_fee_zero_days_noop():
    v = deposit(_vault(), depositor_id="bob", amount_usd=1000.0)
    v2 = accrue_fee(v, on_date=v.last_accrual_on)
    assert v2.accrued_fee_usd == 0.0


def test_accrue_fee_full_year():
    """Pin: 1.5% × 1000 × 365/365 = 15."""
    v = deposit(_vault(annual_fee=0.015), depositor_id="bob", amount_usd=1000.0)
    v = accrue_fee(v, on_date=v.inception_on + timedelta(days=365))
    assert v.accrued_fee_usd == pytest.approx(15.0)


def test_accrue_fee_simple_interest_not_compound():
    """Pin: two consecutive 6-month accruals == one 12-month accrual."""
    v = deposit(_vault(annual_fee=0.015), depositor_id="bob", amount_usd=1000.0)
    v_one_shot = accrue_fee(v, on_date=v.inception_on + timedelta(days=365))
    v_two_steps = accrue_fee(v, on_date=v.inception_on + timedelta(days=182))
    v_two_steps = accrue_fee(v_two_steps, on_date=v.inception_on + timedelta(days=365))
    # Simple interest: identical (no compounding on the accrued fee).
    assert v_one_shot.accrued_fee_usd == pytest.approx(v_two_steps.accrued_fee_usd, rel=1e-9)


def test_accrue_fee_capped_at_aum():
    v = WakalahVault(
        vault_id="V1",
        manager_id="alice",
        policy=WakalahPolicy(annual_fee_pct=0.029),  # max allowed
        inception_on=date(2026, 5, 1),
        aum_usd=1000.0,
        total_shares=1000.0,
        accrued_fee_usd=0.0,
        last_accrual_on=date(2026, 5, 1),
        holders=(ShareEntry(depositor_id="bob", shares=1000.0),),
    )
    # 50 years of fee → cap at AUM.
    v2 = accrue_fee(v, on_date=date(2026, 5, 1) + timedelta(days=365 * 50))
    assert v2.accrued_fee_usd <= v.aum_usd + 1e-9


def test_accrue_fee_backwards_date_rejected():
    v = deposit(_vault(), depositor_id="bob", amount_usd=1000.0)
    with pytest.raises(ValueError):
        accrue_fee(v, on_date=v.last_accrual_on - timedelta(days=1))


# --- pay_fee_to_manager ------------------------------


def test_pay_fee_drains_accrued_to_zero():
    v = deposit(_vault(annual_fee=0.015), depositor_id="bob", amount_usd=1000.0)
    v = accrue_fee(v, on_date=v.inception_on + timedelta(days=365))
    fee_before = v.accrued_fee_usd
    aum_before = v.aum_usd
    v2 = pay_fee_to_manager(v)
    assert v2.accrued_fee_usd == 0.0
    assert v2.aum_usd == pytest.approx(aum_before - fee_before)


def test_pay_fee_zero_accrued_noop():
    v = deposit(_vault(), depositor_id="bob", amount_usd=1000.0)
    v2 = pay_fee_to_manager(v)
    assert v2.aum_usd == v.aum_usd


# --- FSM ---------------------------------------------


def test_transition_open_to_paused():
    v = _vault()
    v2 = transition_status(v, new_status=VaultStatus.PAUSED)
    assert v2.status is VaultStatus.PAUSED


def test_transition_paused_to_open():
    v = transition_status(_vault(), new_status=VaultStatus.PAUSED)
    v2 = transition_status(v, new_status=VaultStatus.OPEN)
    assert v2.status is VaultStatus.OPEN


def test_transition_open_to_closed():
    v = _vault()
    v2 = transition_status(v, new_status=VaultStatus.CLOSED)
    assert v2.status is VaultStatus.CLOSED


def test_transition_closed_terminal():
    v = transition_status(_vault(), new_status=VaultStatus.CLOSED)
    with pytest.raises(ValueError):
        transition_status(v, new_status=VaultStatus.OPEN)


# --- nav_per_share helpers --------------------------


def test_nav_after_mtm():
    """Pin: NAV reflects MTM."""
    v = deposit(_vault(), depositor_id="bob", amount_usd=1000.0)
    v = mark_to_market(v, new_aum_usd=1500.0)
    assert v.nav_per_share() == 1.5


def test_nav_subtracts_accrued_fee():
    """Pin: NAV uses (aum - accrued_fee)."""
    v = deposit(_vault(annual_fee=0.015), depositor_id="bob", amount_usd=1000.0)
    v = accrue_fee(v, on_date=v.inception_on + timedelta(days=365))
    # NAV = (1000 - 15) / 1000 = 0.985.
    assert v.nav_per_share() == pytest.approx(0.985)


# --- Render -----------------------------------------


def test_render_vault_no_secret_leak():
    v = new_vault(
        vault_id="V1",
        manager_id="alice-mgr@example.com",
        inception_on=date(2026, 5, 1),
    )
    out = render_vault(v)
    assert "alice-mgr@example.com" not in out


def test_render_vault_status_emoji():
    v = _vault()
    out = render_vault(v)
    assert "🟢" in out
