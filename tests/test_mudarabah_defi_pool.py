"""Tests for halal/mudarabah_defi_pool.py — Round-5 Wave 22.C."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.mudarabah_defi_pool import (
    DepositorEntry,
    MudarabahPool,
    PoolStatus,
    deposit,
    distribute_profit,
    mark_to_market,
    new_pool,
    pay_manager,
    render_pool,
    transition_status,
    withdraw,
)


def _pool(
    profit_share: float = 0.20,
    inception: date = date(2026, 5, 1),
) -> MudarabahPool:
    return new_pool(
        pool_id="P1",
        manager_id="alice-mgr",
        manager_profit_share_pct=profit_share,
        inception_on=inception,
    )


# --- MudarabahPool validation ----------------------


def test_pool_clean():
    p = _pool()
    assert p.status is PoolStatus.OPEN
    assert p.aum_usd == 0.0
    assert p.nav_per_unit() == 1.0


def test_pool_empty_id_rejected():
    with pytest.raises(ValueError):
        new_pool(
            pool_id="",
            manager_id="alice",
            manager_profit_share_pct=0.20,
            inception_on=date(2026, 5, 1),
        )


def test_pool_profit_share_at_one_rejected():
    with pytest.raises(ValueError):
        new_pool(
            pool_id="P1",
            manager_id="alice",
            manager_profit_share_pct=1.0,
            inception_on=date(2026, 5, 1),
        )


def test_pool_profit_share_at_zero_rejected():
    with pytest.raises(ValueError):
        new_pool(
            pool_id="P1",
            manager_id="alice",
            manager_profit_share_pct=0.0,
            inception_on=date(2026, 5, 1),
        )


def test_pool_manager_as_depositor_rejected():
    with pytest.raises(ValueError):
        MudarabahPool(
            pool_id="P1",
            manager_id="alice",
            manager_profit_share_pct=0.20,
            inception_on=date(2026, 5, 1),
            aum_usd=1000.0,
            total_units=1000.0,
            manager_owed_usd=0.0,
            depositors=(DepositorEntry(depositor_id="alice", units=1000.0),),
            high_water_aum=1000.0,
        )


def test_pool_units_reconcile():
    with pytest.raises(ValueError):
        MudarabahPool(
            pool_id="P1",
            manager_id="alice",
            manager_profit_share_pct=0.20,
            inception_on=date(2026, 5, 1),
            aum_usd=1000.0,
            total_units=1000.0,
            manager_owed_usd=0.0,
            depositors=(DepositorEntry(depositor_id="bob", units=500.0),),
            high_water_aum=1000.0,
        )


# --- deposit ----------------------------------------


def test_deposit_first_pins_nav_one():
    p = _pool()
    p2 = deposit(p, depositor_id="bob", amount_usd=1000.0)
    assert p2.aum_usd == 1000.0
    assert p2.total_units == 1000.0
    assert p2.nav_per_unit() == 1.0


def test_deposit_high_water_updated():
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    assert p.high_water_aum == 1000.0


def test_deposit_manager_rejected():
    p = _pool()
    with pytest.raises(ValueError):
        deposit(p, depositor_id="alice-mgr", amount_usd=1000.0)


def test_deposit_paused_rejected():
    p = transition_status(_pool(), new_status=PoolStatus.PAUSED)
    with pytest.raises(ValueError):
        deposit(p, depositor_id="bob", amount_usd=1000.0)


def test_deposit_closed_rejected():
    p = transition_status(_pool(), new_status=PoolStatus.CLOSED)
    with pytest.raises(ValueError):
        deposit(p, depositor_id="bob", amount_usd=1000.0)


def test_deposit_zero_rejected():
    p = _pool()
    with pytest.raises(ValueError):
        deposit(p, depositor_id="bob", amount_usd=0)


# --- withdraw ---------------------------------------


def test_withdraw_basic():
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    p = withdraw(p, depositor_id="bob", amount_usd=400.0)
    assert p.aum_usd == 600.0


def test_withdraw_above_balance_rejected():
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    with pytest.raises(ValueError):
        withdraw(p, depositor_id="bob", amount_usd=2000.0)


def test_withdraw_unknown_holder_rejected():
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    with pytest.raises(ValueError):
        withdraw(p, depositor_id="charlie", amount_usd=100.0)


def test_withdraw_closed_rejected():
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    p = transition_status(p, new_status=PoolStatus.CLOSED)
    with pytest.raises(ValueError):
        withdraw(p, depositor_id="bob", amount_usd=500.0)


def test_withdraw_paused_allowed():
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    p = transition_status(p, new_status=PoolStatus.PAUSED)
    p2 = withdraw(p, depositor_id="bob", amount_usd=500.0)
    assert p2.aum_usd == 500.0


# --- distribute_profit ------------------------------


def test_distribute_profit_basic():
    """20% to manager, 80% to depositors.

    Deposit 1000 → HWM 1000; MTM 1200 → excess 200; manager takes
    20% × 200 = 40, depositors keep 160.
    """
    p = deposit(_pool(profit_share=0.20), depositor_id="bob", amount_usd=1000.0)
    p = mark_to_market(p, new_aum_usd=1200.0)
    p = distribute_profit(p)
    assert p.manager_owed_usd == pytest.approx(40.0)
    # New HWM = 1000 + (200 - 40) = 1160; depositor's net AUM = 1160.
    assert p.high_water_aum == pytest.approx(1160.0)
    # Depositor NAV = (1200 - 40) / 1000 = 1.16.
    assert p.nav_per_unit() == pytest.approx(1.16)


def test_distribute_profit_below_hwm_noop():
    """Pin: loss / below HWM → manager earns nothing."""
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    p = mark_to_market(p, new_aum_usd=900.0)  # 10% loss
    p2 = distribute_profit(p)
    assert p2.manager_owed_usd == 0.0
    assert p2.high_water_aum == 1000.0  # HWM unchanged


def test_distribute_profit_only_above_high_water_pin():
    """Pin: manager only earns on excess over HWM (true high-water mark)."""
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    # Push to 1500, distribute (HWM goes to ~1400).
    p = mark_to_market(p, new_aum_usd=1500.0)
    p = distribute_profit(p)
    hwm_after_first = p.high_water_aum
    # Now drop back to 1200 — no distribution should occur.
    p = mark_to_market(p, new_aum_usd=1200.0)
    p_after = distribute_profit(p)
    # Manager owed unchanged.
    assert p_after.manager_owed_usd == p.manager_owed_usd
    # HWM unchanged.
    assert p_after.high_water_aum == hwm_after_first


def test_distribute_after_loss_then_recovery():
    """Pin: after a loss, no distribution until back above HWM."""
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    # Loss to 900.
    p = mark_to_market(p, new_aum_usd=900.0)
    p = distribute_profit(p)
    # Recovery to 1050 (still below HWM 1000? actually above).
    p = mark_to_market(p, new_aum_usd=1050.0)
    p_after = distribute_profit(p)
    # excess = 50; manager takes 20% × 50 = 10.
    assert p_after.manager_owed_usd == pytest.approx(10.0)


# --- mark_to_market ---------------------------------


def test_mtm_updates_aum_only():
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    p2 = mark_to_market(p, new_aum_usd=1200.0)
    assert p2.aum_usd == 1200.0
    assert p2.total_units == 1000.0


def test_mtm_below_manager_owed_rejected():
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    p = mark_to_market(p, new_aum_usd=1500.0)
    p = distribute_profit(p)  # manager_owed > 0
    # Crash to below manager_owed.
    with pytest.raises(ValueError):
        mark_to_market(p, new_aum_usd=p.manager_owed_usd - 1.0)


def test_mtm_negative_rejected():
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    with pytest.raises(ValueError):
        mark_to_market(p, new_aum_usd=-1.0)


# --- pay_manager ----------------------------------


def test_pay_manager_drains_owed():
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    p = mark_to_market(p, new_aum_usd=1200.0)
    p = distribute_profit(p)
    aum_before = p.aum_usd
    owed_before = p.manager_owed_usd
    p2 = pay_manager(p)
    assert p2.manager_owed_usd == 0.0
    assert p2.aum_usd == pytest.approx(aum_before - owed_before)


def test_pay_manager_zero_noop():
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    p2 = pay_manager(p)
    assert p2.aum_usd == p.aum_usd


# --- Loss-bearing structural pin -----------------


def test_depositors_absorb_loss_alone():
    """Pin: Mudarabah loss → depositors absorb; manager_owed unchanged."""
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    p = mark_to_market(p, new_aum_usd=800.0)
    # No distribute_profit needed; loss is reflected in NAV.
    bob_value = next(d for d in p.depositors if d.depositor_id == "bob").units * p.nav_per_unit()
    assert bob_value == pytest.approx(800.0)
    assert p.manager_owed_usd == 0.0


# --- FSM ----------------------------------------


def test_transition_open_to_paused():
    p = _pool()
    p2 = transition_status(p, new_status=PoolStatus.PAUSED)
    assert p2.status is PoolStatus.PAUSED


def test_transition_closed_terminal():
    p = transition_status(_pool(), new_status=PoolStatus.CLOSED)
    with pytest.raises(ValueError):
        transition_status(p, new_status=PoolStatus.OPEN)


# --- NAV helpers --------------------------------


def test_nav_subtracts_manager_owed():
    """Pin: depositor NAV = (aum − manager_owed) / units."""
    p = deposit(_pool(), depositor_id="bob", amount_usd=1000.0)
    p = mark_to_market(p, new_aum_usd=1200.0)
    p = distribute_profit(p)
    # NAV reflects depositor net.
    assert p.nav_per_unit() == pytest.approx(p.net_aum_for_depositors() / p.total_units)


# --- Render ------------------------------------


def test_render_no_secret_leak():
    p = new_pool(
        pool_id="P1",
        manager_id="alice-mgr@example.com",
        manager_profit_share_pct=0.20,
        inception_on=date(2026, 5, 1),
    )
    out = render_pool(p)
    assert "alice-mgr@example.com" not in out


def test_render_status_emoji():
    p = _pool()
    out = render_pool(p)
    assert "🟢" in out
