"""Tests for halal/musharakah.py — Round-5 Wave 7.B."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.musharakah import (
    MusharakahPool,
    Partner,
    PoolStatus,
    advance_status,
    render_distribution,
    render_pool,
    settle_pool,
)


def _pool(**overrides) -> MusharakahPool:
    base = {
        "pool_id": "MP-001",
        "partners": (
            Partner(handle="A", capital_contribution=70000.0, profit_share=0.7),
            Partner(handle="B", capital_contribution=30000.0, profit_share=0.3),
        ),
        "currency": "USD",
        "formation_date": date(2026, 1, 1),
        "status": PoolStatus.FORMING,
    }
    base.update(overrides)
    return MusharakahPool(**base)


# --- Validation ----------------------------------


def test_status_string_values():
    assert PoolStatus.FORMING.value == "forming"
    assert PoolStatus.ACTIVE.value == "active"
    assert PoolStatus.DISSOLVING.value == "dissolving"
    assert PoolStatus.CLOSED.value == "closed"


def test_partner_email_handle_rejected():
    with pytest.raises(ValueError):
        Partner(handle="ops@example.com", capital_contribution=100, profit_share=0.5)


def test_partner_zero_capital_rejected():
    with pytest.raises(ValueError):
        Partner(handle="A", capital_contribution=0, profit_share=0.5)


def test_partner_share_outside_unit_rejected():
    with pytest.raises(ValueError):
        Partner(handle="A", capital_contribution=100, profit_share=1.5)


def test_pool_single_partner_rejected():
    with pytest.raises(ValueError):
        MusharakahPool(
            pool_id="P",
            partners=(Partner(handle="A", capital_contribution=100, profit_share=1.0),),
            currency="USD",
            formation_date=date(2026, 1, 1),
            status=PoolStatus.FORMING,
        )


def test_pool_unbalanced_shares_rejected():
    with pytest.raises(ValueError):
        _pool(
            partners=(
                Partner(handle="A", capital_contribution=100, profit_share=0.5),
                Partner(handle="B", capital_contribution=100, profit_share=0.4),
            )
        )


def test_pool_duplicate_handles_rejected():
    with pytest.raises(ValueError):
        _pool(
            partners=(
                Partner(handle="A", capital_contribution=100, profit_share=0.5),
                Partner(handle="A", capital_contribution=100, profit_share=0.5),
            )
        )


def test_pool_total_capital():
    p = _pool()
    assert p.total_capital() == 100000


def test_capital_share():
    p = _pool()
    assert p.capital_share("A") == pytest.approx(0.7)
    assert p.capital_share("B") == pytest.approx(0.3)


def test_capital_share_unknown_raises():
    p = _pool()
    with pytest.raises(KeyError):
        p.capital_share("Z")


# --- Lifecycle -----------------------------


def test_advance_forming_to_active():
    p = _pool()
    new = advance_status(p, PoolStatus.ACTIVE)
    assert new.status is PoolStatus.ACTIVE


def test_advance_invalid_transition_rejected():
    p = _pool(status=PoolStatus.FORMING)
    with pytest.raises(ValueError):
        advance_status(p, PoolStatus.CLOSED)


# --- Settlement -----------------------------


def test_settle_only_active_or_dissolving():
    p = _pool(status=PoolStatus.FORMING)
    with pytest.raises(ValueError):
        settle_pool(p, final_pool_value=110000)


def test_settle_profit_distributed_per_profit_share():
    p = _pool(status=PoolStatus.ACTIVE)
    d = settle_pool(p, final_pool_value=110000)  # +$10k
    by_handle = dict(d.per_partner)
    assert by_handle["A"] == pytest.approx(7000)  # 70% of $10k
    assert by_handle["B"] == pytest.approx(3000)


def test_settle_loss_distributed_per_capital_contribution():
    """Fiqh rule: loss in proportion to capital, NOT profit share."""
    # Pool with skewed profit split but capital 50/50 → loss split 50/50
    p = _pool(
        partners=(
            Partner(handle="A", capital_contribution=50000, profit_share=0.8),
            Partner(handle="B", capital_contribution=50000, profit_share=0.2),
        ),
        status=PoolStatus.ACTIVE,
    )
    d = settle_pool(p, final_pool_value=80000)  # -$20k loss
    by_handle = dict(d.per_partner)
    assert by_handle["A"] == pytest.approx(-10000)
    assert by_handle["B"] == pytest.approx(-10000)


def test_settle_break_even_zero_distribution():
    p = _pool(status=PoolStatus.ACTIVE)
    d = settle_pool(p, final_pool_value=100000)
    assert d.profit_or_loss == 0
    for _, share in d.per_partner:
        assert share == 0


def test_settle_negative_final_value_rejected():
    p = _pool(status=PoolStatus.ACTIVE)
    with pytest.raises(ValueError):
        settle_pool(p, final_pool_value=-1.0)


# --- Render -----------------------------


def test_render_pool():
    p = _pool()
    out = render_pool(p)
    assert "MP-001" in out
    assert "A" in out
    assert "B" in out


def test_render_distribution():
    p = _pool(status=PoolStatus.ACTIVE)
    d = settle_pool(p, final_pool_value=110000)
    out = render_distribution(d)
    assert "profit" in out
    assert "+7000" in out


# --- E2E -----------------------------


def test_e2e_three_partner_pool_with_loss():
    """3-partner pool, loss distributed per capital contribution."""
    pool = MusharakahPool(
        pool_id="MP-002",
        partners=(
            Partner(handle="A", capital_contribution=60000, profit_share=0.5),
            Partner(handle="B", capital_contribution=30000, profit_share=0.3),
            Partner(handle="C", capital_contribution=10000, profit_share=0.2),
        ),
        currency="USD",
        formation_date=date(2026, 1, 1),
        status=PoolStatus.ACTIVE,
    )
    d = settle_pool(pool, final_pool_value=80000)  # -$20k loss
    by_handle = dict(d.per_partner)
    # Loss split 60/30/10
    assert by_handle["A"] == pytest.approx(-12000)
    assert by_handle["B"] == pytest.approx(-6000)
    assert by_handle["C"] == pytest.approx(-2000)


def test_replay_consistency():
    p = _pool(status=PoolStatus.ACTIVE)
    a = settle_pool(p, final_pool_value=110000)
    b = settle_pool(p, final_pool_value=110000)
    assert a == b
