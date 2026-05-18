"""Tests for core/tax_loss_harvest.py — Round-5 Wave 18.G."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.core.tax_loss_harvest import (
    HarvestCandidate,
    HarvestPolicy,
    render_candidates,
    select_candidates,
    top_n_candidates,
    total_harvestable_loss,
)
from halal_trader.core.tax_lots import TaxLot


def _lot(
    lot_id: str = "L1",
    symbol: str = "AAPL",
    qty: float = 100.0,
    basis: float = 100.0,
    acq: date = date(2024, 1, 1),
) -> TaxLot:
    return TaxLot(
        lot_id=lot_id,
        symbol=symbol,
        quantity=qty,
        cost_basis_per_share=basis,
        acquisition_date=acq,
    )


# --- Validation -------------------------------------------------------------


def test_default_policy():
    p = HarvestPolicy()
    assert p.wash_sale_window_days == 30
    assert p.min_loss_amount == 50.0
    assert p.min_loss_pct == 0.02


def test_policy_negative_window_rejected():
    with pytest.raises(ValueError):
        HarvestPolicy(wash_sale_window_days=-1)


def test_policy_negative_min_loss_rejected():
    with pytest.raises(ValueError):
        HarvestPolicy(min_loss_amount=-1.0)


def test_policy_above_one_min_pct_rejected():
    with pytest.raises(ValueError):
        HarvestPolicy(min_loss_pct=1.5)


def test_candidate_negative_market_rejected():
    with pytest.raises(ValueError):
        HarvestCandidate(lot=_lot(), market_price=-1.0, unrealised_loss=10.0, loss_pct=0.1)


def test_candidate_negative_loss_rejected():
    """Loss is positive number — negative would be a gain."""
    with pytest.raises(ValueError):
        HarvestCandidate(lot=_lot(), market_price=100.0, unrealised_loss=-1.0, loss_pct=0.1)


# --- Selection -------------------------------------------------------------


def test_select_loss_above_threshold_picked():
    pool = (_lot(qty=100, basis=100, acq=date(2024, 1, 1)),)
    prices = {"AAPL": 80.0}  # 20% loss = $2000
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    assert len(candidates) == 1
    assert candidates[0].unrealised_loss == 2000.0


def test_select_gain_excluded():
    pool = (_lot(qty=100, basis=100, acq=date(2024, 1, 1)),)
    prices = {"AAPL": 120.0}  # gain
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    assert candidates == ()


def test_select_break_even_excluded():
    pool = (_lot(qty=100, basis=100, acq=date(2024, 1, 1)),)
    prices = {"AAPL": 100.0}
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    assert candidates == ()


def test_select_below_min_loss_amount_excluded():
    """Loss of $30 < default min_loss_amount $50 → excluded."""
    pool = (_lot(qty=10, basis=100, acq=date(2024, 1, 1)),)
    prices = {"AAPL": 97.0}  # $30 loss
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    assert candidates == ()


def test_select_below_min_loss_pct_excluded():
    """Loss of 1% < default 2% → excluded."""
    pool = (_lot(qty=1000, basis=100, acq=date(2024, 1, 1)),)
    prices = {"AAPL": 99.0}  # 1% loss = $1000 (above $ min, but below %)
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    assert candidates == ()


def test_select_in_wash_zone_excluded():
    """Lot acquired 10 days ago is within 30d wash window."""
    pool = (_lot(qty=100, basis=100, acq=date(2025, 5, 25)),)
    prices = {"AAPL": 80.0}
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    assert candidates == ()


def test_select_outside_wash_zone_included():
    """Lot acquired 60 days ago is outside 30d window."""
    pool = (_lot(qty=100, basis=100, acq=date(2025, 4, 1)),)
    prices = {"AAPL": 80.0}
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    assert len(candidates) == 1


def test_select_at_wash_boundary_excluded():
    """Lot acquired exactly 30 days ago is at boundary — still in wash zone (>=)."""
    pool = (_lot(qty=100, basis=100, acq=date(2025, 5, 2)),)
    prices = {"AAPL": 80.0}
    # today = 2025-06-01, threshold = 2025-05-02 → acq == threshold → in zone
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    assert candidates == ()


def test_select_just_past_wash_boundary_included():
    """Lot acquired 31 days ago is past wash window."""
    pool = (_lot(qty=100, basis=100, acq=date(2025, 5, 1)),)
    prices = {"AAPL": 80.0}
    # today = 2025-06-01, threshold = 2025-05-02 → 5/1 < 5/2 → outside zone
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    assert len(candidates) == 1


def test_select_missing_price_excluded():
    pool = (_lot(qty=100, basis=100, acq=date(2024, 1, 1)),)
    prices: dict[str, float] = {}
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    assert candidates == ()


def test_select_negative_price_excluded():
    pool = (_lot(qty=100, basis=100, acq=date(2024, 1, 1)),)
    prices = {"AAPL": -1.0}  # bad data
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    assert candidates == ()


def test_select_sorted_by_loss_descending():
    pool = (
        _lot("L1", qty=100, basis=100, acq=date(2024, 1, 1)),  # loss=$2000
        _lot("L2", qty=100, basis=100, acq=date(2024, 1, 1), symbol="MSFT"),  # loss=$5000
        _lot("L3", qty=100, basis=100, acq=date(2024, 1, 1), symbol="GOOGL"),  # loss=$1000
    )
    prices = {"AAPL": 80.0, "MSFT": 50.0, "GOOGL": 90.0}
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    losses = [c.unrealised_loss for c in candidates]
    assert losses == sorted(losses, reverse=True)


def test_select_custom_policy_relaxed():
    """Tighter wash window allows recently-acquired lots."""
    pool = (_lot(qty=100, basis=100, acq=date(2025, 5, 25)),)
    prices = {"AAPL": 80.0}
    pol = HarvestPolicy(wash_sale_window_days=0)
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1), policy=pol)
    assert len(candidates) == 1


# --- Helpers ----------------------------------------------------------------


def test_total_harvestable_loss_sums():
    pool = (
        _lot("L1", qty=100, basis=100, acq=date(2024, 1, 1)),
        _lot("L2", qty=100, basis=100, acq=date(2024, 1, 1), symbol="MSFT"),
    )
    prices = {"AAPL": 80.0, "MSFT": 50.0}
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    assert total_harvestable_loss(candidates) == 2000.0 + 5000.0


def test_top_n_candidates_returns_first_n():
    pool = tuple(
        _lot(
            f"L{i}",
            qty=100,
            basis=100,
            acq=date(2024, 1, 1),
            symbol=f"SYM{i}",
        )
        for i in range(5)
    )
    prices = {f"SYM{i}": 90.0 - i * 5 for i in range(5)}
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    top = top_n_candidates(candidates, 2)
    assert len(top) == 2
    # First two are biggest losses
    assert top == candidates[:2]


def test_top_n_zero_rejected():
    with pytest.raises(ValueError):
        top_n_candidates([], 0)


# --- Render -----------------------------------------------------------------


def test_render_empty():
    out = render_candidates(())
    assert "none" in out


def test_render_lists_candidates():
    pool = (_lot("L1", qty=100, basis=100, acq=date(2024, 1, 1)),)
    prices = {"AAPL": 80.0}
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    out = render_candidates(candidates)
    assert "L1" in out
    assert "AAPL" in out
    assert "$2000" in out


def test_render_no_secret_leak():
    pool = (_lot("L1"),)
    prices = {"AAPL": 80.0}
    candidates = select_candidates(pool, prices, today=date(2025, 6, 1))
    out = render_candidates(candidates)
    for token in (
        "@",
        "zoom.us",
        "meet.google",
        "private_email",
        "+1-",
        "Authorization",
        "SSN",
        "TaxID",
    ):
        assert token not in out


# --- E2E -------------------------------------------------------------------


def test_e2e_year_end_harvest():
    """End-of-year harvest: pick top 3 loss positions, exclude wash-zone + small losses."""
    pool = (
        _lot("L1", qty=100, basis=100, acq=date(2025, 1, 1), symbol="A"),
        _lot("L2", qty=200, basis=100, acq=date(2025, 1, 1), symbol="B"),
        _lot("L3", qty=10, basis=100, acq=date(2025, 1, 1), symbol="C"),  # tiny
        _lot("L4", qty=100, basis=100, acq=date(2025, 11, 25), symbol="D"),  # wash zone
        _lot("L5", qty=100, basis=100, acq=date(2025, 1, 1), symbol="E"),  # gain
    )
    prices = {
        "A": 50.0,  # 50% loss = $5000 ✓
        "B": 80.0,  # 20% loss = $4000 ✓
        "C": 50.0,  # 50% but only $500 — passes both thresholds, kept
        "D": 50.0,  # would qualify but in wash zone
        "E": 150.0,  # gain
    }
    candidates = select_candidates(pool, prices, today=date(2025, 12, 15))
    ids = [c.lot.lot_id for c in candidates]
    assert "L1" in ids
    assert "L2" in ids
    assert "L3" in ids
    assert "L4" not in ids
    assert "L5" not in ids


def test_replay_consistency():
    pool = (_lot(),)
    prices = {"AAPL": 80.0}
    a = select_candidates(pool, prices, today=date(2025, 6, 1))
    b = select_candidates(pool, prices, today=date(2025, 6, 1))
    assert a == b
