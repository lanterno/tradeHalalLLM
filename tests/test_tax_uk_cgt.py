"""Tests for core/tax_uk_cgt.py — Round-5 Wave 18.C."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.core.tax_uk_cgt import (
    CgtMatch,
    MatchKind,
    UkAcquisition,
    UkDisposal,
    compute_cgt,
    render_computation,
)

# --- Validation -----------------------------------------------


def test_match_kind_string_values():
    assert MatchKind.SAME_DAY.value == "same_day"
    assert MatchKind.THIRTY_DAY.value == "thirty_day"
    assert MatchKind.S104.value == "s104"


def test_acquisition_empty_id_rejected():
    with pytest.raises(ValueError):
        UkAcquisition(
            acq_id="",
            symbol="AAPL",
            quantity=10,
            cost_per_share=100,
            acq_date=date(2026, 5, 1),
        )


def test_acquisition_negative_qty_rejected():
    with pytest.raises(ValueError):
        UkAcquisition(
            acq_id="A",
            symbol="AAPL",
            quantity=-1,
            cost_per_share=100,
            acq_date=date(2026, 5, 1),
        )


def test_disposal_zero_qty_rejected():
    with pytest.raises(ValueError):
        UkDisposal(
            disp_id="D",
            symbol="AAPL",
            quantity=0,
            proceeds_per_share=110,
            disposal_date=date(2026, 5, 1),
        )


def test_match_negative_proceeds_rejected():
    with pytest.raises(ValueError):
        CgtMatch(kind=MatchKind.SAME_DAY, quantity=10, matched_cost=100, proceeds=-1)


def test_match_zero_qty_rejected():
    with pytest.raises(ValueError):
        CgtMatch(kind=MatchKind.SAME_DAY, quantity=0, matched_cost=100, proceeds=110)


def test_match_gain_property():
    m = CgtMatch(kind=MatchKind.SAME_DAY, quantity=10, matched_cost=100, proceeds=110)
    assert m.gain == 10


# --- Same-day matching ---------------------------------------


def test_same_day_matching_used_first():
    acquisitions = [
        UkAcquisition(
            acq_id="A1",
            symbol="AAPL",
            quantity=100,
            cost_per_share=100,
            acq_date=date(2026, 5, 1),
        )
    ]
    disposal = UkDisposal(
        disp_id="D1",
        symbol="AAPL",
        quantity=100,
        proceeds_per_share=110,
        disposal_date=date(2026, 5, 1),  # same day
    )
    comp = compute_cgt(disposal, acquisitions)
    assert len(comp.matches) == 1
    assert comp.matches[0].kind is MatchKind.SAME_DAY
    assert comp.total_gain == 1000  # (110-100)*100


def test_same_day_partial_consumption():
    """Disposal larger than same-day buy: remainder goes to s104 pool."""
    acquisitions = [
        UkAcquisition("A1", "AAPL", 50, 100, date(2026, 5, 1)),
    ]
    disposal = UkDisposal("D1", "AAPL", 100, 110, date(2026, 5, 1))
    comp = compute_cgt(disposal, acquisitions, s104_pool_quantity=50, s104_pool_cost=4000)
    kinds = [m.kind for m in comp.matches]
    assert MatchKind.SAME_DAY in kinds
    assert MatchKind.S104 in kinds


# --- 30-day rule ---------------------------------------------


def test_thirty_day_rule_matched_against_post_disposal_buy():
    """Disposal followed by re-acquisition within 30d: matched against re-acq."""
    acquisitions = [
        UkAcquisition("A1", "AAPL", 100, 105, date(2026, 5, 15)),  # 14d after disposal
    ]
    disposal = UkDisposal("D1", "AAPL", 100, 110, date(2026, 5, 1))
    comp = compute_cgt(disposal, acquisitions)
    assert len(comp.matches) == 1
    assert comp.matches[0].kind is MatchKind.THIRTY_DAY
    assert comp.total_gain == 500  # (110-105)*100


def test_thirty_day_rule_at_boundary_included():
    """Acquisition exactly 30 days after disposal is matched."""
    acquisitions = [
        UkAcquisition("A1", "AAPL", 100, 105, date(2026, 5, 31)),  # exactly 30d
    ]
    disposal = UkDisposal("D1", "AAPL", 100, 110, date(2026, 5, 1))
    comp = compute_cgt(disposal, acquisitions)
    assert comp.matches[0].kind is MatchKind.THIRTY_DAY


def test_thirty_day_rule_just_past_boundary_excluded():
    """Acquisition 31 days after disposal not matched."""
    acquisitions = [
        UkAcquisition("A1", "AAPL", 100, 105, date(2026, 6, 1)),  # 31d
    ]
    disposal = UkDisposal("D1", "AAPL", 100, 110, date(2026, 5, 1))
    comp = compute_cgt(disposal, acquisitions, s104_pool_quantity=100, s104_pool_cost=10000)
    assert all(m.kind is not MatchKind.THIRTY_DAY for m in comp.matches)


def test_thirty_day_rule_pre_disposal_buys_excluded():
    """Acquisition BEFORE disposal not matched under 30-day rule."""
    acquisitions = [
        UkAcquisition("A1", "AAPL", 100, 105, date(2026, 4, 25)),
    ]
    disposal = UkDisposal("D1", "AAPL", 100, 110, date(2026, 5, 1))
    comp = compute_cgt(disposal, acquisitions, s104_pool_quantity=100, s104_pool_cost=10000)
    # 30-day rule looks AFTER disposal; pre-disposal buys go to s104.
    kinds = [m.kind for m in comp.matches]
    assert MatchKind.S104 in kinds


# --- s104 pool -----------------------------------------------


def test_s104_average_cost_used():
    """Falls through to s104 pool with averaged cost."""
    disposal = UkDisposal("D1", "AAPL", 50, 120, date(2026, 5, 1))
    comp = compute_cgt(
        disposal,
        acquisitions=[],
        s104_pool_quantity=200,
        s104_pool_cost=20000,  # avg £100
    )
    assert comp.matches[0].kind is MatchKind.S104
    assert comp.matches[0].matched_cost == 5000  # 50 * £100
    assert comp.total_gain == 1000  # 50 * (120-100)


def test_s104_partial_consumption():
    """Disposal larger than s104 pool: only s104 portion matched."""
    disposal = UkDisposal("D1", "AAPL", 200, 110, date(2026, 5, 1))
    comp = compute_cgt(
        disposal,
        acquisitions=[],
        s104_pool_quantity=50,
        s104_pool_cost=5000,
    )
    assert comp.total_quantity_matched() == 50
    assert comp.matches[0].kind is MatchKind.S104


# --- Validation ---------------------------------------------


def test_negative_s104_quantity_rejected():
    disposal = UkDisposal("D1", "AAPL", 10, 100, date(2026, 5, 1))
    with pytest.raises(ValueError):
        compute_cgt(disposal, acquisitions=[], s104_pool_quantity=-1, s104_pool_cost=0)


def test_negative_s104_cost_rejected():
    disposal = UkDisposal("D1", "AAPL", 10, 100, date(2026, 5, 1))
    with pytest.raises(ValueError):
        compute_cgt(disposal, acquisitions=[], s104_pool_quantity=10, s104_pool_cost=-1)


# --- Render -------------------------------------------------


def test_render_includes_summary():
    acquisitions = [UkAcquisition("A1", "AAPL", 100, 100, date(2026, 5, 1))]
    disposal = UkDisposal("D1", "AAPL", 100, 110, date(2026, 5, 1))
    comp = compute_cgt(disposal, acquisitions)
    out = render_computation(comp)
    assert "D1" in out
    assert "AAPL" in out
    assert "same_day" in out


def test_render_loss_signed():
    acquisitions = [UkAcquisition("A1", "AAPL", 100, 110, date(2026, 5, 1))]
    disposal = UkDisposal("D1", "AAPL", 100, 100, date(2026, 5, 1))
    comp = compute_cgt(disposal, acquisitions)
    out = render_computation(comp)
    # Loss should appear with negative sign
    assert "-1000" in out


def test_render_no_secret_leak():
    acquisitions = [UkAcquisition("A1", "AAPL", 100, 100, date(2026, 5, 1))]
    disposal = UkDisposal("D1", "AAPL", 100, 110, date(2026, 5, 1))
    comp = compute_cgt(disposal, acquisitions)
    out = render_computation(comp)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ------------------------------------------------


def test_e2e_full_uk_matching_chain():
    """Same-day + 30-day + s104 all engaged in one disposal."""
    acquisitions = [
        UkAcquisition("A1", "AAPL", 30, 100, date(2026, 5, 1)),  # same-day
        UkAcquisition("A2", "AAPL", 40, 105, date(2026, 5, 10)),  # 30-day
    ]
    disposal = UkDisposal("D1", "AAPL", 100, 110, date(2026, 5, 1))
    comp = compute_cgt(disposal, acquisitions, s104_pool_quantity=30, s104_pool_cost=2700)
    kinds = [m.kind for m in comp.matches]
    assert MatchKind.SAME_DAY in kinds
    assert MatchKind.THIRTY_DAY in kinds
    assert MatchKind.S104 in kinds
    assert comp.total_quantity_matched() == 100


def test_replay_consistency():
    acquisitions = [UkAcquisition("A1", "AAPL", 100, 100, date(2026, 5, 1))]
    disposal = UkDisposal("D1", "AAPL", 100, 110, date(2026, 5, 1))
    a = compute_cgt(disposal, acquisitions)
    b = compute_cgt(disposal, acquisitions)
    assert a == b
