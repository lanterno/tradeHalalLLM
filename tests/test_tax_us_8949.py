"""Tests for core/tax_us_8949.py — Round-5 Wave 18.B."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.core.tax_lots import RealisedSlice
from halal_trader.core.tax_us_8949 import (
    BoxTotals,
    Form8949Row,
    FormBox,
    box_totals,
    render_row,
    render_summary,
    slice_to_8949_row,
    slices_to_8949_rows,
)


def _slice(
    qty: float = 100,
    basis: float = 50,
    proceeds: float = 60,
    acq: date = date(2024, 1, 1),
    sale: date = date(2025, 6, 1),
) -> RealisedSlice:
    return RealisedSlice(
        lot_id="L1",
        quantity=qty,
        cost_basis_per_share=basis,
        proceeds_per_share=proceeds,
        acquisition_date=acq,
        sale_date=sale,
    )


# --- Enum + validation ----------------------------------------------------


def test_form_box_string_values():
    assert FormBox.BOX_A.value == "box_a"
    assert FormBox.BOX_B.value == "box_b"
    assert FormBox.BOX_C.value == "box_c"
    assert FormBox.BOX_D.value == "box_d"
    assert FormBox.BOX_E.value == "box_e"
    assert FormBox.BOX_F.value == "box_f"


def test_row_invariant_correct():
    """gain_loss = proceeds - basis + adjustment."""
    row = Form8949Row(
        description="100 AAPL",
        acquired_date=date(2024, 1, 1),
        sold_date=date(2025, 6, 1),
        proceeds=6000,
        cost_basis=5000,
        code="",
        adjustment=0,
        gain_loss=1000,
        box=FormBox.BOX_D,
    )
    assert row.gain_loss == 1000


def test_row_invariant_violation_rejected():
    with pytest.raises(ValueError):
        Form8949Row(
            description="x",
            acquired_date=date(2024, 1, 1),
            sold_date=date(2025, 1, 1),
            proceeds=100,
            cost_basis=50,
            code="",
            adjustment=0,
            gain_loss=999,  # should be 50
            box=FormBox.BOX_A,
        )


def test_row_empty_description_rejected():
    with pytest.raises(ValueError):
        Form8949Row(
            description="",
            acquired_date=date(2024, 1, 1),
            sold_date=date(2025, 1, 1),
            proceeds=100,
            cost_basis=50,
            code="",
            adjustment=0,
            gain_loss=50,
            box=FormBox.BOX_A,
        )


def test_row_negative_proceeds_rejected():
    with pytest.raises(ValueError):
        Form8949Row(
            description="x",
            acquired_date=date(2024, 1, 1),
            sold_date=date(2025, 1, 1),
            proceeds=-1,
            cost_basis=50,
            code="",
            adjustment=0,
            gain_loss=-51,
            box=FormBox.BOX_A,
        )


def test_box_totals_n_negative_rejected():
    with pytest.raises(ValueError):
        BoxTotals(
            box=FormBox.BOX_A,
            n_rows=-1,
            proceeds=0,
            cost_basis=0,
            adjustment=0,
            gain_loss=0,
        )


# --- Box selection -------------------------------------------------------


def test_short_term_basis_reported_box_a():
    s = _slice(acq=date(2025, 1, 1), sale=date(2025, 6, 1))
    row = slice_to_8949_row(s, symbol="AAPL", basis_reported_to_irs=True)
    assert row.box is FormBox.BOX_A


def test_short_term_basis_unreported_box_b():
    s = _slice(acq=date(2025, 1, 1), sale=date(2025, 6, 1))
    row = slice_to_8949_row(s, symbol="AAPL", basis_reported_to_irs=False)
    assert row.box is FormBox.BOX_B


def test_long_term_basis_reported_box_d():
    s = _slice(acq=date(2024, 1, 1), sale=date(2025, 6, 1))  # > 1 year
    row = slice_to_8949_row(s, symbol="AAPL", basis_reported_to_irs=True)
    assert row.box is FormBox.BOX_D


def test_long_term_basis_unreported_box_e():
    s = _slice(acq=date(2024, 1, 1), sale=date(2025, 6, 1))
    row = slice_to_8949_row(s, symbol="AAPL", basis_reported_to_irs=False)
    assert row.box is FormBox.BOX_E


# --- Numbers --------------------------------------------------------------


def test_row_proceeds_basis_correct():
    s = _slice(qty=100, basis=50, proceeds=60)
    row = slice_to_8949_row(s, symbol="AAPL")
    assert row.proceeds == 6000
    assert row.cost_basis == 5000
    assert row.gain_loss == 1000


def test_row_loss_negative_gain_loss():
    s = _slice(qty=100, basis=60, proceeds=50)  # $1000 loss
    row = slice_to_8949_row(s, symbol="AAPL")
    assert row.gain_loss == -1000


def test_row_description_includes_qty_and_symbol():
    s = _slice(qty=50)
    row = slice_to_8949_row(s, symbol="MSFT")
    assert "MSFT" in row.description
    assert "50" in row.description


# --- Wash sale ------------------------------------------------------------


def test_wash_sale_sets_w_code():
    s = _slice(qty=100, basis=60, proceeds=50)
    row = slice_to_8949_row(s, symbol="AAPL", is_wash_sale=True, wash_sale_disallowed=1000)
    assert row.code == "W"
    assert row.adjustment == 1000
    # Adjustment offsets the loss → gain_loss = proceeds - basis + adjustment = 0
    assert row.gain_loss == 0


def test_no_wash_sale_no_code():
    s = _slice()
    row = slice_to_8949_row(s, symbol="AAPL", is_wash_sale=False)
    assert row.code == ""
    assert row.adjustment == 0


# --- Batch + totals ------------------------------------------------------


def test_slices_to_rows_batch():
    slices = [
        _slice(qty=100, acq=date(2024, 1, 1), sale=date(2025, 6, 1)),
        _slice(qty=50, acq=date(2025, 1, 1), sale=date(2025, 6, 1)),
    ]
    rows = slices_to_8949_rows(slices, symbol="AAPL")
    assert len(rows) == 2


def test_box_totals_aggregates_by_box():
    rows = (
        Form8949Row(
            description="A",
            acquired_date=date(2025, 1, 1),
            sold_date=date(2025, 6, 1),
            proceeds=1000,
            cost_basis=500,
            code="",
            adjustment=0,
            gain_loss=500,
            box=FormBox.BOX_A,
        ),
        Form8949Row(
            description="B",
            acquired_date=date(2025, 1, 1),
            sold_date=date(2025, 6, 1),
            proceeds=2000,
            cost_basis=1000,
            code="",
            adjustment=0,
            gain_loss=1000,
            box=FormBox.BOX_A,
        ),
        Form8949Row(
            description="C",
            acquired_date=date(2024, 1, 1),
            sold_date=date(2025, 6, 1),
            proceeds=3000,
            cost_basis=4000,
            code="",
            adjustment=0,
            gain_loss=-1000,
            box=FormBox.BOX_D,
        ),
    )
    totals = box_totals(rows)
    assert totals[FormBox.BOX_A].n_rows == 2
    assert totals[FormBox.BOX_A].gain_loss == 1500
    assert totals[FormBox.BOX_D].n_rows == 1
    assert totals[FormBox.BOX_D].gain_loss == -1000


def test_box_totals_empty_returns_empty_dict():
    assert box_totals([]) == {}


# --- Render ---------------------------------------------------------------


def test_render_row_includes_box_and_description():
    s = _slice(qty=100, basis=50, proceeds=60)
    row = slice_to_8949_row(s, symbol="AAPL")
    out = render_row(row)
    assert "AAPL" in out
    assert "box_d" in out  # 100 shares held > 1y → BOX_D


def test_render_row_loss_marker():
    s = _slice(qty=100, basis=60, proceeds=50)
    row = slice_to_8949_row(s, symbol="AAPL")
    out = render_row(row)
    assert "(loss)" in out


def test_render_summary_empty():
    assert "no rows" in render_summary([])


def test_render_summary_with_rows():
    rows = slices_to_8949_rows(
        [_slice(qty=100), _slice(qty=50, acq=date(2025, 1, 1), sale=date(2025, 6, 1))],
        symbol="AAPL",
    )
    out = render_summary(rows)
    assert "Form 8949" in out
    assert "box_a" in out or "box_d" in out


def test_render_no_secret_leak():
    s = _slice(qty=100)
    row = slice_to_8949_row(s, symbol="AAPL")
    out = render_row(row)
    for token in (
        "@",
        "zoom.us",
        "meet.google",
        "private_email",
        "+1-",
        "Authorization",
        "SSN",
        "TaxID",
        "DOB",
    ):
        assert token not in out


# --- E2E ---------------------------------------------------------------


def test_e2e_year_end_8949_summary():
    """Year-end: mix of short + long, gain + loss → consistent summary."""
    slices = [
        _slice(qty=100, basis=50, proceeds=60, acq=date(2024, 1, 1), sale=date(2025, 6, 1)),  # LT gain
        _slice(qty=100, basis=70, proceeds=60, acq=date(2024, 1, 1), sale=date(2025, 6, 1)),  # LT loss
        _slice(qty=100, basis=50, proceeds=60, acq=date(2025, 1, 1), sale=date(2025, 6, 1)),  # ST gain
    ]
    rows = slices_to_8949_rows(slices, symbol="AAPL")
    totals = box_totals(rows)
    assert totals[FormBox.BOX_A].n_rows == 1  # short-term
    assert totals[FormBox.BOX_D].n_rows == 2  # long-term
    # LT net: +1000 (gain) - 1000 (loss) = 0
    assert totals[FormBox.BOX_D].gain_loss == 0
    # ST net: +1000
    assert totals[FormBox.BOX_A].gain_loss == 1000


def test_replay_consistency():
    s = _slice()
    a = slice_to_8949_row(s, symbol="AAPL")
    b = slice_to_8949_row(s, symbol="AAPL")
    assert a == b
