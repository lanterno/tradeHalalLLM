"""Tests for core/tax_lots.py — Round-5 Wave 18.A."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.core.tax_lots import (
    LotMethod,
    RealisedSlice,
    TaxLot,
    apply_sale,
    render_pool,
    render_realisation,
    split_long_short,
    total_cost_basis,
    total_quantity,
    total_realised_pnl,
)


def _lot(
    lot_id: str = "L1",
    qty: float = 100.0,
    basis: float = 50.0,
    acq: date = date(2024, 1, 1),
) -> TaxLot:
    return TaxLot(
        lot_id=lot_id,
        symbol="AAPL",
        quantity=qty,
        cost_basis_per_share=basis,
        acquisition_date=acq,
    )


# --- Validation -------------------------------------------------------------


def test_method_string_values():
    assert LotMethod.FIFO.value == "fifo"
    assert LotMethod.LIFO.value == "lifo"
    assert LotMethod.HIFO.value == "hifo"


def test_lot_empty_id_rejected():
    with pytest.raises(ValueError):
        _lot(lot_id="")


def test_lot_empty_symbol_rejected():
    with pytest.raises(ValueError):
        TaxLot(
            lot_id="L1",
            symbol="",
            quantity=1.0,
            cost_basis_per_share=10.0,
            acquisition_date=date.today(),
        )


def test_lot_zero_quantity_rejected():
    with pytest.raises(ValueError):
        _lot(qty=0.0)


def test_lot_negative_basis_rejected():
    with pytest.raises(ValueError):
        _lot(basis=-1.0)


def test_lot_immutable():
    lot = _lot()
    with pytest.raises(AttributeError):
        lot.quantity = 0.0  # type: ignore[misc]


def test_realised_slice_pnl():
    s = RealisedSlice(
        lot_id="L1",
        quantity=10,
        cost_basis_per_share=50,
        proceeds_per_share=60,
        acquisition_date=date(2024, 1, 1),
        sale_date=date(2024, 6, 1),
    )
    assert s.realised_pnl == 100  # 10 * (60-50)


def test_realised_slice_long_term_boundary():
    """Held > 365 days → long-term. Exactly 365 → short."""
    short = RealisedSlice(
        lot_id="L1",
        quantity=1,
        cost_basis_per_share=50,
        proceeds_per_share=60,
        acquisition_date=date(2024, 1, 1),
        sale_date=date(2025, 1, 1),  # exactly 366 days? No: 2024-01-01 to 2025-01-01 = 366
    )
    # 2024 was leap year → 2024-01-01 + 365d = 2024-12-31; +366d = 2025-01-01
    # Our rule: > 365 → long. 366 days > 365 → long.
    assert short.is_long_term

    just_under = RealisedSlice(
        lot_id="L1",
        quantity=1,
        cost_basis_per_share=50,
        proceeds_per_share=60,
        acquisition_date=date(2024, 1, 1),
        sale_date=date(2024, 12, 31),  # 365 days exactly
    )
    assert just_under.is_long_term is False


# --- FIFO ----------------------------------------------------------------


def test_fifo_sells_oldest_first():
    pool = (
        _lot("OLD", qty=100, basis=50, acq=date(2024, 1, 1)),
        _lot("NEW", qty=100, basis=70, acq=date(2025, 1, 1)),
    )
    realised, remaining = apply_sale(
        pool,
        quantity=100,
        proceeds_per_share=80,
        sale_date=date(2025, 6, 1),
        method=LotMethod.FIFO,
    )
    assert len(realised) == 1
    assert realised[0].lot_id == "OLD"
    assert remaining[0].lot_id == "NEW"


def test_fifo_partial_sale_keeps_remainder():
    pool = (_lot("L1", qty=100, basis=50),)
    realised, remaining = apply_sale(
        pool, quantity=30, proceeds_per_share=60, sale_date=date(2025, 1, 1)
    )
    assert realised[0].quantity == 30
    assert remaining[0].quantity == 70


def test_fifo_walks_multiple_lots():
    pool = (
        _lot("A", qty=50, basis=50, acq=date(2024, 1, 1)),
        _lot("B", qty=50, basis=60, acq=date(2024, 6, 1)),
        _lot("C", qty=50, basis=70, acq=date(2025, 1, 1)),
    )
    realised, remaining = apply_sale(
        pool, quantity=80, proceeds_per_share=80, sale_date=date(2025, 6, 1)
    )
    # Sells A in full + 30 of B
    assert len(realised) == 2
    assert realised[0].lot_id == "A"
    assert realised[0].quantity == 50
    assert realised[1].lot_id == "B"
    assert realised[1].quantity == 30
    # Remaining: 20 of B + 50 of C
    assert len(remaining) == 2
    qtys = {l.lot_id: l.quantity for l in remaining}
    assert qtys["B"] == 20
    assert qtys["C"] == 50


# --- LIFO ----------------------------------------------------------------


def test_lifo_sells_newest_first():
    pool = (
        _lot("OLD", qty=100, basis=50, acq=date(2024, 1, 1)),
        _lot("NEW", qty=100, basis=70, acq=date(2025, 1, 1)),
    )
    realised, remaining = apply_sale(
        pool,
        quantity=100,
        proceeds_per_share=80,
        sale_date=date(2025, 6, 1),
        method=LotMethod.LIFO,
    )
    assert realised[0].lot_id == "NEW"
    assert remaining[0].lot_id == "OLD"


# --- HIFO ----------------------------------------------------------------


def test_hifo_sells_highest_basis_first():
    pool = (
        _lot("LOW", qty=100, basis=30, acq=date(2024, 1, 1)),
        _lot("HIGH", qty=100, basis=80, acq=date(2025, 1, 1)),
        _lot("MID", qty=100, basis=50, acq=date(2024, 6, 1)),
    )
    realised, remaining = apply_sale(
        pool,
        quantity=100,
        proceeds_per_share=70,
        sale_date=date(2025, 6, 1),
        method=LotMethod.HIFO,
    )
    # HIFO sells HIGH first (basis 80) → realised loss
    assert realised[0].lot_id == "HIGH"
    assert realised[0].realised_pnl == pytest.approx(-1000.0)  # (70-80)*100


def test_hifo_minimises_gain_among_methods():
    pool = (
        _lot("LOW", qty=100, basis=30, acq=date(2024, 1, 1)),
        _lot("HIGH", qty=100, basis=80, acq=date(2025, 1, 1)),
    )

    def total_pnl(method: LotMethod) -> float:
        realised, _ = apply_sale(
            pool,
            quantity=50,
            proceeds_per_share=70,
            sale_date=date(2025, 6, 1),
            method=method,
        )
        return total_realised_pnl(realised)

    fifo_pnl = total_pnl(LotMethod.FIFO)
    lifo_pnl = total_pnl(LotMethod.LIFO)
    hifo_pnl = total_pnl(LotMethod.HIFO)
    # HIFO should produce the smallest (most negative or smallest positive) gain
    assert hifo_pnl <= lifo_pnl
    assert hifo_pnl <= fifo_pnl


# --- Edge cases -----------------------------------------------------------


def test_sale_quantity_exceeds_pool_rejected():
    pool = (_lot("L1", qty=10),)
    with pytest.raises(ValueError):
        apply_sale(pool, quantity=100, proceeds_per_share=50, sale_date=date(2025, 1, 1))


def test_sale_zero_quantity_rejected():
    pool = (_lot("L1", qty=10),)
    with pytest.raises(ValueError):
        apply_sale(pool, quantity=0, proceeds_per_share=50, sale_date=date(2025, 1, 1))


def test_sale_negative_proceeds_rejected():
    pool = (_lot("L1", qty=10),)
    with pytest.raises(ValueError):
        apply_sale(pool, quantity=5, proceeds_per_share=-1, sale_date=date(2025, 1, 1))


def test_sale_full_pool_returns_empty_remaining():
    pool = (_lot("L1", qty=10), _lot("L2", qty=20, acq=date(2024, 6, 1)))
    realised, remaining = apply_sale(
        pool, quantity=30, proceeds_per_share=50, sale_date=date(2025, 1, 1)
    )
    assert remaining == ()
    assert sum(s.quantity for s in realised) == 30


def test_total_quantity_matches_sum():
    pool = (_lot("A", qty=10), _lot("B", qty=20, acq=date(2024, 6, 1)))
    assert total_quantity(pool) == 30


def test_total_cost_basis_matches():
    pool = (_lot("A", qty=10, basis=50), _lot("B", qty=20, basis=60, acq=date(2024, 6, 1)))
    assert total_cost_basis(pool) == 10 * 50 + 20 * 60


def test_split_long_short():
    slices = (
        RealisedSlice(
            lot_id="A",
            quantity=10,
            cost_basis_per_share=50,
            proceeds_per_share=60,
            acquisition_date=date(2024, 1, 1),
            sale_date=date(2025, 6, 1),  # >365d → long
        ),
        RealisedSlice(
            lot_id="B",
            quantity=10,
            cost_basis_per_share=50,
            proceeds_per_share=60,
            acquisition_date=date(2025, 1, 1),
            sale_date=date(2025, 6, 1),  # ~150d → short
        ),
    )
    longs, shorts = split_long_short(slices)
    assert len(longs) == 1
    assert len(shorts) == 1
    assert longs[0].lot_id == "A"
    assert shorts[0].lot_id == "B"


# --- Render --------------------------------------------------------------


def test_render_empty_pool():
    assert "empty" in render_pool(())


def test_render_pool_lists_lots():
    pool = (_lot("L1"), _lot("L2", acq=date(2024, 6, 1)))
    out = render_pool(pool)
    assert "L1" in out
    assert "L2" in out
    assert "AAPL" in out


def test_render_realisation_includes_pnl():
    realised, _ = apply_sale(
        (_lot("L1"),),
        quantity=50,
        proceeds_per_share=60,
        sale_date=date(2025, 6, 1),
    )
    out = render_realisation(realised)
    assert "L1" in out
    assert "$" in out


def test_render_realisation_marks_long_term():
    realised, _ = apply_sale(
        (_lot("L1", acq=date(2023, 1, 1)),),
        quantity=50,
        proceeds_per_share=60,
        sale_date=date(2025, 6, 1),
    )
    out = render_realisation(realised)
    assert "LT" in out


def test_render_no_secret_leak():
    pool = (_lot("L1"),)
    out = render_pool(pool)
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


# --- E2E -----------------------------------------------------------------


def test_e2e_buy_three_lots_sell_one():
    pool = (
        _lot("L1", qty=100, basis=50, acq=date(2024, 1, 1)),
        _lot("L2", qty=100, basis=60, acq=date(2024, 6, 1)),
        _lot("L3", qty=100, basis=70, acq=date(2025, 1, 1)),
    )
    realised_fifo, _ = apply_sale(
        pool, quantity=100, proceeds_per_share=80, sale_date=date(2025, 6, 1), method=LotMethod.FIFO
    )
    realised_hifo, _ = apply_sale(
        pool, quantity=100, proceeds_per_share=80, sale_date=date(2025, 6, 1), method=LotMethod.HIFO
    )
    # FIFO sold L1 ($30 gain/share = $3000)
    assert total_realised_pnl(realised_fifo) == pytest.approx(3000.0)
    # HIFO sold L3 ($10 gain/share = $1000)
    assert total_realised_pnl(realised_hifo) == pytest.approx(1000.0)


def test_replay_consistency():
    pool = (_lot("L1"),)
    a = apply_sale(pool, quantity=10, proceeds_per_share=60, sale_date=date(2025, 1, 1))
    b = apply_sale(pool, quantity=10, proceeds_per_share=60, sale_date=date(2025, 1, 1))
    assert a == b
