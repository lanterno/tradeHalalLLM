"""Tests for halal/musharakah_coinvest.py — Round-5 Wave 6.C."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.musharakah_coinvest import (
    CoInvestmentDeal,
    Commitment,
    DealStatus,
    add_commitment,
    call_capital,
    distribute,
    liquidate,
    render_deal,
    render_distribution,
)


def _commit(
    cid: str = "C1",
    investor: str = "alice",
    amount: float = 100_000.0,
    committed_at: date = date(2026, 6, 1),
    funded: float = 0.0,
) -> Commitment:
    return Commitment(
        commitment_id=cid,
        investor_id=investor,
        amount_usd=amount,
        committed_at=committed_at,
        funded_usd=funded,
    )


def _deal(
    deal_id: str = "D1",
    sponsor: str = "platform",
    soft: float = 500_000.0,
    hard: float = 2_000_000.0,
    target: float = 1_000_000.0,
    open_date: date = date(2026, 6, 1),
    commitments: tuple[Commitment, ...] | None = None,
) -> CoInvestmentDeal:
    if commitments is None:
        commitments = ()
    return CoInvestmentDeal(
        deal_id=deal_id,
        sponsor_id=sponsor,
        soft_cap_usd=soft,
        hard_cap_usd=hard,
        target_raise_usd=target,
        open_date=open_date,
        commitments=commitments,
    )


# --- Commitment validation -----------------------------------------------


def test_commitment_valid():
    c = _commit()
    assert c.amount_usd == 100_000.0


def test_commitment_empty_id_rejected():
    with pytest.raises(ValueError):
        _commit(cid="")


def test_commitment_zero_amount_rejected():
    with pytest.raises(ValueError):
        _commit(amount=0)


def test_commitment_funded_above_amount_rejected():
    with pytest.raises(ValueError):
        _commit(amount=100.0, funded=200.0)


# --- CoInvestmentDeal validation ----------------------------------------


def test_deal_valid():
    d = _deal()
    assert d.computed_status() is DealStatus.OPEN


def test_deal_hard_below_soft_rejected():
    with pytest.raises(ValueError):
        _deal(soft=1_000_000, hard=500_000)


def test_deal_target_outside_band_rejected():
    with pytest.raises(ValueError):
        _deal(soft=500_000, target=200_000, hard=2_000_000)


def test_deal_duplicate_commitment_id_rejected():
    c1 = _commit(cid="C1")
    c2 = _commit(cid="C1", investor="bob")
    with pytest.raises(ValueError):
        _deal(commitments=(c1, c2))


# --- computed_status -----------------------------------------------------


def test_status_open_when_below_soft():
    d = _deal(commitments=(_commit(amount=100_000),))
    assert d.computed_status() is DealStatus.OPEN


def test_status_soft_circled_when_at_soft():
    c = _commit(amount=500_000)
    d = _deal(commitments=(c,))
    assert d.computed_status() is DealStatus.SOFT_CIRCLED


def test_status_hard_closed_when_at_hard():
    c = _commit(amount=2_000_000)
    d = _deal(commitments=(c,))
    assert d.computed_status() is DealStatus.HARD_CLOSED


# --- add_commitment ------------------------------------------------------


def test_add_commitment_basic():
    d = _deal()
    new_d = add_commitment(d, _commit(cid="C1", amount=100_000))
    assert len(new_d.commitments) == 1


def test_add_commitment_chains():
    d = _deal()
    d = add_commitment(d, _commit(cid="C1", amount=100_000, investor="alice"))
    d = add_commitment(d, _commit(cid="C2", amount=200_000, investor="bob"))
    assert len(d.commitments) == 2


def test_add_commitment_above_hard_cap_rejected():
    d = _deal(hard=1_000_000)
    d = add_commitment(d, _commit(cid="C1", amount=900_000))
    with pytest.raises(ValueError):
        add_commitment(d, _commit(cid="C2", amount=200_000))


def test_add_commitment_at_hard_cap_closes_deal():
    d = _deal(hard=1_000_000)
    d = add_commitment(d, _commit(cid="C1", amount=1_000_000))
    assert d.computed_status() is DealStatus.HARD_CLOSED


def test_add_commitment_to_hard_closed_rejected():
    d = _deal(hard=1_000_000)
    d = add_commitment(d, _commit(cid="C1", amount=1_000_000))
    with pytest.raises(ValueError):
        add_commitment(d, _commit(cid="C2", amount=10_000))


def test_add_commitment_duplicate_id_rejected():
    d = _deal()
    d = add_commitment(d, _commit(cid="C1"))
    with pytest.raises(ValueError):
        add_commitment(d, _commit(cid="C1", investor="bob"))


# --- call_capital --------------------------------------------------------


def test_call_capital_fifo_order():
    """Pin: oldest commitment funded first."""
    d = _deal()
    d = add_commitment(
        d, _commit(cid="C1", investor="alice", amount=100_000, committed_at=date(2026, 6, 1))
    )
    d = add_commitment(
        d, _commit(cid="C2", investor="bob", amount=100_000, committed_at=date(2026, 6, 5))
    )
    d = call_capital(d, amount_usd=80_000)
    by_id = {c.commitment_id: c for c in d.commitments}
    assert by_id["C1"].funded_usd == 80_000.0
    assert by_id["C2"].funded_usd == 0.0


def test_call_capital_spans_multiple_commitments():
    d = _deal()
    d = add_commitment(d, _commit(cid="C1", amount=100_000, committed_at=date(2026, 6, 1)))
    d = add_commitment(
        d, _commit(cid="C2", investor="bob", amount=100_000, committed_at=date(2026, 6, 5))
    )
    d = call_capital(d, amount_usd=150_000)
    by_id = {c.commitment_id: c for c in d.commitments}
    assert by_id["C1"].funded_usd == 100_000.0
    assert by_id["C2"].funded_usd == 50_000.0


def test_call_capital_above_uncalled_rejected():
    d = _deal()
    d = add_commitment(d, _commit(amount=100_000))
    with pytest.raises(ValueError):
        call_capital(d, amount_usd=200_000)


def test_call_capital_zero_rejected():
    d = _deal()
    d = add_commitment(d, _commit(amount=100_000))
    with pytest.raises(ValueError):
        call_capital(d, amount_usd=0)


def test_uncalled_capital_helper():
    d = _deal()
    d = add_commitment(d, _commit(amount=100_000))
    d = call_capital(d, amount_usd=30_000)
    assert d.uncalled_capital() == 70_000.0


# --- distribute ----------------------------------------------------------


def test_distribute_proportional_to_funded():
    d = _deal()
    d = add_commitment(
        d, _commit(cid="C1", investor="alice", amount=200_000, committed_at=date(2026, 6, 1))
    )
    d = add_commitment(
        d, _commit(cid="C2", investor="bob", amount=100_000, committed_at=date(2026, 6, 2))
    )
    d = call_capital(d, amount_usd=300_000)  # full
    records = distribute(d, proceeds=30_000.0)
    by_inv = {r.investor_id: r for r in records}
    assert by_inv["alice"].proceeds == pytest.approx(20_000.0)
    assert by_inv["bob"].proceeds == pytest.approx(10_000.0)


def test_distribute_loss_borne_per_capital_share():
    """Pin: loss share == capital share (Standard 12)."""
    d = _deal()
    d = add_commitment(
        d, _commit(cid="C1", investor="alice", amount=200_000, committed_at=date(2026, 6, 1))
    )
    d = add_commitment(
        d, _commit(cid="C2", investor="bob", amount=100_000, committed_at=date(2026, 6, 2))
    )
    d = call_capital(d, amount_usd=300_000)
    records = distribute(d, proceeds=-30_000.0)
    by_inv = {r.investor_id: r for r in records}
    assert by_inv["alice"].proceeds == pytest.approx(-20_000.0)
    assert by_inv["bob"].proceeds == pytest.approx(-10_000.0)


def test_distribute_uncalled_does_not_share():
    """Pin: committed-but-uncalled capital does not earn or lose."""
    d = _deal()
    d = add_commitment(
        d, _commit(cid="C1", investor="alice", amount=200_000, committed_at=date(2026, 6, 1))
    )
    d = add_commitment(
        d, _commit(cid="C2", investor="bob", amount=200_000, committed_at=date(2026, 6, 2))
    )
    d = call_capital(d, amount_usd=200_000)  # only alice funded
    records = distribute(d, proceeds=20_000.0)
    by_inv = {r.investor_id: r for r in records}
    assert by_inv["alice"].proceeds == pytest.approx(20_000.0)
    assert by_inv["bob"].proceeds == pytest.approx(0.0)


def test_distribute_no_funded_rejected():
    d = _deal()
    d = add_commitment(d, _commit())
    with pytest.raises(ValueError):
        distribute(d, proceeds=10_000.0)


# --- liquidate -----------------------------------------------------------


def test_liquidate_returns_capital_plus_pnl():
    d = _deal()
    d = add_commitment(d, _commit(amount=100_000))
    d = call_capital(d, amount_usd=100_000)
    new_d, records = liquidate(d, realised_pnl=20_000.0, close_date=date(2026, 12, 1))
    assert new_d.status is DealStatus.LIQUIDATED
    # Single investor → gets back capital + full PnL.
    assert records[0].proceeds == pytest.approx(120_000.0)


def test_liquidate_loss_path():
    d = _deal()
    d = add_commitment(d, _commit(amount=100_000))
    d = call_capital(d, amount_usd=100_000)
    new_d, records = liquidate(d, realised_pnl=-30_000.0, close_date=date(2026, 12, 1))
    # Loss → 70k back.
    assert records[0].proceeds == pytest.approx(70_000.0)


def test_liquidate_locks_status():
    d = _deal()
    d = add_commitment(d, _commit(amount=100_000))
    d = call_capital(d, amount_usd=100_000)
    new_d, _ = liquidate(d, realised_pnl=10_000.0, close_date=date(2026, 12, 1))
    assert new_d.computed_status() is DealStatus.LIQUIDATED


def test_liquidated_cannot_accept_new_commitments():
    d = _deal()
    d = add_commitment(d, _commit(amount=100_000))
    d = call_capital(d, amount_usd=100_000)
    new_d, _ = liquidate(d, realised_pnl=10_000.0, close_date=date(2026, 12, 1))
    with pytest.raises(ValueError):
        add_commitment(new_d, _commit(cid="C2", investor="bob", amount=10_000))


# --- Render --------------------------------------------------------------


def test_render_deal_no_secret_leak():
    d = _deal(sponsor="platform@example.com")
    d = add_commitment(d, _commit(investor="alice@example.com", amount=100_000))
    out = render_deal(d)
    assert "alice@example.com" not in out
    assert "platform@example.com" not in out


def test_render_distribution_no_records():
    out = render_distribution([])
    assert "No distribution" in out


def test_render_distribution_lists_each():
    d = _deal()
    d = add_commitment(d, _commit(amount=100_000))
    d = call_capital(d, amount_usd=100_000)
    records = distribute(d, proceeds=10_000.0)
    out = render_distribution(records)
    assert "Distribution" in out
    assert "share=" in out
