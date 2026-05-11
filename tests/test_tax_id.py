"""Tests for core/tax_id.py — Round-5 Wave 18.F."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.core.tax_id import (
    AssetCategory,
    DisposalEvent,
    TaxClassification,
    TaxPolicy,
    TaxStatus,
    classify,
    classify_batch,
    needs_accountant,
    render_classification,
    total_final_tax,
)


def _ev(
    event_id: str = "E1",
    asset_category: AssetCategory = AssetCategory.LISTED_EQUITY_IDX,
    proceeds: float = 1_000_000.0,
    cost_basis: float = 800_000.0,
    event_date: date = date(2026, 5, 1),
    daily_trade_count: int = 0,
    is_founder: bool = False,
    corporate_holder_pct: float | None = None,
) -> DisposalEvent:
    return DisposalEvent(
        event_id=event_id,
        asset_category=asset_category,
        proceeds=proceeds,
        cost_basis=cost_basis,
        event_date=event_date,
        daily_trade_count=daily_trade_count,
        is_founder=is_founder,
        corporate_holder_pct=corporate_holder_pct,
    )


# --- TaxPolicy validation ------------------------------------------------


def test_policy_default():
    p = TaxPolicy()
    assert p.listed_equity_final_rate == 0.001
    assert p.dividend_final_rate == 0.10


def test_policy_invalid_rate():
    with pytest.raises(ValueError):
        TaxPolicy(listed_equity_final_rate=-0.01)
    with pytest.raises(ValueError):
        TaxPolicy(dividend_final_rate=0.99)


def test_policy_invalid_threshold():
    with pytest.raises(ValueError):
        TaxPolicy(business_income_trades_per_day=0)


def test_policy_invalid_corp_threshold():
    with pytest.raises(ValueError):
        TaxPolicy(corporate_dividend_threshold=0.0)
    with pytest.raises(ValueError):
        TaxPolicy(corporate_dividend_threshold=1.5)


# --- DisposalEvent validation --------------------------------------------


def test_event_valid():
    e = _ev()
    assert e.proceeds == 1_000_000.0


def test_event_empty_id_rejected():
    with pytest.raises(ValueError):
        _ev(event_id="")


def test_event_negative_proceeds_rejected():
    with pytest.raises(ValueError):
        _ev(proceeds=-1.0)


def test_event_negative_count_rejected():
    with pytest.raises(ValueError):
        _ev(daily_trade_count=-1)


def test_event_invalid_corp_holder_pct():
    with pytest.raises(ValueError):
        _ev(corporate_holder_pct=1.5)


def test_event_immutable():
    e = _ev()
    with pytest.raises(AttributeError):
        e.proceeds = 0.0  # type: ignore[misc]


# --- Listed-equity classification -----------------------------------------


def test_listed_equity_final_tax_on_proceeds():
    """Pin: 0.1% × proceeds (NOT gain)."""
    e = _ev(proceeds=1_000_000.0, cost_basis=900_000.0)
    c = classify(e)
    assert c.status is TaxStatus.LISTED_EQUITY_FINAL
    assert c.taxable_amount == 1_000_000.0
    assert c.tax_due == pytest.approx(1_000.0)  # 0.1% of 1M


def test_listed_equity_loss_still_taxes_proceeds():
    """Pin: even on a losing trade, the 0.1% is on gross proceeds."""
    e = _ev(proceeds=1_000_000.0, cost_basis=1_500_000.0)
    c = classify(e)
    assert c.status is TaxStatus.LISTED_EQUITY_FINAL
    assert c.tax_due == pytest.approx(1_000.0)


def test_listed_equity_founder_uplift():
    """Pin: founders pay 0.1% + 0.5% = 0.6% on proceeds."""
    e = _ev(proceeds=1_000_000.0, is_founder=True)
    c = classify(e)
    assert c.status is TaxStatus.LISTED_EQUITY_FOUNDER
    assert c.tax_due == pytest.approx(6_000.0)


def test_frequent_trader_flips_to_business_income():
    e = _ev(daily_trade_count=25)
    c = classify(e)
    assert c.status is TaxStatus.BUSINESS_INCOME


def test_business_income_taxable_is_gain():
    """BUSINESS_INCOME applies progressive PPh — taxable_amount is the gain."""
    e = _ev(proceeds=1_000_000.0, cost_basis=800_000.0, daily_trade_count=25)
    c = classify(e)
    assert c.taxable_amount == 200_000.0
    assert c.tax_due == 0.0  # progressive — accountant computes


def test_business_income_loss_yields_zero_taxable():
    """Pin: a losing trade still has taxable_amount = max(0, gain)."""
    e = _ev(proceeds=500_000.0, cost_basis=800_000.0, daily_trade_count=25)
    c = classify(e)
    assert c.taxable_amount == 0.0


# --- Dividend classification ---------------------------------------------


def test_dividend_final_10pct():
    e = _ev(asset_category=AssetCategory.DIVIDEND, proceeds=100_000.0)
    c = classify(e)
    assert c.status is TaxStatus.DIVIDEND_FINAL
    assert c.tax_due == pytest.approx(10_000.0)


def test_dividend_individual_no_corp_holder_pct():
    e = _ev(
        asset_category=AssetCategory.DIVIDEND,
        proceeds=100_000.0,
        corporate_holder_pct=None,
    )
    c = classify(e)
    assert c.status is TaxStatus.DIVIDEND_FINAL


def test_dividend_corporate_above_threshold_exempt():
    e = _ev(
        asset_category=AssetCategory.DIVIDEND,
        proceeds=100_000.0,
        corporate_holder_pct=0.30,
    )
    c = classify(e)
    assert c.status is TaxStatus.DIVIDEND_EXEMPT_CORPORATE
    assert c.tax_due == 0.0


def test_dividend_corporate_below_threshold_taxed():
    """Pin: corporate holder < 25% still owes the 10%."""
    e = _ev(
        asset_category=AssetCategory.DIVIDEND,
        proceeds=100_000.0,
        corporate_holder_pct=0.10,
    )
    c = classify(e)
    assert c.status is TaxStatus.DIVIDEND_FINAL


def test_dividend_corporate_at_threshold_exempt():
    e = _ev(
        asset_category=AssetCategory.DIVIDEND,
        proceeds=100_000.0,
        corporate_holder_pct=0.25,
    )
    c = classify(e)
    assert c.status is TaxStatus.DIVIDEND_EXEMPT_CORPORATE


# --- Bond coupon ---------------------------------------------------------


def test_bond_coupon_10pct():
    e = _ev(asset_category=AssetCategory.BOND_COUPON, proceeds=50_000.0)
    c = classify(e)
    assert c.status is TaxStatus.BOND_COUPON_FINAL
    assert c.tax_due == pytest.approx(5_000.0)


# --- Foreign equity ------------------------------------------------------


def test_foreign_equity_progressive():
    e = _ev(
        asset_category=AssetCategory.LISTED_EQUITY_FOREIGN,
        proceeds=1_000_000.0,
        cost_basis=800_000.0,
    )
    c = classify(e)
    assert c.status is TaxStatus.FOREIGN_PROGRESSIVE
    assert c.taxable_amount == 200_000.0
    assert c.tax_due == 0.0  # progressive — accountant


def test_other_category_progressive():
    e = _ev(
        asset_category=AssetCategory.OTHER,
        proceeds=500_000.0,
        cost_basis=400_000.0,
    )
    c = classify(e)
    assert c.status is TaxStatus.FOREIGN_PROGRESSIVE


# --- Batch + summary ----------------------------------------------------


def test_classify_batch():
    events = [
        _ev(event_id="E1", proceeds=1_000_000.0),
        _ev(event_id="E2", asset_category=AssetCategory.DIVIDEND, proceeds=100_000.0),
    ]
    out = classify_batch(events)
    assert len(out) == 2
    assert out[0].status is TaxStatus.LISTED_EQUITY_FINAL
    assert out[1].status is TaxStatus.DIVIDEND_FINAL


def test_total_final_tax_sums_only_final_buckets():
    classifications = (
        TaxClassification(
            event_id="E1",
            status=TaxStatus.LISTED_EQUITY_FINAL,
            taxable_amount=1_000_000.0,
            tax_due=1_000.0,
        ),
        TaxClassification(
            event_id="E2",
            status=TaxStatus.DIVIDEND_FINAL,
            taxable_amount=100_000.0,
            tax_due=10_000.0,
        ),
        TaxClassification(
            event_id="E3",
            status=TaxStatus.BUSINESS_INCOME,
            taxable_amount=200_000.0,
            tax_due=0.0,
        ),
    )
    assert total_final_tax(classifications) == 11_000.0


def test_needs_accountant_filters():
    classifications = (
        TaxClassification(
            event_id="E1",
            status=TaxStatus.LISTED_EQUITY_FINAL,
            taxable_amount=1_000_000.0,
            tax_due=1_000.0,
        ),
        TaxClassification(
            event_id="E2",
            status=TaxStatus.BUSINESS_INCOME,
            taxable_amount=200_000.0,
            tax_due=0.0,
        ),
        TaxClassification(
            event_id="E3",
            status=TaxStatus.FOREIGN_PROGRESSIVE,
            taxable_amount=300_000.0,
            tax_due=0.0,
        ),
    )
    flagged = needs_accountant(classifications)
    assert len(flagged) == 2
    assert {c.event_id for c in flagged} == {"E2", "E3"}


# --- Render -------------------------------------------------------------


def test_render_classification_format():
    c = TaxClassification(
        event_id="E1",
        status=TaxStatus.LISTED_EQUITY_FINAL,
        taxable_amount=1_000_000.0,
        tax_due=1_000.0,
    )
    out = render_classification(c)
    assert "E1" in out
    assert "Rp" in out
    assert "listed_equity_final" in out
