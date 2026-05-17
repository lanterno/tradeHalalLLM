"""Tests for core/tax_my.py — Round-5 Wave 18.E."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.core.tax_my import (
    AssetCategory,
    DisposalEvent,
    TaxClassification,
    TaxPolicy,
    TaxStatus,
    classify,
    classify_batch,
    render_classification,
    total_exempt_dividends,
    total_taxable,
)


def _event(**overrides) -> DisposalEvent:
    base = {
        "event_id": "E-1",
        "asset_category": AssetCategory.LISTED_EQUITY_BURSA,
        "proceeds": 1000.0,
        "cost_basis": 800.0,
        "event_date": date(2026, 5, 1),
        "daily_trade_count": 1,
    }
    base.update(overrides)
    return DisposalEvent(**base)


# --- Validation ----------------------------------------


def test_status_string_values():
    assert TaxStatus.EXEMPT_LISTED.value == "exempt_listed"
    assert TaxStatus.EXEMPT_DIVIDEND.value == "exempt_dividend"
    assert TaxStatus.RPGT_REAL_PROPERTY.value == "rpgt_real_property"
    assert TaxStatus.BUSINESS_INCOME.value == "business_income"
    assert TaxStatus.FOREIGN_TAXABLE.value == "foreign_taxable"


def test_event_empty_id_rejected():
    with pytest.raises(ValueError):
        _event(event_id="")


def test_event_negative_proceeds_rejected():
    with pytest.raises(ValueError):
        _event(proceeds=-1)


def test_event_negative_basis_rejected():
    with pytest.raises(ValueError):
        _event(cost_basis=-1)


def test_event_negative_daily_count_rejected():
    with pytest.raises(ValueError):
        _event(daily_trade_count=-1)


def test_classification_negative_taxable_rejected():
    with pytest.raises(ValueError):
        TaxClassification(
            event_id="x", status=TaxStatus.EXEMPT_LISTED, taxable_amount=-1
        )


def test_default_policy():
    p = TaxPolicy()
    assert p.business_income_trades_per_day == 20


def test_policy_zero_threshold_rejected():
    with pytest.raises(ValueError):
        TaxPolicy(business_income_trades_per_day=0)


# --- Classification ------------------------------------


def test_listed_bursa_exempt():
    c = classify(_event(asset_category=AssetCategory.LISTED_EQUITY_BURSA))
    assert c.status is TaxStatus.EXEMPT_LISTED
    assert c.taxable_amount == 0


def test_dividend_exempt():
    c = classify(_event(asset_category=AssetCategory.DIVIDEND))
    assert c.status is TaxStatus.EXEMPT_DIVIDEND
    assert c.taxable_amount == 0


def test_rpgt_real_property_gain_taxable():
    c = classify(
        _event(
            asset_category=AssetCategory.REAL_PROPERTY_RPGT,
            proceeds=500000,
            cost_basis=400000,
        )
    )
    assert c.status is TaxStatus.RPGT_REAL_PROPERTY
    assert c.taxable_amount == 100000


def test_rpgt_loss_zero_taxable():
    c = classify(
        _event(
            asset_category=AssetCategory.REAL_PROPERTY_RPGT,
            proceeds=400000,
            cost_basis=500000,
        )
    )
    assert c.taxable_amount == 0  # loss, not negative


def test_foreign_equity_taxable():
    c = classify(
        _event(
            asset_category=AssetCategory.LISTED_EQUITY_FOREIGN,
            proceeds=1500,
            cost_basis=1000,
        )
    )
    assert c.status is TaxStatus.FOREIGN_TAXABLE
    assert c.taxable_amount == 500


def test_frequent_trader_classified_as_business():
    c = classify(
        _event(
            asset_category=AssetCategory.LISTED_EQUITY_BURSA,
            daily_trade_count=25,
        )
    )
    assert c.status is TaxStatus.BUSINESS_INCOME
    assert c.taxable_amount == 200


def test_at_threshold_classified_as_business():
    """Exactly at threshold (20) → business income."""
    c = classify(
        _event(
            asset_category=AssetCategory.LISTED_EQUITY_BURSA,
            daily_trade_count=20,
        )
    )
    assert c.status is TaxStatus.BUSINESS_INCOME


def test_below_threshold_classified_as_exempt():
    c = classify(
        _event(
            asset_category=AssetCategory.LISTED_EQUITY_BURSA,
            daily_trade_count=19,
        )
    )
    assert c.status is TaxStatus.EXEMPT_LISTED


def test_other_category_treated_taxable():
    c = classify(_event(asset_category=AssetCategory.OTHER))
    assert c.status is TaxStatus.FOREIGN_TAXABLE
    assert c.taxable_amount == 200


def test_custom_policy_threshold():
    pol = TaxPolicy(business_income_trades_per_day=5)
    c = classify(
        _event(
            asset_category=AssetCategory.LISTED_EQUITY_BURSA,
            daily_trade_count=10,
        ),
        policy=pol,
    )
    assert c.status is TaxStatus.BUSINESS_INCOME


# --- Batch + totals ---------------------------------


def test_classify_batch():
    events = [
        _event(event_id="E1", asset_category=AssetCategory.LISTED_EQUITY_BURSA),
        _event(event_id="E2", asset_category=AssetCategory.DIVIDEND),
    ]
    classes = classify_batch(events)
    assert len(classes) == 2


def test_total_taxable_sums():
    events = [
        _event(event_id="E1", asset_category=AssetCategory.REAL_PROPERTY_RPGT, proceeds=500000, cost_basis=400000),
        _event(event_id="E2", asset_category=AssetCategory.LISTED_EQUITY_FOREIGN, proceeds=1500, cost_basis=1000),
    ]
    classes = classify_batch(events)
    assert total_taxable(classes) == 100000 + 500


def test_total_exempt_dividends():
    events = [
        _event(event_id="D1", asset_category=AssetCategory.DIVIDEND, proceeds=100, cost_basis=0),
        _event(event_id="D2", asset_category=AssetCategory.DIVIDEND, proceeds=200, cost_basis=0),
        _event(event_id="E1", asset_category=AssetCategory.LISTED_EQUITY_BURSA),
    ]
    assert total_exempt_dividends(events) == 300


# --- Render ------------------------------------------


def test_render_includes_status_and_amount():
    c = classify(_event())
    out = render_classification(c)
    assert "E-1" in out
    assert "exempt_listed" in out
    assert "RM" in out


# --- E2E -------------------------------------------


def test_e2e_typical_my_resident_year():
    events = [
        _event(event_id="E1", asset_category=AssetCategory.LISTED_EQUITY_BURSA, daily_trade_count=2),
        _event(event_id="E2", asset_category=AssetCategory.DIVIDEND, proceeds=500, cost_basis=0),
        _event(event_id="E3", asset_category=AssetCategory.REAL_PROPERTY_RPGT, proceeds=600000, cost_basis=500000),
    ]
    classes = classify_batch(events)
    statuses = [c.status for c in classes]
    assert TaxStatus.EXEMPT_LISTED in statuses
    assert TaxStatus.EXEMPT_DIVIDEND in statuses
    assert TaxStatus.RPGT_REAL_PROPERTY in statuses
    # Taxable is from RPGT only
    assert total_taxable(classes) == 100000


def test_replay_consistency():
    a = classify(_event())
    b = classify(_event())
    assert a == b
