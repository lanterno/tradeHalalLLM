"""Tests for halal/salam_forward.py — Round-5 Wave 4.C."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.salam_forward import (
    CounterpartyOffer,
    FungibleClass,
    HedgeIntent,
    HedgeRequest,
    match_counterparties,
    plan_salam,
    render_match,
    render_plan,
)


def _req(
    request_id: str = "R1",
    party_id: str = "alice",
    asset_class: FungibleClass = FungibleClass.PRECIOUS_METAL,
    asset_symbol: str = "XAU",
    quantity: float = 100.0,
    spot: float = 2000.0,
    delivery: date = date(2026, 9, 1),
    request_date: date = date(2026, 6, 1),
    intent: HedgeIntent = HedgeIntent.DOWNSIDE_PROTECTION,
    risk: float = 0.5,
) -> HedgeRequest:
    return HedgeRequest(
        request_id=request_id,
        party_id=party_id,
        asset_class=asset_class,
        asset_symbol=asset_symbol,
        quantity=quantity,
        spot_price=spot,
        delivery_date=delivery,
        request_date=request_date,
        intent=intent,
        risk_tolerance=risk,
    )


def _off(
    offer_id: str = "O1",
    party_id: str = "bob",
    asset_class: FungibleClass = FungibleClass.PRECIOUS_METAL,
    asset_symbol: str = "XAU",
    max_qty: float = 200.0,
    earliest: date = date(2026, 8, 1),
    latest: date = date(2026, 12, 1),
    offer_date: date = date(2026, 6, 1),
    discount: float = 0.05,
) -> CounterpartyOffer:
    return CounterpartyOffer(
        offer_id=offer_id,
        party_id=party_id,
        asset_class=asset_class,
        asset_symbol=asset_symbol,
        max_quantity=max_qty,
        earliest_delivery=earliest,
        latest_delivery=latest,
        offer_date=offer_date,
        discount_rate=discount,
    )


# --- HedgeRequest validation ----------------------------------------------


def test_request_valid():
    r = _req()
    assert r.intent is HedgeIntent.DOWNSIDE_PROTECTION


def test_request_empty_id_rejected():
    with pytest.raises(ValueError):
        _req(request_id="")


def test_request_empty_party_rejected():
    with pytest.raises(ValueError):
        _req(party_id="")


def test_request_negative_quantity_rejected():
    with pytest.raises(ValueError):
        _req(quantity=-1.0)


def test_request_zero_spot_rejected():
    with pytest.raises(ValueError):
        _req(spot=0.0)


def test_request_delivery_before_request_rejected():
    with pytest.raises(ValueError):
        _req(delivery=date(2026, 5, 1), request_date=date(2026, 6, 1))


def test_request_tenor_over_365_rejected():
    """AAOIFI Standard 10 cl. 4.4: tenors > 12 months exceed standard."""
    with pytest.raises(ValueError):
        _req(delivery=date(2027, 7, 1), request_date=date(2026, 6, 1))


def test_request_invalid_risk_tolerance():
    with pytest.raises(ValueError):
        _req(risk=-0.1)
    with pytest.raises(ValueError):
        _req(risk=1.5)


def test_request_immutable():
    r = _req()
    with pytest.raises(AttributeError):
        r.quantity = 50.0  # type: ignore[misc]


# --- CounterpartyOffer validation -----------------------------------------


def test_offer_valid():
    o = _off()
    assert o.discount_rate == 0.05


def test_offer_earliest_after_latest_rejected():
    with pytest.raises(ValueError):
        _off(earliest=date(2026, 12, 1), latest=date(2026, 8, 1))


def test_offer_earliest_before_offer_date_rejected():
    with pytest.raises(ValueError):
        _off(earliest=date(2026, 6, 1), offer_date=date(2026, 6, 1))


def test_offer_unreasonable_discount_rejected():
    with pytest.raises(ValueError):
        _off(discount=0.99)


# --- Fungibility constraint via FungibleClass -----------------------------


def test_fungible_class_excludes_equity():
    """Pinned: equities are not in FungibleClass; the type system
    prevents Salam contracts on stocks. Operators must use Wa'd-puts."""
    members = {m.value for m in FungibleClass}
    assert "equity" not in members
    assert "stock" not in members


def test_fungible_class_includes_grain_metal_currency():
    members = {m.value for m in FungibleClass}
    assert "grain" in members
    assert "precious_metal" in members
    assert "currency" in members


# --- plan_salam -----------------------------------------------------------


def test_plan_basic_arithmetic():
    """Pin: prepayment = quantity × (spot - spot×discount) = qty × spot × (1 - d)."""
    r = _req(quantity=10.0, spot=100.0)
    o = _off(discount=0.10)
    plan = plan_salam(r, o)
    assert abs(plan.prepayment_amount - 10.0 * 100.0 * 0.90) < 1e-9
    assert abs(plan.discount_applied - 100.0 * 0.10) < 1e-9


def test_plan_full_prepayment_pin():
    """AAOIFI Standard 10 cl. 3.1 — Salam is fully prepaid."""
    plan = plan_salam(_req(), _off())
    assert plan.is_full_prepayment()


def test_plan_pnl_positive_when_price_falls():
    """Pin: hedge profits when delivery price < spot."""
    r = _req(spot=100.0, quantity=10.0)
    o = _off(discount=0.05)
    plan = plan_salam(r, o, expected_price_at_delivery=80.0)
    # prepayment = 10 × 95 = 950; cost-to-deliver at 80 = 800. P&L = +150.
    assert plan.expected_pnl_at_delivery == pytest.approx(150.0)


def test_plan_pnl_negative_when_price_rises():
    r = _req(spot=100.0, quantity=10.0)
    o = _off(discount=0.05)
    plan = plan_salam(r, o, expected_price_at_delivery=120.0)
    # prepayment = 950; cost = 1200. P&L = -250.
    assert plan.expected_pnl_at_delivery == pytest.approx(-250.0)


def test_plan_asset_class_mismatch_rejected():
    r = _req(asset_class=FungibleClass.GRAIN)
    o = _off(asset_class=FungibleClass.PRECIOUS_METAL)
    with pytest.raises(ValueError):
        plan_salam(r, o)


def test_plan_asset_symbol_mismatch_rejected():
    r = _req(asset_symbol="XAU")
    o = _off(asset_symbol="XAG")
    with pytest.raises(ValueError):
        plan_salam(r, o)


def test_plan_quantity_exceeds_offer_rejected():
    r = _req(quantity=300.0)
    o = _off(max_qty=200.0)
    with pytest.raises(ValueError):
        plan_salam(r, o)


def test_plan_delivery_outside_offer_window_rejected():
    r = _req(delivery=date(2026, 7, 1), request_date=date(2026, 6, 1))
    o = _off(earliest=date(2026, 8, 1), latest=date(2026, 12, 1))
    with pytest.raises(ValueError):
        plan_salam(r, o)


# --- match_counterparties -------------------------------------------------


def test_match_simple_pair():
    res = match_counterparties([_req()], [_off()])
    assert len(res.plans) == 1
    assert not res.unmatched_requests
    assert not res.unmatched_offers
    assert res.fully_matched()


def test_match_multiple_requests_oldest_first():
    """Pin FIFO — oldest request gets matched first."""
    r_old = _req(request_id="R-old", request_date=date(2026, 5, 1), quantity=100.0)
    r_new = _req(request_id="R-new", request_date=date(2026, 6, 1), quantity=100.0)
    o = _off(max_qty=100.0)  # only enough for one
    res = match_counterparties([r_new, r_old], [o])
    assert len(res.plans) == 1
    assert res.plans[0].request_id == "R-old"
    assert len(res.unmatched_requests) == 1
    assert res.unmatched_requests[0].request_id == "R-new"


def test_match_no_compatible_offer():
    r = _req(asset_class=FungibleClass.GRAIN, asset_symbol="WHEAT")
    o = _off(asset_class=FungibleClass.PRECIOUS_METAL, asset_symbol="XAU")
    res = match_counterparties([r], [o])
    assert not res.plans
    assert len(res.unmatched_requests) == 1
    assert len(res.unmatched_offers) == 1


def test_match_offer_consumed_once():
    r1 = _req(request_id="R1", quantity=50.0)
    r2 = _req(request_id="R2", quantity=50.0, request_date=date(2026, 6, 2))
    o = _off(max_qty=200.0)
    res = match_counterparties([r1, r2], [o])
    # FIFO: r1 takes the offer; r2 is unmatched (offers aren't split).
    assert len(res.plans) == 1
    assert res.plans[0].request_id == "R1"
    assert len(res.unmatched_requests) == 1


def test_match_window_filtering():
    r = _req(delivery=date(2026, 9, 1), request_date=date(2026, 6, 1))
    o_window = _off(earliest=date(2026, 8, 1), latest=date(2026, 12, 1))
    o_outside = _off(
        offer_id="O2",
        earliest=date(2026, 6, 15),
        latest=date(2026, 7, 15),
    )
    res = match_counterparties([r], [o_outside, o_window])
    assert len(res.plans) == 1
    assert res.plans[0].offer_id == "O1"


def test_match_empty():
    res = match_counterparties([], [])
    assert res.fully_matched()
    assert not res.plans


# --- render ---------------------------------------------------------------


def test_render_plan_contains_summary():
    r = _req()
    o = _off()
    plan = plan_salam(r, o, expected_price_at_delivery=1900.0)
    out = render_plan(plan)
    assert "📑" in out
    assert "Salam" in out
    assert "Prepayment" in out


def test_render_match_no_secret_leak():
    """Pin: render output masks party_id; raw IDs do not appear."""
    r = _req(party_id="alice@example.com")
    o = _off(party_id="bob@example.com")
    res = match_counterparties([r], [o])
    out = render_match(res)
    assert "@example.com" not in out
    assert "alice@example.com" not in out


def test_render_match_unmatched_path():
    r = _req()
    res = match_counterparties([r], [])
    out = render_match(res)
    assert "Unmatched requests" in out


def test_render_match_empty():
    res = match_counterparties([], [])
    out = render_match(res)
    assert "0 matched" in out
