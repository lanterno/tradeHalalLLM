"""Tests for marketplace/halal_secondary.py — Round-5 Wave 6.F."""

from __future__ import annotations

from datetime import datetime

import pytest

from halal_trader.marketplace.halal_secondary import (
    BookPolicy,
    CounterpartyHalalError,
    FairValueMark,
    Order,
    OrderSide,
    OrderStatus,
    PriceBandViolation,
    Trade,
    assert_price_in_band,
    cancel_order,
    match_book,
    post_order,
    render_order,
    render_trade,
)


def _always_halal(_: str) -> bool:
    return True


def _fv(asset_id: str = "STARTUP-A", price: float = 100.0) -> FairValueMark:
    return FairValueMark(asset_id=asset_id, price=price, marked_at=datetime(2026, 5, 1))


# --- BookPolicy validation -------------------------------


def test_policy_default():
    p = BookPolicy()
    assert p.lower_band_pct == 0.85


def test_policy_invalid_lower_band():
    with pytest.raises(ValueError):
        BookPolicy(lower_band_pct=1.5)


def test_policy_invalid_upper_band():
    with pytest.raises(ValueError):
        BookPolicy(upper_band_pct=5.0)


def test_policy_invalid_min_quantity():
    with pytest.raises(ValueError):
        BookPolicy(min_quantity=0.0)


# --- FairValueMark validation --------------------------


def test_fair_value_valid():
    f = _fv()
    assert f.price == 100.0


def test_fair_value_negative_rejected():
    with pytest.raises(ValueError):
        FairValueMark(asset_id="A", price=-1.0, marked_at=datetime(2026, 5, 1))


# --- assert_price_in_band ----------------------------


def test_price_in_band_passes():
    assert_price_in_band(100.0, 100.0, policy=BookPolicy())


def test_price_at_lower_band_passes():
    assert_price_in_band(85.0, 100.0, policy=BookPolicy())


def test_price_at_upper_band_passes():
    assert_price_in_band(115.0, 100.0, policy=BookPolicy())


def test_price_below_band_rejected():
    with pytest.raises(PriceBandViolation):
        assert_price_in_band(80.0, 100.0, policy=BookPolicy())


def test_price_above_band_rejected():
    with pytest.raises(PriceBandViolation):
        assert_price_in_band(125.0, 100.0, policy=BookPolicy())


def test_zero_fair_value_rejected():
    with pytest.raises(ValueError):
        assert_price_in_band(100.0, 0.0, policy=BookPolicy())


# --- Order validation --------------------------------


def _order(
    order_id: str = "O1",
    asset_id: str = "STARTUP-A",
    side: OrderSide = OrderSide.SELL,
    user_id: str = "alice",
    quantity: float = 100.0,
    filled_quantity: float = 0.0,
    limit_price: float = 100.0,
    posted_at: datetime = datetime(2026, 5, 1, 10, 0),
    status: OrderStatus = OrderStatus.OPEN,
) -> Order:
    return Order(
        order_id=order_id,
        asset_id=asset_id,
        side=side,
        user_id=user_id,
        quantity=quantity,
        filled_quantity=filled_quantity,
        limit_price=limit_price,
        posted_at=posted_at,
        status=status,
    )


def test_order_valid():
    o = _order()
    assert o.remaining() == 100.0


def test_order_empty_id_rejected():
    with pytest.raises(ValueError):
        _order(order_id="")


def test_order_zero_quantity_rejected():
    with pytest.raises(ValueError):
        _order(quantity=0)


def test_order_negative_filled_rejected():
    with pytest.raises(ValueError):
        _order(filled_quantity=-1.0)


def test_order_filled_above_quantity_rejected():
    with pytest.raises(ValueError):
        _order(filled_quantity=200.0)


def test_order_filled_status_must_match():
    with pytest.raises(ValueError):
        _order(filled_quantity=50.0, status=OrderStatus.FILLED)


def test_order_partial_status_must_match():
    with pytest.raises(ValueError):
        _order(filled_quantity=0.0, status=OrderStatus.PARTIALLY_FILLED)


def test_order_immutable():
    o = _order()
    with pytest.raises(AttributeError):
        o.quantity = 0  # type: ignore[misc]


# --- post_order --------------------------------------


def test_post_clean_order():
    o = post_order(
        order_id="O1",
        asset_id="STARTUP-A",
        side=OrderSide.SELL,
        user_id="alice",
        quantity=100.0,
        limit_price=95.0,
        posted_at=datetime(2026, 5, 1, 10, 0),
        fair_value=_fv(),
        is_asset_halal=_always_halal,
    )
    assert o.status is OrderStatus.OPEN


def test_post_haram_asset_rejected():
    with pytest.raises(ValueError):
        post_order(
            order_id="O1",
            asset_id="HARAM-CO",
            side=OrderSide.BUY,
            user_id="alice",
            quantity=100.0,
            limit_price=100.0,
            posted_at=datetime(2026, 5, 1),
            fair_value=_fv(asset_id="HARAM-CO"),
            is_asset_halal=lambda a: a != "HARAM-CO",
        )


def test_post_outside_band_rejected():
    with pytest.raises(PriceBandViolation):
        post_order(
            order_id="O1",
            asset_id="STARTUP-A",
            side=OrderSide.SELL,
            user_id="alice",
            quantity=100.0,
            limit_price=200.0,
            posted_at=datetime(2026, 5, 1),
            fair_value=_fv(),
            is_asset_halal=_always_halal,
        )


def test_post_below_min_quantity_rejected():
    with pytest.raises(ValueError):
        post_order(
            order_id="O1",
            asset_id="STARTUP-A",
            side=OrderSide.SELL,
            user_id="alice",
            quantity=0.001,
            limit_price=100.0,
            posted_at=datetime(2026, 5, 1),
            fair_value=_fv(),
            is_asset_halal=_always_halal,
            policy=BookPolicy(min_quantity=0.01),
        )


def test_post_asset_mismatch_rejected():
    with pytest.raises(ValueError):
        post_order(
            order_id="O1",
            asset_id="STARTUP-A",
            side=OrderSide.SELL,
            user_id="alice",
            quantity=100.0,
            limit_price=100.0,
            posted_at=datetime(2026, 5, 1),
            fair_value=_fv(asset_id="STARTUP-B"),
            is_asset_halal=_always_halal,
        )


# --- match_book — basic ----------------------------


def test_match_single_pair():
    sell = _order(
        order_id="S1",
        side=OrderSide.SELL,
        user_id="alice",
        quantity=100.0,
        limit_price=100.0,
        posted_at=datetime(2026, 5, 1, 9, 0),
    )
    buy = _order(
        order_id="B1",
        side=OrderSide.BUY,
        user_id="bob",
        quantity=100.0,
        limit_price=105.0,
        posted_at=datetime(2026, 5, 1, 10, 0),
    )
    trades, updated = match_book(
        [sell, buy],
        matched_at=datetime(2026, 5, 1, 11, 0),
        is_counterparty_halal=_always_halal,
    )
    assert len(trades) == 1
    assert trades[0].quantity == 100.0
    # Sell is older → match at sell limit (100).
    assert trades[0].price == 100.0


def test_match_no_cross_when_buy_below_sell():
    sell = _order(order_id="S1", side=OrderSide.SELL, user_id="alice", limit_price=110.0)
    buy = _order(
        order_id="B1",
        side=OrderSide.BUY,
        user_id="bob",
        limit_price=100.0,
        posted_at=datetime(2026, 5, 1, 10, 0),
    )
    trades, _ = match_book(
        [sell, buy],
        matched_at=datetime(2026, 5, 1, 11, 0),
        is_counterparty_halal=_always_halal,
    )
    assert trades == ()


def test_match_partial_fill_status():
    sell = _order(
        order_id="S1",
        side=OrderSide.SELL,
        user_id="alice",
        quantity=100.0,
        limit_price=100.0,
    )
    buy = _order(
        order_id="B1",
        side=OrderSide.BUY,
        user_id="bob",
        quantity=60.0,
        limit_price=105.0,
        posted_at=datetime(2026, 5, 1, 10, 0),
    )
    trades, updated = match_book(
        [sell, buy],
        matched_at=datetime(2026, 5, 1, 11, 0),
        is_counterparty_halal=_always_halal,
    )
    assert len(trades) == 1
    assert trades[0].quantity == 60.0
    by_id = {o.order_id: o for o in updated}
    assert by_id["S1"].status is OrderStatus.PARTIALLY_FILLED
    assert by_id["B1"].status is OrderStatus.FILLED


def test_match_no_self_cross():
    sell = _order(order_id="S1", side=OrderSide.SELL, user_id="alice", limit_price=100.0)
    buy = _order(
        order_id="B1",
        side=OrderSide.BUY,
        user_id="alice",
        limit_price=110.0,
        posted_at=datetime(2026, 5, 1, 10, 0),
    )
    trades, _ = match_book(
        [sell, buy],
        matched_at=datetime(2026, 5, 1, 11, 0),
        is_counterparty_halal=_always_halal,
    )
    assert trades == ()


def test_match_different_assets_dont_cross():
    sell = _order(
        order_id="S1",
        asset_id="A",
        side=OrderSide.SELL,
        user_id="alice",
        limit_price=100.0,
    )
    buy = _order(
        order_id="B1",
        asset_id="B",
        side=OrderSide.BUY,
        user_id="bob",
        limit_price=105.0,
        posted_at=datetime(2026, 5, 1, 10, 0),
    )
    trades, _ = match_book(
        [sell, buy],
        matched_at=datetime(2026, 5, 1, 11, 0),
        is_counterparty_halal=_always_halal,
    )
    assert trades == ()


# --- match_book — counterparty halal -----------------


def test_match_buyer_not_halal_raises():
    sell = _order(order_id="S1", side=OrderSide.SELL, user_id="alice", limit_price=100.0)
    buy = _order(
        order_id="B1",
        side=OrderSide.BUY,
        user_id="mallory",
        limit_price=105.0,
        posted_at=datetime(2026, 5, 1, 10, 0),
    )
    with pytest.raises(CounterpartyHalalError):
        match_book(
            [sell, buy],
            matched_at=datetime(2026, 5, 1, 11, 0),
            is_counterparty_halal=lambda u: u != "mallory",
        )


# --- match_book — priority --------------------------


def test_match_best_price_first():
    """Higher buy limit should match first."""
    cheap_buy = _order(
        order_id="B1",
        side=OrderSide.BUY,
        user_id="bob",
        limit_price=100.0,
        posted_at=datetime(2026, 5, 1, 10, 0),
    )
    expensive_buy = _order(
        order_id="B2",
        side=OrderSide.BUY,
        user_id="charlie",
        limit_price=110.0,
        posted_at=datetime(2026, 5, 1, 10, 5),
    )
    sell = _order(
        order_id="S1",
        side=OrderSide.SELL,
        user_id="alice",
        quantity=50.0,
        limit_price=95.0,
        posted_at=datetime(2026, 5, 1, 9, 0),
    )
    trades, _ = match_book(
        [cheap_buy, expensive_buy, sell],
        matched_at=datetime(2026, 5, 1, 11, 0),
        is_counterparty_halal=_always_halal,
    )
    assert trades[0].buyer_id == "charlie"


def test_match_time_priority_breaks_price_tie():
    sell1 = _order(
        order_id="S1",
        side=OrderSide.SELL,
        user_id="alice",
        quantity=50.0,
        limit_price=100.0,
        posted_at=datetime(2026, 5, 1, 9, 0),
    )
    sell2 = _order(
        order_id="S2",
        side=OrderSide.SELL,
        user_id="alice",
        quantity=50.0,
        limit_price=100.0,
        posted_at=datetime(2026, 5, 1, 9, 30),
    )
    buy = _order(
        order_id="B1",
        side=OrderSide.BUY,
        user_id="bob",
        quantity=50.0,
        limit_price=105.0,
        posted_at=datetime(2026, 5, 1, 10, 0),
    )
    trades, _ = match_book(
        [sell1, sell2, buy],
        matched_at=datetime(2026, 5, 1, 11, 0),
        is_counterparty_halal=_always_halal,
    )
    assert trades[0].sell_order_id == "S1"  # earlier sell wins on tie


def test_match_resting_order_wins_price():
    """Pin: older order's price is used."""
    sell = _order(
        order_id="S1",
        side=OrderSide.SELL,
        user_id="alice",
        limit_price=95.0,
        posted_at=datetime(2026, 5, 1, 9, 0),
    )
    buy = _order(
        order_id="B1",
        side=OrderSide.BUY,
        user_id="bob",
        limit_price=110.0,
        posted_at=datetime(2026, 5, 1, 10, 0),
    )
    trades, _ = match_book(
        [sell, buy],
        matched_at=datetime(2026, 5, 1, 11, 0),
        is_counterparty_halal=_always_halal,
    )
    # Sell (older) → match at 95.
    assert trades[0].price == 95.0


# --- cancel_order ---------------------------------


def test_cancel_open():
    o = _order()
    cancelled = cancel_order(o)
    assert cancelled.status is OrderStatus.CANCELLED


def test_cancel_filled_rejected():
    o = _order(
        filled_quantity=100.0,
        status=OrderStatus.FILLED,
    )
    with pytest.raises(ValueError):
        cancel_order(o)


def test_cancel_already_cancelled_rejected():
    cancelled = cancel_order(_order())
    with pytest.raises(ValueError):
        cancel_order(cancelled)


# --- Render ---------------------------------------


def test_render_order_no_secret_leak():
    o = _order(user_id="alice@example.com")
    out = render_order(o)
    assert "alice@example.com" not in out


def test_render_order_side_emoji():
    buy = _order(side=OrderSide.BUY)
    sell = _order(side=OrderSide.SELL)
    assert "🟢" in render_order(buy)
    assert "🔴" in render_order(sell)


def test_render_trade_format():
    t = Trade(
        trade_id="T1",
        asset_id="A",
        buy_order_id="B1",
        sell_order_id="S1",
        buyer_id="bob",
        seller_id="alice",
        quantity=50.0,
        price=100.0,
        matched_at=datetime(2026, 5, 1, 11, 0),
    )
    out = render_trade(t)
    assert "T1" in out
    assert "🤝" in out
