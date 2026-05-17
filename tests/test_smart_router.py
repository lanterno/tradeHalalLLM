"""Tests for trading/smart_router.py — Round-5 Wave 12.E."""

from __future__ import annotations

import pytest

from halal_trader.trading.smart_router import (
    RouterInputs,
    RoutingMode,
    RoutingPolicy,
    VenueQuote,
    render_decision,
    route,
)
from halal_trader.trading.twap import Side


def _quote(
    venue: str = "NYSE",
    symbol: str = "AAPL",
    bid: float = 99.95,
    ask: float = 100.05,
    qty: float = 1000.0,
    commission: float = 0.0,
) -> VenueQuote:
    return VenueQuote(
        venue=venue,
        symbol=symbol,
        bid_price=bid,
        ask_price=ask,
        available_quantity=qty,
        per_share_commission=commission,
    )


def _inputs(
    side: Side = Side.BUY,
    parent_quantity: float = 500.0,
    quotes: tuple[VenueQuote, ...] = (),
) -> RouterInputs:
    return RouterInputs(
        parent_id="P-001",
        symbol="AAPL",
        side=side,
        parent_quantity=parent_quantity,
        venue_quotes=quotes,
    )


# --- Validation -------------------------------------------------------------


def test_routing_mode_string_values():
    assert RoutingMode.BEST_PRICE.value == "best_price"
    assert RoutingMode.SPLIT.value == "split"
    assert RoutingMode.VENUE_PINNED.value == "venue_pinned"


def test_quote_empty_venue_rejected():
    with pytest.raises(ValueError):
        _quote(venue="")


def test_quote_negative_price_rejected():
    with pytest.raises(ValueError):
        _quote(bid=-1.0)


def test_quote_crossed_rejected():
    with pytest.raises(ValueError):
        _quote(bid=101.0, ask=100.0)


def test_quote_negative_qty_rejected():
    with pytest.raises(ValueError):
        _quote(qty=-1.0)


def test_quote_effective_price_buy_includes_commission():
    q = _quote(ask=100.0, commission=0.05)
    assert q.effective_price(Side.BUY) == 100.05


def test_quote_effective_price_sell_subtracts_commission():
    q = _quote(bid=99.95, commission=0.05)
    assert q.effective_price(Side.SELL) == 99.90


def test_policy_pinned_without_venue_rejected():
    with pytest.raises(ValueError):
        RoutingPolicy(mode=RoutingMode.VENUE_PINNED)


def test_policy_pinned_with_empty_venue_rejected():
    with pytest.raises(ValueError):
        RoutingPolicy(mode=RoutingMode.VENUE_PINNED, pinned_venue=" ")


def test_inputs_symbol_mismatch_rejected():
    with pytest.raises(ValueError):
        RouterInputs(
            parent_id="P",
            symbol="AAPL",
            side=Side.BUY,
            parent_quantity=100,
            venue_quotes=(_quote(symbol="MSFT"),),
        )


def test_inputs_zero_qty_rejected():
    with pytest.raises(ValueError):
        _inputs(parent_quantity=0)


# --- Best-price routing ----------------------------------------------------


def test_best_price_picks_cheapest_buy_venue():
    quotes = (
        _quote(venue="NYSE", ask=100.05, qty=10000),
        _quote(venue="EDGX", ask=100.02, qty=10000),
        _quote(venue="ARCA", ask=100.04, qty=10000),
    )
    decision = route(_inputs(quotes=quotes))
    assert len(decision.allocations) == 1
    assert decision.allocations[0].venue == "EDGX"
    assert decision.allocations[0].expected_price == pytest.approx(100.02)


def test_best_price_picks_highest_sell_venue():
    quotes = (
        _quote(venue="NYSE", bid=99.95, qty=10000),
        _quote(venue="EDGX", bid=99.98, qty=10000),
        _quote(venue="ARCA", bid=99.96, qty=10000),
    )
    decision = route(_inputs(side=Side.SELL, quotes=quotes))
    assert decision.allocations[0].venue == "EDGX"


def test_best_price_includes_commission_in_decision():
    quotes = (
        _quote(venue="A", ask=100.00, commission=0.10, qty=10000),  # eff 100.10
        _quote(venue="B", ask=100.05, commission=0.00, qty=10000),  # eff 100.05
    )
    decision = route(_inputs(quotes=quotes))
    assert decision.allocations[0].venue == "B"


def test_best_price_unallocated_when_no_capacity():
    quotes = (_quote(venue="A", qty=0),)
    decision = route(_inputs(parent_quantity=500, quotes=quotes))
    assert decision.allocations == ()
    assert decision.unallocated_quantity == 500


def test_best_price_partial_unallocated_when_venue_smaller_than_parent():
    quotes = (_quote(venue="A", qty=200),)
    decision = route(_inputs(parent_quantity=500, quotes=quotes))
    assert decision.allocations[0].quantity == 200
    assert decision.unallocated_quantity == 300


def test_best_price_empty_quotes():
    decision = route(_inputs(parent_quantity=100, quotes=()))
    assert decision.allocations == ()
    assert decision.unallocated_quantity == 100


# --- Split routing --------------------------------------------------------


def test_split_routes_across_venues_in_price_order():
    quotes = (
        _quote(venue="A", ask=100.10, qty=200),  # most expensive
        _quote(venue="B", ask=100.02, qty=200),  # cheapest
        _quote(venue="C", ask=100.05, qty=200),  # middle
    )
    decision = route(
        _inputs(parent_quantity=500, quotes=quotes),
        policy=RoutingPolicy(mode=RoutingMode.SPLIT),
    )
    venues = [a.venue for a in decision.allocations]
    assert venues == ["B", "C", "A"]


def test_split_total_quantity_matches():
    quotes = (
        _quote(venue="A", ask=100.10, qty=200),
        _quote(venue="B", ask=100.02, qty=200),
        _quote(venue="C", ask=100.05, qty=200),
    )
    decision = route(
        _inputs(parent_quantity=500, quotes=quotes),
        policy=RoutingPolicy(mode=RoutingMode.SPLIT),
    )
    assert decision.total_allocated() == 500


def test_split_unallocated_when_total_capacity_low():
    quotes = (_quote(venue="A", qty=100), _quote(venue="B", qty=100))
    decision = route(
        _inputs(parent_quantity=500, quotes=quotes),
        policy=RoutingPolicy(mode=RoutingMode.SPLIT),
    )
    assert decision.total_allocated() == 200
    assert decision.unallocated_quantity == 300


def test_split_skips_zero_capacity_venues():
    quotes = (
        _quote(venue="A", ask=100.02, qty=0),
        _quote(venue="B", ask=100.05, qty=500),
    )
    decision = route(
        _inputs(parent_quantity=500, quotes=quotes),
        policy=RoutingPolicy(mode=RoutingMode.SPLIT),
    )
    venues = [a.venue for a in decision.allocations]
    assert "A" not in venues
    assert "B" in venues


# --- Venue-pinned routing -------------------------------------------------


def test_venue_pinned_sends_to_pinned_venue():
    quotes = (
        _quote(venue="A", ask=100.02, qty=10000),  # cheaper but ignored
        _quote(venue="B", ask=100.10, qty=10000),
    )
    decision = route(
        _inputs(quotes=quotes),
        policy=RoutingPolicy(mode=RoutingMode.VENUE_PINNED, pinned_venue="B"),
    )
    assert decision.allocations[0].venue == "B"


def test_venue_pinned_unknown_venue_unallocated():
    quotes = (_quote(venue="A", qty=10000),)
    decision = route(
        _inputs(parent_quantity=500, quotes=quotes),
        policy=RoutingPolicy(mode=RoutingMode.VENUE_PINNED, pinned_venue="NEVER"),
    )
    assert decision.allocations == ()
    assert decision.unallocated_quantity == 500


def test_venue_pinned_zero_capacity_unallocated():
    quotes = (_quote(venue="B", qty=0),)
    decision = route(
        _inputs(parent_quantity=500, quotes=quotes),
        policy=RoutingPolicy(mode=RoutingMode.VENUE_PINNED, pinned_venue="B"),
    )
    assert decision.allocations == ()


# --- Render --------------------------------------------------------------


def test_render_decision_includes_summary():
    quotes = (_quote(venue="NYSE", qty=1000),)
    decision = route(_inputs(quotes=quotes))
    out = render_decision(decision)
    assert "Smart-router" in out
    assert "P-001" in out


def test_render_decision_marks_unallocated():
    decision = route(_inputs(parent_quantity=500, quotes=()))
    assert "UNALLOCATED" in render_decision(decision)


def test_render_no_secret_leak():
    quotes = (_quote(venue="NYSE", qty=1000),)
    decision = route(_inputs(quotes=quotes))
    out = render_decision(decision)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ---------------------------------------------------------------


def test_e2e_split_500_across_3_venues_total_matches():
    quotes = (
        _quote(venue="NYSE", ask=100.05, qty=200),
        _quote(venue="EDGX", ask=100.02, qty=200),
        _quote(venue="ARCA", ask=100.04, qty=200),
    )
    decision = route(
        _inputs(parent_quantity=500, quotes=quotes),
        policy=RoutingPolicy(mode=RoutingMode.SPLIT),
    )
    assert decision.total_allocated() == 500
    assert decision.unallocated_quantity == 0


def test_replay_consistency():
    quotes = (_quote(venue="NYSE", qty=1000),)
    a = route(_inputs(quotes=quotes))
    b = route(_inputs(quotes=quotes))
    assert a == b
