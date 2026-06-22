"""Tests for the shared halal no-short invariant."""

from __future__ import annotations

from halal_trader.core.long_only import clamp_sell_to_long


def test_requested_below_available_passes_through():
    r = clamp_sell_to_long(5.0, 10.0)
    assert r.quantity == 5.0
    assert not r.clamped
    assert not r.blocked


def test_requested_above_available_clamps_to_holding():
    r = clamp_sell_to_long(15.0, 10.0)
    assert r.quantity == 10.0  # at most go flat
    assert r.clamped
    assert not r.blocked


def test_requested_equal_available_not_clamped():
    r = clamp_sell_to_long(10.0, 10.0)
    assert r.quantity == 10.0
    assert not r.clamped
    assert not r.blocked


def test_nothing_held_is_blocked():
    r = clamp_sell_to_long(5.0, 0.0)
    assert r.quantity == 0.0
    assert r.blocked
    # nothing to sell → effectively reduced from the request
    assert r.clamped


def test_negative_available_treated_as_zero_blocked():
    # A negative broker qty (already short / bad data) must never authorise a sell.
    r = clamp_sell_to_long(5.0, -3.0)
    assert r.quantity == 0.0
    assert r.blocked


def test_negative_request_floored_to_zero():
    r = clamp_sell_to_long(-5.0, 10.0)
    assert r.quantity == 0.0
    assert not r.clamped  # nothing was requested to begin with
    assert not r.blocked  # holding exists


def test_never_returns_negative_quantity():
    for req in (-10.0, -1.0, 0.0, 1.0, 100.0):
        for avail in (-5.0, 0.0, 1.0, 50.0):
            r = clamp_sell_to_long(req, avail)
            assert r.quantity >= 0.0
            assert r.quantity <= max(avail, 0.0)  # can never exceed the holding
