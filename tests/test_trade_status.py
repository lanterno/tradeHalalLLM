"""Tests for :class:`TradeStatus` (the trade-lifecycle StrEnum).

This enum drives terminal-vs-open branching across the executor,
monitor, reconciler, and notifier — a wrong assignment here would
cascade into stale orders or duplicate notifications.
"""

from __future__ import annotations

from halal_trader.domain.status import TradeStatus

# ── String compatibility ─────────────────────────────────────


def test_str_enum_equals_its_string_value():
    """Existing string comparisons in the codebase keep working."""
    assert TradeStatus.FILLED == "filled"
    assert "filled" == TradeStatus.FILLED


def test_value_round_trip():
    for status in TradeStatus:
        assert TradeStatus(status.value) is status


def test_pinned_wire_values():
    """Wire format is load-bearing: DB rows + executor return dicts use
    these exact strings."""
    assert TradeStatus.PENDING == "pending"
    assert TradeStatus.SUBMITTED == "submitted"
    assert TradeStatus.FILLED == "filled"
    assert TradeStatus.PARTIALLY_FILLED == "partially_filled"
    assert TradeStatus.REJECTED == "rejected"
    assert TradeStatus.CANCELED == "canceled"
    assert TradeStatus.CLOSED == "closed"
    assert TradeStatus.ERROR == "error"


# ── is_terminal ──────────────────────────────────────────────


# ── is_open ──────────────────────────────────────────────────


def test_is_open_true_for_pending_submitted_partial():
    for s in (
        TradeStatus.PENDING,
        TradeStatus.SUBMITTED,
        TradeStatus.PARTIALLY_FILLED,
    ):
        assert TradeStatus.is_open(s) is True


def test_is_open_false_for_terminal_states():
    for s in (
        TradeStatus.FILLED,
        TradeStatus.REJECTED,
        TradeStatus.CANCELED,
        TradeStatus.CLOSED,
        TradeStatus.ERROR,
    ):
        assert TradeStatus.is_open(s) is False


def test_is_open_works_with_raw_string():
    assert TradeStatus.is_open("submitted") is True
    assert TradeStatus.is_open("filled") is False


# ── is_terminal + is_open are mutually exclusive ─────────────
