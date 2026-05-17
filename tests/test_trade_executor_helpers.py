"""Tests for `TradeExecutor._extract_price`.

This is the snapshot → price helper feeding the buying-power check
in `_execute_buy` — a regression here would let an LLM-suggested
buy slip through without a properly estimated cost (under-spend
check fails open). It has tiered fallback (latest_trade → daily_bar
→ 0.0) and several defensive type checks worth pinning.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from halal_trader.trading.executor import TradeExecutor


def _executor() -> TradeExecutor:
    """Construct an executor with stubbed deps — `_extract_price` doesn't
    touch any of them."""
    return TradeExecutor(
        broker=MagicMock(),
        repo=MagicMock(),
        max_position_pct=0.10,
        max_simultaneous_positions=10,
    )


# ── nested-by-symbol shape ─────────────────────────────────


def test_extract_price_nested_latest_trade():
    """The standard Alpaca snapshot shape — `{symbol: {latest_trade: {price}}}`."""
    snap = {"AAPL": {"latest_trade": {"price": 150.0}}}
    assert _executor()._extract_price(snap, "AAPL") == 150.0


def test_extract_price_returns_float_for_int_input():
    """Defensive: a snapshot may carry int prices (some APIs round) —
    the helper coerces to float so downstream `× quantity` is float."""
    snap = {"AAPL": {"latest_trade": {"price": 150}}}
    out = _executor()._extract_price(snap, "AAPL")
    assert out == 150.0
    assert isinstance(out, float)


def test_extract_price_falls_back_to_daily_bar_close():
    """When `latest_trade.price` is absent, fall back to `daily_bar.close`
    — covers stale-market hours / pre-open snapshots where Alpaca only
    surfaces yesterday's daily bar."""
    snap = {"AAPL": {"daily_bar": {"close": 148.5}}}
    assert _executor()._extract_price(snap, "AAPL") == 148.5


def test_extract_price_prefers_latest_trade_over_daily_bar():
    """When both are present, `latest_trade` wins (it's fresher)."""
    snap = {"AAPL": {"latest_trade": {"price": 150.0}, "daily_bar": {"close": 148.0}}}
    assert _executor()._extract_price(snap, "AAPL") == 150.0


def test_extract_price_zero_latest_trade_falls_through_to_bar():
    """`price=0` is treated as falsy → fall through to the bar.
    Defensive against a market-closed snapshot where the trade
    payload exists but is zeroed."""
    snap = {"AAPL": {"latest_trade": {"price": 0}, "daily_bar": {"close": 148.0}}}
    assert _executor()._extract_price(snap, "AAPL") == 148.0


def test_extract_price_zero_bar_close_falls_through_to_default():
    """Both zero → return 0.0 (the executor will then refuse the buy
    because `cost = 0 × qty = 0` triggers the buying-power check
    against `account.buying_power > 0`, so we never under-spend)."""
    snap = {"AAPL": {"latest_trade": {"price": 0}, "daily_bar": {"close": 0}}}
    assert _executor()._extract_price(snap, "AAPL") == 0.0


# ── flat (no symbol-nesting) shape ─────────────────────────


def test_extract_price_flat_snapshot_uses_root_dict():
    """When the snapshot is the symbol's data directly (caller already
    indexed by symbol), the `snapshot.get(symbol, snapshot)` fallback
    treats the root as the data dict."""
    flat = {"latest_trade": {"price": 150.0}}
    assert _executor()._extract_price(flat, "AAPL") == 150.0


# ── unknown-symbol / wrong-shape defensive ─────────────────


def test_extract_price_unknown_symbol_returns_zero():
    """An empty snapshot dict (or one without the symbol key) → 0.0."""
    assert _executor()._extract_price({}, "AAPL") == 0.0


def test_extract_price_non_dict_snapshot_returns_zero():
    """A list / string / None snapshot → 0.0 (executor refuses the
    buy on insufficient buying power downstream)."""
    e = _executor()
    assert e._extract_price([], "AAPL") == 0.0
    assert e._extract_price("not a dict", "AAPL") == 0.0
    assert e._extract_price(None, "AAPL") == 0.0


def test_extract_price_non_dict_per_symbol_data_returns_zero():
    """`snapshot[symbol]` is a non-dict value (e.g. raw price as int) →
    skip, return 0.0. Pin so a future API shape change doesn't silently
    bypass the price extraction."""
    snap = {"AAPL": 150}  # raw price, not nested
    assert _executor()._extract_price(snap, "AAPL") == 0.0


def test_extract_price_non_dict_trade_skipped():
    """`latest_trade` exists but isn't a dict → fall through (don't
    attempt `.get` on a list)."""
    snap = {"AAPL": {"latest_trade": "not a dict", "daily_bar": {"close": 148.0}}}
    assert _executor()._extract_price(snap, "AAPL") == 148.0


def test_extract_price_non_dict_bar_skipped():
    """`daily_bar` isn't a dict → return 0.0."""
    snap = {"AAPL": {"daily_bar": [148, 149]}}
    assert _executor()._extract_price(snap, "AAPL") == 0.0


def test_extract_price_missing_price_field_in_trade():
    """`latest_trade` dict exists but `price` key is absent → fall
    through to bar."""
    snap = {"AAPL": {"latest_trade": {}, "daily_bar": {"close": 148.0}}}
    assert _executor()._extract_price(snap, "AAPL") == 148.0


def test_extract_price_missing_close_field_in_bar():
    """`daily_bar` dict exists but `close` key is absent → 0.0."""
    snap = {"AAPL": {"daily_bar": {}}}
    assert _executor()._extract_price(snap, "AAPL") == 0.0


# ── multi-symbol disambiguation ────────────────────────────


def test_extract_price_finds_correct_symbol_in_multi_symbol_snapshot():
    """Multi-symbol snapshots route to the right symbol's data."""
    snap = {
        "AAPL": {"latest_trade": {"price": 150.0}},
        "MSFT": {"latest_trade": {"price": 410.0}},
    }
    e = _executor()
    assert e._extract_price(snap, "AAPL") == 150.0
    assert e._extract_price(snap, "MSFT") == 410.0


def test_extract_price_unknown_symbol_in_populated_snapshot_returns_zero():
    """Symbol absent from a populated snapshot → 0.0 (the
    `.get(symbol, snapshot)` fallback hits the snapshot itself, which
    has multiple-symbol keys but no `latest_trade` directly → 0.0)."""
    snap = {
        "AAPL": {"latest_trade": {"price": 150.0}},
        "MSFT": {"latest_trade": {"price": 410.0}},
    }
    assert _executor()._extract_price(snap, "GOOG") == 0.0
