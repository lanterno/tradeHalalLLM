"""Tests for the Alpaca-bars adapter helpers.

Covers ``bars_to_klines`` edge cases (already covered in
``test_trading_risk.py`` for the basic path) plus the new
``compute_indicators_by_symbol`` helper that runs the full
parse+indicators pipeline over a multi-symbol bars payload.
"""

from __future__ import annotations

from typing import Any

from halal_trader.trading.bars import (
    bars_to_klines,
    compute_indicators_by_symbol,
    extract_last_price,
)


def _bar(o: float, h: float, low: float, c: float, v: float = 1_000.0) -> dict[str, Any]:
    return {"o": o, "h": h, "l": low, "c": c, "v": v}


def _series(start: float, n: int, step: float = 0.5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    price = start
    for _ in range(n):
        out.append(_bar(price, price + 0.5, price - 0.5, price + step))
        price += step
    return out


def test_compute_indicators_by_symbol_empty_payload():
    klines, indicators = compute_indicators_by_symbol({})
    assert klines == {}
    assert indicators == {}


def test_compute_indicators_by_symbol_skips_unparseable_symbols():
    """A symbol with all-zero closes is dropped — no klines, no indicators."""
    payload = {
        "AAPL": _series(100, 30),
        "BAD": [_bar(0, 0, 0, 0)],
    }
    klines, indicators = compute_indicators_by_symbol(payload)
    assert "AAPL" in klines
    assert "AAPL" in indicators
    assert "BAD" not in klines
    assert "BAD" not in indicators


def test_compute_indicators_by_symbol_returns_indicator_keys():
    """Successful path produces the indicator vector ``compute_all`` emits."""
    payload = {"AAPL": _series(100, 50)}
    klines, indicators = compute_indicators_by_symbol(payload)
    assert len(klines["AAPL"]) == 50
    aapl_ind = indicators["AAPL"]
    # Spot-check a few well-known keys from compute_all.
    assert "rsi_14" in aapl_ind
    assert "current_price" in aapl_ind
    assert "vwap" in aapl_ind


def test_compute_indicators_by_symbol_handles_alpaca_envelope():
    """Alpaca's ``{"bars": [...]}`` shape should round-trip identically."""
    series = _series(50, 30)
    payload = {"MSFT": {"bars": series}}
    klines, indicators = compute_indicators_by_symbol(payload)
    assert len(klines["MSFT"]) == 30
    assert "rsi_14" in indicators["MSFT"]


def test_compute_indicators_by_symbol_real_cycle_envelope():
    """REGRESSION (integration): the live cycle builds {sym: get_stock_bars(sym)},
    where each value is the symbol-keyed {"bars": {sym: [...]}} envelope. This is
    what feeds the risk engine, regime detection, ML inference, and the LLM prompt
    — it must yield REAL indicators, not the empty dict the old bars_to_klines
    silently produced (which left the stock bot trading with no technicals)."""
    payload = {"NVDA": {"bars": {"NVDA": _series(100, 40)}}}
    klines, indicators = compute_indicators_by_symbol(payload)
    assert len(klines["NVDA"]) == 40
    assert "rsi_14" in indicators["NVDA"] and "atr_14" in indicators["NVDA"]


def test_bars_to_klines_handles_symbol_keyed_envelope():
    """REGRESSION: get_stock_bars actually returns {"bars": {SYMBOL: [...]}} —
    a symbol-keyed level inside the envelope. This previously parsed to an empty
    list, silently starving the monitor's trend-break SMA, ML snapshots, and the
    multi-timeframe analyzer. bars_to_klines must flatten the symbol level."""
    payload = {"bars": {"NVDA": _series(100, 40)}}
    klines = bars_to_klines(payload)
    assert len(klines) == 40
    assert klines[0].close > 0


def test_bars_to_klines_handles_double_nested_mcp_envelope():
    """REGRESSION (2026-07-01 live): the Alpaca MCP server now double-nests
    bars under a security wrapper: {"_alpaca_mcp_security": {...},
    "data": {"bars": {SYMBOL: [...]}}}. The one-level peel yielded ZERO klines,
    silently starving indicators/ML and failing the daily recommendation with
    'no candidate market data available'. Must unwrap data → bars → symbol."""
    payload = {
        "_alpaca_mcp_security": {"trust": "untrusted_tool_output"},
        "data": {"bars": {"NVDA": _series(100, 40)}},
    }
    klines = bars_to_klines(payload)
    assert len(klines) == 40
    assert klines[-1].close > 0


def test_bars_to_klines_handles_data_wrapped_flat_list():
    """A {"data": {"bars": [...]}} (non-symbol-keyed) variant also round-trips."""
    payload = {"data": {"bars": _series(50, 30)}}
    assert len(bars_to_klines(payload)) == 30


def test_bars_to_klines_synthetic_timestamps_are_monotonic():
    """Downstream consumers rely on ordered open_time even though the value is synthetic."""
    klines = bars_to_klines(_series(100, 5))
    times = [k.open_time for k in klines]
    assert times == sorted(times)
    assert all(t >= 0 for t in times)


# ── extract_last_price ──────────────────────────────────────────


def test_extract_last_price_flat_payload():
    snap = {"latestTrade": {"p": 150.5}}
    assert extract_last_price(snap, "AAPL") == 150.5


def test_extract_last_price_nested_by_symbol():
    snap = {"AAPL": {"latestTrade": {"p": 150.5}}}
    assert extract_last_price(snap, "AAPL") == 150.5


def test_extract_last_price_alt_keys():
    snap = {"latest_trade": {"price": 200.0}}
    assert extract_last_price(snap, "AAPL") == 200.0


def test_extract_last_price_missing_returns_none():
    assert extract_last_price({"foo": "bar"}, "AAPL") is None
    assert extract_last_price({}, "AAPL") is None
    assert extract_last_price("not a dict", "AAPL") is None


def test_extract_last_price_unparseable_returns_none():
    snap = {"latestTrade": {"p": "not a number"}}
    assert extract_last_price(snap, "AAPL") is None
