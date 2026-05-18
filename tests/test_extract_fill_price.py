"""Tests for :func:`extract_fill_price` — Binance order-fill price math.

Used downstream by the trade recorder + reconciler. The two paths
(modern `fills` array vs. legacy `executedQty` + `cumulativeQuoteQty`)
must both round-trip cleanly because Binance returns one or the other
depending on order type / API version.
"""

from __future__ import annotations

from halal_trader.crypto.exchange import extract_fill_price

# ── Modern `fills` array ─────────────────────────────────────


def test_returns_avg_when_single_fill():
    result = {"fills": [{"qty": "1.0", "price": "50000.0"}]}
    assert extract_fill_price(result) == 50_000.0


def test_returns_volume_weighted_avg_across_multiple_fills():
    """0.5 @ 50000 + 0.5 @ 51000 → VWAP 50500."""
    result = {
        "fills": [
            {"qty": "0.5", "price": "50000.0"},
            {"qty": "0.5", "price": "51000.0"},
        ]
    }
    assert extract_fill_price(result) == 50_500.0


def test_weighting_skews_toward_larger_fill():
    """0.1 @ 50000 + 0.9 @ 60000 → VWAP closer to 60000."""
    result = {
        "fills": [
            {"qty": "0.1", "price": "50000.0"},
            {"qty": "0.9", "price": "60000.0"},
        ]
    }
    avg = extract_fill_price(result)
    # (5000 + 54000) / 1.0 = 59000.
    assert avg == 59_000.0


# ── Legacy `executedQty` / `cumulativeQuoteQty` fallback ─────


def test_falls_back_to_cumulative_quote_qty():
    """No `fills` array → use executedQty + cumulativeQuoteQty."""
    result = {"executedQty": "2.0", "cumulativeQuoteQty": "100000.0"}
    assert extract_fill_price(result) == 50_000.0


def test_fills_array_takes_priority_over_cumulative():
    """If both shapes are present, the per-fill array wins (more accurate)."""
    result = {
        "fills": [{"qty": "1.0", "price": "50000.0"}],
        "executedQty": "1.0",
        "cumulativeQuoteQty": "999.0",  # would give wildly different answer
    }
    assert extract_fill_price(result) == 50_000.0


# ── None paths ───────────────────────────────────────────────


def test_returns_none_when_no_fills_and_no_executed_qty():
    """Pending order with no fills → None (caller stays at the
    submitted price rather than recording a bogus 0)."""
    assert extract_fill_price({}) is None


def test_returns_none_when_zero_executed_qty():
    """Rejected order shows up with executedQty=0 — must not divide-by-zero."""
    result = {"executedQty": "0", "cumulativeQuoteQty": "0"}
    assert extract_fill_price(result) is None


def test_returns_none_when_fills_list_has_zero_qty():
    """Defensive: empty fills sum → no avg."""
    result = {"fills": [{"qty": "0", "price": "50000.0"}]}
    assert extract_fill_price(result) is None


def test_returns_none_when_only_executed_qty_no_cumulative():
    """Partial state where qty is set but quote-qty isn't — defensive
    skip rather than infer a price."""
    result = {"executedQty": "2.0", "cumulativeQuoteQty": "0"}
    assert extract_fill_price(result) is None
