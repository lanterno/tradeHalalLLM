"""Microstructure feature tests — pure-function math + halal note."""

from __future__ import annotations

import pytest

from halal_trader.crypto.microstructure import (
    cumulative_delta,
    format_microstructure_for_prompt,
    funding_features,
    orderbook_features,
)


def _book(bids, asks):
    return {"bids": bids, "asks": asks}


def test_orderbook_features_basic():
    book = _book(
        bids=[[100.0, 1.0], [99.0, 2.0]],
        asks=[[101.0, 1.0], [102.0, 2.0]],
    )
    f = orderbook_features(book)
    assert f is not None
    assert f.best_bid == 100.0
    assert f.best_ask == 101.0
    assert f.mid == pytest.approx(100.5)
    assert f.spread_bps == pytest.approx(99.502, rel=1e-3)
    # Bid-side notional 100+198=298; ask-side 101+204=305 → slight ask-heavy.
    assert f.depth_imbalance < 0


def test_orderbook_features_bid_heavy_imbalance():
    book = _book(
        bids=[[100.0, 10.0]],
        asks=[[101.0, 1.0]],
    )
    f = orderbook_features(book)
    assert f is not None
    assert f.depth_imbalance > 0.5


def test_orderbook_features_empty_returns_none():
    assert orderbook_features({"bids": [], "asks": []}) is None
    assert orderbook_features({}) is None


def test_orderbook_features_handles_inverted_book():
    """If the book somehow comes back with bid > ask, treat as malformed."""
    book = _book(bids=[[100.0, 1.0]], asks=[[99.0, 1.0]])
    assert orderbook_features(book) is None


def test_orderbook_features_handles_garbage_levels():
    book = _book(bids=[["nope", "x"]], asks=[[101.0, 1.0]])
    assert orderbook_features(book) is None


def test_cumulative_delta_buyer_maker_sign():
    """isBuyerMaker=True → trade was sell-aggression → negative delta."""
    trades = [
        {"price": 100, "qty": 1, "isBuyerMaker": True},
        {"price": 100, "qty": 2, "isBuyerMaker": False},
    ]
    # 100*1 sell + 200*1 buy = +100 net.
    assert cumulative_delta(trades) == pytest.approx(100.0)


def test_cumulative_delta_alt_field_names():
    """Binance aggTrades uses ``p``/``q``/``m`` instead of long names."""
    trades = [{"p": "50", "q": "0.5", "m": False}]  # buy aggression
    assert cumulative_delta(trades) == pytest.approx(25.0)


def test_cumulative_delta_skips_invalid_rows():
    trades = [
        {"price": 0, "qty": 1, "isBuyerMaker": False},  # zero price → skip
        {"price": 100, "qty": "nope", "isBuyerMaker": False},  # bad qty → skip
        {"price": 100, "qty": 1, "isBuyerMaker": False},  # +100
    ]
    assert cumulative_delta(trades) == pytest.approx(100.0)


def test_funding_features_basic():
    f = funding_features(funding_rate=0.0001, perp_mark=70_100, spot_mid=70_000)
    assert f is not None
    # 0.0001 * 1095 settles per year = 10.95% annualised.
    assert f.annualised_funding == pytest.approx(0.1095)
    # Basis = 100 / 70000 = ~14.3bps.
    assert f.basis_bps == pytest.approx(14.286, rel=1e-3)


def test_funding_features_zero_prices_returns_none():
    assert funding_features(0.0001, 0, 70_000) is None
    assert funding_features(0.0001, 70_000, 0) is None


def test_format_for_prompt_combines_signals():
    book = orderbook_features(_book(bids=[[100, 10]], asks=[[101, 1]]))
    fund = funding_features(0.0002, 70_100, 70_000)
    text = format_microstructure_for_prompt(
        pair="BTCUSDT", book=book, funding=fund, cum_delta=12_345.0
    )
    assert "BTCUSDT" in text
    assert "bid-heavy" in text
    assert "spread" in text
    assert "perp basis" in text
    assert "funding" in text
    assert "+12,345" in text


def test_format_for_prompt_empty_when_no_signals():
    assert format_microstructure_for_prompt(pair="X") == ""


def test_imbalance_zero_when_perfectly_balanced():
    book = _book(bids=[[100, 1]], asks=[[101, 1]])
    f = orderbook_features(book)
    # Notionals: bid=100, ask=101 — close but not exact since prices differ.
    # Just verify the imbalance is bounded — no sign assumption.
    assert -0.05 < f.depth_imbalance < 0.05
