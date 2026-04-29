"""Tests for the replay-fitted slippage regression model."""

from __future__ import annotations

from halal_trader.ml.slippage import (
    SlippageModel,
    fit_from_trades,
    load_from_file,
    save_to_file,
    trade_to_sample,
)


def _sample(*, size_usd: float, spread_bps: float, slippage_pct: float) -> dict:
    return {
        "size_usd": size_usd,
        "spread_bps": spread_bps,
        "atr_pct": 0.01,
        "rsi_14": 50.0,
        "kline_volatility_pct": 0.005,
        "hour_of_day": 12.0,
        "slippage_pct": slippage_pct,
    }


def test_identity_predicts_default_slippage() -> None:
    model = SlippageModel.identity()
    pred = model.predict({"size_usd": 100, "spread_bps": 5})
    assert 0.0 <= pred.pct <= 0.001  # ~5 bps default


def test_fit_with_too_few_samples_returns_identity() -> None:
    rows = [_sample(size_usd=100, spread_bps=5, slippage_pct=0.0001) for _ in range(5)]
    model = fit_from_trades(rows)
    assert model.n_samples == 0  # identity sentinel


def test_fit_recovers_size_signal() -> None:
    """Larger orders should predict higher slippage than smaller ones."""
    rows = []
    # Linear-ish relationship: slippage increases with size_usd.
    for i in range(60):
        size = 100 * (i + 1)
        rows.append(
            _sample(
                size_usd=size,
                spread_bps=5,
                slippage_pct=0.00001 * size,  # 1bps per $100
            )
        )
    model = fit_from_trades(rows)
    assert model.n_samples == 60
    big = model.predict({"size_usd": 6000, "spread_bps": 5})
    small = model.predict({"size_usd": 100, "spread_bps": 5})
    assert big.pct > small.pct


def test_save_and_load_round_trip(tmp_path) -> None:
    rows = [_sample(size_usd=100 + i, spread_bps=5, slippage_pct=0.0001) for i in range(40)]
    model = fit_from_trades(rows)
    save_to_file(model, tmp_path)
    loaded = load_from_file(tmp_path)
    assert loaded.n_samples == model.n_samples
    assert loaded.intercept == model.intercept


def test_trade_to_sample_extracts_slippage() -> None:
    trade = {
        "price": 100.0,
        "filled_price": 100.5,
        "filled_quantity": 1.0,
        "timestamp": "2026-04-29T12:00:00+00:00",
    }
    indicators = {
        "rsi_14": 55,
        "atr_14": 1.0,
        "kline_volatility_pct": 0.01,
    }
    sample = trade_to_sample(trade, indicators)
    assert sample is not None
    assert sample["slippage_pct"] == 0.005  # 50 bps adverse
    assert sample["size_usd"] == 100.5


def test_trade_to_sample_skips_missing_fills() -> None:
    sample = trade_to_sample({"price": 100.0}, {})  # no filled_price
    assert sample is None
