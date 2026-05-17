"""Tests for :func:`format_regime_for_prompt`.

`build_regime_text` (the wrapper that runs the detector + this
formatter) is covered in `test_cycle_shared_helpers.py` and
`test_cycle_stages.py`. This file pins the formatter shape directly.
"""

from __future__ import annotations

from halal_trader.crypto.regime import MarketRegime, format_regime_for_prompt


def test_empty_returns_sentinel():
    assert format_regime_for_prompt({}) == "No regime data available."


def test_single_pair_renders_regime_label_uppercase():
    out = format_regime_for_prompt(
        {"BTCUSDT": (MarketRegime.TRENDING_UP, 0.85, "trade with the trend")}
    )
    assert "BTCUSDT" in out
    assert "TRENDING_UP" in out


def test_renders_confidence_as_percent():
    out = format_regime_for_prompt(
        {"BTCUSDT": (MarketRegime.RANGING, 0.6, "use mean reversion")}
    )
    assert "60%" in out


def test_includes_strategy_instructions():
    out = format_regime_for_prompt(
        {"BTCUSDT": (MarketRegime.RANGING, 0.6, "USE MEAN REVERSION HERE")}
    )
    assert "USE MEAN REVERSION HERE" in out


def test_multiple_pairs_sorted_alphabetically():
    out = format_regime_for_prompt(
        {
            "ZRXUSDT": (MarketRegime.RANGING, 0.6, "x"),
            "BTCUSDT": (MarketRegime.TRENDING_UP, 0.8, "y"),
            "ETHUSDT": (MarketRegime.HIGH_VOLATILITY, 0.7, "z"),
        }
    )
    btc = out.find("BTCUSDT")
    eth = out.find("ETHUSDT")
    zrx = out.find("ZRXUSDT")
    assert 0 <= btc < eth < zrx


def test_each_pair_renders_two_lines():
    """One line for the regime label + one for strategy text."""
    out = format_regime_for_prompt(
        {"BTCUSDT": (MarketRegime.TRENDING_DOWN, 0.7, "shrink size")}
    )
    lines = out.split("\n")
    assert len(lines) == 2
    assert "TRENDING_DOWN" in lines[0]
    assert "shrink size" in lines[1]


def test_confidence_zero_renders_zero_percent():
    out = format_regime_for_prompt(
        {"BTCUSDT": (MarketRegime.RANGING, 0.0, "be cautious")}
    )
    assert "0%" in out


def test_confidence_one_renders_hundred_percent():
    out = format_regime_for_prompt(
        {"BTCUSDT": (MarketRegime.TRENDING_UP, 1.0, "max conviction")}
    )
    assert "100%" in out
