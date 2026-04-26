"""KNN past-setup retrieval tests."""

from __future__ import annotations

import pytest

from halal_trader.ml.knn_setups import (
    PastSetup,
    find_nearest_setups,
    format_setups_for_prompt,
)


def _snap(
    rsi=50,
    macd_hist=0.0,
    vol_ratio=1.0,
    atr=0.02,
    bb=0.5,
    ema9=100,
    ema21=99,
    vwap=100,
    pc5m=0.0,
    label=1,
    return_pct=0.01,
    pair="X",
):
    return {
        "rsi_14": rsi,
        "macd_histogram": macd_hist,
        "volume_ratio": vol_ratio,
        "atr_14": atr,
        "bb_position": bb,
        "ema_9": ema9,
        "ema_21": ema21,
        "vwap": vwap,
        "price_change_5m": pc5m,
        "label": label,
        "return_pct": return_pct,
        "pair": pair,
    }


def test_find_nearest_returns_closest_first():
    current = _snap(rsi=50, macd_hist=0.1)
    past = [
        _snap(rsi=80, macd_hist=0.5, pair="far"),
        _snap(rsi=51, macd_hist=0.11, pair="close"),
        _snap(rsi=70, macd_hist=0.3, pair="medium"),
    ]
    result = find_nearest_setups(current, past, k=3)
    assert [s.pair for s in result] == ["close", "medium", "far"]


def test_k_caps_result_length():
    current = _snap()
    past = [_snap(pair=f"p{i}") for i in range(10)]
    result = find_nearest_setups(current, past, k=3)
    assert len(result) == 3


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        find_nearest_setups(_snap(), [_snap()], k=0)


def test_missing_features_in_current_returns_empty():
    """A snapshot can't be a query if any feature is None."""
    incomplete = _snap()
    incomplete.pop("rsi_14")
    assert find_nearest_setups(incomplete, [_snap()], k=3) == []


def test_missing_features_in_past_skipped():
    """A past snap with missing features is dropped, not imputed."""
    bad = _snap(pair="bad")
    bad.pop("vwap")
    good = _snap(pair="good")
    result = find_nearest_setups(_snap(), [bad, good], k=2)
    assert [s.pair for s in result] == ["good"]


def test_missing_label_or_return_pct_skipped():
    """Without a labeled outcome the past row is useless."""
    no_label = _snap(label=None, pair="no-label")
    no_ret = _snap(return_pct=None, pair="no-ret")
    good = _snap(pair="good")
    result = find_nearest_setups(_snap(), [no_label, no_ret, good], k=3)
    assert [s.pair for s in result] == ["good"]


def test_format_for_prompt_empty():
    assert format_setups_for_prompt([]) == ""


def test_format_for_prompt_includes_summary():
    setups = [
        PastSetup(pair="A", features=tuple([0.0] * 9), return_pct=0.05, label=1),
        PastSetup(pair="B", features=tuple([0.0] * 9), return_pct=-0.02, label=0),
    ]
    text = format_setups_for_prompt(setups)
    assert "2 past setups" in text
    assert "1/2 profitable" in text
    assert "+5.00%" in text
    assert "-2.00%" in text
    assert "(W" in text
    assert "(L" in text
