"""Tests for :func:`format_forecasts_for_prompt`.

The Chronos pipeline that produces forecasts needs the model + GPU,
but the prompt formatter is a pure dict walker — what the LLM
actually reads each cycle.
"""

from __future__ import annotations

from halal_trader.ml.forecaster import PriceForecast, format_forecasts_for_prompt


def _fc(
    *,
    pair: str = "BTCUSDT",
    current: float = 100.0,
    predicted: list[float] | None = None,
    upper: list[float] | None = None,
    lower: list[float] | None = None,
    confidence: float = 0.7,
    horizon: int = 5,
) -> PriceForecast:
    return PriceForecast(
        pair=pair,
        current_price=current,
        predicted_prices=predicted or [],
        upper_bound=upper or [],
        lower_bound=lower or [],
        confidence=confidence,
        horizon=horizon,
    )


def test_empty_dict_returns_sentinel():
    assert format_forecasts_for_prompt({}) == "No ML price forecasts available."


def test_skips_forecasts_with_no_predicted_prices():
    """A forecaster that fell back to defaults shouldn't emit a noise row."""
    fc = _fc(predicted=[], upper=[], lower=[])
    out = format_forecasts_for_prompt({"BTCUSDT": fc})
    assert out == "No ML price forecasts available."


def test_renders_up_when_last_predicted_above_current():
    fc = _fc(
        current=100.0,
        predicted=[101.0, 105.0],
        upper=[110.0, 110.0],
        lower=[95.0, 95.0],
    )
    out = format_forecasts_for_prompt({"BTCUSDT": fc})
    assert "UP" in out
    assert "5.00%" in out


def test_renders_down_when_last_predicted_below_current():
    fc = _fc(
        current=100.0,
        predicted=[99.0, 92.0],
        upper=[105.0, 105.0],
        lower=[90.0, 90.0],
    )
    out = format_forecasts_for_prompt({"BTCUSDT": fc})
    assert "DOWN" in out
    assert "8.00%" in out


def test_includes_confidence_as_percent():
    fc = _fc(
        current=100.0,
        predicted=[101.0],
        upper=[105.0],
        lower=[95.0],
        confidence=0.65,
    )
    out = format_forecasts_for_prompt({"BTCUSDT": fc})
    assert "65%" in out


def test_includes_range_from_last_bounds():
    fc = _fc(
        current=100.0,
        predicted=[101.0, 102.0],
        upper=[105.0, 110.0],
        lower=[95.0, 90.0],
        confidence=0.7,
    )
    out = format_forecasts_for_prompt({"BTCUSDT": fc})
    # Should use the last upper/lower (not the first).
    assert "$90.00" in out
    assert "$110.00" in out


def test_zero_current_price_yields_zero_pct_no_division_error():
    """Defensive: a zero current price would divide-by-zero if not guarded."""
    fc = _fc(
        current=0.0,
        predicted=[101.0],
        upper=[105.0],
        lower=[95.0],
    )
    # Must not raise; the renderer falls back to 0% change.
    out = format_forecasts_for_prompt({"BTCUSDT": fc})
    assert "0.00%" in out


def test_multiple_pairs_sorted_alphabetically():
    fc1 = _fc(pair="ZRXUSDT", predicted=[101.0], upper=[105.0], lower=[95.0])
    fc2 = _fc(pair="BTCUSDT", predicted=[101.0], upper=[105.0], lower=[95.0])
    out = format_forecasts_for_prompt({"ZRXUSDT": fc1, "BTCUSDT": fc2})
    btc_idx = out.find("BTCUSDT")
    zrx_idx = out.find("ZRXUSDT")
    assert btc_idx >= 0 and zrx_idx >= 0
    assert btc_idx < zrx_idx  # alphabetical
