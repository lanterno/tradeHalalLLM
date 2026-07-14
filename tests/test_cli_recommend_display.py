"""Tests for the CLI recommendation display helper (quant range line)."""

from __future__ import annotations

from halal_trader.cli.recommend import _quant_line


def _rec(symbol: str, cand: dict) -> dict:
    return {"symbol": symbol, "candidates": {symbol: cand}}


def test_shows_calibrated_band_and_implied_move():
    rec = _rec(
        "NVDA",
        {
            "band5d_lo": 186.25,
            "band5d_hi": 222.41,
            "vol_pctl": 0.88,
            "quant_bands": {"calibrated": True},
            "impl_move_pct": 2.15,
            "impl_dte": 2,
            "impl_low": 199.15,
            "impl_high": 207.91,
        },
    )
    line = _quant_line(rec)
    assert "5d band" in line and "$186.25..$222.41" in line
    assert "(cal)" in line
    assert "vol pctl 88%" in line
    assert "implied ±2.1%/2d" in line
    assert "$199.15..$207.91" in line


def test_uncalibrated_tag():
    rec = _rec(
        "AAPL",
        {"band5d_lo": 300.0, "band5d_hi": 340.0, "quant_bands": {"calibrated": False}},
    )
    assert "(uncal)" in _quant_line(rec)


def test_band_only_when_no_implied():
    rec = _rec("AAPL", {"band5d_lo": 300.0, "band5d_hi": 340.0, "quant_bands": {}})
    line = _quant_line(rec)
    assert "5d band" in line
    assert "implied" not in line


def test_implied_only_when_no_band():
    rec = _rec("AAPL", {"impl_move_pct": 1.4, "impl_dte": 2, "impl_low": 312.9, "impl_high": 321.7})
    line = _quant_line(rec)
    assert "implied ±1.4%/2d" in line
    assert "5d band" not in line


def test_empty_when_no_quant_data():
    assert _quant_line(_rec("AAPL", {})) == ""
    assert _quant_line({"symbol": "AAPL"}) == ""
    assert _quant_line({}) == ""
