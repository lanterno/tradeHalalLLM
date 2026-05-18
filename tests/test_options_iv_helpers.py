"""Tests for the pure helpers in :mod:`trading.options_iv`.

The HTTP fetch path is exercised in `test_options_iv.py`. This file
adds the helpers underneath: `_reduce` (the chain → snapshot reducer
that decides ATM IV and put-call skew), `_safe_float`, `_safe_int`.
"""

from __future__ import annotations

from halal_trader.trading.options_iv import _reduce, _safe_float, _safe_int

# ── _safe_float ──────────────────────────────────────────────


def test_safe_float_valid_input():
    assert _safe_float(1.5) == 1.5
    assert _safe_float("2.5") == 2.5
    assert _safe_float(0) == 0.0


def test_safe_float_returns_zero_on_none():
    """Yahoo sometimes returns null fields — must not crash the reducer."""
    assert _safe_float(None) == 0.0


def test_safe_float_returns_zero_on_garbage():
    assert _safe_float("not-a-number") == 0.0


def test_safe_float_returns_zero_on_dict():
    assert _safe_float({"x": 1}) == 0.0


# ── _safe_int ────────────────────────────────────────────────


def test_safe_int_valid_input():
    assert _safe_int(42) == 42
    assert _safe_int("100") == 100


def test_safe_int_returns_zero_on_none():
    assert _safe_int(None) == 0


def test_safe_int_returns_zero_on_garbage():
    assert _safe_int("not-an-int") == 0


def test_safe_int_returns_zero_on_float_string():
    """`int("3.14")` raises — must default to 0 not crash."""
    assert _safe_int("3.14") == 0


# ── _reduce ─────────────────────────────────────────────────


def test_reduce_returns_none_when_no_atm_options():
    """No options inside the band → can't compute ATM IV → None."""
    out = _reduce(
        symbol="AAPL",
        spot=100.0,
        calls=[{"strike": 200.0, "impliedVolatility": 0.3}],
        puts=[],
        band=0.05,
    )
    assert out is None


def test_reduce_computes_atm_iv_as_mean_of_band_options():
    """ATM IV = mean of all options inside ±band of spot."""
    calls = [
        {"strike": 99.0, "impliedVolatility": 0.30, "volume": 10, "openInterest": 100},
        {"strike": 101.0, "impliedVolatility": 0.40, "volume": 5, "openInterest": 50},
    ]
    out = _reduce(symbol="AAPL", spot=100.0, calls=calls, puts=[], band=0.05)
    assert out is not None
    assert out.atm_iv == 0.35  # (0.30 + 0.40) / 2


def test_reduce_skips_zero_iv_options():
    """An option with iv=0 (Yahoo placeholder) doesn't contribute to ATM."""
    calls = [
        {"strike": 100.0, "impliedVolatility": 0.0, "volume": 0, "openInterest": 0},
        {"strike": 100.5, "impliedVolatility": 0.4, "volume": 1, "openInterest": 1},
    ]
    out = _reduce(symbol="AAPL", spot=100.0, calls=calls, puts=[], band=0.05)
    assert out is not None
    assert out.atm_iv == 0.4


def test_reduce_aggregates_volume_and_open_interest():
    calls = [
        {"strike": 100.0, "impliedVolatility": 0.3, "volume": 100, "openInterest": 1000},
    ]
    puts = [
        {"strike": 100.0, "impliedVolatility": 0.3, "volume": 50, "openInterest": 500},
    ]
    out = _reduce(symbol="AAPL", spot=100.0, calls=calls, puts=puts, band=0.05)
    assert out is not None
    assert out.call_volume == 100
    assert out.put_volume == 50
    assert out.call_open_interest == 1000
    assert out.put_open_interest == 500


def test_reduce_computes_positive_skew_when_otm_puts_richer():
    """Out-of-the-money puts at higher IV than OTM calls → positive skew
    (puts are "richer" — market is paying up for downside protection)."""
    calls = [
        {"strike": 100.0, "impliedVolatility": 0.3, "volume": 1, "openInterest": 1},
        {"strike": 115.0, "impliedVolatility": 0.25, "volume": 0, "openInterest": 0},  # OTM call
    ]
    puts = [
        {"strike": 100.0, "impliedVolatility": 0.3, "volume": 1, "openInterest": 1},
        {"strike": 85.0, "impliedVolatility": 0.45, "volume": 0, "openInterest": 0},  # OTM put
    ]
    out = _reduce(symbol="AAPL", spot=100.0, calls=calls, puts=puts, band=0.05)
    assert out is not None
    assert out.put_call_skew > 0  # 0.45 - 0.25 = 0.20


def test_reduce_skew_is_zero_when_one_wing_missing():
    """No OTM puts in the band → can't compute skew → defaults to 0."""
    calls = [
        {"strike": 100.0, "impliedVolatility": 0.3, "volume": 1, "openInterest": 1},
        {"strike": 115.0, "impliedVolatility": 0.25, "volume": 0, "openInterest": 0},
    ]
    out = _reduce(symbol="AAPL", spot=100.0, calls=calls, puts=[], band=0.05)
    assert out is not None
    assert out.put_call_skew == 0.0


def test_reduce_skips_negative_strike():
    """Defensive: a -ve strike (corrupt feed) doesn't poison the ATM mean."""
    calls = [
        {"strike": -1.0, "impliedVolatility": 0.3, "volume": 1, "openInterest": 1},
        {"strike": 100.0, "impliedVolatility": 0.4, "volume": 1, "openInterest": 1},
    ]
    out = _reduce(symbol="AAPL", spot=100.0, calls=calls, puts=[], band=0.05)
    assert out is not None
    assert out.atm_iv == 0.4
