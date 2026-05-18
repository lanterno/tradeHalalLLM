"""Tests for `crypto.indicators.format_indicators_for_prompt`.

`test_crypto_indicators.py` covers `compute_all` end-to-end and the
component math (RSI / EMA / Bollinger). The prompt-side renderer
that turns the indicator dict into the LLM prompt block has its own
contract (signal labels, conditional sections, threshold semantics)
that's untested today. A regression here would silently mis-label
RSI/ADX/MACD signals — operators would see "OVERSOLD" on a perfectly
neutral pair, or vice-versa.
"""

from __future__ import annotations

from halal_trader.crypto.indicators import format_indicators_for_prompt

# ── Error sentinel ─────────────────────────────────────────


def test_format_error_indicator_renders_insufficient_data():
    """A `{"error": ...}` dict (degenerate series) renders as a
    one-liner with the candle count — operator sees the gap rather
    than a missing block."""
    out = format_indicators_for_prompt("BTCUSDT", {"error": "too few candles", "candle_count": 5})
    assert "BTCUSDT" in out
    assert "insufficient data" in out
    assert "5 candles" in out


def test_format_error_indicator_default_candle_count_zero():
    """Missing `candle_count` defaults to 0."""
    out = format_indicators_for_prompt("BTCUSDT", {"error": "x"})
    assert "0 candles" in out


# ── Header ────────────────────────────────────────────────


def test_format_includes_symbol_and_current_price():
    """Header line has the symbol + current price formatted with
    thousands separator + 2dp."""
    out = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_123.4567})
    assert "BTCUSDT" in out
    assert "$50,123.46" in out


# ── Price change section ──────────────────────────────────


def test_format_price_change_renders_each_horizon():
    """All three horizons (1m / 5m / 15m) appear when present."""
    out = format_indicators_for_prompt(
        "BTCUSDT",
        {
            "current_price": 50_000.0,
            "price_change_1m": 0.001,
            "price_change_5m": 0.005,
            "price_change_15m": -0.012,
        },
    )
    assert "1m: +0.100%" in out
    assert "5m: +0.500%" in out
    assert "15m: -1.200%" in out


def test_format_price_change_section_omitted_when_no_horizons():
    """No price_change_* keys → no "Price change:" line at all."""
    out = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_000.0})
    assert "Price change:" not in out


# ── RSI labelling ─────────────────────────────────────────


def test_format_rsi_overbought_label_above_70():
    """RSI > 70 → "(OVERBOUGHT)" suffix. Pin the boundary so a
    refactor doesn't shift the threshold."""
    out = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_000.0, "rsi_14": 75.5})
    assert "RSI(14): 75.5" in out
    assert "(OVERBOUGHT)" in out


def test_format_rsi_oversold_label_below_30():
    """RSI < 30 → "(OVERSOLD)" suffix."""
    out = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_000.0, "rsi_14": 25.0})
    assert "(OVERSOLD)" in out


def test_format_rsi_no_label_in_neutral_zone():
    """50 RSI → no signal label (neither over- nor under-bought)."""
    out = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_000.0, "rsi_14": 50.0})
    assert "OVERBOUGHT" not in out
    assert "OVERSOLD" not in out


def test_format_rsi_at_70_boundary_no_label():
    """Exactly 70 → no label (the check is `> 70`, exclusive)."""
    out = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_000.0, "rsi_14": 70.0})
    assert "OVERBOUGHT" not in out


def test_format_rsi_at_30_boundary_no_label():
    """Exactly 30 → no label (the check is `< 30`, exclusive)."""
    out = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_000.0, "rsi_14": 30.0})
    assert "OVERSOLD" not in out


# ── MACD direction ─────────────────────────────────────────


def test_format_macd_bullish_label_when_histogram_positive():
    """MACD histogram > 0 → BULLISH direction tag."""
    out = format_indicators_for_prompt(
        "BTCUSDT",
        {
            "current_price": 50_000.0,
            "macd": 12.5,
            "macd_signal": 10.0,
            "macd_histogram": 2.5,
        },
    )
    assert "(BULLISH)" in out


def test_format_macd_bearish_label_when_histogram_negative_or_zero():
    """MACD histogram ≤ 0 → BEARISH (zero falls into BEARISH because
    the check is `> 0`, exclusive)."""
    out_neg = format_indicators_for_prompt(
        "BTCUSDT",
        {
            "current_price": 50_000.0,
            "macd": 5.0,
            "macd_signal": 7.0,
            "macd_histogram": -2.0,
        },
    )
    assert "(BEARISH)" in out_neg

    out_zero = format_indicators_for_prompt(
        "BTCUSDT",
        {
            "current_price": 50_000.0,
            "macd": 5.0,
            "macd_signal": 5.0,
            "macd_histogram": 0.0,
        },
    )
    assert "(BEARISH)" in out_zero


def test_format_macd_default_histogram_when_missing():
    """If `macd_histogram` is absent but `macd` is present, fall back
    to 0 → BEARISH default."""
    out = format_indicators_for_prompt(
        "BTCUSDT",
        {"current_price": 50_000.0, "macd": 12.5, "macd_signal": 10.0},
    )
    assert "(BEARISH)" in out


# ── Bollinger Bands ─────────────────────────────────────────


def test_format_bollinger_renders_all_three_bands():
    out = format_indicators_for_prompt(
        "BTCUSDT",
        {
            "current_price": 50_000.0,
            "bb_upper": 51_000.0,
            "bb_middle": 50_000.0,
            "bb_lower": 49_000.0,
            "bb_position": 0.5,
        },
    )
    assert "Upper=51000.00" in out
    assert "Mid=50000.00" in out
    assert "Lower=49000.00" in out
    assert "Position=0.5" in out


def test_format_bollinger_position_na_when_missing():
    """Defensive: bb_position absent → "N/A" rather than KeyError."""
    out = format_indicators_for_prompt(
        "BTCUSDT",
        {
            "current_price": 50_000.0,
            "bb_upper": 51_000.0,
            "bb_middle": 50_000.0,
            "bb_lower": 49_000.0,
        },
    )
    assert "Position=N/A" in out


# ── EMAs ───────────────────────────────────────────────────


def test_format_emas_renders_each_period():
    """EMA9 / EMA21 / EMA50 each render when present."""
    out = format_indicators_for_prompt(
        "BTCUSDT",
        {
            "current_price": 50_000.0,
            "ema_9": 49_800.0,
            "ema_21": 49_500.0,
            "ema_50": 49_000.0,
        },
    )
    assert "EMA9=49800.00" in out
    assert "EMA21=49500.00" in out
    assert "EMA50=49000.00" in out


def test_format_emas_skip_unavailable_periods():
    """Only EMA9 → only that one renders; the line still appears."""
    out = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_000.0, "ema_9": 49_800.0})
    assert "EMA9=49800.00" in out
    assert "EMA21" not in out
    assert "EMA50" not in out


def test_format_emas_section_omitted_when_none_present():
    out = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_000.0})
    assert "EMAs:" not in out


# ── ATR ────────────────────────────────────────────────────


def test_format_atr_when_present():
    out = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_000.0, "atr_14": 250.5})
    assert "ATR(14): 250.50" in out


# ── ADX strength label ─────────────────────────────────────


def test_format_adx_strong_trend_above_25():
    """ADX > 25 → STRONG TREND label."""
    out = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_000.0, "adx_14": 35.0})
    assert "STRONG TREND" in out


def test_format_adx_weak_ranging_at_or_below_25():
    """ADX ≤ 25 → WEAK/RANGING (the check is `> 25` exclusive, so 25
    itself counts as ranging)."""
    out_low = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_000.0, "adx_14": 15.0})
    assert "WEAK/RANGING" in out_low

    out_at = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_000.0, "adx_14": 25.0})
    assert "WEAK/RANGING" in out_at


# ── VWAP ───────────────────────────────────────────────────


def test_format_vwap_renders_when_present():
    out = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_000.0, "vwap": 49_950.123})
    assert "VWAP: 49950.12" in out


# ── Volume ────────────────────────────────────────────────


def test_format_volume_renders_current_avg_and_ratio():
    """Volume section needs all three keys to render the full line."""
    out = format_indicators_for_prompt(
        "BTCUSDT",
        {
            "current_price": 50_000.0,
            "volume_current": 1_500.0,
            "volume_avg_20": 1_000.0,
            "volume_ratio": 1.5,
        },
    )
    assert "current=1500" in out
    assert "avg20=1000" in out
    assert "ratio=1.50x" in out


# ── Composite shape ────────────────────────────────────────


def test_format_minimal_indicators_just_renders_header():
    """`current_price` only → just the header line, nothing else."""
    out = format_indicators_for_prompt("BTCUSDT", {"current_price": 50_000.0})
    # One line only — no sections triggered.
    assert "\n" not in out


def test_format_full_indicator_set_renders_all_sections():
    """Smoke: a fully-populated dict yields every section in order."""
    indicators = {
        "current_price": 50_000.0,
        "price_change_5m": 0.005,
        "rsi_14": 55.0,
        "macd": 1.0,
        "macd_signal": 0.5,
        "macd_histogram": 0.5,
        "bb_upper": 51_000.0,
        "bb_middle": 50_000.0,
        "bb_lower": 49_000.0,
        "bb_position": 0.5,
        "ema_9": 49_800.0,
        "ema_21": 49_500.0,
        "atr_14": 250.0,
        "adx_14": 30.0,
        "vwap": 49_950.0,
        "volume_current": 1_500.0,
        "volume_avg_20": 1_000.0,
        "volume_ratio": 1.5,
    }
    out = format_indicators_for_prompt("BTCUSDT", indicators)
    # Every section title is present.
    assert "Price change" in out
    assert "RSI(14)" in out
    assert "MACD" in out
    assert "Bollinger" in out
    assert "EMAs" in out
    assert "ATR(14)" in out
    assert "ADX(14)" in out
    assert "VWAP" in out
    assert "Volume" in out
