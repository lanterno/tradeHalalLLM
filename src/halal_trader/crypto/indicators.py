"""Technical indicators for crypto trading — pure computation, no I/O.

All functions accept lists of Kline objects and return indicator values.
Uses numpy for efficient array operations.
"""

from typing import Any

import numpy as np

from halal_trader.domain.models import Kline


def compute_all(klines: list[Kline]) -> dict[str, Any]:
    """Compute all standard indicators from a list of klines.

    Returns a dict with all indicator values, ready to be formatted
    into an LLM prompt.
    """
    if len(klines) < 2:
        return {"error": "insufficient data", "candle_count": len(klines)}

    closes = np.array([k.close for k in klines])
    highs = np.array([k.high for k in klines])
    lows = np.array([k.low for k in klines])
    volumes = np.array([k.volume for k in klines])

    result: dict[str, Any] = {
        "candle_count": len(klines),
        "current_price": float(closes[-1]),
        "price_change_1m": _pct_change(closes, 1),
        "price_change_5m": _pct_change(closes, 5),
        "price_change_15m": _pct_change(closes, 15),
    }

    # RSI (14-period)
    if len(closes) >= 15:
        result["rsi_14"] = round(rsi(closes, period=14), 2)

    # MACD (12, 26, 9)
    if len(closes) >= 35:
        macd_line, signal_line, histogram = macd(closes)
        result["macd"] = round(float(macd_line[-1]), 6)
        result["macd_signal"] = round(float(signal_line[-1]), 6)
        result["macd_histogram"] = round(float(histogram[-1]), 6)

    # Bollinger Bands (20, 2)
    if len(closes) >= 20:
        upper, middle, lower = bollinger_bands(closes)
        result["bb_upper"] = round(float(upper[-1]), 2)
        result["bb_middle"] = round(float(middle[-1]), 2)
        result["bb_lower"] = round(float(lower[-1]), 2)
        # Position within bands (0 = lower, 1 = upper)
        band_width = upper[-1] - lower[-1]
        if band_width > 0:
            result["bb_position"] = round(float((closes[-1] - lower[-1]) / band_width), 3)

    # EMAs
    for period in (9, 21, 50):
        if len(closes) >= period:
            ema_val = ema(closes, period=period)
            result[f"ema_{period}"] = round(float(ema_val[-1]), 2)

    # ATR (14-period)
    if len(highs) >= 15:
        result["atr_14"] = round(float(atr(highs, lows, closes, period=14)), 2)

    # VWAP (from available candles)
    if len(closes) >= 2:
        result["vwap"] = round(float(vwap(highs, lows, closes, volumes)), 2)

    # Volume analysis
    if len(volumes) >= 20:
        avg_vol = float(np.mean(volumes[-20:]))
        current_vol = float(volumes[-1])
        result["volume_current"] = round(current_vol, 2)
        result["volume_avg_20"] = round(avg_vol, 2)
        result["volume_ratio"] = round(current_vol / avg_vol, 2) if avg_vol > 0 else 0

    return result


# ── Individual Indicator Functions ──────────────────────────────


def rsi(closes: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index (Wilder's smoothing)."""
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Initial average using SMA
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Wilder's smoothing for remaining periods
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, and histogram."""
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    closes: np.ndarray, period: int = 20, std_dev: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands: upper, middle (SMA), lower."""
    middle = _sma(closes, period)
    rolling_std = _rolling_std(closes, period)
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    return upper, middle, lower


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    alpha = 2.0 / (period + 1)
    result = np.empty_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1]
    return result


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Average True Range."""
    high_low = highs[1:] - lows[1:]
    high_close = np.abs(highs[1:] - closes[:-1])
    low_close = np.abs(lows[1:] - closes[:-1])
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))

    if len(true_range) < period:
        return float(np.mean(true_range))

    # Wilder's smoothing
    atr_val = np.mean(true_range[:period])
    for i in range(period, len(true_range)):
        atr_val = (atr_val * (period - 1) + true_range[i]) / period

    return float(atr_val)


def vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float:
    """Volume-Weighted Average Price."""
    typical_price = (highs + lows + closes) / 3.0
    total_volume = np.sum(volumes)
    if total_volume == 0:
        return float(closes[-1])
    return float(np.sum(typical_price * volumes) / total_volume)


# ── Formatting for LLM Prompt ──────────────────────────────────


def format_indicators_for_prompt(symbol: str, indicators: dict[str, Any]) -> str:
    """Format computed indicators into a concise text block for the LLM."""
    if "error" in indicators:
        return f"  {symbol}: insufficient data ({indicators.get('candle_count', 0)} candles)"

    lines = [f"  {symbol} (price: ${indicators['current_price']:,.2f}):"]

    # Price changes
    changes = []
    for key, label in [
        ("price_change_1m", "1m"),
        ("price_change_5m", "5m"),
        ("price_change_15m", "15m"),
    ]:
        if key in indicators:
            changes.append(f"{label}: {indicators[key]:+.3%}")
    if changes:
        lines.append(f"    Price change: {', '.join(changes)}")

    # RSI
    if "rsi_14" in indicators:
        rsi_val = indicators["rsi_14"]
        signal = ""
        if rsi_val > 70:
            signal = " (OVERBOUGHT)"
        elif rsi_val < 30:
            signal = " (OVERSOLD)"
        lines.append(f"    RSI(14): {rsi_val}{signal}")

    # MACD
    if "macd" in indicators:
        hist = indicators.get("macd_histogram", 0)
        direction = "BULLISH" if hist > 0 else "BEARISH"
        lines.append(
            f"    MACD: {indicators['macd']:.6f}, Signal: {indicators['macd_signal']:.6f}, "
            f"Histogram: {hist:.6f} ({direction})"
        )

    # Bollinger Bands
    if "bb_upper" in indicators:
        lines.append(
            f"    Bollinger: Upper={indicators['bb_upper']:.2f}, "
            f"Mid={indicators['bb_middle']:.2f}, Lower={indicators['bb_lower']:.2f}, "
            f"Position={indicators.get('bb_position', 'N/A')}"
        )

    # EMAs
    emas = []
    for period in (9, 21, 50):
        key = f"ema_{period}"
        if key in indicators:
            emas.append(f"EMA{period}={indicators[key]:.2f}")
    if emas:
        lines.append(f"    EMAs: {', '.join(emas)}")

    # ATR
    if "atr_14" in indicators:
        lines.append(f"    ATR(14): {indicators['atr_14']:.2f}")

    # VWAP
    if "vwap" in indicators:
        lines.append(f"    VWAP: {indicators['vwap']:.2f}")

    # Volume
    if "volume_ratio" in indicators:
        lines.append(
            f"    Volume: current={indicators['volume_current']:.0f}, "
            f"avg20={indicators['volume_avg_20']:.0f}, "
            f"ratio={indicators['volume_ratio']:.2f}x"
        )

    return "\n".join(lines)


# ── Private Helpers ─────────────────────────────────────────────


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average (returns array padded with NaN for early values)."""
    result = np.full_like(data, np.nan)
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1 : i + 1])
    return result


def _rolling_std(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation."""
    result = np.full_like(data, np.nan)
    for i in range(period - 1, len(data)):
        result[i] = np.std(data[i - period + 1 : i + 1])
    return result


def _pct_change(closes: np.ndarray, periods: int) -> float | None:
    """Percentage change over N periods."""
    if len(closes) <= periods:
        return None
    prev = closes[-periods - 1]
    if prev == 0:
        return None
    return float((closes[-1] - prev) / prev)
