"""Extended feature set for the next-gen ML pipeline.

The legacy ``FEATURE_KEYS`` is 9 columns of indicator state. ML papers
on price-action classifiers typically benefit from feature counts in
the 20-50 range — derived ratios, multi-window aggregates,
microstructure summaries — so this module assembles a wider vector
without hard-coding new database columns.

Each builder takes the same kline + indicator dicts the cycle already
produces and returns a name → float mapping. Composing them gives the
new feature vector. Order is stable so an SGDClassifier trained on
this set can be reloaded across processes without surprise drift.
"""

from __future__ import annotations

from typing import Final, Mapping, Sequence

import numpy as np

from halal_trader.domain.models import Kline


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den == 0:
        return default
    return num / den


def base_indicator_features(indicators: Mapping[str, float | None]) -> dict[str, float]:
    """Lift the 9 legacy indicator values into the extended set."""
    out: dict[str, float] = {}
    for key in (
        "rsi_14",
        "macd_histogram",
        "volume_ratio",
        "atr_14",
        "bb_position",
        "ema_9",
        "ema_21",
        "vwap",
        "price_change_5m",
    ):
        v = indicators.get(key)
        if v is None:
            continue
        try:
            out[key] = float(v)
        except Exception:
            continue
    return out


def derived_features(indicators: Mapping[str, float | None]) -> dict[str, float]:
    """Compute ratios + cross-indicator signals from the base set.

    Each derived feature has an obvious operator interpretation so we
    can debug surprises ("why is the model recommending sells in this
    regime?") without a SHAP plot.
    """
    out: dict[str, float] = {}
    ema9 = indicators.get("ema_9")
    ema21 = indicators.get("ema_21")
    ema50 = indicators.get("ema_50")
    price = indicators.get("current_price")
    vwap = indicators.get("vwap")
    bb_upper = indicators.get("bb_upper")
    bb_lower = indicators.get("bb_lower")
    atr = indicators.get("atr_14")

    if ema9 and ema21:
        out["ema_9_minus_21"] = float(ema9) - float(ema21)
        out["ema_9_over_21"] = _safe_div(float(ema9), float(ema21))
    if ema21 and ema50:
        out["ema_21_minus_50"] = float(ema21) - float(ema50)
    if price and vwap:
        out["price_minus_vwap"] = float(price) - float(vwap)
        out["price_over_vwap"] = _safe_div(float(price), float(vwap))
    if bb_upper and bb_lower and price:
        width = float(bb_upper) - float(bb_lower)
        out["bb_width"] = width
        out["price_in_bb"] = _safe_div(float(price) - float(bb_lower), width)
    if atr and price:
        out["atr_pct"] = _safe_div(float(atr), float(price))

    return out


def kline_window_features(klines: Sequence[Kline], *, window: int = 20) -> dict[str, float]:
    """Quantitative summaries over the last ``window`` candles.

    Captures texture beyond the latest indicator snapshot — recent vol,
    drawdown depth, body/wick balance — that an averaged indicator
    might miss.
    """
    if not klines:
        return {}
    closes = np.asarray([k.close for k in klines[-window:]], dtype=float)
    highs = np.asarray([k.high for k in klines[-window:]], dtype=float)
    lows = np.asarray([k.low for k in klines[-window:]], dtype=float)
    opens = np.asarray([k.open for k in klines[-window:]], dtype=float)
    vols = np.asarray([k.volume for k in klines[-window:]], dtype=float)

    if closes.size < 2:
        return {}

    returns = np.diff(closes) / closes[:-1]
    range_pct = float((highs.max() - lows.min()) / closes[-1]) if closes[-1] > 0 else 0.0
    out: dict[str, float] = {
        "ret_window_mean": float(np.mean(returns)),
        "ret_window_std": float(np.std(returns)),
        "ret_window_skew": float(_safe_skew(returns)),
        "drawdown_window": _max_drawdown(closes),
        "high_low_range_pct": range_pct,
        "body_to_range_ratio": float(
            np.mean(np.abs(closes - opens) / np.maximum(highs - lows, 1e-9))
        ),
        "volume_change_pct": float(
            _safe_div(vols[-1] - vols[:-1].mean(), vols[:-1].mean()) if vols.size > 1 else 0.0
        ),
        "up_candle_ratio": float(np.mean(closes > opens)),
    }
    return out


def _safe_skew(arr: np.ndarray) -> float:
    """Population skew without requiring scipy; defensive against zero-variance windows."""
    if arr.size < 3:
        return 0.0
    std = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - np.mean(arr)) / std) ** 3))


def _max_drawdown(closes: np.ndarray) -> float:
    if closes.size == 0:
        return 0.0
    peak = closes[0]
    max_dd = 0.0
    for v in closes:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
    return float(max_dd)


def assemble_features(
    indicators: Mapping[str, float | None],
    klines: Sequence[Kline] | None = None,
) -> dict[str, float]:
    """Compose all feature builders into a single name → float dict."""
    out: dict[str, float] = {}
    out.update(base_indicator_features(indicators))
    out.update(derived_features(indicators))
    if klines:
        out.update(kline_window_features(klines))
    return out


# Stable ordering for the SGDClassifier vector. New features land at the
# end so the prefix of an existing model's coefficient vector stays
# meaningful.
EXTENDED_FEATURE_ORDER: Final[tuple[str, ...]] = (
    # Base
    "rsi_14",
    "macd_histogram",
    "volume_ratio",
    "atr_14",
    "bb_position",
    "ema_9",
    "ema_21",
    "vwap",
    "price_change_5m",
    # Derived
    "ema_9_minus_21",
    "ema_9_over_21",
    "ema_21_minus_50",
    "price_minus_vwap",
    "price_over_vwap",
    "bb_width",
    "price_in_bb",
    "atr_pct",
    # Window aggregates
    "ret_window_mean",
    "ret_window_std",
    "ret_window_skew",
    "drawdown_window",
    "high_low_range_pct",
    "body_to_range_ratio",
    "volume_change_pct",
    "up_candle_ratio",
)


def to_vector(features: Mapping[str, float], default: float = 0.0) -> list[float]:
    """Coerce a feature dict into the canonical vector ordering."""
    return [float(features.get(k, default)) for k in EXTENDED_FEATURE_ORDER]
