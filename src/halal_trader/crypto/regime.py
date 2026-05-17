"""Market regime detection — classifies current conditions for strategy adaptation."""

from __future__ import annotations

import logging
import pickle
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


_REGIME_INSTRUCTIONS = {
    MarketRegime.TRENDING_UP: (
        "Market is TRENDING UP. Trade with the trend: buy on pullbacks to EMA21/EMA50, "
        "set wider take-profits (1.5-3%), use tight stop-losses on counter-trend side."
    ),
    MarketRegime.TRENDING_DOWN: (
        "Market is TRENDING DOWN. Only sell/hold existing positions, no new buys unless "
        "strong reversal signals (RSI < 25 + volume spike + positive sentiment)."
    ),
    MarketRegime.RANGING: (
        "Market is RANGING. Use mean-reversion: buy near Bollinger Band lower + RSI < 35, "
        "sell near Bollinger Band upper + RSI > 65. Keep tight stop-losses."
    ),
    MarketRegime.HIGH_VOLATILITY: (
        "Market is HIGHLY VOLATILE. Reduce position sizes by 50%, widen stop-losses to 1.5-2%, "
        "only trade with strong sentiment confirmation and multiple indicator alignment."
    ),
}


class RegimeDetector:
    """Classifies market regime using rules and optional ML."""

    def __init__(self, *, models_dir: Path | None = None) -> None:
        self._models_dir = models_dir or Path("models")
        self._ml_model = None
        self._ml_path = self._models_dir / "regime_classifier.pkl"
        self._load_ml_model()

    def _load_ml_model(self) -> None:
        if self._ml_path.exists():
            try:
                with open(self._ml_path, "rb") as f:
                    self._ml_model = pickle.load(f)
                logger.info("Regime ML classifier loaded")
            except Exception as e:
                logger.debug("Failed to load regime classifier: %s", e)

    def detect(self, indicators: dict[str, Any]) -> tuple[MarketRegime, float, str]:
        """Detect the current market regime.

        Returns (regime, confidence, strategy_instructions).
        """
        if self._ml_model is not None:
            try:
                return self._detect_ml(indicators)
            except Exception:
                pass

        return self._detect_rules(indicators)

    def _detect_rules(self, indicators: dict[str, Any]) -> tuple[MarketRegime, float, str]:
        """Rule-based regime detection (always works, no training needed)."""
        adx = indicators.get("adx_14")
        bb_width = self._compute_bb_width(indicators)
        volume_ratio = indicators.get("volume_ratio", 1.0)
        rsi = indicators.get("rsi_14", 50)
        ema9 = indicators.get("ema_9")
        ema50 = indicators.get("ema_50")
        price = indicators.get("current_price")

        # High volatility check: ATR spike or extreme BB width + volume surge
        if bb_width is not None and bb_width > 0.06 and volume_ratio > 1.5:
            return (
                MarketRegime.HIGH_VOLATILITY,
                0.8,
                _REGIME_INSTRUCTIONS[MarketRegime.HIGH_VOLATILITY],
            )

        # Trending detection via ADX or EMA spread
        is_trending = False
        trend_direction = None

        if adx is not None and adx > 25:
            is_trending = True
        elif ema9 is not None and ema50 is not None:
            spread = abs(ema9 - ema50) / ema50 if ema50 > 0 else 0
            if spread > 0.005:
                is_trending = True

        if is_trending and price is not None and ema50 is not None:
            trend_direction = "up" if price > ema50 else "down"
        elif is_trending and rsi is not None:
            trend_direction = "up" if rsi > 55 else ("down" if rsi < 45 else None)

        if is_trending and trend_direction == "up":
            confidence = 0.7
            if adx is not None:
                confidence = min(0.95, adx / 40)
            return (
                MarketRegime.TRENDING_UP,
                confidence,
                _REGIME_INSTRUCTIONS[MarketRegime.TRENDING_UP],
            )

        if is_trending and trend_direction == "down":
            confidence = 0.7
            if adx is not None:
                confidence = min(0.95, adx / 40)
            return (
                MarketRegime.TRENDING_DOWN,
                confidence,
                _REGIME_INSTRUCTIONS[MarketRegime.TRENDING_DOWN],
            )

        # Default: ranging
        confidence = 0.6
        if adx is not None and adx < 20:
            confidence = 0.8
        return (
            MarketRegime.RANGING,
            confidence,
            _REGIME_INSTRUCTIONS[MarketRegime.RANGING],
        )

    def _detect_ml(self, indicators: dict[str, Any]) -> tuple[MarketRegime, float, str]:
        """ML-based regime detection (requires trained model)."""
        features = self._extract_features(indicators)
        if features is None:
            return self._detect_rules(indicators)

        X = np.array([features])
        pred = self._ml_model.predict(X)[0]
        proba = self._ml_model.predict_proba(X)[0]
        confidence = float(max(proba))

        regime = MarketRegime(pred)
        return regime, confidence, _REGIME_INSTRUCTIONS[regime]

    def train(self, samples: list[dict], labels: list[str]) -> bool:
        """Train the ML regime classifier on labeled market windows."""
        if len(samples) < 200:
            return False

        try:
            from sklearn.ensemble import RandomForestClassifier

            X = []
            for s in samples:
                f = self._extract_features(s)
                if f is not None:
                    X.append(f)

            if len(X) < 100:
                return False

            y = labels[: len(X)]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(np.array(X), y)

            self._models_dir.mkdir(parents=True, exist_ok=True)
            with open(self._ml_path, "wb") as f:
                pickle.dump(model, f)

            self._ml_model = model
            logger.info("Regime classifier trained on %d samples", len(X))
            return True
        except ImportError:
            logger.info("scikit-learn not installed — ML regime detection disabled")
            return False
        except Exception as e:
            logger.warning("Regime classifier training failed: %s", e)
            return False

    def _compute_bb_width(self, indicators: dict[str, Any]) -> float | None:
        upper = indicators.get("bb_upper")
        lower = indicators.get("bb_lower")
        middle = indicators.get("bb_middle")
        if upper is not None and lower is not None and middle is not None and middle > 0:
            return (upper - lower) / middle
        return None

    def _extract_features(self, indicators: dict[str, Any]) -> list[float] | None:
        feat_keys = ["rsi_14", "macd_histogram", "volume_ratio", "atr_14", "bb_position"]
        values = []
        for k in feat_keys:
            v = indicators.get(k)
            if v is None:
                return None
            values.append(float(v))

        adx = indicators.get("adx_14")
        values.append(float(adx) if adx is not None else 25.0)

        bb_w = self._compute_bb_width(indicators)
        values.append(bb_w if bb_w is not None else 0.03)

        return values


def format_regime_for_prompt(
    regimes: dict[str, tuple[MarketRegime, float, str]],
) -> str:
    """Format regime detection results for the LLM prompt."""
    if not regimes:
        return "No regime data available."

    lines = []
    for pair, (regime, confidence, instructions) in sorted(regimes.items()):
        lines.append(f"  {pair}: {regime.value.upper()} (confidence: {confidence:.0%})")
        lines.append(f"    Strategy: {instructions}")

    return "\n".join(lines)


def build_regime_text(
    detector: RegimeDetector | None,
    indicators_by_symbol: dict[str, dict[str, Any]],
) -> str:
    """Run the detector over each symbol's indicator vector and format the result.

    Shared between :class:`CryptoCycleService` and ``TradingCycleService``
    so both bots produce identical regime blocks. Symbols whose
    indicators carry an ``error`` key are skipped — their bars failed
    parse / didn't have enough history.

    Returns ``""`` when the detector is missing, raises silently on
    detector errors (returns empty string), or formats the per-symbol
    detection via :func:`format_regime_for_prompt`.
    """
    if detector is None or not indicators_by_symbol:
        return ""
    try:
        regimes: dict[str, tuple[MarketRegime, float, str]] = {}
        for symbol, indicators in indicators_by_symbol.items():
            if not indicators or "error" in indicators:
                continue
            regimes[symbol] = detector.detect(indicators)
        if regimes:
            return format_regime_for_prompt(regimes)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Regime detection failed: %s", exc)
    return ""
