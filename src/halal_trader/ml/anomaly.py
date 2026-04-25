"""Market anomaly detection — IsolationForest on indicator snapshots."""

from __future__ import annotations

import logging
import pickle

import numpy as np

from halal_trader.ml.hub import ModelHub

logger = logging.getLogger(__name__)

_FEATURES = ["rsi_14", "macd_histogram", "volume_ratio", "atr_14", "bb_position"]


class MarketAnomalyDetector:
    """Detects unusual market microstructure using IsolationForest."""

    def __init__(self, hub: ModelHub, *, min_samples: int = 100) -> None:
        self._hub = hub
        self._min_samples = min_samples
        self._model = None
        self._samples: list[list[float]] = []
        self._model_path = hub.models_dir / "anomaly_detector.pkl"
        self._load_model()

    def _load_model(self) -> None:
        """Load a previously trained model from disk."""
        if self._model_path.exists():
            try:
                with open(self._model_path, "rb") as f:
                    self._model = pickle.load(f)
                logger.info("Anomaly detector loaded from %s", self._model_path)
            except Exception as e:
                logger.warning("Failed to load anomaly model: %s", e)

    def add_sample(self, indicators: dict) -> None:
        """Add an indicator snapshot as a training sample."""
        features = self._extract_features(indicators)
        if features is not None:
            self._samples.append(features)

    def train(self) -> bool:
        """Train the IsolationForest on collected samples."""
        if len(self._samples) < self._min_samples:
            return False

        try:
            from sklearn.ensemble import IsolationForest

            X = np.array(self._samples[-5000:])
            self._model = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42,
            )
            self._model.fit(X)

            with open(self._model_path, "wb") as f:
                pickle.dump(self._model, f)

            logger.info("Anomaly detector trained on %d samples", len(X))
            return True
        except ImportError:
            logger.info("scikit-learn not installed — anomaly detection disabled")
            return False
        except Exception as e:
            logger.warning("Anomaly detector training failed: %s", e)
            return False

    def detect(self, indicators: dict) -> tuple[bool, float]:
        """Check if the current indicator snapshot is anomalous.

        Returns (is_anomaly, anomaly_score) where lower score = more anomalous.
        """
        if self._model is None:
            return False, 0.0

        features = self._extract_features(indicators)
        if features is None:
            return False, 0.0

        try:
            X = np.array([features])
            score = self._model.decision_function(X)[0]
            is_anomaly = self._model.predict(X)[0] == -1
            return bool(is_anomaly), float(score)
        except Exception as e:
            logger.debug("Anomaly detection failed: %s", e)
            return False, 0.0

    def _extract_features(self, indicators: dict) -> list[float] | None:
        """Extract feature vector from indicator dict."""
        values = []
        for feat in _FEATURES:
            val = indicators.get(feat)
            if val is None:
                return None
            values.append(float(val))
        return values


class MLSignalClassifier:
    """XGBoost classifier trained on our own trade history."""

    def __init__(self, hub: ModelHub) -> None:
        self._hub = hub
        self._model = None
        self._model_path = hub.models_dir / "signal_classifier.pkl"
        self._samples: list[list[float]] = []
        self._labels: list[int] = []
        self._load_model()

    def add_sample(self, indicators: dict, label: int) -> None:
        """Add a labeled indicator snapshot for incremental training."""
        features = []
        for feat in _FEATURES:
            val = indicators.get(feat)
            if val is None:
                return
            features.append(float(val))
        self._samples.append(features)
        self._labels.append(label)

    def auto_train(self, min_samples: int = 50) -> bool:
        """Train on accumulated samples if enough data is available."""
        if len(self._samples) < min_samples:
            return False
        return self.train(self._samples[-5000:], self._labels[-5000:])

    def _load_model(self) -> None:
        if self._model_path.exists():
            try:
                with open(self._model_path, "rb") as f:
                    self._model = pickle.load(f)
                logger.info("Signal classifier loaded from %s", self._model_path)
            except Exception as e:
                logger.warning("Failed to load signal classifier: %s", e)

    def train(self, features: list[list[float]], labels: list[int]) -> bool:
        """Train the classifier on historical trade outcomes.

        features: indicator values at entry time
        labels: 1 = profitable, 0 = unprofitable
        """
        if len(features) < 50:
            return False

        try:
            from xgboost import XGBClassifier

            X = np.array(features)
            y = np.array(labels)
            self._model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )
            self._model.fit(X, y)

            with open(self._model_path, "wb") as f:
                pickle.dump(self._model, f)

            logger.info("Signal classifier trained on %d samples", len(features))
            return True
        except ImportError:
            logger.info("xgboost not installed — signal classification disabled")
            return False
        except Exception as e:
            logger.warning("Signal classifier training failed: %s", e)
            return False

    def predict_confidence(self, indicators: dict) -> float | None:
        """Predict the probability that a trade with these indicators will be profitable."""
        if self._model is None:
            return None

        features = []
        for feat in _FEATURES:
            val = indicators.get(feat)
            if val is None:
                return None
            features.append(float(val))

        try:
            X = np.array([features])
            proba = self._model.predict_proba(X)[0]
            return float(proba[1])
        except Exception:
            return None


def format_ml_signals_for_prompt(
    forecasts_text: str,
    anomalies: dict[str, tuple[bool, float]] | None = None,
    ml_confidence: dict[str, float] | None = None,
) -> str:
    """Format all ML signals into a combined prompt section."""
    lines = []

    if forecasts_text and forecasts_text != "No ML price forecasts available.":
        lines.append("Price Forecasts (Chronos-T5):")
        lines.append(forecasts_text)

    if anomalies:
        anomaly_lines = []
        for pair, (is_anomaly, score) in anomalies.items():
            if is_anomaly:
                anomaly_lines.append(f"  {pair}: ANOMALY DETECTED (score: {score:.3f})")
        if anomaly_lines:
            lines.append("Anomaly Detection:")
            lines.extend(anomaly_lines)

    if ml_confidence:
        conf_lines = []
        for pair, conf in sorted(ml_confidence.items()):
            label = "HIGH" if conf > 0.7 else ("MEDIUM" if conf > 0.5 else "LOW")
            conf_lines.append(f"  {pair}: ML confidence={conf:.0%} ({label})")
        if conf_lines:
            lines.append("Trade Confidence (from our history):")
            lines.extend(conf_lines)

    return "\n".join(lines) if lines else "No ML model data available."
