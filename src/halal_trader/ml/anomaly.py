"""Market anomaly detection — IsolationForest on indicator snapshots."""

from __future__ import annotations

import asyncio
import logging
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np

from halal_trader.ml.features import FEATURE_KEYS
from halal_trader.ml.hub import ModelHub

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# Backwards-compat alias; the canonical list now lives in ml/features.py.
_FEATURES = list(FEATURE_KEYS)

# Bump on any feature-set / preprocessing change. Pickles tagged with
# a different version are discarded on load so retraining picks up the
# new shape automatically.
_MODEL_VERSION = 2

# Wave K — artefact names that key into the ``ml_artefacts`` table.
# Bump these only when you intentionally orphan all prior DB rows
# (rare). The ``_MODEL_VERSION`` above orphans pickles on a feature
# change; the artefact name is the DB-side equivalent.
_ANOMALY_ARTEFACT_NAME = "anomaly_detector"
_ANOMALY_STATE_ARTEFACT_NAME = "anomaly_state"
_SIGNAL_ARTEFACT_NAME = "signal_classifier"

# Pickled payload format (both detector + classifier):
#   {"version": int, "features": list[str], "model": <sklearn / xgb model>}


class MarketAnomalyDetector:
    """Detects unusual market microstructure using IsolationForest."""

    # Persist incremental sample buffer + last-train timestamp every
    # _PERSIST_EVERY_N additions so a restart doesn't lose the warm-up.
    _PERSIST_EVERY_N = 25
    _MAX_BUFFER_SIZE = 5000

    def __init__(
        self,
        hub: ModelHub,
        *,
        min_samples: int = 100,
        engine: "AsyncEngine | None" = None,
    ) -> None:
        self._hub = hub
        self._min_samples = min_samples
        self._model = None
        self._samples: list[list[float]] = []
        self._last_trained_at: float | None = None
        self._adds_since_persist = 0
        self._model_path = hub.models_dir / "anomaly_detector.pkl"
        self._state_path = hub.models_dir / "anomaly_state.pkl"
        # Wave K: when an engine is set, save/load goes through the
        # ``ml_artefacts`` table instead of the disk pickle. The disk
        # path stays as a one-shot fallback for in-flight installs.
        # The buffer-state cache (samples + last_trained_at) also moves
        # to the DB when wired; ``load_state_from_db`` is called from
        # the composition root alongside ``load_latest``.
        self._engine = engine
        # Handle to the most recent fire-and-forget DB save. Kept on the
        # instance so the task isn't garbage-collected mid-flight (an
        # un-referenced create_task() can be), and so tests can await it
        # deterministically instead of sleeping.
        self._save_task: "asyncio.Task[None] | None" = None
        self._load_model()
        if engine is None:
            self._load_state()

    def _load_model(self) -> None:
        """Load a previously trained model from disk."""
        if not self._model_path.exists():
            return
        try:
            with open(self._model_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning("Failed to load anomaly model: %s", e)
            return

        if not _is_current_payload(payload):
            logger.info(
                "Discarding stale anomaly model at %s (version mismatch)",
                self._model_path,
            )
            return
        self._model = payload["model"]
        logger.info("Anomaly detector loaded from %s", self._model_path)

    def _load_state(self) -> None:
        """Restore the incremental sample buffer + last-train timestamp."""
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path, "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            logger.warning("Failed to load anomaly state: %s", e)
            return

        if not isinstance(state, dict):
            return
        if state.get("version") != _MODEL_VERSION:
            logger.info(
                "Discarding stale anomaly state at %s (version mismatch)",
                self._state_path,
            )
            return
        if list(state.get("features") or []) != list(_FEATURES):
            return

        samples = state.get("samples")
        if isinstance(samples, list):
            self._samples = [list(s) for s in samples][-self._MAX_BUFFER_SIZE :]
        last_trained = state.get("last_trained_at")
        if isinstance(last_trained, (int, float)):
            self._last_trained_at = float(last_trained)
        logger.info(
            "Anomaly state restored: %d samples buffered, last_trained_at=%s",
            len(self._samples),
            self._last_trained_at,
        )

    def _save_state(self) -> None:
        """Persist the sample buffer + metadata so a restart resumes warm.

        DB-first when ``engine`` is wired: schedules an async write to
        the ``ml_artefacts`` table as a JSON payload (the samples are
        list-of-float, JSON-serialises cleanly). Falls back to the
        legacy disk pickle when no engine — keeps dev / test contexts
        without Postgres working.
        """
        payload = {
            "version": _MODEL_VERSION,
            "features": list(_FEATURES),
            "samples": self._samples,
            "last_trained_at": self._last_trained_at,
        }
        if self._engine is not None:
            # Fire-and-forget — _save_state is sync (called from
            # add_sample, sync path). add_sample runs inside the
            # cycle's async stage, so a running loop is available. Keep the
            # task handle so it isn't GC'd mid-write and tests can await it.
            try:
                loop = asyncio.get_running_loop()
                self._save_task = loop.create_task(self._save_state_to_db(payload))
                return
            except RuntimeError:
                # No running loop — fall through to disk path.
                logger.debug("anomaly _save_state: no running loop, using disk")
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_path, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            logger.debug("Could not persist anomaly state: %s", e)

    async def _save_state_to_db(self, payload: dict[str, Any]) -> None:
        """Async tail of :meth:`_save_state` — writes the JSON artefact row."""
        if self._engine is None:
            return
        try:
            from halal_trader.db.ml_artefacts import save_artefact

            await save_artefact(
                engine=self._engine,
                name=_ANOMALY_STATE_ARTEFACT_NAME,
                payload_json=payload,
                feature_hash=str(_MODEL_VERSION),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("anomaly state DB save failed: %s", exc)

    async def load_state_from_db(self) -> bool:
        """Restore the buffer from the latest ``anomaly_state`` DB row.

        Mirrors :meth:`load_latest` for the warm-up cache. Called from
        the composition root after constructing the detector. Returns
        True when a row was found + loaded.
        """
        if self._engine is None:
            return False
        try:
            from halal_trader.db.ml_artefacts import load_artefact

            state = await load_artefact(engine=self._engine, name=_ANOMALY_STATE_ARTEFACT_NAME)
            if state is None or not isinstance(state, dict):
                return False
            if state.get("version") != _MODEL_VERSION:
                logger.info("Discarding stale anomaly state DB row (version mismatch)")
                return False
            if list(state.get("features") or []) != list(_FEATURES):
                return False
            samples = state.get("samples")
            if isinstance(samples, list):
                self._samples = [list(s) for s in samples][-self._MAX_BUFFER_SIZE :]
            last_trained = state.get("last_trained_at")
            if isinstance(last_trained, (int, float)):
                self._last_trained_at = float(last_trained)
            logger.info(
                "Anomaly state restored from DB: %d samples buffered, last_trained_at=%s",
                len(self._samples),
                self._last_trained_at,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("anomaly state DB load failed: %s", exc)
            return False

    def add_sample(self, indicators: dict) -> None:
        """Add an indicator snapshot as a training sample."""
        features = self._extract_features(indicators)
        if features is None:
            return
        self._samples.append(features)
        # Cap buffer so memory doesn't grow unbounded.
        if len(self._samples) > self._MAX_BUFFER_SIZE:
            self._samples = self._samples[-self._MAX_BUFFER_SIZE :]
        self._adds_since_persist += 1
        if self._adds_since_persist >= self._PERSIST_EVERY_N:
            self._save_state()
            self._adds_since_persist = 0

    def train(self) -> bool:
        """Train the IsolationForest on collected samples.

        Wave K: training only updates the in-memory model; durable
        persistence lives in :meth:`persist_model` (called by the
        retrainer after a successful train). When no engine is wired,
        ``persist_model`` falls back to the legacy disk path.
        """
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
            import time as _time

            self._last_trained_at = _time.time()
            self._save_state()
            logger.info("Anomaly detector trained on %d samples", len(X))
            return True
        except ImportError:
            logger.info("scikit-learn not installed — anomaly detection disabled")
            return False
        except Exception as e:
            logger.warning("Anomaly detector training failed: %s", e)
            return False

    async def persist_model(self) -> bool:
        """Durable model write — DB-first when ``engine`` is set,
        legacy disk pickle otherwise.

        Called by the retrainer after a successful :meth:`train`.
        Returns True when the persist succeeded by either path.
        """
        if self._model is None:
            return False
        if self._engine is not None:
            try:
                from halal_trader.db.ml_artefacts import (
                    pickle_dumps,
                    save_artefact,
                )

                await save_artefact(
                    engine=self._engine,
                    name=_ANOMALY_ARTEFACT_NAME,
                    payload_bytes=pickle_dumps(_versioned_payload(self._model)),
                    feature_hash=str(_MODEL_VERSION),
                )
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("anomaly DB persist failed (falling back to file): %s", exc)
        try:
            with open(self._model_path, "wb") as f:
                pickle.dump(_versioned_payload(self._model), f)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("anomaly file persist failed: %s", exc)
            return False

    async def load_latest(self) -> bool:
        """DB-first refresh — replaces whatever ``__init__`` loaded from disk.

        Returns True when a DB row was found and loaded. When no engine
        is wired, returns False without touching the disk model — that
        load already ran in ``__init__``.
        """
        if self._engine is None:
            return False
        try:
            from halal_trader.db.ml_artefacts import load_artefact_bytes

            payload_bytes = await load_artefact_bytes(
                engine=self._engine, name=_ANOMALY_ARTEFACT_NAME
            )
            if payload_bytes is None:
                return False
            payload = pickle.loads(payload_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("anomaly DB load failed: %s", exc)
            return False
        if not _is_current_payload(payload):
            logger.info("Discarding stale anomaly DB row (version mismatch)")
            return False
        self._model = payload["model"]
        logger.info("Anomaly detector loaded from ml_artefacts DB row")
        return True

    def auto_train(self) -> bool:
        """Train if enough samples have accumulated since the last training.

        Used by the hot path so we don't need a separate scheduler tick.
        Returns ``True`` when training actually ran.
        """
        if len(self._samples) < self._min_samples:
            return False
        return self.train()

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

    def __init__(self, hub: ModelHub, *, engine: "AsyncEngine | None" = None) -> None:
        self._hub = hub
        self._model = None
        self._model_path = hub.models_dir / "signal_classifier.pkl"
        self._samples: list[list[float]] = []
        self._labels: list[int] = []
        self._engine = engine
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
        if not self._model_path.exists():
            return
        try:
            with open(self._model_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning("Failed to load signal classifier: %s", e)
            return

        if not _is_current_payload(payload):
            logger.info(
                "Discarding stale signal classifier at %s (version mismatch)",
                self._model_path,
            )
            return
        self._model = payload["model"]
        logger.info("Signal classifier loaded from %s", self._model_path)

    def train(self, features: list[list[float]], labels: list[int]) -> bool:
        """Train the classifier on historical trade outcomes.

        features: indicator values at entry time
        labels: 1 = profitable, 0 = unprofitable

        Wave K: only updates the in-memory model; durable persistence
        lives in :meth:`persist_model` (called by the retrainer).
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
            logger.info("Signal classifier trained on %d samples", len(features))
            return True
        except ImportError:
            logger.info("xgboost not installed — signal classification disabled")
            return False
        except Exception as e:
            logger.warning("Signal classifier training failed: %s", e)
            return False

    async def persist_model(self) -> bool:
        """Durable model write — DB-first when ``engine`` is set, file fallback."""
        if self._model is None:
            return False
        if self._engine is not None:
            try:
                from halal_trader.db.ml_artefacts import (
                    pickle_dumps,
                    save_artefact,
                )

                await save_artefact(
                    engine=self._engine,
                    name=_SIGNAL_ARTEFACT_NAME,
                    payload_bytes=pickle_dumps(_versioned_payload(self._model)),
                    feature_hash=str(_MODEL_VERSION),
                )
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("signal classifier DB persist failed: %s", exc)
        try:
            with open(self._model_path, "wb") as f:
                pickle.dump(_versioned_payload(self._model), f)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("signal classifier file persist failed: %s", exc)
            return False

    async def load_latest(self) -> bool:
        """DB-first refresh — replaces whatever ``__init__`` loaded from disk."""
        if self._engine is None:
            return False
        try:
            from halal_trader.db.ml_artefacts import load_artefact_bytes

            payload_bytes = await load_artefact_bytes(
                engine=self._engine, name=_SIGNAL_ARTEFACT_NAME
            )
            if payload_bytes is None:
                return False
            payload = pickle.loads(payload_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("signal classifier DB load failed: %s", exc)
            return False
        if not _is_current_payload(payload):
            logger.info("Discarding stale signal classifier DB row (version mismatch)")
            return False
        self._model = payload["model"]
        logger.info("Signal classifier loaded from ml_artefacts DB row")
        return True

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


def _versioned_payload(model: object) -> dict:
    """Wrap ``model`` with a (version, features) tag so loaders can validate it."""
    return {"version": _MODEL_VERSION, "features": list(_FEATURES), "model": model}


def _is_current_payload(payload: object) -> bool:
    """Return True if the loaded pickle is tagged for the current feature set."""
    if not isinstance(payload, dict):
        return False
    if payload.get("version") != _MODEL_VERSION:
        return False
    if list(payload.get("features") or []) != list(_FEATURES):
        return False
    return "model" in payload


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


def build_ml_signals_text(
    *,
    indicators_by_symbol: dict[str, dict[str, Any]],
    anomaly_detector: Any | None = None,
    signal_classifier: Any | None = None,
    forecasts_text: str = "",
) -> str:
    """Run anomaly + signal classifier over the per-symbol indicators.

    Shared between :class:`CryptoCycleService` and ``TradingCycleService``
    so both bots produce identical ML-signal blocks. Symbols whose
    indicators carry an ``error`` key are skipped — their bars failed
    parse / didn't have enough history.

    ``forecasts_text`` is an optional pre-computed forecasts block (the
    crypto cycle runs Chronos before this; stocks doesn't have enough
    daily-bar history for the forecaster). When all three signals are
    absent, returns ``forecasts_text`` so any pre-computed forecast
    survives.
    """
    if not anomaly_detector and not signal_classifier:
        return forecasts_text
    if not indicators_by_symbol:
        return forecasts_text
    try:
        anomalies: dict[str, tuple[bool, float]] = {}
        ml_confidence: dict[str, float] = {}
        for symbol, indicators in indicators_by_symbol.items():
            if not indicators or "error" in indicators:
                continue
            if anomaly_detector:
                anomaly_detector.add_sample(indicators)
                anomalies[symbol] = anomaly_detector.detect(indicators)
            if signal_classifier:
                conf = signal_classifier.predict_confidence(indicators)
                if conf is not None:
                    ml_confidence[symbol] = conf
        return format_ml_signals_for_prompt(
            forecasts_text, anomalies or None, ml_confidence or None
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("ML signals unavailable: %s", exc)
        return forecasts_text
