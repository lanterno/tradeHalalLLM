"""Tests for online ML state persistence (warm-start)."""

from __future__ import annotations

import pickle
import time

from halal_trader.ml.anomaly import _FEATURES, _MODEL_VERSION, MarketAnomalyDetector
from halal_trader.ml.hub import ModelHub


def _hub(tmp_path):
    return ModelHub(models_dir=tmp_path)


def _full_indicators(seed: float = 0.5) -> dict:
    return {feat: seed + i * 0.01 for i, feat in enumerate(_FEATURES)}


def test_state_persists_after_n_additions(tmp_path):
    """Buffer is flushed to disk every PERSIST_EVERY_N additions."""
    detector = MarketAnomalyDetector(_hub(tmp_path), min_samples=1000)
    state_path = tmp_path / "anomaly_state.pkl"

    # Below the threshold — no flush yet.
    for _ in range(MarketAnomalyDetector._PERSIST_EVERY_N - 1):
        detector.add_sample(_full_indicators())
    assert not state_path.exists()

    # The next add crosses the threshold and flushes.
    detector.add_sample(_full_indicators())
    assert state_path.exists()


def test_warm_start_restores_buffer(tmp_path):
    """A fresh detector picks up the previous instance's buffer + timestamp."""
    hub = _hub(tmp_path)

    first = MarketAnomalyDetector(hub, min_samples=1000)
    for _ in range(MarketAnomalyDetector._PERSIST_EVERY_N):
        first.add_sample(_full_indicators(seed=0.7))
    first._last_trained_at = 12345.0
    first._save_state()

    second = MarketAnomalyDetector(hub, min_samples=1000)
    assert len(second._samples) == MarketAnomalyDetector._PERSIST_EVERY_N
    assert second._last_trained_at == 12345.0


def test_warm_start_caps_samples_at_max_buffer(tmp_path):
    hub = _hub(tmp_path)
    state_path = tmp_path / "anomaly_state.pkl"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    over = MarketAnomalyDetector._MAX_BUFFER_SIZE + 100
    payload = {
        "version": _MODEL_VERSION,
        "features": list(_FEATURES),
        "samples": [[0.0] * len(_FEATURES) for _ in range(over)],
        "last_trained_at": time.time(),
    }
    with open(state_path, "wb") as f:
        pickle.dump(payload, f)

    detector = MarketAnomalyDetector(hub, min_samples=1000)
    assert len(detector._samples) == MarketAnomalyDetector._MAX_BUFFER_SIZE


def test_state_with_wrong_version_is_discarded(tmp_path):
    hub = _hub(tmp_path)
    state_path = tmp_path / "anomaly_state.pkl"
    payload = {
        "version": _MODEL_VERSION + 99,
        "features": list(_FEATURES),
        "samples": [[0.5] * len(_FEATURES)],
        "last_trained_at": 1.0,
    }
    with open(state_path, "wb") as f:
        pickle.dump(payload, f)

    detector = MarketAnomalyDetector(hub, min_samples=1000)
    assert detector._samples == []
    assert detector._last_trained_at is None


def test_state_with_wrong_features_is_discarded(tmp_path):
    hub = _hub(tmp_path)
    state_path = tmp_path / "anomaly_state.pkl"
    payload = {
        "version": _MODEL_VERSION,
        "features": _FEATURES[:5],  # old 5-feature shape
        "samples": [[0.5] * 5],
        "last_trained_at": 1.0,
    }
    with open(state_path, "wb") as f:
        pickle.dump(payload, f)

    detector = MarketAnomalyDetector(hub, min_samples=1000)
    assert detector._samples == []


def test_auto_train_is_noop_below_min_samples(tmp_path):
    detector = MarketAnomalyDetector(_hub(tmp_path), min_samples=50)
    detector.add_sample(_full_indicators())
    assert detector.auto_train() is False
    assert detector._model is None


def test_buffer_capped_at_max_size_during_adds(tmp_path):
    detector = MarketAnomalyDetector(_hub(tmp_path), min_samples=10**9)
    for _ in range(MarketAnomalyDetector._MAX_BUFFER_SIZE + 50):
        detector.add_sample(_full_indicators())
    assert len(detector._samples) == MarketAnomalyDetector._MAX_BUFFER_SIZE
