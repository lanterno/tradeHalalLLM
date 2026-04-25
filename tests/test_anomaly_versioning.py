"""Tests for ML pickle versioning — stale 5-feature pickles are auto-discarded."""

from __future__ import annotations

import pickle

import pytest

from halal_trader.ml.anomaly import (
    _FEATURES,
    _MODEL_VERSION,
    MarketAnomalyDetector,
    MLSignalClassifier,
    _is_current_payload,
    _versioned_payload,
)


def test_features_now_include_all_nine():
    assert _FEATURES == [
        "rsi_14",
        "macd_histogram",
        "volume_ratio",
        "atr_14",
        "bb_position",
        "ema_9",
        "ema_21",
        "vwap",
        "price_change_5m",
    ]


def test_versioned_payload_round_trips():
    payload = _versioned_payload({"fake": "model"})
    assert payload["version"] == _MODEL_VERSION
    assert payload["features"] == _FEATURES
    assert _is_current_payload(payload)


def test_is_current_payload_rejects_old_format():
    # Pre-versioning pickles were just the bare model — no dict wrapper.
    assert not _is_current_payload(b"raw bytes")
    assert not _is_current_payload({"model": "foo"})  # missing version
    assert not _is_current_payload({"version": 1, "features": _FEATURES, "model": "x"})
    assert not _is_current_payload(
        {"version": _MODEL_VERSION, "features": _FEATURES[:5], "model": "x"}
    )


def _hub(tmp_path):
    from halal_trader.ml.hub import ModelHub

    return ModelHub(models_dir=tmp_path)


def test_anomaly_load_skips_stale_pickle(tmp_path):
    """A pickle without the version wrapper is treated as missing."""
    hub = _hub(tmp_path)
    stale_path = hub.models_dir / "anomaly_detector.pkl"
    with open(stale_path, "wb") as f:
        pickle.dump({"old": "format"}, f)

    detector = MarketAnomalyDetector(hub, min_samples=2)
    assert detector._model is None  # rejected — caller will retrain


def test_anomaly_load_accepts_current_pickle(tmp_path):
    hub = _hub(tmp_path)
    sentinel = {"i_am": "the_loaded_model"}
    with open(hub.models_dir / "anomaly_detector.pkl", "wb") as f:
        pickle.dump(_versioned_payload(sentinel), f)

    detector = MarketAnomalyDetector(hub, min_samples=2)
    assert detector._model == sentinel


def test_classifier_load_skips_stale_pickle(tmp_path):
    hub = _hub(tmp_path)
    stale = hub.models_dir / "signal_classifier.pkl"
    with open(stale, "wb") as f:
        pickle.dump("just a model", f)

    classifier = MLSignalClassifier(hub)
    assert classifier._model is None


def test_classifier_load_accepts_current_pickle(tmp_path):
    hub = _hub(tmp_path)
    sentinel = {"i_am": "the_loaded_model"}
    with open(hub.models_dir / "signal_classifier.pkl", "wb") as f:
        pickle.dump(_versioned_payload(sentinel), f)

    classifier = MLSignalClassifier(hub)
    assert classifier._model == sentinel


def test_anomaly_extract_features_requires_all_nine(tmp_path):
    hub = _hub(tmp_path)
    detector = MarketAnomalyDetector(hub)
    full = {feat: 0.5 for feat in _FEATURES}
    assert detector._extract_features(full) == [0.5] * 9
    # Missing one new feature → returns None so the model isn't fed bad data.
    minus_one = dict(full)
    minus_one.pop("ema_9")
    assert detector._extract_features(minus_one) is None
