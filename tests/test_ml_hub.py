"""Tests for :class:`ml.hub.ModelHub` — the lazy-loading model registry.

Small surface but load-bearing: every ML inference path (regime,
forecaster, anomaly, signal classifier) goes through `is_loaded` /
`get_model` to short-circuit when the optional model isn't installed.
"""

from __future__ import annotations

from pathlib import Path

from halal_trader.ml.hub import ModelHub


def test_default_device_is_cpu():
    hub = ModelHub()
    assert hub.device == "cpu"


def test_custom_device_round_trips(tmp_path: Path):
    hub = ModelHub(device="cuda", models_dir=tmp_path)
    assert hub.device == "cuda"


def test_models_dir_is_created_on_init(tmp_path: Path):
    """Init must auto-create the dir — downstream loaders save weights
    there assuming it exists."""
    target = tmp_path / "fresh-subdir"
    assert not target.exists()
    ModelHub(models_dir=target)
    assert target.is_dir()


def test_models_dir_default_path():
    """Default points at `models/` (relative to CWD). The constructor
    will mkdir it if missing — so we just assert the property is set."""
    hub = ModelHub()
    assert hub.models_dir == Path("models")


def test_get_model_returns_none_when_unloaded(tmp_path: Path):
    """Unknown name returns None rather than raising — callers branch
    on this to gracefully degrade."""
    hub = ModelHub(models_dir=tmp_path)
    assert hub.get_model("never-registered") is None


def test_register_makes_model_retrievable(tmp_path: Path):
    hub = ModelHub(models_dir=tmp_path)
    sentinel = object()
    hub.register("regime", sentinel)
    assert hub.get_model("regime") is sentinel


def test_is_loaded_predicate(tmp_path: Path):
    hub = ModelHub(models_dir=tmp_path)
    assert hub.is_loaded("regime") is False
    hub.register("regime", object())
    assert hub.is_loaded("regime") is True


def test_unload_removes_model(tmp_path: Path):
    hub = ModelHub(models_dir=tmp_path)
    hub.register("regime", object())
    hub.unload("regime")
    assert hub.is_loaded("regime") is False
    assert hub.get_model("regime") is None


def test_unload_unknown_name_is_idempotent(tmp_path: Path):
    """Unloading something that was never loaded must not raise —
    cleanup paths call this defensively."""
    hub = ModelHub(models_dir=tmp_path)
    hub.unload("never-registered")  # must not raise


def test_register_overwrites_prior_model(tmp_path: Path):
    """Re-registering the same name swaps the model — used when a
    retrainer hot-swaps a freshly-fit model in place."""
    hub = ModelHub(models_dir=tmp_path)
    first = object()
    second = object()
    hub.register("anomaly", first)
    hub.register("anomaly", second)
    assert hub.get_model("anomaly") is second


def test_register_persists_across_unrelated_names(tmp_path: Path):
    """Loading B doesn't affect A — independent slots."""
    hub = ModelHub(models_dir=tmp_path)
    a = object()
    b = object()
    hub.register("a", a)
    hub.register("b", b)
    assert hub.get_model("a") is a
    assert hub.get_model("b") is b
    assert hub.is_loaded("a") and hub.is_loaded("b")
