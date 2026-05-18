"""Wave K wiring tests — ml artefacts moved from .pkl to the DB.

The ``ml_artefacts`` table + ``save_artefact`` / ``load_artefact`` /
``list_versions`` / ``pickle_dumps`` helpers are covered by
``tests/test_ml_artefacts.py``. This file covers the *new wiring*:

* ``load_artefact_bytes`` raw-bytes loader the ML classes use.
* Each ML class's ``persist_model`` writes through the DB when an
  engine is wired, falls back to the legacy disk pickle otherwise.
* ``load_latest`` reads the latest DB row and overwrites whatever the
  legacy ``__init__`` loaded from disk.
* The CLI ``halal-trader ml versions`` group registers and parses.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _require_xgboost_runtime() -> None:
    """Skip the test cleanly when xgboost can't load its native lib.

    ``import xgboost`` triggers ``_load_lib()`` at module-import time;
    on a macOS dev box without ``brew install libomp`` that raises
    ``xgboost.core.XGBoostError``. We need a guard that survives
    BOTH ImportError (extra not installed) and XGBoostError
    (libomp missing) without eating pytest.skip's own exception.
    """
    try:
        import xgboost  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"xgboost unavailable: {exc}")


# ── load_artefact_bytes helper ──────────────────────────────────


async def test_load_artefact_bytes_returns_raw_pickle(engine) -> None:
    """Anomaly + signal + regime go through the bytes loader — confirm
    it returns the latest pickle payload as-is for the caller to unpack."""
    from halal_trader.db.ml_artefacts import (
        load_artefact_bytes,
        pickle_dumps,
        save_artefact,
    )

    blob = pickle_dumps({"model": "fake", "v": 1})
    await save_artefact(engine=engine, name="anomaly_detector", payload_bytes=blob)
    out = await load_artefact_bytes(engine=engine, name="anomaly_detector")
    assert out == blob


async def test_load_artefact_bytes_picks_latest_version(engine) -> None:
    """Multiple rows under the same name → newest wins."""
    from halal_trader.db.ml_artefacts import (
        load_artefact_bytes,
        pickle_dumps,
        save_artefact,
    )

    await save_artefact(engine=engine, name="regime_classifier", payload_bytes=pickle_dumps("v1"))
    await save_artefact(engine=engine, name="regime_classifier", payload_bytes=pickle_dumps("v2"))
    out = await load_artefact_bytes(engine=engine, name="regime_classifier")
    assert out == pickle_dumps("v2")


async def test_load_artefact_bytes_returns_none_for_json_rows(engine) -> None:
    """A JSON-payload row (e.g. slippage model) is *not* returned by
    the bytes loader — callers that expect a pickle should fall through
    to defaults rather than try to ``pickle.loads`` a JSON string."""
    from halal_trader.db.ml_artefacts import load_artefact_bytes, save_artefact

    await save_artefact(engine=engine, name="slippage_v1", payload_json={"k": "v"})
    out = await load_artefact_bytes(engine=engine, name="slippage_v1")
    assert out is None


async def test_load_artefact_bytes_returns_none_when_missing(engine) -> None:
    from halal_trader.db.ml_artefacts import load_artefact_bytes

    out = await load_artefact_bytes(engine=engine, name="never_saved")
    assert out is None


# ── MarketAnomalyDetector DB persistence ────────────────────────


async def test_anomaly_detector_persists_to_db_when_engine_wired(engine, tmp_path) -> None:
    """train() updates in-memory; persist_model() lands a DB row."""
    pytest.importorskip("sklearn")
    from halal_trader.db.ml_artefacts import list_versions
    from halal_trader.ml.anomaly import MarketAnomalyDetector
    from halal_trader.ml.hub import ModelHub

    hub = ModelHub(models_dir=tmp_path)
    det = MarketAnomalyDetector(hub, min_samples=10, engine=engine)
    # Seed enough samples to train.
    for i in range(20):
        det._samples.append([float(j + i) for j in range(7)])
    assert det.train() is True

    # train() must NOT have written a disk pickle anymore — DB only.
    assert not (tmp_path / "anomaly_detector.pkl").exists()

    persisted = await det.persist_model()
    assert persisted is True
    rows = await list_versions(engine=engine, name="anomaly_detector")
    assert rows, "anomaly_detector row missing after persist_model"


async def test_anomaly_detector_load_latest_overwrites_disk_model(engine, tmp_path) -> None:
    """Even when a fresh disk pickle exists, ``load_latest`` should
    take the DB row (which is by definition the newer artefact)."""
    pytest.importorskip("sklearn")
    # Train a fresh model to use as the DB row payload.
    from sklearn.ensemble import IsolationForest

    from halal_trader.db.ml_artefacts import pickle_dumps, save_artefact
    from halal_trader.ml.anomaly import (
        MarketAnomalyDetector,
        _versioned_payload,
    )
    from halal_trader.ml.hub import ModelHub

    db_model = IsolationForest(n_estimators=5, random_state=42)
    import numpy as np

    db_model.fit(np.random.rand(50, 7))

    await save_artefact(
        engine=engine,
        name="anomaly_detector",
        payload_bytes=pickle_dumps(_versioned_payload(db_model)),
    )

    hub = ModelHub(models_dir=tmp_path)
    det = MarketAnomalyDetector(hub, min_samples=10, engine=engine)
    # __init__ saw no disk pickle, so _model is None.
    assert det._model is None
    assert await det.load_latest() is True
    assert det._model is not None  # came from DB


async def test_anomaly_detector_no_engine_falls_back_to_disk(tmp_path) -> None:
    """Default-wired anomaly detector with no engine still writes the
    disk pickle (in-flight installs / dev / tests without Postgres)."""
    pytest.importorskip("sklearn")
    from halal_trader.ml.anomaly import MarketAnomalyDetector
    from halal_trader.ml.hub import ModelHub

    hub = ModelHub(models_dir=tmp_path)
    det = MarketAnomalyDetector(hub, min_samples=10, engine=None)
    for i in range(20):
        det._samples.append([float(j + i) for j in range(7)])
    det.train()
    # train() no longer writes disk; persist_model() does.
    assert not (tmp_path / "anomaly_detector.pkl").exists()
    persisted = await det.persist_model()
    assert persisted is True
    assert (tmp_path / "anomaly_detector.pkl").exists()


async def test_anomaly_load_latest_returns_false_without_engine(tmp_path) -> None:
    from halal_trader.ml.anomaly import MarketAnomalyDetector
    from halal_trader.ml.hub import ModelHub

    det = MarketAnomalyDetector(ModelHub(models_dir=tmp_path), engine=None)
    assert await det.load_latest() is False


# ── MLSignalClassifier DB persistence ───────────────────────────


async def test_signal_classifier_persists_to_db(engine, tmp_path) -> None:
    _require_xgboost_runtime()
    from halal_trader.db.ml_artefacts import list_versions
    from halal_trader.ml.anomaly import MLSignalClassifier
    from halal_trader.ml.hub import ModelHub

    hub = ModelHub(models_dir=tmp_path)
    clf = MLSignalClassifier(hub, engine=engine)
    features = [[float(i + j) for j in range(7)] for i in range(60)]
    labels = [i % 2 for i in range(60)]
    assert clf.train(features, labels) is True
    assert not (tmp_path / "signal_classifier.pkl").exists()
    assert await clf.persist_model() is True
    rows = await list_versions(engine=engine, name="signal_classifier")
    assert rows, "signal_classifier row missing after persist_model"


async def test_signal_classifier_persist_without_engine_writes_disk(
    tmp_path,
) -> None:
    _require_xgboost_runtime()
    from halal_trader.ml.anomaly import MLSignalClassifier
    from halal_trader.ml.hub import ModelHub

    hub = ModelHub(models_dir=tmp_path)
    clf = MLSignalClassifier(hub, engine=None)
    features = [[float(i + j) for j in range(7)] for i in range(60)]
    labels = [i % 2 for i in range(60)]
    clf.train(features, labels)
    await clf.persist_model()
    assert (tmp_path / "signal_classifier.pkl").exists()


# ── RegimeDetector DB persistence ───────────────────────────────


async def test_regime_detector_persists_to_db(engine, tmp_path) -> None:
    pytest.importorskip("sklearn")
    from halal_trader.crypto.regime import RegimeDetector
    from halal_trader.db.ml_artefacts import list_versions

    det = RegimeDetector(models_dir=tmp_path, engine=engine)
    # Use the regime detector's `train` API. The detector's
    # ``_extract_features`` requires several keys; build full-shaped
    # samples.
    # RegimeDetector._extract_features requires:
    # rsi_14, macd_histogram, volume_ratio, atr_14, bb_position
    # (adx_14 + bb_upper/bb_lower are optional with defaults).
    sample = {
        "rsi_14": 50.0,
        "macd_histogram": 0.1,
        "bb_position": 0.5,
        "atr_14": 100.0,
        "volume_ratio": 1.0,
        "adx_14": 25.0,
        "bb_upper": 51_000.0,
        "bb_lower": 49_000.0,
        "ema_9": 50_000.0,
        "ema_50": 50_000.0,
        "current_price": 50_000.0,
    }
    samples = [dict(sample) for _ in range(250)]
    labels = ["trending_up"] * 125 + ["ranging"] * 125
    assert det.train(samples, labels) is True
    # train() must NOT have written disk pickle (Wave K).
    assert not (tmp_path / "regime_classifier.pkl").exists()
    assert await det.persist_model() is True
    rows = await list_versions(engine=engine, name="regime_classifier")
    assert rows, "regime_classifier row missing after persist_model"


async def test_regime_detector_load_latest_picks_db_over_disk(engine, tmp_path) -> None:
    """A stale disk pickle exists but the DB has a newer row — DB wins."""
    pytest.importorskip("sklearn")
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    from halal_trader.crypto.regime import RegimeDetector
    from halal_trader.db.ml_artefacts import pickle_dumps, save_artefact

    db_model = RandomForestClassifier(n_estimators=5, random_state=42)
    db_model.fit(np.random.rand(50, 7), np.random.randint(0, 2, 50))

    await save_artefact(
        engine=engine,
        name="regime_classifier",
        payload_bytes=pickle_dumps(db_model),
    )

    det = RegimeDetector(models_dir=tmp_path, engine=engine)
    # Nothing on disk → __init__ left _ml_model as None.
    assert det._ml_model is None
    assert await det.load_latest() is True
    assert det._ml_model is not None


async def test_regime_load_latest_returns_false_without_engine(tmp_path) -> None:
    from halal_trader.crypto.regime import RegimeDetector

    det = RegimeDetector(models_dir=tmp_path, engine=None)
    assert await det.load_latest() is False


# ── CLI ml versions ─────────────────────────────────────────────


def test_cli_ml_group_registered() -> None:
    """`halal-trader ml versions` is available."""
    from halal_trader.cli import cli

    cmds = list(cli.commands.keys())
    assert "ml" in cmds
    ml_grp = cli.commands["ml"]
    assert "versions" in list(ml_grp.commands.keys())


def test_cli_ml_versions_help_parses() -> None:
    from click.testing import CliRunner

    from halal_trader.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["ml", "versions", "--help"])
    assert result.exit_code == 0
    assert "ml_artefacts" in result.output


# ── Acceptance bar: no production .pkl writes when engine is wired ──


async def test_train_with_engine_does_not_write_pkl(engine, tmp_path) -> None:
    """Wave K acceptance: ``find data models -name '*.pkl'`` returns
    no results in production. With an engine wired, neither train()
    nor persist_model() touches the legacy disk path."""
    pytest.importorskip("sklearn")
    from halal_trader.ml.anomaly import MarketAnomalyDetector
    from halal_trader.ml.hub import ModelHub

    hub = ModelHub(models_dir=tmp_path)
    det = MarketAnomalyDetector(hub, min_samples=10, engine=engine)
    for i in range(20):
        det._samples.append([float(j + i) for j in range(7)])
    det.train()
    await det.persist_model()
    # Acceptance: no MODEL pkl on disk when DB wired. The trainer's
    # sample-buffer warm-up cache (``anomaly_state.pkl``) is *not* a
    # versioned model artefact — it's a transient incremental-training
    # cache and stays on disk for now (deferred in cleanup_roadmap.md).
    model_pkls = [p for p in Path(tmp_path).glob("*.pkl") if not p.name.endswith("_state.pkl")]
    assert model_pkls == [], f"Wave K leaked model pickle files: {model_pkls}"
