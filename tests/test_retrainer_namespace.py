"""Retrainer namespace tests — separate model dirs for crypto vs stocks."""

from __future__ import annotations

from pathlib import Path

from halal_trader.ml.retrainer import RetrainingScheduler


def _repo_stub() -> object:
    class _R:
        pass

    return _R()


def test_default_namespace_is_crypto(tmp_path: Path):
    sched = RetrainingScheduler(repo=_repo_stub(), models_dir=tmp_path)
    assert sched.namespace == "crypto"
    assert sched._models_dir == tmp_path / "crypto"


def test_explicit_stock_namespace_isolates_dir(tmp_path: Path):
    sched = RetrainingScheduler(repo=_repo_stub(), models_dir=tmp_path, namespace="stock")
    assert sched.namespace == "stock"
    assert sched._models_dir == tmp_path / "stock"


def test_separate_schedulers_use_disjoint_dirs(tmp_path: Path):
    crypto = RetrainingScheduler(repo=_repo_stub(), models_dir=tmp_path)
    stock = RetrainingScheduler(repo=_repo_stub(), models_dir=tmp_path, namespace="stock")
    assert crypto._models_dir != stock._models_dir
    assert crypto._models_dir.parent == stock._models_dir.parent
