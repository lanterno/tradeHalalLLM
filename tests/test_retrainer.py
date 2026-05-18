"""Tests for ml/retrainer.py — labeling closed trades + retrain triggers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from halal_trader.ml.retrainer import RetrainingScheduler


class _FakeRepo:
    def __init__(self, labelled: list[dict[str, Any]] | None = None):
        self.labelled = labelled or []
        self.label_calls: list[tuple[int, int, float]] = []

    async def label_indicator_snapshot(self, trade_id: int, label: int, return_pct: float) -> None:
        self.label_calls.append((trade_id, label, return_pct))

    async def get_labeled_snapshots(self, *, min_samples: int) -> list[dict[str, Any]]:
        return list(self.labelled)


@pytest.mark.asyncio
async def test_on_trade_closed_labels_profitable_correctly():
    repo = _FakeRepo()
    sched = RetrainingScheduler(repo, retrain_every_n_trades=10**6)
    await sched.on_trade_closed(trade_id=42, return_pct=0.025)
    assert repo.label_calls == [(42, 1, 0.025)]


@pytest.mark.asyncio
async def test_on_trade_closed_labels_unprofitable_correctly():
    repo = _FakeRepo()
    sched = RetrainingScheduler(repo, retrain_every_n_trades=10**6)
    await sched.on_trade_closed(trade_id=7, return_pct=-0.01)
    assert repo.label_calls == [(7, 0, -0.01)]


@pytest.mark.asyncio
async def test_retrain_triggered_after_n_trades(monkeypatch):
    """The N-th close kicks retrain; counter resets afterwards."""
    repo = _FakeRepo()
    sched = RetrainingScheduler(repo, retrain_every_n_trades=3, min_samples=10**6)
    fake_retrain = AsyncMock(return_value={"samples": 0})
    monkeypatch.setattr(sched, "retrain", fake_retrain)

    await sched.on_trade_closed(1, 0.01)
    await sched.on_trade_closed(2, 0.01)
    fake_retrain.assert_not_awaited()
    await sched.on_trade_closed(3, 0.01)
    fake_retrain.assert_awaited_once()
    # Counter resets — next 2 closes don't retrigger
    await sched.on_trade_closed(4, 0.01)
    await sched.on_trade_closed(5, 0.01)
    assert fake_retrain.await_count == 1


@pytest.mark.asyncio
async def test_retrain_aborts_when_too_few_snapshots():
    repo = _FakeRepo(labelled=[])  # zero rows from DB
    sched = RetrainingScheduler(repo, min_samples=50)
    result = await sched.retrain()
    assert result == {
        "anomaly": False,
        "classifier": False,
        "slippage": False,
        "samples": 0,
    }


@pytest.mark.asyncio
async def test_retrain_skips_snapshots_with_missing_features(monkeypatch, tmp_path):
    """Snapshots missing any of the 9 features are dropped before training."""
    full = {
        "rsi_14": 50.0,
        "macd_histogram": 0.1,
        "volume_ratio": 1.0,
        "atr_14": 0.02,
        "bb_position": 0.5,
        "ema_9": 100.0,
        "ema_21": 99.0,
        "vwap": 99.5,
        "price_change_5m": 0.001,
        "label": 1,
    }
    missing = dict(full)
    missing["ema_9"] = None  # one None makes the row invalid

    repo = _FakeRepo(labelled=[full, missing] * 30)  # 60 rows, half valid
    sched = RetrainingScheduler(repo, models_dir=tmp_path, min_samples=20)

    # Replace the actual ML training so we can inspect what was passed in.
    captured: dict[str, list] = {}

    class _StubAnomaly:
        def __init__(self, *_, **__):
            self._samples: list = []

        def train(self) -> bool:
            captured["anomaly_samples"] = list(self._samples)
            return True

        async def persist_model(self) -> None:
            # Wave K: retrainer awaits this after a successful train().
            captured["anomaly_persisted"] = True

    class _StubClassifier:
        def __init__(self, *_, **__):
            pass

        def train(self, features: list, labels: list) -> bool:
            captured["features"] = features
            captured["labels"] = labels
            return True

        async def persist_model(self) -> None:
            captured["classifier_persisted"] = True

    monkeypatch.setattr("halal_trader.ml.anomaly.MarketAnomalyDetector", _StubAnomaly)
    monkeypatch.setattr("halal_trader.ml.anomaly.MLSignalClassifier", _StubClassifier)

    result = await sched.retrain()
    assert result["anomaly"] is True
    assert result["classifier"] is True
    # Only the 30 valid rows survive after the missing-feature filter.
    assert len(captured["features"]) == 30
    assert len(captured["anomaly_samples"]) == 30


@pytest.mark.asyncio
async def test_label_failure_does_not_abort(monkeypatch):
    """A repo that raises on labelling shouldn't blow up on_trade_closed."""

    class _BrokenRepo(_FakeRepo):
        async def label_indicator_snapshot(self, *args, **kwargs):
            raise RuntimeError("DB locked")

    repo = _BrokenRepo()
    sched = RetrainingScheduler(repo, retrain_every_n_trades=10**6)
    await sched.on_trade_closed(1, 0.05)  # must not raise
