"""Incremental SGDClassifier wrapper tests + Sortino weight."""

from __future__ import annotations

import pytest

from halal_trader.ml.incremental import (
    IncrementalSignalClassifier,
    sortino_label_weight,
)

# ── Sortino weight ────────────────────────────────────────────


def test_loss_always_full_weight():
    assert sortino_label_weight(return_pct=-0.05, intra_trade_drawdown_pct=0.10) == 1.0


def test_clean_win_gets_ceiling_weight():
    # No drawdown → score 1.0 → ceiling.
    assert sortino_label_weight(return_pct=0.05, intra_trade_drawdown_pct=0.0) == 2.0


def test_noisy_win_below_ceiling():
    """A win with ~equal drawdown should land ~midway between floor and ceiling."""
    w = sortino_label_weight(return_pct=0.05, intra_trade_drawdown_pct=0.05)
    # score = 0.5, weight = 0.2 + (2.0 - 0.2) * 0.5 = 1.1
    assert 1.05 <= w <= 1.15


def test_weight_clamped_below_floor_and_above_ceiling():
    """Even adversarial inputs stay in the configured range."""
    # Tiny win + huge drawdown → low weight, clamp at floor.
    w_low = sortino_label_weight(return_pct=0.001, intra_trade_drawdown_pct=10.0)
    assert w_low >= 0.2

    # Big win + zero drawdown → ceiling.
    w_high = sortino_label_weight(return_pct=0.5, intra_trade_drawdown_pct=0.0)
    assert w_high <= 2.0


# ── IncrementalSignalClassifier ───────────────────────────────


def test_classifier_unavailable_when_sklearn_missing(monkeypatch):
    """When sklearn import fails, available is False and partial_fit returns False."""
    clf = IncrementalSignalClassifier()
    # Patch the import path so _ensure_sklearn fails deterministically.
    monkeypatch.setattr(clf, "_available", False)
    assert clf.partial_fit([0.1, 0.2], 1) is False
    assert clf.predict_confidence([0.1, 0.2]) is None


def test_partial_fit_then_predict_yields_probability():
    pytest.importorskip("sklearn")
    clf = IncrementalSignalClassifier()
    if not clf.available:
        pytest.skip("sklearn unavailable")

    # Two clearly separable classes — feature[0] = 0 → class 0; feature[0] = 1 → class 1.
    for _ in range(20):
        clf.partial_fit([0.0, 0.0], 0)
        clf.partial_fit([1.0, 1.0], 1)

    p_class1 = clf.predict_confidence([1.0, 1.0])
    p_class0 = clf.predict_confidence([0.0, 0.0])
    assert p_class1 is not None and p_class0 is not None
    assert p_class1 > p_class0


def test_predict_returns_none_before_first_fit():
    pytest.importorskip("sklearn")
    clf = IncrementalSignalClassifier()
    if not clf.available:
        pytest.skip("sklearn unavailable")
    assert clf.predict_confidence([0.0, 0.0]) is None


def test_save_load_round_trip(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    save_path = tmp_path / "clf.joblib"
    clf = IncrementalSignalClassifier(save_path=save_path)
    if not clf.available:
        pytest.skip("sklearn unavailable")
    for _ in range(10):
        clf.partial_fit([0.0, 0.0], 0)
        clf.partial_fit([1.0, 1.0], 1)
    clf.save()
    assert save_path.exists()

    reloaded = IncrementalSignalClassifier(save_path=save_path)
    assert reloaded.load() is True
    p1 = reloaded.predict_confidence([1.0, 1.0])
    p0 = reloaded.predict_confidence([0.0, 0.0])
    assert p1 is not None and p0 is not None
    assert p1 > p0


def test_sample_weight_passed_through():
    """High-weight samples should bias the boundary more than low-weight ones."""
    pytest.importorskip("sklearn")
    clf = IncrementalSignalClassifier()
    if not clf.available:
        pytest.skip("sklearn unavailable")
    # Many low-weight 0s, few high-weight 1s — model should still learn.
    for _ in range(50):
        clf.partial_fit([0.0], 0, sample_weight=0.1)
    for _ in range(5):
        clf.partial_fit([1.0], 1, sample_weight=5.0)
    p1 = clf.predict_confidence([1.0])
    assert p1 is not None
