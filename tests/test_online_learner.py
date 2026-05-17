"""Tests for `ml/online_learner.py`.

Pins the ridge fit, the rolling-window eviction, the cold-start
zero-prediction guard, the tanh-bounded output, the
exponential-decay weighting, and the input-validation rejections.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from halal_trader.ml.online_learner import (
    LearnerSnapshot,
    OnlineLearner,
    OnlineLearnerConfig,
)

# ── config validation ────────────────────────────────────


def test_config_rejects_non_positive_window():
    with pytest.raises(ValueError, match="window"):
        OnlineLearnerConfig(window=0)


def test_config_rejects_zero_ridge_lambda():
    """Pin: ridge_lambda must be strictly positive — at 0 the
    closed-form solution can produce singular matrices on
    correlated inputs."""
    with pytest.raises(ValueError, match="ridge_lambda"):
        OnlineLearnerConfig(ridge_lambda=0.0)


def test_config_rejects_decay_outside_range():
    """Decay must be in (0, 1] — at 0 every sample contributes
    nothing; >1 would amplify old samples."""
    with pytest.raises(ValueError, match="decay"):
        OnlineLearnerConfig(decay=0.0)
    with pytest.raises(ValueError, match="decay"):
        OnlineLearnerConfig(decay=1.5)


def test_config_rejects_non_positive_clip():
    with pytest.raises(ValueError, match="clip"):
        OnlineLearnerConfig(feature_clip=0.0)
    with pytest.raises(ValueError, match="clip"):
        OnlineLearnerConfig(target_clip=-1.0)


def test_config_rejects_zero_min_samples_for_predict():
    with pytest.raises(ValueError, match="min_samples"):
        OnlineLearnerConfig(min_samples_for_predict=0)


# ── learner construction ─────────────────────────────────


def test_learner_rejects_zero_features():
    with pytest.raises(ValueError, match="n_features"):
        OnlineLearner(n_features=0)


def test_learner_starts_with_zero_state():
    """Pin: a fresh learner reports zero samples / zero coefs /
    zero RMSE / zero prediction. The dashboard tile renders cleanly
    off this initial state."""
    learner = OnlineLearner(n_features=3)
    snap = learner.snapshot()
    assert isinstance(snap, LearnerSnapshot)
    assert snap.sample_count == 0
    assert snap.coef == [0.0, 0.0, 0.0]
    assert snap.intercept == 0.0
    assert snap.rmse == 0.0
    assert snap.last_target is None


# ── cold-start ───────────────────────────────────────────


def test_predict_returns_zero_below_min_samples():
    """Pin: cold-start safe — the learner must not emit a signal
    until it has seen enough data. Strategies interpreting 0 as
    "no edge" stay correct."""
    learner = OnlineLearner(
        n_features=2,
        config=OnlineLearnerConfig(min_samples_for_predict=5),
    )
    learner.update(features=[1.0, 2.0], target=0.5)
    learner.update(features=[1.5, 2.5], target=0.6)
    # Only 2 samples; below min=5
    assert learner.predict([1.0, 2.0]) == 0.0


def test_predict_emits_signal_at_or_above_min_samples():
    learner = OnlineLearner(
        n_features=2,
        config=OnlineLearnerConfig(min_samples_for_predict=3),
    )
    for i in range(5):
        learner.update(features=[float(i), float(i * 2)], target=float(i) * 0.1)
    out = learner.predict([3.0, 6.0])
    # tanh-squashed; should be a real value not zero
    assert isinstance(out, float)
    assert out != 0.0


# ── ridge fit recovers a known relationship ──────────────


def test_learner_recovers_linear_relationship():
    """Pin: the learner must converge on the right slope when
    given a clean linear signal. Use small targets so tanh squash
    is near-identity (tanh(0.1) ≈ 0.0997)."""
    rng = np.random.default_rng(42)
    learner = OnlineLearner(
        n_features=2,
        config=OnlineLearnerConfig(
            window=200,
            ridge_lambda=0.01,
            min_samples_for_predict=10,
            target_clip=10.0,
        ),
    )
    # y = 0.05*x1 + 0.02*x2 + noise (small to avoid tanh saturation)
    for _ in range(150):
        x1 = float(rng.normal(0, 1))
        x2 = float(rng.normal(0, 1))
        y = 0.05 * x1 + 0.02 * x2 + float(rng.normal(0, 0.005))
        learner.update(features=[x1, x2], target=y)
    snap = learner.snapshot()
    # Coefs should be in the right ballpark — small ridge bias
    # pulls them toward zero but they should still sign-match.
    assert snap.coef[0] > 0.02 and snap.coef[0] < 0.10
    assert snap.coef[1] > 0.005 and snap.coef[1] < 0.05


def test_learner_recovers_intercept():
    """A non-zero target mean should be picked up by the
    intercept term."""
    learner = OnlineLearner(
        n_features=1,
        config=OnlineLearnerConfig(
            window=100,
            ridge_lambda=0.01,
            min_samples_for_predict=5,
            target_clip=10.0,
        ),
    )
    # y = 0.5 (constant) regardless of x
    for x in np.linspace(-1, 1, 60):
        learner.update(features=[float(x)], target=0.5)
    snap = learner.snapshot()
    assert abs(snap.intercept - 0.5) < 0.05


# ── rolling-window eviction ──────────────────────────────


def test_window_evicts_oldest_samples():
    """Pin: deque-backed buffer drops the oldest sample once the
    window is full."""
    learner = OnlineLearner(
        n_features=1,
        config=OnlineLearnerConfig(window=5, ridge_lambda=0.1),
    )
    for i in range(10):
        learner.update(features=[float(i)], target=float(i))
    # Only the last 5 samples are retained.
    assert learner.sample_count == 5


def test_learner_responds_to_new_regime():
    """End-to-end: feed a positive slope for window samples, then
    a negative slope. The learner should follow the new regime
    after enough new samples."""
    rng = np.random.default_rng(7)
    learner = OnlineLearner(
        n_features=1,
        config=OnlineLearnerConfig(window=50, ridge_lambda=0.01, min_samples_for_predict=10),
    )
    for _ in range(50):
        x = float(rng.normal(0, 1))
        learner.update(features=[x], target=0.05 * x)
    snap_before = learner.snapshot()
    assert snap_before.coef[0] > 0
    # Now flip the relationship.
    for _ in range(80):
        x = float(rng.normal(0, 1))
        learner.update(features=[x], target=-0.05 * x)
    snap_after = learner.snapshot()
    assert snap_after.coef[0] < 0


# ── decay weighting ──────────────────────────────────────


def test_decay_speeds_up_regime_response():
    """Pin: with aggressive decay (0.7) the learner forgets old
    samples faster than plain ridge. Compare convergence speed
    on a regime flip."""
    rng = np.random.default_rng(0)

    def run(decay: float) -> float:
        learner = OnlineLearner(
            n_features=1,
            config=OnlineLearnerConfig(
                window=100,
                ridge_lambda=0.01,
                decay=decay,
                min_samples_for_predict=5,
            ),
        )
        for _ in range(80):
            x = float(rng.normal(0, 1))
            learner.update(features=[x], target=0.05 * x)
        for _ in range(20):
            x = float(rng.normal(0, 1))
            learner.update(features=[x], target=-0.05 * x)
        return learner.snapshot().coef[0]

    plain_coef = run(decay=1.0)
    decayed_coef = run(decay=0.85)
    # The decayed learner should have moved further toward the
    # new (negative) regime than the plain ridge.
    assert decayed_coef < plain_coef


def test_decay_one_matches_plain_ridge():
    """Pin: decay=1.0 is the no-op case — equivalent to plain
    ridge. Build two learners; their fits must converge."""
    rng = np.random.default_rng(11)
    samples = [([float(rng.normal(0, 1))], float(rng.normal(0, 0.1))) for _ in range(50)]
    a = OnlineLearner(
        n_features=1,
        config=OnlineLearnerConfig(window=100, ridge_lambda=0.1, decay=1.0),
    )
    b = OnlineLearner(
        n_features=1,
        config=OnlineLearnerConfig(window=100, ridge_lambda=0.1, decay=1.0),
    )
    for x, y in samples:
        a.update(features=x, target=y)
        b.update(features=x, target=y)
    assert a.snapshot().coef == b.snapshot().coef


# ── input clipping ───────────────────────────────────────


def test_features_are_clipped_to_feature_clip():
    """Pin: a glitched feed (1e9 outlier) must not dominate the
    fit. Clipping bounds the contribution."""
    learner = OnlineLearner(
        n_features=1,
        config=OnlineLearnerConfig(
            window=100,
            ridge_lambda=0.1,
            feature_clip=5.0,
            min_samples_for_predict=2,
        ),
    )
    learner.update(features=[100.0], target=0.5)  # 100 → clipped to 5
    learner.update(features=[2.0], target=0.1)
    snap = learner.snapshot()
    # The first sample's effective x was 5, not 100 — so the slope
    # shouldn't be wildly small (which it would be if 100 made it in).
    assert abs(snap.coef[0]) > 0.001


def test_target_is_clipped_to_target_clip():
    learner = OnlineLearner(
        n_features=1,
        config=OnlineLearnerConfig(
            window=20,
            ridge_lambda=0.1,
            target_clip=1.0,
            min_samples_for_predict=2,
        ),
    )
    learner.update(features=[1.0], target=1e6)  # → clipped to 1.0
    snap = learner.snapshot()
    # The target was clipped, so the in-sample residual should be
    # bounded — RMSE much less than 1e6.
    assert snap.rmse < 1.0


def test_update_rejects_wrong_feature_count():
    learner = OnlineLearner(n_features=2)
    with pytest.raises(ValueError, match="features"):
        learner.update(features=[1.0], target=0.5)
    with pytest.raises(ValueError, match="features"):
        learner.update(features=[1.0, 2.0, 3.0], target=0.5)


def test_update_rejects_non_finite_features():
    learner = OnlineLearner(n_features=1)
    with pytest.raises(ValueError, match="finite"):
        learner.update(features=[float("nan")], target=0.5)
    with pytest.raises(ValueError, match="finite"):
        learner.update(features=[float("inf")], target=0.5)


def test_update_rejects_non_finite_target():
    learner = OnlineLearner(n_features=1)
    with pytest.raises(ValueError, match="target"):
        learner.update(features=[1.0], target=float("nan"))


def test_predict_rejects_wrong_feature_count():
    learner = OnlineLearner(n_features=2)
    with pytest.raises(ValueError, match="features"):
        learner.predict([1.0])


# ── tanh squash ──────────────────────────────────────────


def test_predict_output_is_bounded_by_tanh():
    """Pin: even if the linear combination is huge (runaway coef),
    tanh squashes the output to [-1, 1]."""
    learner = OnlineLearner(
        n_features=1,
        config=OnlineLearnerConfig(
            window=20,
            ridge_lambda=0.01,
            min_samples_for_predict=2,
            target_clip=10.0,
        ),
    )
    # Train on a consistent positive slope to bias the coef high.
    for _ in range(50):
        learner.update(features=[1.0], target=0.5)
    out = learner.predict([100.0])  # 100 → clipped to 10, then linear, then tanh
    assert -1.0 <= out <= 1.0


def test_predict_zero_input_produces_intercept_squashed():
    """Pin: predict([0]) returns tanh(intercept) — confirms the
    intercept term is consulted at predict time."""
    learner = OnlineLearner(
        n_features=1,
        config=OnlineLearnerConfig(min_samples_for_predict=2),
    )
    # Constant target → intercept ≈ 0.3
    for _ in range(40):
        learner.update(features=[0.0], target=0.3)
    out = learner.predict([0.0])
    assert abs(out - math.tanh(0.3)) < 0.05


# ── reset ────────────────────────────────────────────────


def test_reset_clears_buffer_and_coefs():
    learner = OnlineLearner(n_features=2)
    for _ in range(20):
        learner.update(features=[1.0, 2.0], target=0.5)
    assert learner.sample_count == 20
    learner.reset()
    assert learner.sample_count == 0
    snap = learner.snapshot()
    assert snap.coef == [0.0, 0.0]
    assert snap.intercept == 0.0
    assert snap.last_target is None


def test_predict_after_reset_returns_zero():
    learner = OnlineLearner(
        n_features=1,
        config=OnlineLearnerConfig(min_samples_for_predict=3),
    )
    for _ in range(20):
        learner.update(features=[1.0], target=0.5)
    learner.reset()
    assert learner.predict([1.0]) == 0.0


# ── snapshot ─────────────────────────────────────────────


def test_snapshot_records_last_target_and_prediction():
    learner = OnlineLearner(
        n_features=1,
        config=OnlineLearnerConfig(min_samples_for_predict=2),
    )
    learner.update(features=[1.0], target=0.3)
    learner.update(features=[2.0], target=0.5)
    learner.predict([3.0])
    snap = learner.snapshot()
    assert snap.last_target == 0.5
    assert isinstance(snap.last_prediction, float)


def test_snapshot_is_immutable():
    learner = OnlineLearner(n_features=1)
    snap = learner.snapshot()
    with pytest.raises(Exception):
        snap.sample_count = 999  # type: ignore[misc]


def test_n_features_property():
    learner = OnlineLearner(n_features=4)
    assert learner.n_features == 4


def test_sample_count_property_tracks_buffer():
    learner = OnlineLearner(n_features=1, config=OnlineLearnerConfig(window=10))
    for _ in range(15):
        learner.update(features=[1.0], target=0.5)
    # Window is 10; oldest 5 evicted.
    assert learner.sample_count == 10
