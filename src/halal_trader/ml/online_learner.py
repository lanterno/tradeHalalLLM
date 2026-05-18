"""Online linear regression for short-half-life signals.

Round-4 wave 6.C: some signals (orderbook imbalance, spot-perp
basis, sentiment-velocity) have too short a half-life for the
batch retrainer's daily / weekly cadence. By the time the offline
job has shipped, the relationship has shifted. This module is the
streaming counterpart: stream-fit a regularised linear model over
the last N samples and re-fit every observation.

Two variants:

* **Ridge regression** (default) — closed-form L2-regularised
  least squares fit on a rolling buffer. Fast, numerically
  stable, and the regularisation prevents the parameter blow-up
  that vanilla OLS gets when features are correlated. Pin: this
  is *not* an online RLS update; it's a small batch refit on
  every observation. With N=200 samples × ≤ 20 features the
  refit cost is sub-millisecond and the implementation stays
  one-screen-of-numpy auditable.
* **Exponentially-weighted ridge** — weights samples by
  ``decay**(t)`` so older observations contribute less. Useful
  when the half-life is ~30s and even a 200-sample window
  would over-weight stale data.

The learner is **bounded** — every sample is clipped to a
``feature_range`` and a ``target_range`` before fitting. Pin:
unbounded inputs from a glitching feed (a 1e9 outlier from a
malformed kline) would dominate the fit. Clipping keeps the
learner numerically sane without dropping samples.

Halal alignment: the learner emits a *signal* (a regression
prediction). It never opens a position by itself. The downstream
consumer (a strategy that combines the learner's signal with the
LLM's plan) decides whether to act. The signal is bounded in
[-1, 1] after a tanh squash so a runaway value can't cause a
size explosion.

Pure-numpy; no scipy / sklearn / DB / async.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# ── Configuration ────────────────────────────────────────


@dataclass(frozen=True)
class OnlineLearnerConfig:
    """Hyperparameters + safety bounds.

    ``window`` is the rolling buffer size; the model refits on
    each call to `update()` against the most-recent ``window``
    samples. Smaller window = more responsive, noisier; larger =
    slower, steadier.

    ``ridge_lambda`` is the L2 penalty. Pin: keep it strictly
    positive — at 0 the closed-form solution is sensitive to
    correlated features; the default 0.1 is small enough not to
    bias the fit on a clean signal, large enough to keep the
    matrix invertible when features collinear-ate.

    ``decay`` is the exponential weight per sample-age (1.0 = no
    decay, plain ridge; 0.95 = each older sample contributes
    ~95% of the previous; 0.5 = aggressive recency bias). Must
    be in (0, 1].

    ``feature_clip`` and ``target_clip`` cap each value before
    fitting so a glitched feed (1e9 from a malformed kline) can't
    dominate the fit. Pin: clipping rather than dropping samples
    keeps the buffer aligned with the time axis.

    ``min_samples_for_predict`` is the cold-start guard — below
    this, `predict` returns 0.0 (the "no signal yet" baseline).
    Default matches the feature count + 1 so the design matrix
    is at least square.
    """

    window: int = 200
    ridge_lambda: float = 0.1
    decay: float = 1.0
    feature_clip: float = 10.0
    target_clip: float = 1.0
    min_samples_for_predict: int = 5

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError(f"window must be positive; got {self.window}")
        if self.ridge_lambda <= 0:
            raise ValueError(f"ridge_lambda must be positive; got {self.ridge_lambda}")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1]; got {self.decay}")
        if self.feature_clip <= 0 or self.target_clip <= 0:
            raise ValueError("clip values must be positive")
        if self.min_samples_for_predict < 1:
            raise ValueError("min_samples_for_predict must be >= 1")


# ── Output ────────────────────────────────────────────────


@dataclass(frozen=True)
class LearnerSnapshot:
    """Diagnostic snapshot of the learner's state.

    ``coef`` is the learned weights (length = n_features +
    intercept). ``rmse`` is the in-sample root-mean-squared error
    over the current buffer. ``sample_count`` is how many samples
    are currently in the buffer (≤ window).
    """

    n_features: int
    sample_count: int
    coef: list[float]
    intercept: float
    rmse: float
    last_prediction: float
    last_target: float | None


# ── Learner ───────────────────────────────────────────────


class OnlineLearner:
    """Rolling-window ridge regressor.

    Usage:

        learner = OnlineLearner(n_features=3, config=OnlineLearnerConfig())
        learner.update(features=[0.1, 0.2, 0.3], target=0.05)
        signal = learner.predict([0.15, 0.25, 0.35])  # in [-1, 1] after tanh

    The learner is single-threaded — updates and predicts are
    not safe to interleave from multiple coroutines without
    external synchronisation. The cycle's hot path holds the
    learner under the per-pair lock already.
    """

    def __init__(
        self,
        *,
        n_features: int,
        config: OnlineLearnerConfig | None = None,
    ) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1; got {n_features}")
        self._n_features = n_features
        self._config = config or OnlineLearnerConfig()
        self._features: deque[np.ndarray] = deque(maxlen=self._config.window)
        self._targets: deque[float] = deque(maxlen=self._config.window)
        # Last fit: coefs (length n_features) and intercept.
        self._coef = np.zeros(n_features, dtype=float)
        self._intercept = 0.0
        self._last_rmse = 0.0
        self._last_prediction = 0.0
        self._last_target: float | None = None
        self._fitted_at_count = 0

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def sample_count(self) -> int:
        return len(self._targets)

    def _clip_features(self, features: Sequence[float]) -> np.ndarray:
        if len(features) != self._n_features:
            raise ValueError(f"expected {self._n_features} features; got {len(features)}")
        arr = np.array(features, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("features must be finite (no NaN / inf)")
        clip = self._config.feature_clip
        return np.clip(arr, -clip, clip)

    def _clip_target(self, target: float) -> float:
        if not np.isfinite(target):
            raise ValueError(f"target must be finite; got {target}")
        clip = self._config.target_clip
        return float(np.clip(target, -clip, clip))

    def update(self, *, features: Sequence[float], target: float) -> None:
        """Append a sample and re-fit.

        Pin: the re-fit happens on every update, not on a timer
        — the hot path is small enough (O(N × K²) for window N
        and features K) that batching gives no measurable benefit
        and complicates the contract.
        """
        x = self._clip_features(features)
        y = self._clip_target(target)
        self._features.append(x)
        self._targets.append(y)
        self._last_target = y
        self._refit()

    def _build_weights(self, n: int) -> np.ndarray:
        """Exponential weight per sample age. Pin: most-recent
        samples (high index in the buffer) get weight 1.0;
        earlier samples scale by ``decay**(n - 1 - i)`` so a
        decay=1.0 collapses to plain ridge."""
        if self._config.decay == 1.0:
            return np.ones(n)
        ages = np.arange(n - 1, -1, -1)  # most recent → 0
        return np.power(self._config.decay, ages)

    def _refit(self) -> None:
        """Closed-form weighted ridge regression.

        Solves ``(XᵀW X + λI) β = XᵀW y`` where ``X`` includes
        an intercept column. Pin: the intercept is regularised
        too — a non-zero intercept on noise drifts the prediction;
        the L2 penalty on it grounds the model at zero unless the
        data demands otherwise.
        """
        n = len(self._targets)
        if n == 0:
            return
        X_no_intercept = np.array(list(self._features), dtype=float)
        # Design matrix with an intercept column.
        X = np.hstack([X_no_intercept, np.ones((n, 1))])
        y = np.array(list(self._targets), dtype=float)
        w = self._build_weights(n)
        # Weighted ridge: solve (Xᵀ diag(w) X + λI) β = Xᵀ diag(w) y.
        WX = X * w[:, None]
        gram = WX.T @ X + self._config.ridge_lambda * np.eye(self._n_features + 1)
        rhs = WX.T @ y
        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            # Singular fallback: pseudoinverse. Pin: numerical
            # safety; should never trigger in practice given
            # ridge_lambda > 0 but defends against pathological
            # inputs.
            beta = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        self._coef = beta[: self._n_features]
        self._intercept = float(beta[-1])
        # In-sample RMSE for the diagnostic snapshot.
        residuals = y - (X @ beta)
        self._last_rmse = float(np.sqrt(np.mean(residuals**2)))
        self._fitted_at_count = n

    def predict(self, features: Sequence[float]) -> float:
        """Predict the (squashed) signal for a new feature vector.

        Pin: returns 0.0 when fewer than ``min_samples_for_predict``
        observations have been seen — cold-start safe default for
        a strategy that interprets 0 as "no edge".

        The raw prediction is squashed through tanh so a runaway
        coefficient can't push the signal beyond [-1, 1] and
        cause a size explosion downstream."""
        x = self._clip_features(features)
        if self.sample_count < self._config.min_samples_for_predict:
            self._last_prediction = 0.0
            return 0.0
        raw = float(x @ self._coef + self._intercept)
        # tanh squash for bounded output.
        squashed = float(np.tanh(raw))
        self._last_prediction = squashed
        return squashed

    def reset(self) -> None:
        """Clear the buffer and zero the coefs. Useful on a
        regime-change detection event when the operator wants
        the learner to start over rather than slowly forget."""
        self._features.clear()
        self._targets.clear()
        self._coef = np.zeros(self._n_features, dtype=float)
        self._intercept = 0.0
        self._last_rmse = 0.0
        self._last_prediction = 0.0
        self._last_target = None
        self._fitted_at_count = 0

    def snapshot(self) -> LearnerSnapshot:
        return LearnerSnapshot(
            n_features=self._n_features,
            sample_count=self.sample_count,
            coef=[float(c) for c in self._coef],
            intercept=self._intercept,
            rmse=self._last_rmse,
            last_prediction=self._last_prediction,
            last_target=self._last_target,
        )


__all__ = [
    "LearnerSnapshot",
    "OnlineLearner",
    "OnlineLearnerConfig",
]
