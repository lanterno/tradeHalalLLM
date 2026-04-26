"""Incremental ML signal classifier — partial_fit instead of full retrain.

The legacy retrainer rebuilds the IsolationForest + classifier from
scratch every 20 closed trades. That re-reads the entire labeled
snapshot table, refits, and writes a new model file — a fine bootstrap
strategy but a slow upgrade once we've seen thousands of trades.

This module replaces the classifier path with an SGDClassifier wrapped
in ``partial_fit``: each closed trade contributes one new sample and
the model updates in place. We keep a small in-memory ring buffer for
the most recent N samples so a freshly-spawned process can reload
state from disk and still see recent history when it warms back up.

Sortino-weighted labels: a profitable trade that survived a 20%
intra-trade drawdown is a worse training signal than a profitable
trade that climbed steadily. Pass ``intra_trade_drawdown_pct`` so we
can downweight noisy wins.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IncrementalSample:
    features: list[float]
    label: int
    sample_weight: float = 1.0


def sortino_label_weight(
    return_pct: float,
    intra_trade_drawdown_pct: float,
    *,
    floor: float = 0.2,
    ceiling: float = 2.0,
) -> float:
    """Map a trade outcome to a Sortino-style training weight.

    A return of +5% with no drawdown gets full weight; the same return
    after a -3% intra-trade drawdown is a *noisier* win and gets
    discounted. Bounded so a single weird sample can't dominate the
    fit.
    """
    if return_pct <= 0:
        # Losses always get weight 1.0 — they're the signal we want to
        # not repeat, and downweighting them risks the classifier
        # learning that losses don't matter.
        return 1.0
    if intra_trade_drawdown_pct <= 0:
        return ceiling
    # Heuristic: weight = return / (return + dd), clamped.
    score = return_pct / (return_pct + intra_trade_drawdown_pct)
    weight = floor + (ceiling - floor) * score
    return max(floor, min(ceiling, weight))


class IncrementalSignalClassifier:
    """Thin wrapper around scikit's ``SGDClassifier`` with partial_fit.

    Reuses the same ``ModelHub`` save/load surface as the legacy
    classifier so existing model directories don't need a structural
    migration. Lazy-imports sklearn so the bot still starts when the
    ``[ml]`` extra isn't installed (the ``available()`` flag tells the
    caller whether to wire it in or skip).
    """

    def __init__(self, *, save_path: Path | None = None, classes: Sequence[int] = (0, 1)) -> None:
        self._save_path = save_path
        self._classes = list(classes)
        self._model = None
        self._initialised = False
        self._available: bool | None = None

    def _ensure_sklearn(self) -> bool:
        if self._available is None:
            try:
                from sklearn.linear_model import SGDClassifier  # noqa: F401

                self._available = True
            except ImportError:
                self._available = False
                logger.info("sklearn not installed — incremental classifier disabled")
        return self._available

    @property
    def available(self) -> bool:
        return self._ensure_sklearn()

    def _new_model(self):
        from sklearn.linear_model import SGDClassifier

        return SGDClassifier(
            loss="log_loss",
            learning_rate="optimal",
            alpha=1e-4,
            random_state=42,
        )

    def partial_fit(
        self,
        features: Sequence[float],
        label: int,
        *,
        sample_weight: float = 1.0,
    ) -> bool:
        """Update the model with one labeled sample. Returns ``True`` on success."""
        if not self._ensure_sklearn():
            return False
        if self._model is None:
            self._model = self._new_model()

        x = np.asarray([list(features)], dtype=float)
        y = np.asarray([int(label)])
        sw = np.asarray([float(sample_weight)])

        if not self._initialised:
            self._model.partial_fit(x, y, classes=np.asarray(self._classes), sample_weight=sw)
            self._initialised = True
        else:
            self._model.partial_fit(x, y, sample_weight=sw)
        return True

    def predict_confidence(self, features: Sequence[float]) -> float | None:
        """Return P(positive) ∈ [0, 1] or ``None`` if the model isn't ready."""
        if not self._ensure_sklearn() or self._model is None or not self._initialised:
            return None
        x = np.asarray([list(features)], dtype=float)
        try:
            proba = self._model.predict_proba(x)[0]
        except Exception as e:
            logger.debug("predict_proba failed: %s", e)
            return None
        if len(proba) < 2:
            return None
        # Class 1 = profitable.
        classes = list(self._model.classes_)
        if 1 in classes:
            return float(proba[classes.index(1)])
        return float(proba[-1])

    def save(self) -> None:
        """Persist to ``save_path`` using joblib if available."""
        if self._save_path is None or self._model is None:
            return
        try:
            import joblib

            self._save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._model, self._save_path)
        except Exception as e:
            logger.debug("Failed to persist incremental classifier: %s", e)

    def load(self) -> bool:
        if self._save_path is None or not self._save_path.exists():
            return False
        try:
            import joblib

            self._model = joblib.load(self._save_path)
            self._initialised = True
            return True
        except Exception as e:
            logger.debug("Failed to load incremental classifier: %s", e)
            return False


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))
