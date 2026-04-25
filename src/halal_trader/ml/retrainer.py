"""Automated ML retraining — labels closed trades and retrains models on schedule."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from halal_trader.domain.ports import TradeRepository

logger = logging.getLogger(__name__)

_FEATURE_KEYS = [
    "rsi_14",
    "macd_histogram",
    "volume_ratio",
    "atr_14",
    "bb_position",
]


class RetrainingScheduler:
    """Labels closed trades with their indicator snapshots, then retrains ML models.

    Designed to run periodically (e.g. nightly or after N closed trades).
    """

    def __init__(
        self,
        repo: TradeRepository,
        *,
        models_dir: Path = Path("models"),
        min_samples: int = 50,
        retrain_every_n_trades: int = 20,
    ) -> None:
        self._repo = repo
        self._models_dir = models_dir
        self._min_samples = min_samples
        self._retrain_threshold = retrain_every_n_trades
        self._trades_since_retrain = 0

    async def on_trade_closed(self, trade_id: int, return_pct: float) -> None:
        """Called when a trade is closed — labels its snapshot and checks retrain trigger."""
        label = 1 if return_pct > 0 else 0
        try:
            await self._repo.label_indicator_snapshot(trade_id, label, return_pct)
            logger.debug(
                "Labeled snapshot for trade #%d: %s (%.2f%%)",
                trade_id,
                "profitable" if label else "unprofitable",
                return_pct * 100,
            )
        except Exception as e:
            logger.debug("Failed to label snapshot for trade #%d: %s", trade_id, e)

        self._trades_since_retrain += 1
        if self._trades_since_retrain >= self._retrain_threshold:
            await self.retrain()
            self._trades_since_retrain = 0

    async def retrain(self) -> dict[str, Any]:
        """Pull labeled snapshots and retrain all ML models."""
        results: dict[str, Any] = {"anomaly": False, "classifier": False, "samples": 0}

        try:
            snapshots = await self._repo.get_labeled_snapshots(min_samples=self._min_samples)
        except Exception as e:
            logger.warning("Failed to fetch labeled snapshots: %s", e)
            return results

        if not snapshots:
            logger.info(
                "Not enough labeled snapshots for retraining (need %d)",
                self._min_samples,
            )
            return results

        results["samples"] = len(snapshots)
        features: list[list[float]] = []
        labels: list[int] = []

        for snap in snapshots:
            feat = []
            skip = False
            for key in _FEATURE_KEYS:
                val = snap.get(key)
                if val is None:
                    skip = True
                    break
                feat.append(float(val))
            if skip:
                continue
            features.append(feat)
            labels.append(snap.get("label", 0))

        if len(features) < self._min_samples:
            logger.info(
                "Not enough valid snapshots after filtering: %d/%d",
                len(features),
                self._min_samples,
            )
            return results

        self._models_dir.mkdir(parents=True, exist_ok=True)

        try:
            from halal_trader.ml.hub import ModelHub

            hub = ModelHub(models_dir=self._models_dir)

            from halal_trader.ml.anomaly import MarketAnomalyDetector

            detector = MarketAnomalyDetector(hub, min_samples=self._min_samples)
            for feat in features:
                detector._samples.append(feat)
            if detector.train():
                results["anomaly"] = True
                logger.info("Anomaly detector retrained on %d samples", len(features))
        except Exception as e:
            logger.warning("Anomaly detector retraining failed: %s", e)

        try:
            from halal_trader.ml.anomaly import MLSignalClassifier
            from halal_trader.ml.hub import ModelHub

            hub = ModelHub(models_dir=self._models_dir)
            classifier = MLSignalClassifier(hub)
            if classifier.train(features, labels):
                results["classifier"] = True
                logger.info("Signal classifier retrained on %d samples", len(features))
        except Exception as e:
            logger.warning("Signal classifier retraining failed: %s", e)

        logger.info(
            "Retraining complete: %d samples, anomaly=%s, classifier=%s",
            len(features),
            results["anomaly"],
            results["classifier"],
        )
        return results
