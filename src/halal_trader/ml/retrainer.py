"""Automated ML retraining — labels closed trades and retrains models on schedule."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from halal_trader.db.repos import IndicatorSnapshotRepo
from halal_trader.ml.features import FEATURE_KEYS

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from halal_trader.db.repos import CryptoTradeRepo

logger = logging.getLogger(__name__)

_FEATURE_KEYS = list(FEATURE_KEYS)


class RetrainingScheduler:
    """Labels closed trades with their indicator snapshots, then retrains ML models.

    Designed to run periodically (e.g. nightly or after N closed trades).

    The ``namespace`` argument lets us run separate schedulers for crypto
    vs stocks — each persists its models under a distinct subdirectory
    (``models/<namespace>/``) so a crypto retrain never clobbers a stock
    model trained on different feature distributions, even though both
    share the same ``IndicatorSnapshot`` table for label storage.

    Pass ``crypto_trade_repo`` + ``engine`` to also refit the Wave G
    slippage model alongside the anomaly + signal classifiers — both
    optional, so the stocks-namespace retrainer can omit them.
    """

    def __init__(
        self,
        repo: IndicatorSnapshotRepo,
        *,
        models_dir: Path = Path("models"),
        min_samples: int = 50,
        retrain_every_n_trades: int = 20,
        namespace: str = "crypto",
        crypto_trade_repo: "CryptoTradeRepo | None" = None,
        engine: "AsyncEngine | None" = None,
    ) -> None:
        self._repo = repo
        # Models live under models/<namespace>/ so two retrainers don't fight
        # over the same on-disk file.
        self._models_dir = models_dir / namespace
        self._namespace = namespace
        self._min_samples = min_samples
        self._retrain_threshold = retrain_every_n_trades
        self._trades_since_retrain = 0
        self._crypto_trade_repo = crypto_trade_repo
        self._engine = engine

    @property
    def namespace(self) -> str:
        return self._namespace

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
        results: dict[str, Any] = {
            "anomaly": False,
            "classifier": False,
            "slippage": False,
            "samples": 0,
        }

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

        # Wave G: slippage refit — separate sample set (closed trades
        # with both intent + filled prices), so it doesn't gate on the
        # anomaly/classifier sample minimum.
        try:
            results["slippage"] = await self._retrain_slippage()
        except Exception as exc:  # noqa: BLE001
            logger.debug("slippage retrain failed: %s", exc)

        logger.info(
            "Retraining complete: %d samples, anomaly=%s, classifier=%s, slippage=%s",
            len(features),
            results["anomaly"],
            results["classifier"],
            results["slippage"],
        )
        return results

    async def _retrain_slippage(self) -> bool:
        """Fit + persist the Wave G slippage model from recent filled trades.

        Returns True when a fresh model was saved, False when the refit
        was skipped (no trade repo, no rows, too few valid samples, etc).
        Each step is best-effort; failures are debug-logged.
        """
        if self._crypto_trade_repo is None:
            return False
        from halal_trader.ml.slippage import (
            fit_from_trades,
            save_to_file,
            trade_to_sample,
        )

        try:
            trades = await self._crypto_trade_repo.get_filled_trades(limit=500)
        except Exception as exc:  # noqa: BLE001
            logger.debug("slippage retrain: trade fetch failed: %s", exc)
            return False
        if not trades:
            return False
        try:
            snapshots = await self._repo.get_labeled_snapshots(min_samples=0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("slippage retrain: snapshot fetch failed: %s", exc)
            snapshots = []
        # Index snapshots by trade_id so we can join in O(n+m).
        snap_by_trade: dict[int, dict[str, Any]] = {}
        for s in snapshots:
            tid = s.get("trade_id")
            if tid is not None:
                snap_by_trade[int(tid)] = s

        samples: list[dict[str, Any]] = []
        for trade in trades:
            tid = trade.get("id")
            if tid is None:
                continue
            indicators = snap_by_trade.get(int(tid), {})
            sample = trade_to_sample(trade, indicators)
            if sample is not None:
                samples.append(sample)

        if not samples:
            return False
        model = fit_from_trades(samples)
        if model.n_samples == 0:
            return False
        self._models_dir.mkdir(parents=True, exist_ok=True)
        try:
            save_to_file(model, self._models_dir)
        except Exception as exc:  # noqa: BLE001
            logger.debug("slippage refit: file save failed: %s", exc)
        if self._engine is not None:
            try:
                from halal_trader.ml.slippage import save_to_db

                await save_to_db(model, self._engine)
            except Exception as exc:  # noqa: BLE001
                logger.debug("slippage refit: DB save failed: %s", exc)
        logger.info("Slippage model refit on %d samples", model.n_samples)
        return True
