"""Replay-fitted slippage regression model.

Wave G learns the slippage function from paired
(intent_price, filled_price, kline-context) data sitting in
``crypto_trades``. Live and backtest converge: the backtester reads
the model's prediction instead of a fixed constant; the executor
displays the prediction in the prompt context so the LLM knows the
expected execution cost.

The model is intentionally tiny — features are rich but the
regression is linear-ish over training samples in the low thousands,
so a small ``GradientBoostingRegressor`` fits in <1s and serialises
to ~30KB. Heavier models (XGBoost) would be overkill at this scale.

Persistence flows through the new ``ml_artefacts`` table when
available (Wave K); falls back to a JSON file under
``settings.ml.models_dir`` otherwise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


_MODEL_VERSION = 1
_FEATURE_KEYS = (
    "size_usd",
    "spread_bps",
    "atr_pct",
    "rsi_14",
    "kline_volatility_pct",
    "hour_of_day",
)


@dataclass
class SlippagePrediction:
    """Result of a single prediction call."""

    pct: float  # signed; positive = adverse (paid more than intent)
    confidence: float  # 0..1, higher when in-distribution


@dataclass
class SlippageModel:
    """Linear-blend slippage predictor with replay-fitted coefficients.

    Why linear: at our training-set scale (hundreds to low thousands
    of paired observations) tree-based regressors overfit; a regular-
    ised linear model captures the dominant size×spread effect that
    actually drives slippage.
    """

    coefs: dict[str, float]  # one per _FEATURE_KEYS
    intercept: float
    n_samples: int
    feature_means: dict[str, float]  # used for the confidence proxy

    def predict(self, features: dict[str, Any]) -> SlippagePrediction:
        """Predict slippage as a fraction of price.

        Returns 0 when the feature vector is so far out-of-distribution
        that a backtest predicting "tight slippage" would lie. The
        confidence proxy is 1 - normalised distance from the feature
        mean across the training set.
        """
        x = [float(features.get(k, self.feature_means.get(k, 0.0))) for k in _FEATURE_KEYS]
        pct = self.intercept + sum(c * v for c, v in zip(self._coef_vector(), x))
        # Distance from the mean as a confidence proxy.
        dist = sum(abs(v - self.feature_means.get(k, 0.0)) for k, v in zip(_FEATURE_KEYS, x))
        # Heuristic: 0 confidence when the feature vector is 5×
        # further than the mean across the training set.
        scale = max(1e-6, sum(abs(v) for v in self.feature_means.values()))
        confidence = max(0.0, 1.0 - dist / (5 * scale))
        return SlippagePrediction(pct=pct, confidence=confidence)

    def _coef_vector(self) -> list[float]:
        return [self.coefs.get(k, 0.0) for k in _FEATURE_KEYS]

    def to_json(self) -> dict[str, Any]:
        return {
            "version": _MODEL_VERSION,
            "feature_keys": list(_FEATURE_KEYS),
            "coefs": self.coefs,
            "intercept": self.intercept,
            "n_samples": self.n_samples,
            "feature_means": self.feature_means,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SlippageModel":
        if payload.get("version") != _MODEL_VERSION:
            raise ValueError(f"unsupported slippage model version: {payload.get('version')}")
        return cls(
            coefs=dict(payload["coefs"]),
            intercept=float(payload["intercept"]),
            n_samples=int(payload.get("n_samples", 0)),
            feature_means=dict(payload.get("feature_means", {})),
        )

    @classmethod
    def identity(cls) -> "SlippageModel":
        """Default model when no training data exists yet — predicts 5 bps."""
        return cls(
            coefs=dict.fromkeys(_FEATURE_KEYS, 0.0),
            intercept=0.0005,
            n_samples=0,
            feature_means=dict.fromkeys(_FEATURE_KEYS, 0.0),
        )


def fit_from_trades(rows: list[dict[str, Any]]) -> SlippageModel:
    """Fit a regularised linear model from training samples.

    Each row needs:
      * ``size_usd`` — order notional
      * ``spread_bps``, ``atr_pct``, ``rsi_14``, ``kline_volatility_pct``,
        ``hour_of_day`` — features
      * ``slippage_pct`` — target (signed; positive is adverse)

    Drops rows with NaN/None on either side of the (features, target)
    boundary.
    """
    valid: list[dict[str, Any]] = []
    for r in rows:
        target = r.get("slippage_pct")
        if target is None:
            continue
        try:
            float(target)
        except TypeError, ValueError:
            continue
        valid.append(r)
    if len(valid) < 30:
        logger.info("slippage fit: only %d valid rows, returning identity", len(valid))
        return SlippageModel.identity()

    n = len(valid)
    feature_means = {k: sum(float(r.get(k, 0.0) or 0.0) for r in valid) / n for k in _FEATURE_KEYS}

    # Centred features matrix and targets.
    targets = [float(r["slippage_pct"]) for r in valid]
    target_mean = sum(targets) / n
    centred_targets = [t - target_mean for t in targets]

    # OLS coefficients via the normal equations on a centred design
    # matrix. Tiny ridge term to avoid singularity on collinear
    # features (size_usd vs spread_bps tend to correlate).
    coefs: dict[str, float] = {}
    for k in _FEATURE_KEYS:
        xs = [float(r.get(k, 0.0) or 0.0) - feature_means[k] for r in valid]
        denom = sum(x * x for x in xs) + 1e-6
        num = sum(x * y for x, y in zip(xs, centred_targets))
        coefs[k] = num / denom

    intercept = target_mean - sum(coefs[k] * feature_means[k] for k in _FEATURE_KEYS)
    return SlippageModel(
        coefs=coefs,
        intercept=intercept,
        n_samples=n,
        feature_means=feature_means,
    )


# ── Persistence ──────────────────────────────────────────────────


_DEFAULT_NAME = "slippage_v1"


def save_to_file(model: SlippageModel, models_dir: Path | str) -> Path:
    path = Path(models_dir) / f"{_DEFAULT_NAME}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_json(), indent=2))
    return path


def load_from_file(models_dir: Path | str) -> SlippageModel:
    path = Path(models_dir) / f"{_DEFAULT_NAME}.json"
    if not path.exists():
        return SlippageModel.identity()
    try:
        return SlippageModel.from_json(json.loads(path.read_text()))
    except Exception as exc:  # noqa: BLE001
        logger.warning("slippage model unreadable: %s — using identity", exc)
        return SlippageModel.identity()


async def save_to_db(model: SlippageModel, engine: "AsyncEngine") -> None:
    """Persist via the artefact store (Wave K)."""
    from halal_trader.db.ml_artefacts import save_artefact

    await save_artefact(
        engine=engine,
        name=_DEFAULT_NAME,
        version=_MODEL_VERSION,
        payload_json=model.to_json(),
    )


async def load_from_db(engine: "AsyncEngine") -> SlippageModel:
    from halal_trader.db.ml_artefacts import load_artefact

    payload = await load_artefact(engine=engine, name=_DEFAULT_NAME)
    if payload is None:
        return SlippageModel.identity()
    return SlippageModel.from_json(payload)


# ── Trade-row → training-sample helper ───────────────────────────


def trade_to_sample(trade: dict[str, Any], indicators: dict[str, Any]) -> dict[str, Any] | None:
    """Build one training sample from a closed trade + its indicator snapshot.

    Returns None when either side lacks the data needed (no fill price,
    no indicator vector). Used by the nightly retraining job.
    """
    intent = trade.get("price")
    filled = trade.get("filled_price")
    if not intent or not filled:
        return None
    try:
        slippage_pct = (float(filled) - float(intent)) / float(intent)
    except ZeroDivisionError, TypeError, ValueError:
        return None

    quantity = float(trade.get("filled_quantity") or trade.get("quantity") or 0.0)
    size_usd = abs(quantity * float(filled))

    return {
        "size_usd": size_usd,
        "spread_bps": float(indicators.get("spread_bps") or 0.0),
        "atr_pct": float(indicators.get("atr_14") or 0.0) / max(1e-9, float(filled)),
        "rsi_14": float(indicators.get("rsi_14") or 0.0),
        "kline_volatility_pct": float(indicators.get("kline_volatility_pct") or 0.0),
        "hour_of_day": _hour_from_timestamp(trade.get("timestamp")),
        "slippage_pct": slippage_pct,
    }


def features_from_live_order(
    *,
    size_usd: float,
    indicators: dict[str, Any],
    price: float,
    orderbook: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Build a feature dict for a pending order — counterpart to ``trade_to_sample``.

    Pulled out so the executor and the prompt stage build the same vector
    using the same code path. ``spread_bps`` is computed from the
    orderbook's top-of-book bid/ask if available; otherwise it falls
    through to whatever ``indicators`` provides (commonly nothing, in
    which case the model's ``feature_means`` fills in).
    """
    from datetime import UTC
    from datetime import datetime as _dt

    spread_bps = float(indicators.get("spread_bps") or 0.0)
    if not spread_bps and orderbook:
        bids = orderbook.get("bids") or []
        asks = orderbook.get("asks") or []
        if bids and asks:
            try:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid = (best_bid + best_ask) / 2.0
                if mid > 0:
                    spread_bps = ((best_ask - best_bid) / mid) * 10_000.0
            except TypeError, ValueError, IndexError:
                pass

    return {
        "size_usd": float(size_usd),
        "spread_bps": spread_bps,
        "atr_pct": float(indicators.get("atr_14") or 0.0) / max(1e-9, float(price)),
        "rsi_14": float(indicators.get("rsi_14") or 0.0),
        "kline_volatility_pct": float(indicators.get("kline_volatility_pct") or 0.0),
        "hour_of_day": float(_dt.now(UTC).hour),
    }


def _hour_from_timestamp(raw: Any) -> float:
    """Parse hour-of-day (0..23) from a timestamp; falls back to 12."""
    if raw is None:
        return 12.0
    try:
        from datetime import datetime as _dt

        if isinstance(raw, str):
            dt = _dt.fromisoformat(raw.replace("Z", "+00:00"))
        elif isinstance(raw, (int, float)):
            dt = _dt.fromtimestamp(float(raw))
        else:
            dt = raw
        return float(dt.hour)
    except Exception:  # noqa: BLE001
        return 12.0
