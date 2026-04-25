"""Stock-side indicator snapshot writer.

Mirrors the crypto pattern: when a stock BUY fills we record the same
9-feature indicator vector keyed by ``trade_id`` so the shared
``RetrainingScheduler`` can label outcomes and retrain ML models on
stock trades too.

The ``IndicatorSnapshot.pair`` column is reused — it stores whichever
trading symbol the snapshot is for (BTCUSDT, AAPL, …) regardless of
market. Adding a "market" column would mean another migration; the
trade_id FK is already enough to disambiguate.
"""

from __future__ import annotations

import logging
from typing import Any

from halal_trader.crypto.indicators import compute_all
from halal_trader.domain.models import Kline
from halal_trader.db.repository import Repository
from halal_trader.trading.risk import _bars_to_klines

logger = logging.getLogger(__name__)


_FEATURE_KEYS = (
    "rsi_14",
    "macd_histogram",
    "volume_ratio",
    "atr_14",
    "bb_position",
    "ema_9",
    "ema_21",
    "vwap",
    "price_change_5m",
)


async def record_stock_snapshot(
    *,
    repo: Repository,
    trade_id: int,
    symbol: str,
    bars: Any,
) -> int | None:
    """Compute indicators from Alpaca bars and persist a snapshot row.

    Returns the snapshot id on success, ``None`` on insufficient data or
    persistence failure (the cycle should not abort on a snapshot miss).
    """
    klines: list[Kline] = _bars_to_klines(bars)
    if len(klines) < 30:
        logger.debug(
            "Skipping snapshot for %s #%d: only %d klines available",
            symbol,
            trade_id,
            len(klines),
        )
        return None

    indicators = compute_all(klines)
    if "error" in indicators:
        logger.debug(
            "Skipping snapshot for %s #%d: indicator error %s",
            symbol,
            trade_id,
            indicators.get("error"),
        )
        return None

    payload: dict[str, float] = {}
    for key in _FEATURE_KEYS:
        val = indicators.get(key)
        if val is None:
            continue
        try:
            payload[key] = float(val)
        except TypeError, ValueError:
            continue

    try:
        return await repo.record_indicator_snapshot(
            trade_id=trade_id, pair=symbol, indicators=payload
        )
    except Exception as exc:
        logger.debug("Failed to record stock snapshot for #%d: %s", trade_id, exc)
        return None
