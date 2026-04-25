"""Single source of truth for the ML feature vector.

The IndicatorSnapshot table records 9 features at trade entry; the
anomaly detector + signal classifier + retrainer all train on the same
list. Any new feature added here must also be populated by
:mod:`crypto.indicators.compute_all` (the snapshot writer pulls from
that mapping).
"""

from __future__ import annotations

from typing import Final

FEATURE_KEYS: Final[tuple[str, ...]] = (
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
