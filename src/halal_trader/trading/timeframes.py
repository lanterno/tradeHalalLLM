"""Stocks multi-timeframe analyzer — Alpaca-backed sibling of crypto's.

Reuses ``crypto.timeframes.TimeframeAnalyzer`` for the alignment and
support/resistance math — the only broker-specific piece is the
per-timeframe bar fetch, which goes through Alpaca via the existing
``Broker`` port. Bars are coerced into ``Kline`` shape via
:mod:`trading.bars` so ``compute_all`` runs unchanged.

Timeframes are chosen for daily-cadence equity trading: 1Hour gives
intraday context, 1Day covers the recent trend, 1Week catches longer-
horizon momentum. (1Min/5Min are skipped — Alpaca's free tier is rate-
limited and the LLM cycle runs every 15 min, so finer bars don't pay
their cost.)
"""

from __future__ import annotations

import logging
from typing import Any

from halal_trader.crypto.timeframes import TimeframeAnalyzer
from halal_trader.domain.models import Kline
from halal_trader.domain.ports import Broker
from halal_trader.trading.bars import bars_to_klines

logger = logging.getLogger(__name__)


_STOCK_TIMEFRAMES: list[tuple[str, int]] = [
    ("1Hour", 1800),  # 30-min cache
    ("1Day", 21600),  # 6-hour cache
    ("1Week", 86400),  # 1-day cache
]

# Alpaca's bar-fetch ``days`` arg is approximate; pick a window per
# interval that yields ≥30 candles for indicator stability.
_DAYS_BY_INTERVAL: dict[str, int] = {
    "1Hour": 7,  # ~7 trading days × ~6.5 hours = ~45 hourly bars
    "1Day": 60,  # 60 trading days
    "1Week": 365,  # 52 weekly bars
}


class StockTimeframeAnalyzer(TimeframeAnalyzer):
    """TimeframeAnalyzer subclass that pulls bars from Alpaca."""

    _timeframes = _STOCK_TIMEFRAMES

    def __init__(self, broker: Broker) -> None:
        # Bypass parent __init__ — we want a Broker, not a BinanceClient.
        self._broker = broker  # type: ignore[assignment]
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    async def _fetch_klines(self, pair: str, interval: str, *, limit: int) -> list[Kline]:
        """Pull Alpaca bars at the requested timeframe and coerce to Kline."""
        days = _DAYS_BY_INTERVAL.get(interval, 30)
        try:
            bars = await self._broker.get_stock_bars(pair, days=days, timeframe=interval)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Alpaca bars fetch failed for %s @ %s: %s", pair, interval, exc)
            return []
        return bars_to_klines(bars)
