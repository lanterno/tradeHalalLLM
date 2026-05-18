"""Tests for ``trading.timeframes.StockTimeframeAnalyzer``.

The Alpaca-fetch adapter delegates the alignment / S-R / summary math
to the crypto base class; the only stock-specific surface is the
``_fetch_klines`` override (interval + days mapping → Alpaca bars →
``Kline`` list).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.trading.timeframes import StockTimeframeAnalyzer


def _bar(o: float, h: float, low: float, c: float, v: float = 1_000.0) -> dict[str, Any]:
    return {"o": o, "h": h, "l": low, "c": c, "v": v}


def _series(start: float, n: int, step: float = 0.5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    price = start
    for _ in range(n):
        out.append(_bar(price, price + 0.5, price - 0.5, price + step))
        price += step
    return out


@pytest.mark.asyncio
async def test_fetch_klines_passes_per_interval_lookback_to_broker():
    """Each timeframe maps to a sensible ``days`` window for indicator stability."""
    broker = MagicMock()
    broker.get_stock_bars = AsyncMock(return_value=_series(100, 30))
    analyzer = StockTimeframeAnalyzer(broker)

    await analyzer._fetch_klines("AAPL", "1Hour", limit=100)
    broker.get_stock_bars.assert_called_once()
    kwargs = broker.get_stock_bars.call_args.kwargs
    # 1Hour requests ~7 days of bars; 1Day requests 60; 1Week requests 365.
    assert kwargs == {"days": 7, "timeframe": "1Hour"}


@pytest.mark.asyncio
async def test_fetch_klines_returns_empty_on_broker_failure():
    broker = MagicMock()
    broker.get_stock_bars = AsyncMock(side_effect=RuntimeError("alpaca down"))
    analyzer = StockTimeframeAnalyzer(broker)

    klines = await analyzer._fetch_klines("AAPL", "1Day", limit=100)
    assert klines == []


@pytest.mark.asyncio
async def test_analyze_runs_all_three_timeframes():
    """A single ``analyze()`` call hits Alpaca once per configured TF."""
    broker = MagicMock()
    broker.get_stock_bars = AsyncMock(return_value=_series(100, 50))
    analyzer = StockTimeframeAnalyzer(broker)

    results = await analyzer.analyze(["AAPL"])

    assert "AAPL" in results
    # Three timeframes (1Hour / 1Day / 1Week) → three fetches.
    assert broker.get_stock_bars.call_count == 3
    intervals = {call.kwargs["timeframe"] for call in broker.get_stock_bars.call_args_list}
    assert intervals == {"1Hour", "1Day", "1Week"}
    # Per-TF summaries surface the indicator output.
    assert "per_tf" in results["AAPL"]
    assert set(results["AAPL"]["per_tf"]).issubset({"1Hour", "1Day", "1Week"})


@pytest.mark.asyncio
async def test_analyze_skips_timeframes_with_too_few_bars():
    """Timeframes returning < 20 bars are dropped (indicator stability floor)."""
    broker = MagicMock()
    broker.get_stock_bars = AsyncMock(return_value=_series(100, 5))  # too few
    analyzer = StockTimeframeAnalyzer(broker)

    results = await analyzer.analyze(["AAPL"])
    # Analyzer ran but produced no per-TF entries.
    assert results.get("AAPL", {}).get("per_tf") == {}
