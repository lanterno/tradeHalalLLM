"""Tests for the cross-cycle shared helpers.

``signals.timeframes.build_timeframe_text`` (analyzer + symbols → text)
owns the per-symbol loop the stock cycle's timeframe stage runs. The
cycle-level tests cover the stage wrapper; these tests cover the helper
directly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.signals.timeframes import build_timeframe_text

# ── build_timeframe_text ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_timeframe_text_empty_when_analyzer_missing():
    assert await build_timeframe_text(None, ["AAPL"]) == ""


@pytest.mark.asyncio
async def test_timeframe_text_empty_when_symbols_empty():
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(return_value={})
    assert await build_timeframe_text(analyzer, []) == ""
    assert analyzer.analyze.call_count == 0


@pytest.mark.asyncio
async def test_timeframe_text_formats_results():
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(
        return_value={
            "AAPL": {
                "alignment_score": 0.72,
                "per_tf": {"1Day": "RSI=58, MACD=bullish"},
                "support_resistance": [],
            }
        }
    )
    text = await build_timeframe_text(analyzer, ["AAPL"])
    assert "AAPL" in text
    assert "BULLISH" in text


@pytest.mark.asyncio
async def test_timeframe_text_swallows_analyzer_failure():
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(side_effect=RuntimeError("alpaca down"))
    assert await build_timeframe_text(analyzer, ["AAPL"]) == ""


@pytest.mark.asyncio
async def test_timeframe_text_empty_when_analyzer_returns_nothing():
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(return_value={})
    assert await build_timeframe_text(analyzer, ["AAPL"]) == ""
