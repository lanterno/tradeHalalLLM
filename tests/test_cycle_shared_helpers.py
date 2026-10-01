"""Tests for the cross-cycle shared helpers.

Two small functions own per-symbol loops the stock cycle's stages run:

* ``ml.anomaly.build_ml_signals_text`` — anomaly + signal classifier
  + optional pre-rendered forecasts → text
* ``signals.timeframes.build_timeframe_text`` — analyzer + symbols → text

The cycle-level tests cover the stage wrappers; these tests cover the
helpers directly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.ml.anomaly import build_ml_signals_text
from halal_trader.signals.timeframes import build_timeframe_text

# ── build_ml_signals_text ────────────────────────────────────────


def test_ml_signals_returns_forecasts_when_no_detectors():
    """With both detectors off, the helper just passes the forecasts through."""
    text = build_ml_signals_text(
        indicators_by_symbol={"AAPL": {}},
        anomaly_detector=None,
        signal_classifier=None,
        forecasts_text="some forecast text",
    )
    assert text == "some forecast text"


def test_ml_signals_returns_forecasts_when_indicators_empty():
    anomaly = MagicMock()
    text = build_ml_signals_text(
        indicators_by_symbol={},
        anomaly_detector=anomaly,
        forecasts_text="forecast block",
    )
    assert text == "forecast block"
    assert anomaly.detect.call_count == 0


def test_ml_signals_combines_anomaly_and_confidence():
    anomaly = MagicMock()
    anomaly.detect.return_value = (True, 0.92)
    signal = MagicMock()
    signal.predict_confidence.return_value = 0.71
    text = build_ml_signals_text(
        indicators_by_symbol={"AAPL": {"rsi_14": 50}},
        anomaly_detector=anomaly,
        signal_classifier=signal,
    )
    # add_sample fed the rolling distribution; detect produced the anomaly tuple.
    assert anomaly.add_sample.call_count == 1
    assert "ANOMALY DETECTED" in text
    assert "AAPL" in text
    assert "ML confidence" in text
    assert "71%" in text


def test_ml_signals_skips_error_indicators():
    anomaly = MagicMock()
    anomaly.detect.return_value = (False, 0.1)
    signal = MagicMock()
    signal.predict_confidence.return_value = 0.5
    indicators = {
        "AAPL": {"rsi_14": 50},
        "BAD": {"error": "insufficient data"},
    }
    build_ml_signals_text(
        indicators_by_symbol=indicators,
        anomaly_detector=anomaly,
        signal_classifier=signal,
    )
    assert anomaly.add_sample.call_count == 1
    assert signal.predict_confidence.call_count == 1


def test_ml_signals_swallows_detector_failure():
    anomaly = MagicMock()
    anomaly.detect.side_effect = RuntimeError("anomaly down")
    text = build_ml_signals_text(
        indicators_by_symbol={"AAPL": {"rsi_14": 50}},
        anomaly_detector=anomaly,
        forecasts_text="forecast survives",
    )
    # Failure is swallowed; pre-rendered forecasts still pass through.
    assert text == "forecast survives"


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
