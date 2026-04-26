"""Multi-timeframe analyzer tests — alignment scoring and prompt formatting."""

from __future__ import annotations

import pytest

from halal_trader.crypto.timeframes import TimeframeAnalyzer, format_timeframes_for_prompt


def _ind(
    *,
    rsi=None,
    macd_hist=None,
    ema9=None,
    ema21=None,
    ema50=None,
    price=None,
    bb_upper=None,
    bb_lower=None,
    vwap=None,
):
    out: dict = {}
    if rsi is not None:
        out["rsi_14"] = rsi
    if macd_hist is not None:
        out["macd_histogram"] = macd_hist
    if ema9 is not None:
        out["ema_9"] = ema9
    if ema21 is not None:
        out["ema_21"] = ema21
    if ema50 is not None:
        out["ema_50"] = ema50
    if price is not None:
        out["current_price"] = price
    if bb_upper is not None:
        out["bb_upper"] = bb_upper
    if bb_lower is not None:
        out["bb_lower"] = bb_lower
    if vwap is not None:
        out["vwap"] = vwap
    return out


@pytest.fixture
def analyzer() -> TimeframeAnalyzer:
    """An analyzer with a None broker — we exercise the pure-logic methods only."""
    return TimeframeAnalyzer(broker=None)  # type: ignore[arg-type]


def test_alignment_all_bullish_returns_positive(analyzer):
    tf_data = {
        "5m": _ind(macd_hist=0.5, ema9=110, ema21=100, ema50=95, price=115),
        "1h": _ind(macd_hist=0.3, ema9=108, ema21=105, ema50=100, price=110),
        "4h": _ind(macd_hist=0.2, ema9=107, ema21=104, ema50=100, price=109),
    }
    score = analyzer._compute_alignment(tf_data)
    assert score == pytest.approx(1.0)


def test_alignment_all_bearish_returns_negative(analyzer):
    tf_data = {
        "5m": _ind(macd_hist=-0.5, ema9=90, ema21=100, ema50=110, price=85),
        "1h": _ind(macd_hist=-0.3, ema9=95, ema21=100, ema50=105, price=90),
    }
    score = analyzer._compute_alignment(tf_data)
    assert score == pytest.approx(-1.0)


def test_alignment_mixed_signals_returns_near_zero(analyzer):
    """When the two TFs disagree the alignment cancels."""
    tf_data = {
        "5m": _ind(macd_hist=1, ema9=110, ema21=100, ema50=95, price=115),
        "1h": _ind(macd_hist=-1, ema9=95, ema21=105, ema50=110, price=90),
    }
    score = analyzer._compute_alignment(tf_data)
    assert abs(score) < 0.1


def test_alignment_empty_data_returns_zero(analyzer):
    assert analyzer._compute_alignment({}) == 0.0


def test_alignment_skips_error_indicators(analyzer):
    tf_data = {
        "5m": {"error": "no data"},
        "1h": _ind(macd_hist=0.5, ema9=110, ema21=100, ema50=95, price=115),
    }
    # Only the 1h TF contributes — full bullish.
    assert analyzer._compute_alignment(tf_data) == pytest.approx(1.0)


def test_support_resistance_extracts_bb_vwap_ema50(analyzer):
    tf_data = {
        "1h": _ind(bb_upper=110, bb_lower=90, vwap=100, ema50=95),
    }
    levels = analyzer._compute_support_resistance(tf_data)
    types = {lv["type"] for lv in levels}
    assert {"resistance", "support", "pivot", "dynamic"}.issubset(types)
    # Sorted by level value.
    vals = [lv["level"] for lv in levels]
    assert vals == sorted(vals)


def test_summarize_tf_combines_signals(analyzer):
    text = analyzer._summarize_tf(_ind(rsi=42, macd_hist=0.4, ema9=105, ema21=100))
    assert "RSI=42" in text
    assert "MACD=bullish" in text
    assert "EMA=bull cross" in text


def test_summarize_tf_handles_error_payload(analyzer):
    assert analyzer._summarize_tf({"error": "x"}) == "insufficient data"


def test_format_for_prompt_labels_alignment_buckets():
    results = {
        "BTCUSDT": {"alignment_score": 0.5, "per_tf": {"1h": "RSI=55"}, "support_resistance": []},
        "ETHUSDT": {"alignment_score": -0.5, "per_tf": {}, "support_resistance": []},
        "SOLUSDT": {"alignment_score": 0.0, "per_tf": {}, "support_resistance": []},
    }
    text = format_timeframes_for_prompt(results)
    assert "BULLISH" in text
    assert "BEARISH" in text
    assert "MIXED" in text
    # Sorted alphabetically by symbol so output is deterministic for caching.
    assert text.find("BTCUSDT") < text.find("ETHUSDT") < text.find("SOLUSDT")


def test_format_for_prompt_handles_empty():
    assert "No multi-timeframe data" in format_timeframes_for_prompt({})
