"""Tests for :func:`format_ml_signals_for_prompt`.

`build_ml_signals_text` is covered in `test_cycle_shared_helpers`.
This file pins the underlying formatter that combines forecasts +
anomalies + per-trade ML confidence into one prompt block.
"""

from __future__ import annotations

from halal_trader.ml.anomaly import format_ml_signals_for_prompt

# ── Empty / sentinel paths ──────────────────────────────────


def test_empty_inputs_returns_no_data_sentinel():
    assert format_ml_signals_for_prompt("") == "No ML model data available."


def test_only_forecasts_sentinel_text_is_skipped():
    """When the forecaster has no data, it returns a sentinel string —
    we mustn't render that as if it were a real forecast section."""
    out = format_ml_signals_for_prompt("No ML price forecasts available.")
    assert out == "No ML model data available."


def test_anomalies_dict_with_no_true_entries_yields_no_section():
    """A dict full of non-anomalies emits nothing — saves prompt space."""
    out = format_ml_signals_for_prompt(
        "",
        anomalies={"BTCUSDT": (False, 0.1), "ETHUSDT": (False, 0.05)},
    )
    assert out == "No ML model data available."


# ── Forecast section ────────────────────────────────────────


def test_forecast_text_renders_with_chronos_header():
    out = format_ml_signals_for_prompt("  BTCUSDT: ML predicts UP 2%")
    assert "Price Forecasts (Chronos-T5):" in out
    assert "BTCUSDT" in out


# ── Anomaly section ─────────────────────────────────────────


def test_only_true_anomalies_are_rendered():
    """Mixed dict — only the ANOMALY=True rows get a line."""
    out = format_ml_signals_for_prompt(
        "",
        anomalies={
            "BTCUSDT": (True, 0.95),
            "ETHUSDT": (False, 0.10),
        },
    )
    assert "Anomaly Detection:" in out
    assert "BTCUSDT: ANOMALY DETECTED" in out
    assert "ETHUSDT" not in out


def test_anomaly_score_rendered_to_3_decimals():
    out = format_ml_signals_for_prompt("", anomalies={"BTCUSDT": (True, 0.123456)})
    assert "0.123" in out


# ── Confidence section ─────────────────────────────────────


def test_confidence_label_high_above_0_7():
    out = format_ml_signals_for_prompt("", ml_confidence={"BTCUSDT": 0.85})
    assert "HIGH" in out
    assert "85%" in out


def test_confidence_label_medium_between_0_5_and_0_7():
    out = format_ml_signals_for_prompt("", ml_confidence={"BTCUSDT": 0.6})
    assert "MEDIUM" in out


def test_confidence_label_low_at_or_below_0_5():
    out = format_ml_signals_for_prompt("", ml_confidence={"BTCUSDT": 0.4})
    assert "LOW" in out


def test_confidence_pairs_sorted_alphabetically():
    out = format_ml_signals_for_prompt("", ml_confidence={"ZRXUSDT": 0.7, "BTCUSDT": 0.7})
    btc = out.find("BTCUSDT")
    zrx = out.find("ZRXUSDT")
    assert 0 <= btc < zrx


# ── Combined sections ──────────────────────────────────────


def test_all_three_sections_render_when_present():
    out = format_ml_signals_for_prompt(
        "  BTCUSDT: ML predicts UP 2%",
        anomalies={"ETHUSDT": (True, 0.91)},
        ml_confidence={"BTCUSDT": 0.75},
    )
    assert "Price Forecasts" in out
    assert "Anomaly Detection" in out
    assert "Trade Confidence" in out
