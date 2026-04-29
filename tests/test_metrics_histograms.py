"""Tests for the prometheus_client histograms."""

from __future__ import annotations

from halal_trader.core.metrics import (
    observe_broker_call,
    observe_llm_call,
    observe_stage_latency,
    render_prometheus_text,
)


def test_render_includes_stage_histogram_after_observation() -> None:
    observe_stage_latency(name="fetch_klines", ms=12.5, error=None)
    text = render_prometheus_text().decode("utf-8")
    assert "halal_trader_stage_latency_ms_bucket" in text
    assert 'stage="fetch_klines"' in text


def test_render_includes_llm_histogram() -> None:
    observe_llm_call(provider="anthropic", model="claude-opus-4-7", ms=1234.0)
    text = render_prometheus_text().decode("utf-8")
    assert "halal_trader_llm_call_ms" in text
    assert 'provider="anthropic"' in text


def test_render_includes_broker_histogram_with_error_label() -> None:
    observe_broker_call(broker="binance", method="place_order", ms=85.0, error=False)
    observe_broker_call(broker="binance", method="place_order", ms=200.0, error=True)
    text = render_prometheus_text().decode("utf-8")
    assert 'error="0"' in text
    assert 'error="1"' in text
