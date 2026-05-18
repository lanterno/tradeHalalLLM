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


# ── Wave-J wiring: confirm observation sites actually fire ───────


def test_timed_broker_call_decorator_records_success() -> None:
    """The ``timed_broker_call`` decorator should observe a successful
    call with ``error=False`` and a finite millisecond value."""
    import asyncio

    from halal_trader.core.metrics import timed_broker_call

    @timed_broker_call("binance", "fake_method")
    async def fake_call() -> str:
        return "ok"

    result = asyncio.run(fake_call())
    assert result == "ok"
    text = render_prometheus_text().decode("utf-8")
    assert 'broker="binance"' in text
    assert 'method="fake_method"' in text


def test_timed_broker_call_decorator_records_failure_and_reraises() -> None:
    """Exceptions in the wrapped coro propagate; the histogram still gets
    the ``error="1"`` observation."""
    import asyncio

    from halal_trader.core.metrics import timed_broker_call

    @timed_broker_call("binance", "fails")
    async def boom() -> None:
        raise RuntimeError("api down")

    try:
        asyncio.run(boom())
    except RuntimeError:
        pass
    else:
        raise AssertionError("decorator should not swallow exceptions")
    text = render_prometheus_text().decode("utf-8")
    assert 'method="fails"' in text
    assert 'error="1"' in text


def test_record_usage_emits_llm_histogram() -> None:
    """``BaseLLM._record_usage`` should both stamp ``last_usage`` and emit
    the ``halal_trader_llm_call_ms`` observation."""
    from halal_trader.core.llm.base import BaseLLM, CallUsage

    class _DummyLLM(BaseLLM):
        async def generate(self, prompt, system=None):  # type: ignore[override]
            return ""

    llm = _DummyLLM(model="claude-test-model")
    usage = CallUsage(
        provider="anthropic",
        model="claude-test-model",
        input_tokens=100,
        output_tokens=20,
        elapsed_ms=1234,
    )
    llm._record_usage(usage)
    assert llm.last_usage is usage
    text = render_prometheus_text().decode("utf-8")
    assert "halal_trader_llm_call_ms" in text
    assert 'model="claude-test-model"' in text


def test_record_usage_skips_emit_when_provider_missing() -> None:
    """If ``CallUsage`` is missing provider or model (defensive fallback),
    the histogram is skipped — never want a label called ``""``."""
    from halal_trader.core.llm.base import BaseLLM, CallUsage

    class _DummyLLM(BaseLLM):
        async def generate(self, prompt, system=None):  # type: ignore[override]
            return ""

    llm = _DummyLLM(model="claude-x")
    usage = CallUsage(provider="", model="", elapsed_ms=123)
    llm._record_usage(usage)
    text = render_prometheus_text().decode("utf-8")
    assert 'provider=""' not in text
    assert 'model=""' not in text


def test_event_bus_publish_increments_events_counter() -> None:
    """``EventBus.publish`` should bump
    ``halal_trader_events_published_total{topic=...}``."""
    import asyncio

    from halal_trader.core.event_bus import EventBus

    bus = EventBus()
    asyncio.run(bus.publish("test.topic", {"x": 1}))
    text = render_prometheus_text().decode("utf-8")
    assert "halal_trader_events_published_total" in text
    assert 'topic="test.topic"' in text


def test_event_bus_publish_increments_per_topic() -> None:
    """Two publishes on the same topic → counter ≥ 2."""
    import asyncio

    from halal_trader.core.event_bus import EventBus

    bus = EventBus()
    asyncio.run(bus.publish("repeated.topic"))
    asyncio.run(bus.publish("repeated.topic"))
    text = render_prometheus_text().decode("utf-8")
    # The counter line for this topic should report at least 2.
    counter_line = next(
        (
            ln
            for ln in text.splitlines()
            if 'halal_trader_events_published_total{topic="repeated.topic"}' in ln
        ),
        None,
    )
    assert counter_line is not None
    value = float(counter_line.split()[-1])
    assert value >= 2.0
