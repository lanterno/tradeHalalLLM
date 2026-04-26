"""Tests for in-process tracing."""

from __future__ import annotations

import time

import pytest

from halal_trader.core.observability import cycle_context
from halal_trader.core.tracing import (
    InMemorySpanExporter,
    Span,
    Tracer,
    get_active_span,
)


def test_span_records_duration() -> None:
    exp = InMemorySpanExporter()
    tr = Tracer(exporters=[exp])
    with tr.span("test.op") as sp:
        time.sleep(0.005)
        assert sp.name == "test.op"
        assert not sp.closed
    spans = exp.spans()
    assert len(spans) == 1
    assert spans[0].name == "test.op"
    assert spans[0].closed
    assert spans[0].duration_ms >= 1.0
    assert spans[0].error is None


def test_span_attrs_and_events() -> None:
    exp = InMemorySpanExporter()
    tr = Tracer(exporters=[exp])
    with tr.span("test.op", pair="BTCUSDT") as sp:
        sp.set_attr("klines", 100)
        sp.add_event("fetched", count=100)
    span = exp.spans()[0]
    assert span.attrs["pair"] == "BTCUSDT"
    assert span.attrs["klines"] == 100
    assert span.events[0].name == "fetched"
    assert span.events[0].attrs["count"] == 100


def test_span_records_error() -> None:
    exp = InMemorySpanExporter()
    tr = Tracer(exporters=[exp])
    with pytest.raises(RuntimeError):
        with tr.span("test.op"):
            raise RuntimeError("boom")
    span = exp.spans()[0]
    assert span.error is not None
    assert "RuntimeError" in span.error
    assert "boom" in span.error


def test_nested_spans_track_parent() -> None:
    exp = InMemorySpanExporter()
    tr = Tracer(exporters=[exp])
    with tr.span("outer") as outer:
        with tr.span("inner") as inner:
            assert inner.parent_id == outer.span_id
            assert get_active_span() is inner
        assert get_active_span() is outer
    assert get_active_span() is None
    spans = {s.name: s for s in exp.spans()}
    assert spans["inner"].parent_id == spans["outer"].span_id
    assert spans["outer"].parent_id is None


def test_span_attaches_cycle_id() -> None:
    exp = InMemorySpanExporter()
    tr = Tracer(exporters=[exp])
    with cycle_context("cycle-deadbeef"):
        with tr.span("test.in_cycle"):
            pass
    span = exp.spans()[0]
    assert span.cycle_id == "cycle-deadbeef"


@pytest.mark.asyncio
async def test_async_span() -> None:
    import asyncio

    exp = InMemorySpanExporter()
    tr = Tracer(exporters=[exp])
    async with tr.aspan("async.op", k=1) as sp:
        await asyncio.sleep(0.005)
        assert sp.name == "async.op"
    span = exp.spans()[0]
    assert span.duration_ms >= 1.0
    assert span.attrs["k"] == 1


@pytest.mark.asyncio
async def test_async_span_records_error() -> None:
    exp = InMemorySpanExporter()
    tr = Tracer(exporters=[exp])
    with pytest.raises(ValueError):
        async with tr.aspan("async.op"):
            raise ValueError("no")
    span = exp.spans()[0]
    assert "ValueError" in span.error


def test_in_memory_exporter_capacity() -> None:
    exp = InMemorySpanExporter(capacity=3)
    tr = Tracer(exporters=[exp])
    for i in range(7):
        with tr.span(f"op-{i}"):
            pass
    spans = exp.spans()
    assert len(spans) == 3
    assert [s.name for s in spans] == ["op-4", "op-5", "op-6"]


def test_span_to_dict_serializable() -> None:
    import json

    exp = InMemorySpanExporter()
    tr = Tracer(exporters=[exp])
    with tr.span("test.op", n=5) as sp:
        sp.add_event("midpoint", k="v")
    d = exp.spans()[0].to_dict()
    s = json.dumps(d)  # must serialize cleanly
    assert "test.op" in s
    assert "midpoint" in s


def test_exporter_failure_does_not_break_span() -> None:
    class _BoomExporter:
        def export(self, span: Span) -> None:
            raise RuntimeError("exporter died")

    captured = InMemorySpanExporter()
    tr = Tracer(exporters=[_BoomExporter(), captured])
    # Should NOT raise — exporter failures swallowed
    with tr.span("test.op"):
        pass
    assert len(captured.spans()) == 1
