"""In-process tracing for the cycle pipeline.

OpenTelemetry would be the right long-term home for this — but pulling
the OTel SDK in just to time five span kinds is heavy and brings a
config surface (collectors, exporters, attributes) we don't need yet.

This module gives the same *shape* (Tracer, Span, attributes, events,
context-managed lifetime) on stdlib only, so:

* Cycle code can decorate stages today and read timings out via the
  in-memory exporter or the JSON-log exporter.
* When the team is ready, swapping to real OTel is a one-file change to
  the exporter — call sites don't move.

A single global :data:`tracer` is exposed; callers do::

    from halal_trader.core.tracing import tracer

    async def run_cycle(...):
        with tracer.span("cycle.fetch_klines", pair_count=len(pairs)) as sp:
            sp.set_attr("source", "websocket")
            ...
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from halal_trader.core.observability import (
    cycle_id_var,
    monitor_id_var,
    new_id,
    request_id_var,
)

logger = logging.getLogger(__name__)

_active_span_var: ContextVar["Span | None"] = ContextVar("_active_span", default=None)


# ── Span ──────────────────────────────────────────────────────────


@dataclass
class SpanEvent:
    name: str
    timestamp_ns: int
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A single named timing slice with attributes and child events."""

    name: str
    span_id: str
    parent_id: str | None = None
    start_ns: int = field(default_factory=time.monotonic_ns)
    end_ns: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    error: str | None = None
    cycle_id: str = ""
    monitor_id: str = ""
    request_id: str = ""

    @property
    def duration_ms(self) -> float:
        if self.end_ns is None:
            return 0.0
        return (self.end_ns - self.start_ns) / 1_000_000

    @property
    def closed(self) -> bool:
        return self.end_ns is not None

    def set_attr(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def add_event(self, name: str, **attrs: Any) -> None:
        self.events.append(SpanEvent(name=name, timestamp_ns=time.monotonic_ns(), attrs=attrs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration_ms, 3),
            "attrs": self.attrs,
            "events": [
                {"name": e.name, "ts_ns": e.timestamp_ns, "attrs": e.attrs} for e in self.events
            ],
            "error": self.error,
            "cycle_id": self.cycle_id,
            "monitor_id": self.monitor_id,
            "request_id": self.request_id,
        }


# ── Exporter contract ─────────────────────────────────────────────


class SpanExporter:
    """Exports finished spans somewhere — log, memory, OTLP later."""

    def export(self, span: Span) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class LogSpanExporter(SpanExporter):
    """Default: write each span as one structured ``debug`` log record.

    Goes through the same JSON filter the rest of the app uses, so spans
    flow into the existing log pipeline without extra plumbing.
    """

    def __init__(self, level: int = logging.DEBUG) -> None:
        self.level = level

    def export(self, span: Span) -> None:
        logger.log(
            self.level,
            "trace.span %s closed (%.2fms)",
            span.name,
            span.duration_ms,
            extra={"event": "trace.span", **span.to_dict()},
        )


class InMemorySpanExporter(SpanExporter):
    """Keeps the last N spans in process — for tests and the dashboard."""

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self._spans: list[Span] = []
        self._lock = threading.Lock()

    def export(self, span: Span) -> None:
        with self._lock:
            self._spans.append(span)
            if len(self._spans) > self.capacity:
                self._spans = self._spans[-self.capacity :]

    def spans(self) -> list[Span]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


# ── Tracer ────────────────────────────────────────────────────────


class Tracer:
    """Cheap span factory.

    One global instance; tests can swap the exporter for assertions.
    Use ``tracer.span("name", **attrs)`` as a context manager for sync
    code, ``async with tracer.aspan(...)`` for async.
    """

    def __init__(self, exporters: list[SpanExporter] | None = None) -> None:
        self._exporters = list(exporters or [LogSpanExporter()])

    def add_exporter(self, exporter: SpanExporter) -> None:
        self._exporters.append(exporter)

    def set_exporters(self, exporters: list[SpanExporter]) -> None:
        self._exporters = list(exporters)

    def _new_span(self, name: str, attrs: Mapping[str, Any]) -> Span:
        parent = _active_span_var.get()
        return Span(
            name=name,
            span_id=new_id("sp"),
            parent_id=parent.span_id if parent else None,
            attrs=dict(attrs),
            cycle_id=cycle_id_var.get(),
            monitor_id=monitor_id_var.get(),
            request_id=request_id_var.get(),
        )

    def _close(self, span: Span, error: BaseException | None) -> None:
        span.end_ns = time.monotonic_ns()
        if error is not None:
            span.error = f"{type(error).__name__}: {error}"
        for exp in self._exporters:
            try:
                exp.export(span)
            except Exception as exc:  # noqa: BLE001
                logger.warning("span exporter failed: %s", exc)

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        sp = self._new_span(name, attrs)
        token = _active_span_var.set(sp)
        err: BaseException | None = None
        try:
            yield sp
        except BaseException as e:
            err = e
            raise
        finally:
            _active_span_var.reset(token)
            self._close(sp, err)

    def aspan(self, name: str, **attrs: Any) -> "AsyncSpanContext":
        """Async context manager equivalent to :meth:`span`."""
        return AsyncSpanContext(self, name, attrs)


class AsyncSpanContext:
    """Async ``with`` wrapper that manages a single :class:`Span`."""

    def __init__(self, tracer: Tracer, name: str, attrs: Mapping[str, Any]) -> None:
        self._tracer = tracer
        self._name = name
        self._attrs = attrs
        self._span: Span | None = None
        self._token: Any = None

    async def __aenter__(self) -> Span:
        sp = self._tracer._new_span(self._name, self._attrs)
        self._token = _active_span_var.set(sp)
        self._span = sp
        return sp

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        assert self._span is not None
        _active_span_var.reset(self._token)
        self._tracer._close(self._span, exc)


# Module-level singleton (mirrors OTel's pattern).
tracer = Tracer()


def get_active_span() -> Span | None:
    return _active_span_var.get()
