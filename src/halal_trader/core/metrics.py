"""Prometheus histograms + counters for the cycle pipeline.

Optional dependency: if ``prometheus_client`` isn't installed, every
helper becomes a no-op. The dashboard's ``/metrics`` endpoint also
exposes the gauge-shaped snapshots from ``web/prometheus.py``; this
module is the histogram side.

Histograms exposed (Wave J):

* ``halal_trader_stage_latency_ms`` — labels ``stage``, ``error``;
  per-stage cycle latency including failed runs.
* ``halal_trader_llm_call_ms`` — labels ``provider``, ``model``;
  end-to-end LLM-call latency.
* ``halal_trader_broker_call_ms`` — labels ``broker``, ``method``,
  ``error``; broker-side latency.

Buckets are tuned for human-scale latencies (1ms – 60s).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Latency buckets in milliseconds — covers the full range of
# observed values from "parsed JSON in 2ms" to "30s LLM call".
_LATENCY_BUCKETS_MS = (
    1.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1_000.0,
    2_500.0,
    5_000.0,
    10_000.0,
    30_000.0,
    60_000.0,
)


_metrics: dict[str, Any] = {}


def _ensure_metrics() -> dict[str, Any]:
    """Lazily build the metric objects on first use.

    Building inside a try/except so a missing ``prometheus_client``
    install doesn't blow up at import time — every helper checks
    the dict and no-ops when empty.
    """
    if _metrics:
        return _metrics
    try:
        from prometheus_client import Counter, Histogram

        _metrics["stage_latency"] = Histogram(
            "halal_trader_stage_latency_ms",
            "Per-stage cycle latency in milliseconds",
            labelnames=("stage", "error"),
            buckets=_LATENCY_BUCKETS_MS,
        )
        _metrics["llm_call_ms"] = Histogram(
            "halal_trader_llm_call_ms",
            "LLM-provider call latency in milliseconds",
            labelnames=("provider", "model"),
            buckets=_LATENCY_BUCKETS_MS,
        )
        _metrics["broker_call_ms"] = Histogram(
            "halal_trader_broker_call_ms",
            "Broker call latency in milliseconds",
            labelnames=("broker", "method", "error"),
            buckets=_LATENCY_BUCKETS_MS,
        )
        _metrics["events_published"] = Counter(
            "halal_trader_events_published_total",
            "Total events published to the in-process bus",
            labelnames=("topic",),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("prometheus_client unavailable: %s", exc)
        _metrics["_disabled"] = True
    return _metrics


def observe_stage_latency(*, name: str, ms: float, error: str | None) -> None:
    m = _ensure_metrics()
    if "_disabled" in m:
        return
    m["stage_latency"].labels(stage=name, error="1" if error else "0").observe(ms)


def observe_llm_call(*, provider: str, model: str, ms: float) -> None:
    m = _ensure_metrics()
    if "_disabled" in m:
        return
    m["llm_call_ms"].labels(provider=provider, model=model).observe(ms)


def observe_broker_call(*, broker: str, method: str, ms: float, error: bool) -> None:
    m = _ensure_metrics()
    if "_disabled" in m:
        return
    m["broker_call_ms"].labels(broker=broker, method=method, error="1" if error else "0").observe(
        ms
    )


def event_published(topic: str) -> None:
    m = _ensure_metrics()
    if "_disabled" in m:
        return
    m["events_published"].labels(topic=topic).inc()


def render_prometheus_text() -> bytes:
    """Render all registered Prometheus metrics in exposition format."""
    try:
        from prometheus_client import REGISTRY, generate_latest

        _ensure_metrics()
        out: bytes = generate_latest(REGISTRY)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.debug("prometheus_client unavailable: %s", exc)
        return b""
