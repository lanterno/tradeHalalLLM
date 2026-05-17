"""OpenTelemetry-compatible span translation for cycle events.

Round-4 wave 8.D: the bot already emits structured `cycle.stage.start`
/ `cycle.stage.end` events on its in-process bus (Wave 5.A
`core/cycle_timeline.py` aggregates them). Operators running the
bot at scale want those events visible in their existing
observability stack (Tempo / Jaeger / Honeycomb / Datadog APM)
alongside broker / LLM / database calls. This module is the
**translation layer** — it converts the bus's stage events into
OTLP-compatible span structures, without pulling in the
`opentelemetry-sdk` as a hard dependency.

Why hand-rolled OTLP rather than the official SDK:

* The SDK has 30+ transitive dependencies and a non-trivial
  startup cost. The bot's per-cycle hot path can't afford that
  unless every operator is willing to pay for it.
* The OTLP wire format is well-specified: the v1.x JSON encoding
  is < 100 lines of dataclass-to-dict mapping. Operators who
  want to ship to a real backend can use this module's output
  with any HTTP client.
* Pure-Python keeps the translator testable without a collector
  or the SDK installed.

The translator handles three concerns:

* **Trace + span ID generation.** Each cycle gets a fresh
  trace_id; each stage gets a span_id. IDs are 16 / 8 bytes of
  cryptographically random hex per the OTLP spec.
* **Span hierarchy.** The cycle is a root span; every stage is
  a child. Pin: future nested stages (a stage that itself runs
  child stages) just add another parent_span_id link — the
  translator already supports it.
* **OTLP JSON encoding.** Each span dataclass exposes a
  `to_otlp_json()` method that produces the canonical JSON
  shape an OTLP/HTTP exporter accepts.

Halal alignment: tracing is observability-only — never opens a
position, never logs operator-identifying data, never includes
prompt content (LLM rationales contain operator strategy IP and
shouldn't ship to a third-party APM). The translator's `attrs`
filter explicitly drops keys matching a small denylist before
exporting.

Pure-Python (stdlib `secrets` + `json`); no opentelemetry-sdk /
DB / network. The caller wires the output to their preferred
HTTP client (httpx / requests / urllib).
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# ── ID generation ────────────────────────────────────────


def new_trace_id() -> str:
    """Generate a 16-byte (32-hex-char) trace ID per OTLP spec.

    Pin: 32 hex chars, lowercase, no separators — the OTLP wire
    format requires this exact shape; mismatched IDs are silently
    dropped by collectors."""
    return secrets.token_hex(16)


def new_span_id() -> str:
    """Generate an 8-byte (16-hex-char) span ID per OTLP spec."""
    return secrets.token_hex(8)


def _datetime_to_nanos(dt: datetime) -> int:
    """Convert a Python `datetime` to integer Unix nanoseconds.

    Pin: `datetime.timestamp() * 1e9` accumulates float error
    (a 42ms duration starting at 2026 lands ~128ns off). Compute
    integer microseconds first (Python's datetime is microsecond-
    precise), then multiply by 1000 — exact integer arithmetic
    preserves the precision the operator's APM displays.
    """
    seconds_int = int(dt.timestamp())
    return seconds_int * 1_000_000_000 + dt.microsecond * 1000


def _validate_id(value: str, *, expected_chars: int, label: str) -> None:
    if len(value) != expected_chars:
        raise ValueError(f"{label} must be {expected_chars} hex chars; got {len(value)}")
    if not all(c in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be lowercase hex; got {value!r}")


# ── Span vocabulary ──────────────────────────────────────


class SpanStatusCode(str, Enum):
    """OTLP span status codes.

    * ``UNSET`` — default; no explicit status set.
    * ``OK`` — operation completed successfully.
    * ``ERROR`` — operation failed; ``status_message`` carries
      the error string.

    Pin: an unfinished span (the cycle was killed mid-stage)
    keeps `UNSET` rather than `ERROR` — a missing END event
    isn't necessarily an error from the OTLP perspective; the
    operator sees the span as in-progress in their APM.
    """

    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class SpanKind(str, Enum):
    """OTLP span kinds.

    Pinned to three the bot uses:

    * ``INTERNAL`` — pure in-process work (most cycle stages).
    * ``CLIENT`` — outgoing call to broker / LLM / database.
    * ``SERVER`` — handling an incoming request (dashboard
      endpoint serving the operator).
    """

    INTERNAL = "SPAN_KIND_INTERNAL"
    CLIENT = "SPAN_KIND_CLIENT"
    SERVER = "SPAN_KIND_SERVER"


# ── Span dataclass ───────────────────────────────────────


@dataclass(frozen=True)
class Span:
    """One OTLP-compatible span.

    ``trace_id`` is shared across every span in one cycle;
    ``span_id`` is unique per span. ``parent_span_id`` is empty
    for the cycle root; populated for every child stage.

    Times are nanoseconds since Unix epoch — pin: OTLP spec
    insists on nanoseconds even though the Python `datetime`
    layer uses microseconds. We multiply by 1000 in `from_event`
    when constructing.
    """

    trace_id: str
    span_id: str
    parent_span_id: str
    name: str
    kind: SpanKind
    start_unix_nano: int
    end_unix_nano: int
    status_code: SpanStatusCode
    status_message: str
    attributes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_id(self.trace_id, expected_chars=32, label="trace_id")
        _validate_id(self.span_id, expected_chars=16, label="span_id")
        if self.parent_span_id and self.parent_span_id != "":
            _validate_id(self.parent_span_id, expected_chars=16, label="parent_span_id")
        if self.start_unix_nano < 0:
            raise ValueError(f"start_unix_nano must be >= 0; got {self.start_unix_nano}")
        if self.end_unix_nano < self.start_unix_nano:
            raise ValueError(
                f"end_unix_nano ({self.end_unix_nano}) must be >= "
                f"start_unix_nano ({self.start_unix_nano})"
            )

    @property
    def duration_nanos(self) -> int:
        return self.end_unix_nano - self.start_unix_nano

    def to_otlp_json(self) -> dict[str, Any]:
        """Encode as the canonical OTLP/HTTP JSON shape.

        Pin: the field names match the protobuf-derived JSON
        spec — `traceId`, `spanId`, `parentSpanId`, `kind`,
        `startTimeUnixNano`, `endTimeUnixNano`, `attributes` as
        a list of `{key, value: {stringValue: …}}` records.
        Operators ship this to the OTLP/HTTP `/v1/traces`
        endpoint without further transformation.
        """
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "startTimeUnixNano": str(self.start_unix_nano),
            "endTimeUnixNano": str(self.end_unix_nano),
            "status": {
                "code": self.status_code.value,
                "message": self.status_message,
            },
            "attributes": [
                {"key": k, "value": {"stringValue": v}} for k, v in sorted(self.attributes.items())
            ],
        }


# ── Stage event input ────────────────────────────────────


@dataclass(frozen=True)
class StageEvent:
    """A stage-event observation the translator consumes.

    Mirrors the shape of `core/cycle_pipeline.py`'s
    `cycle.stage.start` / `cycle.stage.end` bus events but with
    just the fields the translator needs.

    ``elapsed_ms`` is set on END events; ``error`` is set on
    failed END events. The translator uses both to build the
    span's timing + status.
    """

    name: str
    started_at: datetime
    ended_at: datetime | None
    elapsed_ms: float | None = None
    error: str | None = None
    kind: SpanKind = SpanKind.INTERNAL
    attributes: dict[str, str] = field(default_factory=dict)


# ── PII / IP filter ──────────────────────────────────────


# Keys that must NOT ship to a third-party APM. The cycle's
# stage attrs occasionally carry these for in-process logging;
# the translator strips them before exporting.
#
# Pin: this list is conservative-by-default. Adding a new attr
# without considering its operator-IP / PII implications would
# be a leak; pin so the lint / review process catches it.
_REDACTED_ATTRIBUTE_KEYS: frozenset[str] = frozenset(
    {
        "prompt",
        "rationale",
        "raw_response",
        "thinking",
        "api_key",
        "secret",
        "operator_id",
        "broker_key",
        "llm_token",
    }
)


def filter_attributes(attrs: dict[str, str]) -> dict[str, str]:
    """Drop denylisted keys; cap remaining values at 256 chars.

    Pin: the cap is the operator's belt-and-braces against an
    accidentally-large value sneaking into the export payload.
    A 50KB rationale would otherwise bloat every span."""
    out: dict[str, str] = {}
    for k, v in attrs.items():
        if k in _REDACTED_ATTRIBUTE_KEYS:
            continue
        if len(v) > 256:
            v = v[:253] + "..."
        out[k] = v
    return out


# ── Translation ──────────────────────────────────────────


def span_from_event(
    event: StageEvent,
    *,
    trace_id: str,
    parent_span_id: str = "",
) -> Span:
    """Convert one stage event into an OTLP-compatible span.

    Pin: an open stage (no ended_at) gets `end_unix_nano ==
    start_unix_nano` and `status=UNSET` — the operator sees
    the span as in-progress in their APM rather than missing
    entirely.
    """
    start_nanos = _datetime_to_nanos(event.started_at)
    if event.ended_at is None:
        end_nanos = start_nanos
        status_code = SpanStatusCode.UNSET
        status_message = "stage in progress"
    else:
        end_nanos = _datetime_to_nanos(event.ended_at)
        if event.error:
            status_code = SpanStatusCode.ERROR
            status_message = event.error
        else:
            status_code = SpanStatusCode.OK
            status_message = ""

    return Span(
        trace_id=trace_id,
        span_id=new_span_id(),
        parent_span_id=parent_span_id,
        name=event.name,
        kind=event.kind,
        start_unix_nano=start_nanos,
        end_unix_nano=end_nanos,
        status_code=status_code,
        status_message=status_message,
        attributes=filter_attributes(event.attributes),
    )


# ── Cycle-level builder ──────────────────────────────────


@dataclass(frozen=True)
class TraceBundle:
    """A complete cycle's worth of spans ready for export.

    ``trace_id`` is the cycle's trace ID; ``root_span`` is the
    overall cycle span (encompassing every stage); ``stage_spans``
    are children of the root.
    """

    trace_id: str
    root_span: Span
    stage_spans: list[Span] = field(default_factory=list)

    def all_spans(self) -> list[Span]:
        return [self.root_span, *self.stage_spans]

    def to_otlp_payload(self, *, service_name: str = "halal-trader") -> dict[str, Any]:
        """Build the full OTLP/HTTP request body.

        Pin: the wrapper shape is `resourceSpans[0].scopeSpans[0]`
        per the OTLP/HTTP v1 spec. The collector rejects payloads
        without this nesting.
        """
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": service_name},
                            },
                        ],
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "halal_trader.core.otel_translator"},
                            "spans": [s.to_otlp_json() for s in self.all_spans()],
                        }
                    ],
                }
            ]
        }

    def to_json(self, *, service_name: str = "halal-trader") -> str:
        """Convenience: dump the OTLP payload as a JSON string
        ready for `httpx.post(..., content=...)`."""
        return json.dumps(self.to_otlp_payload(service_name=service_name))


def build_trace(
    *,
    cycle_started_at: datetime,
    cycle_ended_at: datetime | None,
    stage_events: list[StageEvent],
    trace_id: str | None = None,
    cycle_attributes: dict[str, str] | None = None,
) -> TraceBundle:
    """Build a complete trace bundle for one cycle.

    The cycle gets a root span spanning [cycle_started_at,
    cycle_ended_at]; every stage event becomes a child span.

    Pin: when `cycle_ended_at` is None (cycle didn't complete),
    the root span uses the latest stage END as its end time, or
    the start time itself if no stage completed — pin so an
    incomplete cycle still produces a valid (non-empty-duration)
    trace rather than a zero-duration span the collector might
    reject.
    """
    tid = trace_id or new_trace_id()

    # Determine root span end time.
    root_end = cycle_ended_at
    if root_end is None:
        # Pick the latest END from any stage; fall back to start.
        completed_ends = [e.ended_at for e in stage_events if e.ended_at is not None]
        root_end = max(completed_ends) if completed_ends else cycle_started_at

    root = Span(
        trace_id=tid,
        span_id=new_span_id(),
        parent_span_id="",
        name="cycle",
        kind=SpanKind.INTERNAL,
        start_unix_nano=_datetime_to_nanos(cycle_started_at),
        end_unix_nano=_datetime_to_nanos(root_end),
        status_code=SpanStatusCode.UNSET if cycle_ended_at is None else SpanStatusCode.OK,
        status_message="cycle in progress" if cycle_ended_at is None else "",
        attributes=filter_attributes(cycle_attributes or {}),
    )

    stage_spans = [
        span_from_event(event, trace_id=tid, parent_span_id=root.span_id) for event in stage_events
    ]

    return TraceBundle(
        trace_id=tid,
        root_span=root,
        stage_spans=stage_spans,
    )


__all__ = [
    "Span",
    "SpanKind",
    "SpanStatusCode",
    "StageEvent",
    "TraceBundle",
    "build_trace",
    "filter_attributes",
    "new_span_id",
    "new_trace_id",
    "span_from_event",
]
