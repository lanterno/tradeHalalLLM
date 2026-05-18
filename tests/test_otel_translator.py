"""Tests for `core/otel_translator.py`.

Pins ID generation length + alphabet, span-validation invariants,
the open-stage UNSET status pin, the redacted-attribute filter,
the OTLP/HTTP wrapper structure, and the build_trace
behaviour around incomplete cycles.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.core.otel_translator import (
    _REDACTED_ATTRIBUTE_KEYS,
    Span,
    SpanKind,
    SpanStatusCode,
    StageEvent,
    TraceBundle,
    build_trace,
    filter_attributes,
    new_span_id,
    new_trace_id,
    span_from_event,
)

# ── ID generation ────────────────────────────────────────


def test_new_trace_id_is_32_lowercase_hex():
    """Pin: 32 hex chars, lowercase, no separators — OTLP wire
    format requires this exact shape; mismatched IDs are silently
    dropped by collectors."""
    tid = new_trace_id()
    assert len(tid) == 32
    assert re.fullmatch(r"[0-9a-f]{32}", tid)


def test_new_span_id_is_16_lowercase_hex():
    sid = new_span_id()
    assert len(sid) == 16
    assert re.fullmatch(r"[0-9a-f]{16}", sid)


def test_trace_ids_are_unique_under_repeated_calls():
    """Pin against a determinism bug — two calls must produce
    different IDs."""
    ids = {new_trace_id() for _ in range(20)}
    assert len(ids) == 20


def test_span_ids_are_unique_under_repeated_calls():
    ids = {new_span_id() for _ in range(20)}
    assert len(ids) == 20


# ── Span validation ──────────────────────────────────────


def _good_span(**overrides):
    base = dict(
        trace_id="0" * 32,
        span_id="1" * 16,
        parent_span_id="",
        name="cycle",
        kind=SpanKind.INTERNAL,
        start_unix_nano=1_000_000,
        end_unix_nano=2_000_000,
        status_code=SpanStatusCode.OK,
        status_message="",
        attributes={},
    )
    base.update(overrides)
    return Span(**base)


def test_span_rejects_short_trace_id():
    with pytest.raises(ValueError, match="trace_id"):
        _good_span(trace_id="abc")


def test_span_rejects_uppercase_hex_in_id():
    """Pin: lowercase hex only; uppercase produces wire-format
    drops at the collector."""
    with pytest.raises(ValueError, match="lowercase hex"):
        _good_span(trace_id="A" * 32)


def test_span_rejects_short_span_id():
    with pytest.raises(ValueError, match="span_id"):
        _good_span(span_id="abc")


def test_span_rejects_negative_start_time():
    with pytest.raises(ValueError, match="start_unix_nano"):
        _good_span(start_unix_nano=-1)


def test_span_rejects_end_before_start():
    """Pin: end < start would produce negative durations; reject."""
    with pytest.raises(ValueError, match="end_unix_nano"):
        _good_span(start_unix_nano=2_000_000, end_unix_nano=1_000_000)


def test_span_accepts_zero_duration():
    """Pin: end == start is valid — represents an open stage
    where the start/end aliases."""
    s = _good_span(start_unix_nano=1_000_000, end_unix_nano=1_000_000)
    assert s.duration_nanos == 0


def test_span_parent_span_id_empty_string_accepted():
    """The cycle root span has empty parent_span_id; pin no
    validation on the empty case."""
    s = _good_span(parent_span_id="")
    assert s.parent_span_id == ""


def test_span_parent_span_id_validated_when_set():
    """Pin: a non-empty parent_span_id must follow the same
    16-hex-char rule."""
    with pytest.raises(ValueError, match="parent_span_id"):
        _good_span(parent_span_id="abc")


# ── duration helper ──────────────────────────────────────


def test_duration_nanos_computed_correctly():
    s = _good_span(start_unix_nano=1_000, end_unix_nano=5_000)
    assert s.duration_nanos == 4_000


# ── filter_attributes ────────────────────────────────────


def test_filter_drops_redacted_keys():
    """Pin: every key in the denylist must be dropped — the
    denylist exists to prevent operator IP / PII from leaving
    the bot to a third-party APM."""
    attrs = {
        "prompt": "secret-strategy",
        "rationale": "secret-reasoning",
        "raw_response": "...",
        "thinking": "...",
        "api_key": "sk-...",
        "secret": "...",
        "operator_id": "user-1",
        "broker_key": "...",
        "llm_token": "...",
        "pair": "BTCUSDT",  # safe
    }
    out = filter_attributes(attrs)
    assert out == {"pair": "BTCUSDT"}


def test_filter_caps_long_values():
    """Pin: 256-char cap so a runaway attr can't bloat the
    payload."""
    attrs = {"long": "x" * 1000}
    out = filter_attributes(attrs)
    assert len(out["long"]) == 256
    assert out["long"].endswith("...")


def test_filter_passes_short_values_unchanged():
    out = filter_attributes({"pair": "BTCUSDT", "side": "buy"})
    assert out == {"pair": "BTCUSDT", "side": "buy"}


def test_filter_handles_empty_input():
    assert filter_attributes({}) == {}


def test_redacted_keys_are_a_frozenset():
    """Pin: immutable so a runtime mutation can't add a non-
    redacted key by mistake. Operators add to the set via code +
    review, not at runtime."""
    assert isinstance(_REDACTED_ATTRIBUTE_KEYS, frozenset)


# ── span_from_event ──────────────────────────────────────


def _event(
    *,
    name: str = "broker.fetch_klines",
    started_at: datetime = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC),
    ended_at: datetime | None = None,
    elapsed_ms: float | None = None,
    error: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, str] | None = None,
) -> StageEvent:
    return StageEvent(
        name=name,
        started_at=started_at,
        ended_at=ended_at,
        elapsed_ms=elapsed_ms,
        error=error,
        kind=kind,
        attributes=attributes or {},
    )


def test_span_from_event_completed_returns_ok_status():
    end = datetime(2026, 5, 1, 10, 0, 1, tzinfo=UTC)
    span = span_from_event(
        _event(ended_at=end),
        trace_id=new_trace_id(),
    )
    assert span.status_code == SpanStatusCode.OK
    assert span.status_message == ""


def test_span_from_event_errored_returns_error_status_with_message():
    end = datetime(2026, 5, 1, 10, 0, 1, tzinfo=UTC)
    span = span_from_event(
        _event(ended_at=end, error="ConnectionError: timeout"),
        trace_id=new_trace_id(),
    )
    assert span.status_code == SpanStatusCode.ERROR
    assert "ConnectionError" in span.status_message


def test_span_from_event_open_stage_unset_status():
    """Pin: an open stage (no ended_at) gets UNSET status — the
    OTLP perspective is "in progress", not "errored". Operator's
    APM shows the span as in-flight."""
    span = span_from_event(_event(ended_at=None), trace_id=new_trace_id())
    assert span.status_code == SpanStatusCode.UNSET
    assert "progress" in span.status_message.lower()


def test_span_from_event_open_stage_zero_duration():
    """Pin: open stage gets end_nanos == start_nanos (zero duration)
    rather than missing end_nanos — pin so the OTLP encoder
    produces a valid span."""
    span = span_from_event(_event(ended_at=None), trace_id=new_trace_id())
    assert span.start_unix_nano == span.end_unix_nano


def test_span_from_event_passes_through_kind():
    span = span_from_event(
        _event(kind=SpanKind.CLIENT, ended_at=datetime(2026, 5, 1, 10, 0, 1, tzinfo=UTC)),
        trace_id=new_trace_id(),
    )
    assert span.kind == SpanKind.CLIENT


def test_span_from_event_strips_redacted_attributes():
    """Pin: the redacted-key filter runs at translation time so
    operator IP can never leak from a stage event."""
    span = span_from_event(
        _event(
            ended_at=datetime(2026, 5, 1, 10, 0, 1, tzinfo=UTC),
            attributes={"prompt": "secret", "pair": "BTCUSDT"},
        ),
        trace_id=new_trace_id(),
    )
    assert "prompt" not in span.attributes
    assert span.attributes["pair"] == "BTCUSDT"


def test_span_from_event_parent_span_id_set_for_children():
    parent = "f" * 16
    span = span_from_event(
        _event(ended_at=datetime(2026, 5, 1, 10, 0, 1, tzinfo=UTC)),
        trace_id=new_trace_id(),
        parent_span_id=parent,
    )
    assert span.parent_span_id == parent


# ── to_otlp_json ─────────────────────────────────────────


def test_otlp_json_uses_camel_case_field_names():
    """Pin: OTLP/HTTP wire format uses camelCase. Mismatched
    field names get silently dropped by the collector."""
    span = _good_span()
    payload = span.to_otlp_json()
    assert "traceId" in payload
    assert "spanId" in payload
    assert "parentSpanId" in payload
    assert "startTimeUnixNano" in payload
    assert "endTimeUnixNano" in payload


def test_otlp_json_times_are_strings():
    """Pin: OTLP wire format requires nano timestamps as strings
    (some JSON parsers can't handle int64 precision otherwise)."""
    span = _good_span()
    payload = span.to_otlp_json()
    assert isinstance(payload["startTimeUnixNano"], str)
    assert isinstance(payload["endTimeUnixNano"], str)


def test_otlp_json_attributes_are_list_of_records():
    """Pin: attributes are NOT a flat dict; they're a list of
    `{key, value: {stringValue: …}}` records per the OTLP spec."""
    span = _good_span(attributes={"pair": "BTCUSDT", "side": "buy"})
    payload = span.to_otlp_json()
    assert isinstance(payload["attributes"], list)
    assert len(payload["attributes"]) == 2
    keys = {a["key"] for a in payload["attributes"]}
    assert keys == {"pair", "side"}
    for a in payload["attributes"]:
        assert "stringValue" in a["value"]


def test_otlp_json_attributes_sorted_for_deterministic_output():
    """Pin: sorted by key — ensures regression-test fixtures
    don't flake on dict-iteration order."""
    span = _good_span(attributes={"z": "z-val", "a": "a-val"})
    keys = [a["key"] for a in span.to_otlp_json()["attributes"]]
    assert keys == ["a", "z"]


def test_otlp_json_status_includes_code_and_message():
    span = _good_span(
        status_code=SpanStatusCode.ERROR,
        status_message="stage failed",
    )
    payload = span.to_otlp_json()
    assert payload["status"]["code"] == "ERROR"
    assert payload["status"]["message"] == "stage failed"


# ── build_trace ──────────────────────────────────────────


def _started_at(s: int) -> datetime:
    return datetime(2026, 5, 1, 10, 0, s, tzinfo=UTC)


def test_build_trace_creates_root_plus_per_stage_spans():
    events = [
        _event(name="a", started_at=_started_at(0), ended_at=_started_at(1)),
        _event(name="b", started_at=_started_at(1), ended_at=_started_at(2)),
    ]
    bundle = build_trace(
        cycle_started_at=_started_at(0),
        cycle_ended_at=_started_at(2),
        stage_events=events,
    )
    assert bundle.root_span.name == "cycle"
    assert len(bundle.stage_spans) == 2
    assert {s.name for s in bundle.stage_spans} == {"a", "b"}


def test_build_trace_stage_spans_share_trace_id_with_root():
    events = [_event(ended_at=_started_at(1))]
    bundle = build_trace(
        cycle_started_at=_started_at(0),
        cycle_ended_at=_started_at(2),
        stage_events=events,
    )
    assert bundle.stage_spans[0].trace_id == bundle.root_span.trace_id == bundle.trace_id


def test_build_trace_stage_spans_parent_to_root():
    """Pin: the cycle is the root span; every stage is a child.
    Future nested stages just chain another parent_span_id."""
    events = [_event(ended_at=_started_at(1))]
    bundle = build_trace(
        cycle_started_at=_started_at(0),
        cycle_ended_at=_started_at(2),
        stage_events=events,
    )
    assert bundle.stage_spans[0].parent_span_id == bundle.root_span.span_id


def test_build_trace_uses_supplied_trace_id():
    """Operators may correlate by passing in their own trace
    ID (e.g. tying to an inbound request)."""
    custom = "a" * 32
    bundle = build_trace(
        cycle_started_at=_started_at(0),
        cycle_ended_at=_started_at(1),
        stage_events=[],
        trace_id=custom,
    )
    assert bundle.trace_id == custom
    assert bundle.root_span.trace_id == custom


def test_build_trace_handles_incomplete_cycle_with_completed_stages():
    """Pin: when the cycle didn't complete (cycle_ended_at=None)
    but stages did, root span uses the latest stage END as its
    end. Pin so the trace is still valid (non-zero duration)."""
    events = [
        _event(name="a", started_at=_started_at(0), ended_at=_started_at(2)),
    ]
    bundle = build_trace(
        cycle_started_at=_started_at(0),
        cycle_ended_at=None,
        stage_events=events,
    )
    assert bundle.root_span.end_unix_nano > bundle.root_span.start_unix_nano
    assert bundle.root_span.status_code == SpanStatusCode.UNSET


def test_build_trace_handles_incomplete_cycle_with_no_completed_stages():
    """Worst case: cycle didn't complete and no stage finished.
    Root span gets zero duration but valid IDs."""
    bundle = build_trace(
        cycle_started_at=_started_at(0),
        cycle_ended_at=None,
        stage_events=[],
    )
    assert bundle.root_span.start_unix_nano == bundle.root_span.end_unix_nano
    assert len(bundle.stage_spans) == 0


def test_build_trace_complete_cycle_has_ok_root_status():
    bundle = build_trace(
        cycle_started_at=_started_at(0),
        cycle_ended_at=_started_at(1),
        stage_events=[],
    )
    assert bundle.root_span.status_code == SpanStatusCode.OK


def test_build_trace_passes_cycle_attributes_through_filter():
    bundle = build_trace(
        cycle_started_at=_started_at(0),
        cycle_ended_at=_started_at(1),
        stage_events=[],
        cycle_attributes={"prompt": "SECRET", "cycle_id": "abc"},
    )
    assert "prompt" not in bundle.root_span.attributes
    assert bundle.root_span.attributes["cycle_id"] == "abc"


# ── TraceBundle helpers ──────────────────────────────────


def test_all_spans_includes_root_first():
    bundle = build_trace(
        cycle_started_at=_started_at(0),
        cycle_ended_at=_started_at(1),
        stage_events=[_event(name="a", started_at=_started_at(0), ended_at=_started_at(1))],
    )
    spans = bundle.all_spans()
    assert spans[0].name == "cycle"
    assert spans[1].name == "a"


# ── OTLP/HTTP wrapper ────────────────────────────────────


def test_otlp_payload_has_resource_spans_wrapper():
    """Pin: collector requires `resourceSpans[0].scopeSpans[0]`
    nesting; payloads without it are rejected."""
    bundle = build_trace(
        cycle_started_at=_started_at(0),
        cycle_ended_at=_started_at(1),
        stage_events=[],
    )
    payload = bundle.to_otlp_payload()
    assert "resourceSpans" in payload
    assert len(payload["resourceSpans"]) == 1
    assert "scopeSpans" in payload["resourceSpans"][0]
    assert "spans" in payload["resourceSpans"][0]["scopeSpans"][0]


def test_otlp_payload_includes_service_name_attribute():
    bundle = build_trace(
        cycle_started_at=_started_at(0),
        cycle_ended_at=_started_at(1),
        stage_events=[],
    )
    payload = bundle.to_otlp_payload(service_name="halal-trader-prod")
    resource_attrs = payload["resourceSpans"][0]["resource"]["attributes"]
    service_name_attr = next(a for a in resource_attrs if a["key"] == "service.name")
    assert service_name_attr["value"]["stringValue"] == "halal-trader-prod"


def test_otlp_payload_default_service_name_is_halal_trader():
    bundle = build_trace(
        cycle_started_at=_started_at(0),
        cycle_ended_at=_started_at(1),
        stage_events=[],
    )
    payload = bundle.to_otlp_payload()
    resource_attrs = payload["resourceSpans"][0]["resource"]["attributes"]
    service_name_attr = next(a for a in resource_attrs if a["key"] == "service.name")
    assert service_name_attr["value"]["stringValue"] == "halal-trader"


def test_to_json_round_trips_via_json_loads():
    """Pin: the JSON output is valid JSON the operator can ship
    via httpx / requests / urllib without further transformation."""
    bundle = build_trace(
        cycle_started_at=_started_at(0),
        cycle_ended_at=_started_at(1),
        stage_events=[_event(name="a", started_at=_started_at(0), ended_at=_started_at(1))],
    )
    text = bundle.to_json()
    decoded = json.loads(text)
    assert "resourceSpans" in decoded


# ── output structure ─────────────────────────────────────


def test_span_is_immutable():
    s = _good_span()
    with pytest.raises(Exception):
        s.name = "tampered"  # type: ignore[misc]


def test_trace_bundle_is_immutable():
    bundle = build_trace(
        cycle_started_at=_started_at(0),
        cycle_ended_at=_started_at(1),
        stage_events=[],
    )
    assert isinstance(bundle, TraceBundle)
    with pytest.raises(Exception):
        bundle.trace_id = "x"  # type: ignore[misc]


def test_stage_event_is_immutable():
    e = _event()
    with pytest.raises(Exception):
        e.name = "x"  # type: ignore[misc]


# ── timing precision ─────────────────────────────────────


def test_unix_nano_conversion_preserves_microsecond_precision():
    """Pin: Python's datetime is microsecond-precise; the
    conversion to nanos multiplies by 1000 (cycles measured in
    microseconds shouldn't lose precision)."""
    started = datetime(2026, 5, 1, 10, 0, 0, microsecond=500_000, tzinfo=UTC)
    ended = started + timedelta(milliseconds=42)
    event = _event(started_at=started, ended_at=ended)
    span = span_from_event(event, trace_id=new_trace_id())
    # 42ms = 42_000_000 nanos
    assert span.duration_nanos == 42_000_000
