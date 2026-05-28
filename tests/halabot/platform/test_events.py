"""Event model — construction, tier classification, causal chaining."""

from __future__ import annotations

from datetime import UTC, datetime

from halabot.platform.clock import FakeClock
from halabot.platform.events import (
    CONTROL_EVENT_TYPES,
    EventType,
    new_event,
)


def _clock() -> FakeClock:
    return FakeClock(datetime(2026, 5, 28, 12, 0, tzinfo=UTC))


def test_new_event_stamps_id_and_ts_from_clock():
    c = _clock()
    e = new_event(c, EventType.OBSERVATION_NEWS, source="finnhub", asset="NVDA")
    assert e.ts == c.now()
    assert e.id is not None
    assert e.asset == "NVDA"
    assert e.source == "finnhub"
    assert e.schema_version == 1


def test_new_event_without_causation_starts_fresh_chain():
    e = new_event(_clock(), EventType.OBSERVATION_BAR, source="alpaca")
    assert e.causation_id is None
    assert e.correlation_id is not None  # a fresh chain id


def test_new_event_with_causation_inherits_correlation_and_links_causation():
    c = _clock()
    root = new_event(c, EventType.OBSERVATION_NEWS, source="finnhub", asset="NVDA")
    child = new_event(
        c, EventType.BELIEF_UPDATED, source="belief", asset="NVDA", causation=root
    )
    assert child.causation_id == root.id
    assert child.correlation_id == root.correlation_id  # same causal chain
    assert child.id != root.id


def test_control_events_are_flagged():
    c = _clock()
    halt = new_event(c, EventType.RISK_HALT, source="risk")
    fill = new_event(c, EventType.ORDER_FILLED, source="exec", asset="NVDA")
    beat = new_event(c, EventType.SYSTEM_HEARTBEAT, source="heartbeat")
    invalid = new_event(c, EventType.BELIEF_INVALIDATED, source="belief", asset="NVDA")
    for e in (halt, fill, beat, invalid):
        assert e.is_control, e.type


def test_durable_events_are_not_control():
    c = _clock()
    for t in (
        EventType.OBSERVATION_PRICE,
        EventType.OBSERVATION_NEWS,
        EventType.BELIEF_UPDATED,
        EventType.CONVICTION_SCORED,
        EventType.POLICY_TARGET_CHANGED,
    ):
        assert not new_event(c, t, source="x").is_control, t


def test_control_set_matches_is_control():
    # The frozenset and the property must agree.
    c = _clock()
    for t in EventType:
        e = new_event(c, t, source="x")
        assert e.is_control == (t in CONTROL_EVENT_TYPES)


def test_event_is_frozen():
    e = new_event(_clock(), EventType.OBSERVATION_BAR, source="alpaca")
    try:
        e.asset = "MSFT"  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("Event should be immutable")
