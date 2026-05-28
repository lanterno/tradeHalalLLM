"""Ingress payload validation (fail-closed, INV-4)."""

from __future__ import annotations

from datetime import UTC, datetime

from halabot.platform.clock import FakeClock
from halabot.platform.events import EventType, new_event
from halabot.schemas import validate_payload

CLOCK = FakeClock(datetime(2026, 5, 28, tzinfo=UTC))


def _ev(t: EventType, *, asset: str | None = "NVDA", payload: dict):
    return new_event(CLOCK, t, source="test", asset=asset, payload=payload)


def test_well_formed_bar_passes():
    e = _ev(EventType.OBSERVATION_BAR, payload={"o": 1, "h": 2, "low": 0.5, "c": 1.5})
    assert validate_payload(e) is None


def test_bar_missing_keys_reported():
    e = _ev(EventType.OBSERVATION_BAR, payload={"o": 1})
    err = validate_payload(e)
    assert err is not None and "missing keys" in err


def test_ingress_requires_asset():
    e = _ev(
        EventType.OBSERVATION_PRICE, asset=None, payload={"price": 1.0}
    )
    assert validate_payload(e) == "missing asset"


def test_news_requires_headline_and_url():
    assert validate_payload(_ev(EventType.OBSERVATION_NEWS, payload={"headline": "x"})) is not None
    assert (
        validate_payload(_ev(EventType.OBSERVATION_NEWS, payload={"headline": "x", "url": "u"}))
        is None
    )


def test_internal_event_types_pass_without_validation():
    # Trusted internal events carry arbitrary payloads — not key-checked.
    e = _ev(EventType.SYSTEM_HEARTBEAT, asset=None, payload={})
    assert validate_payload(e) is None
