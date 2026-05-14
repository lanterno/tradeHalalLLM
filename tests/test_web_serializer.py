"""Tests for the recursive JSON-friendly serializer.

Used by every route module to walk dicts/lists and turn datetimes
into ISO strings before FastAPI serialises them. A bug here would
hit every API endpoint at once.
"""

from __future__ import annotations

from datetime import UTC, datetime

from halal_trader.web._serializer import serialize


def test_passes_through_primitives():
    assert serialize(42) == 42
    assert serialize("hello") == "hello"
    assert serialize(3.14) == 3.14
    assert serialize(True) is True
    assert serialize(None) is None


def test_converts_datetime_to_iso_string():
    dt = datetime(2026, 5, 1, 14, 30, tzinfo=UTC)
    out = serialize(dt)
    assert isinstance(out, str)
    assert out == dt.isoformat()


def test_walks_into_dict_values():
    dt = datetime(2026, 5, 1, tzinfo=UTC)
    out = serialize({"ts": dt, "n": 1})
    assert isinstance(out["ts"], str)
    assert out["n"] == 1


def test_walks_into_list_items():
    dt = datetime(2026, 5, 1, tzinfo=UTC)
    out = serialize([dt, 1, "x"])
    assert isinstance(out[0], str)
    assert out[1] == 1
    assert out[2] == "x"


def test_walks_into_nested_structure():
    """Mixed nesting: dict → list → dict → datetime."""
    dt = datetime(2026, 5, 1, tzinfo=UTC)
    payload = {"items": [{"ts": dt, "n": 5}]}
    out = serialize(payload)
    assert isinstance(out["items"][0]["ts"], str)
    assert out["items"][0]["n"] == 5


def test_dict_keys_are_passed_through_unchanged():
    """Only values are serialised — int / non-string keys stay as-is."""
    out = serialize({1: "a", "b": 2})
    assert out == {1: "a", "b": 2}


def test_empty_collections_round_trip():
    assert serialize([]) == []
    assert serialize({}) == {}


def test_naive_datetime_iso_format():
    """Naive datetimes (no tzinfo) still serialise — isoformat works
    for both."""
    dt = datetime(2026, 5, 1, 12, 0, 0)
    out = serialize(dt)
    assert "2026-05-01T12:00:00" == out


def test_does_not_mutate_input_dict():
    dt = datetime(2026, 5, 1, tzinfo=UTC)
    src = {"ts": dt}
    serialize(src)
    # Original unchanged.
    assert isinstance(src["ts"], datetime)


def test_does_not_mutate_input_list():
    dt = datetime(2026, 5, 1, tzinfo=UTC)
    src = [dt]
    serialize(src)
    assert isinstance(src[0], datetime)
