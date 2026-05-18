"""Tests for the pure helpers in :mod:`mcp.client`.

`AlpacaMCPClient` itself is integration (spawns the MCP server as a
subprocess), but `_flex_get` is a pure key-aliasing helper that
underpins every Alpaca response parser — it walks snake_case /
camelCase / version-specific key names and returns the first hit.
"""

from __future__ import annotations

from halal_trader.mcp.client import _flex_get


def test_returns_first_matching_key():
    """The first key in the lookup order wins, even if later keys exist."""
    d = {"buying_power": 1000, "buyingPower": 9999}
    assert _flex_get(d, "buying_power", "buyingPower") == 1000


def test_falls_through_to_camelcase_when_snake_case_missing():
    d = {"buyingPower": 1500}
    assert _flex_get(d, "buying_power", "buyingPower") == 1500


def test_falls_through_to_default_when_no_keys_match():
    d = {"unrelated": 1}
    assert _flex_get(d, "a", "b", "c", default="missing") == "missing"


def test_default_is_none_by_default():
    """Caller passing zero variants gets None (mirrors `dict.get`)."""
    d = {"x": 1}
    assert _flex_get(d, "missing") is None


def test_returns_zero_value_not_default():
    """A literal 0 / "" / False value must not silently fall through to
    default — that would mask explicit-zero responses."""
    d = {"cash": 0}
    assert _flex_get(d, "cash", default=999) == 0


def test_returns_falsy_string_not_default():
    d = {"status": ""}
    assert _flex_get(d, "status", default="UNKNOWN") == ""


def test_returns_false_not_default():
    d = {"is_open": False}
    assert _flex_get(d, "is_open", "isOpen", default=True) is False


def test_handles_empty_dict():
    assert _flex_get({}, "a", "b", default=42) == 42


def test_handles_no_keys_specified():
    """Defensive: zero key variants should return default."""
    assert _flex_get({"a": 1}, default=99) == 99
