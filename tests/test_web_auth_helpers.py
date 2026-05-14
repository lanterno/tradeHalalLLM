"""Tests for the pure helpers in :mod:`web.middleware.auth`.

The full middleware path is exercised in `test_web_auth_middleware.py`.
This file pins the synchronous helpers underneath: `_is_mutation_request`
(deciding what's gated) and `_constant_time_eq` (the timing-safe
token compare).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from halal_trader.web.middleware.auth import (
    _constant_time_eq,
    _is_mutation_request,
)


def _req(method: str, path: str) -> MagicMock:
    request = MagicMock()
    request.method = method
    request.url.path = path
    return request


# ── _is_mutation_request ─────────────────────────────────────


def test_admin_prefix_always_gated_regardless_of_method():
    """Even GET on `/api/admin/*` is treated as a mutation — admin
    routes are mutation-only-by-design, defensive against accidentally
    exposing a read endpoint there."""
    assert _is_mutation_request(_req("GET", "/api/admin/halt")) is True
    assert _is_mutation_request(_req("HEAD", "/api/admin/halt")) is True


def test_post_on_admin_gated():
    assert _is_mutation_request(_req("POST", "/api/admin/halt")) is True


def test_put_patch_delete_under_api_gated():
    for method in ("PUT", "PATCH", "DELETE"):
        assert _is_mutation_request(_req(method, "/api/positions/AAPL")) is True


def test_get_under_api_not_gated():
    """Read endpoints stay open."""
    assert _is_mutation_request(_req("GET", "/api/positions")) is False


def test_post_outside_api_not_gated():
    """Static asset routes / health / metrics are out of scope."""
    assert _is_mutation_request(_req("POST", "/health")) is False
    assert _is_mutation_request(_req("POST", "/static/upload")) is False


def test_method_normalised_to_uppercase():
    """A weirdly-cased method string still gets recognised as mutation."""
    assert _is_mutation_request(_req("post", "/api/positions")) is True


# ── _constant_time_eq ────────────────────────────────────────


def test_constant_time_eq_true_for_match():
    assert _constant_time_eq("secret-token", "secret-token") is True


def test_constant_time_eq_false_for_mismatch():
    assert _constant_time_eq("secret-token", "wrong-token!") is False


def test_constant_time_eq_false_on_length_mismatch():
    """Different-length strings → False (and constant time, per the comment)."""
    assert _constant_time_eq("a", "ab") is False
    assert _constant_time_eq("ab", "a") is False


def test_constant_time_eq_handles_empty_strings():
    assert _constant_time_eq("", "") is True
    assert _constant_time_eq("", "x") is False
    assert _constant_time_eq("x", "") is False
