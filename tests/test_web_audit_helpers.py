"""Tests for the pure helpers in :mod:`web.audit`.

The full middleware path needs the Starlette test client + a DB
engine; this file pins the synchronous helpers underneath:
`_is_audit_request` (deciding what to log) and `_truncate_payload`
(keeping the payload column from blowing up the table).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from halal_trader.web.audit import _is_audit_request, _truncate_payload


def _req(method: str, path: str) -> MagicMock:
    request = MagicMock()
    request.method = method
    request.url.path = path
    return request


# ── _is_audit_request ───────────────────────────────────────


def test_is_audit_true_for_post_under_api():
    assert _is_audit_request(_req("POST", "/api/admin/halt")) is True


def test_is_audit_true_for_put_patch_delete():
    for method in ("PUT", "PATCH", "DELETE"):
        assert _is_audit_request(_req(method, "/api/admin/halt")) is True, method


def test_is_audit_false_for_get_under_api():
    """Reads aren't mutations — the audit table is mutation-only."""
    assert _is_audit_request(_req("GET", "/api/positions")) is False


def test_is_audit_false_for_post_outside_api():
    """Static asset uploads / health endpoints stay out of the audit table."""
    assert _is_audit_request(_req("POST", "/health")) is False


def test_is_audit_method_normalised_to_uppercase():
    """A weirdly-cased method string still gets recognised."""
    assert _is_audit_request(_req("post", "/api/admin/halt")) is True


# ── _truncate_payload ──────────────────────────────────────


def test_truncate_returns_none_for_empty_body():
    assert _truncate_payload(b"") is None


def test_truncate_returns_decoded_string_below_threshold():
    assert _truncate_payload(b'{"a":1}') == '{"a":1}'


def test_truncate_caps_at_4kb_with_ellipsis_marker():
    """Anything above 4000 bytes gets cut + a '…[truncated]' suffix
    so the operator can see the row was clipped."""
    body = b"x" * 5_000
    out = _truncate_payload(body)
    assert out is not None
    assert out.endswith("…[truncated]")
    # Body itself is exactly 4000 chars.
    assert out.count("x") == 4_000


def test_truncate_replaces_non_utf8_bytes():
    """A non-UTF-8 byte sequence shouldn't crash the audit write —
    the `errors='replace'` decode swaps in U+FFFD."""
    body = b"\xff\xfeabc"
    out = _truncate_payload(body)
    assert out is not None
    assert "abc" in out


def test_truncate_exact_4kb_no_ellipsis():
    """Boundary case: exactly the cap → no truncation marker."""
    body = b"x" * 4_000
    out = _truncate_payload(body)
    assert out is not None
    assert "[truncated]" not in out
