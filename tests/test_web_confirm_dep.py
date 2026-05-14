"""Tests for :func:`require_confirmation` — the destructive-action gate.

Existing `test_web_confirm_activity.py` tests the audit-log side; this
file pins the dependency itself: header-gate semantics, normalisation,
and the ``WEB_REQUIRE_CONFIRMATION=false`` test bypass.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from halal_trader.web.middleware.confirm import require_confirmation


def _request(header: str | None = None) -> MagicMock:
    request = MagicMock()
    request.headers = {"X-Trader-Confirm": header} if header is not None else {}
    return request


def _settings(*, require: bool) -> MagicMock:
    settings = MagicMock()
    settings.web.require_confirmation = require
    return settings


# ── Bypass when disabled ─────────────────────────────────────


@patch("halal_trader.web.middleware.confirm.get_settings")
def test_returns_none_when_requirement_disabled(mock_get_settings):
    """`WEB_REQUIRE_CONFIRMATION=false` → never gates; returns None."""
    mock_get_settings.return_value = _settings(require=False)
    # Even with no header, this should not raise.
    require_confirmation(_request())


@patch("halal_trader.web.middleware.confirm.get_settings")
def test_returns_none_when_disabled_even_with_wrong_header(mock_get_settings):
    mock_get_settings.return_value = _settings(require=False)
    require_confirmation(_request("nope"))


# ── Enforcement when enabled ─────────────────────────────────


@patch("halal_trader.web.middleware.confirm.get_settings")
def test_returns_none_with_correct_header(mock_get_settings):
    mock_get_settings.return_value = _settings(require=True)
    require_confirmation(_request("true"))


@patch("halal_trader.web.middleware.confirm.get_settings")
def test_normalises_header_case(mock_get_settings):
    """Header value compares case-insensitively (`TRUE` / `True` / `true`)."""
    mock_get_settings.return_value = _settings(require=True)
    require_confirmation(_request("TRUE"))
    require_confirmation(_request("True"))


@patch("halal_trader.web.middleware.confirm.get_settings")
def test_normalises_whitespace(mock_get_settings):
    mock_get_settings.return_value = _settings(require=True)
    require_confirmation(_request("  true  "))


@patch("halal_trader.web.middleware.confirm.get_settings")
def test_raises_412_when_header_missing(mock_get_settings):
    mock_get_settings.return_value = _settings(require=True)
    with pytest.raises(HTTPException) as exc:
        require_confirmation(_request())
    assert exc.value.status_code == 412
    assert "X-Trader-Confirm" in exc.value.detail


@patch("halal_trader.web.middleware.confirm.get_settings")
def test_raises_412_when_header_wrong_value(mock_get_settings):
    mock_get_settings.return_value = _settings(require=True)
    with pytest.raises(HTTPException) as exc:
        require_confirmation(_request("yes"))
    assert exc.value.status_code == 412


@patch("halal_trader.web.middleware.confirm.get_settings")
def test_raises_412_when_header_empty_string(mock_get_settings):
    mock_get_settings.return_value = _settings(require=True)
    with pytest.raises(HTTPException) as exc:
        require_confirmation(_request(""))
    assert exc.value.status_code == 412


@patch("halal_trader.web.middleware.confirm.get_settings")
def test_raises_412_when_header_whitespace_only(mock_get_settings):
    """Strip-then-compare means `"   "` falls through to the empty-string
    branch and is rejected — operator can't accidentally bypass with whitespace."""
    mock_get_settings.return_value = _settings(require=True)
    with pytest.raises(HTTPException):
        require_confirmation(_request("   "))


@patch("halal_trader.web.middleware.confirm.get_settings")
def test_error_message_mentions_dashboard_modal(mock_get_settings):
    """Make the error actionable — the message points the operator to
    where the confirm modal lives."""
    mock_get_settings.return_value = _settings(require=True)
    with pytest.raises(HTTPException) as exc:
        require_confirmation(_request())
    assert "dashboard" in exc.value.detail.lower()
