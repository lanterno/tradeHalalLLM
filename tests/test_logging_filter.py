"""Tests for the logging filters in :mod:`logging`.

`ThirdPartyConsoleFilter` is the gate keeping INFO chatter from noisy
third-party libraries off the operator's terminal. A bug here means
the operator either misses real warnings (over-filtering) or drowns
in INFO noise (under-filtering).
"""

from __future__ import annotations

import logging

from halal_trader.logging import ThirdPartyConsoleFilter


def _record(name: str, level: int) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname="x.py",
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )


def _filt() -> ThirdPartyConsoleFilter:
    return ThirdPartyConsoleFilter()


# ── Allow paths ──────────────────────────────────────────────


def test_allows_warning_from_any_logger():
    """A WARNING-level message from `apscheduler` (noisy) should pass —
    operators want to see scheduler warnings."""
    assert _filt().filter(_record("apscheduler", logging.WARNING)) is True


def test_allows_error_from_noisy_logger():
    assert _filt().filter(_record("httpx", logging.ERROR)) is True


def test_allows_critical_from_noisy_logger():
    assert _filt().filter(_record("mcp", logging.CRITICAL)) is True


def test_allows_info_from_application_logger():
    """Application loggers (`halal_trader.*`) keep their INFO messages."""
    assert _filt().filter(_record("halal_trader.trading.cycle", logging.INFO)) is True


def test_allows_debug_from_application_logger():
    assert _filt().filter(_record("halal_trader.trading.cycle", logging.DEBUG)) is True


# ── Block paths ──────────────────────────────────────────────


def test_blocks_info_from_apscheduler():
    """The point of the filter — keep schedule chatter off the console."""
    assert _filt().filter(_record("apscheduler", logging.INFO)) is False


def test_blocks_info_from_httpcore():
    assert _filt().filter(_record("httpcore.connection", logging.INFO)) is False


def test_blocks_info_from_asyncio():
    assert _filt().filter(_record("asyncio", logging.INFO)) is False


def test_blocks_debug_from_noisy_logger():
    """DEBUG is even chattier than INFO — also blocked."""
    assert _filt().filter(_record("httpx", logging.DEBUG)) is False


# ── Submodule routing ───────────────────────────────────────


def test_blocks_nested_noisy_logger():
    """`httpx.client` matches `httpx` at the top — also blocked."""
    assert _filt().filter(_record("httpx.client._main", logging.INFO)) is False


def test_only_top_level_name_is_consulted():
    """A logger like `mybot.apscheduler` should NOT match — only the
    leading segment is checked."""
    assert _filt().filter(_record("mybot.apscheduler", logging.INFO)) is True


def _access(path: str, status: int) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="x.py",
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:5000", "GET", path, "1.1", status),
        exc_info=None,
    )


def test_health_probe_access_lines_are_dropped_unless_they_fail() -> None:
    from halal_trader.logging import HealthAccessFilter

    f = HealthAccessFilter()
    assert not f.filter(_access("/api/health", 200))
    assert not f.filter(_access("/api/health/bot", 200))
    assert f.filter(_access("/api/health/bot", 503))  # a failing probe is news
    assert f.filter(_access("/api/home", 200))
    assert f.filter(_record("uvicorn.access", logging.INFO))  # not an access tuple
