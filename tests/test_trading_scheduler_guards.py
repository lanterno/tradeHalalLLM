"""Tests for `TradingBot._require_initialized`.

The stocks scheduler uses this guard at the top of every scheduled
job (`pre_market`, `trading_cycle`, `end_of_day`). A regression here
would either let a job run with a `None` component (silent NoneType
crash buried in the cycle) or raise a generic error that doesn't
tell the operator what to fix.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from halal_trader.trading.scheduler import TradingBot


def test_require_initialized_raises_when_nothing_set():
    """Fresh bot — none of the four components are set yet. Helper
    must raise with a list of all four names."""
    bot = TradingBot()
    with pytest.raises(RuntimeError) as exc_info:
        bot._require_initialized()
    msg = str(exc_info.value)
    assert "must be called before using" in msg
    assert "screener" in msg
    assert "executor" in msg
    assert "portfolio" in msg
    assert "cycle_service" in msg


def test_require_initialized_returns_tuple_when_all_set():
    """When all four are set, returns a tuple — the call sites
    destructure it via `(screener, executor, portfolio, cycle) = self._require_initialized()`."""
    bot = TradingBot()
    bot.screener = MagicMock()
    bot.executor = MagicMock()
    bot.portfolio = MagicMock()
    bot.cycle_service = MagicMock()

    out = bot._require_initialized()
    assert len(out) == 4
    # Order matters — pin so a refactor doesn't reorder it under callers.
    assert out[0] is bot.screener
    assert out[1] is bot.executor
    assert out[2] is bot.portfolio
    assert out[3] is bot.cycle_service


def test_require_initialized_only_screener_missing():
    """If only one component is missing, only that one is named in
    the error — operators get a precise diagnostic."""
    bot = TradingBot()
    bot.executor = MagicMock()
    bot.portfolio = MagicMock()
    bot.cycle_service = MagicMock()
    # screener stays None

    with pytest.raises(RuntimeError) as exc_info:
        bot._require_initialized()
    msg = str(exc_info.value)
    assert "screener" in msg
    assert "executor" not in msg
    assert "portfolio" not in msg
    assert "cycle_service" not in msg


def test_require_initialized_only_executor_missing():
    bot = TradingBot()
    bot.screener = MagicMock()
    bot.portfolio = MagicMock()
    bot.cycle_service = MagicMock()

    with pytest.raises(RuntimeError) as exc_info:
        bot._require_initialized()
    msg = str(exc_info.value)
    assert "executor" in msg
    assert "screener" not in msg


def test_require_initialized_only_portfolio_missing():
    bot = TradingBot()
    bot.screener = MagicMock()
    bot.executor = MagicMock()
    bot.cycle_service = MagicMock()

    with pytest.raises(RuntimeError) as exc_info:
        bot._require_initialized()
    msg = str(exc_info.value)
    assert "portfolio" in msg
    assert "executor" not in msg


def test_require_initialized_only_cycle_service_missing():
    """The most common forgot-to-init: cycle_service. Pin so a
    refactor that drops cycle_service from the check (or renames it)
    breaks here first."""
    bot = TradingBot()
    bot.screener = MagicMock()
    bot.executor = MagicMock()
    bot.portfolio = MagicMock()

    with pytest.raises(RuntimeError) as exc_info:
        bot._require_initialized()
    msg = str(exc_info.value)
    assert "cycle_service" in msg


def test_require_initialized_partial_init_lists_all_missing():
    """Two of four set → two names in the error message, in order."""
    bot = TradingBot()
    bot.screener = MagicMock()
    bot.executor = MagicMock()
    # portfolio + cycle_service still None

    with pytest.raises(RuntimeError) as exc_info:
        bot._require_initialized()
    msg = str(exc_info.value)
    assert "portfolio" in msg
    assert "cycle_service" in msg
    # The set components are NOT mentioned (cleaner diagnostic).
    assert "screener" not in msg
    assert "executor" not in msg


def test_require_initialized_message_starts_with_initialize_hint():
    """The error message must mention `initialize()` so operators
    know what to call. Pin the actionable instruction."""
    bot = TradingBot()
    with pytest.raises(RuntimeError, match="initialize"):
        bot._require_initialized()


def test_require_initialized_does_not_mutate_state():
    """The guard is read-only — calling it doesn't set any fields."""
    bot = TradingBot()
    with pytest.raises(RuntimeError):
        bot._require_initialized()
    # Components are still None (the guard didn't lazy-init them).
    assert bot.screener is None
    assert bot.executor is None
    assert bot.portfolio is None
    assert bot.cycle_service is None
