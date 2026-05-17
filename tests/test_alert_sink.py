"""Tests for AlertSink — rate-limited Telegram error alerter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.notifications.telegram import AlertSink, TelegramNotifier


def _enabled_notifier() -> MagicMock:
    notifier = MagicMock(spec=TelegramNotifier)
    notifier.enabled = True
    notifier.notify_error = AsyncMock()
    return notifier


@pytest.mark.asyncio
async def test_disabled_notifier_is_noop():
    sink = AlertSink(notifier=None)
    assert not sink.enabled
    sent = await sink.notify("anything", "details")
    assert sent is False


@pytest.mark.asyncio
async def test_disabled_notifier_when_credentials_missing():
    notifier = MagicMock(spec=TelegramNotifier)
    notifier.enabled = False
    sink = AlertSink(notifier=notifier)
    assert not sink.enabled
    assert await sink.notify("e", "d") is False
    notifier.notify_error.assert_not_called()


@pytest.mark.asyncio
async def test_first_alert_sends():
    notifier = _enabled_notifier()
    sink = AlertSink(notifier=notifier)
    sent = await sink.notify("cycle.failed", "boom")
    assert sent is True
    # AlertSink forwards optional market + severity kwargs to the
    # underlying notifier (defaults: "" / "warning").
    notifier.notify_error.assert_awaited_once_with(
        "cycle.failed", "boom", market="", severity="warning"
    )


@pytest.mark.asyncio
async def test_repeated_alert_within_window_suppressed():
    notifier = _enabled_notifier()
    sink = AlertSink(notifier=notifier, cooldown_seconds=900)
    await sink.notify("cycle.failed", "boom 1")
    await sink.notify("cycle.failed", "boom 2")
    await sink.notify("cycle.failed", "boom 3")
    assert notifier.notify_error.await_count == 1


@pytest.mark.asyncio
async def test_distinct_error_types_independent():
    notifier = _enabled_notifier()
    sink = AlertSink(notifier=notifier)
    await sink.notify("cycle.failed", "a")
    await sink.notify("llm.chain.backoff", "b")
    assert notifier.notify_error.await_count == 2


@pytest.mark.asyncio
async def test_after_window_alert_includes_suppressed_count(monkeypatch):
    notifier = _enabled_notifier()
    sink = AlertSink(notifier=notifier, cooldown_seconds=900)

    fake_time = [1000.0]

    def now() -> float:
        return fake_time[0]

    monkeypatch.setattr("halal_trader.notifications.telegram.time.monotonic", now)

    await sink.notify("cycle.failed", "first")
    fake_time[0] += 60
    await sink.notify("cycle.failed", "second-suppressed")
    fake_time[0] += 60
    await sink.notify("cycle.failed", "third-suppressed")

    fake_time[0] += 1000  # past cooldown
    await sink.notify("cycle.failed", "fourth")

    assert notifier.notify_error.await_count == 2
    last_call_details = notifier.notify_error.await_args_list[-1].args[1]
    assert "suppressed 2" in last_call_details
