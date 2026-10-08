"""Tests for :class:`TradingBot`'s audit-log prune and engine teardown.

The prune's retention logic: operators disable it by setting
`WEB_AUDIT_RETENTION_DAYS=0`, and a failing prune on a long-lived
deployment shouldn't crash the bot.
"""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.trading.scheduler import TradingBot


def _bot(*, retention: int = 7, bundle: object | None = None) -> TradingBot:
    """Build a bot that bypasses the real init path, audit retention overridden."""
    bot = TradingBot.__new__(TradingBot)
    bot._engine = None
    bot._running = False
    bot._repo = None
    bot._bundle = bundle  # type: ignore[assignment]
    web = MagicMock()
    web.audit_retention_days = retention
    settings = MagicMock()
    settings.web = web
    bot.settings = settings
    return bot


def _shutdown_ready(bot: TradingBot) -> TradingBot:
    """Stub every resource ``shutdown`` touches besides the engine."""
    bot.scheduler = MagicMock(running=False)
    bot._news_reactor_task = None
    bot._monitor_task = None
    bot._stocks_news = None
    bot._broker_client = MagicMock(disconnect=AsyncMock())
    bot._release_trading_lock = AsyncMock()  # type: ignore[method-assign]
    return bot


@pytest.mark.asyncio
async def test_prune_skipped_when_retention_zero():
    """Operators disable pruning by setting retention=0."""
    bundle = MagicMock()
    bundle.web_audit.delete_old_web_actions = AsyncMock(return_value=0)
    bot = _bot(retention=0, bundle=bundle)
    await bot._prune_audit_log()
    bundle.web_audit.delete_old_web_actions.assert_not_awaited()


@pytest.mark.asyncio
async def test_prune_skipped_when_no_bundle():
    """If init never ran, the bundle is None — must early-return cleanly."""
    bot = _bot(retention=30, bundle=None)
    await bot._prune_audit_log()  # must not raise


@pytest.mark.asyncio
async def test_prune_calls_bundle_with_correct_window():
    bundle = MagicMock()
    bundle.web_audit.delete_old_web_actions = AsyncMock(return_value=42)
    bot = _bot(retention=14, bundle=bundle)
    await bot._prune_audit_log()
    bundle.web_audit.delete_old_web_actions.assert_awaited_once_with(older_than=timedelta(days=14))


@pytest.mark.asyncio
async def test_prune_swallows_repo_exception():
    """A failed prune must not abort the daily-end routine."""
    bundle = MagicMock()
    bundle.web_audit.delete_old_web_actions = AsyncMock(side_effect=RuntimeError("DB locked"))
    bot = _bot(retention=7, bundle=bundle)
    await bot._prune_audit_log()  # must not raise


@pytest.mark.asyncio
async def test_prune_handles_negative_retention_as_disabled():
    """Negative values mean disabled (treated like zero) — defensive."""
    bundle = MagicMock()
    bundle.web_audit.delete_old_web_actions = AsyncMock()
    bot = _bot(retention=-1, bundle=bundle)
    await bot._prune_audit_log()
    bundle.web_audit.delete_old_web_actions.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_disposes_engine_and_clears():
    bot = _shutdown_ready(_bot())
    engine = MagicMock()
    engine.dispose = AsyncMock()
    bot._engine = engine
    bot._running = True
    await bot.shutdown()
    engine.dispose.assert_awaited_once()
    assert bot._engine is None
    assert bot._running is False


@pytest.mark.asyncio
async def test_shutdown_no_op_when_not_initialized():
    """`shutdown` is safe to call even if init never ran."""
    bot = _shutdown_ready(_bot())
    bot._engine = None
    await bot.shutdown()  # must not raise
    assert bot._engine is None
