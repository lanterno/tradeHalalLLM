"""Tests for :class:`BaseTradingBot._prune_audit_log` retention logic.

The shared bot lifecycle (init / run_once / shutdown) is exercised by
the per-bot integration paths. The one piece worth direct coverage is
the audit-log prune called from each bot's `_daily_end` — operators
disable it by setting `WEB_AUDIT_RETENTION_DAYS=0`, and a runaway
prune on a long-lived deployment shouldn't crash the bot.
"""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.scheduler import BaseTradingBot


class _StubBot(BaseTradingBot):
    """Minimal bot that bypasses the real init path."""

    async def _create_components(self) -> None:
        return

    async def _daily_start(self) -> None:
        return

    async def _daily_end(self) -> None:
        return

    def _get_cycle_service(self):
        return MagicMock()

    async def run(self) -> None:
        return


def _bot(*, retention: int = 7, bundle: object | None = None) -> _StubBot:
    """Build a stub bot with audit retention overridden."""
    bot = _StubBot.__new__(_StubBot)
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
    bot = _bot()
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
    bot = _bot()
    bot._engine = None
    await bot.shutdown()  # must not raise
    assert bot._engine is None
