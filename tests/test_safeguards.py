"""Tests for live-mode safeguards (core/safeguards.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.config import (
    AlpacaSettings,
    BinanceSettings,
    CryptoSettings,
    LiveModeSettings,
    Settings,
    StockSettings,
)
from halal_trader.core import halt
from halal_trader.core.safeguards import (
    LiveModeChecker,
    LiveModeError,
    check_live_mode_token,
    expected_token,
    is_live_mode,
)
from halal_trader.notifications.telegram import AlertSink, TelegramNotifier


def _settings(
    *,
    binance_testnet: bool = True,
    alpaca_paper_trade: bool = True,
    confirmation: str = "",
    max_account_balance_usd: float = 500.0,
    max_single_order_usd: float = 100.0,
    max_daily_loss_pct: float = 0.02,
    crypto_max_position_pct: float = 0.20,
    crypto_daily_loss_limit: float = 0.02,
    stocks_max_position_pct: float = 0.20,
    stocks_daily_loss_limit: float = 0.02,
    **rest: Any,
) -> Settings:
    """Build a Settings tree with nested sub-models populated for safeguard tests."""
    return Settings(
        _env_file=None,
        binance=BinanceSettings(_env_file=None, testnet=binance_testnet),
        alpaca=AlpacaSettings(_env_file=None, paper_trade=alpaca_paper_trade),
        live_mode=LiveModeSettings(
            _env_file=None,
            confirmation=confirmation,
            max_account_balance_usd=max_account_balance_usd,
            max_single_order_usd=max_single_order_usd,
            max_daily_loss_pct=max_daily_loss_pct,
        ),
        crypto=CryptoSettings(
            _env_file=None,
            max_position_pct=crypto_max_position_pct,
            daily_loss_limit=crypto_daily_loss_limit,
        ),
        stocks=StockSettings(
            _env_file=None,
            max_position_pct=stocks_max_position_pct,
            daily_loss_limit=stocks_daily_loss_limit,
        ),
        **rest,
    )


def _alert_sink() -> tuple[AlertSink, MagicMock]:
    notifier = MagicMock(spec=TelegramNotifier)
    notifier.enabled = True
    notifier.notify_error = AsyncMock()
    return AlertSink(notifier=notifier), notifier


# ── Token gate ─────────────────────────────────────────────────


def test_expected_token_is_dated():
    today = datetime(2026, 4, 25, tzinfo=UTC)
    assert expected_token(today) == "I-UNDERSTAND-REAL-MONEY-2026-04-25"


def test_is_live_mode_true_when_testnet_off():
    s = _settings(binance_testnet=False, alpaca_paper_trade=True)
    assert is_live_mode(s, market="crypto") is True
    assert is_live_mode(s, market="stocks") is False


def test_check_live_mode_token_skips_when_paper():
    s = _settings(binance_testnet=True, alpaca_paper_trade=True)
    check_live_mode_token(s, market="crypto")  # no raise
    check_live_mode_token(s, market="stocks")


def test_check_live_mode_token_refuses_without_token():
    s = _settings(binance_testnet=False)
    with pytest.raises(LiveModeError, match="Set LIVE_MODE_CONFIRMATION="):
        check_live_mode_token(s, market="crypto")


def test_check_live_mode_token_accepts_today_token():
    today = datetime(2026, 4, 25, tzinfo=UTC)
    s = _settings(
        binance_testnet=False,
        confirmation="I-UNDERSTAND-REAL-MONEY-2026-04-25",
    )
    check_live_mode_token(s, market="crypto", now=today)  # no raise


def test_check_live_mode_token_rejects_yesterday(monkeypatch):
    today = datetime(2026, 4, 25, tzinfo=UTC)
    s = _settings(
        binance_testnet=False,
        confirmation="I-UNDERSTAND-REAL-MONEY-2026-04-24",
    )
    with pytest.raises(LiveModeError):
        check_live_mode_token(s, market="crypto", now=today)


# ── LiveModeChecker.assert_safe ───────────────────────────────


@pytest.mark.asyncio
async def test_checker_inactive_in_paper_mode(engine):
    s = _settings(binance_testnet=True, alpaca_paper_trade=True)
    chk = LiveModeChecker(settings=s, market="crypto")
    assert chk.active is False
    assert await chk.assert_safe(account_balance=10_000, engine=engine, alerts=None) is True


@pytest.mark.asyncio
async def test_checker_passes_within_limits(engine):
    s = _settings(
        binance_testnet=False,
        max_account_balance_usd=500.0,
        max_single_order_usd=100.0,
        crypto_max_position_pct=0.20,
        crypto_daily_loss_limit=0.02,
        max_daily_loss_pct=0.02,
    )
    chk = LiveModeChecker(settings=s, market="crypto")
    sink, notifier = _alert_sink()
    assert await chk.assert_safe(account_balance=200.0, engine=engine, alerts=sink) is True
    notifier.notify_error.assert_not_called()
    assert not chk.tripped


@pytest.mark.asyncio
async def test_checker_trips_on_balance_too_high(engine):
    s = _settings(
        binance_testnet=False,
        max_account_balance_usd=500.0,
        max_single_order_usd=100.0,
        crypto_max_position_pct=0.20,
    )
    chk = LiveModeChecker(settings=s, market="crypto")
    sink, notifier = _alert_sink()
    safe = await chk.assert_safe(account_balance=600.0, engine=engine, alerts=sink)
    assert safe is False
    assert chk.tripped
    assert await halt.is_halted(engine)
    notifier.notify_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_checker_trips_on_single_order_too_large(engine):
    s = _settings(
        binance_testnet=False,
        max_account_balance_usd=10_000.0,
        max_single_order_usd=100.0,
        crypto_max_position_pct=0.50,  # 0.50 * 500 = $250 > $100
    )
    chk = LiveModeChecker(settings=s, market="crypto")
    sink, _ = _alert_sink()
    safe = await chk.assert_safe(account_balance=500.0, engine=engine, alerts=sink)
    assert safe is False
    assert await halt.is_halted(engine)


@pytest.mark.asyncio
async def test_checker_trips_on_loose_loss_limit(engine):
    s = _settings(
        binance_testnet=False,
        max_account_balance_usd=10_000.0,
        max_single_order_usd=10_000.0,
        crypto_max_position_pct=0.001,
        crypto_daily_loss_limit=0.10,  # 10% > 2% live floor
        max_daily_loss_pct=0.02,
    )
    chk = LiveModeChecker(settings=s, market="crypto")
    sink, _ = _alert_sink()
    safe = await chk.assert_safe(account_balance=100.0, engine=engine, alerts=sink)
    assert safe is False


@pytest.mark.asyncio
async def test_checker_idempotent_after_trip(engine):
    s = _settings(
        binance_testnet=False,
        max_account_balance_usd=1.0,
        max_single_order_usd=1.0,
        crypto_max_position_pct=0.001,
    )
    chk = LiveModeChecker(settings=s, market="crypto")
    await chk.assert_safe(account_balance=1000.0, engine=engine, alerts=None)
    assert chk.tripped
    # Subsequent calls return False without re-engaging the kill-switch
    # (already engaged) — important so we don't spam Telegram.
    assert await chk.assert_safe(account_balance=10.0, engine=engine, alerts=None) is False
