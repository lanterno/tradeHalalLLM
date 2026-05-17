"""Tests for :class:`TelegramNotifier`'s message-formatting paths.

Each `notify_*` method does its own formatting before delegating to
``send``. We mock `send` so we can assert on the message payload
without any network. AlertSink rate-limiting is covered separately
in `test_alert_sink.py`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from halal_trader.notifications.telegram import TelegramNotifier


def _notifier() -> TelegramNotifier:
    """Build a notifier with placeholder creds so `send` is gated by
    the test's monkeypatched ``send`` rather than the `enabled` check."""
    return TelegramNotifier(bot_token="real-token", chat_id="123")


# ── enabled property ──────────────────────────────────────────


def test_enabled_true_when_real_creds():
    n = TelegramNotifier(bot_token="real-token", chat_id="42")
    assert n.enabled is True


def test_enabled_false_when_placeholder_token():
    n = TelegramNotifier(bot_token="your_bot_token", chat_id="42")
    assert n.enabled is False


def test_enabled_false_when_placeholder_chat():
    n = TelegramNotifier(bot_token="real", chat_id="your_chat_id")
    assert n.enabled is False


def test_enabled_false_when_empty():
    n = TelegramNotifier(bot_token="", chat_id="")
    assert n.enabled is False


# ── notify_trade ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_notify_trade_buy_uses_green_emoji():
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_trade(pair="BTCUSDT", side="buy", quantity=0.001, price=42_000.0)
        msg = mock_send.await_args.args[0]
    assert "BUY" in msg
    assert "BTCUSDT" in msg
    assert "42,000" in msg or "42000" in msg
    # green circle for buy
    assert "\U0001f7e2" in msg


@pytest.mark.asyncio
async def test_notify_trade_sell_uses_red_emoji():
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_trade(pair="ETHUSDT", side="sell", quantity=0.5, price=2500.0)
        msg = mock_send.await_args.args[0]
    assert "SELL" in msg
    # red circle for sell
    assert "\U0001f534" in msg


@pytest.mark.asyncio
async def test_notify_trade_truncates_long_reasoning():
    """Reasoning is capped at 200 chars to keep the message readable."""
    n = _notifier()
    long_reason = "x" * 500
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_trade(
            pair="BTCUSDT",
            side="buy",
            quantity=0.001,
            price=42_000.0,
            reasoning=long_reason,
        )
        msg = mock_send.await_args.args[0]
    # The reasoning section should be ≤ 200 chars of x's.
    assert "x" * 200 in msg
    assert "x" * 201 not in msg


# ── notify_sl_tp ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_notify_sl_tp_take_profit_uses_check_emoji():
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_sl_tp(
            pair="BTCUSDT",
            exit_reason="take_profit",
            entry_price=40_000.0,
            exit_price=42_000.0,
            pnl=200.0,
        )
        msg = mock_send.await_args.args[0]
    assert "Take-Profit" in msg
    assert "✅" in msg  # ✅
    assert "+$200" in msg or "$+200" in msg


@pytest.mark.asyncio
async def test_notify_sl_tp_stop_loss_uses_x_emoji():
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_sl_tp(
            pair="BTCUSDT",
            exit_reason="stop_loss",
            entry_price=40_000.0,
            exit_price=39_000.0,
            pnl=-100.0,
        )
        msg = mock_send.await_args.args[0]
    assert "Stop-Loss" in msg
    assert "❌" in msg  # ❌
    assert "-$100" in msg or "$-100" in msg


# ── notify_daily_summary ──────────────────────────────────────


@pytest.mark.asyncio
async def test_notify_daily_summary_renders_required_keys():
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_daily_summary({"realized_pnl": 250.0, "trades_count": 4, "win_rate": 0.75})
        msg = mock_send.await_args.args[0]
    assert "Daily Summary" in msg
    assert "+$250" in msg or "$+250" in msg
    assert "Trades: 4" in msg
    # Compact format: "Win: 75%" instead of "Win Rate: 75%".
    assert "Win: 75%" in msg


@pytest.mark.asyncio
async def test_notify_daily_summary_falls_back_to_total_pnl_key():
    """The crypto + stocks `record_day_end` summary uses `realized_pnl`,
    but older callers may pass `total_pnl` — accept both."""
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_daily_summary({"total_pnl": 50.0})
        msg = mock_send.await_args.args[0]
    assert "+$50" in msg or "$+50" in msg


@pytest.mark.asyncio
async def test_notify_daily_summary_omits_best_pair_when_missing():
    """The summary line for `best_pair` is only added when the key is
    present — keeps the message short on quiet days."""
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_daily_summary({"realized_pnl": 0, "trades_count": 0})
        msg = mock_send.await_args.args[0]
    assert "Best:" not in msg
    assert "Worst:" not in msg


# ── notify_buzz ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_notify_buzz_renders_direction_label():
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_buzz(pair="DOGEUSDT", buzz_score=4.2, sentiment=0.6)
        msg = mock_send.await_args.args[0]
    assert "DOGEUSDT" in msg
    assert "bullish" in msg
    # "4.2× normal" — multiplication sign is the polished unicode form.
    assert "4.2" in msg
    assert "normal" in msg


@pytest.mark.asyncio
async def test_notify_buzz_includes_market_prefix():
    """For consistency with notify_trade / notify_sl_tp / notify_error,
    a market= kwarg renders as the [market] prefix."""
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_buzz(pair="DOGEUSDT", buzz_score=3.5, sentiment=0.2, market="crypto")
        msg = mock_send.await_args.args[0]
    assert "[crypto]" in msg
    assert "High Buzz Alert" in msg


@pytest.mark.asyncio
async def test_notify_buzz_negative_sentiment_renders_bearish():
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_buzz(pair="DOGEUSDT", buzz_score=3.0, sentiment=-0.4)
        msg = mock_send.await_args.args[0]
    assert "bearish" in msg


# ── notify_error ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_notify_error_truncates_long_details():
    """Details over 500 chars get smart-truncated: head + ellipsis + tail
    so the actual error code at the end of long stack traces survives.
    Total payload still stays bounded (head 300 + ellipsis + tail 180)."""
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_error("crypto.cycle.failed", "x" * 1_500)
        msg = mock_send.await_args.args[0]
    # The full 1500-char details should NOT be present (truncation fired).
    assert "x" * 1500 not in msg
    # The truncation marker should be there.
    assert "(truncated)" in msg
    # Total length should be bounded.
    assert len(msg) < 700


# ── Improved-format tests (from Telegram messages refactor) ───


@pytest.mark.asyncio
async def test_notify_trade_includes_market_and_notional():
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_trade(
            pair="BTCUSDT",
            side="buy",
            quantity=0.05,
            price=50_000.0,
            market="crypto",
            order_id="1234567890",
        )
        msg = mock_send.await_args.args[0]
    assert "[crypto]" in msg
    assert "BUY" in msg
    # Notional = 0.05 × 50,000 = $2,500
    assert "$2,500" in msg
    # Order ID tail-truncated
    assert "67890" in msg


@pytest.mark.asyncio
async def test_notify_sl_tp_includes_pct_pnl_and_hold_time():
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_sl_tp(
            pair="BTCUSDT",
            exit_reason="stop_loss",
            entry_price=50_000.0,
            exit_price=49_000.0,
            pnl=-100.0,
            quantity=0.1,
            hold_minutes=42.0,
            market="crypto",
        )
        msg = mock_send.await_args.args[0]
    assert "[crypto]" in msg
    assert "Stop-Loss" in msg
    # Pct PnL = (49000-50000)/50000 = -2.00%
    assert "-2.00%" in msg
    assert "Held: 42m" in msg


@pytest.mark.asyncio
async def test_notify_sl_tp_renders_hold_in_hours_when_long():
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_sl_tp(
            pair="BTCUSDT",
            exit_reason="take_profit",
            entry_price=50_000.0,
            exit_price=51_000.0,
            pnl=100.0,
            hold_minutes=125.0,  # 2.1 hours
        )
        msg = mock_send.await_args.args[0]
    assert "Held: 2.1h" in msg


@pytest.mark.asyncio
async def test_notify_daily_summary_includes_llm_cost_when_provided():
    """The new compact summary surfaces ops/spend visibility next to PnL."""
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_daily_summary(
            {
                "realized_pnl": 100,
                "trades_count": 5,
                "win_rate": 0.6,
                "llm_cost_usd": 12.45,
                "llm_calls": 96,
                "cycles_count": 480,
                "market": "crypto",
            }
        )
        msg = mock_send.await_args.args[0]
    assert "crypto" in msg
    assert "$12.45" in msg
    assert "96 calls" in msg
    assert "Cycles: 480" in msg


@pytest.mark.asyncio
async def test_notify_error_critical_severity_uses_distinct_emoji():
    n = _notifier()
    with patch.object(n, "send", new=AsyncMock()) as mock_send:
        await n.notify_error(
            "llm.insufficient_quota",
            "Account out of credits",
            market="crypto",
            severity="critical",
        )
        msg = mock_send.await_args.args[0]
    # Severity label gates the visual treatment; we don't assert on the
    # exact emoji because terminal/encoding can split it into a surrogate
    # pair that doesn't compare equal to the literal codepoint.
    assert "CRITICAL" in msg
    assert "[crypto]" in msg
    assert "llm.insufficient_quota" in msg
