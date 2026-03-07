"""Telegram bot notifications — trade alerts, daily summaries, and system errors."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Sends trading alerts to a Telegram chat."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id

    @property
    def enabled(self) -> bool:
        return bool(self._bot_token and self._chat_id)

    async def send(self, message: str, *, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured Telegram chat."""
        if not self.enabled:
            return False

        url = _API_BASE.format(token=self._bot_token)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    url,
                    json={
                        "chat_id": self._chat_id,
                        "text": message,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True,
                    },
                )
                if resp.status_code == 200:
                    return True
                logger.warning("Telegram API returned %d: %s", resp.status_code, resp.text)
                return False
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)
            return False

    async def notify_trade(
        self,
        pair: str,
        side: str,
        quantity: float,
        price: float,
        reasoning: str = "",
    ) -> None:
        """Send a trade execution alert."""
        emoji = "\U0001f7e2" if side.lower() == "buy" else "\U0001f534"
        msg = (
            f"{emoji} <b>{side.upper()}</b> {pair}\n"
            f"Qty: {quantity:.6f}\n"
            f"Price: ${price:,.2f}\n"
        )
        if reasoning:
            msg += f"Reason: {reasoning[:200]}\n"
        await self.send(msg)

    async def notify_sl_tp(
        self,
        pair: str,
        exit_reason: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
    ) -> None:
        """Send a stop-loss or take-profit alert."""
        emoji = "\u2705" if pnl > 0 else "\u274c"
        label = "Take-Profit" if "take_profit" in exit_reason else "Stop-Loss"
        msg = (
            f"{emoji} <b>{label}</b> {pair}\n"
            f"Entry: ${entry_price:,.2f} \u2192 Exit: ${exit_price:,.2f}\n"
            f"P&L: ${pnl:+,.2f}\n"
        )
        await self.send(msg)

    async def notify_daily_summary(self, stats: dict[str, Any]) -> None:
        """Send end-of-day performance summary."""
        pnl = stats.get("total_pnl", 0)
        emoji = "\U0001f4c8" if pnl >= 0 else "\U0001f4c9"
        msg = (
            f"{emoji} <b>Daily Summary</b>\n"
            f"P&L: ${pnl:+,.2f}\n"
            f"Trades: {stats.get('trades_count', 0)}\n"
            f"Win Rate: {stats.get('win_rate', 0):.0%}\n"
        )
        if stats.get("best_pair"):
            msg += f"Best: {stats['best_pair']} (${stats.get('best_pair_pnl', 0):+,.2f})\n"
        if stats.get("worst_pair"):
            msg += f"Worst: {stats['worst_pair']} (${stats.get('worst_pair_pnl', 0):+,.2f})\n"
        await self.send(msg)

    async def notify_error(self, error_type: str, details: str) -> None:
        """Send a system error alert."""
        msg = (
            f"\u26a0\ufe0f <b>System Alert: {error_type}</b>\n"
            f"{details[:500]}\n"
        )
        await self.send(msg)

    async def notify_buzz(self, pair: str, buzz_score: float, sentiment: float) -> None:
        """Send a high buzz alert for a pair."""
        direction = "bullish" if sentiment > 0 else ("bearish" if sentiment < 0 else "neutral")
        msg = (
            f"\U0001f525 <b>High Buzz Alert</b> {pair}\n"
            f"Buzz: {buzz_score:.1f}x normal\n"
            f"Sentiment: {sentiment:+.2f} ({direction})\n"
            f"Reddit is talking about this coin \u2014 check for opportunities."
        )
        await self.send(msg)
