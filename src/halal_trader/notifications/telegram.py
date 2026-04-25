"""Telegram bot notifications — trade alerts, daily summaries, and system errors."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Sends trading alerts to a Telegram chat."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._client: httpx.AsyncClient | None = None

    _PLACEHOLDER_VALUES = frozenset({"your_bot_token", "your_chat_id", ""})

    @property
    def enabled(self) -> bool:
        return bool(
            self._bot_token
            and self._chat_id
            and self._bot_token not in self._PLACEHOLDER_VALUES
            and self._chat_id not in self._PLACEHOLDER_VALUES
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        """Close the persistent HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def send(self, message: str, *, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured Telegram chat."""
        if not self.enabled:
            return False

        url = _API_BASE.format(token=self._bot_token)
        try:
            client = self._get_client()
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
        msg = f"{emoji} <b>{side.upper()}</b> {pair}\nQty: {quantity:.6f}\nPrice: ${price:,.2f}\n"
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
        pnl = stats.get("realized_pnl", stats.get("total_pnl", 0))
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
        msg = f"\u26a0\ufe0f <b>System Alert: {error_type}</b>\n{details[:500]}\n"
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


class AlertSink:
    """Rate-limited error alerter on top of TelegramNotifier.

    Buckets alerts by ``error_type`` so a flapping error doesn't burn the
    Telegram API quota or spam the operator. The window is ``cooldown_seconds``
    long; the first alert in a window goes through, subsequent ones are
    counted and dropped (with a debug log).

    A no-op `AlertSink` is returned when the underlying notifier is
    disabled (no bot token / chat id), so call sites can stay branchless:

        sink = AlertSink(notifier)
        await sink.notify("cycle.failed", str(exc))   # safe even if disabled
    """

    DEFAULT_COOLDOWN_SECONDS = 900  # 15 minutes

    def __init__(
        self,
        notifier: TelegramNotifier | None,
        *,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        self._notifier = notifier
        self._cooldown = cooldown_seconds
        # NOTE: single-asyncio-loop only \u2014 not thread-safe.
        # ``last_sent`` uses ``None`` as the never-sent sentinel; ``0.0``
        # would falsely engage the cooldown gate on fresh systems where
        # ``time.monotonic()`` starts below the cooldown window.
        self._last_sent: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self._notifier is not None and self._notifier.enabled

    async def notify(self, error_type: str, details: str) -> bool:
        """Send a Telegram alert, deduped by ``error_type`` within the window.

        Returns ``True`` if a message went out, ``False`` if it was suppressed
        (or the notifier is disabled).
        """
        if not self.enabled:
            return False

        now = time.monotonic()
        last = self._last_sent.get(error_type)
        if last is not None and now - last < self._cooldown:
            self._suppressed[error_type] = self._suppressed.get(error_type, 0) + 1
            logger.debug(
                "Alert suppressed for %s (%ds since last; %d total in window)",
                error_type,
                int(now - last),
                self._suppressed[error_type],
            )
            return False

        suppressed = self._suppressed.pop(error_type, 0)
        if suppressed:
            details = f"{details}\n\n(also suppressed {suppressed} similar alerts)"

        assert self._notifier is not None
        await self._notifier.notify_error(error_type, details)
        self._last_sent[error_type] = now
        return True
