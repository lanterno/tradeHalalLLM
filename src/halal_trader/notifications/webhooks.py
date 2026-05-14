"""Slack + Discord webhook notifiers.

Round-4 wave 5.G: gives operators a real notification choice beyond
Telegram. Both adapters speak the same JSON-over-HTTPS webhook
protocols (incoming-webhook on Slack, channel webhook on Discord).
No SDK or OAuth needed — operator pastes a webhook URL into env,
we POST to it.

Both notifiers mirror the public surface of `TelegramNotifier`:

* `enabled` property (False when the webhook URL is empty / a
  placeholder).
* `send(message)` — raw text post.
* `notify_trade`, `notify_sl_tp`, `notify_daily_summary`,
  `notify_buzz` — the same trade-life-cycle alerts the bot already
  emits for Telegram.

Why two adapters in one file: the surface is ~95% identical
(only the JSON payload shape differs); keeping them adjacent
makes the duplication visible and easy to DRY further later.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


_PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "your_slack_webhook_url",
        "your_discord_webhook_url",
        "https://hooks.slack.com/services/your-webhook",
        "https://discord.com/api/webhooks/your-webhook",
    }
)


def _is_placeholder(url: str) -> bool:
    """A URL counts as configured iff it's a real-looking https
    URL that isn't one of the known placeholder strings. Pin
    the check so a half-configured `.env` doesn't silently
    "succeed" by sending to an invalid endpoint."""
    if not url:
        return True
    if url in _PLACEHOLDER_VALUES:
        return True
    if not url.startswith("https://"):
        return True
    return False


# ── Slack ─────────────────────────────────────────────────


class SlackNotifier:
    """Slack incoming-webhook notifier.

    Setup: operator creates an Incoming Webhook in their Slack
    workspace and pastes the URL into ``SLACK_WEBHOOK_URL``.
    Optional ``SLACK_CHANNEL`` overrides the default channel the
    webhook was bound to (only honoured by Slack if the workspace
    permits it).
    """

    def __init__(self, webhook_url: str, *, channel: str = "") -> None:
        self._url = webhook_url
        self._channel = channel
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return not _is_placeholder(self._url)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send(self, message: str, *, blocks: list[dict[str, Any]] | None = None) -> bool:
        """Post ``message`` to the configured channel.

        ``blocks`` is Slack Block Kit; when provided, Slack uses it
        as the rich-formatted body and ``message`` as the
        notification preview text. Returns True iff the POST returned 2xx.
        """
        if not self.enabled:
            return False
        payload: dict[str, Any] = {"text": message}
        if self._channel:
            payload["channel"] = self._channel
        if blocks:
            payload["blocks"] = blocks
        try:
            client = self._get_client()
            resp = await client.post(self._url, json=payload)
            if 200 <= resp.status_code < 300:
                return True
            logger.warning("Slack webhook returned %d: %s", resp.status_code, resp.text[:200])
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("Slack webhook send failed: %s", exc)
            return False

    async def notify_trade(
        self,
        *,
        pair: str,
        side: str,
        quantity: float,
        price: float,
        reasoning: str = "",
    ) -> None:
        emoji = "🟢" if side.lower() == "buy" else "🔴"
        notional = quantity * price
        msg = (
            f"{emoji} *{side.upper()}* `{pair}` qty={quantity} @ ${price:,.2f} (≈${notional:,.2f})"
        )
        if reasoning:
            msg += f"\n>{reasoning[:200]}"
        await self.send(msg)

    async def notify_sl_tp(
        self,
        *,
        pair: str,
        exit_reason: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
    ) -> None:
        emoji = "✅" if pnl_pct >= 0 else "❌"
        msg = (
            f"{emoji} `{pair}` *{exit_reason.upper()}* — "
            f"entry ${entry_price:,.2f} → exit ${exit_price:,.2f} "
            f"({pnl_pct:+.2%})"
        )
        await self.send(msg)

    async def notify_daily_summary(self, stats: dict[str, Any]) -> None:
        pnl = stats.get("realized_pnl", stats.get("total_pnl", 0))
        ret = stats.get("return_pct", 0) or 0
        msg = (
            f"📊 *Daily summary*\n"
            f"P&L: {'+' if pnl >= 0 else ''}${pnl:,.2f} ({ret:+.2%})\n"
            f"Trades: {stats.get('trades_count', 0)}"
        )
        if stats.get("best_pair"):
            msg += f"\nBest: {stats['best_pair']}"
        if stats.get("worst_pair"):
            msg += f"\nWorst: {stats['worst_pair']}"
        await self.send(msg)

    async def notify_buzz(self, pair: str, buzz_score: float, sentiment: float) -> None:
        direction = "bullish" if sentiment > 0 else ("bearish" if sentiment < 0 else "neutral")
        msg = (
            f"🔥 *High Buzz Alert* `{pair}`\n"
            f"Buzz: {buzz_score:.1f}× normal · "
            f"Sentiment: {sentiment:+.2f} ({direction})"
        )
        await self.send(msg)


# ── Discord ───────────────────────────────────────────────


class DiscordNotifier:
    """Discord channel-webhook notifier.

    Setup: operator creates a Webhook on a channel and pastes the URL
    into ``DISCORD_WEBHOOK_URL``. Discord limits webhook messages to
    ~2000 chars; we truncate the reasoning fields so a verbose LLM
    explanation doesn't overflow.
    """

    _MAX_CONTENT_CHARS = 1900  # below Discord's 2000-char hard limit

    def __init__(self, webhook_url: str, *, username: str = "halal-trader") -> None:
        self._url = webhook_url
        self._username = username
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return not _is_placeholder(self._url)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send(self, message: str, *, embeds: list[dict[str, Any]] | None = None) -> bool:
        """Post a message to the configured Discord channel."""
        if not self.enabled:
            return False
        payload: dict[str, Any] = {
            "content": message[: self._MAX_CONTENT_CHARS],
            "username": self._username,
        }
        if embeds:
            payload["embeds"] = embeds
        try:
            client = self._get_client()
            resp = await client.post(self._url, json=payload)
            if 200 <= resp.status_code < 300:
                return True
            logger.warning("Discord webhook returned %d: %s", resp.status_code, resp.text[:200])
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("Discord webhook send failed: %s", exc)
            return False

    async def notify_trade(
        self,
        *,
        pair: str,
        side: str,
        quantity: float,
        price: float,
        reasoning: str = "",
    ) -> None:
        emoji = "🟢" if side.lower() == "buy" else "🔴"
        notional = quantity * price
        msg = (
            f"{emoji} **{side.upper()}** `{pair}` "
            f"qty={quantity} @ ${price:,.2f} (≈${notional:,.2f})"
        )
        if reasoning:
            msg += f"\n> {reasoning[:200]}"
        await self.send(msg)

    async def notify_sl_tp(
        self,
        *,
        pair: str,
        exit_reason: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
    ) -> None:
        emoji = "✅" if pnl_pct >= 0 else "❌"
        msg = (
            f"{emoji} `{pair}` **{exit_reason.upper()}** — "
            f"entry ${entry_price:,.2f} → exit ${exit_price:,.2f} "
            f"({pnl_pct:+.2%})"
        )
        await self.send(msg)

    async def notify_daily_summary(self, stats: dict[str, Any]) -> None:
        pnl = stats.get("realized_pnl", stats.get("total_pnl", 0))
        ret = stats.get("return_pct", 0) or 0
        msg = (
            f"📊 **Daily summary**\n"
            f"P&L: {'+' if pnl >= 0 else ''}${pnl:,.2f} ({ret:+.2%})\n"
            f"Trades: {stats.get('trades_count', 0)}"
        )
        if stats.get("best_pair"):
            msg += f"\nBest: {stats['best_pair']}"
        if stats.get("worst_pair"):
            msg += f"\nWorst: {stats['worst_pair']}"
        await self.send(msg)

    async def notify_buzz(self, pair: str, buzz_score: float, sentiment: float) -> None:
        direction = "bullish" if sentiment > 0 else ("bearish" if sentiment < 0 else "neutral")
        msg = (
            f"🔥 **High Buzz Alert** `{pair}`\n"
            f"Buzz: {buzz_score:.1f}× normal · "
            f"Sentiment: {sentiment:+.2f} ({direction})"
        )
        await self.send(msg)
