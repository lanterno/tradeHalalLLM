"""Tests for `notifications/webhooks.py` (Slack + Discord notifiers).

Both notifiers should mirror the public surface of the existing
`TelegramNotifier` so the bot can route the same trade-life-cycle
events to any wired channel. We use `httpx.MockTransport` to test
the HTTP layer without real webhook URLs.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from halal_trader.notifications.webhooks import DiscordNotifier, SlackNotifier


def _slack_with_handler(handler) -> SlackNotifier:
    n = SlackNotifier("https://hooks.slack.com/services/REAL/URL/HERE")
    n._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return n


def _discord_with_handler(handler) -> DiscordNotifier:
    n = DiscordNotifier("https://discord.com/api/webhooks/12345/REAL-TOKEN")
    n._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return n


# ── enabled-property semantics ────────────────────────────


def test_slack_disabled_when_url_empty():
    assert SlackNotifier("").enabled is False


def test_slack_disabled_when_url_is_placeholder():
    """Round-4 invariant: a half-configured `.env` (placeholder URL
    left in) must NOT silently send to a fake endpoint. Pin so a
    refactor that drops the placeholder check breaks here loudly."""
    for placeholder in (
        "your_slack_webhook_url",
        "https://hooks.slack.com/services/your-webhook",
    ):
        assert SlackNotifier(placeholder).enabled is False


def test_slack_disabled_when_url_lacks_https_scheme():
    """Plaintext HTTP, file://, or just a path are not real
    webhook URLs — refuse to send."""
    assert SlackNotifier("http://example.com/hook").enabled is False
    assert SlackNotifier("/some/path").enabled is False


def test_slack_enabled_when_real_https_url():
    assert SlackNotifier("https://hooks.slack.com/services/T0/B0/realtoken").enabled is True


def test_discord_disabled_when_url_empty():
    assert DiscordNotifier("").enabled is False


def test_discord_disabled_when_url_is_placeholder():
    for placeholder in (
        "your_discord_webhook_url",
        "https://discord.com/api/webhooks/your-webhook",
    ):
        assert DiscordNotifier(placeholder).enabled is False


def test_discord_enabled_when_real_https_url():
    assert DiscordNotifier("https://discord.com/api/webhooks/12345/realtoken-12345").enabled is True


# ── send() contract ───────────────────────────────────────


@pytest.mark.asyncio
async def test_slack_send_disabled_returns_false_no_post():
    """Disabled notifiers must NOT POST."""
    n = SlackNotifier("")
    assert await n.send("hello") is False


@pytest.mark.asyncio
async def test_discord_send_disabled_returns_false_no_post():
    n = DiscordNotifier("")
    assert await n.send("hello") is False


@pytest.mark.asyncio
async def test_slack_send_returns_true_on_2xx():
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"ok": True})

    n = _slack_with_handler(handler)
    assert await n.send("hello") is True
    assert captured["body"]["text"] == "hello"


@pytest.mark.asyncio
async def test_slack_send_returns_false_on_4xx():
    """Webhook failures don't crash the bot — return False so the
    caller can decide. Pin the swallow."""

    def handler(req):
        return httpx.Response(400, json={"error": "no_text"})

    n = _slack_with_handler(handler)
    assert await n.send("hello") is False


@pytest.mark.asyncio
async def test_slack_send_swallows_network_exception():
    """A connection error returning False (not raising) keeps the
    operator's cycle running — Slack down ≠ bot halted."""

    def handler(req):
        raise httpx.ConnectError("simulated DNS failure")

    n = _slack_with_handler(handler)
    assert await n.send("hello") is False


@pytest.mark.asyncio
async def test_slack_send_includes_channel_when_set():
    """`SLACK_CHANNEL` override flows into the payload — Slack uses
    it when the workspace permits cross-channel posting."""
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _slack_with_handler(handler)
    n._channel = "#alerts"
    await n.send("ping")
    assert captured["body"]["channel"] == "#alerts"


@pytest.mark.asyncio
async def test_slack_send_omits_channel_when_unset():
    """Default → no `channel` field; the webhook posts to whatever
    channel it was bound to during creation."""
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _slack_with_handler(handler)
    await n.send("ping")
    assert "channel" not in captured["body"]


@pytest.mark.asyncio
async def test_slack_send_passes_blocks_when_provided():
    """Slack Block Kit support — pin so a refactor doesn't drop
    the optional kwarg."""
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _slack_with_handler(handler)
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "hi"}}]
    await n.send("preview", blocks=blocks)
    assert captured["body"]["blocks"] == blocks
    assert captured["body"]["text"] == "preview"  # preview text preserved


@pytest.mark.asyncio
async def test_discord_send_includes_username():
    """Discord webhooks let the sender override the displayed
    username on a per-post basis. We always send the configured one."""
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _discord_with_handler(handler)
    n._username = "halal-bot-prod"
    await n.send("hello")
    assert captured["body"]["username"] == "halal-bot-prod"


@pytest.mark.asyncio
async def test_discord_send_truncates_at_1900_chars():
    """Discord's hard 2000-char limit on `content` — we cap below it
    so a verbose LLM rationale doesn't reject the post."""
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _discord_with_handler(handler)
    long_msg = "x" * 5000
    await n.send(long_msg)
    assert len(captured["body"]["content"]) <= 1900


@pytest.mark.asyncio
async def test_discord_send_passes_embeds_when_provided():
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _discord_with_handler(handler)
    embeds = [{"title": "Trade", "color": 0x16A34A}]
    await n.send("text", embeds=embeds)
    assert captured["body"]["embeds"] == embeds


# ── notify_trade ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_slack_notify_trade_renders_buy_emoji_and_pair():
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _slack_with_handler(handler)
    await n.notify_trade(pair="BTCUSDT", side="buy", quantity=0.1, price=50_000.0)
    text = captured["body"]["text"]
    assert "🟢" in text  # buy emoji
    assert "BTCUSDT" in text
    assert "$50,000.00" in text
    assert "$5,000.00" in text  # notional


@pytest.mark.asyncio
async def test_slack_notify_trade_renders_sell_emoji():
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _slack_with_handler(handler)
    await n.notify_trade(pair="ETHUSDT", side="sell", quantity=1, price=3000.0)
    assert "🔴" in captured["body"]["text"]


@pytest.mark.asyncio
async def test_slack_notify_trade_truncates_long_reasoning():
    """Verbose LLM rationales get truncated at 200 chars so the
    Slack message stays readable."""
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _slack_with_handler(handler)
    long_reason = "x" * 500
    await n.notify_trade(
        pair="BTCUSDT",
        side="buy",
        quantity=0.1,
        price=50_000.0,
        reasoning=long_reason,
    )
    text = captured["body"]["text"]
    # Reasoning portion (after newline) capped at 200 chars.
    reasoning_part = text.split("\n>", 1)[1] if "\n>" in text else ""
    assert len(reasoning_part) <= 200


@pytest.mark.asyncio
async def test_discord_notify_trade_uses_markdown_bold_not_asterisks_only():
    """Slack uses `*bold*`, Discord uses `**bold**`. Pin so a refactor
    doesn't accidentally use Slack syntax on Discord."""
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _discord_with_handler(handler)
    await n.notify_trade(pair="BTCUSDT", side="buy", quantity=0.1, price=50_000.0)
    content = captured["body"]["content"]
    assert "**BUY**" in content


# ── notify_sl_tp ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_slack_notify_sl_tp_winner_uses_check_emoji():
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _slack_with_handler(handler)
    await n.notify_sl_tp(
        pair="BTCUSDT",
        exit_reason="take_profit",
        entry_price=50_000.0,
        exit_price=52_000.0,
        pnl_pct=0.04,
    )
    text = captured["body"]["text"]
    assert "✅" in text
    assert "+4.00%" in text


@pytest.mark.asyncio
async def test_slack_notify_sl_tp_loser_uses_cross_emoji():
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _slack_with_handler(handler)
    await n.notify_sl_tp(
        pair="BTCUSDT",
        exit_reason="stop_loss",
        entry_price=50_000.0,
        exit_price=49_000.0,
        pnl_pct=-0.02,
    )
    text = captured["body"]["text"]
    assert "❌" in text
    assert "-2.00%" in text


# ── notify_daily_summary ──────────────────────────────────


@pytest.mark.asyncio
async def test_slack_daily_summary_renders_pnl_and_trades():
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _slack_with_handler(handler)
    await n.notify_daily_summary(
        {
            "realized_pnl": 250.50,
            "return_pct": 0.025,
            "trades_count": 3,
            "best_pair": "BTCUSDT",
            "worst_pair": "ETHUSDT",
        }
    )
    text = captured["body"]["text"]
    assert "+$250.50" in text
    assert "Trades: 3" in text
    assert "Best: BTCUSDT" in text
    assert "Worst: ETHUSDT" in text


@pytest.mark.asyncio
async def test_slack_daily_summary_falls_back_to_total_pnl_key():
    """Older callers used `total_pnl` instead of `realized_pnl`. Both
    must work — pin the back-compat fallback."""
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _slack_with_handler(handler)
    await n.notify_daily_summary({"total_pnl": 100.0, "trades_count": 1})
    assert "+$100.00" in captured["body"]["text"]


@pytest.mark.asyncio
async def test_slack_daily_summary_omits_best_worst_when_absent():
    """No best/worst keys → those lines aren't rendered. Pin so an
    empty-day summary doesn't have dangling 'Best:' headers."""
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _slack_with_handler(handler)
    await n.notify_daily_summary({"realized_pnl": 0, "trades_count": 0})
    text = captured["body"]["text"]
    assert "Best:" not in text
    assert "Worst:" not in text


@pytest.mark.asyncio
async def test_discord_daily_summary_renders():
    """Symmetric Discord coverage."""
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _discord_with_handler(handler)
    await n.notify_daily_summary({"realized_pnl": -50, "return_pct": -0.005, "trades_count": 2})
    content = captured["body"]["content"]
    assert "$-50.00" in content
    assert "Trades: 2" in content


# ── notify_buzz ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_slack_buzz_renders_bullish_label():
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _slack_with_handler(handler)
    await n.notify_buzz(pair="DOGEUSDT", buzz_score=4.2, sentiment=0.6)
    text = captured["body"]["text"]
    assert "🔥" in text
    assert "DOGEUSDT" in text
    assert "bullish" in text


@pytest.mark.asyncio
async def test_slack_buzz_renders_bearish_label():
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _slack_with_handler(handler)
    await n.notify_buzz(pair="DOGEUSDT", buzz_score=3.0, sentiment=-0.4)
    assert "bearish" in captured["body"]["text"]


@pytest.mark.asyncio
async def test_slack_buzz_renders_neutral_label():
    """Sentiment exactly 0 → 'neutral' label (not bullish or bearish)."""
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200)

    n = _slack_with_handler(handler)
    await n.notify_buzz(pair="DOGEUSDT", buzz_score=2.5, sentiment=0.0)
    assert "neutral" in captured["body"]["text"]


# ── close() ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_slack_close_releases_client():
    n = SlackNotifier("https://hooks.slack.com/services/T0/B0/x")
    _ = n._get_client()
    assert n._client is not None
    await n.close()
    assert n._client is None


@pytest.mark.asyncio
async def test_slack_close_no_op_when_never_used():
    n = SlackNotifier("https://hooks.slack.com/services/T0/B0/x")
    await n.close()  # must not raise


@pytest.mark.asyncio
async def test_discord_close_releases_client():
    n = DiscordNotifier("https://discord.com/api/webhooks/0/x")
    _ = n._get_client()
    await n.close()
    assert n._client is None


# ── Settings integration ──────────────────────────────────


def test_slack_settings_default_empty_url():
    from halal_trader.config import SlackSettings

    s = SlackSettings()
    assert s.webhook_url == ""
    assert s.channel == ""
    # Pin: a fresh Settings → notifier is disabled by default.
    assert SlackNotifier(s.webhook_url, channel=s.channel).enabled is False


def test_discord_settings_default_username():
    from halal_trader.config import DiscordSettings

    s = DiscordSettings()
    assert s.webhook_url == ""
    assert s.username == "halal-trader"
