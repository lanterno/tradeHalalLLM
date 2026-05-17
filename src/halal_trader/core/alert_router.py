"""Multi-channel alert router with severity routing + dedup.

Round-4 wave 8.E: today's `notifications/telegram.py:AlertSink` is
rate-limited but Telegram-only. Operators in production want
different severities reaching different channels — INFO to a Slack
discussion channel, WARN to the on-call Slack room, PAGE to
Telegram and email and PagerDuty. This module is the routing
layer:

* `AlertSpec(type, severity, summary, …)` — typed alert payload
  the bot's modules raise instead of free-form `notify()`.
* `Channel` — Protocol that every notifier implements (already
  matches the existing Telegram / Slack / Discord notifiers'
  shape).
* `AlertRoute(severity_min, channel)` — declares "send PAGE+
  alerts to this channel".
* `AlertRouter` — composes routes, routes one alert across every
  matching channel, applies a per-`(type, channel)` dedup window.

Why generalise the existing `AlertSink` rather than extend it:

* `AlertSink` deduplicates per `error_type`; the router needs
  per-`(error_type, channel)` dedup so a Slack-suppressed alert
  can still reach Telegram (the operator only needs to see it
  once *per channel*, not once *globally*).
* The router is strategy-agnostic (`core/`); existing
  `AlertSink` lives next to a specific notifier (`notifications/`).
  The router can compose Telegram + Slack + Discord +
  email + a future PagerDuty without circular imports.

Halal alignment: the router moves alert messages, never opens a
position. A delivery failure is logged, not retried — operator
deals with the underlying error rather than the router masking
it. Rate-limiting is per-(type, channel) so a flapping error
can't pager-storm any single channel.

Pure-Python; no DB / network. Channels are Protocol-typed so the
router can be tested with stub channels that don't talk to real
APIs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Iterable, Protocol

logger = logging.getLogger(__name__)


# ── Vocabulary ────────────────────────────────────────────


class Severity(IntEnum):
    """Severity ladder. IntEnum so `severity >= warn` reads
    naturally in route matching.

    * ``INFO`` — observability ticks, daily summary, position
      open / close. Informative; the operator usually mutes.
    * ``WARN`` — single drift breach, single broker error, partial
      backoff. Operator should glance.
    * ``PAGE`` — kill-switch tripped, halt engaged, repeated chain
      backoffs, snapshot store failure. Wake the operator.
    """

    INFO = 10
    WARN = 20
    PAGE = 30


# ── Alert payload ─────────────────────────────────────────


@dataclass(frozen=True)
class AlertSpec:
    """Typed alert payload.

    ``type`` is a stable identifier for the alert family
    (`"chain.backoff"`, `"halt.engaged"`, `"drift.breach"`) — used
    as the dedup key so a flapping condition collapses to one
    message per window.

    ``runbook_url`` points the on-call at the recovery procedure.
    Pin: empty string when no runbook exists yet, but the renderer
    surfaces "no runbook yet" so the operator can spot the gap
    and write one.

    ``context`` carries free-form key/value pairs the channel
    can render (cycle_id, pair, threshold value, …). Pin: never
    used for dedup — only `type` controls suppression.
    """

    type: str
    severity: Severity
    summary: str
    runbook_url: str = ""
    context: dict[str, str] = field(default_factory=dict)


# ── Channel protocol ──────────────────────────────────────


class Channel(Protocol):
    """A notification channel the router can dispatch to.

    The Protocol matches the existing Telegram / Slack / Discord
    notifiers' shape (an async `send` returning a bool — True if
    the message went out, False if suppressed / failed). The
    `name` is used in logs and dedup keys.
    """

    @property
    def name(self) -> str: ...

    @property
    def enabled(self) -> bool: ...

    async def send(self, message: str) -> bool: ...


# ── Routing rules ─────────────────────────────────────────


@dataclass(frozen=True)
class AlertRoute:
    """One routing rule: channel X receives alerts at severity ≥ Y.

    Multiple routes can target the same channel (e.g. a "WARN to
    Slack #alerts; PAGE to Slack #alerts AND telegram"); the
    router de-duplicates channel matches so the channel still gets
    one message per alert.
    """

    severity_min: Severity
    channel: Channel

    def matches(self, alert: AlertSpec) -> bool:
        return alert.severity >= self.severity_min


# ── Dedup window ──────────────────────────────────────────


@dataclass
class _DedupTracker:
    """Per-(type, channel) cooldown.

    Pin: cooldown is per-channel pair so a Slack-suppressed alert
    can still reach Telegram. Operators see each alert once per
    channel they subscribe to.
    """

    cooldown_seconds: float
    _last_sent: dict[tuple[str, str], float] = field(default_factory=dict)
    _suppressed: dict[tuple[str, str], int] = field(default_factory=dict)
    _now: Callable[[], float] = field(default_factory=lambda: time.monotonic)

    def should_send(self, alert_type: str, channel_name: str) -> bool:
        """Check whether (type, channel) is outside its cooldown
        window. Returns True iff the alert should go out."""
        key = (alert_type, channel_name)
        last = self._last_sent.get(key)
        now = self._now()
        if last is None or (now - last) >= self.cooldown_seconds:
            return True
        self._suppressed[key] = self._suppressed.get(key, 0) + 1
        return False

    def mark_sent(self, alert_type: str, channel_name: str) -> None:
        key = (alert_type, channel_name)
        self._last_sent[key] = self._now()

    def suppressed_count(self, alert_type: str, channel_name: str) -> int:
        return self._suppressed.get((alert_type, channel_name), 0)

    def reset_suppressed(self, alert_type: str, channel_name: str) -> int:
        """Return the suppressed count and clear it. Used to surface
        "X suppressed in last window" in the next message."""
        key = (alert_type, channel_name)
        count = self._suppressed.pop(key, 0)
        return count


# ── Router ────────────────────────────────────────────────


@dataclass(frozen=True)
class DispatchResult:
    """Per-channel outcome of one alert dispatch.

    ``sent`` is the channel names the alert reached;
    ``suppressed`` lists channels where the alert was within its
    cooldown window; ``failed`` lists channels where send raised
    or returned False.
    """

    alert_type: str
    severity: Severity
    sent: list[str] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def reached_any(self) -> bool:
        return len(self.sent) > 0


class AlertRouter:
    """Compose routes and dispatch alerts.

    ``cooldown_seconds`` is per-(type, channel). Default 15 min
    matches the existing `AlertSink.DEFAULT_COOLDOWN_SECONDS`.
    Pass a callable for `now_fn` in tests to control the clock —
    pin so dedup behaviour is deterministic in regression tests.
    """

    DEFAULT_COOLDOWN_SECONDS = 900  # 15 min

    def __init__(
        self,
        routes: Iterable[AlertRoute],
        *,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._routes = list(routes)
        self._dedup = _DedupTracker(
            cooldown_seconds=cooldown_seconds,
            _now=now_fn or time.monotonic,
        )

    def matching_channels(self, alert: AlertSpec) -> list[Channel]:
        """Channels that match the alert. Pin: deduplicated by
        channel.name so two routes targeting the same channel still
        produce one delivery."""
        seen: set[str] = set()
        out: list[Channel] = []
        for route in self._routes:
            if not route.matches(alert):
                continue
            if route.channel.name in seen:
                continue
            seen.add(route.channel.name)
            out.append(route.channel)
        return out

    async def dispatch(self, alert: AlertSpec) -> DispatchResult:
        """Send ``alert`` through every matching channel.

        Pin: a per-channel send failure does NOT abort dispatch to
        other channels — the operator wants Telegram delivered
        even if Slack is down. Failures are recorded in the
        result so the caller can decide what to do.
        """
        result = DispatchResult(alert_type=alert.type, severity=alert.severity)
        message = render_message(alert)
        for channel in self.matching_channels(alert):
            if not channel.enabled:
                # Channel deliberately disabled (no token, etc.) —
                # not a failure, not a suppression. Skip silently.
                continue
            if not self._dedup.should_send(alert.type, channel.name):
                result.suppressed.append(channel.name)
                continue
            try:
                ok = await channel.send(message)
            except Exception as exc:  # noqa: BLE001
                logger.warning("alert dispatch to %s failed: %s", channel.name, exc)
                result.failed.append(channel.name)
                continue
            if ok:
                self._dedup.mark_sent(alert.type, channel.name)
                result.sent.append(channel.name)
            else:
                result.failed.append(channel.name)
        return result


# ── Renderer ──────────────────────────────────────────────


_SEVERITY_EMOJI: dict[Severity, str] = {
    Severity.INFO: "ℹ️",
    Severity.WARN: "⚠️",
    Severity.PAGE: "🚨",
}


def render_message(alert: AlertSpec) -> str:
    """Operator-readable text payload for one alert.

    Format mirrors the other Round-4 render helpers — emoji
    severity prefix, type tag in backticks, summary, runbook
    pointer (or "no runbook yet" if empty), context k/v pairs.
    """
    emoji = _SEVERITY_EMOJI.get(alert.severity, "•")
    lines = [f"{emoji} `{alert.type}` [{alert.severity.name}]"]
    lines.append(alert.summary)
    if alert.runbook_url:
        lines.append(f"Runbook: {alert.runbook_url}")
    else:
        lines.append("Runbook: (none yet — write one in docs/runbooks/)")
    if alert.context:
        for k, v in sorted(alert.context.items()):
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


__all__ = [
    "AlertRoute",
    "AlertRouter",
    "AlertSpec",
    "Channel",
    "DispatchResult",
    "Severity",
    "render_message",
]
