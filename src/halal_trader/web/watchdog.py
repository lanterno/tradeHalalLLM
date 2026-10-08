"""The fleet's watchdog: alert when a process or a daily job stops beating.

It runs in the web, a separate container from the bot and the shadow, so a
dead bot cannot silence its own alarm. Every few minutes it judges the
heartbeat rows (core/heartbeat.py:assess) and sends a Telegram message when
a watched component goes stale, and another when it comes back.

What it has alerted is kept in its own heartbeat row (``web.watchdog``), so
a restarted web neither repeats an alert nor forgets to send the recovery.
A component must fail two checks in a row before it alerts: a deploy that
restarts the bot for a minute or two is not an outage.
"""

from __future__ import annotations

import asyncio
import html
import logging
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.heartbeat import (
    CONTINUOUS,
    DAILY_JOBS,
    MARKET_SNAPSHOT,
    STOCK_CYCLE,
    WATCHDOG,
    Beat,
    Status,
    assess,
    beat,
    cycles_due_at,
    read_beats,
)

logger = logging.getLogger(__name__)

WATCHED: tuple[str, ...] = (*CONTINUOUS, STOCK_CYCLE, MARKET_SNAPSHOT, *DAILY_JOBS)

_WHAT = {
    "stock.process": "The stock bot",
    "shadow.process": "The shadow engine",
    "stock.monitor": "The stop-loss / take-profit monitor",
    "stock.cycle": "The day-trader's cycle",
    "market.snapshot": "The per-minute market snapshot",
    "recommendation.daily": "The daily recommendation job",
    "stock.eod": "The end-of-day job",
    "stock.ledger": "The broker-ledger sync",
    "research.daily": "The evening research run",
    "core.trade": "The core portfolio's 15:40 run",
    "digest.weekly": "The weekly digest",
}


class Sender(Protocol):
    @property
    def enabled(self) -> bool: ...

    async def send(self, message: str, *, parse_mode: str = ...) -> bool: ...


def _state(beats: dict[str, Beat]) -> tuple[set[str], set[str]]:
    detail = beats[WATCHDOG].detail if WATCHDOG in beats else None
    detail = detail if isinstance(detail, dict) else {}
    return set(detail.get("alerting") or []), set(detail.get("suspect") or [])


async def check_once(
    engine: AsyncEngine,
    sender: Sender | None,
    *,
    now: datetime | None = None,
) -> dict[str, list[str]]:
    """One watchdog pass. Returns what it alerted and what recovered (for tests)."""
    now = now or datetime.now(UTC)
    beats = await read_beats(engine)
    alerting, suspect = _state(beats)
    statuses = assess(
        beats,
        now=now,
        cycles_due=cycles_due_at(now),
    )
    failing = {c for c in WATCHED if statuses.get(c, Status("ok")).failing}
    can_send = sender is not None and sender.enabled

    sent: list[str] = []
    for component in sorted((failing & suspect) - alerting):
        st = statuses[component]
        message = (
            f"\U0001f6a8 <b>{html.escape(_WHAT.get(component, component))} stopped "
            f"({html.escape(component)})</b>\n{html.escape(st.reason or st.status)}"
        )
        if not can_send or await _send(sender, message):
            alerting.add(component)
            sent.append(component)

    recovered: list[str] = []
    for component in sorted(alerting - failing):
        message = (
            f"✅ <b>{html.escape(_WHAT.get(component, component))} is back "
            f"({html.escape(component)})</b>"
        )
        if not can_send or await _send(sender, message):
            alerting.discard(component)
            recovered.append(component)

    if sent or recovered:
        logger.warning("watchdog: alerted %s, recovered %s", sent, recovered)
    detail: dict[str, Any] = {"alerting": sorted(alerting), "suspect": sorted(failing)}
    await beat(engine, WATCHDOG, detail, now=now)
    return {"alerted": sent, "recovered": recovered}


async def _send(sender: Sender | None, message: str) -> bool:
    if sender is None:
        return False
    try:
        return bool(await sender.send(message))
    except Exception as exc:  # noqa: BLE001 -- retried on the next pass
        logger.warning("watchdog alert not sent: %r", exc)
        return False


async def run(
    engine: AsyncEngine,
    sender: Sender | None,
    *,
    interval_s: float,
) -> None:
    """Check every ``interval_s`` until cancelled, starting one interval in.

    The first pass waits a full interval so a fleet restart (every container
    at once) has time to beat before anything is judged.
    """
    while True:
        await asyncio.sleep(interval_s)
        try:
            await check_once(engine, sender)
        except Exception as exc:  # noqa: BLE001 -- the watchdog must outlive a bad pass
            logger.warning("watchdog pass failed: %r", exc)
