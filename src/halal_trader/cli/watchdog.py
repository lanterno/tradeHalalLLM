"""Dead-man-switch watchdog — alerts Telegram when no trading cycle
has fired recently during market hours.

Designed to run out-of-process (cron / launchd) so it survives the
asyncio-loop suspensions that the in-process heartbeat cannot detect.
Each invocation is short-lived: it tails the structured log file, finds
the most recent ``cycle.start`` / ``cycle.complete`` event, compares it
to wall-clock now, and fires a single Telegram message if the gap
exceeds the threshold. A state file rate-limits repeat alerts so a
prolonged outage doesn't spam the chat every 3 minutes.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click


def _tail_bytes(path: Path, max_bytes: int = 512 * 1024) -> bytes:
    """Return the last ``max_bytes`` of ``path`` (or the whole file)."""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            f.readline()  # discard partial first line
        return f.read()


def _parse_ts(ts_str: str) -> datetime | None:
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        try:
            return datetime.strptime(ts_str.split(",")[0], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


def _find_last_activity(
    log_file: Path,
    logger_prefix: str,
) -> tuple[str, datetime] | None:
    """Walk the tail of ``log_file`` backwards, return (label, ts) of
    the most recent log record whose ``name`` starts with ``logger_prefix``.

    ``label`` is the matched logger name (or the event tag, if present)
    purely for the alert message. Returning "any recent activity" rather
    than only ``cycle.start``/``cycle.complete`` avoids false-negatives
    during pre-market/EOD windows and false-positives from a parallel
    bot that fires the same generic event names (e.g. crypto cycles
    failing every 60 s would otherwise mask a stocks-side hang).
    """
    # Scan the current file first, then the most recent rotated sibling
    # (``.log.1``). Reading moments after a rotation would otherwise see a
    # fresh/near-empty log, return None, and spuriously fire (or crash) the
    # dead-man switch — the bot floods + rotates the log under load.
    name_needle = f'"name": "{logger_prefix}'.encode()
    for path in (log_file, log_file.with_name(log_file.name + ".1")):
        try:
            blob = _tail_bytes(path)
        except FileNotFoundError:
            continue
        for raw in reversed(blob.splitlines()):
            if not raw or name_needle not in raw:
                continue
            try:
                rec = json.loads(raw)
            except (ValueError, TypeError):
                continue
            name = rec.get("name", "")
            if not isinstance(name, str) or not name.startswith(logger_prefix):
                continue
            ts_str = rec.get("timestamp")
            if not ts_str:
                continue
            ts = _parse_ts(ts_str)
            if ts is None:
                continue
            label = rec.get("event") or name
            return str(label), ts
    return None


def _read_state(state_file: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(state_file.read_text())
    except (FileNotFoundError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_state(state_file: Path, payload: dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(payload, indent=2))


async def _send_telegram(message: str) -> bool:
    from halal_trader.config import get_settings
    from halal_trader.notifications.telegram import TelegramNotifier

    settings = get_settings()
    notifier = TelegramNotifier(
        bot_token=settings.telegram.bot_token,
        chat_id=settings.telegram.chat_id,
    )
    try:
        return await notifier.send(message)
    finally:
        await notifier.close()


@click.command()
@click.option(
    "--threshold-minutes",
    type=int,
    default=20,
    show_default=True,
    help="Alert if no cycle event has fired in this many minutes.",
)
@click.option(
    "--dedup-minutes",
    type=int,
    default=30,
    show_default=True,
    help="Suppress repeat alerts within this window.",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default=Path("logs/halal_trader.log"),
    show_default=True,
    help="Structured JSON log file to tail.",
)
@click.option(
    "--state-file",
    type=click.Path(path_type=Path),
    default=Path("logs/.watchdog_state.json"),
    show_default=True,
    help="Where to record the last-alert timestamp for deduplication.",
)
@click.option(
    "--require-market-open/--any-time",
    default=True,
    show_default=True,
    help="By default the watchdog only alerts during US market hours.",
)
@click.option(
    "--logger-prefix",
    default="halal_trader.trading.",
    show_default=True,
    help="Only count log lines whose ``name`` starts with this prefix as "
    "evidence of life. Use halal_trader.crypto. for a crypto watchdog.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the would-be alert to stdout instead of pinging Telegram.",
)
def watchdog(
    threshold_minutes: int,
    dedup_minutes: int,
    log_file: Path,
    state_file: Path,
    require_market_open: bool,
    logger_prefix: str,
    dry_run: bool,
) -> None:
    """Dead-man switch: alert Telegram if no trading cycle is firing."""
    from halal_trader.market_hours import is_market_open_local, now_eastern

    if require_market_open and not is_market_open_local():
        click.echo("watchdog: market closed, skipping check")
        return

    last = _find_last_activity(log_file, logger_prefix)
    if last is None:
        message = (
            f"\U0001f6a8 halabot dead-man alert\n"
            f"No log activity from <code>{logger_prefix}*</code> found in {log_file}."
        )
        gap_minutes = None
    else:
        evt, ts = last
        now_local = datetime.now()
        gap = now_local - ts
        gap_minutes = gap.total_seconds() / 60.0
        if gap_minutes < threshold_minutes:
            click.echo(
                f"watchdog: healthy — last {logger_prefix}* activity "
                f"({evt}) {gap_minutes:.1f} min ago "
                f"(< {threshold_minutes} min threshold)"
            )
            return
        et = now_eastern()
        message = (
            f"\U0001f6a8 <b>halabot dead-man alert</b>\n"
            f"No <code>{logger_prefix}*</code> log line in "
            f"<b>{gap_minutes:.0f} min</b> (threshold {threshold_minutes} min)\n"
            f"Last seen: <code>{evt}</code> at {ts:%Y-%m-%d %H:%M:%S} local\n"
            f"Now: {et:%Y-%m-%d %H:%M} ET (market_open={is_market_open_local()})\n"
            f"PID file: {'present' if Path('halal_trader.pid').exists() else 'absent'}"
        )

    state = _read_state(state_file)
    last_alert_iso = state.get("last_alert_at")
    if last_alert_iso:
        try:
            last_alert = datetime.fromisoformat(last_alert_iso)
            since = (datetime.now() - last_alert).total_seconds() / 60.0
            if since < dedup_minutes:
                # gap_minutes is None when no matching log activity was found
                # (e.g. a log-rotation race) — guard the format so the dedup
                # path doesn't TypeError and crash the watchdog's alert branch.
                gap_str = (
                    f"{gap_minutes:.0f} min gap"
                    if gap_minutes is not None
                    else "no recent activity"
                )
                click.echo(
                    f"watchdog: would alert ({gap_str}) but "
                    f"last alert was {since:.0f} min ago (< {dedup_minutes} min dedup)"
                )
                return
        except ValueError:
            pass

    if dry_run:
        click.echo("--- DRY RUN — message below would be sent to Telegram ---")
        click.echo(message)
        return

    sent = asyncio.run(_send_telegram(message))
    if sent:
        _write_state(
            state_file,
            {
                "last_alert_at": datetime.now().isoformat(timespec="seconds"),
                "last_alert_gap_minutes": gap_minutes,
            },
        )
        click.echo(f"watchdog: alert sent ({gap_minutes:.0f} min gap)")
    else:
        click.echo(
            "watchdog: alert NOT sent — Telegram notifier disabled or send failed",
            err=True,
        )
        sys.exit(1)


__all__ = ["watchdog"]


if __name__ == "__main__":
    # Allow `python -m halal_trader.cli.watchdog` for ad-hoc launchd testing
    os.environ.setdefault("HALAL_TRADER_SKIP_LOGGING_SETUP", "1")
    watchdog()
