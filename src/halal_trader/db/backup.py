"""Daily SQLite backup with gzip + retention.

Operationally simple: copy the live DB via the ``sqlite3`` CLI's
``.backup`` command (which is concurrent-safe), gzip the result, drop it
in ``settings.backup.dir`` named ``halal_trader_YYYY-MM-DD.db.gz``. The
prune step keeps the last ``retention_days`` daily files and the last
``weekly_count`` Sunday files on top of that.

Ships with a CLI (`halal-trader backup`) and is wired into both bots'
end-of-day routines.
"""

from __future__ import annotations

import gzip
import logging
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from halal_trader.notifications.telegram import AlertSink

logger = logging.getLogger(__name__)

_FILENAME_RE = re.compile(r"^halal_trader_(\d{4}-\d{2}-\d{2})\.db\.gz$")


@dataclass(frozen=True)
class BackupResult:
    path: Path
    size_bytes: int
    backed_up_at: datetime


def _build_filename(today: date) -> str:
    return f"halal_trader_{today.isoformat()}.db.gz"


def _safe_sqlite_backup(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` using SQLite's online backup API.

    sqlite3.Connection.backup() acquires a shared lock for the duration
    and is safe against a writer in progress, unlike a plain file copy.
    """
    with sqlite3.connect(str(src)) as src_conn:
        with sqlite3.connect(str(dst)) as dst_conn:
            src_conn.backup(dst_conn)


def run_backup(
    *,
    db_path: Path,
    backup_dir: Path,
    today: date | None = None,
) -> BackupResult:
    """Create a gzipped backup of ``db_path`` in ``backup_dir``."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    today = today or datetime.now(UTC).date()
    target = backup_dir / _build_filename(today)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "halal_trader.db"
        _safe_sqlite_backup(db_path, staging)
        with open(staging, "rb") as src, gzip.open(target, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)

    size = target.stat().st_size
    logger.info(
        "Backup complete: %s (%.1f KB)",
        target,
        size / 1024,
        extra={
            "event": "backup.complete",
            "path": str(target),
            "size_bytes": size,
        },
    )
    return BackupResult(path=target, size_bytes=size, backed_up_at=datetime.now(UTC))


def prune_backups(
    *,
    backup_dir: Path,
    retention_days: int,
    weekly_count: int = 0,
    today: date | None = None,
) -> list[Path]:
    """Remove backups outside the retention windows. Returns deleted paths.

    Daily window: any file dated within the last ``retention_days`` days
    (today inclusive) is kept.
    Weekly window: on top of the daily window, the most recent
    ``weekly_count`` Sunday backups are kept.
    """
    if not backup_dir.exists():
        return []

    today = today or datetime.now(UTC).date()
    cutoff = today - timedelta(days=retention_days)

    files_by_date: dict[date, Path] = {}
    for path in backup_dir.iterdir():
        match = _FILENAME_RE.match(path.name)
        if not match:
            continue
        try:
            d = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        files_by_date[d] = path

    # Sundays (weekday() == 6) sorted newest-first.
    sundays = sorted(
        (d for d in files_by_date if d.weekday() == 6),
        reverse=True,
    )
    keep_sundays = set(sundays[:weekly_count])

    deleted: list[Path] = []
    for d, path in files_by_date.items():
        if d > cutoff and d <= today:
            continue
        if d in keep_sundays:
            continue
        try:
            path.unlink()
            deleted.append(path)
            logger.info(
                "Pruned old backup: %s",
                path,
                extra={"event": "backup.pruned", "path": str(path)},
            )
        except OSError as e:
            logger.warning("Could not prune %s: %s", path, e)
    return deleted


def list_backups(backup_dir: Path) -> list[BackupResult]:
    """Inventory existing backups for the dashboard / CLI."""
    if not backup_dir.exists():
        return []
    out: list[BackupResult] = []
    for path in backup_dir.iterdir():
        match = _FILENAME_RE.match(path.name)
        if not match:
            continue
        try:
            d = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        stat = path.stat()
        out.append(
            BackupResult(
                path=path,
                size_bytes=stat.st_size,
                backed_up_at=datetime(d.year, d.month, d.day, tzinfo=UTC),
            )
        )
    out.sort(key=lambda r: r.backed_up_at, reverse=True)
    return out


async def run_with_alerts(
    *,
    db_path: Path,
    backup_dir: Path,
    retention_days: int,
    weekly_count: int,
    alerts: "AlertSink | None" = None,
) -> BackupResult | None:
    """Wrapper that runs backup + prune and routes failures to AlertSink.

    Async signature so end-of-day hooks can ``await`` it without spinning
    a thread; the heavy lifting is sync but cheap (single-MB DB file).
    """
    try:
        result = run_backup(db_path=db_path, backup_dir=backup_dir)
        prune_backups(
            backup_dir=backup_dir,
            retention_days=retention_days,
            weekly_count=weekly_count,
        )
        return result
    except Exception as e:
        logger.error(
            "Backup failed: %s",
            e,
            exc_info=True,
            extra={"event": "backup.failed"},
        )
        if alerts is not None:
            await alerts.notify("backup.failed", f"{type(e).__name__}: {e}")
        return None
