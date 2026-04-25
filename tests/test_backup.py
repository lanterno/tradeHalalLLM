"""Tests for db/backup.py — gzipped SQLite snapshots + retention."""

from __future__ import annotations

import gzip
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.db import backup
from halal_trader.notifications.telegram import AlertSink, TelegramNotifier


@pytest.fixture
def sample_db(tmp_path: Path) -> Path:
    """A non-empty SQLite file we can copy via the backup API."""
    import sqlite3

    db_path = tmp_path / "halal_trader.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE t(id INTEGER, value TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'hello'), (2, 'world')")
        conn.commit()
    return db_path


def test_run_backup_writes_gzipped_snapshot(sample_db, tmp_path):
    backup_dir = tmp_path / "backups"
    result = backup.run_backup(
        db_path=sample_db,
        backup_dir=backup_dir,
        today=date(2026, 4, 25),
    )
    assert result.path.name == "halal_trader_2026-04-25.db.gz"
    assert result.path.exists()
    assert result.size_bytes > 0

    # The gzipped content should decompress to a valid SQLite file
    # containing our seeded row.
    import sqlite3

    extracted = tmp_path / "extracted.db"
    with gzip.open(result.path, "rb") as src, open(extracted, "wb") as dst:
        dst.write(src.read())
    with sqlite3.connect(str(extracted)) as conn:
        rows = list(conn.execute("SELECT id, value FROM t ORDER BY id"))
    assert rows == [(1, "hello"), (2, "world")]


def test_run_backup_creates_backup_dir(sample_db, tmp_path):
    backup_dir = tmp_path / "deep" / "nested" / "backups"
    backup.run_backup(db_path=sample_db, backup_dir=backup_dir, today=date(2026, 1, 1))
    assert backup_dir.is_dir()


def _seed_backup(dir: Path, d: date) -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    p = dir / f"halal_trader_{d.isoformat()}.db.gz"
    p.write_bytes(b"\x1f\x8b\x08")  # gzip magic; size irrelevant for prune
    return p


def test_prune_drops_old_files_outside_retention_window(tmp_path):
    backup_dir = tmp_path / "b"
    today = date(2026, 4, 25)
    keep = _seed_backup(backup_dir, today - timedelta(days=5))
    drop = _seed_backup(backup_dir, today - timedelta(days=60))

    deleted = backup.prune_backups(
        backup_dir=backup_dir, retention_days=30, weekly_count=0, today=today
    )
    assert keep.exists()
    assert not drop.exists()
    assert drop in deleted


def test_prune_keeps_weekly_sundays(tmp_path):
    backup_dir = tmp_path / "b"
    today = date(2026, 4, 25)  # Saturday
    # Create one Sunday-dated backup that's beyond the daily window:
    sunday = date(2025, 1, 5)  # historic Sunday
    sunday_path = _seed_backup(backup_dir, sunday)
    weekday_path = _seed_backup(backup_dir, sunday + timedelta(days=1))  # Monday

    backup.prune_backups(backup_dir=backup_dir, retention_days=7, weekly_count=4, today=today)
    assert sunday_path.exists(), "Sunday should be retained as a weekly"
    assert not weekday_path.exists(), "Off-week Monday should have been pruned"


def test_prune_keeps_only_most_recent_n_sundays(tmp_path):
    backup_dir = tmp_path / "b"
    today = date(2026, 4, 25)
    # Three Sundays in the past, all outside the 1-day daily window.
    s1 = _seed_backup(backup_dir, date(2026, 1, 4))
    s2 = _seed_backup(backup_dir, date(2026, 2, 1))
    s3 = _seed_backup(backup_dir, date(2026, 3, 1))

    backup.prune_backups(backup_dir=backup_dir, retention_days=1, weekly_count=2, today=today)
    # Only the two most recent Sundays survive; the oldest is gone.
    assert not s1.exists()
    assert s2.exists()
    assert s3.exists()


def test_list_backups_sorted_newest_first(tmp_path):
    backup_dir = tmp_path / "b"
    a = _seed_backup(backup_dir, date(2026, 1, 1))
    b = _seed_backup(backup_dir, date(2026, 4, 25))
    listed = backup.list_backups(backup_dir)
    assert [r.path for r in listed] == [b, a]


def test_list_backups_skips_unknown_filenames(tmp_path):
    backup_dir = tmp_path / "b"
    backup_dir.mkdir()
    valid = _seed_backup(backup_dir, date(2026, 4, 25))
    (backup_dir / "random.txt").write_text("noise")
    (backup_dir / "halal_trader_BAD.db.gz").write_bytes(b"\x00")

    listed = backup.list_backups(backup_dir)
    assert [r.path for r in listed] == [valid]


@pytest.mark.asyncio
async def test_run_with_alerts_swallows_failure_and_alerts(monkeypatch, tmp_path):
    notifier = MagicMock(spec=TelegramNotifier)
    notifier.enabled = True
    notifier.notify_error = AsyncMock()
    sink = AlertSink(notifier=notifier)

    def _boom(**_):
        raise OSError("disk full")

    monkeypatch.setattr(backup, "run_backup", _boom)

    result = await backup.run_with_alerts(
        db_path=tmp_path / "x.db",
        backup_dir=tmp_path / "b",
        retention_days=30,
        weekly_count=12,
        alerts=sink,
    )
    assert result is None
    notifier.notify_error.assert_awaited_once()
    err_type, details = notifier.notify_error.await_args.args
    assert err_type == "backup.failed"
    assert "disk full" in details


@pytest.mark.asyncio
async def test_run_with_alerts_returns_result_on_success(sample_db, tmp_path):
    sink = AlertSink(notifier=None)
    result = await backup.run_with_alerts(
        db_path=sample_db,
        backup_dir=tmp_path / "b",
        retention_days=30,
        weekly_count=12,
        alerts=sink,
    )
    assert result is not None
    assert result.path.exists()
