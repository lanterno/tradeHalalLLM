"""The evening run reports a backup that stopped, or a restore drill that lapsed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.heartbeat import beat
from halal_trader.research.daily import _backup_health


async def test_missing_stale_and_fresh_backups(engine: AsyncEngine) -> None:
    assert await _backup_health(engine) == ["backup: no nightly dump recorded yet"]
    now = datetime.now(UTC)
    await beat(engine, "backup.nightly", {"dump_mb": 391}, now=now - timedelta(hours=40))
    assert await _backup_health(engine) == ["backup: no nightly dump in the last 36 h"]
    await beat(engine, "backup.nightly", {"dump_mb": 391}, now=now)
    await beat(engine, "backup.restore_drill", {"ok": True}, now=now - timedelta(days=45))
    (problem,) = await _backup_health(engine)
    assert "restore drill" in problem
    await beat(engine, "backup.restore_drill", {"ok": True}, now=now)
    assert await _backup_health(engine) == []
