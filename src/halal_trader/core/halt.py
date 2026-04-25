"""Operator kill-switch — read/write the single-row ``kill_switch`` table.

Both bots check :func:`is_halted` at the top of every cycle. The
monitor checks once per minute and refuses NEW entries; in-flight SL/TP
exits still run because closing risk is preferable to holding overnight
under unknown failure.
"""

from __future__ import annotations

import getpass
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlmodel import select

from halal_trader.db.models import KillSwitch


@dataclass(frozen=True)
class HaltStatus:
    enabled: bool
    reason: str | None
    set_by: str | None
    set_at: datetime | None


async def get_status(engine: AsyncEngine) -> HaltStatus:
    """Return the current kill-switch state."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        result = await session.execute(select(KillSwitch).where(KillSwitch.id == 1))
        row = result.scalars().first()
        if row is None:
            return HaltStatus(False, None, None, None)
        return HaltStatus(
            enabled=bool(row.enabled),
            reason=row.reason,
            set_by=row.set_by,
            set_at=row.set_at,
        )


async def is_halted(engine: AsyncEngine) -> bool:
    return (await get_status(engine)).enabled


async def set_halt(engine: AsyncEngine, *, reason: str, set_by: str | None = None) -> HaltStatus:
    """Engage the kill-switch."""
    if set_by is None:
        try:
            set_by = getpass.getuser()
        except Exception:
            set_by = "unknown"

    set_at = datetime.now(UTC)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        result = await session.execute(select(KillSwitch).where(KillSwitch.id == 1))
        row = result.scalars().first()
        if row is None:
            row = KillSwitch(id=1, enabled=True, reason=reason, set_by=set_by, set_at=set_at)
            session.add(row)
        else:
            row.enabled = True
            row.reason = reason
            row.set_by = set_by
            row.set_at = set_at
            session.add(row)
        await session.commit()

    return HaltStatus(enabled=True, reason=reason, set_by=set_by, set_at=set_at)


async def clear_halt(engine: AsyncEngine) -> HaltStatus:
    """Disengage the kill-switch (keeps the audit fields populated)."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        result = await session.execute(select(KillSwitch).where(KillSwitch.id == 1))
        row = result.scalars().first()
        if row is None:
            return HaltStatus(False, None, None, None)
        row.enabled = False
        session.add(row)
        prev_reason = row.reason
        prev_set_by = row.set_by
        prev_set_at = row.set_at
        await session.commit()

    return HaltStatus(enabled=False, reason=prev_reason, set_by=prev_set_by, set_at=prev_set_at)
