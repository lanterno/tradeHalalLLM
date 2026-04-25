"""Tests for the operator kill-switch (core/halt.py)."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from halal_trader.core import halt
from halal_trader.core.cycle import BaseCycleService


@pytest.fixture
async def engine(tmp_path):
    db_path = tmp_path / "halt.db"
    eng = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with eng.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield eng
    await eng.dispose()


async def test_status_when_uninitialized(engine):
    s = await halt.get_status(engine)
    assert s.enabled is False
    assert s.reason is None
    assert s.set_by is None
    assert s.set_at is None


async def test_set_halt_persists_and_engages(engine):
    s = await halt.set_halt(engine, reason="manual fire-drill", set_by="alice")
    assert s.enabled
    assert s.reason == "manual fire-drill"
    assert s.set_by == "alice"
    assert s.set_at is not None

    again = await halt.get_status(engine)
    assert again.enabled
    assert again.reason == "manual fire-drill"


async def test_clear_halt_disengages_but_keeps_audit(engine):
    await halt.set_halt(engine, reason="r", set_by="alice")
    cleared = await halt.clear_halt(engine)
    assert cleared.enabled is False
    assert cleared.reason == "r"
    assert cleared.set_by == "alice"

    s = await halt.get_status(engine)
    assert s.enabled is False
    assert s.reason == "r"


async def test_is_halted_shortcut(engine):
    assert await halt.is_halted(engine) is False
    await halt.set_halt(engine, reason="r", set_by="alice")
    assert await halt.is_halted(engine) is True
    await halt.clear_halt(engine)
    assert await halt.is_halted(engine) is False


async def test_clear_halt_when_uninitialized(engine):
    s = await halt.clear_halt(engine)
    assert s.enabled is False


# ── Cycle integration ──────────────────────────────────────────


class _CountingCycle(BaseCycleService):
    def __init__(self, engine) -> None:
        super().__init__(engine=engine)
        self.impl_calls = 0

    async def _pre_cycle_checks(self) -> bool:
        return True

    async def _should_halt(self) -> bool:
        return False

    async def _run_cycle_impl(self) -> None:
        self.impl_calls += 1


async def test_run_cycle_skips_when_killswitch_engaged(engine):
    cycle = _CountingCycle(engine)
    await cycle.run_cycle()
    assert cycle.impl_calls == 1

    await halt.set_halt(engine, reason="r", set_by="bob")
    await cycle.run_cycle()
    assert cycle.impl_calls == 1  # still 1 — kill-switch blocked it

    await halt.clear_halt(engine)
    await cycle.run_cycle()
    assert cycle.impl_calls == 2
