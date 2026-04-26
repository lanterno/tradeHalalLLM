"""Order-time halal gate tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from halal_trader.db import admin
from halal_trader.db.repository import Repository
from halal_trader.halal.gate import halal_gate


async def _engine_repo(tmp_path):
    db_path = tmp_path / "gate.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    head = admin.head()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await conn.execute(sa.text(f"INSERT INTO alembic_version (version_num) VALUES ('{head}')"))
    return engine, Repository(engine)


def _screener(*, halal: bool, refresh_raises: Exception | None = None):
    s = MagicMock()
    s.is_halal = AsyncMock(return_value=halal)
    s.refresh_if_stale = AsyncMock(side_effect=refresh_raises)
    return s


async def test_halal_symbol_returns_screening_id(tmp_path):
    engine, repo = await _engine_repo(tmp_path)
    try:
        sid = await halal_gate(
            repo, screener=_screener(halal=True), symbol="AAPL", asset_class="stock"
        )
        assert sid is not None and sid > 0

        loaded = await repo.get_halal_screening(sid)
        assert loaded["decision"] == "halal"
    finally:
        await engine.dispose()


async def test_non_halal_symbol_returns_none_but_records_audit(tmp_path):
    """Even when blocking a trade, we persist the screening row for audit."""
    engine, repo = await _engine_repo(tmp_path)
    try:
        sid = await halal_gate(
            repo, screener=_screener(halal=False), symbol="GMBL", asset_class="stock"
        )
        assert sid is None
        # Audit row exists with decision='not_halal'.
        async with engine.begin() as conn:
            row = await conn.execute(
                sa.text("SELECT decision FROM halal_screenings WHERE symbol = 'GMBL'")
            )
            assert row.scalar_one() == "not_halal"
    finally:
        await engine.dispose()


async def test_screener_exception_treated_as_non_halal(tmp_path):
    """A screener outage must never let a non-compliant trade through."""
    engine, repo = await _engine_repo(tmp_path)
    try:
        screener = _screener(halal=True)
        screener.is_halal = AsyncMock(side_effect=RuntimeError("network gone"))
        sid = await halal_gate(repo, screener=screener, symbol="AAPL", asset_class="stock")
        assert sid is None
    finally:
        await engine.dispose()


async def test_refresh_failure_does_not_abort_gate(tmp_path):
    """A flaky cache refresh should still allow the is_halal check to run."""
    engine, repo = await _engine_repo(tmp_path)
    try:
        screener = _screener(halal=True, refresh_raises=RuntimeError("zoya 503"))
        sid = await halal_gate(repo, screener=screener, symbol="AAPL", asset_class="stock")
        assert sid is not None
    finally:
        await engine.dispose()


async def test_invalid_asset_class_raises(tmp_path):
    engine, repo = await _engine_repo(tmp_path)
    try:
        with pytest.raises(ValueError):
            await halal_gate(
                repo, screener=_screener(halal=True), symbol="X", asset_class="options"
            )
    finally:
        await engine.dispose()


async def test_screener_without_refresh_method_still_works(tmp_path):
    """Older screeners may not expose refresh_if_stale — gate should tolerate."""
    engine, repo = await _engine_repo(tmp_path)
    try:
        screener = MagicMock(spec=["is_halal"])
        screener.is_halal = AsyncMock(return_value=True)
        sid = await halal_gate(repo, screener=screener, symbol="AAPL", asset_class="stock")
        assert sid is not None
    finally:
        await engine.dispose()
