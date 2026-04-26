"""Persisted purification ledger — repository round-trip + totals."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from halal_trader.db import admin
from halal_trader.db.repository import Repository


async def _engine_repo(tmp_path):
    db_path = tmp_path / "purif.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    head = admin.head()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await conn.execute(sa.text(f"INSERT INTO alembic_version (version_num) VALUES ('{head}')"))
    return engine, Repository(engine)


async def test_record_and_read_outstanding(tmp_path):
    engine, repo = await _engine_repo(tmp_path)
    try:
        eid = await repo.record_purification(
            symbol="aapl",
            dividend_usd=100.0,
            haram_pct=0.05,
            purification_usd=5.0,
            notes="Q1 dividend",
        )
        assert eid > 0

        outstanding = await repo.get_outstanding_purification()
        assert len(outstanding) == 1
        assert outstanding[0]["symbol"] == "AAPL"  # uppercased
        assert outstanding[0]["purification_usd"] == 5.0
        assert outstanding[0]["paid_at"] is None
    finally:
        await engine.dispose()


async def test_mark_paid_moves_entry_out_of_outstanding(tmp_path):
    engine, repo = await _engine_repo(tmp_path)
    try:
        eid = await repo.record_purification(
            symbol="A", dividend_usd=100, haram_pct=0.05, purification_usd=5
        )
        ok = await repo.mark_purification_paid(eid)
        assert ok is True

        outstanding = await repo.get_outstanding_purification()
        assert outstanding == []

        totals = await repo.get_purification_totals()
        assert totals["outstanding_usd"] == 0.0
        assert totals["paid_usd"] == 5.0
    finally:
        await engine.dispose()


async def test_mark_paid_unknown_id_returns_false(tmp_path):
    engine, repo = await _engine_repo(tmp_path)
    try:
        ok = await repo.mark_purification_paid(9999)
        assert ok is False
    finally:
        await engine.dispose()


async def test_totals_sum_across_many_entries(tmp_path):
    engine, repo = await _engine_repo(tmp_path)
    try:
        for sym, div in [("A", 100), ("B", 200), ("C", 50)]:
            await repo.record_purification(
                symbol=sym, dividend_usd=div, haram_pct=0.10, purification_usd=div * 0.10
            )
        # Mark the second one paid.
        await repo.mark_purification_paid(2)
        totals = await repo.get_purification_totals()
        # Outstanding: 10 + 5 = 15; paid: 20.
        assert totals["outstanding_usd"] == 15.0
        assert totals["paid_usd"] == 20.0
    finally:
        await engine.dispose()


async def test_outstanding_sorted_newest_first(tmp_path):
    engine, repo = await _engine_repo(tmp_path)
    try:
        for sym in ("A", "B", "C"):
            await repo.record_purification(
                symbol=sym, dividend_usd=10, haram_pct=0.05, purification_usd=0.5
            )
        rows = await repo.get_outstanding_purification()
        # Most recent insert (id=3 = "C") should be first.
        assert rows[0]["symbol"] == "C"
        assert rows[-1]["symbol"] == "A"
    finally:
        await engine.dispose()
