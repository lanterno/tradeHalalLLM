"""Halal screening audit FK — repository round-trip + trade linkage."""

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from halal_trader.db import admin
from halal_trader.db.repository import Repository


async def _make_repo(tmp_path):
    db_path = tmp_path / "halal.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    head = admin.head()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await conn.execute(sa.text(f"INSERT INTO alembic_version (version_num) VALUES ('{head}')"))
    return Repository(engine), engine


async def test_record_screening_round_trip(tmp_path):
    repo, engine = await _make_repo(tmp_path)
    try:
        sid = await repo.record_halal_screening(
            symbol="BTC",
            asset_class="crypto",
            source="coingecko_rules",
            decision="halal",
            criteria={"market_cap_usd": 1_400_000_000_000, "category": "layer-1"},
            cache_hit=False,
        )
        assert sid > 0

        loaded = await repo.get_halal_screening(sid)
        assert loaded is not None
        assert loaded["symbol"] == "BTC"
        assert loaded["decision"] == "halal"
        assert loaded["source"] == "coingecko_rules"
        assert loaded["cache_hit"] is False
        assert loaded["criteria"]["category"] == "layer-1"
    finally:
        await engine.dispose()


async def test_crypto_trade_carries_screening_fk(tmp_path):
    repo, engine = await _make_repo(tmp_path)
    try:
        sid = await repo.record_halal_screening(
            symbol="ETH",
            asset_class="crypto",
            source="cache",
            decision="halal",
            cache_hit=True,
        )
        trade_id = await repo.record_crypto_trade(
            pair="ETHUSDT",
            side="buy",
            quantity=0.1,
            price=3500.0,
            halal_screening_id=sid,
        )
        # Read back via raw SQL — repository keeps no get_trade_by_id today.
        async with engine.begin() as conn:
            row = await conn.execute(
                sa.text("SELECT halal_screening_id FROM crypto_trades WHERE id = :i"),
                {"i": trade_id},
            )
            assert row.scalar_one() == sid
    finally:
        await engine.dispose()


async def test_stock_trade_carries_screening_fk(tmp_path):
    repo, engine = await _make_repo(tmp_path)
    try:
        sid = await repo.record_halal_screening(
            symbol="AAPL",
            asset_class="stock",
            source="zoya",
            decision="halal",
        )
        trade_id = await repo.record_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=200.0,
            halal_screening_id=sid,
        )
        async with engine.begin() as conn:
            row = await conn.execute(
                sa.text("SELECT halal_screening_id FROM trades WHERE id = :i"),
                {"i": trade_id},
            )
            assert row.scalar_one() == sid
    finally:
        await engine.dispose()


async def test_screening_id_is_optional_for_back_compat(tmp_path):
    """Existing call sites that don't pass screening_id must keep working."""
    repo, engine = await _make_repo(tmp_path)
    try:
        trade_id = await repo.record_crypto_trade(
            pair="BTCUSDT",
            side="buy",
            quantity=0.001,
            price=70000.0,
        )
        async with engine.begin() as conn:
            row = await conn.execute(
                sa.text("SELECT halal_screening_id FROM crypto_trades WHERE id = :i"),
                {"i": trade_id},
            )
            assert row.scalar_one() is None
    finally:
        await engine.dispose()
