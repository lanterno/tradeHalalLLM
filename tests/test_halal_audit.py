"""Halal audit-receipt exporter tests."""

import json

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from halal_trader.db import admin
from halal_trader.db.repository import Repository
from halal_trader.halal import audit


async def _make_repo(tmp_path):
    db_path = tmp_path / "audit.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    head = admin.head()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await conn.execute(sa.text(f"INSERT INTO alembic_version (version_num) VALUES ('{head}')"))
    return Repository(engine), engine


async def test_export_receipt_returns_none_for_unknown_trade(tmp_path):
    _repo, engine = await _make_repo(tmp_path)
    try:
        assert await audit.export_receipt(engine, trade_id=9999, asset_class="crypto") is None
    finally:
        await engine.dispose()


async def test_export_receipt_joins_screening(tmp_path):
    repo, engine = await _make_repo(tmp_path)
    try:
        sid = await repo.record_halal_screening(
            symbol="BTC",
            asset_class="crypto",
            source="coingecko_rules",
            decision="halal",
            criteria={"market_cap_usd": 1_400_000_000_000},
        )
        trade_id = await repo.record_crypto_trade(
            pair="BTCUSDT",
            side="buy",
            quantity=0.01,
            price=70_000.0,
            halal_screening_id=sid,
        )
        receipt = await audit.export_receipt(engine, trade_id=trade_id, asset_class="crypto")
        assert receipt is not None
        assert receipt.payload["compliance_status"] == "halal"
        assert receipt.payload["asset_class"] == "crypto"
        assert receipt.payload["trade"]["pair"] == "BTCUSDT"
        # criteria comes back as a parsed dict, not a JSON string.
        assert receipt.payload["screening"]["criteria"]["market_cap_usd"] == 1_400_000_000_000
    finally:
        await engine.dispose()


async def test_export_receipt_marks_legacy_trade_as_unattested(tmp_path):
    """Trades without a screening FK get an explicit ``unattested`` status."""
    repo, engine = await _make_repo(tmp_path)
    try:
        trade_id = await repo.record_crypto_trade(
            pair="ETHUSDT", side="buy", quantity=0.1, price=3500.0
        )
        receipt = await audit.export_receipt(engine, trade_id=trade_id, asset_class="crypto")
        assert receipt is not None
        assert receipt.payload["compliance_status"] == "unattested"
        assert receipt.payload["screening"] is None
    finally:
        await engine.dispose()


async def test_export_receipt_to_json_serialises_datetimes(tmp_path):
    repo, engine = await _make_repo(tmp_path)
    try:
        trade_id = await repo.record_crypto_trade(
            pair="BTCUSDT", side="buy", quantity=0.01, price=70_000.0
        )
        receipt = await audit.export_receipt(engine, trade_id=trade_id, asset_class="crypto")
        as_json = receipt.to_json()
        # Must round-trip — i.e. all values must be JSON-serialisable.
        parsed = json.loads(as_json)
        assert parsed["trade"]["pair"] == "BTCUSDT"
    finally:
        await engine.dispose()


async def test_export_for_symbol_paginates_and_links_screenings(tmp_path):
    repo, engine = await _make_repo(tmp_path)
    try:
        sid = await repo.record_halal_screening(
            symbol="AAPL", asset_class="stock", source="zoya", decision="halal"
        )
        for _ in range(3):
            await repo.record_trade(
                symbol="AAPL", side="buy", quantity=10, price=200.0, halal_screening_id=sid
            )
        # Throw in a non-AAPL trade that should not appear in results.
        await repo.record_trade(symbol="MSFT", side="buy", quantity=5, price=420.0)

        receipts = await audit.export_for_symbol(
            engine, symbol="AAPL", asset_class="stock", limit=10
        )
        assert len(receipts) == 3
        for r in receipts:
            assert r.payload["trade"]["symbol"] == "AAPL"
            assert r.payload["compliance_status"] == "halal"
    finally:
        await engine.dispose()
