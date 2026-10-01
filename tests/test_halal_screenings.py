"""Halal screening audit FK — repository round-trip + trade linkage."""

import sqlalchemy as sa

from halal_trader.db.repository import Repository


async def test_record_screening_round_trip(engine):
    repo = Repository(engine)
    sid = await repo.record_halal_screening(
        symbol="AAPL",
        asset_class="stock",
        source="zoya",
        decision="halal",
        criteria={"market_cap_usd": 3_000_000_000_000, "category": "technology"},
        cache_hit=False,
    )
    assert sid > 0

    loaded = await repo.get_halal_screening(sid)
    assert loaded is not None
    assert loaded["symbol"] == "AAPL"
    assert loaded["decision"] == "halal"
    assert loaded["source"] == "zoya"
    assert loaded["cache_hit"] is False
    assert loaded["criteria"]["category"] == "technology"


async def test_stock_trade_carries_screening_fk(engine):
    repo = Repository(engine)
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


async def test_screening_id_is_optional_for_back_compat(engine):
    """Existing call sites that don't pass screening_id must keep working."""
    repo = Repository(engine)
    trade_id = await repo.record_trade(
        symbol="AAPL",
        side="buy",
        quantity=10,
        price=200.0,
    )
    async with engine.begin() as conn:
        row = await conn.execute(
            sa.text("SELECT halal_screening_id FROM trades WHERE id = :i"),
            {"i": trade_id},
        )
        assert row.scalar_one() is None
