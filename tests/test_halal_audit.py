"""Halal audit-receipt exporter tests."""

import json

from halal_trader.db.repository import Repository
from halal_trader.halal import audit


async def test_export_receipt_returns_none_for_unknown_trade(engine):
    assert await audit.export_receipt(engine, trade_id=9999, asset_class="crypto") is None


async def test_export_receipt_joins_screening(engine):
    repo = Repository(engine)
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
    assert receipt.payload["screening"]["criteria"]["market_cap_usd"] == 1_400_000_000_000


async def test_export_receipt_marks_legacy_trade_as_unattested(engine):
    """Trades without a screening FK get an explicit ``unattested`` status."""
    repo = Repository(engine)
    trade_id = await repo.record_crypto_trade(
        pair="ETHUSDT", side="buy", quantity=0.1, price=3500.0
    )
    receipt = await audit.export_receipt(engine, trade_id=trade_id, asset_class="crypto")
    assert receipt is not None
    assert receipt.payload["compliance_status"] == "unattested"
    assert receipt.payload["screening"] is None


async def test_export_receipt_to_json_serialises_datetimes(engine):
    repo = Repository(engine)
    trade_id = await repo.record_crypto_trade(
        pair="BTCUSDT", side="buy", quantity=0.01, price=70_000.0
    )
    receipt = await audit.export_receipt(engine, trade_id=trade_id, asset_class="crypto")
    as_json = receipt.to_json()
    parsed = json.loads(as_json)
    assert parsed["trade"]["pair"] == "BTCUSDT"


async def test_export_for_symbol_paginates_and_links_screenings(engine):
    repo = Repository(engine)
    sid = await repo.record_halal_screening(
        symbol="AAPL", asset_class="stock", source="zoya", decision="halal"
    )
    for _ in range(3):
        await repo.record_trade(
            symbol="AAPL", side="buy", quantity=10, price=200.0, halal_screening_id=sid
        )
    await repo.record_trade(symbol="MSFT", side="buy", quantity=5, price=420.0)

    receipts = await audit.export_for_symbol(engine, symbol="AAPL", asset_class="stock", limit=10)
    assert len(receipts) == 3
    for r in receipts:
        assert r.payload["trade"]["symbol"] == "AAPL"
        assert r.payload["compliance_status"] == "halal"
