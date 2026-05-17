"""Halal audit-receipt exporter tests."""

import json
from datetime import UTC, datetime

from halal_trader.db.models import CryptoTrade, HalalScreening, Trade
from halal_trader.db.repository import Repository
from halal_trader.halal import audit
from halal_trader.halal.audit import Receipt, build_receipt


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


# ── Pure-helper unit tests (no DB) ─────────────────────────────


def _stock_trade(**overrides) -> Trade:
    base = dict(
        id=1,
        symbol="AAPL",
        side="buy",
        quantity=10.0,
        price=180.0,
        timestamp=datetime(2026, 5, 1, 14, 30, tzinfo=UTC),
        halal_screening_id=42,
    )
    base.update(overrides)
    return Trade(**base)


def _crypto_trade(**overrides) -> CryptoTrade:
    base = dict(
        id=2,
        pair="BTCUSDT",
        side="buy",
        quantity=0.001,
        price=42_000.0,
        timestamp=datetime(2026, 5, 1, 14, 30, tzinfo=UTC),
        halal_screening_id=43,
    )
    base.update(overrides)
    return CryptoTrade(**base)


def _screening_obj(**overrides) -> HalalScreening:
    base = dict(
        id=42,
        symbol="AAPL",
        asset_class="stock",
        decision="HALAL",
        source="zoya",
        timestamp=datetime(2026, 5, 1, 14, 0, tzinfo=UTC),
        criteria='{"debt_ratio": 0.18, "interest_income": 0.02}',
    )
    base.update(overrides)
    return HalalScreening(**base)


def test_build_receipt_marks_stock_when_given_trade():
    receipt = build_receipt(_stock_trade(), _screening_obj())
    assert receipt.payload["asset_class"] == "stock"


def test_build_receipt_marks_crypto_when_given_crypto_trade():
    receipt = build_receipt(_crypto_trade(), _screening_obj(symbol="BTCUSDT"))
    assert receipt.payload["asset_class"] == "crypto"


def test_build_receipt_marks_unattested_without_screening():
    receipt = build_receipt(_stock_trade(halal_screening_id=None), None)
    assert receipt.payload["screening"] is None
    assert receipt.payload["compliance_status"] == "unattested"


def test_build_receipt_decodes_criteria_json():
    receipt = build_receipt(_stock_trade(), _screening_obj())
    crit = receipt.payload["screening"]["criteria"]
    assert isinstance(crit, dict)
    assert crit["debt_ratio"] == 0.18


def test_build_receipt_keeps_criteria_as_string_when_not_json():
    receipt = build_receipt(_stock_trade(), _screening_obj(criteria="not-json"))
    assert receipt.payload["screening"]["criteria"] == "not-json"


def test_receipt_to_json_default_serializer_handles_datetimes():
    r = Receipt(payload={"now": datetime(2026, 5, 1, tzinfo=UTC)})
    parsed = json.loads(r.to_json())
    assert "2026-05-01" in parsed["now"]
