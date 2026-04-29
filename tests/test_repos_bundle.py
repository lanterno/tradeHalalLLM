"""Tests for the per-table repo protocols + RepoBundle facade."""

from __future__ import annotations

from halal_trader.db.repos import RepoBundle


async def test_bundle_from_engine_round_trips_a_trade(engine) -> None:
    bundle = RepoBundle.from_engine(engine)
    trade_id = await bundle.trades.record_trade(
        symbol="AAPL",
        side="buy",
        quantity=10,
        price=200.0,
    )
    assert trade_id > 0
    rows = await bundle.trades.get_recent_trades(limit=10)
    assert any(r["id"] == trade_id for r in rows)


async def test_bundle_web_audit_helpers(engine) -> None:
    bundle = RepoBundle.from_engine(engine)
    aid = await bundle.web_audit.begin_web_action(
        actor="rid-1", method="POST", path="/api/admin/halt", payload="{}"
    )
    await bundle.web_audit.complete_web_action(aid, status_code=200)
    rows = await bundle.web_audit.get_recent_web_actions(limit=5)
    assert any(r["id"] == aid for r in rows)


async def test_bundle_runtime_config_round_trip(engine) -> None:
    bundle = RepoBundle.from_engine(engine)
    await bundle.runtime_config.set_runtime_config("CRYPTO_MAX_POSITION_PCT", 0.05)
    cfg = await bundle.runtime_config.list_runtime_config()
    assert cfg["CRYPTO_MAX_POSITION_PCT"] == 0.05
