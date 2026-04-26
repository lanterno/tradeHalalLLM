"""CrossAssetAnalytics tests — switches between crypto and stock round-trips."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from halal_trader.core.analytics import CrossAssetAnalytics
from halal_trader.db import admin
from halal_trader.db.repository import Repository


async def _engine_repo(tmp_path):
    db_path = tmp_path / "x.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    head = admin.head()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await conn.execute(sa.text(f"INSERT INTO alembic_version (version_num) VALUES ('{head}')"))
    return engine, Repository(engine)


async def _seed_stock_round_trips(repo: Repository) -> None:
    """Two winning + one losing stock round-trip, all closed."""
    now = datetime.now(UTC)
    for symbol, entry, exit_p, side_pnl in [
        ("AAPL", 200.0, 210.0, "win"),
        ("MSFT", 420.0, 415.0, "loss"),
        ("GOOG", 150.0, 160.0, "win"),
    ]:
        tid = await repo.record_trade(symbol=symbol, side="buy", quantity=10, price=entry)
        # Bypass close_trade's UTC-now stamp so closed_at is in the
        # lookback window and our test is deterministic.
        from sqlalchemy.ext.asyncio import AsyncSession as _S
        from sqlmodel.ext.asyncio.session import AsyncSession

        from halal_trader.db.models import Trade

        async with AsyncSession(repo._engine) as session:
            trade = await session.get(Trade, tid)
            trade.filled_price = entry
            trade.exit_price = exit_p
            trade.exit_reason = "take_profit" if side_pnl == "win" else "stop_loss"
            trade.closed_at = now - timedelta(hours=1)
            trade.status = "closed"
            session.add(trade)
            await session.commit()
        del _S


async def test_stock_analytics_returns_zero_when_no_trades(tmp_path):
    _engine, repo = await _engine_repo(tmp_path)
    try:
        analytics = CrossAssetAnalytics(repo, asset_class="stock")
        stats = await analytics.compute_stats(lookback_days=7)
        assert stats.total_trades == 0
        assert stats.win_rate == 0
    finally:
        await _engine.dispose()


async def test_stock_analytics_aggregates_round_trips(tmp_path):
    engine, repo = await _engine_repo(tmp_path)
    try:
        await _seed_stock_round_trips(repo)
        analytics = CrossAssetAnalytics(repo, asset_class="stock")
        stats = await analytics.compute_stats(lookback_days=7)
        assert stats.total_trades == 3
        assert stats.wins == 2
        assert stats.losses == 1
        assert stats.win_rate > 0.6
        # AAPL gained 100, GOOG gained 100, MSFT lost 50 → net +150.
        assert stats.total_pnl == 150
        # Best pair has the highest cumulative pnl.
        assert stats.best_pair in {"AAPL", "GOOG"}
        assert stats.worst_pair == "MSFT"
    finally:
        await engine.dispose()


async def test_crypto_analytics_path_unchanged(tmp_path):
    """Default asset_class='crypto' must delegate to the original analytics."""
    engine, repo = await _engine_repo(tmp_path)
    try:
        analytics = CrossAssetAnalytics(repo)  # default crypto
        stats = await analytics.compute_stats(lookback_days=7)
        assert stats.total_trades == 0  # no crypto trades seeded
    finally:
        await engine.dispose()


async def test_format_for_prompt_renders_text(tmp_path):
    engine, repo = await _engine_repo(tmp_path)
    try:
        await _seed_stock_round_trips(repo)
        analytics = CrossAssetAnalytics(repo, asset_class="stock")
        stats = await analytics.compute_stats(lookback_days=7)
        text = analytics.format_for_prompt(stats)
        assert "Win rate" in text or "win" in text.lower()
    finally:
        await engine.dispose()
