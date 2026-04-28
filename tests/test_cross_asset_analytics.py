"""CrossAssetAnalytics tests — switches between crypto and stock round-trips."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.core.analytics import CrossAssetAnalytics
from halal_trader.db.models import Trade
from halal_trader.db.repository import Repository


async def _seed_stock_round_trips(engine, repo: Repository) -> None:
    """Two winning + one losing stock round-trip, all closed."""
    now = datetime.now(UTC)
    for symbol, entry, exit_p, side_pnl in [
        ("AAPL", 200.0, 210.0, "win"),
        ("MSFT", 420.0, 415.0, "loss"),
        ("GOOG", 150.0, 160.0, "win"),
    ]:
        tid = await repo.record_trade(symbol=symbol, side="buy", quantity=10, price=entry)
        async with AsyncSession(engine) as session:
            trade = await session.get(Trade, tid)
            trade.filled_price = entry
            trade.exit_price = exit_p
            trade.exit_reason = "take_profit" if side_pnl == "win" else "stop_loss"
            trade.closed_at = now - timedelta(hours=1)
            trade.status = "closed"
            session.add(trade)
            await session.commit()


async def test_stock_analytics_returns_zero_when_no_trades(engine):
    repo = Repository(engine)
    analytics = CrossAssetAnalytics(repo, asset_class="stock")
    stats = await analytics.compute_stats(lookback_days=7)
    assert stats.total_trades == 0
    assert stats.win_rate == 0


async def test_stock_analytics_aggregates_round_trips(engine):
    repo = Repository(engine)
    await _seed_stock_round_trips(engine, repo)
    analytics = CrossAssetAnalytics(repo, asset_class="stock")
    stats = await analytics.compute_stats(lookback_days=7)
    assert stats.total_trades == 3
    assert stats.wins == 2
    assert stats.losses == 1
    assert stats.win_rate > 0.6
    assert stats.total_pnl == 150
    assert stats.best_pair in {"AAPL", "GOOG"}
    assert stats.worst_pair == "MSFT"


async def test_crypto_analytics_path_unchanged(engine):
    """Default asset_class='crypto' must delegate to the original analytics."""
    repo = Repository(engine)
    analytics = CrossAssetAnalytics(repo)
    stats = await analytics.compute_stats(lookback_days=7)
    assert stats.total_trades == 0


async def test_format_for_prompt_renders_text(engine):
    repo = Repository(engine)
    await _seed_stock_round_trips(engine, repo)
    analytics = CrossAssetAnalytics(repo, asset_class="stock")
    stats = await analytics.compute_stats(lookback_days=7)
    text = analytics.format_for_prompt(stats)
    assert "Win rate" in text or "win" in text.lower()
