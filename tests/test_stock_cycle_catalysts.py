"""Stock cycle's catalyst-feed wiring."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from halal_trader.trading.catalysts import Catalyst, StockCatalystFeed
from halal_trader.trading.cycle import TradingCycleService


def _bare_cycle(catalyst_feed=None) -> TradingCycleService:
    return TradingCycleService(
        broker=MagicMock(),
        screener=MagicMock(),
        strategy=MagicMock(),
        executor=MagicMock(),
        portfolio=MagicMock(),
        catalyst_feed=catalyst_feed,
    )


async def test_no_feed_returns_empty_string():
    cycle = _bare_cycle()
    assert await cycle._gather_catalysts(["AAPL"]) == ""


async def test_feed_with_results_renders_for_prompt():
    src = MagicMock()
    src.fetch = AsyncMock(
        return_value=[
            Catalyst(
                symbol="AAPL",
                kind="news",
                title="Apple beats Q1",
                timestamp=datetime.now(timezone.utc),
                sentiment="positive",
                source="Bloomberg",
            )
        ]
    )
    feed = StockCatalystFeed([src])
    cycle = _bare_cycle(catalyst_feed=feed)
    text = await cycle._gather_catalysts(["AAPL"])
    assert "Apple beats Q1" in text


async def test_feed_exception_returns_empty():
    feed = MagicMock()
    feed.fetch_all = AsyncMock(side_effect=RuntimeError("boom"))
    cycle = _bare_cycle(catalyst_feed=feed)
    assert await cycle._gather_catalysts(["AAPL"]) == ""


async def test_no_matching_symbols_returns_empty():
    src = MagicMock()
    src.fetch = AsyncMock(
        return_value=[
            Catalyst(
                symbol="DOGE",  # not in our halal universe
                kind="news",
                title="dog news",
                timestamp=datetime.now(timezone.utc),
            )
        ]
    )
    feed = StockCatalystFeed([src])
    cycle = _bare_cycle(catalyst_feed=feed)
    assert await cycle._gather_catalysts(["AAPL"]) == ""
