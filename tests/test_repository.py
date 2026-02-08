"""Tests for the SQLite repository."""

import pytest

from halal_trader.db.models import init_db
from halal_trader.db.repository import Repository


@pytest.fixture
async def repo(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = await init_db(db_path)
    return Repository(db)


class TestTradeRecording:
    async def test_record_trade(self, repo):
        trade_id = await repo.record_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=190.50,
            order_id="test-order-123",
            status="submitted",
            llm_reasoning="Strong momentum",
        )
        assert trade_id is not None
        assert trade_id > 0

    async def test_update_trade_status(self, repo):
        trade_id = await repo.record_trade(
            symbol="AAPL", side="buy", quantity=10, status="submitted"
        )
        await repo.update_trade_status(trade_id, "filled", price=191.00)

        trades = await repo.get_recent_trades(limit=1)
        assert len(trades) == 1
        assert trades[0]["status"] == "filled"
        assert trades[0]["price"] == 191.00

    async def test_get_today_trades(self, repo):
        await repo.record_trade(symbol="AAPL", side="buy", quantity=10)
        await repo.record_trade(symbol="NVDA", side="sell", quantity=5)

        trades = await repo.get_today_trades()
        assert len(trades) == 2


class TestDailyPnL:
    async def test_start_and_end_day(self, repo):
        await repo.start_day(100000.0)
        await repo.end_day(
            ending_equity=101500.0,
            realized_pnl=1500.0,
            trades_count=5,
        )

        history = await repo.get_pnl_history(limit=1)
        assert len(history) == 1
        row = history[0]
        assert row["starting_equity"] == 100000.0
        assert row["ending_equity"] == 101500.0
        assert row["realized_pnl"] == 1500.0
        assert row["trades_count"] == 5
        assert abs(row["return_pct"] - 0.015) < 0.001


class TestLLMDecisions:
    async def test_record_decision(self, repo):
        decision_id = await repo.record_decision(
            provider="ollama",
            model="qwen2.5:32b",
            prompt_summary="Test analysis",
            raw_response='{"decisions": []}',
            parsed_action={"buys": 0, "sells": 0},
            symbols=["AAPL", "NVDA"],
            execution_ms=1500,
        )
        assert decision_id is not None
        assert decision_id > 0
