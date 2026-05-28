"""Executor must hard-block BUYs of recently-closed symbols.

Loop observation, 2026-05-21 cycles 14:00 → 14:30: pathological
CSCO ping-pong (4 transactions in 90 min on the same symbol). Three
escalating prompt-level warnings — rule 8 transaction-cost, RECENTLY
CLOSED block, "FOMC volatility is NOT a thesis change" — were all
visibly ignored by the LLM. The hard-rule fix: refuse a BUY at the
executor for any symbol whose `closed_at` is within the configured
cooldown window. No LLM judgment, no escape hatch.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.domain.models import Account, TradeAction, TradeDecision
from halal_trader.trading.executor import TradeExecutor


def _account() -> Account:
    return Account(
        equity=100_000,
        buying_power=100_000,
        cash=100_000,
        portfolio_value=100_000,
        status="ACTIVE",
    )


def _decision(symbol: str = "CSCO", quantity: int = 100):
    return TradeDecision(
        action=TradeAction.BUY,
        symbol=symbol,
        quantity=quantity,
        confidence=0.8,
        reasoning="t",
    )


def _executor(repo, *, cooldown_minutes: int = 30, sl_cooldown_minutes: int = 120) -> TradeExecutor:
    broker = MagicMock()
    broker.get_account_info = AsyncMock(return_value=_account())
    broker.get_stock_snapshot = AsyncMock(return_value={"latest_trade": {"price": 100.0}})
    broker.place_order = AsyncMock(return_value={"id": "x", "status": "filled"})
    broker.get_order_by_id = AsyncMock(
        return_value={
            "id": "x",
            "status": "filled",
            "filled_qty": "100",
            "filled_avg_price": "100",
            "filled_at": "2026-05-21T19:00:00Z",
        }
    )
    return TradeExecutor(
        broker,
        repo,
        max_position_pct=1.0,
        max_simultaneous_positions=10,
        max_sector_pct=0,
        recent_close_cooldown_minutes=cooldown_minutes,
        stop_loss_reentry_cooldown_minutes=sl_cooldown_minutes,
    )


@pytest.mark.asyncio
async def test_buy_blocked_within_cooldown_window():
    """A symbol closed 15 min ago must be rejected by the 30-min cooldown."""
    repo = MagicMock()
    repo.get_recently_closed = AsyncMock(
        return_value=[
            {"symbol": "CSCO", "closed_at": datetime.now(UTC) - timedelta(minutes=15)},
        ]
    )
    repo.record_trade = AsyncMock()
    executor = _executor(repo)

    result = await executor._execute_buy(_decision("CSCO"), positions=[])

    assert result["status"] == "rejected"
    assert "cooldown" in result["reason"].lower()
    assert "CSCO" in result["reason"]
    # Critical: no broker calls when the cooldown blocks — saves an MCP round-trip.
    executor._broker.get_account_info.assert_not_called()
    executor._broker.place_order.assert_not_called()
    repo.record_trade.assert_not_called()


@pytest.mark.asyncio
async def test_buy_allowed_outside_cooldown_window():
    """A symbol closed 45 min ago is past the 30-min cooldown → BUY proceeds."""
    repo = MagicMock()
    repo.get_recently_closed = AsyncMock(
        return_value=[
            {"symbol": "CSCO", "closed_at": datetime.now(UTC) - timedelta(minutes=45)},
        ]
    )
    repo.record_trade = AsyncMock(return_value=42)
    executor = _executor(repo)

    result = await executor._execute_buy(_decision("CSCO"), positions=[])

    assert result["status"] != "rejected" or "cooldown" not in result.get("reason", "")


@pytest.mark.asyncio
async def test_other_symbols_not_affected():
    """The cooldown only blocks the symbol with a recent close, not the whole universe."""
    repo = MagicMock()
    repo.get_recently_closed = AsyncMock(
        return_value=[
            {"symbol": "CSCO", "closed_at": datetime.now(UTC) - timedelta(minutes=5)},
        ]
    )
    repo.record_trade = AsyncMock(return_value=43)
    executor = _executor(repo)

    # MSFT was not closed recently → goes through normally
    result = await executor._execute_buy(_decision("MSFT"), positions=[])

    assert result["status"] != "rejected" or "cooldown" not in result.get("reason", "")


@pytest.mark.asyncio
async def test_cooldown_zero_disables_check():
    """Operator escape hatch: cooldown=0 means no blocking."""
    repo = MagicMock()
    repo.get_recently_closed = AsyncMock(
        return_value=[
            {"symbol": "CSCO", "closed_at": datetime.now(UTC) - timedelta(minutes=1)},
        ]
    )
    repo.record_trade = AsyncMock(return_value=44)
    # Disable BOTH re-entry gates (recent-close + stop-loss) for the
    # full escape-hatch behaviour.
    executor = _executor(repo, cooldown_minutes=0, sl_cooldown_minutes=0)

    result = await executor._execute_buy(_decision("CSCO"), positions=[])

    # Cooldown disabled → not rejected by cooldown (may still be rejected
    # by other gates, but the cooldown is bypassed).
    assert "cooldown" not in result.get("reason", "")
    # The repo lookup is skipped entirely when both gates are <= 0.
    repo.get_recently_closed.assert_not_called()


@pytest.mark.asyncio
async def test_iso_string_closed_at_parsed():
    """`closed_at` from a `model_dump()` row is an ISO string — the cooldown
    helper must parse both shapes."""
    repo = MagicMock()
    iso_15_min_ago = (datetime.now(UTC) - timedelta(minutes=15)).isoformat()
    repo.get_recently_closed = AsyncMock(
        return_value=[{"symbol": "CSCO", "closed_at": iso_15_min_ago}]
    )
    repo.record_trade = AsyncMock()
    executor = _executor(repo)

    result = await executor._execute_buy(_decision("CSCO"), positions=[])
    assert result["status"] == "rejected"
    assert "cooldown" in result["reason"].lower()


@pytest.mark.asyncio
async def test_repo_error_does_not_block_trade():
    """A DB blip on the cooldown lookup must NOT prevent a legitimate
    trade — degrade to allow."""
    repo = MagicMock()
    repo.get_recently_closed = AsyncMock(side_effect=RuntimeError("db down"))
    repo.record_trade = AsyncMock(return_value=45)
    executor = _executor(repo)

    result = await executor._execute_buy(_decision("CSCO"), positions=[])

    # Trade should proceed (status != rejected by cooldown).
    assert "cooldown" not in result.get("reason", "")


@pytest.mark.asyncio
async def test_recent_sell_without_closed_at_blocks_buy():
    """A SELL row from the window must block re-buy even if its
    underlying BUY's `closed_at` is still NULL (legacy data from
    before the close-on-sell fix, or in-flight lag)."""
    repo = MagicMock()
    repo.get_recently_closed = AsyncMock(return_value=[])  # no closed BUYs
    sell_ts = datetime.now(UTC) - timedelta(minutes=12)
    repo.get_recent_sells = AsyncMock(
        return_value=[{"symbol": "CSCO", "side": "sell", "timestamp": sell_ts}]
    )
    repo.record_trade = AsyncMock()
    executor = _executor(repo)

    result = await executor._execute_buy(_decision("CSCO"), positions=[])

    assert result["status"] == "rejected"
    assert "cooldown" in result["reason"].lower()
    # Latest exit was a SELL — reason should reflect that vocabulary.
    assert "sold" in result["reason"].lower()
    executor._broker.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_uses_latest_exit_across_closed_and_sold():
    """Both sources have a hit; the cooldown uses whichever is more recent."""
    repo = MagicMock()
    # closed 25 min ago, sold 8 min ago → SELL is more recent
    repo.get_recently_closed = AsyncMock(
        return_value=[
            {"symbol": "CSCO", "closed_at": datetime.now(UTC) - timedelta(minutes=25)},
        ]
    )
    sell_ts2 = datetime.now(UTC) - timedelta(minutes=8)
    repo.get_recent_sells = AsyncMock(
        return_value=[{"symbol": "CSCO", "side": "sell", "timestamp": sell_ts2}]
    )
    repo.record_trade = AsyncMock()
    executor = _executor(repo)

    result = await executor._execute_buy(_decision("CSCO"), positions=[])
    assert result["status"] == "rejected"
    # Should report 8 min (the SELL), not 25 min (the close).
    assert "8 min ago" in result["reason"] or "9 min ago" in result["reason"]
    assert "sold" in result["reason"].lower()


@pytest.mark.asyncio
async def test_picks_most_recent_close_when_symbol_appears_twice():
    """If the symbol shows up multiple times (e.g. bought + closed twice
    in the window), the cooldown uses the LATEST close timestamp."""
    repo = MagicMock()
    repo.get_recently_closed = AsyncMock(
        return_value=[
            {"symbol": "CSCO", "closed_at": datetime.now(UTC) - timedelta(minutes=45)},
            {"symbol": "CSCO", "closed_at": datetime.now(UTC) - timedelta(minutes=10)},
        ]
    )
    repo.record_trade = AsyncMock()
    executor = _executor(repo)

    result = await executor._execute_buy(_decision("CSCO"), positions=[])
    # 10 min ago is INSIDE the 30-min cooldown — must reject.
    assert result["status"] == "rejected"
    assert "cooldown" in result["reason"].lower()
    # Reported gap should reflect the LATEST close (10 min), not the 45-min one.
    assert "10 min ago" in result["reason"] or "11 min ago" in result["reason"]


# ── stop-loss re-entry gate (falling-knife guard) ───────────────


@pytest.mark.asyncio
async def test_stop_loss_reentry_blocked_past_normal_cooldown():
    """A symbol stopped out 60 min ago is PAST the 30-min recent-close
    cooldown but still inside the 120-min stop-loss gate → blocked.
    This is the MSFT falling-knife loop the gate exists to stop."""
    repo = MagicMock()
    repo.get_recently_closed = AsyncMock(
        return_value=[
            {
                "symbol": "MSFT",
                "exit_reason": "stop_loss",
                "closed_at": datetime.now(UTC) - timedelta(minutes=60),
            },
        ]
    )
    repo.get_recent_sells = AsyncMock(return_value=[])
    repo.record_trade = AsyncMock()
    executor = _executor(repo)  # default stop_loss_reentry_cooldown=120

    result = await executor._execute_buy(_decision("MSFT"), positions=[])

    assert result["status"] == "rejected"
    assert "stop-loss re-entry gate" in result["reason"]
    executor._broker.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_stop_loss_reentry_allowed_after_gate_window():
    """Past the 120-min gate → re-entry allowed again."""
    repo = MagicMock()
    repo.get_recently_closed = AsyncMock(
        return_value=[
            {
                "symbol": "MSFT",
                "exit_reason": "stop_loss",
                "closed_at": datetime.now(UTC) - timedelta(minutes=130),
            },
        ]
    )
    repo.get_recent_sells = AsyncMock(return_value=[])
    repo.record_trade = AsyncMock(return_value=1)
    executor = _executor(repo)

    result = await executor._execute_buy(_decision("MSFT"), positions=[])
    assert "stop-loss re-entry gate" not in result.get("reason", "")


@pytest.mark.asyncio
async def test_llm_sell_does_not_trigger_stop_loss_gate():
    """An LLM-chosen sell (not a stop-out) only gets the short reason-
    agnostic cooldown — past 30 min it's allowed, the 120-min gate does
    NOT apply."""
    repo = MagicMock()
    repo.get_recently_closed = AsyncMock(
        return_value=[
            {
                "symbol": "GOOG",
                "exit_reason": "llm_sell",
                "closed_at": datetime.now(UTC) - timedelta(minutes=45),
            },
        ]
    )
    repo.get_recent_sells = AsyncMock(return_value=[])
    repo.record_trade = AsyncMock(return_value=1)
    executor = _executor(repo)

    result = await executor._execute_buy(_decision("GOOG"), positions=[])
    assert "stop-loss re-entry gate" not in result.get("reason", "")


@pytest.mark.asyncio
async def test_stop_loss_reentry_zero_disables_gate():
    """stop_loss_reentry_cooldown=0 disables the gate (escape hatch)."""
    repo = MagicMock()
    repo.get_recently_closed = AsyncMock(
        return_value=[
            {
                "symbol": "MSFT",
                "exit_reason": "stop_loss",
                "closed_at": datetime.now(UTC) - timedelta(minutes=5),
            },
        ]
    )
    repo.get_recent_sells = AsyncMock(return_value=[])
    repo.record_trade = AsyncMock(return_value=1)
    broker = MagicMock()
    broker.get_account_info = AsyncMock(return_value=_account())
    broker.get_stock_snapshot = AsyncMock(return_value={"latest_trade": {"price": 100.0}})
    broker.place_order = AsyncMock(return_value={"id": "x", "status": "filled"})
    broker.get_order_by_id = AsyncMock(
        return_value={"id": "x", "status": "filled", "filled_qty": "100", "filled_avg_price": "100"}
    )
    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10,
        max_sector_pct=0, recent_close_cooldown_minutes=0,
        stop_loss_reentry_cooldown_minutes=0,
    )

    result = await executor._execute_buy(_decision("MSFT"), positions=[])
    assert "stop-loss re-entry gate" not in result.get("reason", "")
