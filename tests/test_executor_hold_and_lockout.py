"""Two new hard rules added 2026-05-22 night to reduce churn:

* ``_check_min_hold`` — block LLM SELLs of positions younger than
  ``min_hold_minutes``. Symmetric to the buy-side recent-close cooldown.
  Monitor-driven SL/TP exits use a separate path (``close_trade``) and
  are unaffected.
* ``_check_market_close_lockout`` — refuse new BUYs in the last N min
  before market close so EOD reconciliation isn't holding bag-positions.

Observed yesterday: AVGO 15-min flip (cycle-166585c8), the final
hour of trades couldn't be managed before EOD forced close.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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


def _broker():
    b = MagicMock()
    b.get_account_info = AsyncMock(return_value=_account())
    b.get_stock_snapshot = AsyncMock(return_value={"latest_trade": {"price": 100.0}})
    b.place_order = AsyncMock(return_value={"id": "x", "status": "filled"})
    b.get_order_by_id = AsyncMock(
        return_value={
            "id": "x",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "100",
            "filled_at": "2026-05-22T15:00:00Z",
        }
    )
    return b


def _decision_buy(symbol: str = "AAPL", quantity: int = 10):
    return TradeDecision(
        action=TradeAction.BUY, symbol=symbol, quantity=quantity, confidence=0.8, reasoning="t"
    )


def _decision_sell(symbol: str = "AAPL", quantity: int = 10):
    return TradeDecision(
        action=TradeAction.SELL, symbol=symbol, quantity=quantity, confidence=0.8, reasoning="exit"
    )


def _executor(
    repo,
    *,
    min_hold: int = 30,
    close_lockout: int = 30,
    cooldown: int = 30,
):
    return TradeExecutor(
        _broker(),
        repo,
        max_position_pct=1.0,
        max_simultaneous_positions=10,
        max_sector_pct=0,
        recent_close_cooldown_minutes=cooldown,
        min_hold_minutes=min_hold,
        no_new_positions_minutes_before_close=close_lockout,
    )


# ── _check_min_hold ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sell_blocked_when_youngest_buy_is_fresh():
    """A BUY opened 15 min ago must block a SELL when min_hold=30."""
    repo = MagicMock()
    open_trade = SimpleNamespace(
        symbol="AAPL",
        side="buy",
        timestamp=datetime.now(UTC) - timedelta(minutes=15),
    )
    repo.get_open_trades = AsyncMock(return_value=[open_trade])
    repo.record_trade = AsyncMock()
    executor = _executor(repo, cooldown=0)

    result = await executor._execute_sell(_decision_sell("AAPL"))

    assert result["status"] == "rejected"
    assert "min-hold" in result["reason"].lower()
    assert "15 min" in result["reason"]
    repo.record_trade.assert_not_called()


@pytest.mark.asyncio
async def test_sell_allowed_when_youngest_buy_past_hold():
    """A BUY opened 45 min ago is past the 30-min hold → sell proceeds."""
    repo = MagicMock()
    open_trade = SimpleNamespace(
        symbol="AAPL",
        side="buy",
        timestamp=datetime.now(UTC) - timedelta(minutes=45),
    )
    repo.get_open_trades = AsyncMock(return_value=[open_trade])
    repo.record_trade = AsyncMock(return_value=1)
    repo.close_open_trades_for_symbol = AsyncMock(return_value=1)
    executor = _executor(repo, cooldown=0)

    result = await executor._execute_sell(_decision_sell("AAPL"))
    assert result["status"] != "rejected"


@pytest.mark.asyncio
async def test_sell_blocked_uses_youngest_buy_when_multiple_open():
    """Multiple BUYs open: youngest one determines the gate."""
    repo = MagicMock()
    old_ts = datetime.now(UTC) - timedelta(hours=2)
    young_ts = datetime.now(UTC) - timedelta(minutes=10)
    repo.get_open_trades = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="AAPL", side="buy", timestamp=old_ts),
            SimpleNamespace(symbol="AAPL", side="buy", timestamp=young_ts),
        ]
    )
    repo.record_trade = AsyncMock()
    executor = _executor(repo, cooldown=0)

    result = await executor._execute_sell(_decision_sell("AAPL"))
    assert result["status"] == "rejected"
    # Reported age should be the YOUNGEST (10 min), not 2h.
    assert "10 min" in result["reason"]


@pytest.mark.asyncio
async def test_min_hold_zero_disables():
    """Operator escape: min_hold=0 → check skipped entirely."""
    repo = MagicMock()
    repo.get_open_trades = AsyncMock()  # should NOT be called
    repo.record_trade = AsyncMock()
    repo.close_open_trades_for_symbol = AsyncMock(return_value=0)
    executor = _executor(repo, min_hold=0, cooldown=0)

    result = await executor._execute_sell(_decision_sell("AAPL"))
    # Not rejected by min-hold (may be rejected by other gates).
    assert "min-hold" not in result.get("reason", "").lower()
    repo.get_open_trades.assert_not_called()


@pytest.mark.asyncio
async def test_min_hold_other_symbol_not_affected():
    """A young AAPL BUY shouldn't block a SELL of MSFT."""
    repo = MagicMock()
    aapl_ts = datetime.now(UTC) - timedelta(minutes=10)
    repo.get_open_trades = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="AAPL", side="buy", timestamp=aapl_ts),
        ]
    )
    repo.record_trade = AsyncMock(return_value=1)
    repo.close_open_trades_for_symbol = AsyncMock(return_value=0)
    executor = _executor(repo, cooldown=0)

    result = await executor._execute_sell(_decision_sell("MSFT"))
    assert result["status"] != "rejected" or "min-hold" not in result.get("reason", "").lower()


# ── _check_market_close_lockout ─────────────────────────────────


def _at_et(hour: int, minute: int):
    """Return a tz-aware ET datetime with hour/minute on a trading day."""
    from halal_trader.market_hours import MARKET_TZ

    # Pick a fixed Tuesday so weekend / holiday checks pass.
    return datetime(2026, 5, 19, hour, minute, tzinfo=MARKET_TZ)


@pytest.mark.asyncio
async def test_buy_blocked_in_lockout_window():
    """At 15:45 ET (15 min to 16:00 close), BUYs are rejected."""
    repo = MagicMock()
    repo.get_recently_closed = AsyncMock(return_value=[])
    repo.get_recent_sells = AsyncMock(return_value=[])
    repo.record_trade = AsyncMock()
    executor = _executor(repo)

    with patch(
        "halal_trader.market_hours.now_eastern",
        return_value=_at_et(15, 45),
    ):
        result = await executor._execute_buy(_decision_buy("AAPL"), positions=[])

    assert result["status"] == "rejected"
    assert "close lockout" in result["reason"].lower()
    assert "15 min" in result["reason"] or "14 min" in result["reason"]
    executor._broker.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_buy_allowed_outside_lockout_window():
    """At 15:00 ET (60 min to close), well outside the 30-min lockout."""
    repo = MagicMock()
    repo.get_recently_closed = AsyncMock(return_value=[])
    repo.get_recent_sells = AsyncMock(return_value=[])
    repo.record_trade = AsyncMock(return_value=1)
    executor = _executor(repo)

    with patch(
        "halal_trader.market_hours.now_eastern",
        return_value=_at_et(15, 0),
    ):
        result = await executor._execute_buy(_decision_buy("AAPL"), positions=[])

    # Not rejected by the lockout (may pass through to the broker which fills).
    assert "close lockout" not in result.get("reason", "").lower()


@pytest.mark.asyncio
async def test_lockout_zero_disables():
    """Operator escape: no_new_positions_minutes_before_close=0 → no lockout."""
    repo = MagicMock()
    repo.get_recently_closed = AsyncMock(return_value=[])
    repo.get_recent_sells = AsyncMock(return_value=[])
    repo.record_trade = AsyncMock(return_value=1)
    executor = _executor(repo, close_lockout=0)

    with patch(
        "halal_trader.market_hours.now_eastern",
        return_value=_at_et(15, 59),  # 1 min before close
    ):
        result = await executor._execute_buy(_decision_buy("AAPL"), positions=[])

    assert "close lockout" not in result.get("reason", "").lower()


@pytest.mark.asyncio
async def test_lockout_does_not_block_sells():
    """SELLs go through even in the lockout window (operator can always close)."""
    repo = MagicMock()
    old_buy = SimpleNamespace(
        symbol="AAPL", side="buy", timestamp=datetime.now(UTC) - timedelta(hours=2)
    )
    repo.get_open_trades = AsyncMock(return_value=[old_buy])
    repo.record_trade = AsyncMock(return_value=1)
    repo.close_open_trades_for_symbol = AsyncMock(return_value=1)
    executor = _executor(repo, cooldown=0)

    with patch(
        "halal_trader.market_hours.now_eastern",
        return_value=_at_et(15, 55),  # 5 min to close, lockout active
    ):
        result = await executor._execute_sell(_decision_sell("AAPL"))

    # SELL not rejected by lockout (lockout only gates BUYs).
    assert "close lockout" not in result.get("reason", "").lower()
