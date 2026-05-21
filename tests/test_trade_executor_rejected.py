"""Executor must persist `status='rejected'` when the broker returns a
malformed response (no order id).

Yesterday's outage: the Alpaca MCP server returned a Pydantic
validation error STRING as the response payload. The old code's
``order_id = result.get('id', '') if isinstance(result, dict) else ''``
extraction gave an empty string, then ``_confirm_fill`` returned
``status='pending'``, then ``record_trade`` persisted a phantom
``pending`` row with quantity=380. That row then re-appeared as 100%
drift on every subsequent reconcile pass until the operator dug it
out by hand. This test pins the fix: a malformed broker response now
produces a ``rejected`` row.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.domain.models import Account, TradeAction, TradeDecision
from halal_trader.trading.executor import TradeExecutor, _extract_order_id


def _account() -> Account:
    return Account(
        equity=100_000,
        buying_power=100_000,
        cash=100_000,
        portfolio_value=100_000,
        status="ACTIVE",
    )


def _decision(symbol: str = "AMZN", quantity: int = 380) -> TradeDecision:
    return TradeDecision(
        action=TradeAction.BUY, symbol=symbol, quantity=quantity, confidence=0.8, reasoning="t"
    )


def _broker(*, place_order_return: object) -> MagicMock:
    b = MagicMock()
    b.get_account_info = AsyncMock(return_value=_account())
    b.get_stock_snapshot = AsyncMock(return_value={"latest_trade": {"price": 200.0}})
    b.place_order = AsyncMock(return_value=place_order_return)
    return b


# ── unit: _extract_order_id ──────────────────────────────────────


def test_extract_order_id_normal_dict():
    assert _extract_order_id({"id": "abc-123", "status": "filled"}) == "abc-123"


def test_extract_order_id_dict_without_id():
    assert _extract_order_id({"error": "boom"}) == ""


def test_extract_order_id_dict_with_empty_id():
    assert _extract_order_id({"id": ""}) == ""


def test_extract_order_id_non_dict():
    assert _extract_order_id("Pydantic validation error: ...") == ""
    assert _extract_order_id(None) == ""
    assert _extract_order_id([{"id": "should-be-ignored"}]) == ""


def test_extract_order_id_non_string_id_coerced():
    assert _extract_order_id({"id": 42}) == "42"


def test_extract_order_id_unwraps_result_envelope():
    """Upstream Alpaca MCP wraps responses as {"result": {"id": ...}} —
    mirror the same unwrap the get_all_positions parser does."""
    assert _extract_order_id({"result": {"id": "wrapped-1", "status": "filled"}}) == "wrapped-1"


def test_extract_order_id_wrapped_without_id_returns_empty():
    """{"result": {...no id...}} → empty, treated as rejected."""
    assert _extract_order_id({"result": {"error": "boom"}}) == ""


def test_extract_order_id_wrapped_result_non_dict():
    """{"result": [...]} (list, not dict) → empty."""
    assert _extract_order_id({"result": [1, 2, 3]}) == ""


# ── close_position orphan path ──────────────────────────────────


def _position(symbol: str, qty: float):
    from halal_trader.domain.models import Position
    return Position(symbol=symbol, qty=qty, avg_entry_price=100.0, current_price=101.0)


def _sell_close_decision(symbol: str = "AAPL"):
    """A `decision.quantity == 0` sentinel triggers the close_position path."""
    return TradeDecision(
        action=TradeAction.SELL, symbol=symbol, quantity=0, confidence=0.8, reasoning="exit"
    )


@pytest.mark.asyncio
async def test_close_position_no_order_id_records_filled_when_position_gone():
    """The classic close_position flow: broker returns no id but the
    position is gone afterwards. We must record a filled SELL with the
    pre-close quantity so the BUY is closed in the reconciler's view."""
    broker = MagicMock()
    broker.get_account_info = AsyncMock(return_value=_account())
    # Position present pre-close, gone after close
    broker.get_all_positions = AsyncMock(
        side_effect=[
            [_position("AAPL", 25.0)],  # pre-close snapshot
            [],  # post-close: nothing left
        ]
    )
    # close_position returns a payload with no id (real-world shape)
    broker.close_position = AsyncMock(return_value={"closed": True, "status": "ok"})

    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=200)

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0
    )

    result = await executor._execute_sell(_sell_close_decision("AAPL"))
    assert result["status"] == "filled"
    kwargs = repo.record_trade.await_args.kwargs
    assert kwargs["status"] == "filled"
    assert kwargs["filled_quantity"] == 25.0
    assert kwargs["side"] == "sell"
    assert kwargs["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_close_position_no_change_records_rejected():
    """close_position returned, but the position is still there → broker
    didn't actually close. Mark rejected so the reconciler doesn't
    accidentally write off an open BUY."""
    broker = MagicMock()
    broker.get_account_info = AsyncMock(return_value=_account())
    broker.get_all_positions = AsyncMock(
        side_effect=[[_position("AAPL", 25.0)], [_position("AAPL", 25.0)]]
    )
    broker.close_position = AsyncMock(return_value={"closed": False})

    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=201)

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0
    )

    result = await executor._execute_sell(_sell_close_decision("AAPL"))
    assert result["status"] == "rejected"
    kwargs = repo.record_trade.await_args.kwargs
    assert kwargs["status"] == "rejected"
    assert kwargs["filled_quantity"] == 0.0


@pytest.mark.asyncio
async def test_close_position_no_open_position_skipped():
    """If nothing is open for the symbol, close_position is a no-op."""
    broker = MagicMock()
    broker.get_account_info = AsyncMock(return_value=_account())
    broker.get_all_positions = AsyncMock(return_value=[])  # nothing to close
    broker.close_position = AsyncMock()

    repo = MagicMock()
    repo.record_trade = AsyncMock()

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0
    )

    result = await executor._execute_sell(_sell_close_decision("AAPL"))
    assert result["status"] == "skipped"
    # Critical: must NOT have called close_position (or recorded a trade)
    # — pre-fetch told us there's nothing there.
    broker.close_position.assert_not_called()
    repo.record_trade.assert_not_called()


@pytest.mark.asyncio
async def test_sell_fill_closes_open_buy_row():
    """An LLM-initiated SELL must stamp closed_at on the underlying open
    BUY(s). Without this, the recent-close cooldown query missed LLM
    sells and same-symbol re-buys leaked through (observed 2026-05-21
    14:45 ET on QCOM)."""
    broker = MagicMock()
    broker.get_account_info = AsyncMock(return_value=_account())
    broker.place_order = AsyncMock(return_value={"id": "sell-1", "status": "filled"})
    broker.get_order_by_id = AsyncMock(
        return_value={
            "id": "sell-1",
            "status": "filled",
            "filled_qty": "40",
            "filled_avg_price": "207",
            "filled_at": "2026-05-21T18:00:00Z",
        }
    )
    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=300)
    repo.close_open_trades_for_symbol = AsyncMock(return_value=1)

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0
    )

    sized_sell = TradeDecision(
        action=TradeAction.SELL, symbol="QCOM", quantity=40, confidence=0.8, reasoning="exit"
    )
    result = await executor._execute_sell(sized_sell)

    assert result["status"] == "filled"
    # The critical assertion: the executor closed the open BUY(s).
    repo.close_open_trades_for_symbol.assert_awaited_once()
    args = repo.close_open_trades_for_symbol.await_args.args
    assert args[0] == "QCOM"
    assert args[1] == 207.0  # filled_avg_price from broker
    assert args[2] == "llm_sell"


@pytest.mark.asyncio
async def test_close_position_with_order_id_uses_confirm_fill_path():
    """When close_position DOES return an order id, the normal poll path
    runs (no synthesized fill)."""
    broker = MagicMock()
    broker.get_account_info = AsyncMock(return_value=_account())
    broker.get_all_positions = AsyncMock(return_value=[_position("AAPL", 25.0)])
    broker.close_position = AsyncMock(return_value={"id": "close-1", "status": "accepted"})
    broker.get_order_by_id = AsyncMock(
        return_value={
            "id": "close-1",
            "status": "filled",
            "filled_qty": "25",
            "filled_avg_price": "200",
            "filled_at": "2026-04-26T15:00:00Z",
        }
    )

    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=202)

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0
    )

    result = await executor._execute_sell(_sell_close_decision("AAPL"))
    assert result["status"] == "filled"
    kwargs = repo.record_trade.await_args.kwargs
    assert kwargs["status"] == "filled"
    assert kwargs["filled_quantity"] == 25.0
    assert kwargs["order_id"] == "close-1"


# ── integration: _execute_buy on malformed response ─────────────


@pytest.mark.asyncio
async def test_buy_with_string_response_records_rejected():
    """The exact yesterday-bug shape: broker returns a non-dict string."""
    broker = _broker(place_order_return="2 validation errors for place_stock_order …")
    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=99)

    executor = TradeExecutor(
        broker,
        repo,
        max_position_pct=1.0,
        max_simultaneous_positions=10,
        max_sector_pct=0,  # disable sector cap for this test
    )

    result = await executor._execute_buy(_decision("AMZN", quantity=380), positions=[])

    assert result["status"] == "rejected"
    assert "validation" in result["reason"]
    repo.record_trade.assert_awaited_once()
    kwargs = repo.record_trade.await_args.kwargs
    assert kwargs["status"] == "rejected"
    assert kwargs["filled_quantity"] == 0.0
    assert kwargs["order_id"] == ""
    assert kwargs["symbol"] == "AMZN"
    assert kwargs["quantity"] == 380


@pytest.mark.asyncio
async def test_buy_with_dict_missing_id_records_rejected():
    """Broker dict that lacks an `id` field → rejected, not pending."""
    broker = _broker(place_order_return={"error": "rate limit", "code": 429})
    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=100)

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0
    )

    result = await executor._execute_buy(_decision("NVDA", quantity=10), positions=[])
    assert result["status"] == "rejected"
    kwargs = repo.record_trade.await_args.kwargs
    assert kwargs["status"] == "rejected"
    assert kwargs["filled_quantity"] == 0.0


@pytest.mark.asyncio
async def test_buy_with_none_response_records_rejected():
    """Broker returns None → still safe, recorded as rejected."""
    broker = _broker(place_order_return=None)
    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=101)

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0
    )

    result = await executor._execute_buy(_decision(), positions=[])
    assert result["status"] == "rejected"
    repo.record_trade.assert_awaited_once()


@pytest.mark.asyncio
async def test_buy_with_valid_response_unchanged_path():
    """Sanity: valid response still flows through normally."""
    broker = _broker(place_order_return={"id": "ord-1", "status": "filled"})
    broker.get_order_by_id = AsyncMock(
        return_value={
            "id": "ord-1",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "200",
            "filled_at": "2026-04-26T15:00:00Z",
        }
    )
    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=42)

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0
    )

    result = await executor._execute_buy(_decision("NVDA", quantity=10), positions=[])
    assert result["status"] == "filled"
    kwargs = repo.record_trade.await_args.kwargs
    assert kwargs["status"] == "filled"
    assert kwargs["filled_quantity"] == 10.0
