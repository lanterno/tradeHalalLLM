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

from types import SimpleNamespace
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
    # Held long ≥ the sell quantity so the short-guard clamp lets it through.
    broker.get_all_positions = AsyncMock(
        return_value=[SimpleNamespace(symbol="QCOM", qty=40.0)]
    )
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
async def test_close_all_records_synthetic_sell_for_each_symbol():
    """EOD close-all must record matching SELL Trade rows so the
    reconciler's signed-net math (BUY+ minus SELL-) cancels. Without
    this, the next morning's reconcile shows db=<qty> broker=0 =
    100% drift — exactly what we hit on 2026-05-22 morning on
    SHOP/NOW/MSFT after the previous EOD."""
    from types import SimpleNamespace

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="SHOP", qty=290, current_price=42.5),
        ]
    )
    broker.close_all_positions = AsyncMock(return_value={"result": "closed"})

    repo = MagicMock()
    # Two open BUYs for SHOP — total 290 shares.
    repo.get_open_trades = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="SHOP", side="buy", filled_quantity=145, filled_price=40.0),
            SimpleNamespace(symbol="SHOP", side="buy", filled_quantity=145, filled_price=41.0),
        ]
    )
    repo.close_open_trades_for_symbol = AsyncMock(return_value=2)
    repo.record_trade = AsyncMock(return_value=999)

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0
    )

    await executor.close_all()

    # Synthetic SELL recorded with the SUM of open BUY quantities.
    repo.record_trade.assert_awaited_once()
    kwargs = repo.record_trade.await_args.kwargs
    assert kwargs["symbol"] == "SHOP"
    assert kwargs["side"] == "sell"
    assert kwargs["quantity"] == 290.0
    assert kwargs["filled_quantity"] == 290.0
    assert kwargs["filled_price"] == 42.5  # pre-snapshot current_price
    assert kwargs["status"] == "filled"


@pytest.mark.asyncio
async def test_close_all_stamps_closed_at_on_orphan_buys():
    """The EOD close-all path now walks the DB and stamps closed_at on
    every open BUY. Without this the previous EOD left orphans that
    showed as 100% reconcile drift the next morning (observed
    2026-05-22 10:30 ET on SHOP/NOW/MSFT)."""
    from types import SimpleNamespace

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="SHOP", qty=290, current_price=42.5),
            SimpleNamespace(symbol="NOW", qty=90, current_price=820.0),
        ]
    )
    broker.close_all_positions = AsyncMock(return_value={"result": "closed"})

    repo = MagicMock()
    repo.get_open_trades = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="SHOP", side="buy", filled_price=40.0),
            SimpleNamespace(symbol="NOW", side="buy", filled_price=800.0),
            SimpleNamespace(symbol="MSFT", side="buy", filled_price=410.0),  # not in broker
        ]
    )
    repo.close_open_trades_for_symbol = AsyncMock(return_value=1)

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0
    )

    await executor.close_all()

    # All three symbols should get a close call — even MSFT which the
    # broker didn't have (orphan: DB thought we held it but broker
    # didn't, exactly the case we're fixing).
    closed_symbols = {
        call.args[0] for call in repo.close_open_trades_for_symbol.await_args_list
    }
    assert closed_symbols == {"SHOP", "NOW", "MSFT"}
    # Exit price for SHOP comes from pre-snapshot current_price (42.5).
    shop_call = next(
        c for c in repo.close_open_trades_for_symbol.await_args_list if c.args[0] == "SHOP"
    )
    assert shop_call.args[1] == 42.5
    # MSFT had no snapshot entry → falls back to BUY's filled_price (410).
    msft_call = next(
        c for c in repo.close_open_trades_for_symbol.await_args_list if c.args[0] == "MSFT"
    )
    assert msft_call.args[1] == 410.0
    # Reason is uniform across all close-all entries.
    for call in repo.close_open_trades_for_symbol.await_args_list:
        assert call.args[2] == "eod_close_all"


@pytest.mark.asyncio
async def test_close_all_survives_pre_snapshot_failure():
    """If the broker's pre-position snapshot fails, the close-out still
    walks the DB and closes BUYs with whatever filled_price they have."""
    from types import SimpleNamespace

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(side_effect=RuntimeError("broker down"))
    broker.close_all_positions = AsyncMock(return_value={"result": "closed"})

    repo = MagicMock()
    repo.get_open_trades = AsyncMock(
        return_value=[SimpleNamespace(symbol="SHOP", side="buy", filled_price=40.0)]
    )
    repo.close_open_trades_for_symbol = AsyncMock(return_value=1)

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0
    )

    await executor.close_all()

    # Still closed the BUY, using the filled_price fallback.
    assert repo.close_open_trades_for_symbol.await_count == 1
    call = repo.close_open_trades_for_symbol.await_args_list[0]
    assert call.args[0] == "SHOP"
    assert call.args[1] == 40.0


@pytest.mark.asyncio
async def test_close_all_skips_synthetic_sell_when_unpriceable():
    """A reverse-orphan BUY (broker has no position, no filled_price, no
    entry estimate) must NOT get a fabricated $0 synthetic SELL — that
    stamped exit_price=0 on the BUY and showed a phantom −100% loss in
    P&L (observed 2026-05-22 on CSCO). Skip it entirely instead."""
    from types import SimpleNamespace

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(return_value=[])  # broker holds nothing
    broker.close_all_positions = AsyncMock(return_value={"result": "closed"})

    repo = MagicMock()
    repo.get_open_trades = AsyncMock(
        return_value=[
            SimpleNamespace(
                symbol="CSCO",
                side="buy",
                filled_quantity=50,
                filled_price=None,  # never confirmed
                price=None,  # no entry estimate either
            )
        ]
    )
    repo.close_open_trades_for_symbol = AsyncMock(return_value=1)
    repo.record_trade = AsyncMock(return_value=1)

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0
    )

    await executor.close_all()

    # No $0 synthetic SELL, and no $0 close stamp.
    repo.record_trade.assert_not_awaited()
    repo.close_open_trades_for_symbol.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_all_falls_back_to_entry_price_when_no_fill():
    """When a BUY has no confirmed filled_price but does carry an entry
    estimate (``price``), the EOD close uses that as a breakeven exit
    rather than skipping or fabricating $0."""
    from types import SimpleNamespace

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(return_value=[])  # not on broker
    broker.close_all_positions = AsyncMock(return_value={"result": "closed"})

    repo = MagicMock()
    repo.get_open_trades = AsyncMock(
        return_value=[
            SimpleNamespace(
                symbol="CSCO",
                side="buy",
                filled_quantity=50,
                filled_price=None,
                price=48.0,  # entry estimate → breakeven exit
            )
        ]
    )
    repo.close_open_trades_for_symbol = AsyncMock(return_value=1)
    repo.record_trade = AsyncMock(return_value=1)

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0
    )

    await executor.close_all()

    close_call = repo.close_open_trades_for_symbol.await_args_list[0]
    assert close_call.args[0] == "CSCO"
    assert close_call.args[1] == 48.0
    sell_kwargs = repo.record_trade.await_args.kwargs
    assert sell_kwargs["filled_price"] == 48.0
    assert sell_kwargs["quantity"] == 50.0


@pytest.mark.asyncio
async def test_close_all_holds_reactor_positions_overnight():
    """Reactor-momentum positions are exempt from the EOD flatten — the
    broker batch close is skipped (it would flatten them), non-reactor
    positions are closed individually, and reactor symbols get neither a
    closed_at stamp nor a synthetic SELL."""
    from types import SimpleNamespace

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="NVDA", qty=20, current_price=210.0),  # reactor
            SimpleNamespace(symbol="SHOP", qty=10, current_price=42.0),  # cycle
        ]
    )
    broker.close_all_positions = AsyncMock(return_value={"result": "closed"})
    broker.close_position = AsyncMock(return_value={"id": "x"})

    repo = MagicMock()
    repo.get_open_trades = AsyncMock(
        return_value=[
            SimpleNamespace(
                symbol="NVDA", side="buy", filled_quantity=20, filled_price=200.0,
                entry_type="reactor_momentum",
            ),
            SimpleNamespace(
                symbol="SHOP", side="buy", filled_quantity=10, filled_price=40.0,
                entry_type=None,
            ),
        ]
    )
    repo.close_open_trades_for_symbol = AsyncMock(return_value=1)
    repo.record_trade = AsyncMock(return_value=1)

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0,
        reactor_hold_overnight=True,
    )

    await executor.close_all()

    # Batch flatten NOT used (would have closed NVDA too).
    broker.close_all_positions.assert_not_awaited()
    # Only SHOP closed individually on the broker; NVDA held.
    closed_syms = {c.args[0] for c in broker.close_position.await_args_list}
    assert closed_syms == {"SHOP"}
    # DB cleanup touches SHOP only.
    db_closed = {c.args[0] for c in repo.close_open_trades_for_symbol.await_args_list}
    assert db_closed == {"SHOP"}
    sold_syms = {c.kwargs["symbol"] for c in repo.record_trade.await_args_list}
    assert "NVDA" not in sold_syms


@pytest.mark.asyncio
async def test_close_all_flattens_reactor_when_hold_disabled():
    """With reactor_hold_overnight=False, reactor positions flatten at EOD
    like everything else (batch close)."""
    from types import SimpleNamespace

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(
        return_value=[SimpleNamespace(symbol="NVDA", qty=20, current_price=210.0)]
    )
    broker.close_all_positions = AsyncMock(return_value={"result": "closed"})

    repo = MagicMock()
    repo.get_open_trades = AsyncMock(
        return_value=[
            SimpleNamespace(
                symbol="NVDA", side="buy", filled_quantity=20, filled_price=200.0,
                entry_type="reactor_momentum",
            )
        ]
    )
    repo.close_open_trades_for_symbol = AsyncMock(return_value=1)
    repo.record_trade = AsyncMock(return_value=1)

    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0,
        reactor_hold_overnight=False,
    )

    await executor.close_all()

    broker.close_all_positions.assert_awaited_once()
    db_closed = {c.args[0] for c in repo.close_open_trades_for_symbol.await_args_list}
    assert db_closed == {"NVDA"}


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


# ── cumulative per-name position cap ────────────────────────────
#
# The per-name cap (max_position_pct) must count the EXISTING holding plus
# the new order — checking only the new order's notional let repeated adds
# to one symbol blow past the cap (observed 2026-06-17: AAPL → ~34% of
# equity via four buys each individually under the 20% cap).


def _pos(symbol: str, qty: float, price: float = 200.0):
    return SimpleNamespace(
        symbol=symbol, qty=qty, current_price=price, avg_entry_price=price
    )


def test_existing_position_value_held_and_absent():
    from halal_trader.trading.executor import _existing_position_value

    positions = [_pos("AAPL", 60, 200.0), _pos("MSFT", 10, 410.0)]
    assert _existing_position_value("AAPL", positions) == 60 * 200.0
    assert _existing_position_value("aapl", positions) == 60 * 200.0  # case-insensitive
    assert _existing_position_value("NVDA", positions) == 0.0  # not held
    assert _existing_position_value("AAPL", []) == 0.0
    assert _existing_position_value("AAPL", None) == 0.0


@pytest.mark.asyncio
async def test_position_cap_counts_existing_holding():
    """Held $12k (60@200) + new $10k (50@200) = 22% of $100k > 20% cap →
    rejected, even though the new order alone is only 10%."""
    broker = _broker(place_order_return={"id": "x", "status": "filled"})
    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=1)
    executor = TradeExecutor(
        broker, repo, max_position_pct=0.20, max_simultaneous_positions=10, max_sector_pct=0
    )

    result = await executor._execute_buy(
        _decision("AAPL", quantity=50), positions=[_pos("AAPL", 60, 200.0)]
    )

    assert result["status"] == "rejected"
    assert "exceeds" in result["reason"] and "limit" in result["reason"]
    broker.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_position_cap_allows_when_cumulative_under_limit():
    """Held $12k + new $6k (30@200) = 18% < 20% → the cap does NOT block it
    (it proceeds and is rejected later for the malformed response, proving
    the cap check passed)."""
    broker = _broker(place_order_return="2 validation errors for place_stock_order …")
    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=1)
    executor = TradeExecutor(
        broker, repo, max_position_pct=0.20, max_simultaneous_positions=10, max_sector_pct=0
    )

    result = await executor._execute_buy(
        _decision("AAPL", quantity=30), positions=[_pos("AAPL", 60, 200.0)]
    )

    # Rejected, but NOT by the position cap — it got past the cap to the broker.
    assert result["status"] == "rejected"
    assert "exceeds" not in result["reason"]
    broker.place_order.assert_awaited_once()
