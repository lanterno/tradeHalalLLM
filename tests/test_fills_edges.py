"""Edge-case tests for :mod:`core.fills`.

`test_fills.py` covers the main paths; this file pins the remaining
branches: the `confirm_binance` status-map's `expired` / `new` /
`pending_new` translations + unknown-raw-status passthrough,
`filled_at`'s populate-only-when-filled-and-qty-positive invariant,
and `confirm_alpaca`'s sync-callback path + `_build_alpaca_result`'s
field-key fallback (`filled_qty` vs `filled_quantity`,
`filled_avg_price` vs `avg_price`).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from halal_trader.core.fills import (
    FillResult,
    _build_alpaca_result,
    confirm_alpaca,
    confirm_binance,
)
from halal_trader.domain.status import TradeStatus


def _now() -> datetime:
    return datetime(2026, 4, 25, 12, 0, tzinfo=UTC)


# ── confirm_binance status-map edges ────────────────────────


def test_binance_expired_status_maps_to_rejected():
    """`expired` (Binance returns this when an order's TIF lapses)
    funnels into REJECTED — not its own state. Pinned because the
    rest of the pipeline only branches on FILLED / REJECTED / etc."""
    resp = {"orderId": 1, "status": "EXPIRED", "executedQty": 0, "cumulativeQuoteQty": 0}
    out = confirm_binance(resp, _now())
    assert out.status == TradeStatus.REJECTED


def test_binance_new_status_maps_to_pending():
    """A bare ``status='NEW'`` ack from Binance is PENDING, not FILLED."""
    resp = {"orderId": 1, "status": "NEW", "executedQty": 0, "cumulativeQuoteQty": 0}
    out = confirm_binance(resp, _now())
    assert out.status == TradeStatus.PENDING


def test_binance_pending_new_status_also_maps_to_pending():
    resp = {"orderId": 1, "status": "PENDING_NEW"}
    out = confirm_binance(resp, _now())
    assert out.status == TradeStatus.PENDING


def test_binance_unknown_raw_status_passes_through_unchanged():
    """Defensive: an unrecognised broker status (e.g. a future Binance
    state) passes through as-is rather than crashing or being swallowed
    into a wrong category. Operator can grep for the unknown value."""
    resp = {"orderId": 1, "status": "WEIRD_FUTURE_STATE"}
    out = confirm_binance(resp, _now())
    assert out.status == "weird_future_state"


def test_binance_empty_status_falls_back_to_pending():
    """No `status` key (malformed response) → "pending" sentinel."""
    resp = {"orderId": 1}
    out = confirm_binance(resp, _now())
    # Empty raw_status normalises to "pending" string; status_map doesn't
    # have it so it passes through.
    assert out.status == "pending"


def test_binance_filled_at_set_only_when_status_is_filled_and_qty_positive():
    """`filled_at` is the timestamp callers stamp on the DB row; it
    must be None for non-filled states (we don't have a real fill time).
    Pin both halves: status=FILLED + qty>0 → set; either absent → None."""
    # Filled + qty > 0 → set.
    resp = {"orderId": 1, "status": "FILLED", "fills": [{"price": "100", "qty": "1"}]}
    out = confirm_binance(resp, _now())
    assert out.filled_at is not None
    assert out.filled_at.tzinfo is UTC

    # Filled status but qty=0 (some exchange edge case) → None.
    resp_zero = {"orderId": 2, "status": "FILLED", "fills": []}
    out_zero = confirm_binance(resp_zero, _now())
    assert out_zero.filled_at is None

    # Partially filled → None.
    resp_partial = {
        "orderId": 3,
        "status": "PARTIALLY_FILLED",
        "fills": [{"price": "100", "qty": "0.5"}],
    }
    out_partial = confirm_binance(resp_partial, _now())
    assert out_partial.filled_at is None


def test_binance_orderid_coerced_to_str():
    """Binance returns numeric orderIds; our DB column is text. Coerce
    so downstream string concatenation / dict lookups don't blow up."""
    resp = {"orderId": 12345, "status": "FILLED", "fills": [{"price": "100", "qty": "1"}]}
    out = confirm_binance(resp, _now())
    assert out.order_id == "12345"
    assert isinstance(out.order_id, str)


def test_binance_missing_orderid_yields_empty_string():
    """Defensive: no orderId in the response → "" rather than crash."""
    resp = {"status": "FILLED", "fills": [{"price": "100", "qty": "1"}]}
    out = confirm_binance(resp, _now())
    assert out.order_id == ""


def test_binance_raw_payload_preserved_on_result():
    """The full broker response is stashed on `raw` so the operator can
    reconstruct the wire-level state during incident triage."""
    resp = {"orderId": 1, "status": "FILLED", "fills": [{"price": "100", "qty": "1"}]}
    out = confirm_binance(resp, _now())
    assert out.raw is resp  # exact same dict, not a copy


# ── confirm_alpaca sync-callback path ───────────────────────


@pytest.mark.asyncio
async def test_alpaca_accepts_sync_poll_callback():
    """`poll` may return either a dict or a coroutine. The sync-return
    path is reached when the callback is a plain function (e.g. a unit
    test stub). Both must work without the loop hanging."""

    def sync_poll() -> dict[str, Any]:
        return {
            "status": "filled",
            "filled_qty": "1.0",
            "filled_avg_price": "100.0",
        }

    out = await confirm_alpaca(
        sync_poll,
        order_id="order-1",
        submitted_at=_now(),
        timeout=1.0,
        interval=0.05,
    )
    assert out.status == "filled"
    assert out.filled_quantity == 1.0
    assert out.filled_price == 100.0


@pytest.mark.asyncio
async def test_alpaca_canceled_terminal_short_circuits_polling():
    """`canceled` is in `_TERMINAL_STATES` — first poll returning it
    must exit the loop immediately, not keep polling till timeout."""
    poll_count = [0]

    def poll() -> dict[str, Any]:
        poll_count[0] += 1
        return {"status": "canceled", "filled_qty": 0}

    out = await confirm_alpaca(
        poll,
        order_id="order-cx",
        submitted_at=_now(),
        timeout=10.0,
        interval=0.05,
    )
    assert out.status == "canceled"
    assert poll_count[0] == 1  # exited on the very first poll


@pytest.mark.asyncio
async def test_alpaca_async_poll_path():
    """`poll` returning a coroutine — common in production where the
    callback hits the MCP subprocess."""

    async def async_poll() -> dict[str, Any]:
        await asyncio.sleep(0)
        return {"status": "filled", "filled_qty": "2", "avg_price": "50"}

    out = await confirm_alpaca(
        async_poll,
        order_id="ord-2",
        submitted_at=_now(),
        timeout=1.0,
        interval=0.05,
    )
    assert out.status == "filled"
    assert out.filled_quantity == 2.0
    assert out.filled_price == 50.0  # via `avg_price` fallback


# ── _build_alpaca_result field-key fallback ─────────────────


def test_build_alpaca_uses_filled_qty_first():
    """`filled_qty` is the modern Alpaca field; the helper picks it
    over `filled_quantity` when both are present."""
    raw = {"filled_qty": "1.5", "filled_quantity": "9.9", "filled_avg_price": "100"}
    out = _build_alpaca_result("o1", "filled", raw, _now())
    assert out.filled_quantity == 1.5


def test_build_alpaca_falls_back_to_filled_quantity():
    """When `filled_qty` is missing, use `filled_quantity` — covers
    older SDK responses or test stubs that use the verbose name."""
    raw = {"filled_quantity": "2.0", "avg_price": "50"}
    out = _build_alpaca_result("o2", "filled", raw, _now())
    assert out.filled_quantity == 2.0


def test_build_alpaca_zero_quantity_when_neither_field_present():
    """Defensive: missing both keys → 0 (the `or 0` fallback) rather
    than crashing with a TypeError."""
    raw = {"filled_avg_price": "100"}
    out = _build_alpaca_result("o3", "filled", raw, _now())
    assert out.filled_quantity == 0.0


def test_build_alpaca_uses_filled_avg_price_first():
    """`filled_avg_price` is the modern field; takes priority over
    `avg_price` when both are set."""
    raw = {"filled_qty": "1", "filled_avg_price": "100", "avg_price": "999"}
    out = _build_alpaca_result("o4", "filled", raw, _now())
    assert out.filled_price == 100.0


def test_build_alpaca_falls_back_to_avg_price():
    """`filled_avg_price` missing → use `avg_price`."""
    raw = {"filled_qty": "1", "avg_price": "75"}
    out = _build_alpaca_result("o5", "filled", raw, _now())
    assert out.filled_price == 75.0


def test_build_alpaca_none_price_when_neither_field_present():
    """No price field at all → `filled_price=None`. The DB column is
    nullable so this is an acceptable terminal state for a partial
    fill that didn't surface the average price yet."""
    raw = {"filled_qty": "1"}
    out = _build_alpaca_result("o6", "partially_filled", raw, _now())
    assert out.filled_price is None


def test_build_alpaca_filled_at_only_for_filled_status_with_qty():
    """Mirror the Binance `filled_at` invariant — set only when status
    is `filled` AND `qty > 0`. Other terminals (rejected, canceled,
    pending) leave it None even if qty data is present."""
    raw = {"filled_qty": "1.0", "filled_avg_price": "100"}
    # filled + qty>0 → set
    out_filled = _build_alpaca_result("o7", "filled", raw, _now())
    assert out_filled.filled_at is not None
    # rejected (even with qty data) → None
    out_rej = _build_alpaca_result("o8", "rejected", raw, _now())
    assert out_rej.filled_at is None
    # filled but qty=0 → None
    out_zero = _build_alpaca_result("o9", "filled", {"filled_qty": "0"}, _now())
    assert out_zero.filled_at is None


def test_build_alpaca_preserves_raw_response():
    """The raw dict round-trips on the result — same identity check."""
    raw = {"filled_qty": "1", "anything": "else"}
    out = _build_alpaca_result("o10", "filled", raw, _now())
    assert out.raw is raw


def test_build_alpaca_preserves_submitted_at():
    """`submitted_at` is captured at place_order time and threads through
    untouched."""
    submitted = _now()
    out = _build_alpaca_result("o11", "filled", {"filled_qty": "1"}, submitted)
    assert out.submitted_at is submitted


def test_fill_result_is_frozen():
    """`FillResult` is a frozen dataclass — caller can stash it without
    fear of mutation. Pin so a refactor that drops `frozen=True` is
    caught."""
    out = _build_alpaca_result("x", "filled", {"filled_qty": "1"}, _now())
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        out.status = "different"  # type: ignore[misc]
    assert isinstance(out, FillResult)
