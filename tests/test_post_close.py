"""Tests for the post-close analytics fan-out."""

from __future__ import annotations

from halal_trader.core.llm.rag_db import DBRationaleStore
from halal_trader.core.post_close import (
    CloseEvent,
    CloseRecorders,
    record_close,
)
from halal_trader.halal.round_trip_purification import (
    RoundTripLedger,
    RoundTripRule,
)


def _event(pnl: float = 0.02, gain_usd: float = 50.0, **kwargs) -> CloseEvent:
    base = dict(
        trade_id="t1",
        symbol="BTCUSDT",
        side="buy",
        entry_price=100.0,
        exit_price=100.0 * (1.0 + pnl),
        exit_reason="take_profit",
        realized_pnl_usd=gain_usd,
        return_pct=pnl,
        quantity=0.5,
        hold_seconds=600,
        reasoning="momentum",
    )
    base.update(kwargs)
    return CloseEvent(**base)


# ── Purification dispatch ────────────────────────────────────────


async def test_purification_recorded_on_winning_trade(engine) -> None:
    led = RoundTripLedger(engine=engine)
    rules = {"BTCUSDT": RoundTripRule(symbol="BTCUSDT", impure_ratio=0.02)}
    rec = CloseRecorders(purification_ledger=led, purification_rules=rules)
    summary = await record_close(_event(gain_usd=100.0), rec)
    assert summary.get("purification_due_usd") == 2.0
    assert await led.outstanding() == 2.0


async def test_purification_skipped_on_loss(engine) -> None:
    led = RoundTripLedger(engine=engine)
    rules = {"BTCUSDT": RoundTripRule(symbol="BTCUSDT", impure_ratio=0.02)}
    rec = CloseRecorders(purification_ledger=led, purification_rules=rules)
    summary = await record_close(_event(gain_usd=-50.0), rec)
    assert "purification_due_usd" not in summary
    assert await led.outstanding() == 0.0


async def test_purification_skipped_when_no_rule(engine) -> None:
    led = RoundTripLedger(engine=engine)
    rec = CloseRecorders(purification_ledger=led, purification_rules={})
    summary = await record_close(_event(gain_usd=100.0), rec)
    assert "purification_due_usd" not in summary


# ── Resilience ───────────────────────────────────────────────────


async def test_record_close_never_raises_on_recorder_failure() -> None:
    class _BoomStore:
        async def add(self, *a, **kw):
            raise RuntimeError("boom")

    rec = CloseRecorders(rag_store=_BoomStore())
    summary = await record_close(_event(), rec)
    assert "rag_added" not in summary


async def test_full_fan_out(engine) -> None:
    """End-to-end: every recorder fires for one event."""
    rag = DBRationaleStore(engine=engine)
    led = RoundTripLedger(engine=engine)
    rules = {"BTCUSDT": RoundTripRule(symbol="BTCUSDT", impure_ratio=0.02)}
    rec = CloseRecorders(
        rag_store=rag,
        purification_ledger=led,
        purification_rules=rules,
    )
    summary = await record_close(_event(pnl=0.02, gain_usd=100), rec)
    assert await rag.size() == 1
    assert await led.outstanding() == 2.0
    for key in ("purification_due_usd", "rag_added"):
        assert key in summary
