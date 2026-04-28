"""End-to-end integration: the close hook fires every recorder."""

from __future__ import annotations

from halal_trader.core.insights_hub import InsightsHub
from halal_trader.core.post_close import (
    CloseEvent,
    CloseRecorders,
    record_close,
)
from halal_trader.core.regret_db import DBRegretRecorder
from halal_trader.core.thesis_db import DBThesisTagStore
from halal_trader.halal.round_trip_purification import (
    RoundTripLedger,
    RoundTripRule,
)


def _e(symbol: str, *, pnl: float, gain: float, **kw):
    base = dict(
        trade_id=f"{symbol}-1",
        symbol=symbol,
        side="buy",
        entry_price=100.0,
        exit_price=100.0 * (1.0 + pnl),
        exit_reason="take_profit",
        realized_pnl_usd=gain,
        return_pct=pnl,
        quantity=1.0,
        hold_seconds=300,
        reasoning="momentum",
    )
    base.update(kw)
    return CloseEvent(**base)


async def test_full_post_close_flow_to_dashboard_shape(engine) -> None:
    """One winning close → drift observed + thesis tagged + regret + purification."""
    hub = InsightsHub()
    thesis_store = DBThesisTagStore(engine=engine)
    regret_recorder = DBRegretRecorder(engine=engine)
    ledger = RoundTripLedger(engine=engine)
    rec = CloseRecorders(
        hub=hub,
        thesis_store=thesis_store,
        regret_recorder=regret_recorder,
        purification_ledger=ledger,
        purification_rules={"BTCUSDT": RoundTripRule(symbol="BTCUSDT", impure_ratio=0.02)},
    )
    summary = await record_close(_e("BTCUSDT", pnl=0.02, gain=100.0), rec)

    assert hub.drift.n == 1
    assert hub.drift.state in ("warming_up", "stable", "drift")

    snap = hub.to_app_state()
    assert "drift_monitor" in snap
    assert "shadow_ledger" in snap
    assert "regime_memory" in snap

    assert await thesis_store.get("BTCUSDT-1") is not None

    rows = await regret_recorder.all()
    assert len(rows) == 1
    assert rows[0]["trade_id"] == "BTCUSDT-1"
    assert rows[0]["pnl_pct"] == 0.02

    assert await ledger.outstanding() == 2.0
    assert summary["purification_due_usd"] == 2.0


async def test_loss_close_skips_purification(engine) -> None:
    hub = InsightsHub()
    thesis_store = DBThesisTagStore(engine=engine)
    regret_recorder = DBRegretRecorder(engine=engine)
    ledger = RoundTripLedger(engine=engine)
    rec = CloseRecorders(
        hub=hub,
        thesis_store=thesis_store,
        regret_recorder=regret_recorder,
        purification_ledger=ledger,
        purification_rules={"BTCUSDT": RoundTripRule(symbol="BTCUSDT", impure_ratio=0.02)},
    )
    await record_close(_e("BTCUSDT", pnl=-0.02, gain=-100.0), rec)
    assert await ledger.outstanding() == 0.0
    assert hub.drift.n == 1
    assert await thesis_store.get("BTCUSDT-1") is not None
    assert len(await regret_recorder.all()) == 1


async def test_multiple_closes_aggregate(engine) -> None:
    hub = InsightsHub()
    thesis_store = DBThesisTagStore(engine=engine)
    regret_recorder = DBRegretRecorder(engine=engine)
    ledger = RoundTripLedger(engine=engine)
    rec = CloseRecorders(
        hub=hub,
        thesis_store=thesis_store,
        regret_recorder=regret_recorder,
        purification_ledger=ledger,
        purification_rules={"AAPL": RoundTripRule(symbol="AAPL", impure_ratio=0.01)},
    )
    for i in range(20):
        sym = "AAPL" if i % 2 == 0 else "MSFT"
        await record_close(
            _e(
                sym,
                pnl=0.01 if i % 3 == 0 else -0.005,
                gain=50 if i % 3 == 0 else -25,
                trade_id=f"{sym}-{i}",
            ),
            rec,
        )
    assert hub.drift.n == 20
    assert await ledger.outstanding() > 0
    assert len(await regret_recorder.all()) == 20
