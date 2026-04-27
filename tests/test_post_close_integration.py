"""End-to-end integration: the close hook fires every recorder.

This test exercises the real PositionMonitor wiring path — same
``record_close`` fan-out the live cycle uses — to assert that one
simulated trade close populates drift, thesis, regret and purification
state in the way the dashboard reads them.

Avoids touching network: scripts the broker's ``buy()`` and price
queries with light mocks.
"""

from __future__ import annotations

from pathlib import Path

from halal_trader.core.insights_hub import InsightsHub
from halal_trader.core.post_close import (
    CloseEvent,
    CloseRecorders,
    RegretSidecar,
    record_close,
)
from halal_trader.core.thesis import ThesisTagStore
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


async def test_full_post_close_flow_to_dashboard_shape(tmp_path: Path) -> None:
    """One winning close → drift observed + thesis tagged + regret
    sidecar appended + purification accrual."""
    hub = InsightsHub()
    rec = CloseRecorders(
        hub=hub,
        thesis_store=ThesisTagStore(path=tmp_path / "thesis.json"),
        regret_sidecar=RegretSidecar(path=tmp_path / "regret.json"),
        purification_ledger=RoundTripLedger(path=tmp_path / "purif.json"),
        purification_rules={"BTCUSDT": RoundTripRule(symbol="BTCUSDT", impure_ratio=0.02)},
    )
    summary = await record_close(_e("BTCUSDT", pnl=0.02, gain=100.0), rec)

    # Dashboard JSON shape — every key the /api/insights/* routes read.
    assert hub.drift.n == 1
    assert hub.drift.state in ("warming_up", "stable", "drift")

    snap = hub.to_app_state()
    assert "drift_monitor" in snap
    assert "shadow_ledger" in snap
    assert "regime_memory" in snap

    # Thesis tag is persisted.
    assert rec.thesis_store.get("BTCUSDT-1") is not None

    # Regret sidecar got the row.
    rows = rec.regret_sidecar.all()
    assert len(rows) == 1
    assert rows[0]["trade_id"] == "BTCUSDT-1"
    assert rows[0]["pnl_pct"] == 0.02

    # Purification accrued + reachable from outstanding ledger.
    assert rec.purification_ledger.outstanding() == 2.0
    assert summary["purification_due_usd"] == 2.0


async def test_loss_close_skips_purification(tmp_path: Path) -> None:
    hub = InsightsHub()
    rec = CloseRecorders(
        hub=hub,
        thesis_store=ThesisTagStore(path=tmp_path / "thesis.json"),
        regret_sidecar=RegretSidecar(path=tmp_path / "regret.json"),
        purification_ledger=RoundTripLedger(path=tmp_path / "purif.json"),
        purification_rules={"BTCUSDT": RoundTripRule(symbol="BTCUSDT", impure_ratio=0.02)},
    )
    await record_close(_e("BTCUSDT", pnl=-0.02, gain=-100.0), rec)
    assert rec.purification_ledger.outstanding() == 0.0
    # But drift + thesis + regret still recorded.
    assert hub.drift.n == 1
    assert rec.thesis_store.get("BTCUSDT-1") is not None
    assert len(rec.regret_sidecar.all()) == 1


async def test_multiple_closes_aggregate(tmp_path: Path) -> None:
    hub = InsightsHub()
    rec = CloseRecorders(
        hub=hub,
        thesis_store=ThesisTagStore(path=tmp_path / "thesis.json"),
        regret_sidecar=RegretSidecar(path=tmp_path / "regret.json"),
        purification_ledger=RoundTripLedger(path=tmp_path / "purif.json"),
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
    # Only AAPL has purification rules — and only winners.
    assert rec.purification_ledger.outstanding() > 0
    assert len(rec.regret_sidecar.all()) == 20
