"""Tests for the post-close analytics fan-out."""

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


# ── Drift dispatch ───────────────────────────────────────────────


async def test_drift_observed() -> None:
    hub = InsightsHub()
    rec = CloseRecorders(hub=hub)
    summary = await record_close(_event(pnl=0.01), rec)
    assert hub.drift.n == 1
    assert "drift_state" in summary


async def test_no_hub_skips_drift_silently() -> None:
    rec = CloseRecorders()
    summary = await record_close(_event(), rec)
    assert "drift_state" not in summary


# ── Thesis dispatch ──────────────────────────────────────────────


async def test_thesis_recorded(tmp_path: Path) -> None:
    store = ThesisTagStore(path=tmp_path / "tags.json")
    rec = CloseRecorders(thesis_store=store)
    summary = await record_close(_event(hold_seconds=120), rec)  # short hold → scalp
    assert summary["thesis_tag"] == "scalp"
    assert store.get("t1") == "scalp"


# ── Regret dispatch ──────────────────────────────────────────────


async def test_regret_appended(tmp_path: Path) -> None:
    side = RegretSidecar(path=tmp_path / "regret.json")
    rec = CloseRecorders(regret_sidecar=side)
    summary = await record_close(_event(pnl=0.02), rec)
    rows = side.all()
    assert len(rows) == 1
    assert rows[0]["trade_id"] == "t1"
    assert rows[0]["pnl_pct"] == 0.02
    assert "regret" in summary


async def test_regret_resilient_to_corrupt_sidecar(tmp_path: Path) -> None:
    p = tmp_path / "regret.json"
    p.write_text("{not json")
    side = RegretSidecar(path=p)
    rec = CloseRecorders(regret_sidecar=side)
    await record_close(_event(), rec)
    assert len(side.all()) == 1


# ── Purification dispatch ────────────────────────────────────────


async def test_purification_recorded_on_winning_trade(tmp_path: Path) -> None:
    led = RoundTripLedger(path=tmp_path / "purif.json")
    rules = {"BTCUSDT": RoundTripRule(symbol="BTCUSDT", impure_ratio=0.02)}
    rec = CloseRecorders(purification_ledger=led, purification_rules=rules)
    summary = await record_close(_event(gain_usd=100.0), rec)
    assert summary.get("purification_due_usd") == 2.0
    assert led.outstanding() == 2.0


async def test_purification_skipped_on_loss(tmp_path: Path) -> None:
    led = RoundTripLedger(path=tmp_path / "purif.json")
    rules = {"BTCUSDT": RoundTripRule(symbol="BTCUSDT", impure_ratio=0.02)}
    rec = CloseRecorders(purification_ledger=led, purification_rules=rules)
    summary = await record_close(_event(gain_usd=-50.0), rec)
    assert "purification_due_usd" not in summary
    assert led.outstanding() == 0.0


async def test_purification_skipped_when_no_rule(tmp_path: Path) -> None:
    led = RoundTripLedger(path=tmp_path / "purif.json")
    rec = CloseRecorders(purification_ledger=led, purification_rules={})
    summary = await record_close(_event(gain_usd=100.0), rec)
    assert "purification_due_usd" not in summary


# ── Resilience ───────────────────────────────────────────────────


async def test_record_close_never_raises_on_recorder_failure() -> None:
    class _BoomStore:
        def set(self, *a, **kw):
            raise RuntimeError("boom")

    rec = CloseRecorders(thesis_store=_BoomStore())
    # Must not raise
    await record_close(_event(), rec)


async def test_full_fan_out(tmp_path: Path) -> None:
    """End-to-end: every recorder fires for one event."""
    hub = InsightsHub()
    store = ThesisTagStore(path=tmp_path / "tags.json")
    side = RegretSidecar(path=tmp_path / "regret.json")
    led = RoundTripLedger(path=tmp_path / "purif.json")
    rules = {"BTCUSDT": RoundTripRule(symbol="BTCUSDT", impure_ratio=0.02)}
    rec = CloseRecorders(
        hub=hub,
        thesis_store=store,
        regret_sidecar=side,
        purification_ledger=led,
        purification_rules=rules,
    )
    summary = await record_close(_event(pnl=0.02, gain_usd=100), rec)
    assert hub.drift.n == 1
    assert store.get("t1") is not None
    assert len(side.all()) == 1
    assert led.outstanding() == 2.0
    # All four keys present
    for key in ("drift_state", "thesis_tag", "regret", "purification_due_usd"):
        assert key in summary
