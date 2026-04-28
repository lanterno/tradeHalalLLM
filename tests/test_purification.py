"""Tests for round-trip purification accounting.

Note: the existing ``halal_trader.halal.purification`` module covers
the *dividend* flavour; this test set covers the new round-trip
(capital-gains) flavour in ``halal_trader.halal.round_trip_purification``.
"""

from __future__ import annotations

import pytest

from halal_trader.halal.round_trip_purification import (
    RoundTripEntry,
    RoundTripLedger,
    RoundTripRule,
    compute_round_trip_purification,
    load_rules_from_dicts,
    outstanding_round_trip_due,
    record_round_trip,
)

# ── Pure compute ─────────────────────────────────────────────────


def test_compute_basic_rounding() -> None:
    assert compute_round_trip_purification(gain_usd=100.0, impure_ratio=0.025) == 2.50


def test_compute_zero_when_loss() -> None:
    assert compute_round_trip_purification(gain_usd=-50.0, impure_ratio=0.05) == 0.0


def test_compute_zero_when_no_impure() -> None:
    assert compute_round_trip_purification(gain_usd=100.0, impure_ratio=0.0) == 0.0


def test_compute_decimal_precision() -> None:
    out = compute_round_trip_purification(gain_usd=1000.0, impure_ratio=0.01666666)
    assert out == pytest.approx(16.67, abs=0.01)


# ── Rules ────────────────────────────────────────────────────────


def test_rule_invalid_ratio_raises() -> None:
    with pytest.raises(ValueError):
        RoundTripRule(symbol="X", impure_ratio=1.2)
    with pytest.raises(ValueError):
        RoundTripRule(symbol="X", impure_ratio=-0.1)


def test_load_rules_from_dicts() -> None:
    rules = load_rules_from_dicts(
        [
            {"symbol": "aapl", "impure_ratio": 0.02, "source": "aaoifi"},
            {"symbol": "msft", "impure_ratio": 0.005},
            {"symbol": "", "impure_ratio": 0.5},
            {"symbol": "junk", "impure_ratio": "not-a-number"},
        ]
    )
    assert "AAPL" in rules
    assert rules["AAPL"].impure_ratio == 0.02
    assert "MSFT" in rules
    assert "JUNK" not in rules


# ── Ledger ───────────────────────────────────────────────────────


async def test_ledger_record_idempotent(engine) -> None:
    led = RoundTripLedger(engine=engine)
    e = RoundTripEntry(
        entry_id="AAPL:t1",
        symbol="AAPL",
        gain_amount_usd=100,
        impure_ratio=0.02,
        purification_due_usd=2.0,
        timestamp="2026-04-26T00:00:00+00:00",
    )
    assert await led.record(e) is True
    assert await led.record(e) is False
    assert await led.outstanding() == 2.0


async def test_ledger_disburse_marks_entry(engine) -> None:
    led = RoundTripLedger(engine=engine)
    e = RoundTripEntry(
        entry_id="X:1",
        symbol="X",
        gain_amount_usd=100,
        impure_ratio=0.05,
        purification_due_usd=5.0,
        timestamp="2026-04-26T00:00:00+00:00",
    )
    await led.record(e)
    assert await led.outstanding() == 5.0
    assert await led.disbursed_total() == 0.0
    assert await led.mark_disbursed("X:1", to="charity-x") is True
    assert await led.outstanding() == 0.0
    assert await led.disbursed_total() == 5.0
    assert await led.mark_disbursed("X:1") is False


async def test_ledger_by_symbol_aggregates_outstanding(engine) -> None:
    led = RoundTripLedger(engine=engine)
    for entry_id, sym, due in [
        ("AAPL:1", "AAPL", 2.0),
        ("AAPL:2", "AAPL", 4.0),
        ("MSFT:1", "MSFT", 1.0),
    ]:
        await led.record(
            RoundTripEntry(
                entry_id=entry_id,
                symbol=sym,
                gain_amount_usd=100,
                impure_ratio=0.02,
                purification_due_usd=due,
                timestamp="2026-04-26T00:00:00+00:00",
            )
        )
    await led.mark_disbursed("AAPL:2")
    by_sym = await led.by_symbol()
    assert by_sym == {"AAPL": 2.0, "MSFT": 1.0}


# ── record_round_trip ────────────────────────────────────────────


async def test_record_round_trip_no_rule_returns_none(engine) -> None:
    led = RoundTripLedger(engine=engine)
    out = await record_round_trip(led, {}, trade_id="t1", symbol="AAPL", gain_usd=100)
    assert out is None
    assert await led.outstanding() == 0.0


async def test_record_round_trip_zero_ratio_returns_none(engine) -> None:
    led = RoundTripLedger(engine=engine)
    rules = {"AAPL": RoundTripRule(symbol="AAPL", impure_ratio=0.0)}
    out = await record_round_trip(led, rules, trade_id="t1", symbol="AAPL", gain_usd=100)
    assert out is None


async def test_record_round_trip_loss_returns_none(engine) -> None:
    led = RoundTripLedger(engine=engine)
    rules = {"AAPL": RoundTripRule(symbol="AAPL", impure_ratio=0.02)}
    out = await record_round_trip(led, rules, trade_id="t1", symbol="AAPL", gain_usd=-50)
    assert out is None


async def test_record_round_trip_writes_entry(engine) -> None:
    led = RoundTripLedger(engine=engine)
    rules = {"AAPL": RoundTripRule(symbol="AAPL", impure_ratio=0.02)}
    out = await record_round_trip(led, rules, trade_id="t1", symbol="AAPL", gain_usd=100)
    assert out is not None
    assert out.purification_due_usd == 2.0
    assert await led.outstanding() == 2.0
    out2 = await record_round_trip(led, rules, trade_id="t1", symbol="AAPL", gain_usd=100)
    assert out2 is None
    assert await led.outstanding() == 2.0


async def test_outstanding_summary_shape(engine) -> None:
    led = RoundTripLedger(engine=engine)
    rules = {"AAPL": RoundTripRule(symbol="AAPL", impure_ratio=0.02)}
    await record_round_trip(led, rules, trade_id="t1", symbol="AAPL", gain_usd=100)
    await record_round_trip(led, rules, trade_id="t2", symbol="AAPL", gain_usd=200)
    summary = await outstanding_round_trip_due(led)
    assert summary["total_usd"] == 6.0
    assert summary["by_symbol"]["AAPL"] == 6.0
    assert summary["n_entries"] == 2
    assert summary["disbursed_total_usd"] == 0.0
