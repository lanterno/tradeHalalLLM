"""Tests for round-trip purification accounting.

Note: the existing ``halal_trader.halal.purification`` module covers
the *dividend* flavour; this test set covers the new round-trip
(capital-gains) flavour in ``halal_trader.halal.round_trip_purification``.
"""

from __future__ import annotations

from pathlib import Path

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
            {"symbol": "", "impure_ratio": 0.5},  # skipped (no symbol)
            {"symbol": "junk", "impure_ratio": "not-a-number"},  # skipped (bad ratio)
        ]
    )
    assert "AAPL" in rules
    assert rules["AAPL"].impure_ratio == 0.02
    assert "MSFT" in rules
    assert "JUNK" not in rules


# ── Ledger ───────────────────────────────────────────────────────


def test_ledger_record_idempotent(tmp_path: Path) -> None:
    led = RoundTripLedger(path=tmp_path / "purif.json")
    e = RoundTripEntry(
        entry_id="AAPL:t1",
        symbol="AAPL",
        gain_amount_usd=100,
        impure_ratio=0.02,
        purification_due_usd=2.0,
        timestamp="2026-04-26T00:00:00Z",
    )
    assert led.record(e) is True
    assert led.record(e) is False
    assert led.outstanding() == 2.0


def test_ledger_disburse_marks_entry(tmp_path: Path) -> None:
    led = RoundTripLedger(path=tmp_path / "purif.json")
    e = RoundTripEntry(
        entry_id="X:1",
        symbol="X",
        gain_amount_usd=100,
        impure_ratio=0.05,
        purification_due_usd=5.0,
        timestamp="2026-04-26T00:00:00Z",
    )
    led.record(e)
    assert led.outstanding() == 5.0
    assert led.disbursed_total() == 0.0
    assert led.mark_disbursed("X:1", to="charity-x") is True
    assert led.outstanding() == 0.0
    assert led.disbursed_total() == 5.0
    assert led.mark_disbursed("X:1") is False


def test_ledger_persists_across_instances(tmp_path: Path) -> None:
    p = tmp_path / "purif.json"
    led1 = RoundTripLedger(path=p)
    led1.record(
        RoundTripEntry(
            entry_id="X:1",
            symbol="X",
            gain_amount_usd=100,
            impure_ratio=0.02,
            purification_due_usd=2.0,
            timestamp="2026-04-26T00:00:00Z",
        )
    )
    led2 = RoundTripLedger(path=p)
    assert led2.outstanding() == 2.0


def test_ledger_resilient_to_corrupt_file(tmp_path: Path) -> None:
    p = tmp_path / "purif.json"
    p.write_text("{not json")
    led = RoundTripLedger(path=p)
    assert led.outstanding() == 0.0


def test_ledger_by_symbol_aggregates_outstanding(tmp_path: Path) -> None:
    led = RoundTripLedger(path=tmp_path / "purif.json")
    led.record(
        RoundTripEntry(
            entry_id="AAPL:1",
            symbol="AAPL",
            gain_amount_usd=100,
            impure_ratio=0.02,
            purification_due_usd=2.0,
            timestamp="t",
        )
    )
    led.record(
        RoundTripEntry(
            entry_id="AAPL:2",
            symbol="AAPL",
            gain_amount_usd=200,
            impure_ratio=0.02,
            purification_due_usd=4.0,
            timestamp="t",
        )
    )
    led.record(
        RoundTripEntry(
            entry_id="MSFT:1",
            symbol="MSFT",
            gain_amount_usd=100,
            impure_ratio=0.01,
            purification_due_usd=1.0,
            timestamp="t",
        )
    )
    led.mark_disbursed("AAPL:2")
    by_sym = led.by_symbol()
    assert by_sym == {"AAPL": 2.0, "MSFT": 1.0}


# ── record_round_trip ────────────────────────────────────────────


def test_record_round_trip_no_rule_returns_none(tmp_path: Path) -> None:
    led = RoundTripLedger(path=tmp_path / "purif.json")
    out = record_round_trip(led, {}, trade_id="t1", symbol="AAPL", gain_usd=100)
    assert out is None
    assert led.outstanding() == 0.0


def test_record_round_trip_zero_ratio_returns_none(tmp_path: Path) -> None:
    led = RoundTripLedger(path=tmp_path / "purif.json")
    rules = {"AAPL": RoundTripRule(symbol="AAPL", impure_ratio=0.0)}
    out = record_round_trip(led, rules, trade_id="t1", symbol="AAPL", gain_usd=100)
    assert out is None


def test_record_round_trip_loss_returns_none(tmp_path: Path) -> None:
    led = RoundTripLedger(path=tmp_path / "purif.json")
    rules = {"AAPL": RoundTripRule(symbol="AAPL", impure_ratio=0.02)}
    out = record_round_trip(led, rules, trade_id="t1", symbol="AAPL", gain_usd=-50)
    assert out is None


def test_record_round_trip_writes_entry(tmp_path: Path) -> None:
    led = RoundTripLedger(path=tmp_path / "purif.json")
    rules = {"AAPL": RoundTripRule(symbol="AAPL", impure_ratio=0.02)}
    out = record_round_trip(led, rules, trade_id="t1", symbol="AAPL", gain_usd=100)
    assert out is not None
    assert out.purification_due_usd == 2.0
    assert led.outstanding() == 2.0
    out2 = record_round_trip(led, rules, trade_id="t1", symbol="AAPL", gain_usd=100)
    assert out2 is None
    assert led.outstanding() == 2.0


def test_outstanding_summary_shape(tmp_path: Path) -> None:
    led = RoundTripLedger(path=tmp_path / "purif.json")
    rules = {"AAPL": RoundTripRule(symbol="AAPL", impure_ratio=0.02)}
    record_round_trip(led, rules, trade_id="t1", symbol="AAPL", gain_usd=100)
    record_round_trip(led, rules, trade_id="t2", symbol="AAPL", gain_usd=200)
    summary = outstanding_round_trip_due(led)
    assert summary["total_usd"] == 6.0
    assert summary["by_symbol"]["AAPL"] == 6.0
    assert summary["n_entries"] == 2
    assert summary["disbursed_total_usd"] == 0.0
