"""Tests for trading/post_trade.py — Round-5 Wave 12.H."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from halal_trader.trading.post_trade import (
    Benchmark,
    ExecutionInputs,
    ExecutionReport,
    Fill,
    analyze,
    render_report,
)
from halal_trader.trading.twap import Side


def _fill(
    fill_id: str = "F1",
    qty: float = 100,
    price: float = 100.0,
    t: datetime = datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
) -> Fill:
    return Fill(fill_id=fill_id, quantity=qty, price=price, fill_time=t)


def _inputs(**overrides) -> ExecutionInputs:
    base = {
        "parent_id": "P-001",
        "symbol": "AAPL",
        "side": Side.BUY,
        "fills": (_fill(),),
        "arrival_price": 100.0,
        "twap_price": None,
        "vwap_price": None,
        "close_price": None,
    }
    base.update(overrides)
    return ExecutionInputs(**base)


# --- Validation ---------------------------------------------------------


def test_benchmark_string_values():
    assert Benchmark.ARRIVAL.value == "arrival"
    assert Benchmark.TWAP.value == "twap"
    assert Benchmark.VWAP.value == "vwap"
    assert Benchmark.CLOSE.value == "close"


def test_fill_empty_id_rejected():
    with pytest.raises(ValueError):
        _fill(fill_id="")


def test_fill_zero_qty_rejected():
    with pytest.raises(ValueError):
        _fill(qty=0)


def test_fill_zero_price_rejected():
    with pytest.raises(ValueError):
        _fill(price=0)


def test_fill_naive_time_rejected():
    with pytest.raises(ValueError):
        Fill(
            fill_id="F1",
            quantity=10,
            price=100,
            fill_time=datetime(2026, 5, 5, 9, 30),
        )


def test_inputs_empty_fills_rejected():
    with pytest.raises(ValueError):
        _inputs(fills=())


def test_inputs_zero_arrival_rejected():
    with pytest.raises(ValueError):
        _inputs(arrival_price=0)


def test_inputs_negative_twap_rejected():
    with pytest.raises(ValueError):
        _inputs(twap_price=-1.0)


# --- Slippage analysis -------------------------------------------------


def test_buy_above_arrival_positive_slippage():
    """Buy filled at 101 vs arrival 100 → positive (bad) slippage."""
    fills = (_fill(price=101.0),)
    report = analyze(_inputs(fills=fills, arrival_price=100.0))
    assert report.arrival_slippage_bps == pytest.approx(100.0)  # 1% = 100 bps


def test_buy_below_arrival_negative_slippage():
    """Buy filled at 99 vs arrival 100 → negative (favourable) slippage."""
    fills = (_fill(price=99.0),)
    report = analyze(_inputs(fills=fills, arrival_price=100.0))
    assert report.arrival_slippage_bps == pytest.approx(-100.0)


def test_sell_below_arrival_positive_slippage():
    """Sell filled at 99 vs arrival 100 → positive (bad) slippage for sell."""
    fills = (_fill(price=99.0),)
    report = analyze(_inputs(fills=fills, arrival_price=100.0, side=Side.SELL))
    assert report.arrival_slippage_bps == pytest.approx(100.0)


def test_sell_above_arrival_negative_slippage():
    fills = (_fill(price=101.0),)
    report = analyze(_inputs(fills=fills, arrival_price=100.0, side=Side.SELL))
    assert report.arrival_slippage_bps == pytest.approx(-100.0)


def test_total_quantity_sums_fills():
    fills = (
        _fill(qty=100),
        _fill("F2", qty=50, t=datetime(2026, 5, 5, 9, 35, tzinfo=timezone.utc)),
        _fill("F3", qty=25, t=datetime(2026, 5, 5, 9, 40, tzinfo=timezone.utc)),
    )
    report = analyze(_inputs(fills=fills))
    assert report.total_quantity == 175


def test_avg_fill_price_volume_weighted():
    """50 @ 100 + 50 @ 102 = avg 101."""
    fills = (
        _fill(qty=50, price=100),
        _fill("F2", qty=50, price=102, t=datetime(2026, 5, 5, 9, 35, tzinfo=timezone.utc)),
    )
    report = analyze(_inputs(fills=fills))
    assert report.average_fill_price == 101


def test_market_impact_positive_for_above_arrival_buy():
    fills = (_fill(price=102.0),)
    report = analyze(_inputs(fills=fills, arrival_price=100.0))
    assert report.market_impact_pct == pytest.approx(0.02)


def test_market_impact_signed_for_sell():
    fills = (_fill(price=98.0),)
    report = analyze(_inputs(fills=fills, arrival_price=100.0, side=Side.SELL))
    assert report.market_impact_pct == pytest.approx(0.02)


def test_fill_duration_first_to_last():
    fills = (
        _fill(qty=50, t=datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc)),
        _fill("F2", qty=50, t=datetime(2026, 5, 5, 9, 45, tzinfo=timezone.utc)),
    )
    report = analyze(_inputs(fills=fills))
    assert report.fill_duration_seconds == pytest.approx(15 * 60)


def test_fill_duration_single_fill_zero():
    fills = (_fill(),)
    report = analyze(_inputs(fills=fills))
    assert report.fill_duration_seconds == 0


# --- Optional benchmarks ----------------------------------------------


def test_twap_benchmark_emitted_when_provided():
    fills = (_fill(price=100.5),)
    report = analyze(_inputs(fills=fills, arrival_price=100.0, twap_price=100.0))
    assert report.twap_slippage_bps == pytest.approx(50.0)


def test_vwap_benchmark_emitted_when_provided():
    fills = (_fill(price=100.5),)
    report = analyze(_inputs(fills=fills, arrival_price=100.0, vwap_price=100.0))
    assert report.vwap_slippage_bps == pytest.approx(50.0)


def test_close_benchmark_emitted_when_provided():
    fills = (_fill(price=100.5),)
    report = analyze(_inputs(fills=fills, arrival_price=100.0, close_price=100.0))
    assert report.close_slippage_bps == pytest.approx(50.0)


def test_optional_benchmarks_omitted_when_none():
    report = analyze(_inputs())
    assert report.twap_slippage_bps is None
    assert report.vwap_slippage_bps is None
    assert report.close_slippage_bps is None


# --- Report invariants ------------------------------------------------


def test_report_zero_quantity_rejected():
    with pytest.raises(ValueError):
        ExecutionReport(
            parent_id="P",
            side=Side.BUY,
            total_quantity=0,
            average_fill_price=100,
            arrival_slippage_bps=0,
            twap_slippage_bps=None,
            vwap_slippage_bps=None,
            close_slippage_bps=None,
            market_impact_pct=0,
            fill_duration_seconds=10,
        )


def test_report_zero_avg_fill_rejected():
    with pytest.raises(ValueError):
        ExecutionReport(
            parent_id="P",
            side=Side.BUY,
            total_quantity=10,
            average_fill_price=0,
            arrival_slippage_bps=0,
            twap_slippage_bps=None,
            vwap_slippage_bps=None,
            close_slippage_bps=None,
            market_impact_pct=0,
            fill_duration_seconds=10,
        )


# --- Render -----------------------------------------------------------


def test_render_includes_summary():
    fills = (_fill(price=100.5),)
    report = analyze(
        _inputs(fills=fills, arrival_price=100.0, twap_price=100.0, vwap_price=100.0)
    )
    out = render_report(report)
    assert "Post-trade" in out
    assert "P-001" in out
    assert "TWAP slippage" in out
    assert "VWAP slippage" in out


def test_render_no_secret_leak():
    fills = (_fill(),)
    report = analyze(_inputs(fills=fills))
    out = render_report(report)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ---------------------------------------------------------


def test_e2e_multi_fill_buy_with_slippage():
    fills = (
        _fill(qty=100, price=100.10),
        _fill("F2", qty=200, price=100.15, t=datetime(2026, 5, 5, 9, 35, tzinfo=timezone.utc)),
        _fill("F3", qty=200, price=100.20, t=datetime(2026, 5, 5, 9, 40, tzinfo=timezone.utc)),
    )
    report = analyze(
        _inputs(
            fills=fills,
            arrival_price=100.00,
            twap_price=100.18,
            vwap_price=100.16,
        )
    )
    # Avg fill ≈ (100*100.10 + 200*100.15 + 200*100.20)/500 = 100.16
    assert report.average_fill_price == pytest.approx(100.16)
    assert report.arrival_slippage_bps > 0  # paid more than arrival


def test_replay_consistency():
    a = analyze(_inputs())
    b = analyze(_inputs())
    assert a == b
