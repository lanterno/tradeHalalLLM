"""Paper-vs-live slippage divergence tracker tests."""

from __future__ import annotations

from halal_trader.core.divergence import (
    TradeSlippage,
    build_report,
    compute_slippage,
    format_report,
)

# ── compute_slippage ──────────────────────────────────────────


def test_buy_filled_above_intended_is_positive():
    """Paid more than expected → positive slippage (cost us)."""
    s = compute_slippage(intended_price=100, actual_fill_price=101, side="buy")
    assert s == 0.01


def test_sell_filled_below_intended_is_positive():
    """Received less than expected → positive slippage."""
    s = compute_slippage(intended_price=100, actual_fill_price=99, side="sell")
    assert s == 0.01


def test_buy_filled_better_than_intended_negative():
    s = compute_slippage(intended_price=100, actual_fill_price=99, side="buy")
    assert s == -0.01


def test_invalid_side_returns_none():
    assert compute_slippage(intended_price=100, actual_fill_price=101, side="hold") is None


def test_invalid_price_returns_none():
    assert compute_slippage(intended_price=0, actual_fill_price=101, side="buy") is None
    assert compute_slippage(intended_price=100, actual_fill_price=0, side="buy") is None


# ── build_report ──────────────────────────────────────────────


def test_report_empty_samples_returns_zero():
    r = build_report([])
    assert r.sample_size == 0
    assert r.exceeds_threshold is False


def test_report_skips_partial_samples():
    """Trades with one side missing aren't comparable."""
    samples = [
        TradeSlippage(trade_id=1, paper_slippage_pct=0.0005, live_slippage_pct=None),
        TradeSlippage(trade_id=2, paper_slippage_pct=None, live_slippage_pct=0.0008),
        TradeSlippage(trade_id=3, paper_slippage_pct=0.0005, live_slippage_pct=0.0010),
    ]
    r = build_report(samples)
    assert r.sample_size == 1


def test_report_under_threshold_does_not_flag():
    samples = [
        TradeSlippage(trade_id=i, paper_slippage_pct=0.0005, live_slippage_pct=0.0006)
        for i in range(20)
    ]
    r = build_report(samples, threshold_bps=10)
    assert r.exceeds_threshold is False


def test_report_over_threshold_flags():
    samples = [
        TradeSlippage(trade_id=i, paper_slippage_pct=0.0005, live_slippage_pct=0.0050)
        for i in range(20)
    ]
    r = build_report(samples, threshold_bps=10)
    # mean_div = 0.0045 = 45bps → exceeds 10bps threshold.
    assert r.exceeds_threshold is True
    assert abs(r.mean_divergence_bps - 45) < 1e-6


def test_report_p95_isolates_tail():
    """A handful of outliers should pull p95 above the mean."""
    samples = [
        TradeSlippage(trade_id=i, paper_slippage_pct=0.0005, live_slippage_pct=0.0006)
        for i in range(80)
    ]
    # Five clear outliers — 5% of 100 samples = the 95th percentile lives here.
    for i in range(80, 100):
        samples.append(
            TradeSlippage(trade_id=i, paper_slippage_pct=0.0005, live_slippage_pct=0.0500)
        )
    r = build_report(samples)
    # p95 must be materially higher than mean — outliers visible in the tail.
    assert r.p95_divergence_bps > r.mean_divergence_bps


def test_format_report_includes_flag_when_exceeded():
    samples = [
        TradeSlippage(trade_id=i, paper_slippage_pct=0.0005, live_slippage_pct=0.0050)
        for i in range(10)
    ]
    text = format_report(build_report(samples, threshold_bps=10))
    assert "EXCEEDS" in text


def test_format_report_no_samples():
    text = format_report(build_report([]))
    assert "unknown" in text.lower()
