"""Tests for sentiment/insider_pattern.py — Round-5 Wave 11.F."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.sentiment.insider_pattern import (
    ClusterPattern,
    DetectorPolicy,
    Direction,
    InsiderTrade,
    detect,
    render_detection,
)


def _trade(
    trade_id: str = "T-1",
    insider_handle: str = "i-A",
    symbol: str = "AAPL",
    direction: Direction = Direction.BUY,
    shares: float = 1000,
    trade_date: date = date(2026, 5, 1),
) -> InsiderTrade:
    return InsiderTrade(
        trade_id=trade_id,
        insider_handle=insider_handle,
        symbol=symbol,
        direction=direction,
        shares=shares,
        trade_date=trade_date,
    )


# --- Validation -----------------------------


def test_direction_string_values():
    assert Direction.BUY.value == "buy"
    assert Direction.SELL.value == "sell"


def test_pattern_string_values():
    assert ClusterPattern.NORMAL.value == "normal"
    assert ClusterPattern.CLUSTER_BUY.value == "cluster_buy"
    assert ClusterPattern.CLUSTER_SELL.value == "cluster_sell"
    assert ClusterPattern.PRE_NEWS_SALE.value == "pre_news_sale"


def test_default_policy():
    p = DetectorPolicy()
    assert p.cluster_window_days == 5
    assert p.min_trades_for_cluster == 3


def test_policy_low_min_trades_rejected():
    with pytest.raises(ValueError):
        DetectorPolicy(min_trades_for_cluster=1)


def test_policy_low_min_insiders_rejected():
    with pytest.raises(ValueError):
        DetectorPolicy(min_distinct_insiders=1)


def test_trade_empty_id_rejected():
    with pytest.raises(ValueError):
        _trade(trade_id="")


def test_trade_zero_shares_rejected():
    with pytest.raises(ValueError):
        _trade(shares=0)


# --- Detection -------------------------


def test_no_trades_normal():
    d = detect("AAPL", [])
    assert d.pattern is ClusterPattern.NORMAL
    assert d.n_trades == 0


def test_filters_by_symbol():
    trades = [_trade(symbol="MSFT") for _ in range(5)]
    d = detect("AAPL", trades)
    assert d.n_trades == 0


def test_below_threshold_normal():
    """Two trades by two insiders → below 3-trade threshold."""
    trades = [
        _trade("T1", "i-A", direction=Direction.BUY),
        _trade("T2", "i-B", direction=Direction.BUY, trade_date=date(2026, 5, 2)),
    ]
    d = detect("AAPL", trades)
    assert d.pattern is ClusterPattern.NORMAL


def test_cluster_buy_detected():
    """3+ buys by 2+ insiders within 5 days → CLUSTER_BUY."""
    trades = [
        _trade("T1", "i-A", direction=Direction.BUY, trade_date=date(2026, 5, 1)),
        _trade("T2", "i-B", direction=Direction.BUY, trade_date=date(2026, 5, 2)),
        _trade("T3", "i-C", direction=Direction.BUY, trade_date=date(2026, 5, 3)),
    ]
    d = detect("AAPL", trades)
    assert d.pattern is ClusterPattern.CLUSTER_BUY
    assert d.n_distinct_insiders == 3


def test_cluster_sell_detected():
    trades = [
        _trade("T1", "i-A", direction=Direction.SELL, trade_date=date(2026, 5, 1)),
        _trade("T2", "i-B", direction=Direction.SELL, trade_date=date(2026, 5, 2)),
        _trade("T3", "i-C", direction=Direction.SELL, trade_date=date(2026, 5, 3)),
    ]
    d = detect("AAPL", trades)
    assert d.pattern is ClusterPattern.CLUSTER_SELL


def test_one_insider_many_trades_not_cluster():
    """3 trades but only 1 insider → not a cluster."""
    trades = [
        _trade("T1", "i-A", direction=Direction.BUY, trade_date=date(2026, 5, 1)),
        _trade("T2", "i-A", direction=Direction.BUY, trade_date=date(2026, 5, 2)),
        _trade("T3", "i-A", direction=Direction.BUY, trade_date=date(2026, 5, 3)),
    ]
    d = detect("AAPL", trades)
    assert d.pattern is ClusterPattern.NORMAL


def test_outside_window_not_cluster():
    """3 trades spread over 30 days → outside 5-day window."""
    trades = [
        _trade("T1", "i-A", direction=Direction.BUY, trade_date=date(2026, 5, 1)),
        _trade("T2", "i-B", direction=Direction.BUY, trade_date=date(2026, 5, 15)),
        _trade("T3", "i-C", direction=Direction.BUY, trade_date=date(2026, 5, 30)),
    ]
    d = detect("AAPL", trades)
    assert d.pattern is ClusterPattern.NORMAL


def test_pre_news_sale_pattern():
    """Multiple insider sales in week before known upcoming news → PRE_NEWS_SALE."""
    trades = [
        _trade("T1", "i-A", direction=Direction.SELL, trade_date=date(2026, 5, 25)),
        _trade("T2", "i-B", direction=Direction.SELL, trade_date=date(2026, 5, 26)),
        _trade("T3", "i-C", direction=Direction.SELL, trade_date=date(2026, 5, 27)),
    ]
    d = detect("AAPL", trades, upcoming_news_date=date(2026, 5, 31))
    assert d.pattern is ClusterPattern.PRE_NEWS_SALE


def test_pre_news_buys_not_flagged():
    """Pre-news buying is not the suspicious pattern."""
    trades = [
        _trade("T1", "i-A", direction=Direction.BUY, trade_date=date(2026, 5, 25)),
        _trade("T2", "i-B", direction=Direction.BUY, trade_date=date(2026, 5, 26)),
        _trade("T3", "i-C", direction=Direction.BUY, trade_date=date(2026, 5, 27)),
    ]
    d = detect("AAPL", trades, upcoming_news_date=date(2026, 5, 31))
    # Pre-news SALE pattern doesn't fire on BUYS — this clusters as CLUSTER_BUY
    assert d.pattern is ClusterPattern.CLUSTER_BUY


# --- Render --------------------------


def test_render_normal():
    d = detect("AAPL", [_trade()])
    out = render_detection(d)
    assert "AAPL" in out


def test_render_cluster_buy_emoji():
    trades = [
        _trade("T1", "i-A", direction=Direction.BUY, trade_date=date(2026, 5, 1)),
        _trade("T2", "i-B", direction=Direction.BUY, trade_date=date(2026, 5, 2)),
        _trade("T3", "i-C", direction=Direction.BUY, trade_date=date(2026, 5, 3)),
    ]
    d = detect("AAPL", trades)
    out = render_detection(d)
    assert "🟢" in out


def test_render_pre_news_alert_emoji():
    trades = [
        _trade("T1", "i-A", direction=Direction.SELL, trade_date=date(2026, 5, 25)),
        _trade("T2", "i-B", direction=Direction.SELL, trade_date=date(2026, 5, 26)),
        _trade("T3", "i-C", direction=Direction.SELL, trade_date=date(2026, 5, 27)),
    ]
    d = detect("AAPL", trades, upcoming_news_date=date(2026, 5, 31))
    out = render_detection(d)
    assert "🚨" in out


def test_render_no_secret_leak():
    d = detect("AAPL", [_trade()])
    out = render_detection(d)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization", "SSN"):
        assert token not in out


# --- E2E -----------------------


def test_e2e_real_world_cluster_buy():
    """Three c-suite execs all buy stock within a week — strong bullish signal."""
    trades = [
        _trade("T1", "ceo-handle", direction=Direction.BUY, trade_date=date(2026, 5, 10)),
        _trade("T2", "cfo-handle", direction=Direction.BUY, trade_date=date(2026, 5, 11)),
        _trade("T3", "cto-handle", direction=Direction.BUY, trade_date=date(2026, 5, 12)),
    ]
    d = detect("AAPL", trades)
    assert d.pattern is ClusterPattern.CLUSTER_BUY


def test_replay_consistency():
    trades = [_trade()]
    a = detect("AAPL", trades)
    b = detect("AAPL", trades)
    assert a == b
