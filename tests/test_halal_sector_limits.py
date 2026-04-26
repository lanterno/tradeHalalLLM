"""Sector-rotation halal limit tests."""

from __future__ import annotations

from halal_trader.halal.sector_limits import (
    UNKNOWN_SECTOR,
    check_buy_against_limits,
    compute_allocation,
    sector_for,
)


def test_sector_for_known_symbol():
    assert sector_for("AAPL") == "Technology"
    assert sector_for("XOM") == "Energy"


def test_sector_for_unknown_falls_back():
    assert sector_for("ZZZZ") == UNKNOWN_SECTOR


def test_compute_allocation_buckets_by_sector():
    alloc = compute_allocation(
        {"AAPL": 5_000, "MSFT": 3_000, "JNJ": 2_000},
        total_equity=20_000,
    )
    assert alloc.by_sector["Technology"] == 8_000
    assert alloc.by_sector["Healthcare"] == 2_000
    assert alloc.pct("Technology") == 0.40


def test_check_buy_allowed_under_cap():
    alloc = compute_allocation({"AAPL": 5_000}, total_equity=20_000)
    ok, reason = check_buy_against_limits(
        symbol="MSFT", notional_usd=2_000, allocation=alloc, max_sector_pct=0.50
    )
    assert ok is True
    assert reason == ""


def test_check_buy_blocked_when_post_trade_breaches_cap():
    alloc = compute_allocation({"AAPL": 7_000}, total_equity=20_000)
    ok, reason = check_buy_against_limits(
        symbol="MSFT", notional_usd=2_000, allocation=alloc, max_sector_pct=0.40
    )
    # 7k + 2k = 9k = 45% > 40% cap.
    assert ok is False
    assert "Technology" in reason
    assert "40%" in reason


def test_check_buy_uses_post_trade_total_not_pre():
    """A 1% buy on top of 39% should still trip a 40% cap (post = 40.05%)."""
    alloc = compute_allocation({"AAPL": 7_800}, total_equity=20_000)  # 39%
    ok, _ = check_buy_against_limits(
        symbol="MSFT", notional_usd=210, allocation=alloc, max_sector_pct=0.40
    )
    assert ok is False


def test_check_buy_against_zero_equity_allows():
    """Cold start (no equity) shouldn't block trades — defensive default."""
    alloc = compute_allocation({}, total_equity=0)
    ok, _ = check_buy_against_limits(
        symbol="AAPL", notional_usd=100, allocation=alloc, max_sector_pct=0.40
    )
    assert ok is True


def test_unknown_symbols_share_unknown_bucket():
    alloc = compute_allocation(
        {"ZZZZ": 5_000, "YYYY": 5_000},
        total_equity=20_000,
    )
    assert alloc.by_sector[UNKNOWN_SECTOR] == 10_000
