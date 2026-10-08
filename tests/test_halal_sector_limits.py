"""Sector-rotation halal limit tests."""

from __future__ import annotations

from halal_trader.halal.sector_limits import (
    DEFAULT_EXEMPT_SECTORS,
    UNKNOWN_SECTOR,
    cap_sector,
    check_buy_against_limits,
    compute_allocation,
    sector_for,
)

SECTORS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JNJ": "Healthcare",
    "LLY": "Healthcare",
    "XOM": "Energy",
}


def test_the_cap_sector_comes_from_the_screens_industry():
    assert cap_sector("AAPL", "Electronic Computers") == "Technology"
    assert cap_sector("MU", "Semiconductors & Related Devices") == "Technology"
    assert cap_sector("MSFT", "Services-Prepackaged Software") == "Technology"
    assert cap_sector("XOM", "Petroleum Refining") == "Energy"
    assert cap_sector("ABT", "Pharmaceutical Preparations") == "Healthcare"
    # Overrides: a life-science tool maker filed as measuring devices is not
    # Technology; a chip-equipment maker filed as machinery is.
    assert cap_sector("TMO", "Measuring & Controlling Devices, NEC") == "Healthcare"
    assert cap_sector("LRCX", "Special Industry Machinery, NEC") == "Technology"
    assert sector_for("MSFT", sector_map=SECTORS) == "Technology"
    assert sector_for("TMO") == "Healthcare"  # an override, without the screen


def test_sector_for_unknown_falls_back():
    assert sector_for("ZZZZ") == UNKNOWN_SECTOR


def test_compute_allocation_buckets_by_sector():
    alloc = compute_allocation(
        {"AAPL": 5_000, "MSFT": 3_000, "JNJ": 2_000}, total_equity=20_000, sector_map=SECTORS
    )
    assert alloc.by_sector["Technology"] == 8_000
    assert alloc.by_sector["Healthcare"] == 2_000
    assert alloc.pct("Technology") == 0.40


def test_check_buy_allowed_under_cap():
    alloc = compute_allocation({"AAPL": 5_000}, total_equity=20_000, sector_map=SECTORS)
    ok, reason = check_buy_against_limits(
        symbol="MSFT", notional_usd=2_000, allocation=alloc, max_sector_pct=0.50, sector_map=SECTORS
    )
    assert ok is True
    assert reason == ""


def test_check_buy_blocked_when_post_trade_breaches_cap():
    # JNJ + LLY are both Healthcare — non-exempt, so the cap fires.
    alloc = compute_allocation({"JNJ": 7_000}, total_equity=20_000, sector_map=SECTORS)
    ok, reason = check_buy_against_limits(
        symbol="LLY", notional_usd=2_000, allocation=alloc, max_sector_pct=0.40, sector_map=SECTORS
    )
    # 7k + 2k = 9k = 45% > 40% cap.
    assert ok is False
    assert "Healthcare" in reason
    assert "40%" in reason


def test_check_buy_uses_post_trade_total_not_pre():
    """A 1% buy on top of 39% should still trip a 40% cap (post = 40.05%)."""
    alloc = compute_allocation({"JNJ": 7_800}, total_equity=20_000, sector_map=SECTORS)  # 39%
    ok, _ = check_buy_against_limits(
        symbol="LLY", notional_usd=210, allocation=alloc, max_sector_pct=0.40, sector_map=SECTORS
    )
    assert ok is False


# ── Tech exemption ──────────────────────────────────────────────


def test_default_exempt_sectors_includes_technology():
    """Operator policy: halal universe is Tech-heavy so Tech is exempt."""
    assert "Technology" in DEFAULT_EXEMPT_SECTORS


def test_tech_buy_above_cap_allowed_by_default():
    """A Tech buy that would naively push past 40% is allowed because
    Technology is in DEFAULT_EXEMPT_SECTORS."""
    # 8k Tech (40%) + 5k more Tech = 65% — far past the cap.
    alloc = compute_allocation({"AAPL": 8_000}, total_equity=20_000, sector_map=SECTORS)
    ok, reason = check_buy_against_limits(
        symbol="MSFT", notional_usd=5_000, allocation=alloc, max_sector_pct=0.40, sector_map=SECTORS
    )
    assert ok is True
    assert reason == ""


def test_non_tech_still_capped_when_tech_exempt():
    """Default exemption only frees Tech — Healthcare, Energy, etc.
    still hit the cap as before."""
    alloc = compute_allocation({"JNJ": 7_000}, total_equity=20_000, sector_map=SECTORS)
    ok, _ = check_buy_against_limits(
        symbol="LLY", notional_usd=2_000, allocation=alloc, max_sector_pct=0.40, sector_map=SECTORS
    )
    assert ok is False


def test_explicit_empty_exempt_set_re_enables_tech_cap():
    """Operator escape hatch: pass exempt_sectors=frozenset() to
    re-arm the cap on every sector."""
    alloc = compute_allocation({"AAPL": 8_000}, total_equity=20_000, sector_map=SECTORS)
    ok, reason = check_buy_against_limits(
        symbol="MSFT",
        notional_usd=2_000,
        allocation=alloc,
        max_sector_pct=0.40,
        exempt_sectors=frozenset(),
        sector_map=SECTORS,
    )
    assert ok is False
    assert "Technology" in reason


def test_check_buy_against_zero_equity_allows():
    """Cold start (no equity) shouldn't block trades — defensive default."""
    alloc = compute_allocation({}, total_equity=0, sector_map=SECTORS)
    ok, _ = check_buy_against_limits(
        symbol="AAPL", notional_usd=100, allocation=alloc, max_sector_pct=0.40, sector_map=SECTORS
    )
    assert ok is True


def test_unknown_symbols_share_unknown_bucket():
    alloc = compute_allocation(
        {"ZZZZ": 5_000, "YYYY": 5_000}, total_equity=20_000, sector_map=SECTORS
    )
    assert alloc.by_sector[UNKNOWN_SECTOR] == 10_000
