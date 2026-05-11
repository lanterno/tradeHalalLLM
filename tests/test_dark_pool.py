"""Tests for trading/dark_pool.py — Round-5 Wave 12.F."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from halal_trader.trading.dark_pool import (
    PrintType,
    RouteVerdict,
    TickerDarkProfile,
    TradePrint,
    classify_print,
    decide_route,
    filter_lit_only,
    is_dark,
    render_decision,
    render_profile,
    summarise_ticker,
)


def _print(
    print_id: str = "P1",
    ticker: str = "AAPL",
    timestamp: datetime = datetime(2026, 5, 10, 14, 0, 0),
    price: float = 200.0,
    quantity: float = 100.0,
    venue: str = "NSDQ",
    is_dark_indicator: bool = False,
    is_internal_indicator: bool = False,
    is_auction_indicator: bool = False,
) -> TradePrint:
    return TradePrint(
        print_id=print_id,
        ticker=ticker,
        timestamp=timestamp,
        price=price,
        quantity=quantity,
        venue_code=venue,
        is_dark_indicator=is_dark_indicator,
        is_internal_indicator=is_internal_indicator,
        is_auction_indicator=is_auction_indicator,
    )


# --- TradePrint validation ---------------------------------------------


def test_print_valid():
    p = _print()
    assert p.ticker == "AAPL"


def test_print_empty_id_rejected():
    with pytest.raises(ValueError):
        _print(print_id="")


def test_print_empty_venue_rejected():
    with pytest.raises(ValueError):
        _print(venue=" ")


def test_print_zero_price_rejected():
    with pytest.raises(ValueError):
        _print(price=0)


def test_print_negative_quantity_rejected():
    with pytest.raises(ValueError):
        _print(quantity=-1.0)


def test_print_immutable():
    p = _print()
    with pytest.raises(AttributeError):
        p.price = 0  # type: ignore[misc]


# --- classify_print + is_dark -----------------------------------------


def test_classify_lit_default():
    p = _print()
    assert classify_print(p) is PrintType.LIT


def test_classify_dark_trf():
    p = _print(is_dark_indicator=True)
    assert classify_print(p) is PrintType.DARK_TRF


def test_classify_dark_internal_priority():
    """Pin: internal indicator wins over dark indicator."""
    p = _print(is_dark_indicator=True, is_internal_indicator=True)
    assert classify_print(p) is PrintType.DARK_INTERNAL


def test_classify_auction_above_dark():
    """Pin: auction wins over dark when both flagged."""
    p = _print(is_dark_indicator=True, is_auction_indicator=True)
    assert classify_print(p) is PrintType.AUCTION


def test_is_dark_helper():
    assert is_dark(PrintType.DARK_TRF)
    assert is_dark(PrintType.DARK_INTERNAL)
    assert not is_dark(PrintType.LIT)
    assert not is_dark(PrintType.AUCTION)


# --- summarise_ticker --------------------------------------------------


def test_summarise_basic():
    base = datetime(2026, 5, 10, 14, 0, 0)
    prints = [
        _print(print_id="P1", quantity=100, timestamp=base + timedelta(seconds=1)),
        _print(
            print_id="P2",
            quantity=200,
            timestamp=base + timedelta(seconds=2),
            is_dark_indicator=True,
        ),
        _print(print_id="P3", quantity=200, timestamp=base + timedelta(seconds=3)),
    ]
    profile = summarise_ticker(
        prints,
        ticker="AAPL",
        window_seconds=60,
        as_of=base + timedelta(seconds=10),
    )
    assert profile.n_prints == 3
    assert profile.total_volume == 500.0
    assert profile.dark_volume == 200.0
    assert profile.dark_fraction == pytest.approx(0.40)


def test_summarise_window_excludes_old():
    base = datetime(2026, 5, 10, 14, 0, 0)
    prints = [
        _print(
            print_id="P0",
            quantity=1000,
            timestamp=base - timedelta(seconds=120),
            is_dark_indicator=True,
        ),
        _print(print_id="P1", quantity=100, timestamp=base),
    ]
    profile = summarise_ticker(prints, ticker="AAPL", window_seconds=60, as_of=base)
    assert profile.n_prints == 1
    assert profile.dark_volume == 0.0


def test_summarise_filters_by_ticker():
    base = datetime(2026, 5, 10, 14, 0, 0)
    prints = [
        _print(print_id="P1", ticker="AAPL", quantity=100, timestamp=base),
        _print(print_id="P2", ticker="MSFT", quantity=500, timestamp=base),
    ]
    profile = summarise_ticker(
        prints, ticker="AAPL", window_seconds=60, as_of=base + timedelta(seconds=5)
    )
    assert profile.total_volume == 100.0


def test_summarise_invalid_window_rejected():
    with pytest.raises(ValueError):
        summarise_ticker([], ticker="AAPL", window_seconds=0, as_of=datetime.now())


def test_summarise_empty_returns_zero_fractions():
    profile = summarise_ticker([], ticker="AAPL", window_seconds=60, as_of=datetime(2026, 5, 10))
    assert profile.n_prints == 0
    assert profile.dark_fraction == 0.0


def test_summarise_internal_fraction_pinned():
    base = datetime(2026, 5, 10, 14, 0, 0)
    prints = [
        _print(
            print_id="P1",
            quantity=100,
            timestamp=base,
            is_internal_indicator=True,
        ),
        _print(
            print_id="P2",
            quantity=100,
            timestamp=base,
            is_dark_indicator=True,
        ),
    ]
    profile = summarise_ticker(
        prints, ticker="AAPL", window_seconds=60, as_of=base + timedelta(seconds=5)
    )
    # internal_fraction is internal / total = 100/200 = 0.5.
    assert profile.internal_fraction == pytest.approx(0.50)
    # dark_fraction also captures internal (it's a dark type).
    assert profile.dark_fraction == pytest.approx(1.0)


# --- decide_route -----------------------------------------------------


def _profile(
    dark_fraction: float = 0.10,
    ticker: str = "AAPL",
) -> TickerDarkProfile:
    return TickerDarkProfile(
        ticker=ticker,
        window_seconds=60,
        n_prints=10,
        total_volume=1000.0,
        dark_volume=dark_fraction * 1000.0,
        dark_fraction=dark_fraction,
        internal_fraction=0.0,
    )


def test_decide_allow_normal_below_warn():
    decision = decide_route(_profile(dark_fraction=0.20))
    assert decision.verdict is RouteVerdict.ALLOW


def test_decide_warn_normal_between_30_50():
    decision = decide_route(_profile(dark_fraction=0.40))
    assert decision.verdict is RouteVerdict.WARN


def test_decide_opt_out_normal_above_50():
    decision = decide_route(_profile(dark_fraction=0.55))
    assert decision.verdict is RouteVerdict.OPT_OUT


def test_decide_halal_sensitive_warn_at_15pct():
    decision = decide_route(_profile(dark_fraction=0.15), halal_sensitive=True)
    assert decision.verdict is RouteVerdict.WARN


def test_decide_halal_sensitive_opt_out_at_30pct():
    decision = decide_route(_profile(dark_fraction=0.30), halal_sensitive=True)
    assert decision.verdict is RouteVerdict.OPT_OUT


def test_decide_overrides_used_when_passed():
    decision = decide_route(
        _profile(dark_fraction=0.25),
        overrides=(0.20, 0.40),
    )
    assert decision.verdict is RouteVerdict.WARN


def test_decide_invalid_overrides_rejected():
    with pytest.raises(ValueError):
        decide_route(_profile(), overrides=(0.5, 0.3))


def test_decide_reason_explains():
    d_warn = decide_route(_profile(dark_fraction=0.40))
    assert "dark_fraction" in d_warn.reason
    d_allow = decide_route(_profile(dark_fraction=0.10))
    assert "tolerance" in d_allow.reason


# --- filter_lit_only --------------------------------------------------


def test_filter_lit_only():
    prints = [
        _print(print_id="P1"),
        _print(print_id="P2", is_dark_indicator=True),
        _print(print_id="P3", is_internal_indicator=True),
        _print(print_id="P4", is_auction_indicator=True),
    ]
    out = filter_lit_only(prints)
    ids = {p.print_id for p in out}
    assert "P1" in ids
    assert "P4" in ids  # auction is allowed
    assert "P2" not in ids
    assert "P3" not in ids


# --- Render -----------------------------------------------------------


def test_render_decision_emoji_per_verdict():
    d_allow = decide_route(_profile(dark_fraction=0.10))
    d_warn = decide_route(_profile(dark_fraction=0.40))
    d_opt = decide_route(_profile(dark_fraction=0.60))
    assert "✅" in render_decision(d_allow)
    assert "⚠️" in render_decision(d_warn)
    assert "🛑" in render_decision(d_opt)


def test_render_decision_marks_halal_sensitive():
    d = decide_route(_profile(dark_fraction=0.05), halal_sensitive=True)
    assert "halal-sensitive" in render_decision(d)


def test_render_profile_format():
    p = _profile(dark_fraction=0.25)
    out = render_profile(p)
    assert "AAPL" in out
    assert "25.00%" in out
