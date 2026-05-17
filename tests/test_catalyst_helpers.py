"""Tests for the smaller catalyst helpers in :mod:`trading.catalysts`.

`AlpacaNewsSource`, `_parse_news_item`, `format_catalysts_for_prompt`,
and `StockCatalystFeed` are already covered in `test_stock_catalysts.py`.
This file adds the gaps: `StaticCatalystSource`, `EarningsCalendarSource`,
`CatalystRiskPolicy.size_multiplier_for`, `next_catalyst_window`, and
the `_parse_calendar_ts` parser.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.trading.catalysts import (
    Catalyst,
    CatalystRiskPolicy,
    EarningsCalendarSource,
    StaticCatalystSource,
    _parse_calendar_ts,
    next_catalyst_window,
)


def _cat(
    symbol: str = "AAPL",
    kind: str = "news",
    *,
    when: datetime | None = None,
    title: str = "headline",
) -> Catalyst:
    return Catalyst(
        symbol=symbol,
        kind=kind,
        title=title,
        timestamp=when or datetime.now(UTC),
        source="test",
    )


# ── StaticCatalystSource ─────────────────────────────────────


@pytest.mark.asyncio
async def test_static_source_returns_all_when_no_symbols():
    """Empty symbols list = "everything" (replay / dry-run uses this)."""
    cats = [_cat("AAPL"), _cat("MSFT")]
    src = StaticCatalystSource(cats)
    out = await src.fetch([])
    assert out == cats


@pytest.mark.asyncio
async def test_static_source_filters_by_symbol():
    cats = [_cat("AAPL"), _cat("MSFT"), _cat("GOOG")]
    src = StaticCatalystSource(cats)
    out = await src.fetch(["AAPL", "MSFT"])
    assert {c.symbol for c in out} == {"AAPL", "MSFT"}


@pytest.mark.asyncio
async def test_static_source_filter_is_case_insensitive():
    src = StaticCatalystSource([_cat("AAPL")])
    out = await src.fetch(["aapl"])
    assert len(out) == 1


# ── EarningsCalendarSource ──────────────────────────────────


@pytest.mark.asyncio
async def test_earnings_source_no_get_calendar_method_returns_empty():
    """Defensive: client without `get_calendar` (legacy mock) → []."""
    src = EarningsCalendarSource(client=object())
    out = await src.fetch(["AAPL"])
    assert out == []


@pytest.mark.asyncio
async def test_earnings_source_empty_symbols_returns_empty():
    client = MagicMock()
    client.get_calendar = AsyncMock()
    src = EarningsCalendarSource(client=client)
    out = await src.fetch([])
    assert out == []
    client.get_calendar.assert_not_awaited()


@pytest.mark.asyncio
async def test_earnings_source_renders_one_catalyst_per_row():
    client = MagicMock()
    client.get_calendar = AsyncMock(
        return_value=[
            {"symbol": "AAPL", "date": "2026-05-10", "eps_estimate": 1.50},
        ]
    )
    src = EarningsCalendarSource(client=client)
    out = await src.fetch(["AAPL"])
    assert len(out) == 1
    cat = out[0]
    assert cat.symbol == "AAPL"
    assert cat.kind == "earnings"
    assert "1.5" in cat.title
    assert cat.source == "alpaca-calendar"


@pytest.mark.asyncio
async def test_earnings_source_filters_to_requested_symbols():
    client = MagicMock()
    client.get_calendar = AsyncMock(
        return_value=[
            {"symbol": "AAPL", "date": "2026-05-10"},
            {"symbol": "MSFT", "date": "2026-05-10"},
        ]
    )
    src = EarningsCalendarSource(client=client)
    out = await src.fetch(["AAPL"])
    assert {c.symbol for c in out} == {"AAPL"}


@pytest.mark.asyncio
async def test_earnings_source_swallows_client_exception():
    """A blowing-up calendar API must not crash the cycle."""
    client = MagicMock()
    client.get_calendar = AsyncMock(side_effect=RuntimeError("rate limited"))
    src = EarningsCalendarSource(client=client)
    out = await src.fetch(["AAPL"])
    assert out == []


# ── _parse_calendar_ts ──────────────────────────────────────


def test_parse_calendar_ts_iso_string():
    ts = _parse_calendar_ts({"date": "2026-05-10T14:00:00+00:00"})
    assert ts == datetime(2026, 5, 10, 14, 0, tzinfo=UTC)


def test_parse_calendar_ts_z_suffix():
    """`Z` is shorthand for `+00:00`; the parser must accept it."""
    ts = _parse_calendar_ts({"date": "2026-05-10T14:00:00Z"})
    assert ts == datetime(2026, 5, 10, 14, 0, tzinfo=UTC)


def test_parse_calendar_ts_naive_iso_assumes_utc():
    """Defensive: a naive datetime gets stamped with UTC rather than
    crashing downstream tz-aware comparisons."""
    ts = _parse_calendar_ts({"date": "2026-05-10T14:00:00"})
    assert ts.tzinfo == UTC


def test_parse_calendar_ts_passes_datetime_through():
    dt = datetime(2026, 5, 10, 14, 0, tzinfo=UTC)
    assert _parse_calendar_ts({"date": dt}) == dt


def test_parse_calendar_ts_returns_none_on_garbage():
    assert _parse_calendar_ts({"date": "not-a-date"}) is None
    assert _parse_calendar_ts({}) is None
    assert _parse_calendar_ts({"date": 12345}) is None


def test_parse_calendar_ts_falls_back_through_alt_keys():
    """Alpaca uses `date`, but other sources may emit `ts` or `when`."""
    ts = _parse_calendar_ts({"ts": "2026-05-10T14:00:00Z"})
    assert ts == datetime(2026, 5, 10, 14, 0, tzinfo=UTC)


# ── CatalystRiskPolicy.size_multiplier_for ──────────────────


def test_size_multiplier_full_when_no_relevant_catalysts():
    pol = CatalystRiskPolicy()
    assert pol.size_multiplier_for("AAPL", []) == 1.0


def test_size_multiplier_shrinks_inside_pre_event_window():
    """Earnings 2 hours away → 0.5× (within the 4-hour pre-event window)."""
    now = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
    earnings = _cat("AAPL", kind="earnings", when=datetime(2026, 5, 10, 14, 0, tzinfo=UTC))
    pol = CatalystRiskPolicy()
    assert pol.size_multiplier_for("AAPL", [earnings], now=now) == 0.5


def test_size_multiplier_full_outside_pre_event_window():
    """Earnings 8 hours away → 1.0× (outside the 4h window)."""
    now = datetime(2026, 5, 10, 6, 0, tzinfo=UTC)
    earnings = _cat("AAPL", kind="earnings", when=datetime(2026, 5, 10, 14, 0, tzinfo=UTC))
    pol = CatalystRiskPolicy()
    assert pol.size_multiplier_for("AAPL", [earnings], now=now) == 1.0


def test_size_multiplier_full_for_past_catalyst():
    """Catalysts in the past don't shrink size — the event already happened."""
    now = datetime(2026, 5, 10, 18, 0, tzinfo=UTC)
    earnings = _cat("AAPL", kind="earnings", when=datetime(2026, 5, 10, 14, 0, tzinfo=UTC))
    pol = CatalystRiskPolicy()
    assert pol.size_multiplier_for("AAPL", [earnings], now=now) == 1.0


def test_size_multiplier_ignores_low_impact_kinds():
    """A regular news catalyst doesn't trigger pre-event sizing."""
    now = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
    news = _cat("AAPL", kind="news", when=datetime(2026, 5, 10, 14, 0, tzinfo=UTC))
    pol = CatalystRiskPolicy()
    assert pol.size_multiplier_for("AAPL", [news], now=now) == 1.0


def test_size_multiplier_only_applies_to_matching_symbol():
    now = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
    earnings = _cat("MSFT", kind="earnings", when=datetime(2026, 5, 10, 14, 0, tzinfo=UTC))
    pol = CatalystRiskPolicy()
    assert pol.size_multiplier_for("AAPL", [earnings], now=now) == 1.0


# ── next_catalyst_window ────────────────────────────────────


def test_next_catalyst_returns_soonest_upcoming():
    now = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
    later = _cat("AAPL", when=now + timedelta(hours=10))
    sooner = _cat("AAPL", when=now + timedelta(hours=2))
    out = next_catalyst_window([later, sooner], now=now)
    assert out is sooner


def test_next_catalyst_returns_none_when_none_upcoming():
    now = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
    past = _cat("AAPL", when=now - timedelta(hours=2))
    assert next_catalyst_window([past], now=now) is None


def test_next_catalyst_respects_look_ahead_window():
    """A catalyst beyond the look-ahead is excluded."""
    now = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
    far_off = _cat("AAPL", when=now + timedelta(hours=48))
    out = next_catalyst_window([far_off], now=now, look_ahead_hours=24)
    assert out is None


def test_next_catalyst_filters_by_symbol():
    now = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
    msft = _cat("MSFT", when=now + timedelta(hours=2))
    out = next_catalyst_window([msft], symbol="AAPL", now=now)
    assert out is None
