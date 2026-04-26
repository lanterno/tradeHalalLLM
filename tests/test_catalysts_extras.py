"""Tests for catalyst extras: static source, earnings, pre-event sizing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.trading.catalysts import (
    Catalyst,
    CatalystRiskPolicy,
    EarningsCalendarSource,
    StaticCatalystSource,
    StockCatalystFeed,
    next_catalyst_window,
)


def _t(hours_from_now: float) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=hours_from_now)


# ── Static source ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_static_source_returns_all_when_no_filter() -> None:
    src = StaticCatalystSource(
        [
            Catalyst(symbol="AAPL", kind="news", title="x", timestamp=_t(-1)),
            Catalyst(symbol="MSFT", kind="earnings", title="y", timestamp=_t(2)),
        ]
    )
    out = await src.fetch([])
    assert len(out) == 2


@pytest.mark.asyncio
async def test_static_source_filters_by_symbol() -> None:
    src = StaticCatalystSource(
        [
            Catalyst(symbol="AAPL", kind="news", title="x", timestamp=_t(-1)),
            Catalyst(symbol="MSFT", kind="earnings", title="y", timestamp=_t(2)),
        ]
    )
    out = await src.fetch(["AAPL"])
    assert len(out) == 1
    assert out[0].symbol == "AAPL"


# ── Earnings calendar source ─────────────────────────────────────


class _CalendarStub:
    def __init__(self, rows):
        self._rows = rows

    async def get_calendar(self, symbols):
        return self._rows


@pytest.mark.asyncio
async def test_earnings_source_maps_calendar_rows() -> None:
    when = _t(48).isoformat()
    stub = _CalendarStub(
        [
            {"symbol": "AAPL", "date": when, "eps_estimate": 1.45},
            {"symbol": "MSFT", "date": when, "eps_estimate": 2.10},
        ]
    )
    src = EarningsCalendarSource(stub)
    out = await src.fetch(["AAPL", "MSFT"])
    assert {c.symbol for c in out} == {"AAPL", "MSFT"}
    assert all(c.kind == "earnings" for c in out)


@pytest.mark.asyncio
async def test_earnings_source_filters_to_requested_symbols() -> None:
    when = _t(2).isoformat()
    stub = _CalendarStub(
        [
            {"symbol": "AAPL", "date": when},
            {"symbol": "GOOG", "date": when},
        ]
    )
    src = EarningsCalendarSource(stub)
    out = await src.fetch(["AAPL"])
    assert len(out) == 1
    assert out[0].symbol == "AAPL"


@pytest.mark.asyncio
async def test_earnings_source_no_calendar_method_returns_empty() -> None:
    class _NoCal:
        pass

    src = EarningsCalendarSource(_NoCal())
    assert await src.fetch(["AAPL"]) == []


@pytest.mark.asyncio
async def test_earnings_source_handles_unparseable_date() -> None:
    stub = _CalendarStub([{"symbol": "AAPL", "date": "not-a-date"}])
    src = EarningsCalendarSource(stub)
    out = await src.fetch(["AAPL"])
    assert out == []


@pytest.mark.asyncio
async def test_earnings_source_failure_is_swallowed() -> None:
    class _Boom:
        async def get_calendar(self, **_kwargs):
            raise RuntimeError("api down")

    src = EarningsCalendarSource(_Boom())
    assert await src.fetch(["AAPL"]) == []


# ── Pre-event risk policy ────────────────────────────────────────


def test_risk_policy_full_size_outside_window() -> None:
    cat = Catalyst(symbol="AAPL", kind="earnings", title="x", timestamp=_t(48))
    pol = CatalystRiskPolicy(pre_event_hours=4.0, pre_event_size_multiplier=0.5)
    m = pol.size_multiplier_for("AAPL", [cat])
    assert m == 1.0


def test_risk_policy_shrinks_inside_window() -> None:
    cat = Catalyst(symbol="AAPL", kind="earnings", title="x", timestamp=_t(2))
    pol = CatalystRiskPolicy(pre_event_hours=4.0, pre_event_size_multiplier=0.4)
    m = pol.size_multiplier_for("AAPL", [cat])
    assert m == 0.4


def test_risk_policy_ignores_other_symbols() -> None:
    cat = Catalyst(symbol="MSFT", kind="earnings", title="x", timestamp=_t(2))
    pol = CatalystRiskPolicy()
    m = pol.size_multiplier_for("AAPL", [cat])
    assert m == 1.0


def test_risk_policy_ignores_low_impact_kinds() -> None:
    cat = Catalyst(symbol="AAPL", kind="news", title="x", timestamp=_t(2))
    pol = CatalystRiskPolicy(high_impact_kinds=("earnings",))
    m = pol.size_multiplier_for("AAPL", [cat])
    assert m == 1.0


def test_risk_policy_picks_min_when_multiple_overlap() -> None:
    cats = [
        Catalyst(symbol="AAPL", kind="earnings", title="a", timestamp=_t(3)),
        Catalyst(symbol="AAPL", kind="fomc", title="b", timestamp=_t(2)),
    ]
    pol = CatalystRiskPolicy(pre_event_size_multiplier=0.3)
    m = pol.size_multiplier_for("AAPL", cats)
    assert m == 0.3


# ── Window helpers ───────────────────────────────────────────────


def test_next_catalyst_window_picks_soonest() -> None:
    cats = [
        Catalyst(symbol="AAPL", kind="news", title="far", timestamp=_t(20)),
        Catalyst(symbol="AAPL", kind="news", title="near", timestamp=_t(5)),
        Catalyst(symbol="AAPL", kind="news", title="past", timestamp=_t(-1)),
    ]
    n = next_catalyst_window(cats)
    assert n is not None
    assert n.title == "near"


def test_next_catalyst_window_filters_symbol() -> None:
    cats = [
        Catalyst(symbol="MSFT", kind="news", title="msft", timestamp=_t(2)),
        Catalyst(symbol="AAPL", kind="news", title="aapl", timestamp=_t(10)),
    ]
    n = next_catalyst_window(cats, symbol="AAPL")
    assert n is not None
    assert n.symbol == "AAPL"


def test_next_catalyst_window_respects_horizon() -> None:
    cats = [
        Catalyst(symbol="AAPL", kind="news", title="far", timestamp=_t(48)),
    ]
    n = next_catalyst_window(cats, look_ahead_hours=24)
    assert n is None


# ── Feed integration ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_feed_combines_static_plus_earnings() -> None:
    static = StaticCatalystSource(
        [Catalyst(symbol="AAPL", kind="news", title="news", timestamp=_t(-1))]
    )
    cal = EarningsCalendarSource(
        _CalendarStub([{"symbol": "AAPL", "date": _t(48).isoformat()}])
    )
    feed = StockCatalystFeed(sources=[static, cal])
    out = await feed.fetch_all(["AAPL"])
    kinds = {c.kind for c in out}
    assert "news" in kinds
    assert "earnings" in kinds
