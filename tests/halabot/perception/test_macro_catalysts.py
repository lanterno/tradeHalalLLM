"""MacroCatalystSource — maps FRED-style catalysts to observation.macro."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from halabot.perception.sources.macro_catalysts import MacroCatalystSource
from halabot.platform.clock import FakeClock
from halabot.platform.events import Event, EventType

CLOCK = FakeClock(datetime(2026, 7, 3, 12, 0, tzinfo=UTC))


@dataclass(frozen=True)
class _LegacyCatalyst:
    """Shape-compatible stand-in for trading.catalysts.Catalyst."""

    symbol: str
    kind: str
    title: str
    timestamp: datetime
    extra: dict[str, Any] = field(default_factory=dict)


class _FakeFetcher:
    def __init__(self, items: list[_LegacyCatalyst]) -> None:
        self.items = items
        self.calls: list[list[str]] = []

    async def fetch(self, symbols) -> list[_LegacyCatalyst]:
        self.calls.append(list(symbols))
        return self.items


def _src(items: list[_LegacyCatalyst], symbols: list[str]) -> MacroCatalystSource:
    async def universe() -> list[str]:
        return symbols

    return MacroCatalystSource(_FakeFetcher(items), universe, CLOCK)


def _collect() -> tuple[list[Event], Any]:
    sink: list[Event] = []

    async def emit(e: Event) -> None:
        sink.append(e)

    return sink, emit


IN_3_DAYS = CLOCK.now() + timedelta(days=3)


@pytest.mark.asyncio
async def test_maps_fred_catalyst_to_observation_macro():
    items = [_LegacyCatalyst("AAPL", "cpi", "CPI release", IN_3_DAYS)]
    sink, emit = _collect()
    n = await _src(items, ["AAPL"]).poll_once(emit)
    assert n == 1
    e = sink[0]
    assert e.type == EventType.OBSERVATION_MACRO
    assert e.asset == "AAPL"
    assert e.payload["kind"] == "CPI"
    assert e.payload["scheduled_for"] == IN_3_DAYS.isoformat()
    assert e.payload["expected_impact"] == pytest.approx(0.9)
    assert e.payload["detail"] == "CPI release"


@pytest.mark.asyncio
async def test_impact_table_and_unknown_kind_default():
    items = [
        _LegacyCatalyst("AAPL", "fomc", "FOMC", IN_3_DAYS),
        _LegacyCatalyst("AAPL", "gdp", "GDP", IN_3_DAYS),
        _LegacyCatalyst("AAPL", "retail_sales", "Retail", IN_3_DAYS),
    ]
    sink, emit = _collect()
    await _src(items, ["AAPL"]).poll_once(emit)
    impacts = {e.payload["kind"]: e.payload["expected_impact"] for e in sink}
    assert impacts["FOMC"] == pytest.approx(0.9)
    assert impacts["GDP"] == pytest.approx(0.6)  # below threshold: priced in
    assert impacts["RETAIL_SALES"] == pytest.approx(0.5)  # conservative default


@pytest.mark.asyncio
async def test_stale_releases_dropped():
    """A release >1h past its schedule is calendar noise, not pending."""
    items = [
        _LegacyCatalyst("AAPL", "cpi", "old", CLOCK.now() - timedelta(hours=2)),
        _LegacyCatalyst("AAPL", "nfp", "fresh", IN_3_DAYS),
    ]
    sink, emit = _collect()
    n = await _src(items, ["AAPL"]).poll_once(emit)
    assert n == 1
    assert sink[0].payload["kind"] == "NFP"


@pytest.mark.asyncio
async def test_dedup_across_polls():
    """FRED re-returns the whole calendar every poll — dedup by
    symbol:kind:date so the belief isn't churned each tick."""
    items = [_LegacyCatalyst("AAPL", "cpi", "CPI release", IN_3_DAYS)]
    src = _src(items, ["AAPL"])
    sink, emit = _collect()
    assert await src.poll_once(emit) == 1
    assert await src.poll_once(emit) == 0  # same calendar → nothing new


@pytest.mark.asyncio
async def test_malformed_items_dropped_not_fatal():
    good = _LegacyCatalyst("AAPL", "cpi", "CPI", IN_3_DAYS)
    bad_no_symbol = _LegacyCatalyst("", "cpi", "CPI", IN_3_DAYS)
    bad_no_ts = _LegacyCatalyst("MSFT", "cpi", "CPI", None)  # type: ignore[arg-type]
    sink, emit = _collect()
    n = await _src([bad_no_symbol, bad_no_ts, good], ["AAPL", "MSFT"]).poll_once(emit)
    assert n == 1
    assert sink[0].asset == "AAPL"


@pytest.mark.asyncio
async def test_empty_universe_skips_fetch():
    fetcher = _FakeFetcher([])

    async def universe() -> list[str]:
        return []

    src = MacroCatalystSource(fetcher, universe, CLOCK)
    sink, emit = _collect()
    assert await src.poll_once(emit) == 0
    assert fetcher.calls == []  # never hit the feed with no symbols


@pytest.mark.asyncio
async def test_date_only_schedule_anchored_to_release_clock_time():
    """FRED gives date-only (UTC midnight) timestamps; the source must
    anchor them to the release's real ET clock time or the imminence
    window fires the evening before and misses the print entirely."""
    from zoneinfo import ZoneInfo

    midnight = datetime(2026, 7, 9, 0, 0, tzinfo=UTC)  # date-only signature
    items = [
        _LegacyCatalyst("AAPL", "cpi", "CPI", midnight),
        _LegacyCatalyst("AAPL", "fomc", "FOMC", midnight),
    ]
    sink, emit = _collect()
    await _src(items, ["AAPL"]).poll_once(emit)
    by_kind = {e.payload["kind"]: e.payload["scheduled_for"] for e in sink}
    et = ZoneInfo("America/New_York")
    cpi = datetime.fromisoformat(by_kind["CPI"]).astimezone(et)
    fomc = datetime.fromisoformat(by_kind["FOMC"]).astimezone(et)
    assert (cpi.hour, cpi.minute) == (8, 30)
    assert (fomc.hour, fomc.minute) == (14, 0)
    assert cpi.date().isoformat() == "2026-07-09"  # same calendar day, not shifted


@pytest.mark.asyncio
async def test_real_clock_time_schedules_pass_through():
    """A fetcher that provides a genuine release time keeps it."""
    real = datetime(2026, 7, 9, 12, 30, tzinfo=UTC)
    items = [_LegacyCatalyst("AAPL", "cpi", "CPI", real)]
    sink, emit = _collect()
    await _src(items, ["AAPL"]).poll_once(emit)
    assert sink[0].payload["scheduled_for"] == real.isoformat()
