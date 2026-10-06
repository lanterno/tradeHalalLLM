"""AlpacaBarSource — maps the real Alpaca MCP bar shape to observation.bar."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halabot.perception.sources.alpaca_bars import AlpacaBarSource, _extract_bars
from halabot.perception.watermark import InMemoryBarWatermark, PgBarWatermark
from halabot.platform.clock import FakeClock
from halabot.platform.event_log import PgEventLog
from halabot.platform.events import Event, EventType, new_event

CLOCK = FakeClock(datetime(2026, 5, 28, 12, 0, tzinfo=UTC))
LATE = FakeClock(datetime(2026, 5, 28, 20, 0, tzinfo=UTC))  # after the hourly test bars closed


def _rows(*closes):
    return [
        {"t": f"t{i}", "o": c, "h": c + 1, "l": c - 1, "c": c, "v": 100}
        for i, c in enumerate(closes)
    ]


class _FakeMCP:
    """Mirrors the real Alpaca shape: {"bars": {SYMBOL: [...]}}."""

    def __init__(self, closes_by_symbol: dict[str, tuple]):
        self._closes = dict(closes_by_symbol)
        self.fail_for: set[str] = set()

    def set_closes(self, symbol: str, closes: tuple) -> None:
        self._closes[symbol] = closes

    async def get_stock_bars(self, symbol, days=5, timeframe="1Hour"):
        if symbol in self.fail_for:
            raise RuntimeError("mcp down")
        return {"bars": {symbol: _rows(*self._closes.get(symbol, ()))}, "next_page_token": None}


async def _universe(symbols):
    async def u() -> list[str]:
        return symbols

    return u


async def _emit_to(sink: list[Event]):
    async def emit(e: Event) -> None:
        sink.append(e)

    return emit


# ── _extract_bars: the real per-symbol shape + envelope tolerance ──
def test_extract_bars_real_per_symbol_shape():
    resp = {"bars": {"NVDA": [{"c": 1}, {"c": 2}], "MSFT": [{"c": 9}]}}
    assert len(_extract_bars(resp, "NVDA")) == 2
    assert len(_extract_bars(resp, "MSFT")) == 1
    assert _extract_bars(resp, "TSLA") == []  # symbol absent


def test_extract_bars_tolerates_envelopes_and_garbage():
    assert len(_extract_bars({"result": {"bars": {"NVDA": [{"c": 1}]}}}, "NVDA")) == 1
    assert len(_extract_bars({"bars": [{"c": 1}]}, "NVDA")) == 1  # flat-list fallback
    assert len(_extract_bars([{"c": 1}], "NVDA")) == 1  # bare list
    assert _extract_bars("garbage", "NVDA") == []


@pytest.mark.asyncio
async def test_emits_bar_observations_per_symbol():
    mcp = _FakeMCP({"NVDA": (100, 101, 102), "MSFT": (400,)})
    src = AlpacaBarSource(mcp, await _universe(["NVDA", "MSFT"]), CLOCK, interval_s=0)
    sink: list[Event] = []
    n = await src.poll_once(await _emit_to(sink))
    assert n == 4
    assert all(e.type == EventType.OBSERVATION_BAR for e in sink)
    nvda = [e for e in sink if e.asset == "NVDA"]
    assert [e.payload["c"] for e in nvda] == [100, 101, 102]  # chronological


@pytest.mark.asyncio
async def test_dedups_seen_bars_across_polls():
    mcp = _FakeMCP({"NVDA": (100, 101)})
    src = AlpacaBarSource(mcp, await _universe(["NVDA"]), CLOCK, interval_s=0)
    sink: list[Event] = []
    emit = await _emit_to(sink)
    await src.poll_once(emit)
    mcp.set_closes("NVDA", (100, 101, 102))  # one new bar (t2)
    await src.poll_once(emit)
    assert [e.payload["c"] for e in sink] == [100, 101, 102]  # 100/101 not re-emitted


@pytest.mark.asyncio
async def test_one_symbol_failure_does_not_block_others():
    mcp = _FakeMCP({"NVDA": (100,), "MSFT": (400,)})
    mcp.fail_for = {"NVDA"}
    src = AlpacaBarSource(mcp, await _universe(["NVDA", "MSFT"]), CLOCK, interval_s=0)
    sink: list[Event] = []
    n = await src.poll_once(await _emit_to(sink))
    assert n == 1 and sink[0].asset == "MSFT"


@pytest.mark.asyncio
async def test_drops_nonpositive_close():
    class _ZeroMCP(_FakeMCP):
        async def get_stock_bars(self, symbol, days=5, timeframe="1Hour"):
            return {"bars": {symbol: [{"t": "t0", "c": 0}, {"t": "t1", "c": 50}]}}

    mcp = _ZeroMCP({})
    src = AlpacaBarSource(mcp, await _universe(["NVDA"]), CLOCK, interval_s=0)
    sink: list[Event] = []
    n = await src.poll_once(await _emit_to(sink))
    assert n == 1 and sink[0].payload["c"] == 50


class _HourlyMCP:
    """Hourly bars with real timestamps, in whatever order it is given."""

    def __init__(self, hours_closes: list[tuple[int, float]]):
        self.rows = [
            {"t": f"2026-05-28T{h:02d}:00:00Z", "o": c, "h": c, "l": c, "c": c, "v": 1}
            for h, c in hours_closes
        ]

    async def get_stock_bars(self, symbol, days=5, timeframe="1Hour"):
        return {"bars": {symbol: list(self.rows)}}


@pytest.mark.asyncio
async def test_resumes_after_the_logged_watermark():
    # A restart used to publish the whole fetch window again as live bars.
    mcp = _HourlyMCP([(13, 1.0), (14, 2.0), (15, 3.0)])
    marks = InMemoryBarWatermark({"NVDA": datetime(2026, 5, 28, 14, 0, tzinfo=UTC)})
    src = AlpacaBarSource(mcp, await _universe(["NVDA"]), LATE, interval_s=0, watermark=marks)
    sink: list[Event] = []
    emit = await _emit_to(sink)
    await src.poll_once(emit)
    assert [e.payload["c"] for e in sink] == [3.0]  # only the bar after the mark
    await src.poll_once(emit)
    assert len(sink) == 1  # and the mark moved with it


@pytest.mark.asyncio
async def test_a_forming_bar_waits_until_it_is_final():
    # Alpaca returns the forming bar too; published early, its partial close,
    # high, low and volume stood for the whole bar forever after.
    clock = FakeClock(datetime(2026, 5, 28, 15, 10, tzinfo=UTC))
    mcp = _HourlyMCP([(13, 1.0), (14, 2.0), (15, 3.0)])
    src = AlpacaBarSource(mcp, await _universe(["NVDA"]), clock, interval_s=0)
    sink: list[Event] = []
    emit = await _emit_to(sink)
    await src.poll_once(emit)
    # 14:00 closed at 15:00 but the delayed feed may still be filling it in.
    assert [e.payload["c"] for e in sink] == [1.0]
    clock.set(datetime(2026, 5, 28, 15, 16, tzinfo=UTC) + timedelta(hours=1))
    mcp.rows[2]["c"] = 3.5  # the 15:00 bar's final close
    await src.poll_once(emit)
    assert [e.payload["c"] for e in sink] == [1.0, 2.0, 3.5]
    assert sink[-1].payload["interval_s"] == 3600.0


@pytest.mark.asyncio
async def test_emits_bars_oldest_first():
    mcp = _HourlyMCP([(15, 3.0), (13, 1.0), (14, 2.0)])
    src = AlpacaBarSource(mcp, await _universe(["NVDA"]), LATE, interval_s=0)
    sink: list[Event] = []
    await src.poll_once(await _emit_to(sink))
    assert [e.payload["c"] for e in sink] == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_pg_watermark_reads_the_newest_logged_bar_per_asset(halabot_engine):
    log = PgEventLog(halabot_engine)
    now = datetime.now(UTC)
    rows = [
        ("NVDA", "2026-05-28T13:00:00Z", now),
        ("NVDA", "2026-05-28T15:00:00Z", now),
        ("NVDA", "2026-05-28T14:00:00Z", now),  # a later replay of an older bar
        ("MSFT", "not-a-time", now),  # one malformed payload costs only itself
        ("MSFT", "2026-05-28T11:00:00Z", now),
        ("AMD", "2026-05-28T16:00:00Z", now - timedelta(days=30)),  # outside the lookback
    ]
    for asset, bar_ts, ts in rows:
        await log.append(
            new_event(
                FakeClock(ts),
                EventType.OBSERVATION_BAR,
                source="alpaca-bars",
                asset=asset,
                payload={"o": 1, "h": 1, "low": 1, "c": 1, "v": 1, "bar_ts": bar_ts},
            )
        )
    marks = await PgBarWatermark(halabot_engine, lookback_days=7).load()
    assert marks == {
        "NVDA": datetime(2026, 5, 28, 15, 0, tzinfo=UTC),
        "MSFT": datetime(2026, 5, 28, 11, 0, tzinfo=UTC),
    }
