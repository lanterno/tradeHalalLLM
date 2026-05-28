"""AlpacaBarSource — maps the real Alpaca MCP bar shape to observation.bar."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halabot.perception.sources.alpaca_bars import AlpacaBarSource, _extract_bars
from halabot.platform.clock import FakeClock
from halabot.platform.events import Event, EventType

CLOCK = FakeClock(datetime(2026, 5, 28, 12, 0, tzinfo=UTC))


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
