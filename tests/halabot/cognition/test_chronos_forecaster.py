"""Chronos forecaster interpreter (B1) — torch-free via an injected fake."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halabot.cognition.bars import Bar, BarBuffer
from halabot.cognition.chronos_forecaster import ChronosForecasterInterpreter
from halabot.platform.clock import FakeClock
from halabot.platform.events import EventType, new_event

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)
CLOCK = FakeClock(T0)


class _FakeForecaster:
    """Returns a fixed (lo, mid, hi) price triple regardless of input."""

    def __init__(self, lo: float, mid: float, hi: float) -> None:
        self._q = (lo, mid, hi)

    def forecast(self, closes: list[float], *, horizon: int) -> tuple[float, float, float]:
        return self._q


def _fill(buf: BarBuffer, asset: str, n: int, last: float = 100.0) -> None:
    for i in range(n):
        c = last - (n - 1 - i) * 0.0  # flat tail; only the last close matters for the ratio
        buf.append(asset, Bar(o=c, h=c + 1, low=c - 1, c=last, v=1.0, ts=T0))


def _obs(asset: str = "NVDA"):
    return new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset=asset)


async def _run(forecaster, *, n=70, last=100.0, window=64):
    buf = BarBuffer()
    _fill(buf, "NVDA", n, last=last)
    itp = ChronosForecasterInterpreter(buf, forecaster, window=window)
    return await itp.interpret(_obs())


@pytest.mark.asyncio
async def test_confident_up_forecast_votes_long_high_weight():
    # Median +2% with a tight band → strong positive direction, high weight.
    out = await _run(_FakeForecaster(lo=101.0, mid=102.0, hi=103.0), last=100.0)
    assert len(out) == 1
    e = out[0]
    assert e.source == "forecaster"
    assert e.direction > 0.5  # +2% * scale(50) saturates strongly positive
    assert e.weight > 0.3  # tight band relative to a 2% move → confident


@pytest.mark.asyncio
async def test_confident_down_forecast_votes_short():
    out = await _run(_FakeForecaster(lo=97.0, mid=98.0, hi=99.0), last=100.0)
    assert len(out) == 1 and out[0].direction < -0.5


@pytest.mark.asyncio
async def test_wide_band_lowers_weight():
    # Same +1% median move, but a very wide 10–90% band → low signal-to-noise.
    tight = await _run(_FakeForecaster(lo=100.5, mid=101.0, hi=101.5), last=100.0)
    wide = await _run(_FakeForecaster(lo=95.0, mid=101.0, hi=107.0), last=100.0)
    assert tight[0].weight > wide[0].weight


@pytest.mark.asyncio
async def test_abstains_on_too_little_history():
    out = await _run(_FakeForecaster(lo=101.0, mid=105.0, hi=110.0), n=10, window=64)
    assert out == []


@pytest.mark.asyncio
async def test_abstains_on_near_zero_move():
    # Median essentially equal to last → direction below the floor → abstain.
    out = await _run(_FakeForecaster(lo=99.99, mid=100.0001, hi=100.01), last=100.0)
    assert out == []


@pytest.mark.asyncio
async def test_abstains_on_degenerate_band():
    out = await _run(_FakeForecaster(lo=102.0, mid=102.0, hi=102.0), last=100.0)
    assert out == []  # hi <= lo → no usable signal


@pytest.mark.asyncio
async def test_max_weight_scales_emitted_weight():
    # Same forecast, higher max_weight → proportionally higher vote weight. (A
    # backtest A/B showed raising it above the 0.6 default HURTS — overweighting
    # the forecaster unbalances the ensemble — so it ships at 0.6; the knob exists
    # for future tuning.)
    buf = BarBuffer()
    _fill(buf, "NVDA", 70, last=100.0)
    fc = _FakeForecaster(lo=101.0, mid=102.0, hi=103.0)
    lo_w = await ChronosForecasterInterpreter(buf, fc, window=64, max_weight=0.6).interpret(_obs())
    hi_w = await ChronosForecasterInterpreter(buf, fc, window=64, max_weight=1.0).interpret(_obs())
    assert hi_w[0].weight > lo_w[0].weight
    assert hi_w[0].weight / lo_w[0].weight == pytest.approx(1.0 / 0.6, rel=1e-6)
