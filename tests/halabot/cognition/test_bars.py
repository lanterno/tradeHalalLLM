"""Bar buffer + pure indicators."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from halabot.cognition.bars import (
    Bar,
    BarBuffer,
    atr,
    ema,
    momentum_signal,
    rsi,
    swing_points,
)

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


def _bar(c, *, o=None, h=None, low=None, i=0):
    return Bar(
        o=o or c, h=h or c + 1, low=low or c - 1, c=c, v=1000.0, ts=T0 + timedelta(minutes=i)
    )


# ── BarBuffer ──
def test_buffer_appends_and_reads_per_asset():
    buf = BarBuffer(maxlen=10)
    buf.append("NVDA", _bar(100))
    buf.append("NVDA", _bar(101))
    buf.append("MSFT", _bar(400))
    assert buf.closes("NVDA") == [100, 101]
    assert buf.closes("MSFT") == [400]


def test_buffer_evicts_oldest_past_maxlen():
    buf = BarBuffer(maxlen=3)
    for i in range(5):
        buf.append("NVDA", _bar(100 + i, i=i))
    assert buf.closes("NVDA") == [102, 103, 104]


# ── indicators ──
def test_ema_none_when_insufficient():
    assert ema([1, 2], 5) is None


def test_ema_trends_with_data():
    assert ema([10] * 10, 5) == 10.0  # flat series → EMA equals the level


def test_rsi_all_gains_is_100():
    assert rsi(list(range(1, 20)), period=14) == 100.0


def test_rsi_none_when_insufficient():
    assert rsi([1, 2, 3], period=14) is None


def test_atr_positive_on_ranging_data():
    highs = [10 + (i % 2) for i in range(20)]
    lows = [9 - (i % 2) for i in range(20)]
    closes = [9.5] * 20
    out = atr(highs, lows, closes, period=14)
    assert out is not None and out > 0


def test_swing_points_finds_local_extrema():
    highs = [1, 3, 1, 1, 5, 1, 1]
    lows = [1, 1, 0, 1, 1, 1, 1]
    sh, sl = swing_points(highs, lows, lookback=2)
    assert 5 in sh  # the clear local max
    assert 0 in sl  # the clear local min


def test_momentum_signal_positive_on_uptrend():
    closes = [100 + i for i in range(30)]  # steady rise
    direction, weight = momentum_signal(closes)
    assert direction > 0
    assert 0 < weight <= 1.0


def test_momentum_signal_negative_on_downtrend():
    closes = [130 - i for i in range(30)]
    direction, _ = momentum_signal(closes)
    assert direction < 0


def test_momentum_signal_zero_when_insufficient():
    assert momentum_signal([100, 101, 102]) == (0.0, 0.0)
