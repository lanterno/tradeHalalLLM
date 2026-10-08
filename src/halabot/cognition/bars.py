"""Bar buffer + pure indicator math (cheap, deterministic — INV-1).

A rolling per-asset window of OHLCV bars and the indicator functions cognition
interpreters compute over it. All functions return ``None`` on insufficient
data rather than raising, so a cold-start asset degrades cleanly. These are the
LLM-free signals that keep beliefs current when the LLM is unavailable.

(The richer indicator suite in ``halal_trader/signals/indicators.py`` folds in
later; this is a self-contained minimal set to get the loop running.)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Bar:
    o: float
    h: float
    low: float
    c: float
    v: float
    ts: datetime


class BarBuffer:
    """Rolling OHLCV window per asset (bounded; oldest evicted), strictly
    increasing in bar time."""

    def __init__(self, maxlen: int = 200) -> None:
        self._maxlen = maxlen
        self._bars: dict[str, deque[Bar]] = {}

    def append(self, asset: str, bar: Bar) -> bool:
        """Add ``bar`` if it is newer than the asset's last bar; return whether it was.

        A repeat or an older bar is dropped. Every indicator here reads the
        window as one ordered series, and the shadow once fed it the same five
        days again on each restart: the 200-bar window became repeated,
        out-of-order runs of history and RSI, the SMAs and the forecaster all
        computed on that.
        """
        window = self._bars.setdefault(asset, deque(maxlen=self._maxlen))
        if window and bar.ts <= window[-1].ts:
            return False
        window.append(bar)
        return True

    def last_ts(self, asset: str) -> datetime | None:
        """Bar time of the asset's latest bar, or None before its first."""
        window = self._bars.get(asset)
        return window[-1].ts if window else None

    def bars(self, asset: str) -> list[Bar]:
        return list(self._bars.get(asset, ()))

    def closes(self, asset: str) -> list[float]:
        return [b.c for b in self._bars.get(asset, ())]

    def timestamped_closes(self, asset: str) -> list[tuple[datetime, float]]:
        """(bar_ts, close) pairs — lets the risk engine align return series by
        time rather than by position (correlation correctness)."""
        return [(b.ts, b.c) for b in self._bars.get(asset, ())]

    def highs(self, asset: str) -> list[float]:
        return [b.h for b in self._bars.get(asset, ())]

    def lows(self, asset: str) -> list[float]:
        return [b.low for b in self._bars.get(asset, ())]


class BufferPriceSource:
    """Last-close price source over the buffer (the updater's ``PriceSource``).

    A real, free price feed for the invalidation check: the latest bar close.
    Returns None for an asset with no bars yet (no spurious invalidation)."""

    def __init__(self, buffer: BarBuffer) -> None:
        self._buffer = buffer

    def last_price(self, asset: str) -> float | None:
        closes = self._buffer.closes(asset)
        return closes[-1] if closes else None


def returns(closes: list[float]) -> list[float]:
    """Simple per-bar returns from a close series (skips non-positive prior closes)."""
    return [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]


def ema(values: list[float], period: int) -> float | None:
    if len(values) < period or period <= 0:
        return None
    k = 2.0 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def rsi(closes: list[float], period: int = 14) -> float | None:
    """Cutler's RSI in [0, 100] (simple averages of the last ``period`` moves), or
    None if < period+1 closes.

    Not Wilder's smoothed RSI (signals/indicators.py, the day-trader's): the
    engine's interpreters were tuned on this one, so it stays until a backtest
    on out-of-sample windows says otherwise."""
    if len(closes) < period + 1:
        return None
    gains, losses = 0.0, 0.0
    for prev, cur in zip(closes[-period - 1 :], closes[-period:]):
        delta = cur - prev
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain, avg_loss = gains / period, losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> float | None:
    """Average true range: the simple mean of the last ``period`` true ranges (not
    Wilder's smoothing, see :func:`rsi`), or None if insufficient data."""
    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return None
    trs: list[float] = []
    for i in range(n - period, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    return sum(trs) / len(trs) if trs else None


def swing_points(
    highs: list[float], lows: list[float], lookback: int = 2
) -> tuple[list[float], list[float]]:
    """Local extrema: a high is a swing high if it's the max of its
    ±lookback window (mutatis mutandis for lows). Returns (swing_highs,
    swing_lows) in chronological order."""
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    n = min(len(highs), len(lows))
    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback : i + lookback + 1]
        window_l = lows[i - lookback : i + lookback + 1]
        if highs[i] == max(window_h):
            swing_highs.append(highs[i])
        if lows[i] == min(window_l):
            swing_lows.append(lows[i])
    return swing_highs, swing_lows


def momentum_signal(closes: list[float], *, fast: int = 9, slow: int = 21) -> tuple[float, float]:
    """Trend direction + confidence from fast-vs-slow EMA separation.

    Returns ``(direction, weight)`` where direction ∈ [-1, +1] (sign = trend,
    magnitude = relative EMA gap, capped) and weight ∈ [0, 1] (data sufficiency).
    ``(0.0, 0.0)`` when there isn't enough history — a neutral, weightless signal.
    """
    if len(closes) < slow:
        return 0.0, 0.0
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    if fast_ema is None or slow_ema is None or slow_ema == 0:
        return 0.0, 0.0
    gap = (fast_ema - slow_ema) / abs(slow_ema)  # relative separation
    direction = max(-1.0, min(1.0, gap * 20.0))  # scale; ±5% gap saturates
    weight = min(1.0, len(closes) / (2 * slow))  # more history → more confidence
    return direction, weight
