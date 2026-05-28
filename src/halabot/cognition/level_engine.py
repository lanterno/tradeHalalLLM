"""Bar-driven level engine (REARCHITECTURE L2 → L3 collaborator).

Implements the updater's ``LevelEngine`` protocol by reading the rolling
:class:`BarBuffer` and feeding computed swings + ATR + last price into the pure
``update_levels`` (which carries the ratchet-up-only invalidation and the
all-None cold-start guard).
"""

from __future__ import annotations

from halabot.belief.levels import update_levels
from halabot.belief.schema import Levels
from halabot.cognition.bars import BarBuffer, atr, swing_points


class BarLevelEngine:
    def __init__(
        self,
        buffer: BarBuffer,
        *,
        swing_lookback: int = 2,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
    ) -> None:
        self._buffer = buffer
        self._swing_lookback = swing_lookback
        self._atr_period = atr_period
        self._atr_stop_mult = atr_stop_mult

    async def levels_for(self, asset: str, prev: Levels) -> Levels:
        highs = self._buffer.highs(asset)
        lows = self._buffer.lows(asset)
        closes = self._buffer.closes(asset)
        swing_highs, swing_lows = swing_points(highs, lows, lookback=self._swing_lookback)
        return update_levels(
            last_price=closes[-1] if closes else None,
            swing_lows=swing_lows,
            swing_highs=swing_highs,
            atr=atr(highs, lows, closes, period=self._atr_period),
            prev=prev,
            atr_stop_mult=self._atr_stop_mult,
        )
