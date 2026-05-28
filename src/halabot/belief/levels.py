"""Level engine (REARCHITECTURE B.4).

Computes support / resistance / stop / invalidation. The invalidation is the
structural level that, if lost, kills the long thesis; it **ratchets up only**
(tightens as price rises, never loosens) — the structural "slow out". A
cold-start asset with no swing structure and no ATR yields a ``None``
invalidation rather than crashing (fix R, all-None ``max``); the policy simply
won't open a position until a stop can be computed.

This is a pure function over already-computed swings + ATR (the bar→swing
computation lives in cognition), so it's trivially testable.
"""

from __future__ import annotations

from halabot.belief.schema import Levels


def update_levels(
    *,
    last_price: float | None,
    swing_lows: list[float],
    swing_highs: list[float],
    atr: float | None,
    prev: Levels,
    atr_stop_mult: float = 2.0,
) -> Levels:
    """Recompute :class:`Levels` from price structure.

    * ``support`` = nearest swing low below price; ``resistance`` = nearest
      swing high above price.
    * ``invalidation`` = the MAX of {most-recent swing low, price − k·ATR,
      previous invalidation} over whichever are available — the ratchet-up-only
      structural stop. Empty (all-None) → ``None`` (no crash, no premature stop).
    * ``stop`` mirrors ``invalidation`` (the monitor's hard stop).
    """
    support = _nearest_below(last_price, swing_lows)
    resistance = _nearest_above(last_price, swing_highs)

    structural = swing_lows[-1] if swing_lows else None
    atr_floor = (
        last_price - atr_stop_mult * atr
        if last_price is not None and atr is not None and atr > 0
        else None
    )
    candidates = [x for x in (structural, atr_floor, prev.invalidation) if x is not None]
    invalidation = max(candidates) if candidates else None  # fix R: no ValueError on empty

    return Levels(
        support=support,
        resistance=resistance,
        stop=invalidation,
        invalidation=invalidation,
    )


def _nearest_below(price: float | None, levels: list[float]) -> float | None:
    if price is None:
        return None
    below = [x for x in levels if x < price]
    return max(below) if below else None


def _nearest_above(price: float | None, levels: list[float]) -> float | None:
    if price is None:
        return None
    above = [x for x in levels if x > price]
    return min(above) if above else None
