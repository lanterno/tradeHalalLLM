"""Position-sizing primitives — pure, offline-validatable.

The thesis: for a long-only halal book, *sizing* (not signal count) is where the
edge compounds. These are the building blocks — fractional Kelly and a CPPI-style
drawdown throttle — kept pure and tested here so they can be validated in the
backtest before anything is wired to live orders.

Long-only by construction: a negative edge sizes to ZERO, never a negative
(short) bet. Both functions clamp into safe ranges and (Kelly) gate on sample
size so a few lucky trades can't justify a big bet.
"""

from __future__ import annotations

from halal_trader.core.sample_guard import DEFAULT_MIN_SAMPLES, gate_stat


def half_kelly_fraction(
    win_rate: float,
    payoff_ratio: float,
    *,
    n: int,
    min_n: int = DEFAULT_MIN_SAMPLES,
    cap: float = 0.5,
) -> float:
    """Half-Kelly bet fraction, hard-clamped to ``[0, cap]``.

    Kelly f* = p - (1-p)/b, where p = ``win_rate`` and b = ``payoff_ratio``
    (avg win / avg loss). Half-Kelly (f*/2) is the practical standard — full
    Kelly is notoriously over-aggressive and assumes perfectly known edge.

    - A negative edge (or non-positive payoff / out-of-range win rate) → 0.0:
      never a short, never a negative bet.
    - Gated by the sample guard: below ``min_n`` observations → 0.0. A win-rate
      off a handful of trades is noise; don't bet on it.
    """
    if payoff_ratio <= 0.0 or not 0.0 <= win_rate <= 1.0:
        return 0.0
    kelly = win_rate - (1.0 - win_rate) / payoff_ratio
    half = min(max(0.0, kelly / 2.0), cap)
    return gate_stat(half, n, min_n=min_n, fallback=0.0)


def drawdown_throttle(
    drawdown: float,
    *,
    max_drawdown_budget: float,
    floor: float = 0.0,
) -> float:
    """CPPI-style exposure multiplier in ``[floor, 1.0]``.

    ``drawdown`` is a positive fraction below the peak (0.10 = 10% down).
    Returns 1.0 at the peak and scales linearly down to ``floor`` as the
    drawdown reaches ``max_drawdown_budget`` (and stays at ``floor`` beyond) —
    so exposure shrinks as the cushion to the budget erodes, the CPPI idea.
    """
    if max_drawdown_budget <= 0.0 or drawdown <= 0.0:
        return 1.0
    if drawdown >= max_drawdown_budget:
        return floor
    return max(floor, 1.0 - drawdown / max_drawdown_budget)
