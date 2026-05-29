"""Structural regime signal (REARCHITECTURE L2, rank 5).

A market-state label derived purely from PRICE GEOMETRY — the Kaufman
efficiency ratio (trend quality) and a Donchian-channel breakout — and so
INDEPENDENT of the signed evidence vector that drives conviction. This matters
because the existing :class:`EvidenceRegimeClassifier` derives its regime from
the same ``weighted_sum(evidence)`` that triggers an entry, so at entry it is a
constant (``trending_up``) and cannot discriminate good entries from bad
(measured 2026-05-29: all backtest entries were ``trending_up``). A structural
label can differ at entry, so it is a candidate edge signal we can MEASURE
(per-structure P&L segmentation) before letting it touch conviction/sizing.

EMPIRICAL VERDICT (2026-05-29, 5 independent windows — 20/30/45/60/90d, 1H,
5bps, ER>=0.5, 20-bar window): the structural label does NOT robustly
discriminate entry P&L. Donchian-breakout entries LOSE in 4 of 5 windows
(only the n=4 30d bucket was positive — noise), and the "chop" bucket's sign
flips win/lose/win/win/lose across windows. So a breakout filter would NOT
help this engine on 1H bars, and structure is wired as MEASUREMENT/telemetry
ONLY — it does NOT feed conviction, gating, or sizing. Kept because the
per-structure segmentation is useful operator visibility and the functions
are reusable if a future timeframe / market-wide variant proves out.

All pure functions over OHLC lists — no LLM, no network, no [ml] extra (INV-1).
"""

from __future__ import annotations

from typing import Literal

StructuralLabel = Literal["breakout", "trend", "chop", "unknown"]

# Kaufman ER at/above which the window is "efficiently trending" (vs choppy).
_DEFAULT_ER_TREND = 0.5
_DEFAULT_WINDOW = 20


def efficiency_ratio(closes: list[float], window: int = _DEFAULT_WINDOW) -> float | None:
    """Kaufman efficiency ratio over the last ``window`` bars: net directional
    change / total path length, in [0, 1]. ~1 = a clean one-way move; ~0 = the
    price thrashed back and forth (chop). None if too few bars / zero path."""
    if window < 1 or len(closes) < window + 1:
        return None
    seg = closes[-(window + 1) :]
    net = abs(seg[-1] - seg[0])
    path = sum(abs(seg[i] - seg[i - 1]) for i in range(1, len(seg)))
    if path <= 0:
        return None
    return net / path


def donchian_breakout(
    highs: list[float], lows: list[float], closes: list[float], window: int = _DEFAULT_WINDOW
) -> int:
    """+1 if the latest close breaks ABOVE the prior ``window``-bar high
    (Donchian up-breakout), -1 if it breaks BELOW the prior ``window``-bar low,
    else 0. The latest bar is excluded from the channel it must break."""
    if window < 1 or len(highs) < window + 1 or len(lows) < window + 1 or not closes:
        return 0
    prior_high = max(highs[-(window + 1) : -1])
    prior_low = min(lows[-(window + 1) : -1])
    c = closes[-1]
    if c >= prior_high:
        return 1
    if c <= prior_low:
        return -1
    return 0


def sma_trend_state(closes: list[float], window: int = 50) -> Literal["above", "below", "unknown"]:
    """Where the latest close sits relative to its ``window``-bar simple moving
    average: ``above`` (uptrend side) / ``below`` (downtrend side) / ``unknown``
    (too few bars). Used for a market-wide risk-on/off read on the benchmark —
    a single global, non-circular signal independent of any one asset's
    evidence (rank 5, market-regime variant)."""
    if window < 1 or len(closes) < window:
        return "unknown"
    sma = sum(closes[-window:]) / window
    return "above" if closes[-1] >= sma else "below"


def structural_label(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    window: int = _DEFAULT_WINDOW,
    er_trend: float = _DEFAULT_ER_TREND,
) -> StructuralLabel:
    """Classify the current bar's price structure (independent of conviction):

    * ``breakout`` — a fresh Donchian up-breakout on an efficient move (the
      highest-quality momentum entry the strategy can find structurally).
    * ``trend``    — efficiently trending (ER >= ``er_trend``) but not a fresh
      breakout (an established move we're joining late).
    * ``chop``     — inefficient / range-bound (ER < ``er_trend``).
    * ``unknown``  — too few bars to judge.
    """
    er = efficiency_ratio(closes, window)
    if er is None:
        return "unknown"
    breakout = donchian_breakout(highs, lows, closes, window)
    if breakout > 0 and er >= er_trend:
        return "breakout"
    if er >= er_trend:
        return "trend"
    return "chop"
