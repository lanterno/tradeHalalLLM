"""Deterministic price-level families ("where might it stall?").

Phase 1 of ``docs/QUANT_PREDICTION_ROADMAP.md``: algorithmic support/
resistance levels computable from daily OHLCV — no chart-reading, no
parameters a discretionary user could cherry-pick. Three families, chosen
by evidence-per-complexity:

* **Prior extremes** (`prior_extreme_levels`) — prior day / prior completed
  ISO-week / prior completed month high & low. Zero parameters; the one
  family with a concrete documented mechanism (stop and stop-entry orders
  cluster just beyond prior extremes — Osler's order-book work).
* **Round numbers** (`round_number_levels`) — magnitude-scaled $5/$10/$50/
  $100 grid points near the price. Documented as order *clustering*, weak
  as a barrier: a modifier/snap input, never a standalone level.
* **Swing zones** (`swing_zones`) — confirmed rolling-window swing extrema
  clustered into zones by proximity (ATR-scaled), ranked by touch count.
  Swing confirmation needs ``confirm`` bars of future data, so a swing only
  becomes a level ``confirm`` bars after the extreme — the walk-forward
  harness relies on this to avoid the classic lookahead bug.

Honesty contract: **no family in this module has earned a prompt slot**.
Levels ship to the LLM only after their walk-forward touch-and-hold rate
beats a distance-matched placebo (``quant/level_eval.py``) on out-of-sample
windows — the roadmap expects most families to FAIL this test (per-asset
structural signals were halabot's NO-GO class); the survivors are the
product. Pure numpy throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from halal_trader.quant.volatility import FloatArray

# Round-number grids, coarsest first; a grid applies when it is not
# ridiculously fine for the price (price / grid <= _MAX_GRID_STEPS).
_ROUND_GRIDS = (100.0, 50.0, 25.0, 10.0, 5.0, 1.0)
_MAX_GRID_STEPS = 60
_REACH_ATR = 3.0  # levels farther than this are context, not test targets


@dataclass(frozen=True, slots=True)
class Level:
    """One deterministic price level.

    ``kind`` names the family + member (e.g. ``prior_week_high``,
    ``round_50``, ``swing_zone``); ``strength`` is the touch count for
    swing zones and ``None`` for parameterless families.
    """

    price: float
    kind: str
    strength: float | None = None


def atr_series(
    highs: FloatArray, lows: FloatArray, closes: FloatArray, window: int = 14
) -> npt.NDArray[np.float64]:
    """Wilder ATR as a series (EWMA of true range, alpha=1/window).

    True range uses the previous close (gaps count); slot 0 seeds with the
    plain high-low range. Same daily price units as the inputs.
    """
    h = np.asarray(highs, dtype=np.float64)
    lo = np.asarray(lows, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    if not (h.size == lo.size == c.size) or h.size == 0:
        raise ValueError("highs/lows/closes must be non-empty and equal length")
    tr = h - lo
    if h.size > 1:
        prev_c = c[:-1]
        tr = np.concatenate(
            [
                tr[:1],
                np.maximum.reduce(
                    [h[1:] - lo[1:], np.abs(h[1:] - prev_c), np.abs(lo[1:] - prev_c)]
                ),
            ]
        )
    out = np.empty_like(tr)
    alpha = 1.0 / window
    acc = tr[0]
    out[0] = acc
    for i in range(1, tr.size):
        acc = (1.0 - alpha) * acc + alpha * tr[i]
        out[i] = acc
    return out


def prior_extreme_levels(
    dates: list[str],
    highs: FloatArray,
    lows: FloatArray,
) -> list[Level]:
    """Prior day / prior completed week / prior completed month high-low.

    ``dates`` are ascending ``YYYY-MM-DD`` session dates aligned with the
    bars. Prior day = the last completed session; prior week/month = the
    last ISO week / calendar month **strictly before** the one containing
    the last bar (i.e. fully completed periods only).
    """
    h = np.asarray(highs, dtype=np.float64)
    lo = np.asarray(lows, dtype=np.float64)
    if len(dates) != h.size or h.size != lo.size:
        raise ValueError("dates/highs/lows must be equal length")
    if h.size == 0:
        return []
    out = [
        Level(float(h[-1]), "prior_day_high"),
        Level(float(lo[-1]), "prior_day_low"),
    ]
    from datetime import date as _date

    def _iso_week(d: str) -> tuple[int, int]:
        y, m, day = int(d[:4]), int(d[5:7]), int(d[8:10])
        cal = _date(y, m, day).isocalendar()
        return (cal[0], cal[1])

    last_week = _iso_week(dates[-1])
    last_month = dates[-1][:7]
    week_keys = [_iso_week(d) for d in dates]
    prior_week_idx = [i for i, wk in enumerate(week_keys) if wk < last_week]
    if prior_week_idx:
        wk = week_keys[prior_week_idx[-1]]
        sel = [i for i in prior_week_idx if week_keys[i] == wk]
        out.append(Level(float(h[sel].max()), "prior_week_high"))
        out.append(Level(float(lo[sel].min()), "prior_week_low"))
    month_keys = [d[:7] for d in dates]
    prior_month_idx = [i for i, mk in enumerate(month_keys) if mk < last_month]
    if prior_month_idx:
        mk = month_keys[prior_month_idx[-1]]
        sel = [i for i in prior_month_idx if month_keys[i] == mk]
        out.append(Level(float(h[sel].max()), "prior_month_high"))
        out.append(Level(float(lo[sel].min()), "prior_month_low"))
    return out


def round_number_levels(price: float) -> list[Level]:
    """Nearest round-number grid points above and below ``price``.

    Magnitude-aware: a grid is emitted only when it is coarse enough to be
    salient for the price (``price / grid <= 60`` — the $1 grid matters for
    a $30 stock, not a $500 one). Coarser grids are more salient; the kind
    encodes the grid (``round_100`` > ``round_10`` …). Modifier/snap input
    only, never a standalone signal.
    """
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    out: list[Level] = []
    seen: set[float] = set()
    for grid in _ROUND_GRIDS:
        if price / grid > _MAX_GRID_STEPS:
            continue
        below = np.floor(price / grid) * grid
        for lvl in (below, below + grid):
            p = round(float(lvl), 2)
            if p > 0 and p not in seen:
                seen.add(p)
                out.append(Level(p, f"round_{int(grid)}"))
    return out


def swing_zones(
    highs: FloatArray,
    lows: FloatArray,
    atr: float,
    *,
    confirm: int = 3,
    cluster_eps_atr: float = 0.5,
    top_k: int = 8,
) -> list[Level]:
    """Confirmed swing extrema clustered into touch-ranked zones.

    A bar is a swing high (low) when its high (low) is the strict maximum
    (minimum) of the ``confirm``-bar neighbourhood on BOTH sides — so the
    last ``confirm`` bars can never contribute a swing (that is the
    anti-lookahead property the harness depends on, not a defect). Swing
    prices within ``cluster_eps_atr·ATR`` merge into one zone whose price
    is the touch-weighted mean and whose ``strength`` is the touch count;
    the ``top_k`` strongest zones are returned.
    """
    h = np.asarray(highs, dtype=np.float64)
    lo = np.asarray(lows, dtype=np.float64)
    if h.size != lo.size:
        raise ValueError("highs/lows must be equal length")
    if atr <= 0:
        raise ValueError(f"atr must be positive, got {atr}")
    n = h.size
    if n < 2 * confirm + 1:
        return []
    touches: list[float] = []
    for i in range(confirm, n - confirm):
        lo_w = slice(i - confirm, i + confirm + 1)
        if h[i] >= h[lo_w].max():
            touches.append(float(h[i]))
        if lo[i] <= lo[lo_w].min():
            touches.append(float(lo[i]))
    if not touches:
        return []
    touches.sort()
    eps = cluster_eps_atr * atr
    clusters: list[list[float]] = [[touches[0]]]
    for p in touches[1:]:
        if p - clusters[-1][-1] <= eps:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    zones = [Level(float(np.mean(c)), "swing_zone", strength=float(len(c))) for c in clusters]
    zones.sort(key=lambda z: (-(z.strength or 0), z.price))
    return zones[:top_k]


def level_map(
    dates: list[str],
    highs: FloatArray,
    lows: FloatArray,
    closes: FloatArray,
    *,
    atr: float,
) -> list[Level]:
    """Compose all families into one price-sorted map, round-snap merged.

    A non-round level within ``0.15·ATR`` of a round-number grid point is
    snapped to it (the round number absorbs it — confluence per the
    research), keeping the stronger ``kind`` in the name.
    """
    c = np.asarray(closes, dtype=np.float64)
    if c.size == 0:
        raise ValueError("closes must be non-empty")
    price = float(c[-1])
    levels = prior_extreme_levels(dates, highs, lows) + swing_zones(highs, lows, atr)
    rounds = round_number_levels(price)
    snap_eps = 0.15 * atr
    snapped: list[Level] = []
    for lvl in levels:
        near = next((r for r in rounds if abs(r.price - lvl.price) <= snap_eps), None)
        if near is not None:
            snapped.append(Level(near.price, f"{lvl.kind}+{near.kind}", strength=lvl.strength))
        else:
            snapped.append(lvl)
    out = snapped + rounds
    out.sort(key=lambda z: z.price)
    return out
