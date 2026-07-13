"""Data-hygiene and expectations-setting diagnostics (Phase 0).

Two small, honest tools:

* **Overnight/intraday decomposition** — most equity drift accrues
  close→open, not open→close (Lou–Polk–Skouras). An intraday-only strategy
  structurally forfeits that drift; this diagnostic quantifies the split on
  our own universe so the expectation is set by measurement, not folklore.
  Diagnostic only — never a signal.
* **Split-gap guard** — the Alpaca MCP ``get_stock_bars`` tool exposes no
  ``adjustment`` parameter (verified 2026-07-13 against the live tool
  schema), so bars must be assumed RAW: a stock split appears as a huge
  overnight gap and silently corrupts vol estimates, levels, labels and
  calibrations. ``suspect_split_gaps`` flags overnight moves beyond a
  threshold chosen so that essentially only corporate actions trigger it
  (default 40 %: a 2:1 split gaps −50 %, while true one-day moves that
  size are ~nonexistent for AAOIFI-20 mega caps). Deliberately NOT lower:
  skipping genuine crash days would bias the track record upward —
  excluding the worst outcomes is a worse lie than including a rare
  artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from halal_trader.quant.volatility import FloatArray

SPLIT_GAP_THRESHOLD = 0.40


@dataclass(frozen=True, slots=True)
class OvernightSplit:
    """Decomposition of daily drift into overnight and intraday legs."""

    n_days: int
    mean_overnight_pct: float  # avg close→open return, %
    mean_intraday_pct: float  # avg open→close return, %
    cum_overnight_pct: float  # compounded overnight-only leg, %
    cum_intraday_pct: float  # compounded intraday-only leg, %


def overnight_intraday_split(opens: FloatArray, closes: FloatArray) -> OvernightSplit:
    """Split daily returns into overnight (close→open) and intraday legs.

    Requires ≥ 2 bars of strictly positive opens/closes; raises
    ``ValueError`` otherwise.
    """
    o = np.asarray(opens, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    if o.size != c.size:
        raise ValueError(f"length mismatch: opens={o.size}, closes={c.size}")
    if o.size < 2:
        raise ValueError("need at least 2 bars")
    if (o <= 0).any() or (c <= 0).any():
        raise ValueError("prices must be strictly positive")
    overnight = o[1:] / c[:-1] - 1.0
    intraday = c[1:] / o[1:] - 1.0
    return OvernightSplit(
        n_days=int(overnight.size),
        mean_overnight_pct=round(float(overnight.mean()) * 100, 4),
        mean_intraday_pct=round(float(intraday.mean()) * 100, 4),
        cum_overnight_pct=round(float(np.prod(1.0 + overnight) - 1.0) * 100, 2),
        cum_intraday_pct=round(float(np.prod(1.0 + intraday) - 1.0) * 100, 2),
    )


def suspect_split_gaps(
    opens: FloatArray,
    closes: FloatArray,
    threshold: float = SPLIT_GAP_THRESHOLD,
) -> list[tuple[int, float]]:
    """Indices (and gap fractions) of overnight moves beyond ``threshold``.

    On raw (unadjusted) bars these are almost certainly splits or other
    corporate actions; consumers should refuse to label/calibrate across
    them rather than ingest a fake ±50 % move. Index ``i`` means the gap
    is between bar ``i-1``'s close and bar ``i``'s open.
    """
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    o = np.asarray(opens, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    if o.size != c.size:
        raise ValueError(f"length mismatch: opens={o.size}, closes={c.size}")
    if o.size < 2:
        return []
    gaps = o[1:] / c[:-1] - 1.0
    return [(int(i) + 1, round(float(g), 4)) for i, g in enumerate(gaps) if abs(g) > threshold]
