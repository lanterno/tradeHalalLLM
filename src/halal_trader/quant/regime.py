"""VIX term-structure market-regime gate (Phase 3).

The one *market-relative* signal in this roadmap — the class the operator
context says historically DID survive disjoint-OOS ("market-relative
signals pay off; per-asset technicals don't"). It reads the CBOE VIX term
structure (public index data — no trading, halal-trivially fine) and maps
it to a 3-state regime the recommendation prompt can condition on.

Two ratios, per the research:

* ``r_slow = VIX / VIX3M`` — the STABLE regime anchor ("what regime are we
  in"). Backwardation (r_slow > 1: the 1-month fear above the 3-month) is
  the rare, high-signal risk-off state (~15–20 % of days) that historically
  precedes poor short-horizon equity returns.
* ``r_fast = VIX9D / VIX`` — the fast, noisy trigger ("is something
  happening today"). Used only as COLOR, never as the gate: a 9-day
  inversion right before a scheduled event (CPI/FOMC) is *expectation*,
  not stress, so gating on it would false-alarm.

2-day hysteresis (a new state must persist two sessions before it takes)
kills boundary whipsaw. Computed statelessly from the recent ratio series
— no persisted state to drift.

The classifier is pure stdlib; ``fetch_vix_term_structure`` pulls the three
CBOE index CSVs over httpx and degrades to the last-good reading (then
``None``) on any failure — a data outage must never block or delay a cycle.
CBOE is the source because Alpaca doesn't serve indices and Yahoo is
429-blocked here (verified 2026-07-14).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{idx}_History.csv"
_FETCH_TIMEOUT = 8.0
_TAIL_DAYS = 15  # only the recent series is needed for the hysteresis

# r_slow (VIX/VIX3M) thresholds: >=1.0 backwardation (risk-off), the calm
# band below ~0.95 is contango (risk-on), between is caution.
_RISK_OFF = 1.0
_CAUTION = 0.95
_FAST_INVERSION = 1.0  # r_fast >= this = short-end inversion (color only)
_HYSTERESIS_DAYS = 2

RISK_ON = "risk_on"
CAUTION = "caution"
RISK_OFF = "risk_off"


@dataclass(frozen=True, slots=True)
class RegimeReading:
    """One day's VIX term-structure regime."""

    regime: str  # risk_on | caution | risk_off
    r_slow: float  # VIX / VIX3M
    r_fast: float  # VIX9D / VIX
    fast_inverted: bool  # short-end inversion flag (color)
    vix: float


def classify_raw(r_slow: float, r_fast: float) -> str:
    """Per-day regime from the term-structure ratios (no hysteresis).

    Gated on the STABLE ``r_slow`` anchor; ``r_fast`` is not part of the
    gate (it only sets the ``fast_inverted`` color elsewhere).
    """
    if r_slow >= _RISK_OFF:
        return RISK_OFF
    if r_slow >= _CAUTION:
        return CAUTION
    return RISK_ON


def _hysteresis(raw_recent: list[str], current: str) -> str:
    """Apply N-day confirmation: only switch when the last N agree.

    ``raw_recent`` is oldest→newest raw daily states including today. The
    regime moves to a new state only if the most recent ``_HYSTERESIS_DAYS``
    raw states are all that new state; otherwise it holds ``current``.
    """
    if len(raw_recent) < _HYSTERESIS_DAYS:
        return raw_recent[-1] if raw_recent else current
    tail = raw_recent[-_HYSTERESIS_DAYS:]
    if all(s == tail[0] for s in tail) and tail[0] != current:
        return tail[0]
    return current


def regime_from_series(
    vix9d: list[float],
    vix: list[float],
    vix3m: list[float],
) -> RegimeReading | None:
    """Latest regime from aligned ascending VIX / VIX9D / VIX3M close series.

    Applies the 2-day hysteresis over the recent raw states starting from
    the oldest available raw state as the seed. Returns ``None`` when the
    series are empty, misaligned, or carry non-positive values.
    """
    n = min(len(vix9d), len(vix), len(vix3m))
    if n == 0:
        return None
    v9, v, v3 = vix9d[-n:], vix[-n:], vix3m[-n:]
    if any(x <= 0 for x in (*v9[-1:], *v[-1:], *v3[-1:])):
        return None
    raw: list[str] = []
    for a, b, c in zip(v9, v, v3, strict=True):
        if a <= 0 or b <= 0 or c <= 0:
            continue
        raw.append(classify_raw(b / c, a / b))
    if not raw:
        return None
    regime = raw[0]
    for i in range(1, len(raw)):
        regime = _hysteresis(raw[: i + 1], regime)
    r_slow = v[-1] / v3[-1]
    r_fast = v9[-1] / v[-1]
    return RegimeReading(
        regime=regime,
        r_slow=round(r_slow, 4),
        r_fast=round(r_fast, 4),
        fast_inverted=r_fast >= _FAST_INVERSION,
        vix=round(v[-1], 2),
    )


def format_for_prompt(reading: RegimeReading) -> str:
    """One-line market-regime block for the recommendation prompt."""
    label = {
        RISK_ON: "RISK-ON (VIX term structure in contango — normal tape)",
        CAUTION: "CAUTION (VIX curve flattening — near backwardation)",
        RISK_OFF: (
            "RISK-OFF (VIX BACKWARDATION — 1-month fear above 3-month; "
            "poor short-horizon equity odds, size conviction down)"
        ),
    }[reading.regime]
    color = " · short-end inverted (near-term event/stress)" if reading.fast_inverted else ""
    return (
        f"MARKET REGIME: {label}. VIX {reading.vix:.1f}, "
        f"VIX/VIX3M {reading.r_slow:.2f}, VIX9D/VIX {reading.r_fast:.2f}{color}."
    )


def _parse_cboe_csv(text: str) -> dict[str, float]:
    """Parse a CBOE ``DATE,OPEN,HIGH,LOW,CLOSE`` CSV → {ISO date: close}.

    DATE is ``MM/DD/YYYY``; only the CLOSE column is kept. Malformed rows
    are skipped so one bad line can't sink the whole series.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 5 or not parts[0][:2].isdigit():
            continue
        try:
            mm, dd, yyyy = parts[0].split("/")
            iso = f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
            out[iso] = float(parts[4])
        except ValueError, IndexError:
            continue
    return out


_last_good: RegimeReading | None = None


async def fetch_vix_term_structure() -> RegimeReading | None:
    """Fetch VIX9D/VIX/VIX3M from CBOE and return the latest regime reading.

    Fully defensive: on ANY failure (network, parse, misalignment) returns
    the last successful reading if one exists this process, else ``None`` —
    the caller degrades to "regime unavailable", never blocks. Aligns the
    three series on their common trading dates (the tail suffices for
    hysteresis) before classifying.
    """
    global _last_good
    import httpx

    try:
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
            series: dict[str, dict[str, float]] = {}
            for idx in ("VIX9D", "VIX", "VIX3M"):
                resp = await client.get(_CBOE_URL.format(idx=idx))
                resp.raise_for_status()
                series[idx] = _parse_cboe_csv(resp.text)
        common = sorted(set(series["VIX9D"]) & set(series["VIX"]) & set(series["VIX3M"]))[
            -_TAIL_DAYS:
        ]
        if not common:
            raise ValueError("no common trading dates across VIX indices")
        reading = regime_from_series(
            [series["VIX9D"][d] for d in common],
            [series["VIX"][d] for d in common],
            [series["VIX3M"][d] for d in common],
        )
        if reading is not None:
            _last_good = reading
        return reading if reading is not None else _last_good
    except Exception as exc:  # noqa: BLE001 — advisory; a VIX outage is non-fatal
        logger.debug("VIX term-structure fetch failed: %s", exc)
        return _last_good
