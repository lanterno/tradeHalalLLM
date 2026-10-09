"""The strict-halal core portfolio's rules.

Pure functions, shared by the forward book (research/forward_book.py) and,
later, by execution, so the record and the orders follow one rule:

* **targets:** the largest ``TOP_N`` eligible names by market cap, weighted
  by cap; a company's share classes split its cap. Eligible means halal on
  the newest screen and in ``SECTOR``: technology only (software & internet,
  computer hardware, semiconductors and chip equipment, as
  halal/sector_limits.py groups the screen's industries), the operator's
  decision of 2026-10-09;
* **trades:** a holding the screen no longer passes goes to zero; a new name
  enters at its target; a holding whose weight has drifted outside its band
  (``BAND`` of its target, at least ``BAND_FLOOR``) goes back to target;
  everything else is left alone, and the result is renormalised. Doing
  nothing inside the band is what keeps turnover, and so cost, near an
  index fund's.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from halal_trader.halal.sector_limits import TECHNOLOGY

TOP_N = 100
SECTOR: str | None = TECHNOLOGY  # None: every sector
# The day the targets last changed (2026-10-12: technology only). A monthly
# rebalance is due again at the first run on or after it, so the account moves
# to the new targets then rather than at the next month's start.
RULE_SINCE = date(2026, 10, 12)
BAND = 0.25
BAND_FLOOR = 0.002
COST_BPS = 5.0


def split_share_classes(
    caps: Mapping[str, float], ciks: Mapping[str, int | None]
) -> dict[str, float]:
    """Each class of a multi-class company gets an equal part of its one cap."""
    classes: dict[int, int] = {}
    for symbol in caps:
        cik = ciks.get(symbol)
        if cik is not None:
            classes[cik] = classes.get(cik, 0) + 1
    return {
        s: c / classes[ciks[s]] if ciks.get(s) is not None else c  # type: ignore[index]
        for s, c in caps.items()
    }


def targets(caps: Mapping[str, float], top_n: int = TOP_N) -> dict[str, float]:
    """Cap weights over the ``top_n`` largest positive caps."""
    largest = sorted(((c, s) for s, c in caps.items() if c > 0), reverse=True)[:top_n]
    total = sum(c for c, _ in largest)
    return {s: c / total for c, s in largest} if total > 0 else {}


def outside_band(current: float, target: float, band: float = BAND) -> bool:
    return abs(current - target) > max(band * target, BAND_FLOOR)


def rebalance(
    current: Mapping[str, float],
    target: Mapping[str, float],
    eligible: set[str],
    *,
    band: float = BAND,
) -> dict[str, float]:
    """New weights: ineligible to zero, out-of-band names to target, the rest held."""
    new: dict[str, float] = {}
    for symbol in set(current) | set(target):
        if symbol not in eligible:
            continue  # the screen no longer passes it: sold
        now, goal = current.get(symbol, 0.0), target.get(symbol, 0.0)
        if now == 0.0 and goal > 0.0:
            # Entering always trades. The band's floor exists to skip tiny
            # adjustments; applied to an entry it kept every name with a target
            # under 0.2% out for good (the first plan bought 64 of the top 100).
            new[symbol] = goal
        else:
            new[symbol] = goal if outside_band(now, goal, band) else now
    new = {s: w for s, w in new.items() if w > 0}
    total = sum(new.values())
    return {s: w / total for s, w in new.items()} if total > 0 else {}


def turnover(before: Mapping[str, float], after: Mapping[str, float]) -> float:
    return sum(abs(after.get(s, 0.0) - before.get(s, 0.0)) for s in set(before) | set(after))
