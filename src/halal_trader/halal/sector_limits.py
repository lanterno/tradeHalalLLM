"""Halal sector-rotation limits.

Even when every individual ticker passes Shariah screening, a portfolio
that's 100% in one sector breaches diversification guidance and (more
practically) concentrates idiosyncratic risk against us. This module
caps the % of equity allocated to any single sector and returns a reason
when a candidate buy would breach the cap.

A symbol's sector is its industry on the newest strict screen (its SEC SIC
code, grouped by compliance/sectors.py), so every name the bot can trade
has one; ``SECTOR_OVERRIDES`` corrects the few large names whose SIC code
misleads. The three technology industries form one exempt "Technology"
sector. A symbol the screen does not hold is ``"unknown"``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from halal_trader.compliance.sectors import sector_of

TECHNOLOGY = "Technology"

# Sectors that are NOT subject to the per-sector cap. The halal stocks
# universe is structurally heavy on US large-cap Technology (most of
# the Shariah-compliant tickers are software / semis / cloud); capping
# Tech at the same threshold as every other sector would force the bot
# to leave most of its high-conviction setups on the table. Operator
# policy as of 2026-05-21: exempt Technology only.
DEFAULT_EXEMPT_SECTORS: frozenset[str] = frozenset({TECHNOLOGY})

# The screen's industries that make up Technology.
_TECH_INDUSTRIES = frozenset({"Software & internet", "Semiconductors", "Computer hardware"})

# Large halal names whose SIC code says the wrong thing: life-science tool
# makers filed as measuring or lab instruments (they are healthcare, and
# must not ride the Technology exemption), chip-equipment makers filed as
# machinery or optics, an industrial filed as surgical instruments. Since the
# core holds Technology only (2026-10-09), the group was reviewed name by
# name against GICS: instrument, water, radiation, auction and solar-tracker
# businesses are not technology; internet platforms and solar-chip makers
# (GICS semiconductors) stay in it.
SECTOR_OVERRIDES: dict[str, str] = {
    "TMO": "Healthcare",
    "DHR": "Healthcare",
    "A": "Healthcare",
    "WAT": "Healthcare",
    "RVTY": "Healthcare",
    "BIO": "Healthcare",
    "TXG": "Healthcare",
    "BRKR": "Healthcare",
    "LRCX": TECHNOLOGY,
    "KLAC": TECHNOLOGY,
    "ROP": TECHNOLOGY,
    "MMM": "Industrials",
    "ROK": "Industrials",
    "VLTO": "Industrials",
    "FTV": "Industrials",
    "MIR": "Industrials",
    "RBA": "Industrials",
    "NXT": "Industrials",
    "IMAX": "Communications & media",
}


def cap_sector(symbol: str, sic_description: str | None) -> str:
    """The sector the cap counts ``symbol`` in, from its SIC description."""
    if symbol.upper() in SECTOR_OVERRIDES:
        return SECTOR_OVERRIDES[symbol.upper()]
    industry = sector_of(sic_description)
    return TECHNOLOGY if industry in _TECH_INDUSTRIES else industry


UNKNOWN_SECTOR = "unknown"


@dataclass(frozen=True)
class SectorAllocation:
    """How much of the portfolio is currently in each sector (in USD)."""

    by_sector: Mapping[str, float]
    total_equity: float

    def pct(self, sector: str) -> float:
        if self.total_equity <= 0:
            return 0.0
        return self.by_sector.get(sector, 0.0) / self.total_equity


def sector_for(symbol: str, *, sector_map: Mapping[str, str] | None = None) -> str:
    """``symbol``'s cap sector: from ``sector_map`` (HalalScreener.sectors, the
    screen's view), else an override, else unknown."""
    upper = symbol.upper()
    if sector_map and upper in sector_map:
        return sector_map[upper]
    return SECTOR_OVERRIDES.get(upper, UNKNOWN_SECTOR)


def compute_allocation(
    positions_value: Mapping[str, float],
    *,
    total_equity: float,
    sector_map: Mapping[str, str] | None = None,
) -> SectorAllocation:
    """Sum existing position values into per-sector buckets."""
    buckets: dict[str, float] = {}
    for symbol, value in positions_value.items():
        s = sector_for(symbol, sector_map=sector_map)
        buckets[s] = buckets.get(s, 0.0) + float(value)
    return SectorAllocation(by_sector=buckets, total_equity=total_equity)


def check_buy_against_limits(
    *,
    symbol: str,
    notional_usd: float,
    allocation: SectorAllocation,
    max_sector_pct: float = 0.40,
    sector_map: Mapping[str, str] | None = None,
    exempt_sectors: Iterable[str] | None = None,
) -> tuple[bool, str]:
    """Return ``(allowed, reason)`` for a candidate buy.

    The cap is total post-trade exposure — the candidate's notional gets
    added to the existing bucket before the comparison, so a +1% buy on
    top of 39% existing exposure still trips the 40% cap.

    ``exempt_sectors`` short-circuits the cap check for sectors the
    operator has whitelisted (defaults to :data:`DEFAULT_EXEMPT_SECTORS`,
    currently ``{"Technology"}`` — see that constant for the rationale).
    Pass an explicit empty set to disable the exemption.
    """
    if allocation.total_equity <= 0:
        return True, ""
    sector = sector_for(symbol, sector_map=sector_map)
    exempt = DEFAULT_EXEMPT_SECTORS if exempt_sectors is None else frozenset(exempt_sectors)
    if sector in exempt:
        return True, ""
    current = allocation.by_sector.get(sector, 0.0)
    post = current + notional_usd
    post_pct = post / allocation.total_equity
    if post_pct > max_sector_pct:
        return False, (
            f"sector cap: {sector} would be {post_pct:.0%} of equity "
            f"after this buy (cap {max_sector_pct:.0%})"
        )
    return True, ""
