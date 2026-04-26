"""Halal sector-rotation limits.

Even when every individual ticker passes Shariah screening, a portfolio
that's 100% in one sector breaches diversification guidance and (more
practically) concentrates idiosyncratic risk against us. This module
caps the % of equity allocated to any single GICS-style sector and
returns a reason when a candidate buy would breach the cap.

Sectors are looked up via a small in-process map (extend with Alpaca
``get_stock_snapshot`` sector field over time). Symbols missing from the
map default to ``"unknown"`` and are bucketed together — operators can
still trade them but they share the unknown-sector cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# A small, hand-maintained seed map. Real symbols/sectors should come
# from Alpaca's snapshot or a fundamentals provider; this keeps tests
# meaningful without a live data dependency. Only SYMBOLS already in the
# fallback halal whitelist are listed.
_DEFAULT_SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "GOOG": "Technology",
    "META": "Technology",
    "NVDA": "Technology",
    "ORCL": "Technology",
    "CRM": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "ADBE": "Technology",
    "CSCO": "Technology",
    "QCOM": "Technology",
    "AVGO": "Technology",
    "TXN": "Technology",
    "NOW": "Technology",
    "TSM": "Technology",
    "TSLA": "Consumer Discretionary",
    "AMZN": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "MRK": "Healthcare",
    "LLY": "Healthcare",
    "UNH": "Healthcare",
    "ABBV": "Healthcare",
    "DHR": "Healthcare",
    "TMO": "Healthcare",
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "WMT": "Consumer Staples",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "VZ": "Communication Services",
    "T": "Communication Services",
    "DIS": "Communication Services",
    "CMCSA": "Communication Services",
    "LMT": "Industrials",
    "RTX": "Industrials",
    "BA": "Industrials",
    "CAT": "Industrials",
    "DE": "Industrials",
}

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
    table = sector_map or _DEFAULT_SECTOR_MAP
    return table.get(symbol.upper(), UNKNOWN_SECTOR)


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
) -> tuple[bool, str]:
    """Return ``(allowed, reason)`` for a candidate buy.

    The cap is total post-trade exposure — the candidate's notional gets
    added to the existing bucket before the comparison, so a +1% buy on
    top of 39% existing exposure still trips the 40% cap.
    """
    if allocation.total_equity <= 0:
        return True, ""
    sector = sector_for(symbol, sector_map=sector_map)
    current = allocation.by_sector.get(sector, 0.0)
    post = current + notional_usd
    post_pct = post / allocation.total_equity
    if post_pct > max_sector_pct:
        return False, (
            f"sector cap: {sector} would be {post_pct:.0%} of equity "
            f"after this buy (cap {max_sector_pct:.0%})"
        )
    return True, ""
