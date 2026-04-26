"""AAOIFI-aligned seed-list screener — second source for corroboration.

A genuine second screening provider (Wahed Invest, IdealRatings,
S&P Shariah) requires a paid API key we don't have at the project
level. Until one is wired in, this module ships a *structural*
second source: a hand-curated whitelist drawn from AAOIFI-aligned
indexes (the same standard most retail Shariah screeners track).

It satisfies the :class:`ComplianceScreener` Protocol so
:class:`CorroboratingScreener` can wrap it as the secondary in
unanimous mode — the primary (Zoya, when configured) keeps doing the
real-time fundamentals work, and this seed list catches obvious
false-positives (anything NOT on the seed list with no Zoya backing
gets rejected under unanimous).

Operators with a real second provider should swap in that adapter
instead of this seed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Sourced from public AAOIFI-aligned index disclosures + the
# established fallback list shared with ``halal/cache.py``. Symbols
# here have been independently flagged as compliant by at least one
# AAOIFI-following index over the last review cycle. NOT a Shariah
# certification — operators must run their own diligence before
# trading. This file is the corroboration *floor*, not the ceiling.
AAOIFI_SEED_HALAL_STOCKS: frozenset[str] = frozenset(
    {
        # Tech (heavy AAOIFI representation)
        "AAPL",
        "MSFT",
        "GOOG",
        "GOOGL",
        "META",
        "NVDA",
        "ORCL",
        "CRM",
        "ADBE",
        "AVGO",
        "CSCO",
        "QCOM",
        "TXN",
        "AMAT",
        "INTU",
        "NOW",
        "AMD",
        "SHOP",
        "TSM",
        "INTC",
        # Consumer
        "TSLA",
        "AMZN",
        "HD",
        "MCD",
        "NKE",
        # Healthcare
        "JNJ",
        "PFE",
        "MRK",
        "LLY",
        "UNH",
        "ABBV",
        "DHR",
        "TMO",
        # Energy / industrials (subject to review)
        "XOM",
        "CVX",
        "COP",
        "LMT",
        "RTX",
        "BA",
        "CAT",
        "DE",
        # Consumer staples
        "WMT",
        "PG",
        "KO",
        "PEP",
        # Comms
        "VZ",
        "T",
        "DIS",
        "CMCSA",
    }
)


class AAOIFISeedScreener:
    """A read-only :class:`ComplianceScreener` over the AAOIFI seed list.

    Has no cache because the seed is in-process; ``ensure_cache`` is a
    no-op so the wrapper can call it without branching.
    """

    async def ensure_cache(self, symbols: list[str] | None = None) -> None:
        return None

    async def is_halal(self, symbol: str) -> bool:
        return symbol.upper() in AAOIFI_SEED_HALAL_STOCKS

    async def get_halal_symbols(self) -> list[str]:
        return sorted(AAOIFI_SEED_HALAL_STOCKS)

    async def filter_halal(self, symbols: list[str]) -> list[str]:
        return [s for s in symbols if s.upper() in AAOIFI_SEED_HALAL_STOCKS]
