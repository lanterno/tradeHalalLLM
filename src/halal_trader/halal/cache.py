"""Halal stock cache with SQLite persistence and Zoya API integration."""

import logging

from halal_trader.domain.ports import TradeRepository
from halal_trader.halal.zoya import ZoyaClient

logger = logging.getLogger(__name__)

# Well-known halal-compliant large-cap stocks (AAOIFI-screened, commonly accepted).
# Used as a fallback when the Zoya API is not configured.
DEFAULT_HALAL_SYMBOLS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # NVIDIA
    "AVGO",  # Broadcom
    "TSM",  # Taiwan Semiconductor
    "GOOG",  # Alphabet
    "GOOGL",  # Alphabet (class A)
    "AMZN",  # Amazon
    "META",  # Meta Platforms
    "CSCO",  # Cisco
    "ADBE",  # Adobe
    "CRM",  # Salesforce
    "ORCL",  # Oracle
    "QCOM",  # Qualcomm
    "TXN",  # Texas Instruments
    "AMAT",  # Applied Materials
    "INTU",  # Intuit
    "NOW",  # ServiceNow
    "AMD",  # AMD
    "SHOP",  # Shopify
]


class HalalScreener:
    """Screens stocks for Shariah compliance using Zoya API + local cache."""

    def __init__(self, repo: TradeRepository, zoya: ZoyaClient | None = None) -> None:
        self._repo = repo
        self._zoya = zoya

    async def ensure_cache(self, symbols: list[str] | None = None) -> None:
        """Populate the halal cache, using Zoya API or defaults.

        Called at startup / daily refresh.
        """
        if await self._repo.is_cache_fresh(max_age_hours=24):
            logger.info("Halal cache is fresh, skipping refresh")
            return

        if self._zoya and self._zoya.api_key:
            target = symbols or DEFAULT_HALAL_SYMBOLS
            logger.info("Refreshing halal cache via Zoya API for %d symbols", len(target))
            results = await self._zoya.screen_bulk(target)
            for result in results:
                await self._repo.cache_halal_status(
                    symbol=result["symbol"],
                    compliance=result["compliance"],
                    detail=result.get("detail"),
                )
        else:
            logger.info(
                "Zoya API not configured — loading %d default halal symbols",
                len(DEFAULT_HALAL_SYMBOLS),
            )
            for sym in DEFAULT_HALAL_SYMBOLS:
                await self._repo.cache_halal_status(
                    symbol=sym,
                    compliance="halal",
                    detail="Default list (AAOIFI pre-screened large-cap)",
                )

    async def is_halal(self, symbol: str) -> bool:
        """Check if a specific symbol is halal (from cache)."""
        status = await self._repo.get_halal_status(symbol)
        return status == "halal"

    async def get_halal_symbols(self) -> list[str]:
        """Return all cached halal-compliant symbols."""
        return await self._repo.get_halal_symbols()

    async def filter_halal(self, symbols: list[str]) -> list[str]:
        """Filter a list of symbols, keeping only halal ones."""
        halal = set(await self.get_halal_symbols())
        return [s for s in symbols if s in halal]
