"""Halal stock cache with SQLite persistence and Zoya API integration."""

import logging

from halal_trader.config import HalalSettings, get_settings
from halal_trader.db.repos import StockHalalCacheRepo
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

    def __init__(
        self,
        repo: StockHalalCacheRepo,
        zoya: ZoyaClient | None = None,
        *,
        halal_settings: HalalSettings | None = None,
    ) -> None:
        self._repo = repo
        self._zoya = zoya
        # Settings is a singleton; we accept an override only so tests can
        # tighten the TTL without touching the global cache. Live code
        # should leave halal_settings=None and let get_settings() decide.
        self._halal = halal_settings or get_settings().halal

    async def ensure_cache(self, symbols: list[str] | None = None, *, force: bool = False) -> None:
        """Populate the halal cache, using Zoya API or defaults.

        Called at startup, on the configured TTL, and from
        :meth:`refresh_if_stale` (the mid-cycle hook). ``force=True``
        bypasses the freshness check — used by mid-cycle refresh, which
        has already decided to refresh based on its tighter window.
        """
        if not force and await self._repo.is_cache_fresh(
            max_age_hours=self._halal.cache_max_age_hours
        ):
            logger.info(
                "Halal cache fresh (TTL %dh), skipping refresh",
                self._halal.cache_max_age_hours,
            )
            return

        if self._zoya and self._zoya.api_key:
            target = symbols or DEFAULT_HALAL_SYMBOLS
            logger.info("Refreshing halal cache via Zoya API for %d symbols", len(target))
            results = await self._zoya.screen_bulk(target)
            errored = 0
            for result in results:
                # Skip transient API failures — caching them as "doubtful"
                # poisons the cache for the full TTL, so a momentary Zoya
                # outage during the single pre-market pass starves the
                # universe all day (observed 2026-05-27). Leaving the prior
                # verdict (or no row) lets the next refresh retry.
                if result.get("error"):
                    errored += 1
                    continue
                await self._repo.cache_halal_status(
                    symbol=result["symbol"],
                    compliance=result["compliance"],
                    detail=result.get("detail"),
                )
            if errored:
                logger.warning(
                    "Halal refresh: %d/%d symbols failed screening (transient) — "
                    "not cached, prior verdicts preserved; will retry next refresh",
                    errored,
                    len(target),
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

    async def refresh_if_stale(self, symbols: list[str] | None = None) -> bool:
        """Mid-cycle hook — refresh if the cache is older than the soft threshold.

        Distinct from :meth:`ensure_cache`: that uses the *hard* TTL
        (``cache_max_age_hours``) and is called once at startup. This
        one uses ``midcycle_refresh_hours`` (a tighter window) and is
        meant to be called at the top of each cycle so we don't wait
        the full TTL between refreshes.

        Returns ``True`` if a refresh actually ran.
        """
        soft = self._halal.midcycle_refresh_hours
        if await self._repo.is_cache_fresh(max_age_hours=soft):
            return False
        logger.info("Halal cache older than %dh — mid-cycle refresh", soft)
        await self.ensure_cache(symbols=symbols, force=True)
        return True
