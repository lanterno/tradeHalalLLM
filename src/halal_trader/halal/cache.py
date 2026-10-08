"""The day-trader's halal screen: the strict in-house screen, cached as its trading universe.

``HalalScreener`` is the ``ComplianceScreener`` the cycle, the reactor and the
order boundary (``TradeExecutor._check_halal``) use. With a database engine
-- as the bot always runs -- its verdicts are the strict in-house screen's
(halal/strict.py), failing closed on a stale or missing screen, and
``halal_cache`` holds the trading universe: the largest names that screen
passes (``settings.halal.universe_size``).

Without an engine (tests, offline tooling) it falls back to the curated
default list below, which is no longer what the bot trades: seven of its
twenty names fail the strict screen.
"""

import logging
from collections.abc import Callable
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.config import HalalSettings, get_settings
from halal_trader.db.repos import StockHalalCacheRepo
from halal_trader.halal import strict
from halal_trader.market_hours import today_eastern

logger = logging.getLogger(__name__)

# A curated list of large caps, once believed AAOIFI-compliant. Research tools
# still use it as a fixed symbol set; the bot's halal gate does NOT (AMZN,
# INTU, META, NVDA, ORCL, SHOP and TSM fail the strict screen).
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
    """Screens stocks for Shariah compliance: the strict in-house screen, and a cache."""

    def __init__(
        self,
        repo: StockHalalCacheRepo,
        *,
        halal_settings: HalalSettings | None = None,
        engine: AsyncEngine | None = None,
        today: Callable[[], date] = today_eastern,
    ) -> None:
        self._repo = repo
        # Settings is a singleton; we accept an override only so tests can
        # tighten the TTL without touching the global cache. Live code
        # should leave halal_settings=None and let get_settings() decide.
        self._halal = halal_settings or get_settings().halal
        # The strict screen lives in the database; no engine = legacy list mode.
        self._engine = engine
        self._today = today

    async def sectors(self, symbols: list[str]) -> dict[str, str]:
        """Each symbol's sector for the sector cap, from the newest strict
        screen (halal/sector_limits.py); a symbol it does not hold is absent."""
        from halal_trader.halal.sector_limits import cap_sector

        if self._engine is None or not symbols:
            return {}
        as_of = await strict.newest_screen(self._engine, on_or_before=self._today())
        if as_of is None:
            return {}
        rows = await strict.screen_rows(self._engine, as_of, symbols=[s.upper() for s in symbols])
        return {r.symbol: cap_sector(r.symbol, r.sic_description) for r in rows}

    async def ensure_cache(self, symbols: list[str] | None = None, *, force: bool = False) -> None:
        """Populate the halal cache from the strict screen, or the default list.

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

        if self._engine is not None:
            await self._refresh_from_strict_screen(self._engine)
            return

        # No database (tests, offline tooling): the curated default list.
        for sym in DEFAULT_HALAL_SYMBOLS:
            await self._repo.cache_halal_status(
                symbol=sym,
                compliance="halal",
                detail="Default list (AAOIFI pre-screened large-cap)",
            )

    async def _refresh_from_strict_screen(self, engine: AsyncEngine) -> None:
        """Make ``halal_cache`` the strict screen's largest halal names, and nothing else.

        A stale or missing screen empties it (fail closed): the cycle and the
        reactor then have nothing to buy, and the order boundary refuses
        regardless (:meth:`is_halal` reads the screen itself).
        """
        size = self._halal.universe_size
        as_of, names = await strict.halal_universe(engine, today=self._today(), limit=size)
        if not names:
            logger.error(
                "Halal universe EMPTY: no fresh strict screen (newest: %s) -- nothing is "
                "tradable until the weekly screen runs",
                as_of,
            )
        verdicts = dict.fromkeys(names, ("halal", f"strict screen {as_of}"))
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM halal_cache WHERE NOT (symbol = ANY(:keep))"),
                {"keep": sorted(verdicts)},
            )
        for symbol, (compliance, detail) in verdicts.items():
            await self._repo.cache_halal_status(symbol=symbol, compliance=compliance, detail=detail)
        logger.info(
            "Halal universe: %d of the strict screen's largest halal names (screen %s)",
            sum(1 for c, _ in verdicts.values() if c == "halal"),
            as_of,
        )

    async def is_halal(self, symbol: str) -> bool:
        """Is ``symbol`` halal right now? The order boundary's question.

        With an engine: the newest fresh strict screen must pass it, read at
        the moment of asking, not from the cache. Without one: the cached verdict.
        """
        if self._engine is not None:
            v = await strict.verdict(self._engine, symbol, today=self._today())
            if not v.halal:
                logger.info("halal gate: %s", v.reason)
            return v.halal
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
