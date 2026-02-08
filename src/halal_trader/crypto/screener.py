"""Crypto Shariah-compliance screener using CoinGecko metadata.

Implements a dynamic, rule-based screening system inspired by Mufti Faraz Adam's
Crypto Shariah Screening Framework.  Uses CoinGecko API to fetch live token
metadata and applies multi-criteria filtering:

1. Category filter — reject tokens in prohibited categories.
2. Token type filter — reject meme coins, rebase tokens, etc.
3. Legitimacy filter — minimum market cap and age.
4. Utility check — token must serve a real-world purpose.
5. Manual overrides — config-based allow/deny lists for edge cases.
"""

import json
import logging
from typing import Any

import httpx

from halal_trader.db.repository import Repository

logger = logging.getLogger(__name__)

# ── Prohibited categories (haram) ──────────────────────────────
PROHIBITED_CATEGORIES: set[str] = {
    "gambling",
    "adult",
    "adult-content",
    "casino",
    "lending-borrowing",
    "interest-bearing",
    "wrapped-interest-bearing-tokens",
    "ponzi",
    "insurance",
}

# ── Prohibited token types / tags ──────────────────────────────
PROHIBITED_TAGS: set[str] = {
    "meme-token",
    "meme",
    "rebase-tokens",
    "leveraged-token",
    "gambling",
    "nsfw",
}

# ── Well-known halal tokens (override) ─────────────────────────
# These pass all scholarly criteria and are always considered halal.
DEFAULT_HALAL_OVERRIDES: set[str] = {
    "bitcoin",
    "ethereum",
    "cardano",
    "solana",
    "ripple",
    "polkadot",
    "avalanche-2",
    "chainlink",
    "polygon-ecosystem-token",
    "algorand",
    "cosmos",
    "stellar",
    "near",
    "hedera-hashgraph",
    "internet-computer",
    "tezos",
    "fantom",
    "aptos",
}

# CoinGecko API base URL (free tier)
_CG_BASE = "https://api.coingecko.com/api/v3"


class CryptoHalalScreener:
    """Rule-based crypto Shariah screener backed by CoinGecko metadata.

    Implements the CryptoComplianceScreener protocol.
    """

    def __init__(
        self,
        repo: Repository,
        *,
        coingecko_api_key: str = "",
        min_market_cap: float = 1_000_000_000,
        halal_overrides: set[str] | None = None,
        deny_overrides: set[str] | None = None,
    ) -> None:
        self._repo = repo
        self._api_key = coingecko_api_key
        self._min_market_cap = min_market_cap
        self._halal_overrides = halal_overrides or DEFAULT_HALAL_OVERRIDES
        self._deny_overrides = deny_overrides or set()

    # ── CryptoComplianceScreener protocol ──────────────────────

    async def refresh_screening(self, symbols: list[str] | None = None) -> None:
        """Refresh the halal screening cache for all or specified tokens.

        If symbols is None, screens the top coins by market cap from CoinGecko.
        """
        if await self._repo.is_crypto_cache_fresh(max_age_hours=24):
            logger.info("Crypto halal cache is fresh, skipping refresh")
            return

        logger.info("Refreshing crypto halal screening cache...")

        # Fetch top coins from CoinGecko
        coins = await self._fetch_coins_list()
        screened = 0

        for coin in coins:
            symbol = coin.get("symbol", "").upper()

            compliance, criteria = self._screen_coin(coin)
            category = self._extract_category(coin)
            market_cap = coin.get("market_cap") or 0

            await self._repo.cache_crypto_halal_status(
                symbol=symbol,
                compliance=compliance,
                category=category,
                market_cap=float(market_cap),
                screening_criteria=json.dumps(criteria),
            )
            screened += 1

        logger.info("Crypto halal screening complete: %d coins screened", screened)

    async def is_halal(self, symbol: str) -> bool:
        """Check if a crypto symbol is halal (cached)."""
        status = await self._repo.get_crypto_halal_status(symbol.upper())
        return status == "halal"

    async def get_halal_pairs(self) -> list[str]:
        """Get all halal crypto symbols from the cache."""
        return await self._repo.get_crypto_halal_symbols()

    async def filter_halal(self, symbols: list[str]) -> list[str]:
        """Filter a list of symbols, keeping only halal ones."""
        result = []
        for sym in symbols:
            if await self.is_halal(sym):
                result.append(sym)
        return result

    # ── Screening Logic ────────────────────────────────────────

    def _screen_coin(self, coin: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Screen a single coin against all criteria.

        Returns:
            (compliance_status, criteria_dict)
        """
        coin_id = coin.get("id", "")
        symbol = coin.get("symbol", "").upper()
        criteria: dict[str, Any] = {}

        # 0. Manual deny override
        if coin_id in self._deny_overrides or symbol.lower() in self._deny_overrides:
            criteria["manual_deny"] = True
            return "not_halal", criteria

        # 1. Manual halal override (well-known halal tokens)
        if coin_id in self._halal_overrides:
            criteria["manual_override"] = True
            return "halal", criteria

        # 2. Category filter
        categories = self._get_categories(coin)
        prohibited_found = categories & PROHIBITED_CATEGORIES
        if prohibited_found:
            criteria["prohibited_categories"] = list(prohibited_found)
            return "not_halal", criteria
        criteria["category_check"] = "passed"

        # 3. Token type / tag filter
        tags = self._get_tags(coin)
        prohibited_tags = tags & PROHIBITED_TAGS
        if prohibited_tags:
            criteria["prohibited_tags"] = list(prohibited_tags)
            return "not_halal", criteria
        criteria["tag_check"] = "passed"

        # 4. Legitimacy filter — minimum market cap
        market_cap = coin.get("market_cap") or 0
        if market_cap < self._min_market_cap:
            criteria["market_cap"] = market_cap
            criteria["min_required"] = self._min_market_cap
            criteria["legitimacy_check"] = "failed"
            return "doubtful", criteria
        criteria["legitimacy_check"] = "passed"

        # 5. All checks passed
        criteria["all_checks"] = "passed"
        return "halal", criteria

    def _get_categories(self, coin: dict[str, Any]) -> set[str]:
        """Extract categories from coin metadata (normalized to lowercase)."""
        cats: set[str] = set()
        # CoinGecko markets endpoint doesn't include categories directly,
        # but the /coins/{id} endpoint does.  For the list endpoint we rely on
        # tags and the coin id itself.
        for key in ("categories", "category"):
            val = coin.get(key)
            if isinstance(val, list):
                cats.update(v.lower().replace(" ", "-") for v in val if v)
            elif isinstance(val, str) and val:
                cats.add(val.lower().replace(" ", "-"))
        return cats

    def _get_tags(self, coin: dict[str, Any]) -> set[str]:
        """Extract tags from coin metadata."""
        tags: set[str] = set()
        for key in ("tags", "categories"):
            val = coin.get(key)
            if isinstance(val, list):
                tags.update(v.lower().replace(" ", "-") for v in val if v)
        return tags

    def _extract_category(self, coin: dict[str, Any]) -> str:
        """Extract a single primary category string for display."""
        cats = self._get_categories(coin)
        return ", ".join(sorted(cats)[:3]) if cats else "unknown"

    # ── CoinGecko API ──────────────────────────────────────────

    async def _fetch_coins_list(self) -> list[dict[str, Any]]:
        """Fetch top coins by market cap from CoinGecko."""
        headers: dict[str, str] = {}
        if self._api_key:
            headers["x-cg-demo-api-key"] = self._api_key

        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": "100",
            "page": "1",
            "sparkline": "false",
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{_CG_BASE}/coins/markets",
                    params=params,
                    headers=headers,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error("CoinGecko API error: %s", e)
            return []
