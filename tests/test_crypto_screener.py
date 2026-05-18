"""Tests for :class:`CryptoHalalScreener`'s rule engine.

The CoinGecko fetch and the cache write paths are integration; this
file focuses on the pure compliance rules — the bit that decides
``halal`` / ``not_halal`` / ``doubtful`` from a coin's metadata.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from halal_trader.crypto.screener import CryptoHalalScreener


def _screener(**overrides) -> CryptoHalalScreener:
    repo = MagicMock()
    return CryptoHalalScreener(repo=repo, **overrides)


# ── Manual overrides ──────────────────────────────────────────


def test_manual_halal_override_wins_immediately():
    """Tokens in the halal-overrides set return ``halal`` without
    running any other check (well-known compliant coins)."""
    s = _screener()
    coin = {
        "id": "bitcoin",
        "symbol": "btc",
        "tags": ["meme"],  # would otherwise trigger prohibited_tags
    }
    status, criteria = s._screen_coin(coin)
    assert status == "halal"
    assert criteria.get("manual_override") is True


def test_manual_deny_override_wins_immediately():
    """A coin in the deny-list is rejected even if it would otherwise
    pass every check."""
    s = _screener(deny_overrides={"sketchy-coin"})
    coin = {"id": "sketchy-coin", "symbol": "sk", "market_cap": 5_000_000_000}
    status, criteria = s._screen_coin(coin)
    assert status == "not_halal"
    assert criteria["manual_deny"] is True


# ── Category filter ───────────────────────────────────────────


def test_prohibited_category_rejects():
    s = _screener()
    coin = {
        "id": "some-casino-token",
        "symbol": "cas",
        "categories": ["Gambling", "Decentralized Finance"],
    }
    status, criteria = s._screen_coin(coin)
    assert status == "not_halal"
    assert "gambling" in criteria["prohibited_categories"]


def test_lending_borrowing_category_is_prohibited():
    """Interest-bearing protocols (Aave, Compound) → not_halal."""
    s = _screener()
    coin = {"id": "aave", "symbol": "aave", "categories": ["lending-borrowing"]}
    status, _ = s._screen_coin(coin)
    assert status == "not_halal"


# ── Tag filter ───────────────────────────────────────────────


def test_meme_token_tag_rejects():
    s = _screener()
    coin = {
        "id": "shiba-inu",
        "symbol": "shib",
        "tags": ["meme-token", "dog"],
    }
    status, criteria = s._screen_coin(coin)
    assert status == "not_halal"
    assert "meme-token" in criteria["prohibited_tags"]


def test_leveraged_token_tag_rejects():
    s = _screener()
    coin = {"id": "btc-3l", "symbol": "btc3l", "tags": ["leveraged-token"]}
    status, _ = s._screen_coin(coin)
    assert status == "not_halal"


# ── Legitimacy / market cap filter ───────────────────────────


def test_low_market_cap_returns_doubtful():
    """Below the legitimacy threshold but no other red flag → doubtful
    (not outright not_halal — could be a legitimate small-cap)."""
    s = _screener(min_market_cap=1_000_000_000)
    coin = {
        "id": "small-coin",
        "symbol": "sc",
        "categories": [],
        "market_cap": 50_000_000,  # well below threshold
    }
    status, criteria = s._screen_coin(coin)
    assert status == "doubtful"
    assert criteria["legitimacy_check"] == "failed"


def test_zero_or_missing_market_cap_falls_back_to_doubtful():
    s = _screener(min_market_cap=1_000_000_000)
    coin = {"id": "x", "symbol": "x", "categories": []}  # no market_cap
    status, _ = s._screen_coin(coin)
    assert status == "doubtful"


# ── All-passes path ──────────────────────────────────────────


def test_well_capped_clean_token_returns_halal():
    s = _screener(min_market_cap=1_000_000_000)
    coin = {
        "id": "some-defi-token",
        "symbol": "sdt",
        "categories": ["smart-contract-platform"],
        "market_cap": 5_000_000_000,
    }
    status, criteria = s._screen_coin(coin)
    assert status == "halal"
    assert criteria["all_checks"] == "passed"


# ── Helpers ─────────────────────────────────────────────────


def test_get_categories_normalises_to_lowercase_dashes():
    s = _screener()
    cats = s._get_categories({"categories": ["Smart Contract Platform", "Layer 1"]})
    assert "smart-contract-platform" in cats
    assert "layer-1" in cats


def test_get_categories_empty_when_no_keys():
    s = _screener()
    assert s._get_categories({}) == set()


def test_extract_category_returns_unknown_for_no_categories():
    s = _screener()
    assert s._extract_category({}) == "unknown"


def test_extract_category_caps_at_three():
    s = _screener()
    coin = {"categories": [f"c{i}" for i in range(10)]}
    out = s._extract_category(coin)
    # Only 3 should be in the rendered string.
    assert out.count(",") == 2  # 3 items → 2 commas
