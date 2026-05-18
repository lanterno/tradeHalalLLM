"""Tests for `CryptoCycleService._get_tradeable_pairs`.

Existing `test_crypto_cycle.py` has 2 happy-path tests (intersection +
fall-back). This file pins the remaining branches: operator pause
filtering, `max_pairs_per_cycle` truncation, paused-pairs exception
swallow (DB hiccup mustn't block the cycle), USDT/BUSD suffix → base
asset lookup in the halal set (so screener returning ``"BTC"`` matches
configured ``"BTCUSDT"``), and duplicate-pair dedup.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_service(
    *,
    configured_pairs: list[str],
    halal_pairs: list[str] | None = None,
    paused_pairs: set[str] | None = None,
    max_pairs: int = 10,
    paused_raises: Exception | None = None,
):
    """Build a CryptoCycleService for `_get_tradeable_pairs` testing."""
    from halal_trader.crypto.cycle import CryptoCycleService

    screener = AsyncMock()
    screener.get_halal_pairs = AsyncMock(return_value=halal_pairs or [])

    portfolio = AsyncMock()
    if paused_raises is not None:
        portfolio.get_paused_pairs = AsyncMock(side_effect=paused_raises)
    else:
        portfolio.get_paused_pairs = AsyncMock(return_value=paused_pairs or set())

    settings = MagicMock()
    settings.crypto = MagicMock()
    settings.crypto.max_pairs_per_cycle = max_pairs

    svc = CryptoCycleService(
        broker=MagicMock(),
        screener=screener,
        strategy=AsyncMock(),
        executor=AsyncMock(),
        portfolio=portfolio,
        ws_manager=MagicMock(),
        configured_pairs=configured_pairs,
    )
    svc._settings = settings
    return svc


# ── Operator pause filtering ───────────────────────────────


@pytest.mark.asyncio
async def test_paused_pair_excluded_from_tradeable():
    """A pair in `get_paused_pairs()` is filtered out — the dashboard's
    POST /api/admin/pair/.../pause takes effect on the very next cycle
    without needing a bot restart."""
    svc = _make_service(
        configured_pairs=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        halal_pairs=["BTC", "ETH", "SOL"],
        paused_pairs={"BTCUSDT"},
    )
    result = await svc._get_tradeable_pairs()
    assert "BTCUSDT" not in result
    assert "ETHUSDT" in result
    assert "SOLUSDT" in result


@pytest.mark.asyncio
async def test_pause_filter_case_insensitive():
    """Pause set is keyed UPPER; configured pairs may be any case."""
    svc = _make_service(
        configured_pairs=["btcusdt"],  # lowercase
        halal_pairs=["BTC"],
        paused_pairs={"BTCUSDT"},  # upper
    )
    result = await svc._get_tradeable_pairs()
    assert result == []  # paused even though case differs


@pytest.mark.asyncio
async def test_pause_applies_in_no_halal_fallback_path_too():
    """When the halal cache is empty (fallback to configured pairs),
    the pause filter still applies. Operator can disable a pair even
    if the screener is offline."""
    svc = _make_service(
        configured_pairs=["BTCUSDT", "ETHUSDT"],
        halal_pairs=[],  # no cache → fallback path
        paused_pairs={"ETHUSDT"},
    )
    result = await svc._get_tradeable_pairs()
    assert result == ["BTCUSDT"]


# ── max_pairs_per_cycle truncation ─────────────────────────


@pytest.mark.asyncio
async def test_max_pairs_truncates_in_no_halal_fallback():
    """The configured-pairs fallback respects `max_pairs_per_cycle`."""
    svc = _make_service(
        configured_pairs=["P1", "P2", "P3", "P4", "P5"],
        halal_pairs=[],
        max_pairs=3,
    )
    result = await svc._get_tradeable_pairs()
    assert len(result) == 3
    assert result == ["P1", "P2", "P3"]


@pytest.mark.asyncio
async def test_max_pairs_zero_returns_empty():
    """Operator-set 0 → no trading this cycle (pinned defensive)."""
    svc = _make_service(
        configured_pairs=["BTCUSDT", "ETHUSDT"],
        halal_pairs=[],
        max_pairs=0,
    )
    result = await svc._get_tradeable_pairs()
    assert result == []


# ── get_paused_pairs exception swallow ─────────────────────


@pytest.mark.asyncio
async def test_paused_pairs_db_failure_does_not_block_cycle():
    """If the DB hiccup blocks `get_paused_pairs`, the cycle continues
    without the pause filter — log at debug, treat as empty set.
    Pin so an operator's downed dashboard doesn't kill trading."""
    svc = _make_service(
        configured_pairs=["BTCUSDT"],
        halal_pairs=["BTC"],
        paused_raises=RuntimeError("connection refused"),
    )
    result = await svc._get_tradeable_pairs()
    assert result == ["BTCUSDT"]  # cycle proceeds


# ── USDT/BUSD suffix → base lookup ─────────────────────────


@pytest.mark.asyncio
async def test_pair_matches_when_screener_returns_base_only():
    """Screener returns just the base (`BTC`); configured uses the
    full symbol (`BTCUSDT`). The suffix-strip must match them up."""
    svc = _make_service(
        configured_pairs=["BTCUSDT"],
        halal_pairs=["BTC"],  # base, not the pair
    )
    result = await svc._get_tradeable_pairs()
    assert "BTCUSDT" in result


@pytest.mark.asyncio
async def test_pair_matches_full_symbol_in_screener():
    """Screener may return either form — both must match."""
    svc = _make_service(
        configured_pairs=["BTCUSDT"],
        halal_pairs=["BTCUSDT"],
    )
    result = await svc._get_tradeable_pairs()
    assert "BTCUSDT" in result


@pytest.mark.asyncio
async def test_busd_suffix_also_recognised():
    """BUSD is the second supported suffix — pin so a refactor that
    drops BUSD support is intentional."""
    svc = _make_service(
        configured_pairs=["BTCBUSD"],
        halal_pairs=["BTC"],  # base only
    )
    result = await svc._get_tradeable_pairs()
    assert "BTCBUSD" in result


@pytest.mark.asyncio
async def test_pair_without_known_suffix_uses_full_symbol():
    """A pair like `BTCETH` (no USDT/BUSD suffix) matches against the
    screener as the full symbol, not a sliced base."""
    svc = _make_service(
        configured_pairs=["BTCETH"],
        halal_pairs=["BTCETH"],
    )
    result = await svc._get_tradeable_pairs()
    assert result == ["BTCETH"]


@pytest.mark.asyncio
async def test_no_halal_matches_falls_back_to_all_configured():
    """Subtle behaviour: when the halal cache exists but NONE of the
    configured pairs matches it, the cycle falls back to every
    configured pair (minus paused). The intuition is "the screener
    must be misconfigured — don't silently halt trading"; pin so a
    refactor that drops this fallback is intentional."""
    svc = _make_service(
        configured_pairs=["UNKNOWNCOIN", "MYSTERYCOIN"],
        halal_pairs=["BTC", "ETH"],
    )
    result = await svc._get_tradeable_pairs()
    # Falls back to configured, including the unknowns.
    assert "UNKNOWNCOIN" in result
    assert "MYSTERYCOIN" in result


@pytest.mark.asyncio
async def test_no_halal_matches_fallback_still_filters_paused():
    """Even on the no-match fallback, paused pairs stay excluded."""
    svc = _make_service(
        configured_pairs=["UNKNOWNCOIN", "MYSTERYCOIN"],
        halal_pairs=["BTC", "ETH"],
        paused_pairs={"UNKNOWNCOIN"},
    )
    result = await svc._get_tradeable_pairs()
    assert "UNKNOWNCOIN" not in result
    assert "MYSTERYCOIN" in result


# ── Duplicate elimination ──────────────────────────────────


@pytest.mark.asyncio
async def test_duplicate_configured_pairs_deduplicated():
    """If `configured_pairs` accidentally repeats a symbol (legacy
    config bug), the output is still unique. Pin so a typo doesn't
    cause double-buys."""
    svc = _make_service(
        configured_pairs=["BTCUSDT", "BTCUSDT", "ETHUSDT", "BTCUSDT"],
        halal_pairs=["BTC", "ETH"],
    )
    result = await svc._get_tradeable_pairs()
    assert result.count("BTCUSDT") == 1
    assert result.count("ETHUSDT") == 1


@pytest.mark.asyncio
async def test_dedup_preserves_first_occurrence_order():
    """Dedup is order-preserving — the first occurrence wins. Pin
    so a refactor to `list(set(...))` doesn't accidentally
    randomise the order."""
    svc = _make_service(
        configured_pairs=["ETHUSDT", "BTCUSDT", "ETHUSDT"],
        halal_pairs=["BTC", "ETH"],
    )
    result = await svc._get_tradeable_pairs()
    assert result == ["ETHUSDT", "BTCUSDT"]


# ── Empty / edge cases ────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_configured_pairs_returns_empty():
    svc = _make_service(configured_pairs=[], halal_pairs=["BTC"])
    result = await svc._get_tradeable_pairs()
    assert result == []


@pytest.mark.asyncio
async def test_all_pairs_paused_returns_empty():
    svc = _make_service(
        configured_pairs=["BTCUSDT", "ETHUSDT"],
        halal_pairs=["BTC", "ETH"],
        paused_pairs={"BTCUSDT", "ETHUSDT"},
    )
    result = await svc._get_tradeable_pairs()
    assert result == []
