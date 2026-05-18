"""Tests for :class:`GetTradeablePairsStage`.

Pins: operator pause filtering, ``max_pairs_per_cycle`` truncation,
paused-pairs exception swallow (DB hiccup mustn't block the cycle),
USDT/BUSD suffix → base asset lookup in the halal set (so screener
returning ``"BTC"`` matches configured ``"BTCUSDT"``), duplicate-pair
dedup, and the no-match fallback to configured pairs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from halal_trader.core.cycle_pipeline import CycleState
from halal_trader.core.cycle_stages import GetTradeablePairsStage


def _make_stage(
    *,
    configured_pairs: list[str],
    halal_pairs: list[str] | None = None,
    paused_pairs: set[str] | None = None,
    max_pairs: int = 10,
    paused_raises: Exception | None = None,
) -> GetTradeablePairsStage:
    """Build a :class:`GetTradeablePairsStage` for testing."""
    screener = AsyncMock()
    screener.get_halal_pairs = AsyncMock(return_value=halal_pairs or [])

    portfolio = AsyncMock()
    if paused_raises is not None:
        portfolio.get_paused_pairs = AsyncMock(side_effect=paused_raises)
    else:
        portfolio.get_paused_pairs = AsyncMock(return_value=paused_pairs or set())

    return GetTradeablePairsStage(
        screener=screener,
        portfolio=portfolio,
        configured_pairs=configured_pairs,
        max_pairs=max_pairs,
    )


async def _run(stage: GetTradeablePairsStage) -> list[str]:
    state = CycleState()
    await stage.run(state)
    return state.halal_pairs


# ── Operator pause filtering ───────────────────────────────


@pytest.mark.asyncio
async def test_paused_pair_excluded_from_tradeable():
    """A pair in ``get_paused_pairs()`` is filtered out — the dashboard's
    ``POST /api/admin/pair/.../pause`` takes effect on the very next
    cycle without needing a bot restart."""
    stage = _make_stage(
        configured_pairs=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        halal_pairs=["BTC", "ETH", "SOL"],
        paused_pairs={"BTCUSDT"},
    )
    result = await _run(stage)
    assert "BTCUSDT" not in result
    assert "ETHUSDT" in result
    assert "SOLUSDT" in result


@pytest.mark.asyncio
async def test_pause_filter_case_insensitive():
    """Pause set is keyed UPPER; configured pairs may be any case."""
    stage = _make_stage(
        configured_pairs=["btcusdt"],  # lowercase
        halal_pairs=["BTC"],
        paused_pairs={"BTCUSDT"},  # upper
    )
    assert await _run(stage) == []  # paused even though case differs


@pytest.mark.asyncio
async def test_pause_applies_in_no_halal_fallback_path_too():
    """When the halal cache is empty (fallback to configured pairs),
    the pause filter still applies."""
    stage = _make_stage(
        configured_pairs=["BTCUSDT", "ETHUSDT"],
        halal_pairs=[],
        paused_pairs={"ETHUSDT"},
    )
    assert await _run(stage) == ["BTCUSDT"]


# ── max_pairs_per_cycle truncation ─────────────────────────


@pytest.mark.asyncio
async def test_max_pairs_truncates_in_no_halal_fallback():
    """The configured-pairs fallback respects ``max_pairs_per_cycle``."""
    stage = _make_stage(
        configured_pairs=["P1", "P2", "P3", "P4", "P5"],
        halal_pairs=[],
        max_pairs=3,
    )
    result = await _run(stage)
    assert len(result) == 3
    assert result == ["P1", "P2", "P3"]


@pytest.mark.asyncio
async def test_max_pairs_zero_returns_empty():
    """Operator-set 0 → no trading this cycle (pinned defensive)."""
    stage = _make_stage(
        configured_pairs=["BTCUSDT", "ETHUSDT"],
        halal_pairs=[],
        max_pairs=0,
    )
    assert await _run(stage) == []


# ── get_paused_pairs exception swallow ─────────────────────


@pytest.mark.asyncio
async def test_paused_pairs_db_failure_does_not_block_cycle():
    """If the DB hiccup blocks ``get_paused_pairs``, the cycle continues
    without the pause filter — log at debug, treat as empty set."""
    stage = _make_stage(
        configured_pairs=["BTCUSDT"],
        halal_pairs=["BTC"],
        paused_raises=RuntimeError("connection refused"),
    )
    assert await _run(stage) == ["BTCUSDT"]


# ── USDT/BUSD suffix → base lookup ─────────────────────────


@pytest.mark.asyncio
async def test_pair_matches_when_screener_returns_base_only():
    """Screener returns just the base (``BTC``); configured uses the
    full symbol (``BTCUSDT``). The suffix-strip must match them up."""
    stage = _make_stage(
        configured_pairs=["BTCUSDT"],
        halal_pairs=["BTC"],
    )
    assert "BTCUSDT" in await _run(stage)


@pytest.mark.asyncio
async def test_pair_matches_full_symbol_in_screener():
    """Screener may return either form — both must match."""
    stage = _make_stage(
        configured_pairs=["BTCUSDT"],
        halal_pairs=["BTCUSDT"],
    )
    assert "BTCUSDT" in await _run(stage)


@pytest.mark.asyncio
async def test_busd_suffix_also_recognised():
    """BUSD is the second supported suffix."""
    stage = _make_stage(
        configured_pairs=["BTCBUSD"],
        halal_pairs=["BTC"],
    )
    assert "BTCBUSD" in await _run(stage)


@pytest.mark.asyncio
async def test_pair_without_known_suffix_uses_full_symbol():
    """A pair like ``BTCETH`` (no USDT/BUSD suffix) matches against the
    screener as the full symbol, not a sliced base."""
    stage = _make_stage(
        configured_pairs=["BTCETH"],
        halal_pairs=["BTCETH"],
    )
    assert await _run(stage) == ["BTCETH"]


@pytest.mark.asyncio
async def test_no_halal_matches_falls_back_to_all_configured():
    """Subtle behaviour: when the halal cache exists but NONE of the
    configured pairs matches it, the cycle falls back to every
    configured pair (minus paused). The intuition is "the screener
    must be misconfigured — don't silently halt trading"."""
    stage = _make_stage(
        configured_pairs=["UNKNOWNCOIN", "MYSTERYCOIN"],
        halal_pairs=["BTC", "ETH"],
    )
    result = await _run(stage)
    assert "UNKNOWNCOIN" in result
    assert "MYSTERYCOIN" in result


@pytest.mark.asyncio
async def test_no_halal_matches_fallback_still_filters_paused():
    """Even on the no-match fallback, paused pairs stay excluded."""
    stage = _make_stage(
        configured_pairs=["UNKNOWNCOIN", "MYSTERYCOIN"],
        halal_pairs=["BTC", "ETH"],
        paused_pairs={"UNKNOWNCOIN"},
    )
    result = await _run(stage)
    assert "UNKNOWNCOIN" not in result
    assert "MYSTERYCOIN" in result


# ── Duplicate elimination ──────────────────────────────────


@pytest.mark.asyncio
async def test_duplicate_configured_pairs_deduplicated():
    """If ``configured_pairs`` accidentally repeats a symbol, the output
    is still unique. Pin so a typo doesn't cause double-buys."""
    stage = _make_stage(
        configured_pairs=["BTCUSDT", "BTCUSDT", "ETHUSDT", "BTCUSDT"],
        halal_pairs=["BTC", "ETH"],
    )
    result = await _run(stage)
    assert result.count("BTCUSDT") == 1
    assert result.count("ETHUSDT") == 1


@pytest.mark.asyncio
async def test_dedup_preserves_first_occurrence_order():
    """Dedup is order-preserving — the first occurrence wins."""
    stage = _make_stage(
        configured_pairs=["ETHUSDT", "BTCUSDT", "ETHUSDT"],
        halal_pairs=["BTC", "ETH"],
    )
    assert await _run(stage) == ["ETHUSDT", "BTCUSDT"]


# ── Empty / edge cases ────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_configured_pairs_returns_empty():
    stage = _make_stage(configured_pairs=[], halal_pairs=["BTC"])
    assert await _run(stage) == []


@pytest.mark.asyncio
async def test_all_pairs_paused_returns_empty():
    stage = _make_stage(
        configured_pairs=["BTCUSDT", "ETHUSDT"],
        halal_pairs=["BTC", "ETH"],
        paused_pairs={"BTCUSDT", "ETHUSDT"},
    )
    assert await _run(stage) == []


def test_stage_has_stable_name():
    """The stage name appears in instrumentation events; lock it."""
    stage = _make_stage(configured_pairs=[], halal_pairs=[])
    assert stage.name == "get_tradeable_pairs"
