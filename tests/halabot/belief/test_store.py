"""InMemoryBeliefStore — versioning + copy isolation (INV-5)."""

from __future__ import annotations

import pytest

from halabot.belief.schema import BeliefState, Regime
from halabot.belief.store import InMemoryBeliefStore


@pytest.mark.asyncio
async def test_get_missing_returns_none():
    store = InMemoryBeliefStore()
    assert await store.get("NVDA") is None


@pytest.mark.asyncio
async def test_put_increments_version():
    store = InMemoryBeliefStore()
    b = BeliefState.neutral("NVDA")
    assert await store.put(b) == 1
    assert await store.put(b) == 2
    assert await store.put(b) == 3


@pytest.mark.asyncio
async def test_get_returns_latest_version():
    store = InMemoryBeliefStore()
    b = BeliefState.neutral("NVDA")
    b.regime = Regime.TRENDING_UP
    await store.put(b)
    b.regime = Regime.VOLATILE
    await store.put(b)
    latest = await store.get("NVDA")
    assert latest is not None and latest.regime == Regime.VOLATILE
    assert latest.version == 2


@pytest.mark.asyncio
async def test_old_version_is_retained():
    store = InMemoryBeliefStore()
    b = BeliefState.neutral("NVDA")
    b.regime = Regime.TRENDING_UP
    await store.put(b)
    b.regime = Regime.TRENDING_DOWN
    await store.put(b)
    v1 = await store.get_version("NVDA", 1)
    assert v1 is not None and v1.regime == Regime.TRENDING_UP  # history preserved


@pytest.mark.asyncio
async def test_put_isolates_from_caller_mutation():
    """A later in-place mutation of the caller's instance must not corrupt the
    persisted version (the store deep-copies on put)."""
    store = InMemoryBeliefStore()
    b = BeliefState.neutral("NVDA")
    b.conviction = 0.5
    await store.put(b)
    b.conviction = 0.9  # mutate AFTER persisting
    stored = await store.get("NVDA")
    assert stored is not None and stored.conviction == 0.5  # unaffected


@pytest.mark.asyncio
async def test_all_active_returns_latest_per_asset():
    store = InMemoryBeliefStore()
    await store.put(BeliefState.neutral("NVDA"))
    await store.put(BeliefState.neutral("MSFT"))
    actives = await store.all_active()
    assert {b.asset for b in actives} == {"NVDA", "MSFT"}
