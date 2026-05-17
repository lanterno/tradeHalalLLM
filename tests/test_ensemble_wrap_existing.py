"""Test for the public `wrap_existing` helper in ``core.llm.ensemble``.

This helper is a tiny `async def` that returns its arg unchanged — it
exists so the primary plan can feed into `run_ensemble` as one of the
variants without re-calling the LLM. It used to be duplicated as a
private `_wrap_existing` in both `crypto/strategy.py` and
`trading/strategy.py`; now it has a single home.

This test pins:

* Returns the same object by identity (no copy, no mutation).
* Is genuinely async (test with `await`).
* Works with arbitrary inputs (None, dataclass, dict, primitive).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from halal_trader.core.llm.ensemble import wrap_existing


@pytest.mark.asyncio
async def test_wrap_existing_returns_input_by_identity():
    obj = object()
    out = await wrap_existing(obj)
    assert out is obj


@pytest.mark.asyncio
async def test_wrap_existing_handles_none():
    """No type-narrowing — None passes through verbatim."""
    assert await wrap_existing(None) is None


@pytest.mark.asyncio
async def test_wrap_existing_handles_dataclass():
    @dataclass
    class _Plan:
        market_outlook: str = "ok"

    plan = _Plan()
    out = await wrap_existing(plan)
    assert out is plan
    assert out.market_outlook == "ok"


@pytest.mark.asyncio
async def test_wrap_existing_handles_primitive():
    assert await wrap_existing(42) == 42
    assert await wrap_existing("hello") == "hello"


@pytest.mark.asyncio
async def test_wrap_existing_does_not_mutate_input():
    """Defensive: a refactor that converts to `model_copy()` or some
    other transform would break the contract — pin the no-op."""
    plan = {"buys": [], "sells": []}
    out = await wrap_existing(plan)
    out["mutated"] = True
    # The same dict — caller's reference was returned. (This is by
    # design; the helper is a passthrough.)
    assert plan["mutated"] is True
