"""Tests for the `stage()` async context manager in :mod:`core.cycle_pipeline`.

`run_stages` (which wraps each stage in this context) is covered by
the e2e pipeline tests; `CycleState` defaults are covered by
`test_cycle_state.py`. This file pins the stage context's own
contract: bus events fire, exceptions are swallowed by default,
elapsed_ms is captured on the outcome, and `extra` mutation flows
into the end event.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.cycle_pipeline import StageOutcome, stage

# ── Bus events ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_publishes_start_then_end_with_no_bus_no_op():
    """`bus=None` → context still works, just no events fire."""
    async with stage(None, "x"):
        pass
    # No assertions — must not raise.


@pytest.mark.asyncio
async def test_publishes_start_then_end_in_order():
    bus = MagicMock()
    bus.publish = AsyncMock()
    async with stage(bus, "build_regime_text"):
        pass
    topics = [c.args[0] for c in bus.publish.await_args_list]
    assert topics == ["cycle.stage.start", "cycle.stage.end"]


@pytest.mark.asyncio
async def test_start_event_carries_stage_name_and_attrs():
    bus = MagicMock()
    bus.publish = AsyncMock()
    async with stage(bus, "build_x", pair_count=4):
        pass
    start_payload = bus.publish.await_args_list[0].args[1]
    assert start_payload["name"] == "build_x"
    assert start_payload["pair_count"] == 4


@pytest.mark.asyncio
async def test_end_event_carries_elapsed_ms():
    bus = MagicMock()
    bus.publish = AsyncMock()
    async with stage(bus, "build_x"):
        pass
    end_payload = bus.publish.await_args_list[1].args[1]
    assert "elapsed_ms" in end_payload
    assert end_payload["elapsed_ms"] >= 0


@pytest.mark.asyncio
async def test_end_event_includes_outcome_extra_mutations():
    """Stage code can stash arbitrary keys on `outcome.extra`; they
    get merged into the cycle.stage.end payload."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    async with stage(bus, "build_x") as o:
        o.extra["n_klines"] = 42
    end_payload = bus.publish.await_args_list[1].args[1]
    assert end_payload["n_klines"] == 42


# ── Exception swallowing ────────────────────────────────────


@pytest.mark.asyncio
async def test_swallow_default_does_not_re_raise():
    """A stage body that raises is logged but doesn't propagate —
    matches "best-effort augmentation" semantics for most stages."""
    async with stage(None, "buggy"):
        raise RuntimeError("oops")
    # Got here → not re-raised.


@pytest.mark.asyncio
async def test_swallow_default_records_error_on_outcome():
    bus = MagicMock()
    bus.publish = AsyncMock()
    async with stage(bus, "buggy") as o:
        raise RuntimeError("oops")
    assert o.error is not None
    assert "oops" in o.error
    end_payload = bus.publish.await_args_list[1].args[1]
    assert "oops" in end_payload["error"]


@pytest.mark.asyncio
async def test_swallow_false_re_raises():
    """For the LLM call + executor — stages that *must* succeed —
    set `swallow=False` so a real failure halts the cycle."""
    with pytest.raises(RuntimeError, match="oops"):
        async with stage(None, "critical", swallow=False):
            raise RuntimeError("oops")


# ── Bus failure ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bus_publish_failure_swallowed():
    """A misbehaving bus publish must not crash the stage — events
    are best-effort observability."""
    bus = MagicMock()
    bus.publish = AsyncMock(side_effect=RuntimeError("bus down"))
    # Both start (publish raises) and end (publish raises again) get swallowed.
    async with stage(bus, "x"):
        pass


# ── StageOutcome shape ──────────────────────────────────────


def test_stage_outcome_default_fields():
    o = StageOutcome(name="x", elapsed_ms=12.3)
    assert o.name == "x"
    assert o.elapsed_ms == 12.3
    assert o.error is None
    assert o.skipped is False
    assert o.extra == {}
