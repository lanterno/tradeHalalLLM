"""Tests for the cycle stage instrumentation primitive."""

from __future__ import annotations

import asyncio

import pytest

from halal_trader.core.cycle_pipeline import stage
from halal_trader.core.event_bus import EventBus


async def test_stage_publishes_start_and_end() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def consumer() -> None:
        async for event in bus.subscribe("cycle.stage.*"):
            seen.append(event.topic)
            if len(seen) == 2:
                break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    async with stage(bus, "compute_indicators"):
        pass
    await asyncio.wait_for(task, timeout=1.0)
    assert seen == ["cycle.stage.start", "cycle.stage.end"]


async def test_stage_records_elapsed_ms() -> None:
    bus = EventBus()
    out = []

    async def consumer() -> None:
        async for event in bus.subscribe("cycle.stage.end"):
            out.append(event.payload)
            break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    async with stage(bus, "slow") as outcome:
        await asyncio.sleep(0.005)
        outcome.extra["foo"] = "bar"
    await asyncio.wait_for(task, timeout=1.0)
    assert out[0]["name"] == "slow"
    assert out[0]["elapsed_ms"] >= 5.0
    assert out[0]["error"] is None
    assert out[0]["foo"] == "bar"


async def test_stage_swallows_exception_by_default() -> None:
    bus = EventBus()
    end_payload: list[dict] = []

    async def consumer() -> None:
        async for event in bus.subscribe("cycle.stage.end"):
            end_payload.append(event.payload)
            break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    async with stage(bus, "boom"):
        raise RuntimeError("intentional")
    await asyncio.wait_for(task, timeout=1.0)
    assert "RuntimeError" in end_payload[0]["error"]


async def test_stage_reraises_when_swallow_false() -> None:
    bus = EventBus()
    with pytest.raises(RuntimeError, match="intentional"):
        async with stage(bus, "fatal", swallow=False):
            raise RuntimeError("intentional")
