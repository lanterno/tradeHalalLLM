"""Tests for the /ws/cycle live event stream."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from halal_trader.core.context import DashboardContext, RuntimeView
from halal_trader.core.event_bus import EventBus
from halal_trader.core.insights_hub import InsightsHub
from halal_trader.web.routes.streaming import register


def _client_with_bus(bus: EventBus) -> TestClient:
    app = FastAPI()
    ctx = DashboardContext(
        engine=None,  # type: ignore[arg-type]
        repo=None,  # type: ignore[arg-type]
        hub=InsightsHub(),
        analytics=None,  # type: ignore[arg-type]
        settings=None,  # type: ignore[arg-type]
        bus=bus,
        runtime=RuntimeView(),
    )
    app.state.ctx = ctx
    register(app)
    return TestClient(app)


def test_ws_cycle_streams_published_events() -> None:
    bus = EventBus()
    client = _client_with_bus(bus)

    with client.websocket_connect("/ws/cycle") as ws:

        async def push() -> None:
            await asyncio.sleep(0.01)
            await bus.publish("cycle.start", {"cycle_id": "cycle-aaa"})
            await bus.publish(
                "cycle.stage.end",
                {"name": "fetch_klines", "elapsed_ms": 12.5},
            )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(push())
        finally:
            loop.close()

        # Allow the server side to drain — TestClient runs the app in
        # a worker thread, so a brief sleep is enough.
        first = ws.receive_json(mode="text")
        assert first["topic"] == "cycle.start"
        assert first["payload"] == {"cycle_id": "cycle-aaa"}

        second = ws.receive_json(mode="text")
        assert second["topic"] == "cycle.stage.end"
        assert second["payload"]["name"] == "fetch_klines"


def test_ws_cycle_filters_by_topic() -> None:
    bus = EventBus()
    client = _client_with_bus(bus)

    with client.websocket_connect("/ws/cycle?topic=llm.*") as ws:

        async def push() -> None:
            await asyncio.sleep(0.01)
            await bus.publish("cycle.start", {})  # filtered out
            await bus.publish("llm.call.complete", {"tokens": 200})

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(push())
        finally:
            loop.close()

        msg = ws.receive_json(mode="text")
        assert msg["topic"] == "llm.call.complete"
        assert msg["payload"]["tokens"] == 200
