"""FastAPI surface — exercised in-process via ASGITransport (shared loop)."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from halabot.api.app import create_api
from halabot.belief.schema import BeliefState, ComplianceVerdict, Direction, Regime
from halabot.belief.store import PgBeliefStore

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


async def _client(engine):
    app = create_api(engine)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


@pytest.mark.asyncio
async def test_health_and_beliefs_endpoints(halabot_engine):
    await PgBeliefStore(halabot_engine).put(
        BeliefState(
            asset="NVDA", regime=Regime.TRENDING_UP, direction=Direction.LONG_BIAS,
            conviction=0.8, conviction_raw=0.8,
            halal=ComplianceVerdict("NVDA", "halal", screened_at=T0),
        )
    )
    async with await _client(halabot_engine) as c:
        h = await c.get("/health")
        assert h.status_code == 200 and "events" in h.json()

        beliefs = await c.get("/beliefs")
        assert beliefs.status_code == 200
        assert beliefs.json()[0]["asset"] == "NVDA"

        one = await c.get("/beliefs/nvda")  # case-insensitive
        assert one.status_code == 200 and one.json()["conviction"] == 0.8

        missing = await c.get("/beliefs/TSLA")
        assert missing.status_code == 404


@pytest.mark.asyncio
async def test_controls_halt_endpoint(halabot_engine):
    async with await _client(halabot_engine) as c:
        assert (await c.get("/controls/halt")).json()["halted"] is False
        r = await c.post("/controls/halt", json={"halted": True, "reason": "test halt"})
        assert r.status_code == 200 and r.json()["halted"] is True
        assert (await c.get("/controls/halt")).json()["reason"] == "test halt"


@pytest.mark.asyncio
async def test_decision_endpoint_validates_correlation_id(halabot_engine):
    async with await _client(halabot_engine) as c:
        bad = await c.get("/decisions/not-a-uuid")
        assert bad.status_code == 400
        missing = await c.get("/decisions/00000000-0000-0000-0000-000000000000")
        assert missing.status_code == 404
