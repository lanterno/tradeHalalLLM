"""FastAPI surface (REARCHITECTURE L9) — a thin layer over ``queries``.

Routers: beliefs (the belief board), decisions (the replayable decision stream),
risk, system (health), controls (operator kill-switch). Read-only except the
controls POSTs. ``create_api(engine)`` builds the app; the ``halabot dashboard``
CLI command serves it. Heavy import (fastapi) is local to keep the rest of the
package importable without the dashboard extra installed.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.api import queries


class HaltRequest(BaseModel):
    # Module-level (not local) so FastAPI can resolve the stringified annotation
    # under `from __future__ import annotations` and bind it as the request body.
    halted: bool
    reason: str | None = None


def create_api(engine: AsyncEngine) -> Any:  # returns a FastAPI app
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="halabot — market understanding", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return await queries.system_health(engine)

    @app.get("/beliefs")
    async def beliefs() -> list[dict[str, Any]]:
        return await queries.list_beliefs(engine)

    @app.get("/beliefs/{asset}")
    async def belief(asset: str) -> dict[str, Any]:
        b = await queries.get_belief(engine, asset.upper())
        if b is None:
            raise HTTPException(status_code=404, detail=f"no belief for {asset}")
        return b

    @app.get("/beliefs/{asset}/conviction")
    async def conviction(asset: str, limit: int = 100) -> list[dict[str, Any]]:
        return await queries.conviction_history(engine, asset.upper(), limit=limit)

    @app.get("/decisions")
    async def decisions(limit: int = 50) -> list[dict[str, Any]]:
        return await queries.recent_decisions(engine, limit=limit)

    @app.get("/decisions/{correlation_id}")
    async def decision(correlation_id: str) -> list[dict[str, Any]]:
        try:
            cid = UUID(correlation_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid correlation_id") from exc
        chain = await queries.decision_chain(engine, cid)
        if not chain:
            raise HTTPException(status_code=404, detail="no events for that correlation_id")
        return chain

    @app.get("/risk")
    async def risk() -> dict[str, Any]:
        r = await queries.latest_risk(engine)
        return r or {"detail": "no risk state recorded yet"}

    @app.get("/controls/halt")
    async def halt_status() -> dict[str, Any]:
        return await queries.get_halt(engine)

    @app.post("/controls/halt")
    async def set_halt(req: HaltRequest) -> dict[str, Any]:
        return await queries.set_halt(engine, halted=req.halted, reason=req.reason)

    return app
