"""Dashboard surface for prompt-evolution candidates (Wave F)."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx
from halal_trader.web.middleware.confirm import require_confirmation


def register(app: FastAPI) -> None:
    @app.get("/api/prompts/candidates")
    async def list_candidates(
        name: str | None = None,
        limit: int = 50,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halal_trader.core.llm.prompt_evo_runner import list_recent_genomes

        rows = await list_recent_genomes(engine=ctx.engine, name=name, limit=limit)
        return JSONResponse(serialize(rows))

    @app.post(
        "/api/prompts/{genome_id}/promote",
        dependencies=[Depends(require_confirmation)],
    )
    async def promote(
        genome_id: int,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halal_trader.core.llm.prompt_evo_runner import promote_genome

        ok = await promote_genome(engine=ctx.engine, genome_id=genome_id)
        if not ok:
            raise HTTPException(404, f"genome {genome_id} not found")
        return JSONResponse({"id": genome_id, "promoted": True})
