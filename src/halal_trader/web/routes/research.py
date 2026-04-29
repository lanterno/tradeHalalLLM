"""Research surface — replay, prompt-version diff, halal audit export."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response

from halal_trader.core.context import DashboardContext
from halal_trader.db.models import LlmDecision
from halal_trader.halal.audit import export_for_symbol, export_receipt
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/research/replay/{decision_id}")
    async def replay_decision(
        decision_id: int, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        from sqlmodel.ext.asyncio.session import AsyncSession

        async with AsyncSession(ctx.engine) as session:
            row = await session.get(LlmDecision, decision_id)
            if row is None:
                raise HTTPException(404, f"decision {decision_id} not found")
            payload = row.model_dump()
            return JSONResponse(serialize(payload))

    @app.get("/api/research/prompt-versions")
    async def list_prompt_versions(
        limit: int = 200, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        decisions = await ctx.repo.get_recent_decisions(limit=limit)
        by_version: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for d in decisions:
            v = d.get("prompt_version") or "unversioned"
            by_version[v].append(d)

        out = []
        for version, rows in by_version.items():
            costs = [r.get("cost_usd") or 0.0 for r in rows]
            in_toks = [r.get("input_tokens") or 0 for r in rows]
            cache_toks = [r.get("cache_read_tokens") or 0 for r in rows]
            in_total = sum(in_toks)
            cache_total = sum(cache_toks)
            cache_ratio = (cache_total / in_total) if in_total > 0 else 0.0
            out.append(
                {
                    "version": version,
                    "count": len(rows),
                    "total_cost_usd": round(sum(costs), 4),
                    "avg_cost_usd": round(sum(costs) / len(rows), 4) if rows else 0.0,
                    "avg_input_tokens": int(sum(in_toks) / len(rows)) if rows else 0,
                    "cache_read_ratio": round(cache_ratio, 3),
                }
            )
        out.sort(key=lambda r: r["count"], reverse=True)
        return JSONResponse(out)

    @app.get("/api/research/halal-audit/{asset_class}/{trade_id}")
    async def audit_one_trade(
        asset_class: str,
        trade_id: int,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> Response:
        if asset_class not in ("stock", "crypto"):
            raise HTTPException(400, "asset_class must be 'stock' or 'crypto'")
        receipt = await export_receipt(ctx.engine, trade_id=trade_id, asset_class=asset_class)
        if receipt is None:
            raise HTTPException(404, f"no {asset_class} trade with id={trade_id}")
        return Response(content=receipt.to_json(), media_type="application/json")

    @app.get("/api/research/halal-audit/{asset_class}/symbol/{symbol}")
    async def audit_for_symbol(
        asset_class: str,
        symbol: str,
        limit: int = 50,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        if asset_class not in ("stock", "crypto"):
            raise HTTPException(400, "asset_class must be 'stock' or 'crypto'")
        receipts = await export_for_symbol(
            ctx.engine, symbol=symbol, asset_class=asset_class, limit=limit
        )
        return JSONResponse([r.payload for r in receipts])
