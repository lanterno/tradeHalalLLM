"""Research surface — replay, prompt-version diff, halal audit export.

These endpoints consume the audit primitives we shipped in earlier
phases (``LlmDecision`` cost columns, prompt-version registry, halal
audit FK) and surface them in a form the dashboard can render. Three
orthogonal questions an operator wants answered:

1. *Why did the LLM make this decision?* — replay shows the stored
   prompt summary, raw response, parsed action, and prompt version
   for any decision id.
2. *Did changing the prompt help?* — the prompt-diff endpoint groups
   recent decisions by prompt_version and reports per-version cost,
   token use, and decision counts.
3. *Was every trade halal-attested?* — the audit-export endpoint
   surfaces the per-trade compliance receipt JSON for download.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response

from halal_trader.db.models import LlmDecision
from halal_trader.db.repository import Repository
from halal_trader.halal.audit import export_for_symbol, export_receipt
from halal_trader.web._serializer import serialize


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    @app.get("/api/research/replay/{decision_id}")
    async def replay_decision(decision_id: int) -> JSONResponse:
        repo: Repository = app_state["repo"]
        from sqlmodel.ext.asyncio.session import AsyncSession

        async with AsyncSession(repo._engine) as session:
            row = await session.get(LlmDecision, decision_id)
            if row is None:
                raise HTTPException(404, f"decision {decision_id} not found")
            payload = row.model_dump()
            return JSONResponse(serialize(payload))

    @app.get("/api/research/prompt-versions")
    async def list_prompt_versions(limit: int = 200) -> JSONResponse:
        """Group recent decisions by prompt_version with per-version totals.

        Returns a list of ``{version, count, total_cost_usd, avg_input_tokens,
        avg_cache_read_ratio}`` so the dashboard can render an A/B-style
        comparison of two prompt versions side-by-side.
        """
        repo: Repository = app_state["repo"]
        decisions = await repo.get_recent_decisions(limit=limit)
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
    async def audit_one_trade(asset_class: str, trade_id: int) -> Response:
        if asset_class not in ("stock", "crypto"):
            raise HTTPException(400, "asset_class must be 'stock' or 'crypto'")
        engine = app_state["engine"]
        receipt = await export_receipt(engine, trade_id=trade_id, asset_class=asset_class)
        if receipt is None:
            raise HTTPException(404, f"no {asset_class} trade with id={trade_id}")
        return Response(content=receipt.to_json(), media_type="application/json")

    @app.get("/api/research/halal-audit/{asset_class}/symbol/{symbol}")
    async def audit_for_symbol(asset_class: str, symbol: str, limit: int = 50) -> JSONResponse:
        if asset_class not in ("stock", "crypto"):
            raise HTTPException(400, "asset_class must be 'stock' or 'crypto'")
        engine = app_state["engine"]
        receipts = await export_for_symbol(
            engine, symbol=symbol, asset_class=asset_class, limit=limit
        )
        return JSONResponse([r.payload for r in receipts])
