"""Halal & compliance admin endpoints.

Surfaces the purification ledger, on-demand cache refresh, and current
sector allocation so the operator can drive compliance workflows from
the dashboard:

* GET / POST / DELETE on the purification ledger.
* POST /api/admin/halal/refresh to force a halal-cache rebuild.
* GET sector-allocation breakdown (current exposure per sector vs cap).
"""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from halal_trader.core.context import DashboardContext
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx
from halal_trader.web.middleware.confirm import require_confirmation

logger = logging.getLogger(__name__)


class RecordPurificationRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12)
    dividend_usd: float = Field(ge=0)
    haram_pct: float = Field(ge=0, le=1)
    notes: str | None = Field(default=None, max_length=500)


def register(app: FastAPI) -> None:
    @app.get("/api/admin/purification")
    async def list_purification(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        outstanding = await ctx.repo.get_outstanding_purification()
        totals = await ctx.repo.get_purification_totals()
        return JSONResponse(
            {
                "outstanding": serialize(outstanding),
                "totals": totals,
            }
        )

    @app.post(
        "/api/admin/purification",
        dependencies=[Depends(require_confirmation)],
    )
    async def record_purification(
        req: RecordPurificationRequest,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halal_trader.halal.purification import compute_purification

        entry = compute_purification(
            symbol=req.symbol,
            dividend_usd=req.dividend_usd,
            haram_revenue_pct=req.haram_pct,
            notes=req.notes or "",
        )
        eid = await ctx.repo.record_purification(
            symbol=entry.symbol,
            dividend_usd=float(entry.dividend_usd),
            haram_pct=float(entry.haram_pct),
            purification_usd=float(entry.purification_usd),
            notes=entry.notes,
        )
        return JSONResponse(
            {
                "id": eid,
                "symbol": entry.symbol,
                "purification_usd": float(entry.purification_usd),
            }
        )

    @app.post(
        "/api/admin/purification/{entry_id}/mark_paid",
        dependencies=[Depends(require_confirmation)],
    )
    async def mark_paid(entry_id: int, ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        ok = await ctx.repo.mark_purification_paid(entry_id)
        if not ok:
            raise HTTPException(404, f"purification entry {entry_id} not found")
        return JSONResponse({"id": entry_id, "paid": True})

    @app.post(
        "/api/admin/halal/refresh",
        dependencies=[Depends(require_confirmation)],
    )
    async def force_halal_refresh(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halal_trader.halal.cache import HalalScreener

        screener = HalalScreener(ctx.repo, zoya=None)
        await screener.ensure_cache(force=True)
        symbols = await screener.get_halal_symbols()
        return JSONResponse(
            {
                "refreshed": True,
                "halal_symbol_count": len(symbols),
            }
        )

    @app.get("/api/admin/halal/sector-allocation")
    async def sector_allocation(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halal_trader.halal.sector_limits import (
            UNKNOWN_SECTOR,
            compute_allocation,
        )

        positions = ctx.runtime.stock_positions or []
        equity = ctx.runtime.stock_equity or 0.0

        positions_value = {
            getattr(p, "symbol", "?"): float(getattr(p, "qty", 0))
            * float(getattr(p, "current_price", None) or getattr(p, "avg_entry_price", 0))
            for p in positions
        }
        alloc = compute_allocation(positions_value, total_equity=equity)
        # Render as a list of {sector, value, pct} for easy table rendering.
        rows = []
        for sector, value in sorted(alloc.by_sector.items()):
            rows.append(
                {
                    "sector": sector,
                    "value_usd": value,
                    "pct": alloc.pct(sector),
                }
            )
        return JSONResponse(
            {
                "total_equity_usd": equity,
                "unknown_sector_label": UNKNOWN_SECTOR,
                "allocations": rows,
            }
        )
