"""Per-trade operator intervention endpoints.

The crypto monitor + stock monitor watch SL/TP between cycles, but the
LLM occasionally misses one or sets it loose. The dashboard now lets
the operator amend SL/TP, manually close with a reason, and inspect a
trade's full audit drawer (decision id → halal screening → indicator
snapshot → fills) from one panel.

All write paths refuse non-buy and already-closed trades, refuse SL
above entry / TP below entry (long-only sanity), and write a
``web_action`` row via the audit middleware.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import CryptoTrade, Trade
from halal_trader.db.repository import Repository
from halal_trader.web.middleware.confirm import require_confirmation

logger = logging.getLogger(__name__)


class EditSLTPRequest(BaseModel):
    asset_class: str = Field(pattern="^(stock|crypto)$")
    stop_loss: float | None = Field(default=None, ge=0)
    target_price: float | None = Field(default=None, ge=0)


class ManualCloseRequest(BaseModel):
    asset_class: str = Field(pattern="^(stock|crypto)$")
    exit_price: float = Field(ge=0)
    reason: str = Field(default="operator_manual_close", min_length=1, max_length=100)


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    @app.patch(
        "/api/admin/trades/{trade_id}/sl_tp",
        dependencies=[Depends(require_confirmation)],
    )
    async def edit_sl_tp(trade_id: int, req: EditSLTPRequest) -> JSONResponse:
        """Update SL or TP on an open BUY trade."""
        if req.stop_loss is None and req.target_price is None:
            raise HTTPException(422, "must provide stop_loss or target_price")

        engine = app_state["engine"]
        model = CryptoTrade if req.asset_class == "crypto" else Trade

        async with AsyncSession(engine) as session:
            trade = await session.get(model, trade_id)
            if trade is None:
                raise HTTPException(404, f"trade {trade_id} not found")
            if trade.side != "buy":
                raise HTTPException(409, "can only edit SL/TP on a BUY trade")
            if trade.closed_at is not None:
                raise HTTPException(409, "trade is already closed")

            entry = trade.filled_price or trade.price or trade.entry_price or 0
            if req.stop_loss is not None and entry > 0 and req.stop_loss >= entry:
                raise HTTPException(422, f"stop_loss {req.stop_loss} must be below entry {entry}")
            if req.target_price is not None and entry > 0 and req.target_price <= entry:
                raise HTTPException(
                    422, f"target_price {req.target_price} must be above entry {entry}"
                )

            if req.stop_loss is not None:
                trade.stop_loss = req.stop_loss
            if req.target_price is not None:
                trade.target_price = req.target_price
            session.add(trade)
            await session.commit()

            logger.info(
                "Operator edited SL/TP on %s trade %d: SL=%s TP=%s",
                req.asset_class,
                trade_id,
                req.stop_loss,
                req.target_price,
            )

        return JSONResponse(
            {
                "trade_id": trade_id,
                "stop_loss": req.stop_loss,
                "target_price": req.target_price,
            }
        )

    @app.post(
        "/api/admin/trades/{trade_id}/close",
        dependencies=[Depends(require_confirmation)],
    )
    async def manual_close_trade(trade_id: int, req: ManualCloseRequest) -> JSONResponse:
        """Stamp a trade as closed at an operator-supplied exit price.

        Note: this DOES NOT submit a broker order. The expected workflow
        is to use ``/api/admin/positions/{symbol}/close`` (W1) for the
        broker exit, then this endpoint to record the realised exit
        price/reason on the specific trade row. The two are separate so
        a partial fill can be reconciled accurately.
        """
        repo: Repository = app_state["repo"]
        if req.asset_class == "crypto":
            await repo.close_crypto_trade(
                trade_id, exit_price=req.exit_price, exit_reason=req.reason
            )
        else:
            await repo.close_trade(trade_id, exit_price=req.exit_price, exit_reason=req.reason)
        return JSONResponse(
            {
                "trade_id": trade_id,
                "exit_price": req.exit_price,
                "reason": req.reason,
                "closed": True,
            }
        )

    @app.get("/api/trades/{asset_class}/{trade_id}/audit")
    async def trade_audit_drawer(asset_class: str, trade_id: int) -> JSONResponse:
        """Combined audit drawer: trade row + halal receipt + indicator snapshot."""
        if asset_class not in ("stock", "crypto"):
            raise HTTPException(400, "asset_class must be 'stock' or 'crypto'")

        engine = app_state["engine"]
        model = CryptoTrade if asset_class == "crypto" else Trade
        async with AsyncSession(engine) as session:
            trade = await session.get(model, trade_id)
            if trade is None:
                raise HTTPException(404, f"trade {trade_id} not found")

        from halal_trader.halal.audit import export_receipt

        receipt = await export_receipt(engine, trade_id=trade_id, asset_class=asset_class)

        # Indicator snapshot — trades that came in via the cycle have one.
        snapshot: dict | None = None
        async with AsyncSession(engine) as session:
            from sqlmodel import select

            from halal_trader.db.models import IndicatorSnapshot

            result = await session.exec(
                select(IndicatorSnapshot).where(IndicatorSnapshot.trade_id == trade_id)
            )
            row = result.first()
            if row is not None:
                snapshot = row.model_dump()

        from halal_trader.web._serializer import serialize

        return JSONResponse(
            {
                "trade": serialize(trade.model_dump()),
                "receipt": receipt.payload if receipt else None,
                "indicator_snapshot": serialize(snapshot) if snapshot else None,
            }
        )
