"""Mobile-friendly summary endpoint + state-push WebSocket."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from halal_trader.core.context import DashboardContext
from halal_trader.core.halt import get_status
from halal_trader.web.dependencies import get_ctx

logger = logging.getLogger(__name__)


_PUSH_INTERVAL_SECONDS = 5.0


async def _build_summary(ctx: DashboardContext) -> dict[str, Any]:
    """Roll up the dashboard's frequently-needed state into one payload."""
    halt_payload: dict[str, Any] = {"enabled": False, "reason": None}
    try:
        halt = await get_status(ctx.engine)
        halt_payload = {
            "enabled": halt.enabled,
            "reason": halt.reason,
            "set_by": halt.set_by,
            "set_at": halt.set_at.isoformat() if halt.set_at else None,
        }
    except Exception as e:
        logger.debug("halt status read failed: %s", e)

    # Everything below comes from the database: the web is its own process
    # and in-process runtime state never reaches it.
    from sqlalchemy import text

    from halal_trader.core.heartbeat import STOCK_CYCLE, cycle_risk
    from halal_trader.web.routes.system import _bot_liveness

    risk: dict[str, Any] | None = None
    try:
        risk, _ = await cycle_risk(ctx.engine)
    except Exception as e:  # noqa: BLE001
        logger.debug("risk read failed: %s", e)
    drawdown = risk.get("drawdown_pct") if risk else None
    risk_market = "stocks" if risk else None
    alive, components = await _bot_liveness(ctx)
    cycle_beat = (components or {}).get(STOCK_CYCLE)
    llm_cost_today: float | None = None
    try:
        async with ctx.engine.connect() as conn:
            llm_cost_today = float(
                (
                    await conn.execute(
                        text(
                            "SELECT coalesce(sum(spent_usd), 0) FROM llm_spend "
                            "WHERE day = (now() AT TIME ZONE 'UTC')::date"
                        )
                    )
                ).scalar()
                or 0.0
            )
    except Exception as e:  # noqa: BLE001
        logger.debug("llm spend read failed: %s", e)

    # Today's realized P&L — the most recent stocks ``daily_pnl`` row.
    pnl_today_usd: float | None = None
    try:
        most_recent: dict[str, Any] | None = None
        try:
            stocks_pnl = await ctx.repo.get_pnl_history(limit=1)
            if stocks_pnl:
                most_recent = stocks_pnl[0]
        except Exception as exc:  # noqa: BLE001
            logger.debug("stocks pnl read failed: %s", exc)
        if most_recent is not None:
            pnl_today_usd = float(most_recent.get("realized_pnl") or 0.0)
    except Exception as e:
        logger.debug("pnl read failed: %s", e)

    return {
        "ts": time.time(),
        "halt": halt_payload,
        "bot_running": alive,
        "last_cycle": cycle_beat["beat_at"] if cycle_beat else None,
        "drawdown_pct": drawdown,
        "drawdown_market": risk_market,
        "open_positions_by_asset": dict(ctx.runtime.open_positions_by_asset),
        "pnl_today_usd": pnl_today_usd,
        "llm_cost_today_usd": llm_cost_today,
    }


def register(app: FastAPI) -> None:
    @app.get("/api/mobile/summary")
    async def mobile_summary(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        return JSONResponse(await _build_summary(ctx))

    @app.websocket("/ws/state")
    async def ws_state(websocket: WebSocket) -> None:
        ctx: DashboardContext | None = getattr(websocket.app.state, "ctx", None)
        await websocket.accept()
        if ctx is None:
            await websocket.close(code=1011)
            return
        try:
            while True:
                payload = await _build_summary(ctx)
                try:
                    await websocket.send_json(payload)
                except Exception as e:
                    logger.debug("ws state send failed: %s", e)
                    return
                await asyncio.sleep(_PUSH_INTERVAL_SECONDS)
        except WebSocketDisconnect:
            return
        except Exception as e:  # noqa: BLE001
            logger.debug("ws state error: %s", e)
            return
