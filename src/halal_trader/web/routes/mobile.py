"""Mobile-friendly summary endpoint + state-push WebSocket.

The dashboard's existing pages are desktop-tab heavy. The phone view
needs a single rolled-up endpoint so the operator can refresh once and
see the four numbers that matter (P&L, halt status, open positions,
last cycle), plus a halt button. This module backs that view.

It also adds ``/ws/state`` — a WebSocket that pushes the same summary
on each price tick so the phone tab stays live without polling. The
push payload reuses the summary endpoint's shape so the React side
renders it through one component.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from halal_trader.core.halt import get_status
from halal_trader.db.repository import Repository

logger = logging.getLogger(__name__)


_PUSH_INTERVAL_SECONDS = 5.0


async def _build_summary(app_state: dict[str, Any]) -> dict[str, Any]:
    """Roll up the dashboard's frequently-needed state into one payload.

    Tolerant of partially-initialised app_state — anything missing
    falls back to a sane placeholder so the phone view never renders
    "undefined". The frontend can show greyed-out tiles for missing
    fields rather than blowing up.
    """
    engine = app_state.get("engine")
    repo: Repository | None = app_state.get("repo")

    halt_payload: dict[str, Any] = {"enabled": False, "reason": None}
    if engine is not None:
        try:
            halt = await get_status(engine)
            halt_payload = {
                "enabled": halt.enabled,
                "reason": halt.reason,
                "set_by": halt.set_by,
                "set_at": halt.set_at.isoformat() if halt.set_at else None,
            }
        except Exception as e:
            logger.debug("halt status read failed: %s", e)

    last_cycle = app_state.get("last_cycle")
    bot_running = bool(app_state.get("bot_running"))

    risk = app_state.get("risk_state") or {}
    drawdown = risk.get("drawdown_pct") if isinstance(risk, dict) else None

    open_positions_by_asset = app_state.get("open_positions_by_asset") or {}

    pnl_today_usd: float | None = None
    if repo is not None:
        try:
            recent = await repo.get_crypto_pnl_history(limit=1)
            if recent:
                pnl_today_usd = float(recent[0].get("realized_pnl") or 0.0)
        except Exception as e:
            logger.debug("pnl read failed: %s", e)

    return {
        "ts": time.time(),
        "halt": halt_payload,
        "bot_running": bot_running,
        "last_cycle": last_cycle,
        "drawdown_pct": drawdown,
        "open_positions_by_asset": dict(open_positions_by_asset),
        "pnl_today_usd": pnl_today_usd,
        "llm_cost_today_usd": app_state.get("llm_cost_today_usd"),
    }


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    @app.get("/api/mobile/summary")
    async def mobile_summary() -> JSONResponse:
        """Roll-up endpoint for the phone view (4-tile dashboard)."""
        return JSONResponse(await _build_summary(app_state))

    @app.websocket("/ws/state")
    async def ws_state(websocket: WebSocket) -> None:
        """Push the summary on every interval so the phone view stays live."""
        await websocket.accept()
        try:
            while True:
                payload = await _build_summary(app_state)
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
