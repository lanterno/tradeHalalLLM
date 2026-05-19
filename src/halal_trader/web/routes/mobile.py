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

    risk = ctx.runtime.risk_state or {}
    drawdown = risk.get("drawdown_pct") if isinstance(risk, dict) else None
    # The risk_state push carries which bot wrote it ("crypto" / "stocks" /
    # absent → unknown) so a glance at the mobile summary tells the
    # operator which cycle the drawdown belongs to.
    risk_market = risk.get("market") if isinstance(risk, dict) else None

    # Today's realized P&L — try both markets and report the most
    # recent non-zero row (or the most recent row, period). Stocks-only
    # operators were previously stuck on $0 because this only read the
    # crypto ledger.
    pnl_today_usd: float | None = None
    try:
        most_recent: dict[str, Any] | None = None
        try:
            stocks_pnl = await ctx.repo.get_pnl_history(limit=1)
            if stocks_pnl:
                most_recent = stocks_pnl[0]
        except Exception as exc:  # noqa: BLE001
            logger.debug("stocks pnl read failed: %s", exc)
        try:
            crypto_pnl = await ctx.repo.get_crypto_pnl_history(limit=1)
            if crypto_pnl:
                # Pick whichever row is more recent by date string
                # (both ledgers use YYYY-MM-DD which sorts lexically).
                if most_recent is None or str(crypto_pnl[0].get("date") or "") > str(
                    most_recent.get("date") or ""
                ):
                    most_recent = crypto_pnl[0]
        except Exception as exc:  # noqa: BLE001
            logger.debug("crypto pnl read failed: %s", exc)
        if most_recent is not None:
            pnl_today_usd = float(most_recent.get("realized_pnl") or 0.0)
    except Exception as e:
        logger.debug("pnl read failed: %s", e)

    return {
        "ts": time.time(),
        "halt": halt_payload,
        "bot_running": ctx.runtime.bot_running,
        "last_cycle": ctx.runtime.last_cycle,
        "drawdown_pct": drawdown,
        "drawdown_market": risk_market,
        "open_positions_by_asset": dict(ctx.runtime.open_positions_by_asset),
        "pnl_today_usd": pnl_today_usd,
        "llm_cost_today_usd": ctx.runtime.llm_cost_today_usd,
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
