"""SSE /api/sse and WebSocket /ws/prices."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import Depends, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx

logger = logging.getLogger(__name__)


def register(app: FastAPI) -> None:
    @app.get("/api/sse")
    async def sse(ctx: DashboardContext = Depends(get_ctx)) -> StreamingResponse:
        async def event_stream():
            while True:
                trades = await ctx.repo.get_recent_crypto_trades(5)
                data = json.dumps({"trades": trades}, default=str)
                yield f"data: {data}\n\n"
                await asyncio.sleep(10)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.websocket("/ws/prices")
    async def ws_prices(
        websocket: WebSocket,
        symbols: list[str] = Query(default=[]),
    ) -> None:
        # WebSocket routes don't get FastAPI's Depends resolution the
        # same way as HTTP routes; pull the context off the ASGI app.
        ctx: DashboardContext | None = getattr(websocket.app.state, "ctx", None)
        await websocket.accept()
        try:
            while True:
                prices: dict[str, float] = {}
                ws_mgr = ctx.runtime.ws_manager if ctx else None
                for sym in symbols:
                    price = None
                    if ws_mgr and hasattr(ws_mgr, "get_latest_price"):
                        price = ws_mgr.get_latest_price(sym)
                    if price is not None:
                        prices[sym] = price
                if prices:
                    await websocket.send_json(prices)
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("WebSocket connection closed")
