"""SSE /api/sse, WebSocket /ws/prices, and live-cycle stream /ws/cycle."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import Depends, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx

logger = logging.getLogger(__name__)


def register(app: FastAPI) -> None:
    @app.get("/api/sse")
    async def sse(
        ctx: DashboardContext = Depends(get_ctx),
        market: str = "crypto",
    ) -> StreamingResponse:
        """SSE feed of the 5 most-recent trades, refreshing every 10s.

        ``market="crypto"`` (default, back-compat) streams from the
        crypto ledger. ``market="stocks"`` streams from the stocks
        ``trades`` table. The /ws/cycle WebSocket (below) is the
        modern event-bus path; this endpoint is the legacy polling
        stream kept for older frontend integrations.
        """
        market_lower = market.lower()
        is_stocks = market_lower in ("stock", "stocks")

        async def event_stream() -> AsyncIterator[str]:
            while True:
                if is_stocks:
                    trades = await ctx.repo.get_recent_trades(5)
                else:
                    trades = await ctx.repo.get_recent_crypto_trades(5)
                data = json.dumps({"trades": trades, "market": market_lower}, default=str)
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

    @app.websocket("/ws/cycle")
    async def ws_cycle(websocket: WebSocket, topic: str = "*") -> None:
        """Wave I — stream structured cycle events to the dashboard.

        Subscribers see ``cycle.start``, ``cycle.stage.start``,
        ``cycle.stage.end``, ``cycle.complete``, ``cycle.failed``,
        ``llm.call.complete``, ``executor.fill``, etc — anything the
        bot publishes on the EventBus matching ``topic`` (default
        ``*`` — everything).
        """
        ctx: DashboardContext | None = getattr(websocket.app.state, "ctx", None)
        await websocket.accept()
        if ctx is None:
            await websocket.send_json(
                {
                    "topic": "_error",
                    "ts": datetime.utcnow().isoformat(),
                    "payload": {"error": "no context — bot not running in this process"},
                }
            )
            await websocket.close(code=1011)
            return

        try:
            async for event in ctx.bus.subscribe(topic):
                payload = {
                    "topic": event.topic,
                    "ts": event.ts.isoformat(),
                    "payload": event.payload,
                }
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("ws_cycle closed: %s", exc)
