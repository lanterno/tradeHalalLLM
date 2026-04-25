"""GET /api/config."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from halal_trader.config import get_settings


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    del app_state  # unused — settings come from get_settings()

    @app.get("/api/config")
    async def api_config() -> JSONResponse:
        settings = get_settings()
        return JSONResponse(
            {
                "llm_provider": settings.llm_provider.value,
                "llm_model": settings.llm_model,
                "crypto_pairs": settings.crypto_pairs,
                "crypto_trading_interval_seconds": settings.crypto_trading_interval_seconds,
                "crypto_max_position_pct": settings.crypto_max_position_pct,
                "crypto_daily_loss_limit": settings.crypto_daily_loss_limit,
                "crypto_daily_return_target": settings.crypto_daily_return_target,
                "db_path": str(settings.db_path),
            }
        )
