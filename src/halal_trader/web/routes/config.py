"""GET /api/config (current values) + /api/config/schema (envvar metadata)."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

from halal_trader.config import Settings
from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx


def register(app: FastAPI) -> None:
    @app.get("/api/config")
    async def api_config(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        settings = ctx.settings
        return JSONResponse(
            {
                "llm_provider": settings.llm.provider.value,
                "llm_model": settings.llm.model,
                "crypto_pairs": settings.crypto.pairs,
                "crypto_trading_interval_seconds": settings.crypto.trading_interval_seconds,
                "crypto_max_position_pct": settings.crypto.max_position_pct,
                "crypto_daily_loss_limit": settings.crypto.daily_loss_limit,
                "crypto_daily_return_target": settings.crypto.daily_return_target,
                "db_path": str(settings.db_path),
            }
        )

    @app.get("/api/config/schema")
    async def api_config_schema() -> JSONResponse:
        """Expose every Settings leaf field with its env name + default + type.

        Lets the dashboard render a settings form without us hand-coding
        each field. Secrets (anything matching api_key/secret/token) are
        flagged so the UI can render them as masked inputs.
        """
        return JSONResponse(_walk_settings_schema(Settings))


# ── Schema introspection ───────────────────────────────────────


_SECRET_HINTS = ("api_key", "secret", "token", "client_id", "chat_id")


def _walk_settings_schema(model: type[BaseSettings]) -> list[dict[str, Any]]:
    """Yield one entry per scalar leaf field across the Settings tree."""
    out: list[dict[str, Any]] = []
    own_prefix = model.model_config.get("env_prefix", "") or ""
    for name, field in model.model_fields.items():
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseSettings):
            out.extend(_walk_settings_schema(ann))
            continue
        env_name = (
            field.validation_alias
            if isinstance(field.validation_alias, str)
            else (own_prefix + name).upper()
        )
        out.append(
            {
                "env_name": env_name,
                "owner": model.__name__,
                "type": _type_name(field.annotation),
                "default": _default_value(field),
                "description": field.description or "",
                "secret": any(h in env_name.lower() for h in _SECRET_HINTS),
            }
        )
    return out


def _type_name(annotation: Any) -> str:
    """Stringify the field's annotation for the schema response."""
    if annotation is None:
        return "any"
    name = getattr(annotation, "__name__", None)
    return name or str(annotation)


def _default_value(field: FieldInfo) -> Any:
    """Pull a JSON-friendly default out of a FieldInfo.

    Anything that's not a JSON-native type (Path, custom objects) gets
    stringified so the schema endpoint never blows up on serialization.
    """
    if field.default_factory is not None:
        try:
            raw = field.default_factory()
        except TypeError:
            return None
    elif field.default is None or field.default is Ellipsis:
        return None
    elif hasattr(field.default, "value"):  # Enum default
        return field.default.value
    else:
        raw = field.default

    return _to_json_safe(raw)


def _to_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    return str(value)
