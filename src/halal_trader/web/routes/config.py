"""GET /api/config/schema: every environment variable the settings read, with its metadata."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

from halal_trader.config import Settings


def register(app: FastAPI) -> None:
    @app.get("/api/config/schema")
    async def api_config_schema() -> JSONResponse:
        """Expose every Settings leaf field with its env name + default + type.

        Lets the dashboard render a settings form without us hand-coding
        each field. Secrets (anything matching api_key/secret/token) are
        flagged so the UI can render them as masked inputs.
        """
        return JSONResponse(_walk_settings_schema(Settings))


# ── Schema introspection ───────────────────────────────────────


# Substrings of an env name that make it a secret: masked in the UI, and its
# default never sent (DATABASE_URL's default carries the repo-default
# Postgres password). Webhook URLs are bearer credentials; LIVE_MODE_
# CONFIRMATION is the live-trading arming token; EDGAR's user agent names a
# real contact.
_SECRET_HINTS = (
    "api_key",
    "secret",
    "token",
    "client_id",
    "chat_id",
    "password",
    "database_url",
    "dsn",
    "webhook",
    "confirmation",
    "user_agent",
)


def _walk_settings_schema(model: type[BaseSettings]) -> list[dict[str, Any]]:
    """Yield one entry per scalar leaf field across the Settings tree."""
    out: list[dict[str, Any]] = []
    own_prefix = model.model_config.get("env_prefix", "") or ""
    for name, field in model.model_fields.items():
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseSettings):
            out.extend(_walk_settings_schema(ann))
            continue
        if name not in getattr(model, "ENV", frozenset()):
            continue  # a fixed value, not configuration
        env_name = (
            field.validation_alias
            if isinstance(field.validation_alias, str)
            else (own_prefix + name).upper()
        )
        secret = any(h in env_name.lower() for h in _SECRET_HINTS)
        out.append(
            {
                "env_name": env_name,
                "owner": model.__name__,
                "type": _type_name(field.annotation),
                "default": None if secret else _default_value(field),
                "description": field.description or "",
                "secret": secret,
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
            # pydantic v2.12 widened ``default_factory`` to optionally take
            # validated data; the callable still works without args, but
            # mypy types it as 1-arg.
            factory = cast(Callable[[], Any], field.default_factory)
            raw = factory()
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
