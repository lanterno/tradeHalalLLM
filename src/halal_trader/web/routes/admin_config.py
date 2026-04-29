"""Runtime tunable controls — config overrides + prompt picker + A/B weights.

The ``runtime_config`` table is a key/value overlay over ``Settings``.
This endpoint group lets the operator:

* GET the current overlay map (one row per overridden key).
* PATCH a single key with a JSON-typed value (validated against the
  field's declared type from ``Settings`` schema).
* DELETE a key to revert to its .env default.
* GET the registered prompt versions (Phase 0.3 registry) and switch
  the active one.
* GET / PATCH the A/B router's per-variant weights.

The bot picks up the overlay on its next cycle by reading via the new
``Settings.from_runtime_overlay`` helper. Bounds are enforced
*server-side* — the dashboard's UI hints are nice but never the only
defence.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx
from halal_trader.web.middleware.confirm import require_confirmation

logger = logging.getLogger(__name__)


class ConfigPatchRequest(BaseModel):
    """One key/value pair to overlay onto Settings."""

    value: Any  # any JSON-native type; validation happens server-side


def register(app: FastAPI) -> None:
    @app.get("/api/admin/config/runtime")
    async def get_runtime_config(
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        return JSONResponse(await ctx.repo.list_runtime_config())

    @app.patch(
        "/api/admin/config/runtime/{key}",
        dependencies=[Depends(require_confirmation)],
    )
    async def patch_runtime_config(
        key: str,
        body: ConfigPatchRequest,
        ctx: DashboardContext = Depends(get_ctx),
    ) -> JSONResponse:
        from halal_trader.config import Settings
        from halal_trader.web.routes.config import _walk_settings_schema

        schema = {row["env_name"]: row for row in _walk_settings_schema(Settings)}
        if key.upper() not in schema:
            raise HTTPException(404, f"unknown settings key: {key.upper()}")

        if not _value_in_bounds(schema[key.upper()], body.value):
            raise HTTPException(
                422,
                f"value {body.value!r} out of bounds for {key.upper()} "
                f"(type {schema[key.upper()]['type']})",
            )

        await ctx.repo.set_runtime_config(key, body.value, set_by="dashboard")
        return JSONResponse({"key": key.upper(), "value": body.value})

    @app.delete(
        "/api/admin/config/runtime/{key}",
        dependencies=[Depends(require_confirmation)],
    )
    async def delete_runtime_config(
        key: str, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        ok = await ctx.repo.delete_runtime_config(key)
        if not ok:
            raise HTTPException(404, f"no override for {key.upper()}")
        return JSONResponse({"key": key.upper(), "reverted": True})

    @app.get("/api/admin/prompts")
    async def list_prompts(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        import halal_trader.crypto.prompts as crypto_prompts
        import halal_trader.trading.strategy as trading_strategy
        from halal_trader.core.llm.prompts import list_versions, register

        for module, attr in (
            (crypto_prompts, "PROMPT_VERSION"),
            (trading_strategy, "PROMPT_VERSION"),
            (trading_strategy, "USER_PROMPT_VERSION"),
        ):
            pv = getattr(module, attr, None)
            if pv is not None:
                try:
                    register(pv.name, pv.template)
                except ValueError:
                    pass

        active = (await ctx.repo.list_runtime_config()).get("ACTIVE_PROMPT_VERSION")

        out = []
        for name, pv in list_versions().items():
            out.append(
                {
                    "name": name,
                    "version_id": pv.version_id,
                    "short": pv.short,
                    "active": pv.short == active,
                }
            )
        return JSONResponse(sorted(out, key=lambda r: r["name"]))

    @app.post(
        "/api/admin/prompts/active",
        dependencies=[Depends(require_confirmation)],
    )
    async def set_active_prompt(
        body: dict[str, str], ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        version = body.get("version", "").strip()
        if not version or "@" not in version:
            raise HTTPException(422, "body must include 'version' as 'name@hash'")
        await ctx.repo.set_runtime_config("ACTIVE_PROMPT_VERSION", version, set_by="dashboard")
        return JSONResponse({"active": version})

    @app.get("/api/admin/ab/weights")
    async def get_ab_weights(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        weights = (await ctx.repo.list_runtime_config()).get("AB_VARIANT_WEIGHTS", {})
        return JSONResponse(weights)

    @app.patch(
        "/api/admin/ab/weights",
        dependencies=[Depends(require_confirmation)],
    )
    async def patch_ab_weights(
        body: dict[str, float], ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        if not body:
            raise HTTPException(422, "weights map cannot be empty")
        for k, v in body.items():
            if not isinstance(v, (int, float)) or v < 0:
                raise HTTPException(422, f"weight for {k!r} must be a non-negative number")
        await ctx.repo.set_runtime_config("AB_VARIANT_WEIGHTS", dict(body), set_by="dashboard")
        return JSONResponse(body)


def _value_in_bounds(schema_row: dict[str, Any], value: Any) -> bool:
    """Reject out-of-range values for typed scalars before we persist them.

    The schema's ``type`` is the declared Python type name. We don't have
    explicit min/max on every field today, so this is a soft guard: it
    catches obvious type mismatches and refuses negative percentages /
    non-finite floats. A future iteration can read explicit bounds from
    pydantic ``Field(ge=, le=)`` metadata.
    """
    declared = schema_row.get("type", "any")
    if declared in ("int", "float") and not isinstance(value, (int, float)):
        return False
    if declared == "bool" and not isinstance(value, bool):
        return False
    if isinstance(value, float):
        if value != value:  # NaN
            return False
        if value in (float("inf"), float("-inf")):
            return False
    # Heuristic on percentage-shaped knobs (anything ending with _PCT).
    if schema_row["env_name"].upper().endswith("_PCT") and isinstance(value, (int, float)):
        if value < 0 or value > 1:
            return False
    return True
