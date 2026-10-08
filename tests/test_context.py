"""Tests for :mod:`core.context` — `DashboardContext` and `RuntimeView`.

These types replace the old `app_state: dict[str, Any]` bag. Tests
elsewhere (`test_ws_cycle`, `test_prometheus`) use them as construction
helpers — this file pins the contract: field defaults and the frozen +
mutable boundary (static deps frozen, `runtime` view mutable).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from halal_trader.core.context import DashboardContext, RuntimeView


def _ctx_kwargs() -> dict:
    """Sentinel objects for the frozen fields — identity is what we test."""
    return {
        "engine": object(),
        "repo": object(),
        "analytics": object(),
        "settings": object(),
        "bus": object(),
        "runtime": RuntimeView(),
    }


# ── RuntimeView defaults ────────────────────────────────────


def test_runtime_view_is_mutable():
    """The view is intentionally mutable — the cycle pushes into
    `risk_state`, `last_cycle`, etc. on each tick."""
    rv = RuntimeView()
    rv.bot_running = True
    rv.risk_state = {"drawdown": 0.05, "market": "stocks"}
    assert rv.bot_running is True
    assert rv.risk_state == {"drawdown": 0.05, "market": "stocks"}


# ── DashboardContext shape ─────────────────────────────────


def test_dashboard_context_holds_all_six_fields():
    """The dataclass projects exactly six fields — pin so a future
    field add (or removal) is intentional."""
    field_names = {f.name for f in fields(DashboardContext)}
    assert field_names == {
        "engine",
        "repo",
        "analytics",
        "settings",
        "bus",
        "runtime",
    }


def test_dashboard_context_is_frozen():
    """The static deps are frozen — re-pointing a field must raise so
    routes can't accidentally swap an engine mid-flight."""
    ctx = DashboardContext(**_ctx_kwargs())
    with pytest.raises(FrozenInstanceError):
        ctx.engine = object()  # type: ignore[misc]


def test_dashboard_context_runtime_field_remains_mutable():
    """The frozen wrapper guards the *fields* of DashboardContext (you
    can't replace `runtime` itself), but the RuntimeView it holds is
    still a regular dataclass — its fields mutate freely."""
    ctx = DashboardContext(**_ctx_kwargs())
    ctx.runtime.bot_running = True  # ok — mutating the held view
    assert ctx.runtime.bot_running is True
    # But re-pointing `runtime` to a different RuntimeView is still
    # forbidden (the outer container is frozen).
    with pytest.raises(FrozenInstanceError):
        ctx.runtime = RuntimeView()  # type: ignore[misc]


# ── slots invariants ────────────────────────────────────


def test_dashboard_context_uses_slots():
    """``slots=True`` means no `__dict__` — pin via attribute absence
    rather than assignment (slots+frozen interact in confusing ways
    that make `pytest.raises` brittle)."""
    ctx = DashboardContext(**_ctx_kwargs())
    assert not hasattr(ctx, "__dict__")


def test_the_web_knows_only_when_it_started() -> None:
    """The bot's in-process state never reaches the web: no field pretends it does."""
    from dataclasses import fields

    assert [f.name for f in fields(RuntimeView)] == ["started_at"]
