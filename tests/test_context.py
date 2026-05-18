"""Tests for :mod:`core.context` — `BotContext`, `DashboardContext`,
and `RuntimeView`.

These types replace the old `app_state: dict[str, Any]` bag. Tests
elsewhere (`test_web_insights`, `test_ws_cycle`, `test_prometheus`)
use them as construction helpers — this file pins the contract:
field defaults, the projection from `BotContext` to `DashboardContext`,
and the frozen + mutable boundary (static deps frozen, `runtime`
view mutable).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from halal_trader.core.context import BotContext, DashboardContext, RuntimeView


def _ctx_kwargs() -> dict:
    """Sentinel objects for the frozen fields — identity is what we test."""
    return {
        "engine": object(),
        "repo": object(),
        "hub": object(),
        "analytics": object(),
        "settings": object(),
        "bus": object(),
        "runtime": RuntimeView(),
    }


# ── RuntimeView defaults ────────────────────────────────────


def test_runtime_view_constructs_with_no_args():
    """All fields have defaults — no required kwargs."""
    rv = RuntimeView()
    assert rv.bot_running is False
    assert rv.started_at is None
    assert rv.last_cycle is None
    assert rv.risk_state is None
    assert rv.account_snapshot is None


def test_runtime_view_collection_defaults_are_independent():
    """``stock_positions`` and ``open_positions_by_asset`` use
    ``field(default_factory=...)`` — two instances must NOT share the
    same list/dict, otherwise pushing into one leaks into the other."""
    a = RuntimeView()
    b = RuntimeView()
    a.stock_positions.append({"symbol": "AAPL"})
    a.open_positions_by_asset["BTCUSDT"] = []
    assert b.stock_positions == []  # b is unaffected
    assert b.open_positions_by_asset == {}


def test_runtime_view_optional_broker_handles_default_none():
    """``crypto_broker`` / ``stock_broker`` / `ws_manager` /
    `sentiment_manager` are only populated when the bot is co-hosted
    with the dashboard. Dashboard-only processes must see them as
    None to branch correctly."""
    rv = RuntimeView()
    assert rv.crypto_broker is None
    assert rv.stock_broker is None
    assert rv.ws_manager is None
    assert rv.sentiment_manager is None


def test_runtime_view_is_mutable():
    """The view is intentionally mutable — the cycle pushes into
    `risk_state`, `last_cycle`, etc. on each tick."""
    rv = RuntimeView()
    rv.bot_running = True
    rv.risk_state = {"drawdown": 0.05, "market": "crypto"}
    assert rv.bot_running is True
    assert rv.risk_state == {"drawdown": 0.05, "market": "crypto"}


def test_runtime_view_llm_cost_optional_float():
    rv = RuntimeView()
    assert rv.llm_cost_today_usd is None
    rv.llm_cost_today_usd = 1.23
    assert rv.llm_cost_today_usd == 1.23


# ── DashboardContext shape ─────────────────────────────────


def test_dashboard_context_holds_all_seven_fields():
    """The dataclass projects exactly seven fields — pin so a future
    field add (or removal) is intentional."""
    field_names = {f.name for f in fields(DashboardContext)}
    assert field_names == {
        "engine",
        "repo",
        "hub",
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


# ── BotContext shape ──────────────────────────────────────


def test_bot_context_holds_same_seven_fields_as_dashboard():
    """BotContext currently mirrors DashboardContext (the docstring
    says "superset" but no extra fields exist yet). When a bot-only
    field is added, this test breaks — re-evaluate the projection."""
    bot_fields = {f.name for f in fields(BotContext)}
    dash_fields = {f.name for f in fields(DashboardContext)}
    assert bot_fields == dash_fields


def test_bot_context_is_frozen():
    ctx = BotContext(**_ctx_kwargs())
    with pytest.raises(FrozenInstanceError):
        ctx.engine = object()  # type: ignore[misc]


# ── BotContext.to_dashboard_context() projection ───────────


def test_to_dashboard_context_returns_dashboard_context_instance():
    bot = BotContext(**_ctx_kwargs())
    dash = bot.to_dashboard_context()
    assert isinstance(dash, DashboardContext)


def test_to_dashboard_context_passes_field_identity_through():
    """The projection must NOT copy the held objects — engine, repo,
    runtime, etc. all flow through by identity. Otherwise the
    dashboard would observe a stale snapshot of the runtime view
    instead of the live one the cycle is mutating."""
    kwargs = _ctx_kwargs()
    bot = BotContext(**kwargs)
    dash = bot.to_dashboard_context()

    assert dash.engine is kwargs["engine"]
    assert dash.repo is kwargs["repo"]
    assert dash.hub is kwargs["hub"]
    assert dash.analytics is kwargs["analytics"]
    assert dash.settings is kwargs["settings"]
    assert dash.bus is kwargs["bus"]
    assert dash.runtime is kwargs["runtime"]  # critical — same view


def test_to_dashboard_context_runtime_mutations_visible_in_both():
    """The shared RuntimeView is the whole point of the projection —
    the cycle pushes into bot.runtime and the dashboard reads from
    dash.runtime. If they were separate views, the dashboard would
    show stale data."""
    bot = BotContext(**_ctx_kwargs())
    dash = bot.to_dashboard_context()

    bot.runtime.bot_running = True
    bot.runtime.risk_state = {"drawdown": 0.1}

    assert dash.runtime.bot_running is True
    assert dash.runtime.risk_state == {"drawdown": 0.1}


def test_to_dashboard_context_is_idempotent():
    """Calling the projection twice yields equivalent dashboard
    contexts pointing at the same underlying objects (a fresh frozen
    wrapper each time, but identical contents)."""
    bot = BotContext(**_ctx_kwargs())
    a = bot.to_dashboard_context()
    b = bot.to_dashboard_context()
    # Different wrapper instances (frozen dataclass; no caching).
    assert a is not b
    # But every field they hold is the same object.
    assert a.engine is b.engine
    assert a.runtime is b.runtime


# ── slots invariants ────────────────────────────────────


def test_dashboard_context_uses_slots():
    """``slots=True`` means no `__dict__` — pin via attribute absence
    rather than assignment (slots+frozen interact in confusing ways
    that make `pytest.raises` brittle)."""
    ctx = DashboardContext(**_ctx_kwargs())
    assert not hasattr(ctx, "__dict__")


def test_bot_context_uses_slots():
    ctx = BotContext(**_ctx_kwargs())
    assert not hasattr(ctx, "__dict__")
