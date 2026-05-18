"""Wave A wiring tests — typed BotContext replaces the app_state dict.

The DashboardContext / BotContext primitives are covered by existing
tests; this file pins the *kill-the-dict* contract: the bot's
RuntimeView gets the writes that used to land in ``app_state``, the
``app_state`` import path no longer exists, and the BotContext is
projection-compatible with the dashboard's narrower view.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

# ── app_state has been fully removed ────────────────────────────


def test_app_state_no_longer_importable_from_web_app() -> None:
    """The legacy ``app_state: dict[str, Any]`` shim is gone."""
    from halal_trader.web import app as web_app

    assert not hasattr(web_app, "app_state"), (
        "web/app.py still exports `app_state` — Wave A acceptance bar says it should be deleted"
    )


def test_no_app_state_dict_reads_in_src() -> None:
    """``grep -r 'app_state[' src/`` must be zero (acceptance bar)."""
    import subprocess

    proc = subprocess.run(
        ["grep", "-rn", 'app_state\\["', "src/"],
        check=False,
        capture_output=True,
        text=True,
    )
    # grep returns 1 when no matches found — that's success.
    assert proc.returncode == 1, (
        f"Found app_state dict reads in src/:\n{proc.stdout}\n"
        "Wave A acceptance bar: 0 reads of ``app_state['...']`` allowed."
    )


# ── BotContext shape ────────────────────────────────────────────


def test_bot_context_to_dashboard_context_shares_runtime_ref() -> None:
    """The dashboard projection must share the *same* RuntimeView
    instance — otherwise mutations from the bot's cycle wouldn't be
    visible to the dashboard."""
    from halal_trader.core.context import BotContext, RuntimeView

    runtime = RuntimeView(bot_running=True)
    bot_ctx = BotContext(
        engine=MagicMock(),
        repo=MagicMock(),
        hub=MagicMock(),
        analytics=MagicMock(),
        settings=MagicMock(),
        bus=MagicMock(),
        runtime=runtime,
    )
    dash = bot_ctx.to_dashboard_context()
    assert dash.runtime is runtime  # same object, not a copy

    # Mutations on the bot's runtime are visible via the dashboard.
    runtime.last_cycle = {"completed_at": "2026-05-18T12:00:00+00:00"}
    assert dash.runtime.last_cycle == {"completed_at": "2026-05-18T12:00:00+00:00"}


def test_runtime_view_defaults_are_safe() -> None:
    """A bare RuntimeView is constructible — the dashboard process
    creates one before any cycle has run."""
    from halal_trader.core.context import RuntimeView

    rv = RuntimeView()
    assert rv.bot_running is False
    assert rv.started_at is None
    assert rv.last_cycle is None
    assert rv.ws_manager is None
    assert rv.crypto_broker is None


# ── BaseTradingBot owns the runtime ─────────────────────────────


def test_base_trading_bot_init_creates_runtime_and_no_ctx_yet() -> None:
    """Before ``_create_components`` runs, the bot has an empty
    ``RuntimeView`` but no ``BotContext`` yet (engine isn't built)."""
    import os

    from halal_trader.core.scheduler import BaseTradingBot

    # Set required env var so settings construction succeeds.
    os.environ.setdefault("BINANCE_API_KEY", "test")
    os.environ.setdefault("BINANCE_SECRET_KEY", "test")

    class _DummyBot(BaseTradingBot):
        async def _create_components(self) -> None: ...
        async def _daily_start(self) -> None: ...
        async def _daily_end(self) -> None: ...
        def _get_cycle_service(self):
            return None

        async def run(self) -> None: ...

    bot = _DummyBot()
    assert bot._runtime.bot_running is False
    assert bot._ctx is None  # Not built until _create_components runs


def test_runtime_view_mutation_visible_through_dashboard_ctx() -> None:
    """Acceptance: the cycle writes ``runtime.last_cycle``; the
    dashboard's ``ctx.runtime.last_cycle`` reads the same value."""
    from halal_trader.core.context import BotContext, RuntimeView

    runtime = RuntimeView()
    bot_ctx = BotContext(
        engine=MagicMock(),
        repo=MagicMock(),
        hub=MagicMock(),
        analytics=MagicMock(),
        settings=MagicMock(),
        bus=MagicMock(),
        runtime=runtime,
    )
    dash = bot_ctx.to_dashboard_context()

    # Simulate the cycle's mutation
    now = datetime.now(UTC).isoformat()
    runtime.last_cycle = {"completed_at": now, "market": "crypto"}
    runtime.bot_running = True

    assert dash.runtime.last_cycle == {"completed_at": now, "market": "crypto"}
    assert dash.runtime.bot_running is True


# ── insights_hub renamed serialiser ─────────────────────────────


def test_insights_hub_snapshot_replaces_to_app_state() -> None:
    """The legacy ``to_app_state`` was renamed to ``snapshot``;
    the new name shouldn't have surprises."""
    from halal_trader.core.insights_hub import InsightsHub

    hub = InsightsHub()
    assert not hasattr(hub, "to_app_state"), "Legacy method should be removed"
    snap = hub.snapshot()
    assert isinstance(snap, dict)
    assert "drift_monitor" in snap
    assert "shadow_ledger" in snap


# ── Dashboard ctx wiring (smoke) ────────────────────────────────


@pytest.mark.asyncio
async def test_get_ctx_returns_attached_context() -> None:
    """The FastAPI dependency returns whatever the lifespan attached.

    Verifies the indirection in ``web/dependencies.py:get_ctx``
    without spinning up a real FastAPI app — uses a stub request
    object with the expected ``app.state.ctx`` chain."""
    from types import SimpleNamespace

    from halal_trader.core.context import DashboardContext, RuntimeView
    from halal_trader.web.dependencies import get_ctx

    runtime = RuntimeView(bot_running=True)
    ctx = DashboardContext(
        engine=MagicMock(),
        repo=MagicMock(),
        hub=MagicMock(),
        analytics=MagicMock(),
        settings=MagicMock(),
        bus=MagicMock(),
        runtime=runtime,
    )
    fake_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ctx=ctx)))
    out = get_ctx(fake_request)  # type: ignore[arg-type]
    assert out is ctx
    assert out.runtime.bot_running is True


@pytest.mark.asyncio
async def test_get_ctx_raises_when_lifespan_didnt_attach() -> None:
    """A programming error: the lifespan was bypassed or the dependency
    runs before lifespan completes. Surface loudly so a 500 with
    actionable message reaches the operator."""
    from types import SimpleNamespace

    from halal_trader.web.dependencies import get_ctx

    fake_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="DashboardContext not attached"):
        get_ctx(fake_request)  # type: ignore[arg-type]


# ── Item 7: co-host runtime sync ────────────────────────────────


def test_attach_to_app_raises_before_initialize() -> None:
    """``attach_to_app`` is a post-initialize operation — calling it
    before ``initialize()`` has built ``self._ctx`` is a programming
    error that should surface loudly, not silently succeed."""
    import os
    from types import SimpleNamespace

    from halal_trader.core.scheduler import BaseTradingBot

    os.environ.setdefault("BINANCE_API_KEY", "test")
    os.environ.setdefault("BINANCE_SECRET_KEY", "test")

    class _DummyBot(BaseTradingBot):
        async def _create_components(self) -> None: ...
        async def _daily_start(self) -> None: ...
        async def _daily_end(self) -> None: ...
        def _get_cycle_service(self):
            return None

        async def run(self) -> None: ...

    bot = _DummyBot()
    fake_app = SimpleNamespace(state=SimpleNamespace())
    with pytest.raises(RuntimeError, match="initialized"):
        bot.attach_to_app(fake_app)


def test_attach_to_app_projects_bot_ctx_onto_app_state() -> None:
    """Happy path: after init, ``attach_to_app`` writes the bot's
    DashboardContext projection onto the FastAPI app's state. The
    projection MUST share the same RuntimeView so cycle writes are
    visible via the dashboard's ctx.runtime reads."""
    import os
    from types import SimpleNamespace

    from halal_trader.core.context import BotContext, RuntimeView
    from halal_trader.core.scheduler import BaseTradingBot

    os.environ.setdefault("BINANCE_API_KEY", "test")
    os.environ.setdefault("BINANCE_SECRET_KEY", "test")

    class _DummyBot(BaseTradingBot):
        async def _create_components(self) -> None: ...
        async def _daily_start(self) -> None: ...
        async def _daily_end(self) -> None: ...
        def _get_cycle_service(self):
            return None

        async def run(self) -> None: ...

    bot = _DummyBot()
    runtime = RuntimeView(bot_running=True)
    # Simulate what _create_components would have produced.
    bot._ctx = BotContext(
        engine=MagicMock(),
        repo=MagicMock(),
        hub=MagicMock(),
        analytics=MagicMock(),
        settings=MagicMock(),
        bus=MagicMock(),
        runtime=runtime,
    )

    fake_app = SimpleNamespace(state=SimpleNamespace())
    bot.attach_to_app(fake_app)

    # The dashboard projection is on the app state.
    dash = fake_app.state.ctx
    assert dash.engine is bot._ctx.engine
    assert dash.repo is bot._ctx.repo
    assert dash.runtime is runtime  # shared ref, not a copy

    # Cycle writes (simulated) are visible through the dashboard ctx.
    runtime.last_cycle = {"market": "crypto"}
    assert dash.runtime.last_cycle == {"market": "crypto"}


def test_dashboard_lifespan_skips_build_when_ctx_preinstalled(monkeypatch, tmp_path) -> None:
    """When the bot's ``attach_to_app`` runs BEFORE the dashboard
    lifespan, the lifespan must NOT rebuild a parallel DashboardContext
    — it sees the pre-installed ``app.state.ctx`` and yields straight
    through. Pinning this so a co-hosted bot owns the engine."""
    import os
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from halal_trader.core.context import DashboardContext, RuntimeView
    from halal_trader.web import app as web_app

    os.environ.setdefault("BINANCE_API_KEY", "test")
    os.environ.setdefault("BINANCE_SECRET_KEY", "test")
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))

    sentinel_engine = SimpleNamespace(name="bot_engine")
    preinstalled = DashboardContext(
        engine=sentinel_engine,  # type: ignore[arg-type]
        repo=MagicMock(),
        hub=MagicMock(),
        analytics=MagicMock(),
        settings=MagicMock(),
        bus=MagicMock(),
        runtime=RuntimeView(bot_running=True),
    )

    app = web_app.create_app()
    app.state.ctx = preinstalled

    # Patch init_db to fail loudly — if the lifespan tries to rebuild,
    # the test catches it. The co-host path should never call init_db.
    def _boom(*_a, **_k):
        raise AssertionError("lifespan rebuilt despite pre-installed ctx")

    monkeypatch.setattr("halal_trader.web.app.init_db", _boom)

    with TestClient(app) as c:
        # The pre-installed ctx survived lifespan.
        assert c.app.state.ctx is preinstalled
        assert c.app.state.ctx.engine is sentinel_engine
