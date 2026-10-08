"""Wave A wiring tests — the typed DashboardContext replaced the app_state dict.

The DashboardContext / RuntimeView primitives are covered by
``test_context.py``; this file pins the *kill-the-dict* contract: the
``app_state`` import path no longer exists and routes read the context
the lifespan attached.
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

    runtime = RuntimeView(started_at=datetime.now(UTC))
    ctx = DashboardContext(
        engine=MagicMock(),
        repo=MagicMock(),
        analytics=MagicMock(),
        settings=MagicMock(),
        bus=MagicMock(),
        runtime=runtime,
    )
    fake_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ctx=ctx)))
    out = get_ctx(fake_request)  # type: ignore[arg-type]
    assert out is ctx
    assert out.runtime is runtime


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
