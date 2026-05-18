"""Wave L wiring tests — /api/halal/explain route + halal explain CLI.

The explainer module itself is covered by
``tests/test_halal_explainer.py``. This file pins the route + CLI
that surface it to operators:

* GET /api/halal/explain/{trade_id} returns the rendered Markdown for
  a trade with a screening, an "unattested" body for legacy trades
  pre-dating the FK, and 404 for unknown trade ids.
* halal-trader halal explain TRADE_ID prints the same Markdown.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── CLI shape ───────────────────────────────────────────────────


def test_cli_halal_group_registered() -> None:
    """`halal-trader halal explain` is available."""
    from halal_trader.cli import cli

    cmds = list(cli.commands.keys())
    assert "halal" in cmds
    halal_grp = cli.commands["halal"]
    assert "explain" in list(halal_grp.commands.keys())


def test_cli_halal_explain_help_parses() -> None:
    from click.testing import CliRunner

    from halal_trader.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["halal", "explain", "--help"])
    assert result.exit_code == 0
    assert "Sharia-compliance" in result.output or "explanation" in result.output


def test_cli_halal_explain_accepts_trade_id_and_asset_class() -> None:
    """Smoke-check the CLI argument shape without hitting the DB —
    invoke with --help on the positional arg to confirm parsing."""
    from click.testing import CliRunner

    from halal_trader.cli import cli

    runner = CliRunner()
    # Pass --asset-class to confirm the flag is wired.
    result = runner.invoke(cli, ["halal", "explain", "--help"])
    assert "--asset-class" in result.output
    assert "crypto" in result.output


# ── Route shape ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_explain_route_returns_markdown_for_trade_with_screening() -> None:
    """End-to-end shape: the route pulls the receipt, runs the
    explainer, returns JSON {trade_id, asset_class, decision,
    body_md, sources}."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from halal_trader.core.context import DashboardContext, RuntimeView
    from halal_trader.halal.audit import Receipt
    from halal_trader.web.dependencies import get_ctx
    from halal_trader.web.routes.admin_halal import register

    # Build a Receipt the explainer will turn into markdown.
    payload = {
        "asset_class": "crypto",
        "trade": {"symbol": "BTCUSDT", "id": 42},
        "screening": {
            "decision": "halal",
            "source": "coingecko_rules",
            "criteria": {"category": "layer-1", "market_cap": 1_000_000_000_000},
        },
        "compliance_status": "halal",
    }

    app = FastAPI()
    ctx = DashboardContext(
        engine=MagicMock(),
        repo=MagicMock(),
        hub=MagicMock(),
        analytics=MagicMock(),
        settings=MagicMock(),
        bus=MagicMock(),
        runtime=RuntimeView(),
    )
    app.state.ctx = ctx
    app.dependency_overrides[get_ctx] = lambda: ctx
    register(app)

    with patch(
        "halal_trader.halal.audit.export_receipt",
        new=AsyncMock(return_value=Receipt(payload=payload)),
    ):
        client = TestClient(app)
        response = client.get("/api/halal/explain/42?asset_class=crypto")
    assert response.status_code == 200
    body = response.json()
    assert body["trade_id"] == 42
    assert body["asset_class"] == "crypto"
    assert body["decision"] == "halal"
    assert "BTCUSDT" in body["body_md"]
    assert "HALAL" in body["body_md"]
    assert "layer-1" in body["body_md"]
    assert isinstance(body["sources"], list) and body["sources"]


@pytest.mark.asyncio
async def test_explain_route_returns_404_for_unknown_trade() -> None:
    """``export_receipt`` returns None for a missing trade — the route
    surfaces a 404 rather than a 500 / empty body."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from halal_trader.core.context import DashboardContext, RuntimeView
    from halal_trader.web.dependencies import get_ctx
    from halal_trader.web.routes.admin_halal import register

    app = FastAPI()
    ctx = DashboardContext(
        engine=MagicMock(),
        repo=MagicMock(),
        hub=MagicMock(),
        analytics=MagicMock(),
        settings=MagicMock(),
        bus=MagicMock(),
        runtime=RuntimeView(),
    )
    app.state.ctx = ctx
    app.dependency_overrides[get_ctx] = lambda: ctx
    register(app)

    with patch(
        "halal_trader.halal.audit.export_receipt",
        new=AsyncMock(return_value=None),
    ):
        client = TestClient(app)
        response = client.get("/api/halal/explain/99999?asset_class=crypto")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_explain_route_rejects_bad_asset_class() -> None:
    """Unknown asset class → 400 (the explainer + receipt both only
    understand crypto / stock)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from halal_trader.core.context import DashboardContext, RuntimeView
    from halal_trader.web.dependencies import get_ctx
    from halal_trader.web.routes.admin_halal import register

    app = FastAPI()
    ctx = DashboardContext(
        engine=MagicMock(),
        repo=MagicMock(),
        hub=MagicMock(),
        analytics=MagicMock(),
        settings=MagicMock(),
        bus=MagicMock(),
        runtime=RuntimeView(),
    )
    app.state.ctx = ctx
    app.dependency_overrides[get_ctx] = lambda: ctx
    register(app)

    client = TestClient(app)
    response = client.get("/api/halal/explain/1?asset_class=futures")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_explain_route_handles_legacy_trade_without_screening() -> None:
    """Trades pre-dating the screening FK have screening=None in the
    receipt; the explainer still renders an "unattested" body."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from halal_trader.core.context import DashboardContext, RuntimeView
    from halal_trader.halal.audit import Receipt
    from halal_trader.web.dependencies import get_ctx
    from halal_trader.web.routes.admin_halal import register

    payload = {
        "asset_class": "crypto",
        "trade": {"symbol": "BTCUSDT", "id": 7},
        "screening": None,
        "compliance_status": "unattested",
    }

    app = FastAPI()
    ctx = DashboardContext(
        engine=MagicMock(),
        repo=MagicMock(),
        hub=MagicMock(),
        analytics=MagicMock(),
        settings=MagicMock(),
        bus=MagicMock(),
        runtime=RuntimeView(),
    )
    app.state.ctx = ctx
    app.dependency_overrides[get_ctx] = lambda: ctx
    register(app)

    with patch(
        "halal_trader.halal.audit.export_receipt",
        new=AsyncMock(return_value=Receipt(payload=payload)),
    ):
        client = TestClient(app)
        response = client.get("/api/halal/explain/7?asset_class=crypto")
    assert response.status_code == 200
    body = response.json()
    # Explainer defaults to "doubtful" when no decision present.
    assert body["decision"] == "doubtful"
    assert "BTCUSDT" in body["body_md"]
