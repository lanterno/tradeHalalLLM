"""Prometheus exposition + endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from halal_trader.core.context import RuntimeView
from halal_trader.web import app as web_app
from halal_trader.web.prometheus import (
    MetricSnapshot,
    collect_default_snapshots,
    render_metrics,
)

# ── render_metrics ────────────────────────────────────────────


def test_render_emits_help_and_type_headers():
    snaps = [MetricSnapshot(name="x", help_text="a counter", value=42)]
    text = render_metrics(snaps)
    assert "# HELP x a counter" in text
    assert "# TYPE x gauge" in text
    assert "x 42" in text


def test_render_groups_headers_per_metric_name():
    """Multiple snapshots of the same metric only emit headers once."""
    snaps = [
        MetricSnapshot(name="open", help_text="positions", value=1, labels={"a": "X"}),
        MetricSnapshot(name="open", help_text="positions", value=2, labels={"a": "Y"}),
    ]
    text = render_metrics(snaps)
    assert text.count("# HELP open") == 1
    assert text.count("# TYPE open") == 1
    assert 'open{a="X"} 1' in text
    assert 'open{a="Y"} 2' in text


def test_render_escapes_label_values():
    snap = MetricSnapshot(name="x", help_text="x", value=1, labels={"k": 'v"\\n'})
    text = render_metrics([snap])
    # Quote, backslash, newline all escaped.
    assert 'k="v\\"\\\\n"' in text


def test_render_empty_returns_empty_string():
    assert render_metrics([]) == ""


# ── collect_default_snapshots ─────────────────────────────────


def test_collector_emits_bot_running():
    rt = RuntimeView(bot_running=True)
    snaps = collect_default_snapshots(rt)
    names = [s.name for s in snaps]
    assert "halal_trader_bot_running" in names
    assert next(s for s in snaps if s.name == "halal_trader_bot_running").value == 1.0


def test_collector_emits_open_positions_per_asset():
    rt = RuntimeView(open_positions_by_asset={"crypto": [{}, {}, {}], "stock": [{}]})
    snaps = collect_default_snapshots(rt)
    by_label = {s.labels["asset_class"]: s.value for s in snaps if s.labels}
    assert by_label == {"crypto": 3, "stock": 1}


def test_collector_skips_none_drawdown():
    rt = RuntimeView(
        risk_state={"drawdown_pct": None, "portfolio_heat_pct": 0.02},
    )
    snaps = collect_default_snapshots(rt)
    names = [s.name for s in snaps]
    assert "halal_trader_drawdown_pct" not in names
    assert "halal_trader_portfolio_heat_pct" in names


def test_collector_labels_risk_metrics_with_market():
    """The crypto/stocks discriminator on ``risk_state`` flows through
    to a Prometheus ``market="…"`` label so dashboards can graph the
    two bots' heat / drawdown separately."""
    rt = RuntimeView(
        risk_state={
            "market": "stocks",
            "drawdown_pct": 0.012,
            "portfolio_heat_pct": 0.04,
        },
    )
    snaps = collect_default_snapshots(rt)
    by_name = {s.name: s for s in snaps if s.labels.get("market") == "stocks"}
    assert "halal_trader_drawdown_pct" in by_name
    assert "halal_trader_portfolio_heat_pct" in by_name
    assert by_name["halal_trader_drawdown_pct"].value == 0.012


def test_collector_falls_back_to_unknown_market_label():
    """Pre-discriminator runtime pushes (no ``market`` key) still emit
    the metric — labelled ``market="unknown"`` so the collector path
    stays consistent."""
    rt = RuntimeView(risk_state={"drawdown_pct": 0.005})
    snaps = collect_default_snapshots(rt)
    matches = [s for s in snaps if s.name == "halal_trader_drawdown_pct"]
    assert matches and matches[0].labels.get("market") == "unknown"


# ── /metrics endpoint ─────────────────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    web_app.app_state.clear()
    app = web_app.create_app()

    with TestClient(app) as c:
        c.app.state.ctx.runtime.bot_running = True
        c.app.state.ctx.runtime.llm_cost_today_usd = 0.42
        yield c


def test_metrics_endpoint_returns_text(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "halal_trader_bot_running 1" in r.text
    assert "halal_trader_llm_cost_today_usd 0.42" in r.text
