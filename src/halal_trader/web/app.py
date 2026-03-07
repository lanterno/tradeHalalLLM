"""FastAPI dashboard application — REST API + SSE + HTMX templates."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from halal_trader.config import get_settings
from halal_trader.crypto.analytics import PerformanceAnalytics
from halal_trader.db.models import init_db
from halal_trader.db.repository import Repository

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def create_app():
    """Create and configure the FastAPI application."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    app = FastAPI(title="Halal Trader Dashboard", version="0.2.0")

    _state: dict[str, Any] = {}

    @app.on_event("startup")
    async def startup():
        settings = get_settings()
        engine = await init_db(str(settings.db_path))
        _state["engine"] = engine
        _state["repo"] = Repository(engine)
        _state["analytics"] = PerformanceAnalytics(_state["repo"])

    @app.on_event("shutdown")
    async def shutdown():
        if "engine" in _state:
            await _state["engine"].dispose()

    # ── HTML Pages ─────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _render_page("dashboard")

    @app.get("/trades", response_class=HTMLResponse)
    async def trades_page():
        return _render_page("trades")

    @app.get("/analytics", response_class=HTMLResponse)
    async def analytics_page():
        return _render_page("analytics")

    # ── REST API ───────────────────────────────────────────────

    @app.get("/api/trades")
    async def api_trades(limit: int = 100):
        repo: Repository = _state["repo"]
        trades = await repo.get_recent_crypto_trades(limit)
        return JSONResponse(trades)

    @app.get("/api/pnl/daily")
    async def api_daily_pnl(days: int = 30):
        repo: Repository = _state["repo"]
        pnl = await repo.get_crypto_pnl_history(limit=days)
        return JSONResponse(pnl)

    @app.get("/api/analytics")
    async def api_analytics(days: int = 7):
        analytics: PerformanceAnalytics = _state["analytics"]
        stats = await analytics.compute_stats(lookback_days=days)
        return JSONResponse({
            "total_trades": stats.total_trades,
            "wins": stats.wins,
            "losses": stats.losses,
            "win_rate": stats.win_rate,
            "avg_win_pct": stats.avg_win_pct,
            "avg_loss_pct": stats.avg_loss_pct,
            "total_pnl": stats.total_pnl,
            "profit_factor": stats.profit_factor,
            "max_drawdown_pct": stats.max_drawdown_pct,
            "avg_hold_minutes": stats.avg_hold_minutes,
            "best_pair": stats.best_pair,
            "worst_pair": stats.worst_pair,
            "streak": stats.streak,
            "streak_type": stats.streak_type,
            "by_exit_reason": stats.by_exit_reason,
        })

    @app.get("/api/decisions")
    async def api_decisions(limit: int = 50):
        repo: Repository = _state["repo"]
        # Use raw SQL for LLM decisions since we don't have a dedicated method
        from sqlalchemy import text
        async with repo._engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT * FROM llm_decisions ORDER BY timestamp DESC LIMIT :limit"
                ),
                {"limit": limit},
            )
            rows = [dict(row._mapping) for row in result]
            for row in rows:
                for k, v in row.items():
                    if isinstance(v, datetime):
                        row[k] = v.isoformat()
            return JSONResponse(rows)

    @app.get("/api/adjustments")
    async def api_adjustments():
        repo: Repository = _state["repo"]
        adjustments = await repo.get_recent_adjustments()
        return JSONResponse(adjustments)

    @app.get("/api/health")
    async def api_health():
        return JSONResponse({
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "0.2.0",
        })

    # ── SSE Live Updates ───────────────────────────────────────

    @app.get("/api/sse")
    async def sse():
        async def event_stream():
            while True:
                repo: Repository = _state["repo"]
                trades = await repo.get_recent_crypto_trades(5)
                data = json.dumps({"trades": trades}, default=str)
                yield f"data: {data}\n\n"
                await asyncio.sleep(10)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def _render_page(page: str) -> str:
    """Render an HTMX-powered HTML page."""
    template_path = _TEMPLATE_DIR / f"{page}.html"
    if template_path.exists():
        return template_path.read_text()

    # Inline fallback template
    return _INLINE_TEMPLATES.get(page, _INLINE_TEMPLATES["dashboard"])


_INLINE_TEMPLATES = {
    "dashboard": """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Halal Trader Dashboard</title>
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
               background: #0a0a0f; color: #e0e0e0; }
        .header { background: #111118; padding: 1rem 2rem; border-bottom: 1px solid #1a1a2e;
                  display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 1.4rem; color: #4ade80; }
        .nav { display: flex; gap: 1rem; }
        .nav a { color: #888; text-decoration: none; padding: 0.5rem 1rem; border-radius: 6px; }
        .nav a:hover, .nav a.active { color: #fff; background: #1a1a2e; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1rem; padding: 1.5rem; }
        .card { background: #111118; border: 1px solid #1a1a2e; border-radius: 10px;
                padding: 1.2rem; }
        .card h3 { font-size: 0.85rem; color: #888; text-transform: uppercase;
                   letter-spacing: 0.05em; margin-bottom: 0.5rem; }
        .card .value { font-size: 1.8rem; font-weight: 700; }
        .green { color: #4ade80; }
        .red { color: #f87171; }
        .chart-container { background: #111118; border: 1px solid #1a1a2e; border-radius: 10px;
                          padding: 1.2rem; margin: 0 1.5rem 1.5rem; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 0.6rem 1rem; border-bottom: 1px solid #1a1a2e; }
        th { color: #888; font-size: 0.8rem; text-transform: uppercase; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block;
                     margin-right: 6px; }
        .status-dot.live { background: #4ade80; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>Halal Trader</h1>
        <div class="nav">
            <a href="/" class="active">Dashboard</a>
            <a href="/trades">Trades</a>
            <a href="/analytics">Analytics</a>
        </div>
        <div><span class="status-dot live"></span> Live</div>
    </div>
    <div class="grid" id="stats" hx-get="/api/analytics" hx-trigger="load, every 30s"
         hx-swap="innerHTML">
        <div class="card"><h3>Loading...</h3></div>
    </div>
    <div class="chart-container">
        <h3 style="color:#888;font-size:0.85rem;text-transform:uppercase;margin-bottom:1rem;">
            Recent Trades</h3>
        <div id="trades-table" hx-get="/api/trades?limit=20" hx-trigger="load, every 15s"
             hx-swap="innerHTML">
            <p style="color:#666;">Loading trades...</p>
        </div>
    </div>
    <script>
    document.body.addEventListener('htmx:afterSwap', function(e) {
        if (e.detail.target.id === 'stats') {
            try {
                const data = JSON.parse(e.detail.xhr.responseText);
                const pnlClass = data.total_pnl >= 0 ? 'green' : 'red';
                const wrClass = data.win_rate >= 0.5 ? 'green' : 'red';
                e.detail.target.innerHTML = `
                    <div class="card"><h3>Total P&L</h3>
                        <div class="value ${pnlClass}">$${data.total_pnl?.toFixed(2) || '0.00'}</div></div>
                    <div class="card"><h3>Win Rate</h3>
                        <div class="value ${wrClass}">${(data.win_rate*100)?.toFixed(0) || 0}%</div></div>
                    <div class="card"><h3>Total Trades</h3>
                        <div class="value">${data.total_trades || 0}</div></div>
                    <div class="card"><h3>Profit Factor</h3>
                        <div class="value">${data.profit_factor?.toFixed(2) || '0.00'}</div></div>
                    <div class="card"><h3>Max Drawdown</h3>
                        <div class="value red">${(data.max_drawdown_pct*100)?.toFixed(1) || 0}%</div></div>
                    <div class="card"><h3>Streak</h3>
                        <div class="value">${data.streak || 0} ${data.streak_type || ''}</div></div>`;
            } catch(err) {}
        }
        if (e.detail.target.id === 'trades-table') {
            try {
                const trades = JSON.parse(e.detail.xhr.responseText);
                if (!trades.length) { e.detail.target.innerHTML = '<p style="color:#666">No trades yet.</p>'; return; }
                let html = '<table><tr><th>Time</th><th>Pair</th><th>Side</th><th>Qty</th><th>Price</th><th>Status</th></tr>';
                trades.slice(0, 20).forEach(t => {
                    const cls = t.side === 'buy' ? 'green' : 'red';
                    html += `<tr><td>${(t.timestamp||'').slice(0,19)}</td><td>${t.pair||''}</td>
                        <td class="${cls}">${(t.side||'').toUpperCase()}</td>
                        <td>${Number(t.quantity||0).toFixed(6)}</td>
                        <td>$${Number(t.price||0).toFixed(2)}</td><td>${t.status||''}</td></tr>`;
                });
                html += '</table>';
                e.detail.target.innerHTML = html;
            } catch(err) {}
        }
    });
    </script>
</body>
</html>""",
    "trades": """\
<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Trades - Halal Trader</title>
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
<style>* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0; }
.header { background: #111118; padding: 1rem 2rem; border-bottom: 1px solid #1a1a2e;
          display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 1.4rem; color: #4ade80; }
.nav { display: flex; gap: 1rem; }
.nav a { color: #888; text-decoration: none; padding: 0.5rem 1rem; border-radius: 6px; }
.nav a:hover, .nav a.active { color: #fff; background: #1a1a2e; }
.content { padding: 1.5rem; }
table { width: 100%; border-collapse: collapse; background: #111118; border-radius: 10px; }
th, td { text-align: left; padding: 0.6rem 1rem; border-bottom: 1px solid #1a1a2e; }
th { color: #888; font-size: 0.8rem; text-transform: uppercase; }
.green { color: #4ade80; } .red { color: #f87171; }</style></head>
<body>
<div class="header"><h1>Halal Trader</h1>
<div class="nav"><a href="/">Dashboard</a><a href="/trades" class="active">Trades</a>
<a href="/analytics">Analytics</a></div></div>
<div class="content" id="trades" hx-get="/api/trades?limit=100" hx-trigger="load"
     hx-swap="innerHTML"><p>Loading...</p></div>
<script>
document.body.addEventListener('htmx:afterSwap', function(e) {
    if (e.detail.target.id !== 'trades') return;
    try {
        const trades = JSON.parse(e.detail.xhr.responseText);
        let html = '<table><tr><th>Time</th><th>Pair</th><th>Side</th><th>Qty</th><th>Price</th><th>Status</th><th>Reasoning</th></tr>';
        trades.forEach(t => {
            const cls = t.side === 'buy' ? 'green' : 'red';
            html += `<tr><td>${(t.timestamp||'').slice(0,19)}</td><td>${t.pair||''}</td>
                <td class="${cls}">${(t.side||'').toUpperCase()}</td>
                <td>${Number(t.quantity||0).toFixed(6)}</td><td>$${Number(t.price||0).toFixed(2)}</td>
                <td>${t.status||''}</td><td>${(t.llm_reasoning||'').slice(0,60)}</td></tr>`;
        });
        e.detail.target.innerHTML = html + '</table>';
    } catch(err) { e.detail.target.innerHTML = '<p>Error loading trades</p>'; }
});
</script></body></html>""",
    "analytics": """\
<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Analytics - Halal Trader</title>
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
<style>* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0; }
.header { background: #111118; padding: 1rem 2rem; border-bottom: 1px solid #1a1a2e;
          display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 1.4rem; color: #4ade80; }
.nav { display: flex; gap: 1rem; }
.nav a { color: #888; text-decoration: none; padding: 0.5rem 1rem; border-radius: 6px; }
.nav a:hover, .nav a.active { color: #fff; background: #1a1a2e; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem; padding: 1.5rem; }
.card { background: #111118; border: 1px solid #1a1a2e; border-radius: 10px; padding: 1.2rem; }
.card h3 { font-size: 0.85rem; color: #888; text-transform: uppercase; margin-bottom: 0.5rem; }
.card .value { font-size: 1.5rem; font-weight: 700; }
.green { color: #4ade80; } .red { color: #f87171; }</style></head>
<body>
<div class="header"><h1>Halal Trader</h1>
<div class="nav"><a href="/">Dashboard</a><a href="/trades">Trades</a>
<a href="/analytics" class="active">Analytics</a></div></div>
<div class="grid" id="analytics" hx-get="/api/analytics?days=30" hx-trigger="load"
     hx-swap="innerHTML"><div class="card"><h3>Loading...</h3></div></div>
<script>
document.body.addEventListener('htmx:afterSwap', function(e) {
    if (e.detail.target.id !== 'analytics') return;
    try {
        const d = JSON.parse(e.detail.xhr.responseText);
        e.detail.target.innerHTML = `
            <div class="card"><h3>Total Trades (30d)</h3><div class="value">${d.total_trades}</div></div>
            <div class="card"><h3>Win Rate</h3><div class="value ${d.win_rate>=0.5?'green':'red'}">${(d.win_rate*100).toFixed(0)}%</div></div>
            <div class="card"><h3>Total P&L</h3><div class="value ${d.total_pnl>=0?'green':'red'}">$${d.total_pnl.toFixed(2)}</div></div>
            <div class="card"><h3>Profit Factor</h3><div class="value">${d.profit_factor.toFixed(2)}</div></div>
            <div class="card"><h3>Max Drawdown</h3><div class="value red">${(d.max_drawdown_pct*100).toFixed(1)}%</div></div>
            <div class="card"><h3>Avg Win</h3><div class="value green">${(d.avg_win_pct*100).toFixed(2)}%</div></div>
            <div class="card"><h3>Avg Loss</h3><div class="value red">${(d.avg_loss_pct*100).toFixed(2)}%</div></div>
            <div class="card"><h3>Avg Hold</h3><div class="value">${d.avg_hold_minutes.toFixed(0)} min</div></div>
            <div class="card"><h3>Best Pair</h3><div class="value green">${d.best_pair||'N/A'}</div></div>
            <div class="card"><h3>Worst Pair</h3><div class="value red">${d.worst_pair||'N/A'}</div></div>`;
    } catch(err) {}
});
</script></body></html>""",
}
