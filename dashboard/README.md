# Halal Trader Dashboard

React + TypeScript + Vite SPA that surfaces the bot's live state:
positions, P&L, halal screening verdicts, LLM decisions with token
counts and cost, regime / drift / shadow / RAG insights, and a live
cycle event stream.

## Running

The bot's FastAPI server (`web/app.py`) serves the built SPA from
`dashboard/dist/` at `http://localhost:8082`. The fastest path is:

```bash
# from the repo root
cd dashboard && npm install   # one-time
just dashboard-build           # build dist/ for production serve
just dashboard                  # starts the bot's web server (serves dist/)
```

For frontend-only iteration with HMR, run the Vite dev server in
parallel with the bot's web server:

```bash
just dashboard       # in one terminal — bot at :8082
just dashboard-dev   # in another terminal — Vite at :5173 with HMR
```

`vite.config.ts` proxies `/api`, `/ws`, and `/metrics` to `:8082` so
the dev server hot-reloads UI changes while real bot data flows
through.

## Backend surface

| Endpoint group | Purpose |
|---|---|
| `GET /api/positions` | Open trades and balances |
| `GET /api/trades` | Recent fills with cost-basis + P&L |
| `GET /api/decisions` | LLM decision audit trail (token counts, cost, prompt version) |
| `GET /api/risk/state` | Most recent `PortfolioRiskState` (heat / drawdown / correlation), with a `market` discriminator (`crypto` / `stocks`) so the panel labels whose snapshot it is |
| `GET /api/insights/{regret,thesis,drift,stress,shadow,calibration,regime,basis,treasury,purification,replay}` | Per-analytic snapshots |
| `GET /api/halal/explain/{trade_id}` | Markdown Sharia explainer for a trade |
| `GET /api/mobile/summary` / `WS /ws/state` | Phone-friendly summary (halt status, drawdown, open positions by asset, daily P&L, LLM cost, plus the same `drawdown_market` discriminator) |
| `POST /api/admin/halt` / `POST /api/admin/resume` | Operator kill switch (gated by `WEB_API_TOKEN`) |
| `WS /ws/cycle` | Live cycle event stream (cycle.start / stage.complete / llm.call / executor.fill / monitor.exit) |
| `GET /metrics` | Prometheus scrape (per-stage histograms, cost rates, drawdown / heat with `market` label, …) |

## Auth

Mutating routes (`/api/admin/*`, halt/resume, paused-pair toggles)
require the bearer token in `WEB_API_TOKEN` plus a one-time confirm
gate (`WEB_REQUIRE_CONFIRMATION`). Read-only routes are open by
default; tighten via the FastAPI middleware in `web/app.py` if
exposing the dashboard beyond localhost.

## Build commands

| Command | What it does |
|---|---|
| `just dashboard-install` | `npm install` (one-time) |
| `just dashboard-build` | `npm run build` — output to `dist/` |
| `just dashboard-dev` | `npm run dev` — Vite HMR on `:5173` |
| `just dashboard-lint` | `npm run lint` |
