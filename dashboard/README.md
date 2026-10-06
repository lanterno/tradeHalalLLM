# Halal Trader Dashboard

React + TypeScript + Vite SPA over the trader's database: the accounts
and core portfolio, the daily stock pick, the halabot shadow engine's
beliefs, positions, trades and P&L, LLM decisions with token counts and
cost, halal compliance, zakat and purification, and the kill-switch.

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

`vite.config.ts` proxies `/api` and `/ws` to `:8082` so the dev server
hot-reloads UI changes while real bot data flows through.

## Backend surface

What the SPA reads (one `src/api/*.ts` client per group):

| Endpoint | Page |
|---|---|
| `GET /api/home` | Home |
| `GET /api/core` | Core portfolio |
| `GET /api/recommendation{,/history,/scorecard}` | Stock of the Day |
| `GET /api/halabot/{beliefs,overview,decisions}` | Belief Board (shadow engine) |
| `GET /api/positions`, `GET /api/trades` | Positions, Trades |
| `GET /api/analytics`, `GET /api/pnl/daily` | Analytics |
| `GET /api/decisions`, `GET /api/adjustments`, `GET /api/metrics/rejections` | Decisions |
| `GET /api/halal/{compliance,zakat,purification}` | Halal |
| `GET /api/risk/state`, `GET /api/system/{halt,reconcile/recent,backups}` | Risk & Halt |
| `GET /api/insights/purification` | Insights |
| `GET /api/metrics/{cycles,llm}` | Observability |
| `GET /api/health`, `GET /api/system/status`, `GET /api/config` | System, sidebar status |
| `POST` / `DELETE /api/system/halt` | Engage / clear the kill-switch |

## Auth

Every mutation needs the `X-Trader-Token` header matching `WEB_API_TOKEN`
(unset, the dashboard is read-only), and halt/resume additionally need
`X-Halt-Confirm: yes`. The SPA sends only the confirm header, so the Halt
button works only behind a proxy that adds the token. Read-only routes
are open to whatever reaches the socket.

## Build commands

| Command | What it does |
|---|---|
| `just dashboard-install` | `npm install` (one-time) |
| `just dashboard-build` | `npm run build` — output to `dist/` |
| `just dashboard-dev` | `npm run dev` — Vite HMR on `:5173` |
| `just dashboard-lint` | `npm run lint` |
