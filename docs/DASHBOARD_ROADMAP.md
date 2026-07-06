# Stock-bot dashboard — review & improvement roadmap

Review date: 2026-07-07. Scope: the FastAPI + React SPA dashboard
(`dashboard/` frontend served by `src/halal_trader/web/`, on :8082 as the
`trader-web` container). Focus: the **stock** bot. Method: full read of the
SPA (12 pages / 12 hooks / 11 API modules / 11 components) and the backend
(26 wired route modules, ~90 endpoints), cross-checked against the **live**
running dashboard and the Postgres DB (545 trades).

---

## TL;DR — the dashboard barely shows the stock bot

The dashboard is well-built and feature-rich, but as deployed it shows a
stock operator **almost no real data**, for two independent structural
reasons:

1. **It's crypto-first by default.** Every market-aware endpoint
   (`trades`, `positions`, `analytics`, `pnl`) defaults to `market="crypto"`,
   and the SPA **never sends `market=stocks`** — the frontend fetchers don't
   even have a `market` field. So Dashboard / Positions / Trades / Analytics
   render the *empty* crypto tables. Proven live: `/api/trades` → `[]`, but
   `/api/trades?market=stocks` → real rows (ADBE EOD close, id 580); the DB
   holds 545 trades. The backend already fully supports stocks behind the
   param — **the gap is purely that the UI has no market switch.**

2. **It runs standalone, so all in-memory data is empty.** `trader-web` is a
   separate container from `trader-stocks`. The co-host path
   (`app.build_engine`/`attach_to_app`) never fires, so `ctx.hub` and
   `ctx.runtime` are built empty (`web/app.py:63-65`). Every endpoint that
   reads runtime/hub state — `/api/sentiment`, `/api/risk/state`, most of
   `/api/insights/*`, `/ws/cycle`, live WS prices — returns `[]` /
   `{available:false}`. Confirmed live: `system/status` shows
   `bot_running:false` and `/api/sentiment` → `[]` while the bot is actively
   trading in the sibling container.

Everything that is **DB-backed** works today: Stock-of-the-Day
recommendation (real AMZN thesis, conviction 0.72), the LLM Decisions log
(92 KB of real GLM decisions), Belief Board (`hb_*` tables), and — once the
market param is passed — trades/positions/analytics/pnl.

**Net:** the improvement work is mostly **frontend wiring + a deployment
decision**, not new backend. The backend is, if anything, over-built
(~90 endpoints, ~24 consumed; a whole 25-endpoint admin control plane with
no UI; 33 stale dead modules).

---

## State of the dashboard

### Pages (12) — completeness as *built*, and what a stock operator actually sees today

| Route | Purpose | Built | Stock operator sees today |
|---|---|---|---|
| `/` Dashboard | KPI cards + equity curve + recent trades + rec card | 70% | Mostly **zeros/empty** (crypto default); only the Stock-of-the-Day card is real |
| `/recommendation` Stock of the Day | Halal advisory pick + scorecard + history | **90%** | **Works** — best stock page |
| `/beliefs` Belief Board | halabot shadow-engine beliefs | 80% | Works (advisory shadow engine; not trade data) |
| `/positions` Positions | Open positions + allocation + live price | 75% | **Empty** (crypto default); stocks also get no live price → unrealized P&L = 0 |
| `/trades` Trades | Filterable history + CSV export | 80% | **Empty** (crypto default); UI is `pair`-shaped, not `symbol` |
| `/analytics` Analytics | KPI grid + 4 charts | 85% | **All zeros** (crypto default) |
| `/sentiment` Sentiment | News/buzz/narratives | 85% | **Empty** (standalone runtime) + crypto-only sources |
| `/decisions` Decisions | LLM decision log + adjustments | 80% | **Works** (DB-backed) — rich, but crypto-centric cycles, legacy `provider` col |
| `/risk` Risk & Halt | Kill-switch + portfolio risk + reconcile drift + backups | **90%** | Halt control works; risk snapshot **empty** (standalone); backups stub `[]` |
| `/insights` Insights | Model-health tiles | 85% | **Empty** (standalone hub) + crypto/ML-oriented |
| `/observability` Observability | Cycle latency + LLM tokens | 85% | Works (log-parsed); "Cost" label shows tokens, **no $** |
| `/system` System | Health + WS + config dump | 75% | Health hardcoded; config dumps `crypto_*` only; WS health crypto-only |

### Backend surface

- **~90 endpoints wired, ~24 consumed by the SPA.** No 404 risk (every SPA
  call resolves), but a large dead-API surface: the **entire admin control
  plane is built and unsurfaced** — `admin.py` (9), `admin_config.py` (7),
  `admin_trades.py` (3), `admin_halal.py` (6) = **25 mutation endpoints with
  no UI** (halt/resume, pause pair, cancel/close orders, edit SL/TP,
  purification, prompt/AB tuning). Plus `research.py`/`research_jobs.py`
  (9, backtest queue), 9 unused insight tiles, and misc.
- **Stubs/misleading:** `/api/health` is hardcoded `{status:"running"}`
  (never reflects real health); `/api/system/backups` hardcoded `[]`;
  `/api/system/status` carries stock cadence + classifier health but the SPA
  reads the *crypto* interval field.
- **Auth:** reads fully open; mutations gated by constant-time `X-Trader-Token`
  (+ `X-Trader-Confirm` for destructive) with no token ⇒ safe read-only mode.
  Every mutation audited to `web_actions`. **Caveats:** WebSockets bypass the
  auth/audit middleware entirely; the SPA catch-all serves files with no
  path-traversal guard; `/api/config/schema` enumerates env-var names
  (values not returned). All low-risk on loopback, real risk if ever tunneled.
- **Dead code:** 33 stale `.pyc`-only modules in `web/__pycache__/` (removed
  SaaS-platform surface: `billing_state`, `marketplace`, `robo_advisor`,
  `kyc`, `soc2_readiness`, …) — safe to delete.

### Cross-cutting frontend issues

- **No error states** on ~9/12 pages — a 500 is indistinguishable from empty
  or a perpetual "Loading…". No error boundary.
- **Dark-only, no theming** — `app.css` `@theme` tokens are dark-only; chart
  colors are hardcoded hex (won't follow any theme). 0 `prefers-color-scheme`
  / `dark:` / `data-theme`.
- **Inconsistent loading** (skeletons vs bare "Loading…") and **duplication**
  (chart tooltip/colors copy-pasted across 5 charts; `usd()`/`pct()` helpers
  re-defined in pages instead of `lib/utils`).
- **Dead deps/code:** `lightweight-charts` imported nowhere; `lib/utils.pnlBg`
  unused; `provider`/`by_provider` columns vestigial (single GLM provider now).
- **A11y gaps:** status dots convey state by color only; expandable Decisions
  rows lack `aria-expanded`/button semantics; nav lacks `aria-current`.
- Polling cadences (React Query `refetchInterval`) are sensible
  (positions 5s, trades 15s, analytics 30s, health 10s). `refetchOnWindowFocus`
  is globally off despite a hook comment claiming focus-refetch.

---

## Roadmap

Ordered by operator value. Phase 0 is correctness (the dashboard currently
shows the wrong/empty data); later phases add the surfaces a stock day-trader
actually wants. All of this is **safe, non-trading work** (frontend + read
APIs + deployment) — none of it touches live sizing, `core/fills.py`, or
reconcile.

### Phase 0 — Make it show the stock bot at all (correctness; do first)
- [ ] **Global Stocks/Crypto market switch.** Thread a `market` param through
      `api/trades.ts`, `positions.ts`, `analytics.ts`, `pnl` + their hooks,
      and add a toggle in `Layout` (persist to `localStorage`/URL). Since the
      bot is stocks-only today, the fast interim fix is to default the SPA to
      `market=stocks`. Fixes Dashboard / Positions / Trades / Analytics in one
      change. **Highest ROI item on this list.**
- [ ] **Decide the deployment model** (operator call — see below). Either
      co-host the web app inside the `trader-stocks` process via
      `attach_to_app` (unblocks risk/sentiment/insights/live-cycle/backups), or
      make those endpoints DB-backed so the standalone `trader-web` container
      works. Until this is resolved, Sentiment / Risk-snapshot / Insights /
      live prices are structurally empty.
- [ ] **Live stock quotes for positions.** `positions.py:55-70` returns
      `current_price = entry`, `unrealized_pnl = 0` for stocks. Add a REST
      quote enrichment (Alpaca) so open-position P&L is real.
- [ ] **Fix stubs / mislabels:** real `/api/health`; render stock cadence +
      classifier health from `system/status` (SPA reads the crypto interval);
      make `/api/system/backups` real or remove the panel.
- [ ] **Reshape Trades/Positions for stocks:** `symbol` (not `pair`) column +
      filter labels; CSV export uses stock fields (`symbol`, `filled_price`,
      `exit_reason`).

### Phase 1 — Reliability & polish (cheap, high-signal)
- [ ] Render `isError` on every page + a global React error boundary.
- [ ] Standardize on skeleton loaders; drop bare "Loading…" text.
- [ ] Centralize chart theme (tooltip style + color ramp) into shared
      constants tied to CSS tokens; delete duplicated hex.
- [ ] Remove dead weight: `lightweight-charts` dep, `pnlBg`, legacy
      `provider`/`by_provider` columns, and the 33 stale `.pyc` modules.
- [ ] Fix the Observability "Cost" tile to show real $ (wire
      `/api/metrics/llm` cost); a11y pass on nav / status dots / expanders.

### Phase 2 — Stock-operator value surfaces (the real gaps)
- [ ] **Guard / rejection visibility panel** — surface `cycle.no_action` +
      the rejection reason (20% concentration cap, recent-close cooldown,
      SL re-entry gate, min-notional, halal fail). Today the operator can't
      see "the bot wanted to buy X but was blocked by Y" without reading logs.
      **Highest-value new panel.** Needs a small backend endpoint (parse from
      decisions/logs or a rejections view).
- [ ] **True equity curve** from Alpaca account equity (mark-to-market incl.
      open positions), not the current cumulative-`realized_pnl` proxy. The
      `DailyPnl` type already carries start/end equity — it's just unused.
- [ ] **Per-symbol stock P&L** (realized + unrealized) — a stock analogue of
      the crypto `PairBreakdown`.
- [ ] **Fill-quality / slippage panel** — the `trades` table already tracks
      `submitted_at`/`filled_at`/`filled_price`/`paper_slippage_pct`; expose
      submitted-vs-filled and slippage. Also surfaces the cold-start
      fill-timeout pattern seen at the open.
- [ ] **Zoya halal-screen status page** — pass/fail/reason per symbol. Halal
      is non-negotiable; `halal_compliance.py` (AAOIFI summary) and
      `admin_halal` sector-allocation are built and unconsumed — wire them.
- [ ] **Market-hours / next-cycle banner** — open/closed, last cycle time,
      next scheduled cycle (stock bot is a 15-min market-hours cron).
- [ ] **Reconcile-drift annotation** — label the known ~100% stock drift as
      "ledger-only / expected" (per OPERATOR_CONTEXT) so the red % doesn't
      false-alarm.
- [ ] LLM $/day cost figure somewhere prominent (~$0.05/day today).

### Phase 3 — Operator control plane (decide: surface or delete)
- [ ] Triage the 25 built-but-unsurfaced admin mutation endpoints. For a
      single-user paper bot the useful ones are: force-close position, cancel
      order(s), edit SL/TP, and pause/resume — surface these behind the
      existing `X-Trader-Token` + confirm flow. Delete or shelve the rest
      (purification mark-paid, prompt/AB tuning) unless wanted.
- [ ] Optionally surface the backtest job queue (`research_jobs.py`, fully
      built, no UI) as a research page.

### Phase 4 — Platform hygiene (lower priority)
- [ ] Light/dark theming (`data-theme` + token overrides); today it's
      dark-only.
- [ ] WebSocket auth (currently `/ws/*` bypass the auth/audit middleware).
- [ ] Path-traversal guard on the SPA catch-all (`Path.resolve()` +
      `is_relative_to`).
- [ ] Consolidate the two duplicate halal-explain endpoints.

---

## Operator decisions this roadmap needs

1. **Deployment model** — co-host the web server inside the stocks-bot process
   (richest data, one container) vs keep `trader-web` standalone and make its
   data fully DB-backed (simpler ops, but needs the runtime/hub endpoints
   reworked). This gates Phase 0.
2. **Single-bot vs dual-asset** — is the dashboard stocks-only (simplest:
   default everything to stocks, drop the crypto-first framing) or a true
   dual-asset tool with a switch? Crypto is off (empty Binance keys) today.
3. **Admin control plane** — surface the useful mutations, or delete the
   25-endpoint surface as dead weight?

---

*Companion docs: `docs/ROADMAP.md` (bot/strategy backlog),
`docs/OPERATOR_CONTEXT.md` (working agreement + known operator-gated issues),
`docs/ARCHITECTURE.md`.*
