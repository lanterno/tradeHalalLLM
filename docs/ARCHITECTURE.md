# Architecture

Halal Trader as it runs today (rewritten 2026-10-01, after the crypto bot,
the dormant ML stack and the inert LLM modes were removed). Paper trading
only; US stocks; long-only; halal-screened.

## Processes

Three long-running processes and one database. They share nothing in
memory: **the database is the only contract between them.**

```
                 ┌───────────────────────────── Postgres 16 + pgvector ─────────────────────────────┐
                 │ trades · daily_pnl · llm_decisions · halal_cache · kill_switch · heartbeats       │
                 │ broker_activities · broker_equity · llm_spend · hb_* (shadow) · daily_bars · ...  │
                 └───────▲───────────────────────────▲───────────────────────────────▲──────────────┘
                         │                           │                               │
   trader-stocks  halal-trader start          trader-shadow  halabot shadow    trader-web  uvicorn create_app
   (the live bot: cycle, monitor,             (belief engine; proposes,         (FastAPI + React SPA;
    news reactor, after-close jobs)            never trades)                     reads the DB)
                         │                           │
                         └──── Alpaca (paper) via the alpaca-mcp-server subprocess ────┘
```

`infra/docker-compose.yml` runs them, plus a one-shot `trader-migrate` that
brings the schema to the Alembic head before anything starts.

## The live bot (`trading/`)

`trading/scheduler.py:TradingBot` is the composition root. It runs an
APScheduler cron in US/Eastern time and two supervised background tasks.

| Job | When (ET, Mon-Fri) | What |
|---|---|---|
| pre-market | 09:00 (and at startup) | refresh the halal cache; record the day's starting equity (restart-safe) |
| recommendation | 09:05 | advisory "stock of the day" (never trades) |
| trading cycle | 09:30-15:45, every 15 min | one LLM decision, then execution |
| end of day | 15:50 (12:50 on half days) | flatten cycle positions; reactor positions may hold overnight |
| broker ledger | 16:30 | copy Alpaca's fills and equity; reconcile; alert on drift |
| background | always | `StockPositionMonitor` (SL/TP/trailing every 30 s) and `StockNewsEventReactor` (Finnhub every 60 s), each restarted if it dies |

### A trading cycle

`TradingCycleService` (`trading/cycle.py`) on the `BaseCycleService`
template (`core/cycle.py`):

1. **Kill-switch first** (`core/halt.py`); then the live-mode tripwire.
2. **Daily loss limit** (`trading/portfolio.py`), anchored to the day's
   first equity in `daily_pnl`.
3. **Inputs**: the halal universe (`halal/cache.py`), account and positions,
   snapshots and 60 days of bars per symbol, the portfolio risk engine
   (`portfolio/risk.py`), multi-timeframe alignment (`signals/timeframes.py`),
   catalysts (FRED, SEC 8-K, Fed speak), the newest ticker-tagged headlines,
   recent performance (`portfolio/analytics.py`) and active self-review
   adjustments. Stages live in `core/cycle_stages.py`.
4. **One LLM call** (`trading/strategy.py`): GLM-5.2 with a forced
   `submit_decisions` tool call. A reply missing required fields is a
   *failed* call, recorded and alerted, never a silent hold.
5. **Execution** (`trading/executor.py`), then a heartbeat.

### Orders: where the invariants live

Every BUY passes `TradeExecutor._execute_buy`, which applies, in order:
halal membership (fails **closed**), the market-close lockout, re-entry
cooldowns, buying power, the per-name and sector caps. The news reactor
enters through the same method and additionally checks the daily loss
limit, the risk engine's last halt and the position cap. Sells are clamped
to the broker-held quantity (`core/long_only.py`: no shorts); LLM sells
cannot close news-momentum positions (`EntryType.REACTOR_MOMENTUM`), which
only the monitor's rule-based exits close. `tests/invariants/` pins each of
these where it is wired.

### The broker boundary

`mcp/client.py` launches `alpaca-mcp-server` over stdio. In the image it is
a build-time install of the frozen closure in `infra/alpaca-mcp-server.txt`;
on a host it is `uvx alpaca-mcp-server@<ALPACA_MCP_SERVER_VERSION>`. Never
unpinned: an upstream change has broken the bot at startup more than once.
Recorded payloads of the pinned release live in `tests/fixtures/`.
Read-only account data for the books comes from Alpaca's REST API
directly (`execution/alpaca_rest.py`).

`execution/alpaca_broker.py` implements the same `Broker` port over REST,
with no subprocess: timeouts, strict parsing, and a `client_order_id` on
every order so a timed-out submission is looked up before it is retried.
`ALPACA_BROKER_ADAPTER=rest` selects it; the default stays `mcp` until
`halal-trader broker compare` has agreed through live sessions.

## Books, health, spend

- **Ledger** (`execution/ledger.py`): Alpaca's account activities and daily
  equity are the books of truth; the bot's `trades` are reconciled to them;
  performance is measured from broker equity (`halal-trader ledger`).
- **Heartbeats** (`core/heartbeat.py`): process, cycle, monitor and ledger
  beats in the `heartbeats` table. `/api/health/bot` is 503 when the process
  is stale, or when no cycle completed recently during market hours.
- **LLM spend** (`core/llm/spend.py`): every GLM call reports its cost to
  `llm_spend` and asks first; one daily cap across processes; observe or
  enforce mode.
- **Locks**: a Postgres advisory lock allows one stock bot per database.

## The shadow engine (`src/halabot`)

A Sense → Understand → Convict → Act engine around a persistent per-asset
`BeliefState` (`hb_*` tables). It runs in shadow: it records proposals and
outcomes, and its `execution/` layer is never imported by the running
engine (a fresh-interpreter test enforces this). Its halal verdicts come
from the same universe, refreshed hourly. See `docs/REARCHITECTURE.md`.

## The dashboard (`web/`, `dashboard/`)

FastAPI routes read the database through `DashboardContext`; the only
mutation the SPA makes is Halt/Resume (`/api/system/halt`). Fields that
assumed the bot shared the web's process (`RuntimeView`) are empty in this
deployment and are being replaced by database reads.

## Research (plan Phase 3)

Separate from the live path; nothing here trades.

- `data/`: 10 years of daily SIP bars (raw and adjusted) for a liquidity
  universe plus SPY/QQQ/SPUS/HLAL, monthly bars for every listed and
  delisted stock (the point-in-time universe), and annual SEC
  fundamentals, all from free sources (`halal-trader data`).
- `compliance/`: an in-house AAOIFI-style screen from SEC data with a
  point-in-time verdict history, validated against SPUS and HLAL holdings
  from their N-PORT filings (`halal-trader compliance`). Delisted tickers
  are matched to their filer by name. Not yet a gate.
- `research/`: point-in-time backtests (eligible = liquid then x halal
  then) and paper-forward books. Every backtest is a trial in
  `quant_trials`, judged by the Deflated Sharpe Ratio of its returns over
  SPUS across all trials (`halal-trader research`, `quant trials --prefix
  research.`).
- `quant/`: price-range bands and calibration behind the recommendation.

## Configuration

`config.py:Settings`, loaded once from the environment and `.env`; every
field is documented in `.env.example` (a test enforces both
directions). Secrets never enter the repo.

## Schema

Alembic is the only schema authority for the bot's tables; `init_db()`
refuses to start on a wrong revision and never runs DDL. The shadow's `hb_*`
tables are created by halabot itself.
