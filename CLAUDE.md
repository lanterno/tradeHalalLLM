# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LLM-powered halal day-trading bot for **US stocks** (Alpaca paper trading via MCP). Python 3.14+, managed with `uv`. Single developer, working directly on `main`. Paper only, until a strategy passes the written capital gates (see below).

**Crypto trading was abandoned on 2026-10-01**: its code was deleted then, and its tables dropped on 2026-10-08 (migration `732583ae4034`).

**Read `docs/OPERATOR_CONTEXT.md` first.** It holds the non-code-derivable context: the working agreement, the stocks strategy intent (**fast in, slow out**), why the sole LLM provider is GLM-5.2 via OpenRouter (don't undo it; the bot won't start without `GLM_API_KEY`), operator-gated issues you can't fix in code (reconcile drift and the destructive fix-drift tool; don't touch `_aggregate_stocks_positions`), and the `src/halabot` engineering lessons (validate every edge with `halabot backtest` on disjoint OOS windows; the engine is shadow-only and never trades).

The roadmap is `docs/MODERNIZATION_PLAN.md` with its evidence in `docs/assessment/2026-10-01/`. Both are **kept local on purpose and not committed**: the repo is public and they map open weaknesses. Read them, update them, but don't `git add` them.

## Common commands

```bash
just dev                # uv sync --extra dev --extra all
just test               # pytest (needs Postgres on localhost:5433; see Database)
just lint / just format # ruff (one version: the lock's, also used by the pre-commit hook)
just typecheck          # mypy strict over the gated packages (pyproject [tool.mypy] files)
just precommit          # every pre-commit hook over every tracked file

# The deployed fleet (a Hetzner server; docs/DEPLOY.md)
just up                 # postgres + migrate + stocks + shadow + web, in docker
just down / build / rebuild / health / backup <dir> / docker-status
just docker-logs [svc]  # follow one service (default trader-stocks)

# Operator
halal-trader halt --reason "..."   # kill-switch: bots refuse new entries (monitor still exits)
halal-trader resume / halt-status
halal-trader db migrate | current | stamp head | revision -m "..."
halal-trader recommend [--show|--scorecard]   # advisory daily pick, never trades
halabot backtest ... / halabot ab-report      # shadow engine research tools
```

Every compose recipe goes through `compose` in the justfile (`--env-file .env`, project `halabot`). The compose file binds every port to 127.0.0.1 (dashboard 8082, Postgres 5433): on the public server Docker's port rules bypass ufw, so never publish a port on 0.0.0.0. The host side (bootstrap, the nightly off-site backup, health alerts, systemd units) is `infra/server/`. The previous machine was lost with its database and `.env` in 2026-10; a backup counts only once restic has it off the server (`backup.offsite` heartbeat).

**Database**: Postgres 16 + pgvector, Alembic is the single schema authority (`init_db()` refuses to start on a wrong revision; it never runs DDL). The models must match what the migrations build (`tests/test_alembic_migrations.py` compares them): an index or constraint goes in both the model and its migration, or the next `--autogenerate` proposes dropping it. Tests use per-worker `halal_trader_test*` databases on the same server. `tests/conftest.py` refuses any database name that isn't disposable, runs tests without the operator's `.env` (`HALAL_TRADER_ENV_FILE`), and blocks outbound network (`TEST_ALLOW_NETWORK=1` to opt out once).

Dashboard frontend: `cd dashboard && npm install && npm run build` (served from `dashboard/dist` by `web/app.py`); `npm run dev` for hot reload.

## Architecture

Authoritative diagrams: `docs/ARCHITECTURE.md` (where it and the code differ, trust the code).

**One live bot, one shadow engine, one dashboard; three containers, one database.** `trading/scheduler.py:TradingBot` (APScheduler cron, 15-min cycles in market hours) drives `TradingCycleService` → `TradingStrategy` (one GLM tool call) → `TradeExecutor` (Alpaca via the MCP stdio subprocess). Between cycles, `StockPositionMonitor` enforces SL/TP and trailing stops every 30 s, and `StockNewsEventReactor` can place half-size "fast in" momentum entries. `src/halabot` runs alongside as `halabot shadow` and only logs proposals. The web (`web/app.py`) is a separate process: **the database is the only contract between them**. In-process state (`RuntimeView`, `EventBus`) does not reach the web.

**Hex-ish layering.** `domain/ports.py` holds the Protocols (`Broker`, `ComplianceScreener`, `LLMBackend`, …); adapters live in `mcp/` (Alpaca), `halal/` (the strict screen's gate + cache), `core/llm/` (GLM). Shared maths lives in `signals/` (indicators, multi-timeframe) and `portfolio/` (risk engine, performance analytics).

**Single LLM provider: GLM-5.2** via `core/llm/factory.py:create_llm`, OpenAI-compatible (OpenRouter by default; `FallbackLLM` chains a second endpoint if `GLM_FALLBACK_BASE_URL` is set). Strips `<think>…</think>`. A tool call with unparseable arguments, or missing the schema's required keys, is a **failed** call recorded as such, never a silent empty plan.

## Invariants (pinned by `tests/invariants/`)

- **Halal at the order boundary.** Every BUY passes `TradeExecutor._execute_buy`, which refuses any symbol the screener does not hold halal and **fails closed** if the screen can't be read. The prompt's symbol list is advisory; this gate is the rule. Every new order path must go through it.
- **Long-only.** No short action exists; sells are clamped to the broker-held quantity (`core/long_only.py`).
- **Kill-switch first.** `BaseCycleService.run_cycle` checks `core/halt.is_halted` before anything else; the reactor checks it too. The core (`core_executor.run`, shared by the 15:40 job and `core run`) buys nothing while halted but still sells holdings that fail the halal screen, and alerts.
- **Reactor entries obey the cycle's gates**: daily loss limit, the risk engine's last halt, max simultaneous positions. All fail closed.
- **Daily loss limit is anchored to the day's first equity** (`daily_pnl`), not to whatever equity a restarted process sees.
- **halabot execution stays dormant** (`tests/halabot/execution/test_dormant.py`; keep `ENGINE_LIVE` unset).
- **The stock process never loads `binance`** (`tests/test_no_crypto_imports.py`).

## Conventions / gotchas

- **Settings are a singleton** (`config.py:get_settings()`); pass `settings` by DI, never construct `Settings` elsewhere. **Only a group's `ENV` fields come from the environment**: secrets, endpoints and the operator's switches. Everything else is a decided value in `config.py`; make a decision rather than add a variable. `.env.example` lists exactly the `ENV` fields (`tests/test_settings_parity.py`).
- **The broker server is frozen.** The image runs alpaca-mcp-server from `infra/alpaca-mcp-server.txt`, an exact `pip freeze` installed at build time; host runs use `uvx alpaca-mcp-server@<AlpacaSettings.mcp_server_version>`. Never let it float: an unpinned server, or a pinned server with floating dependencies, has broken the bot at startup. Upgrade by re-freezing, bumping the version default (a test keeps the two in sync), and smoke-testing a live cycle.
- **Liveness is in the database.** `core/heartbeat.py`: the bot beats `stock.process` (60 s), `stock.cycle`, `stock.monitor` and one row per daily job; the shadow beats `shadow.process`. `assess()` judges loops by age and daily jobs by the trading calendar (`DAILY_JOBS`); the web's watchdog (`web/watchdog.py`) alerts on Telegram when one goes stale, `/api/health/bot` is 503 when the bot is, and compose healthchecks use `core/healthcheck.py`. Add a beat (and a `DAILY_JOBS` entry for a daily job) for any new long-running component. A restarted bot catches up the daily jobs it missed (`plan_catch_up`).
- **Secrets per container.** One `.env`; `infra/docker-compose.yml` blanks, per service, the secrets that process does not read. A new secret goes into the blank lists of every service that does not need it.
- **Signals.** The bot and the shadow install SIGTERM/SIGINT handlers (they are PID 1 in docker); compose gives them `init: true` and a 30 s grace period. Keep `shutdown()` fast.
- **Deploy live-path changes outside US market hours** (before 09:00 or after 16:00 ET), rebuild with `just build`, recreate only what changed, and watch the first cycle: the failure mode is silent no-action, not a crash.
- **Async repository.** `db/repository.py` is async; one `Repository(engine)` per process. Don't open engines per cycle.
- **Structured events.** `extra={"event": events.X, ...}` with constants from `core/events.py`; correlation ids come from `core/observability.py`.
- **Operator alerts** go through `AlertSink.notify(error_type, details)` (`notifications/telegram.py`), which rate-limits per type.
- **Fill confirmation.** `core/fills.py:confirm_alpaca` fills `submitted_at`/`filled_at`/`filled_price`/`filled_quantity`; never conflate submitted with filled.
- **CLI lazy imports.** Heavy modules are imported inside command functions so `--help` stays fast.
- **Optional extras** (`[ml]`, `[dashboard]`) must degrade gracefully when absent.
- **PEP 758** `except A, B:` (no parentheses) is valid Python 3.14 and what ruff formats to. Don't "fix" it.

## Product strategy

A small home-built bot competing with institutions, now developed full-time. Edge comes from new tech (LLMs, open models), alternative data and fast iteration, **measured** on one research harness before any capital moves. Halal compliance applies to every feature regardless of profitability.
