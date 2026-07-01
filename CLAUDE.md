# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LLM-powered halal day-trading bot for **stocks** (Alpaca paper trading via MCP) and **crypto** (Binance testnet/prod). Python 3.14+, managed with `uv`. Single-user, paper/testnet only — never real money.

## Common commands

Use the `justfile` recipes (each wraps `uv run halal-trader …`). `just` with no args lists recipes.

```bash
just install            # uv sync
just dev                # uv sync --extra dev --extra all  (ml + sentiment + dashboard)
just test               # pytest (asyncio_mode=auto, testpaths=tests)
just lint               # ruff check src/ tests/
just format             # ruff format + ruff check --fix
just typecheck          # mypy (strict on domain/ + core/ only)
just precommit          # run all pre-commit hooks against every tracked file
just precommit-install  # one-time: wire pre-commit into .git/hooks/

# Stocks
just stocks             # halal-trader start  (APScheduler, market hours)
just stocks-once        # halal-trader start --once
just status             # halal-trader status

# Crypto (24/7 asyncio loop)
just crypto             # halal-trader crypto start
just crypto-once        # halal-trader crypto start --once
just crypto-status      # Binance balances + ticker prices
just crypto-stats       # win rate / profit factor / drawdown
just crypto-screen      # refresh CoinGecko-based halal list

just dashboard          # FastAPI + React SPA on :8082 (serves dashboard/dist)
just logs / logs-tail   # pretty-print JSON log files (cycle_id + event tags)
just pg-up / pg-down    # bring the Postgres+pgvector container up/down (port 5433)
just db-reset           # ⚠ DROP+CREATE halal_trader and re-run migrations
just test-db-reset      # drop the test database (safe; recreated by next pytest run)

# Operator
halal-trader halt --reason "..."     # engage kill-switch (bots refuse new entries)
halal-trader resume                  # disengage kill-switch
halal-trader halt-status             # show current state + last audit row
halal-trader db migrate              # apply pending Alembic revisions
halal-trader db current              # show current vs head revision
halal-trader db stamp head           # one-time adopt a pre-Alembic DB
```

**Database**: Postgres 16 + pgvector. Bring up the container with `just pg-up` (uses `infra/docker-compose.yml`); the bot connects to `localhost:5433` per `DATABASE_URL` in `.env.example`. The test suite uses a separate `halal_trader_test` database that `tests/conftest.py` recreates per session and TRUNCATEs per test — running tests requires the same Postgres container reachable on `localhost:5433`.

Run a single test file/case: `uv run pytest tests/test_crypto_executor.py -k test_name`.

Backtest: `uv run halal-trader crypto backtest --pair BTCUSDT --candles 1000 [--llm --cycle-interval 5]`. The `--llm` mode caches results in `models/llm_backtest_cache.json` keyed by prompt hash, so repeated runs skip LLM calls.

Dashboard frontend: `cd dashboard && npm install && npm run build` (Vite output goes to `dashboard/dist`, which `web/app.py` serves). `npm run dev` for hot reload.

DB migrations: Alembic is the single schema authority. `init_db()` opens the engine and verifies the DB is at head; it never runs DDL. Use `halal-trader db migrate` (forwards), `halal-trader db current` (status), `halal-trader db stamp head` (one-time adoption of a pre-Alembic DB), `halal-trader db revision -m "..."` (new migration). The bot refuses to start with `SchemaError` if the DB is at the wrong revision.

## Architecture

Authoritative diagrams + tables live in `docs/ARCHITECTURE.md`. Key points that span multiple files:

**Two parallel bots, shared infrastructure.** The stock bot (`trading/`) and crypto bot (`crypto/`) are independent composition roots that both extend `core/scheduler.py:BaseTradingBot`. They share the LLM abstraction (`core/llm/`), DB layer (`db/`), domain models/ports (`domain/`), and `Settings` (`config.py`). Don't merge their cycle logic — they have very different cadences (15min cron vs 60s asyncio loop) and execution paths (Alpaca MCP subprocess vs `python-binance` async REST + WebSocket).

**Hex-ish layering.** `domain/ports.py` defines `Protocol`s (Broker, LLMProvider, ComplianceScreener, …); concrete adapters live under `mcp/` (Alpaca MCP stdio client), `crypto/exchange.py` (Binance), `core/llm/` (GLM-5.2 over OpenAI-compatible endpoints + `FallbackLLM` chain), `halal/` (Zoya), `crypto/screener.py` (CoinGecko). When adding a provider, implement the port — don't import concrete classes into cycle/strategy code.

**Crypto cycle = `CryptoCycleService.run_cycle()`** in `crypto/cycle.py`. Each cycle: refresh symbol filters → check `PortfolioTracker.should_halt_trading()` → fetch halal pairs → pull klines (throttled, 5 concurrent) via WebSocket buffer → compute indicators → run `PortfolioRiskEngine` (correlation/heat/drawdown) → call LLM with the full prompt context (see ARCHITECTURE.md "What the LLM Receives") → regime gate → `CryptoExecutor.execute_plan()` → snapshot indicators for filled buys. The whole cycle is wrapped in `asyncio.wait_for(interval * 2)` so a stuck cycle doesn't block the bot.

**Position monitor is independent of the cycle.** `crypto/monitor.py:PositionMonitor` runs as a background task every `crypto_monitor_interval` seconds (default 2s), enforcing SL/TP/trailing stops via WebSocket prices. It coordinates with the executor via a shared `exiting_pairs` set to prevent concurrent buy/sell on the same pair. After closing a trade, it calls `retrainer.on_trade_closed()` to label the indicator snapshot for ML training. **Don't put SL/TP enforcement in the cycle** — the cycle's cadence is too slow.

**News reactor can preempt the cycle.** `sentiment/events.py:NewsEventReactor` polls CryptoPanic every 30s; on a high-impact event, it triggers an emergency mini-cycle outside the normal loop. Be mindful of this when adding state that assumes cycles run on a fixed cadence.

**Single LLM provider: GLM-5.2.** There is no provider switch anymore. `core/llm/factory.py:create_llm(settings)` builds a `GLMLLM` (`core/llm/glm.py`) that speaks OpenAI-compatible endpoints — OpenRouter by default (`GLM_API_KEY` + `GLM_BASE_URL`, model `LLM_MODEL=z-ai/glm-5.2`). If `GLM_FALLBACK_BASE_URL` (+ `GLM_FALLBACK_MODEL`/`GLM_FALLBACK_API_KEY`) is set, the primary is wrapped in `FallbackLLM`, which chains the two GLM endpoints (e.g. OpenRouter primary, Z.ai direct fallback; exponential backoff 60s→30min after all fail). The endpoint strips `<think>…</think>` reasoning blocks — keep that behavior when adding a new endpoint.

**Self-improvement.** After each crypto cycle, `crypto/self_improve.py` lets the LLM tune `max_position_pct`, `stop_loss_pct`, `take_profit_pct` within bounded ranges. Changes below `1e-6` are silently dropped. Records `StrategyAdjustment` rows.

**ML retraining loop.** Buys record an `IndicatorSnapshot` (RSI, MACD, volume ratio, ATR, BB position). When the position closes, `ml/retrainer.py:RetrainingScheduler.on_trade_closed()` labels the snapshot with `return_pct` and (every 20 trades) retrains the IsolationForest anomaly detector + signal classifier. The anomaly detector also supports incremental updates (`add_sample` / `auto_train`) — don't re-read the full DB on every cycle.

## Conventions / gotchas

- **Settings are a singleton.** `config.py:get_settings()` caches a `Settings()` instance. Don't construct `Settings` directly elsewhere — pass `settings` via DI.
- **DB connection.** `Settings.database_url` is the canonical async URL (asyncpg); `Settings.database_url_sync()` derives the matching `+psycopg` URL for Alembic / sync admin scripts. Never hardcode a URL — always go through `get_settings()`.
- **Async repository.** `db/repository.py` is fully async; `Repository(engine)` is constructed once in `BaseTradingBot.initialize()` and shared. Don't open new engines per cycle.
- **Three logger sinks.** `logging.py` configures Rich console + JSON `logs/halal_trader.log` (rotated) + `logs/error.log`. The `just logs*` recipes parse the JSON format via `scripts/format_logs.py` — don't switch to plain-text logging without updating those.
- **Structured event logging.** Use `extra={"event": events.X, ...}` with constants from `core/events.py`. `cycle_id`/`monitor_id`/`request_id` ContextVars in `core/observability.py` are auto-attached to every JSON record by `ObservabilityFilter`. `BaseCycleService.run_cycle()` already wraps each cycle in `cycle_context()` — don't manage these manually unless you're starting a sub-scope (e.g. per-trade `monitor_context()` in `crypto/monitor.py`).
- **Operator alerts.** Errors that need human attention go through `AlertSink.notify(error_type, details)` in `notifications/telegram.py`, NOT directly via `notifier.notify_error`. The sink rate-limits per `error_type` (15-min sliding window). The cycle's `cycle.failed` exception path already fires it; surface new failure modes by adding an `extra={"event": ...}` log + a sink call site.
- **Kill-switch is a first-class gate.** `BaseCycleService.run_cycle` checks `core/halt.is_halted(engine)` BEFORE any other logic. Use `halal-trader halt --reason "..."` to engage; the monitor's per-trade SL/TP loop still runs (closing risk is preferred to holding under failure).
- **Fill confirmation.** `core/fills.py:confirm_binance` (immediate-fill response parser) and `confirm_alpaca` (poll loop) populate `submitted_at`/`filled_at`/`filled_price`/`filled_quantity` on every trade row. Don't conflate `submitted` with `filled` anywhere downstream — `core/reconcile.py` aggregates filled quantities only.
- **CLI lazy-imports.** Heavy modules (binance, fastapi, ml) are imported inside command functions in `cli.py` so `--help` stays fast. Keep that pattern when adding commands.
- **Halal compliance is non-negotiable.** Every new tradable asset path must go through the relevant screener (Zoya for stocks, `crypto/screener.py` for crypto). See `.cursor/rules/project-strategy.mdc`.
- **Optional extras.** `[ml]`, `[sentiment]`, `[dashboard]`, `[fingpt]` extras gate their respective imports. Code that uses them must degrade gracefully when the extras aren't installed (see `cli.py:dashboard` for the pattern).
- **Binance error codes.** `-1013` (invalid quantity) and `-2010` (insufficient balance) are treated as rejections, NOT circuit-breaker errors. `-1003` triggers ~30s rate-limit backoff. The per-pair circuit breaker is for *unexpected* errors only.
- **Min buy notional is $50.** Below that the executor refuses the order. Many tests assume this — don't lower it without checking `tests/test_crypto_executor.py`.

## Product strategy (from `.cursor/rules/project-strategy.mdc`)

This is a small home-built bot competing against institutions. The edge is aggressive adoption of new tech (LLMs, HuggingFace models), alternative data (Reddit, news APIs), and rapid iteration. When choosing between conservative and aggressive approaches, lean aggressive. Always prefer integrating an existing OSS model or API over building from scratch. Halal compliance applies to every feature regardless of profitability.
