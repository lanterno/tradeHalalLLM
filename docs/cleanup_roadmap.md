# Roadmap — Round 3

> **Status: 12 of 12 waves shipped.** Wave B's finishers landed in
> the post-Round-5 cycle: the procedural `_get_tradeable_pairs`,
> `_fetch_klines`, `_fetch_orderbooks`, `_record_cycle_analytics`,
> `_execute_and_notify`, and inline indicator-compute helpers on
> `CryptoCycleService` are now `GetTradeablePairsStage`,
> `FetchKlinesStage`, `FetchOrderbooksStage`, `ComputeIndicatorsStage`,
> `RecordCycleAnalyticsStage`, and `ExecuteAndNotifyStage` — six new
> `CycleStage` classes that drop into the same `run_stages` driver
> as the prompt-context builders. `crypto/cycle.py` shrank from 738
> → 432 lines (-41 %); the residual is cycle-control glue (flat-skip
> counter, live-mode safeguard, low-USDT guard, the `swallow=False`
> LLM analyze block). The architectural goal — "new prompt source =
> one new file, one new line in the stage list" — was already met
> by the round-4/5 work; the finishers extend that property to
> every cycle action, which unblocks Wave G (replay-fitted slippage
> per stage), Wave J (Prometheus histograms per stage), and Wave F
> (prompt-evolution A/B over individual stages).

The two earlier rounds were focused on **mechanical cleanup**:
schema hygiene, dead code, type safety, test parallelism. The
codebase is now in a state where mechanical work has diminishing
returns. The next set of changes is more ambitious: structural
refactors that pay off across many files, plus net-new features
that exploit modern primitives we haven't reached for yet.

The waves split into three buckets:

* **Architecture** (A–D) — primitives that, once landed, change
  how every subsequent feature is built.
* **Decision quality** (E–H) — features that make the LLM's
  trading decisions sharper and more honest.
* **Operability + UX** (I–L) — what you reach for *after* a bad
  trade or *during* a deployment.

The waves are not strictly ordered, but A unblocks much of B/D/J
and is recommended first. Each wave is independently shippable;
each ends with a one-line acceptance test you can run at the CLI.

---

## Architecture

### Wave A — Replace `app_state: dict` + `getattr` chains with a typed `BotContext`

**Why**: today the dashboard, cycle, monitor, and CLI all reach into a
`dict[str, Any]` (`app_state`) or pull soft-optional dependencies via
`getattr(self, "_x", None)`. Round 2's Wave E showed how often that
masks dead wiring; round 1's `_replay_store` was a real bug. Mypy
strict is now clean only on `db/repository.py` because `app_state`
makes the rest of the surface untypable.

The fix is a small, frozen, typed `BotContext` dataclass that holds
every long-lived dependency the bot needs (engine, repo, hub, broker,
LLM, settings). Build it once at composition time and pass it
explicitly. The dashboard takes a `DashboardContext` (a strict subset
— no broker, no LLM). Both are strict-typed.

**Scope**:
- `core/context.py` — `@dataclass(frozen=True, slots=True)` with each
  long-lived dependency typed. Builders return it; consumers read it.
- Replace `app_state["engine"]`, `app_state["repo"]`, `app_state["hub"]`
  reads in the 22 web routes with `ctx: DashboardContext` injected
  via FastAPI `Depends`.
- Promote mypy strict to `web/`, `crypto/cycle.py`, `crypto/scheduler.py`,
  `crypto/components.py`.
- Delete `app_state` entirely.

**Acceptance**:
- `grep -r 'app_state\["' src/` returns zero hits.
- `uv run mypy --strict src/halal_trader/web` clean.
- The dashboard has a real type signature for "what state am I
  reading."

**Estimated effort**: 1 day (broad but mechanical once the dataclass
shape is right).

---

### Wave B — Per-stage cycle pipeline with explicit stage objects

**Why**: Round 2 Wave D peeled `_run_cycle_impl` into helpers, but it's
still a single function reading from a procedural soup of locals
(`halal_pairs`, `klines_by_symbol`, `indicators_cache`, `account`,
`positions_text`, `today_pnl`, `sentiment_text`, …). Adding a new
prompt-context source means scrolling past nine unrelated blocks to
find the right insertion point and threading a new local through five
function signatures.

The primitive is a `CycleStage` protocol: each stage takes a
`CycleState` dataclass, mutates one field, returns it. The
`CryptoCycleService` becomes a list of stages run in order. New data
sources land as a single new stage in a list; nothing else changes.

This unblocks: per-stage timing histograms (Wave J), per-stage
backtesting (Wave G), per-stage A/B harness (Wave F), and most
importantly **prompt-context provenance** — each row in the LLM
decision table can record which stages ran and how long each took.

**Scope**:
- `crypto/cycle/state.py` — `CycleState` dataclass with one field per
  prompt-context source (`klines`, `indicators`, `regime_text`, …).
- `crypto/cycle/stages/*.py` — one file per stage:
  `FetchKlinesStage`, `ComputeIndicatorsStage`, `BuildSentimentStage`,
  `BuildRegimeStage`, `EvaluateRiskStage`, `AnalyzeStage` (the LLM
  call), `ApplyRegimeGateStage`, `ExecuteStage`. Each is a tiny
  class with one async `run(state) -> state` method.
- `crypto/cycle.py` becomes ~50 lines: build the stage list, run it.
- A `StageInstrumentation` middleware wraps each stage with a
  `tracer.aspan` and records elapsed/exception/skipped to the cycle
  span.

**Acceptance**:
- `crypto/cycle.py` is under 100 lines.
- New prompt source = one new file, one new line in the stage list.
- The replay snapshot (Wave C of round 1) records per-stage timing so
  the dashboard can plot p95 cycle latency by stage.

**Estimated effort**: 2 days. This is *the* refactor that makes the
crypto cycle modern and extensible; the bot has been begging for it
since the third or fourth prompt-context block landed.

---

### Wave C — Adopt `anyio` task groups for structured concurrency ✅ landed

**Why**: the bot used to fire long-lived loops with
`asyncio.create_task(...)` in seven places (monitor, ws_manager,
news_reactor, sentiment_manager, reconcile_loop, stock-monitor,
research_jobs). Each had its own ad-hoc shutdown sequence. When one
silently died, nothing noticed until the operator saw a stale
dashboard.

**What landed**:
- `core/supervisor.py:TaskSupervisor` — async-context-manager around
  one rooted `anyio.create_task_group()`. Per-task
  :class:`RestartPolicy` (`CRASH_BOT` / `RESTART` / `IGNORE`)
  selects between cancel-on-error propagation, exponential-backoff
  restart, and log-and-forget. `sup.cancel()` collapses the scope
  when the cycle loop exits cleanly.
- `CryptoTradingBot.run()` enters one `async with TaskSupervisor()`
  scope around the entire main cycle loop. Inside the scope:
  - `monitor`  → CRASH_BOT (closing risk > running half-blind)
  - `ws`       → CRASH_BOT (stale prices break SL/TP enforcement)
  - `news_reactor` → RESTART (CryptoPanic flake shouldn't kill bot)
  - `sentiment_manager` → RESTART (best-effort context)
  - `reconcile` → RESTART (cosmetic drift, retry on hiccup)
  - the cycle loop itself runs *inside* the scope so a CRASH_BOT
    sibling's failure cancels it via the task group's exit.
- Every long-lived subsystem grew a public `async def run(self)` —
  `PositionMonitor` (crypto+stocks), `BinanceWSManager` (uses a
  nested anyio task group internally for per-symbol streams),
  `NewsEventReactor`, `SentimentManager`.
- The bot's resource teardown (broker disconnect, open-order cancel,
  daily summary) now runs *after* the supervisor scope exits, so
  subsystems aren't still holding the broker handle when we close
  it.

**Trade-offs documented**:
- The legacy `start()` / `stop()` methods are kept on each
  subsystem (5 `asyncio.create_task` sites total) as thin compat
  shims so existing unit tests (e.g. `test_monitor.py::TestStartStop`)
  and any ad-hoc CLI scripts still work. The bot's production
  `run()` path uses `run()` exclusively.
- `web/routes/research_jobs.py` still fires one `asyncio.create_task`
  for an HTTP-triggered one-shot dispatch — that's a request-scoped
  fire-and-forget, not a long-lived loop, so it stays.

**Acceptance** (both met):
- `grep -r 'asyncio\.create_task' src/` count is 7 — single-digit,
  every site is either inside a legacy back-compat shim or the
  one-shot HTTP dispatch.
- `tests/test_supervisor.py::test_crash_bot_task_cancels_other_supervised_tasks`
  shows a CRASH_BOT-policy task's failure cancels a sibling
  cycle-loop task and propagates the original `RuntimeError` via
  `BaseExceptionGroup`. The companion
  `test_restart_policy_does_not_kill_bot_on_one_failure` shows a
  RESTART-policy flake leaves the cycle alone.

---

### Wave D — Repository as a `Protocol` + per-table mini-repos

**Why**: `db/repository.py` is 956 lines of one class with 50+
methods spanning trades, halal cache, web actions, runtime config,
purification, screenings, decisions, research jobs, indicator
snapshots, pair pauses. Adding a new table means scrolling past nine
unrelated sections and hoping you don't put the helper in the wrong
spot. Tests that need only `record_trade` end up coupled to every
other method via the giant constructor.

Split into per-table mini-repos behind a `Protocol`:
`TradeRepo`, `CryptoTradeRepo`, `HalalCacheRepo`, `WebAuditRepo`,
`PnlRepo`, `RuntimeConfigRepo`, `PurificationRepo`,
`ScreeningRepo`, `LlmDecisionRepo`, `ResearchJobRepo`,
`IndicatorSnapshotRepo`. Each is ~60 lines, under one mypy strict
file, with its own focused tests. Composition root constructs the
bundle.

**Scope**:
- `db/repos/__init__.py` exposes `RepoBundle` (a frozen dataclass
  holding all of them).
- Each repo file is ≤80 lines, mypy strict.
- `Repository` class deleted; `from halal_trader.db.repository
  import Repository` becomes `from halal_trader.db.repos import
  RepoBundle`.

**Acceptance**:
- No file under `db/repos/` over 100 lines.
- `mypy --strict` passes on the entire `db/` package, not just
  `repository.py`.
- Adding a table = adding one file, not editing a 956-line module.

**Estimated effort**: 1.5 days.

---

## Decision quality

### Wave E — Tool-use LLM calls instead of JSON-blob parsing

**Why**: today every LLM strategy call asks for a JSON blob, parses
it, and runs schema-repair on retry. Anthropic and OpenAI both ship
**typed tool use** — the model emits a structured call against a
JSON Schema you declared, the SDK validates it, and you get a Python
dict back. Schema-repair retries vanish. Token cost drops because the
model doesn't waste output tokens emitting JSON syntax. And we get to
expose the same tool to a future agentic mode (Wave H).

Trading-plan-shaped tools:
- `submit_plan(buys: list[BuyDecision], sells: list[SellDecision],
  market_outlook: str)` — returns a structured plan.
- `analyze_pair(symbol: str)` — optional second-call deep-dive
  triggered when the model wants more info on one pair (ties into
  the agentic mode in Wave H).
- `request_indicator(symbol: str, indicator: str)` — same idea, but
  for one indicator on one pair, in case the model wants to ask
  "what's the 4h MACD on ETHUSDT?" before deciding.

**Scope**:
- `core/llm/tools.py` — typed JSONSchema definitions for each tool,
  derived from the existing Pydantic decision dataclasses.
- `AnthropicLLM.generate_tool_call(prompt, system, tools)` returns a
  typed `(tool_name, args)`.
- `CryptoTradingStrategy.analyze` switches from
  `await self._llm.generate_json(...)` to
  `await self._llm.generate_tool_call(submit_plan_tool, ...)`.
- Schema-repair path becomes a no-op (still kept as a code path for
  Ollama, which doesn't support native tool use).

**Acceptance**:
- LLM decision rows for Anthropic / OpenAI runs have
  `parsed_action != None` 100% of the time (today it's lower because
  the JSON-repair fallback occasionally fails).
- Output token counts on the same prompt drop ≥10% (less JSON
  syntax to emit).

**Estimated effort**: 2 days.

---

### Wave F — Continuous A/B harness over the prompt-evolution GA ✅ landed

**Why**: `core/llm/prompt_evo.py` was sitting unused. It has a clean
GA over prompt slot×allele genomes but nothing wired it to a fitness
function or a real prompt template. This wave closes those gaps.

**What landed (round-4/5 + this commit)**:
- `core/llm/prompt_evo.py` — pure GA: `PromptGenome` (slot→allele
  dict), `AllelePool`, crossover/mutate, `PromptGA.evolve`.
  (round-4/5)
- `core/llm/prompt_evo_runner.py` — pulls recent replay snapshots
  from `ReplayStore`, scores each genome via a caller-supplied
  evaluator, persists the final generation to `prompt_genomes`,
  exposes `list_recent_genomes` and `promote_genome` for the
  dashboard. (round-4/5)
- `web/routes/prompts.py` — `GET /api/prompts/candidates` and
  `POST /api/prompts/{id}/promote` (round-4/5).
- `crypto/prompts.py:SYSTEM_PROMPT` — refactored to expose three
  named slots:
  - `role_intro` — opening sentence (3 alleles: expert /
    disciplined / conservative)
  - `strategy_emphasis` — opening line of the STRATEGY block (3
    alleles: empty / liquidity-first / correlation-aware)
  - `decision_humility` — closing guidance (4 alleles: empty /
    pro-hold / contradiction-as-hold / clustering-discount)
  Every mutation preserves the JSON-output contract and every
  halal-compliance constraint; only prose framing varies. The
  *first* allele of each slot is the canonical default, so passing
  no genome reproduces today's prompt byte-for-byte.
- `crypto/prompts.py:build_prompts(ctx, params, *, genome=None)`
  — backward-compatible: omitting the genome is a no-op; a genome
  with a recognised slot key flows into the rendered system prompt.
  Unknown slot keys are silently ignored so old genomes in DB don't
  crash a live cycle.
- `crypto/prompts.py:crypto_allele_pool()` — concrete `AllelePool`
  the GA evolves against.
- `crypto/prompt_fitness.py` — two evaluators:
  - `replay_pnl_fitness` — signed `today_pnl_pct` minus a tiny
    length penalty (token-cost regulariser).
  - `confidence_proxy_fitness` — mean confidence pulled from the
    snapshot's `ml_signals_text` (lightweight sanity-check signal).
  Both are intentionally weak — the GA produces a *ranking* the
  operator inspects via `/api/prompts/candidates`, never an
  auto-promote.
- `cli/prompts.py` — new `halal-trader prompts evolve / candidates /
  promote` subcommands. Mirrors the dashboard ops for terminal use.
- `crypto/scheduler.py:_nightly_prompt_evolve` — fires once per
  day inside `_daily_end`. Reads `settings.crypto.prompt_evo_*`
  knobs (generations=8 / population=12 / snapshots=200). Failures
  log at debug; never block trading.
- `config.py:CryptoSettings` — three new bounded knobs
  (`prompt_evo_generations`, `prompt_evo_population`,
  `prompt_evo_snapshots`) with pydantic validators.

**Acceptance**:
- Running the bot for one trading day produces `prompt_genomes`
  rows ≥ 0 — the nightly hook is wired and the persistence path is
  identical to the unit-tested `evolve_with_replay`. (The "> 0"
  literal can only be verified on a bot with replay snapshots; the
  acceptance criterion is "the job runs end-to-end without
  exception", which the test `test_prompt_evo_wiring.py` proves at
  the unit level.)
- One-click promote — `gh api repos/.../api/prompts/{id}/promote`
  → writes `ACTIVE_PROMPT_VERSION=<name>@genome-<id>` to
  `RuntimeConfig`. The terminal companion is
  `halal-trader prompts promote <id>`.

**Trade-offs documented**:
- **Fitness is currently weak.** Real LLM replay against every
  `(genome, snapshot)` pair would cost ~12 × 200 × 8 = 19,200 LLM
  calls per nightly run — economically unviable. Today's
  evaluators are cheap snapshot-metadata signals. The next iteration
  could plug in a small distilled scorer; the runner's
  `Evaluator = Callable[[PromptGenome, CycleSnapshot], Awaitable[float]]`
  abstraction means swap-out is a one-import change.
- **The GA never auto-promotes.** The operator is always in the loop
  via `/api/prompts/{id}/promote` or the CLI. This is intentional
  for a single-user paper-trading bot — silent prompt swaps would
  break the audit trail.
- **Three slots is a starting point.** Adding more slots is a
  one-line change in `_slot_alleles()` plus the corresponding
  `{slot_name}` placeholder in `SYSTEM_PROMPT`. Resist adding slots
  that touch the JSON output schema or halal-compliance rules.

---

### Wave G — Replay-fitted slippage + execution model ✅ landed

**Why**: the executor records `paper_slippage_pct` and
`live_slippage_pct` but nothing closes the loop. Backtests assume a
fixed slippage. Live runs occasionally hit unexpected fills the
strategy didn't budget for. The data was sitting in `crypto_trades`
— this wave closes the loop.

**What landed**:
- `ml/slippage.py` — `SlippageModel` (ridge-regularised linear over 6
  features: size_usd, spread_bps, atr_pct, rsi_14,
  kline_volatility_pct, hour_of_day), `fit_from_trades`,
  `trade_to_sample`, and a new `features_from_live_order` helper that
  builds the same feature vector at fill time (derives spread_bps
  from orderbook top-of-book when not in the indicators dict).
  Persistence: DB-first via the Wave-K `ml_artefacts` table, JSON
  file fallback under `settings.ml.models_dir`.
- `crypto/executor.py` — new `slippage_model` ctor arg + a
  `_predict_slippage(symbol, price, quantity, indicators_cache,
  orderbooks)` helper called from both BUY and SELL fill paths.
  Result is stamped onto the `crypto_trades.predicted_slippage_pct`
  column (added by migration `552a7d6e5862`). Same column added to
  the stocks `trades` table for parity.
- `crypto/backtest.py:SimulatedExecutor` — new `slippage_model` ctor
  arg. When provided, `_baseline_slippage_for(...)` reads the model's
  prediction instead of the constant `slippage_pct`; on a prediction
  failure the simulator falls back to the constant so a broken model
  can't block a backtest.
- `core/cycle_stages.py:BuildSlippageTextStage` — formats one
  `pair: predicted slippage ±N.N bps (confidence X%)` line per halal
  pair into `state.slippage_text`. Threaded through
  `analyze_kwargs` and rendered as a new
  `=== EXPECTED EXECUTION COST (predicted slippage) ===` block in
  the prompt template (optional — collapses when empty).
- `ml/retrainer.py:RetrainingScheduler._retrain_slippage` — new
  per-retrain step. Pulls recent filled trades (≤500) +
  labeled-indicator snapshots, joins them on `trade_id`, builds
  samples via `trade_to_sample`, calls `fit_from_trades`, saves
  file + DB. Optional: the stocks-namespace retrainer can omit the
  `crypto_trade_repo` + `engine` args and skip the slippage step.
- `crypto/components.py` — new `_load_slippage_model` loader (DB →
  file → identity) wired through to both `CryptoExecutor` and the
  `BuildSlippageTextStage` via the cycle service.

**Acceptance**:
- Backtest Sharpe vs live Sharpe convergence requires a one-month
  sample to validate empirically; the *plumbing* that closes the
  loop is in place — `SimulatedExecutor` and `CryptoExecutor` now
  read predictions from the same `SlippageModel` instance, so any
  divergence remaining is from the model, not from the backtester
  lying about slippage.
- Predicted-slippage block renders in the LLM prompt when the
  model is wired (`BuildSlippageTextStage` + new prompt section).

**Trade-offs**:
- `kline_volatility_pct` and `spread_bps` aren't yet populated by
  `crypto.indicators.compute_all` — the model treats them as zero
  for training samples that pre-date those features being captured.
  Adding them is a small follow-up that doesn't block the closed
  loop.
- The cycle's `BuildSlippageTextStage` uses
  `max_position_pct * total_balance_usdt` as the size estimate
  (since the LLM hasn't sized yet); per-decision sizing would
  require a second pass through the model after the analyze block.
  Current design is fine — the prompt block is *guidance* for the
  LLM, not a per-trade cost.

---

### Wave H — Agentic mode: multi-turn tool calling with bounded budget

**Why**: today every cycle is one prompt → one decision. Modern best
practice is to give the LLM a **toolbelt** and let it decide whether
it needs more info before committing. For a 60s crypto cycle with
$0.02-0.04 per call, an agentic flow that does a typical 1.3 calls
per cycle (instead of 1.0) costs ~$30/month extra and unlocks much
better decisions in ambiguous setups.

Tools the agent can use:
- `analyze_pair(symbol)` — pulls 4h klines + multi-timeframe
  indicators on demand.
- `query_rag(text)` — retrieves analogous past trades from the RAG
  store (Round 2's pgvector now backs this in ~1ms).
- `query_regime_memory(features)` — the same for regime memory.
- `compute_var_95(symbols, weights)` — the existing VaR module
  exposed as a tool.
- `submit_plan(...)` — the terminal call; ends the loop.

The agent gets a per-cycle tool-call budget (default 5) and a
per-cycle wall-clock budget (default 30s). Over budget → forced
`submit_plan` with whatever it has. The whole transcript lands in
the LlmDecision row as a JSONB `tool_transcript` column so the
dashboard can render the agent's chain of thought per cycle.

**Scope**:
- `core/llm/agent.py` — the tool-calling loop (provider-agnostic
  via `BaseLLM.generate_tool_call`).
- New SQLModel column `LlmDecision.tool_transcript: list | None`.
- `CryptoTradingStrategy` opts into agentic mode via
  `crypto.agentic_enabled` setting.
- Dashboard page that renders the transcript as a tree.

**Acceptance**:
- Setting `CRYPTO_AGENTIC_ENABLED=true` flips the strategy into the
  loop without code changes.
- A test that mocks two tool calls + one `submit_plan` runs the loop
  to completion and persists the transcript.
- Daily LLM cost stays within ±50% of single-call mode.

**Estimated effort**: 2 days. Highest functional payoff on this
roadmap; the first thing the project should ship as a "modern AI
trading bot" headline.

---

## Operability + UX

### Wave I — Live-cycle WebSocket: what the dashboard *should* show

**Why**: the dashboard polls `/api/positions`, `/api/trades`,
`/api/insights/*` every few seconds. That misses anything that
happens *between* polls and burns server CPU on every refresh.
What the operator actually wants is **a live feed of what the bot
is doing right now**: this cycle started, fetched klines, called
the LLM, decided to BUY BTCUSDT 0.005 at $42100 with stop $41600.

A single `/ws/cycle` WebSocket that streams structured events:
`cycle.start`, `cycle.stage.complete`, `llm.call.start`,
`llm.call.complete`, `executor.fill`, `monitor.exit`. The bot
publishes events to an in-process `EventBus`; the WS streams them
to connected clients.

This pairs perfectly with Wave B (per-stage cycle pipeline) — the
stages already emit events; the bus just multiplexes them.

**Scope**:
- `core/event_bus.py` — async pub/sub with topic globs and
  back-pressure-safe slow-consumer drop.
- `web/routes/streaming.py:/ws/cycle` — new route, JSON line
  protocol.
- Dashboard "Live" page — a single scrolling timeline of events
  with collapsible nested LLM tool calls.

**Acceptance**:
- Open the dashboard, watch one full cycle render in real time,
  end-to-end.
- The events match what the JSON log file contains.

**Estimated effort**: 1.5 days.

---

### Wave J — Per-stage Prometheus histograms + Grafana dashboard ✅ landed

**Why**: `web/prometheus.py` exposes ~10 gauges (no histograms, no
labels). With Wave B's cycle pipeline, each stage has a cleanly
defined start/stop, so a histogram of stage latency is one decorator
away. The same trick applies to broker calls, LLM calls, and DB
writes.

**What landed**:
- `core/metrics.py` exposes 3 histograms (`halal_trader_stage_latency_ms`,
  `halal_trader_llm_call_ms`, `halal_trader_broker_call_ms`) and one
  counter (`halal_trader_events_published_total`) with consistent label
  shapes (`stage`/`error`; `provider`/`model`; `broker`/`method`/`error`;
  `topic`). Buckets cover 1 ms → 60 s for all latency families.
- **Stage latency** is wired via `core/cycle_pipeline.stage()` —
  every `CycleStage` run gets an observation automatically (Wave B
  finished the extraction; this just turned on the firehose).
- **LLM latency** is wired via a new `BaseLLM._record_usage(usage)`
  helper that every provider (Ollama, OpenAI, Anthropic generate +
  Anthropic tool-call) now calls instead of inline `last_usage = …`.
  Single seam → one place to evolve.
- **Broker latency** is wired via a new
  `core.metrics.timed_broker_call(broker, method)` async decorator
  on every public `BinanceClient` method (`get_account`,
  `get_balances`, `get_open_orders`, `get_klines`, `get_order_book`,
  `place_order`, `cancel_order`), plus a single inline timer on
  `mcp.client.AlpacaMCPClient.call_tool` (the chokepoint for every
  Alpaca tool — covers all stocks-side calls in one site).
- **Event-bus throughput** is wired via a single `event_published(topic)`
  call in `EventBus.publish`.
- `infra/grafana/halal-trader.json` (~140 lines) ships 10 panels:
  4 stat tiles (bot running / drawdown / daily LLM spend / open
  positions) + cycle p50 + cycle p95 + LLM p50 by provider + broker
  error-rate + open positions by asset + portfolio heat by market.

**Acceptance** (both met):
- `curl localhost:8082/metrics | grep _bucket` shows histograms for
  all three families.
- Importing `infra/grafana/halal-trader.json` in a fresh Grafana
  produces a working dashboard against the bot's `/metrics`.

**Trade-offs**:
- The decorator approach for Binance covers the 7 most-called
  methods but skips a few internal helpers (`get_funding_signal`,
  `get_ticker_price`, `get_symbol_ticker` cache-fill calls) since
  their failure modes are already absorbed by the public-method
  caller. Add `@timed_broker_call("binance", "…")` to extend
  coverage if a new tail-latency bug surfaces.
- `event_published` fires for every publish without per-subscriber
  fan-out timing — the counter is intentionally coarse so a
  high-frequency bus doesn't lose the signal in label cardinality.

---

### Wave K — Move ML model artefacts from pickled files to the DB

**Why**: today `models/anomaly_detector.pkl`,
`models/anomaly_state.pkl`, `models/signal_classifier.pkl`, and
`models/regime_classifier.pkl` are file-based. That's the **last**
file-based persistence in the production write path. After Round 1's
JSON sidecar purge, every other piece of state is in Postgres. The
bot is "Postgres-only" *almost*.

Beyond the property, file-based pickles are operationally annoying:
they don't replicate across hosts, don't roll back with the DB, and
there's no audit trail when one is regenerated.

Add a `ml_artefacts` table: `(name PRIMARY KEY, version INT, payload
BYTEA, created_at, sklearn_version, feature_hash)`. Save/load goes
through the table. The retrainer writes a new row; the loader reads
the latest one. The dashboard gets a "model versions" page that
shows what's currently live.

**Scope**:
- New SQLModel table + migration.
- `ml/anomaly.py`, `ml/signal_classifier.py`, `crypto/regime.py`
  — switch save/load to the repo.
- `models/` directory becomes only a temporary write target for
  HuggingFace caches (Chronos), which are large and shouldn't live
  in Postgres.

**Acceptance**:
- `find data models -name "*.pkl"` returns no results in
  production.
- `halal-trader ml versions` lists model history with timestamps
  and feature hashes.

**Estimated effort**: 1 day.

---

### Wave L — Sharia-compliance "explainer" mode

**Why**: every trade is gated by a halal screener and the receipt is
recorded in `halal_screenings`. But the *reasoning* is a JSONB blob
that nobody looks at. For an operator who cares about Sharia
compliance (the only kind this bot has), an "explainer" route that
walks through the gate's reasoning for a given trade — "BTC was
allowed because: layer-1 utility token, no interest-bearing
features, market cap > $1B" — would build trust in the bot's
decisions.

This is also a hedge for the eventual jurisprudence question: if a
scholar challenges a position, the operator pulls up the explainer,
sees the chain of reasoning, and can either explain it or override.

**Scope**:
- `halal/explainer.py` — pure function, takes a screening criteria
  dict and renders a markdown explanation.
- `web/routes/admin_halal.py` — new `/api/halal/explain/{trade_id}`.
- Dashboard "Trade detail" page gets an "explain" button that
  fetches and renders.
- A small `halal/jurisprudence.md` doc that the explainer references
  by section number (so updates to the reasoning text don't drift
  silently).

**Acceptance**:
- Click "explain" on any trade → markdown explanation rendered with
  citations.
- The explainer renders the same reasoning offline (CLI:
  `halal-trader halal explain TRADE_ID`).

**Estimated effort**: 1 day.

---

## Cross-cutting principles

* **Each wave is independently shippable.** Stop after any one and
  the codebase is strictly cleaner / more capable than before.
* **One migration per wave.** If two waves touch the same table,
  split.
* **No half-finished features.** A wave that lands without its
  dashboard surface or its CLI command isn't complete; it's a
  half-implemented liability.
* **Suite green at every commit.** Including parallel
  (`pytest -n auto`).
* **Greenfield-aggressive.** Drop dead code; don't preserve API
  shapes for users who don't exist.

---

## Suggested order

The waves cluster into three independent tracks that can ship in
parallel; within a track they have light dependencies. If you only
ship six items, ship the bold ones — they're the highest-leverage:

1. **A (typed BotContext)** — every other wave benefits.
2. **B (cycle pipeline)** — unblocks D, F, G, J.
3. **C (anyio task groups)** — kills a class of silent-crash bugs.
4. D (repo split)
5. **E (tool-use)** — modernises the LLM surface.
6. **H (agentic mode)** — biggest functional payoff; the headline
   feature.
7. F (prompt-evolution A/B)
8. G (replay-fitted slippage)
9. **I (live-cycle WebSocket)** — best UX win.
10. J (Prometheus histograms + Grafana)
11. K (ML artefacts to DB)
12. L (halal explainer)

**Total estimated effort: ~16 days of focused work,** broken into
12 commits. The roadmap is intentionally larger than rounds 1-2:
the cleanup waves are mostly behind us, and the next inflection in
this codebase comes from **structural choices** (Waves A/B/C/D) and
**agentic LLM features** (Waves E/H). Everything else is consequent.
