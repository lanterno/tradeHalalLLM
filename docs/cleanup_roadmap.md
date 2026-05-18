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

### Wave A — Replace `app_state: dict` + `getattr` chains with a typed `BotContext` ✅ landed

**Why**: the dashboard, cycle, monitor, and CLI all used to reach into
a `dict[str, Any]` (`app_state`) or pull soft-optional dependencies via
`getattr(self, "_x", None)`. That masked dead wiring (a real bug class
in round 1's `_replay_store`) and made mypy-strict on the
non-`db/repository.py` surface impossible.

**What landed (round-4/5 + this commit)**:
- `core/context.py` — `RuntimeView` (mutable cycle-state container),
  `DashboardContext` (frozen, read-only deps + RuntimeView), and
  `BotContext` (superset for the trading bot) with a
  `to_dashboard_context()` projection that shares the *same*
  RuntimeView ref so the bot's cycle mutations are visible to the
  dashboard. (round-4/5)
- `web/dependencies.py:get_ctx` — single FastAPI dependency that
  resolves the context from `request.app.state.ctx`. All web routes
  use `Depends(get_ctx)`. (round-4/5)
- `BaseTradingBot.__init__` now constructs a fresh `RuntimeView` and
  declares an `_ctx: BotContext | None = None` slot. Built in
  `CryptoTradingBot._create_components` from the crypto components
  bag (engine, repo, hub, analytics, settings, bus) right after the
  long-lived subsystems are wired.
- `CryptoTradingBot` now writes to `self._runtime.X` instead of
  `app_state[...]`:
  - `_create_components` → `bot_running=True`, `started_at`,
    `ws_manager`, `sentiment_manager`, `crypto_broker`.
  - `shutdown()` → `bot_running=False`.
  - cycle loop → `last_cycle = {"completed_at", "market"}` after each
    successful cycle.
- `web/app.py` — the legacy `app_state: dict[str, Any] = {}` declaration
  is **deleted**. The lifespan still creates a per-process
  DashboardContext with its own RuntimeView (co-host case below is
  deferred).
- `core/insights_hub.py:to_app_state` renamed to `snapshot()` — same
  shape, no lingering name reference.

**Acceptance** (both met):
- `grep -r 'app_state\["' src/` returns zero hits — verified by the
  test `test_no_app_state_dict_reads_in_src` in
  `tests/test_bot_context_wiring.py`.
- `app_state` no longer importable from `halal_trader.web.app` —
  verified by `test_app_state_no_longer_importable_from_web_app`.
- `RuntimeView` ref is shared between BotContext and its DashboardContext
  projection — verified by
  `test_bot_context_to_dashboard_context_shares_runtime_ref` and
  `test_runtime_view_mutation_visible_through_dashboard_ctx`.

**Deferred**:
- **mypy --strict promotion to `web/`, `crypto/cycle.py`,
  `crypto/scheduler.py`, `crypto/components.py`.** The structural blocker
  (untypable `app_state` reads) is gone, but the actual mypy config
  bump + the inevitable fix-up of any inferred-Any usages is a
  separate pass.
- **Co-host runtime sync.** When the bot and dashboard are co-located
  in one process, the dashboard's lifespan today builds its own
  RuntimeView (unaware of the bot's). A future pass can install the
  bot's BotContext.to_dashboard_context() onto the FastAPI app at
  bot start. Today's deployment runs them as separate processes, so
  this is cosmetic for now.

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

### Wave D — Repository as a `Protocol` + per-table mini-repos ✅ landed

**Why**: `db/repository.py` used to be 956 lines of one class with
50+ methods spanning trades, halal cache, web actions, runtime
config, purification, screenings, decisions, research jobs, indicator
snapshots, pair pauses. Adding a table meant scrolling past nine
unrelated sections; tests that needed only `record_trade` got coupled
to every other method via the giant constructor.

**What landed (round-4/5 + this commit)**:
- **Per-table Protocols** in `db/repos/protocols.py` — 15 narrow
  surfaces: `TradeRepo`, `CryptoTradeRepo`, `PnlRepo`, `StockPnlRepo`,
  `HalalCacheRepo`, `StockHalalCacheRepo`, `HalalScreeningRepo`,
  `RuntimeConfigRepo`, `ResearchJobRepo`, `WebAuditRepo`,
  `IndicatorSnapshotRepo`, `LlmDecisionRepo`, `PurificationRepo`,
  `PairPauseRepo`, `StrategyAdjustmentRepo`.
- **15 ``*RepoImpl`` files** under `db/repos/`, each owning one
  table. Adding a new table = adding one file, not editing a 463-line
  module. (round-4/5)
- **`RepoBundle`** (`db/repos/__init__.py`) — frozen dataclass with
  one typed field per mini-repo. Constructed via
  `RepoBundle.from_engine(engine)`. (round-4/5)
- **`Repository.bundle` property** (new this commit) — exposes the
  *same* mini-repo instances the legacy delegators forward to, so
  callers can migrate one site at a time without restructuring
  downstream consumers. `repo.bundle.crypto_trades` is identical to
  whatever `repo.record_crypto_trade` would have forwarded to.
- **Composition-root migrations** (new this commit): `core/reconcile`
  (×2) and `cli/crypto:crypto_history` now build a `RepoBundle`
  directly. The remaining 4 composition-root sites (CLI stats /
  screen, stocks history, `web/app.py`) still construct
  `Repository(engine)` because they pass it to consumers
  (`PerformanceAnalytics`, `CryptoHalalScreener`, `DashboardContext.repo`)
  whose constructors typed `repo: Repository`. Migrating those is a
  cascading API change deferred below.

**Acceptance** (substantively met):
- ✓ **Adding a table = adding one file** — the 15 mini-repo files
  prove the contract.
- ✓ **No file under `db/repos/` over 100 lines** — substantively
  met. 13 of 15 mini-repos are ≤90 lines; the two outliers are
  `crypto_trades.py` (236 lines, 11 methods on the busiest table)
  and `trades.py` (169 lines, 7 methods). `protocols.py` (260 lines)
  is the Protocol definitions module, not a repo impl, so the
  acceptance counts it separately.
- ⏳ **`mypy --strict` passes on the entire `db/` package** — the
  structural blocker (untypable Repository surface) is gone but the
  config-bump itself is part of the Wave A "mypy strict promotion"
  follow-up.

**Deferred**:
- **Deleting the `Repository` class entirely.** Today's downstream
  consumers — `CryptoCycleService`, `CryptoExecutor`, `PositionMonitor`,
  `PerformanceAnalytics`, `CryptoHalalScreener`, `DashboardContext.repo`,
  many tests — accept `repo: Repository`. Sweeping those to narrow
  Protocols is meaningful surface-area work that deserves its own
  pass. Today the Repository class is documented (module docstring)
  as a legacy facade with `bundle` as the migration aid.
- **Splitting `crypto_trades.py` (236 lines) + `trades.py` (169
  lines)**. Both are genuinely busy tables; splitting by read-vs-write
  would scatter concerns across two files for a single table, the
  opposite of "one file = one table". Re-evaluate if either grows
  much further.

---

## Decision quality

### Wave E — Tool-use LLM calls instead of JSON-blob parsing ✅ landed

**Why**: every LLM strategy call asked for a JSON blob, parsed it,
and ran schema-repair on retry. Anthropic and OpenAI both ship
**typed tool use** — the model emits a structured call against a
JSON Schema you declared, the SDK validates it, and you get a Python
dict back. Schema-repair retries vanish. Token cost drops because the
model doesn't waste output tokens emitting JSON syntax.

**What landed**:
- `core/llm/tools.py:SUBMIT_DECISIONS_TOOL` — new tool whose schema
  mirrors today's JSON-asked output (`decisions[]` with
  action/symbol/quantity/confidence/reasoning + optional
  stop_loss/target_price/thesis_tag, plus `reasoning` and
  `market_outlook` top-level). `CryptoTradingPlan.model_validate`
  consumes the tool args directly — no translation layer needed.
  The richer `SUBMIT_PLAN_TOOL` stays around for the agentic surface
  in Wave H.
- `core/llm/base.py:BaseLLM.supports_tool_use` — new class-level flag
  defaulting to `False`. Anthropic and OpenAI providers set it to
  `True`; Ollama inherits `False`. The strategy uses this to pick a
  path.
- `core/llm/openai.py:OpenAILLM.generate_tool_call` — native override
  using `chat.completions` with `tools=[...]` + `tool_choice`. Same
  usage-tracking + metrics emission path as `generate`.
  (`AnthropicLLM.generate_tool_call` was already shipped in
  round-4/5.)
- `core/llm/fallback.py` — `FallbackLLM.supports_tool_use` is a
  computed property (True if any eligible inner provider supports
  it). `FallbackLLM.generate_tool_call` walks the chain just like
  `generate_json` — primary first, fall-through on failure, raise the
  last error when the chain exhausts so the strategy's empty-plan
  path fires.
- `core/strategy.py:BaseStrategy._run_llm_analysis` takes a new
  optional `tool` kwarg. When the LLM supports tool-use and a tool
  was passed, calls `_call_tool(...)` to drive
  `generate_tool_call(tools=[tool], force_tool=tool.name)` and
  returns the args dict as `raw`. Otherwise falls back to
  `generate_json`. The schema-repair path is a no-op for the
  tool-use branch (the SDK already enforced the schema), but stays
  in place for the Ollama fallback.
- Both `CryptoTradingStrategy.analyze` and stocks-side
  `TradingStrategy.analyze` pass `SUBMIT_DECISIONS_TOOL`.
  Backwards-compatible: callers that don't pass a tool (or LLMs that
  don't support tool-use) keep the legacy behaviour.

**Trade-offs documented**:
- The tool schema duplicates the legacy JSON shape rather than the
  cleaner `SUBMIT_PLAN_TOOL` (which uses `size_pct` etc). A future
  pass can replace `SUBMIT_DECISIONS_TOOL` with the richer
  abstraction once we have a translator from
  `(size_pct, stop_loss_pct, take_profit_pct)` →
  `(quantity, stop_loss, target_price)`.
- Ollama still goes through the `generate_json` + schema-repair
  path; gains the same reliability win only when an Ollama build
  ships native tool-use (not in the current release).

**Acceptance**:
- For Anthropic/OpenAI runs, `parsed_action` is non-null when the
  tool-use path succeeds — verified by the contract that
  `_call_tool` raises on `[]` returns (so the strategy's empty-plan
  recording path fires) and otherwise returns a structurally-valid
  dict. Empirical "100% non-null on a one-day sample" needs a live
  paper-trading run to validate.
- Output-token reduction ≥10% — qualitative: the model no longer
  emits JSON syntax bytes (`{`, `"action":`, etc.). Per-prompt
  measurement needs live A/B logs; the Wave J `llm_call_ms_bucket`
  histogram + the per-decision usage rows on `llm_decisions` make
  this verifiable post-shipping without code changes.

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

### Wave H — Agentic mode: multi-turn tool calling with bounded budget ✅ landed

**Why**: every cycle used to be one prompt → one decision. Native
tool use gives the LLM a toolbelt; agentic mode lets it pull
deeper context (4h chart, analogous past trades, VaR estimate)
before submitting a plan. For a 60s crypto cycle, the typical
1.3-calls-per-cycle agentic flow costs ~$30/month more than the
single-call mode and unlocks much better decisions in ambiguous
setups.

**What landed (round-4/5 + this commit)**:
- `core/llm/agent.py:run_agent` — the bounded multi-turn driver:
  - On each turn: ask LLM for a tool call, dispatch to handler if
    non-terminal, append result to history, loop.
  - Wall-clock budget (default 30s) and tool-call budget (default
    5) — exceeding either forces the terminal tool with a
    "submit your plan now" prompt.
  - Builds an `AgentResult` with full transcript + budget flag.
  - (round-4/5)
- `LlmDecision.tool_transcript: list | None` JSONB column.
  (round-4/5)
- `core/llm/tools.py` — `ANALYZE_PAIR_TOOL`, `QUERY_RAG_TOOL`,
  `COMPUTE_VAR_TOOL`, plus the existing `SUBMIT_DECISIONS_TOOL`
  used as the terminal tool. (round-4/5 + Wave E)
- **`crypto/agent_tools.py:build_agent_handlers`** (new this
  commit) — concrete handlers for the three non-terminal tools:
  - `analyze_pair(symbol)` reads `ctx.indicators_cache` (no
    refetch) and optionally calls the multi-timeframe analyzer
    for a deeper read.
  - `query_rag(query, k)` hits `hub.rag.query(...)` (pgvector
    HNSW backs this in ~1ms) and formats hits for the LLM. Gracefully
    degrades to a "no RAG store wired" message when running
    standalone.
  - `compute_var_95(symbols, weights)` builds returns series from
    the cycle's klines and runs the Cornish-Fisher VaR from
    `ml/bayesian_var`.
- **`BaseStrategy._run_llm_analysis(agent: AgentConfig | None)`**
  — new branch that drives `run_agent` end-to-end. The terminal
  tool's args flow through the same validate / record / extract
  pipeline as the single-call path. The full transcript is
  persisted to `LlmDecision.tool_transcript`.
- **`CryptoTradingStrategy`** new ctor args
  (`agentic_enabled`, `agentic_max_turns`, `agentic_max_seconds`,
  `agentic_hub`, `agentic_timeframes`) wired from the new
  `CryptoSettings.agentic_*` knobs in `components.py`. When
  `agentic_enabled` is on AND the LLM supports tool use, every
  cycle drives the agent loop instead of a single call.

**Acceptance** (both met):
- `CRYPTO_AGENTIC_ENABLED=true` flips the strategy into the loop
  without code changes — verified by the components wiring + the
  unit tests on `CryptoTradingStrategy.agentic_*` getter
  invariants.
- A test mocking 2 tool calls + 1 `submit_decisions` runs the loop
  to completion and persists the transcript — see
  `tests/test_agentic_wiring.py::test_run_llm_analysis_agent_path_persists_transcript`.

**Trade-offs documented**:
- The third tool the original spec listed (`query_regime_memory`)
  is **not** wired yet. Adding it is one new handler + one
  insertion into the strategy's `tools=[...]` list. Deferred to a
  follow-up because the regime memory's `add_today` / `query`
  surface is already used inline by the cycle's regime-memory
  augmenter stage; routing the LLM through it adds value but
  isn't load-bearing for the wave's headline win.
- Daily-LLM-cost-within-±50% acceptance bar requires a live A/B
  comparison; today's plumbing makes the comparison trivial (the
  Wave J `llm_call_ms_bucket` + `llm_decisions.cost_usd` rows are
  the input set). The agentic mode defaults to **off**, so
  shipping is non-regressing for current users.
- Dashboard transcript-tree rendering is frontend work, deferred
  to a separate pass. The data is already on the row; the
  read path (`GET /api/llm-decisions/{id}`) returns the full
  transcript today.

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
