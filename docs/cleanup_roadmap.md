# Roadmap — Round 3

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

### Wave C — Adopt `anyio` task groups for structured concurrency

**Why**: today the bot creates background tasks via
`asyncio.create_task(...)` in seven places (monitor, ws_manager,
news_reactor, sentiment_manager, reconcile_loop, position_monitor,
research_jobs). Each has its own ad-hoc shutdown sequence. When one
silently dies (a coroutine raises before the supervisor logs it),
*nothing* notices until the operator sees a stale dashboard.

`anyio.create_task_group()` is the modern primitive: cancel-on-error
propagation, automatic awaiting on context exit, no fire-and-forget
loss. Switching the scheduler to manage all background work under a
root task group means a crash in one supervisor cancels the whole
bot — which is what we want for a single-user paper-trading bot.

**Scope**:
- Add `anyio` to `[project.dependencies]` (already in via httpx).
- `core/scheduler.py:run` enters `async with anyio.create_task_group()
  as tg:` once, and every long-lived task (monitor, ws, news_reactor,
  reconcile loop) is spawned with `tg.start_soon(...)`.
- Each subsystem's `start()` becomes "register this coroutine with
  the supervisor"; `stop()` becomes "cancel my scope."
- One `RestartPolicy` knob per task: `crash_bot` (the default for
  monitor/ws), `restart` (for sentiment/news_reactor with backoff),
  or `ignore` (for the optional reddit fetcher).

**Acceptance**:
- `grep -r 'asyncio\.create_task' src/` is a single-digit count
  (only inside `RestartPolicy`-aware adapters).
- A test that raises inside the monitor cancels the bot's run loop
  with the original traceback, not a silent hang.

**Estimated effort**: 1 day. Adds a small amount of code; pays for
itself the first time a background task crashes.

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

### Wave F — Continuous A/B harness over the prompt-evolution GA

**Why**: `core/llm/prompt_evo.py` is sitting unused. It has a clean
GA over prompt slot×allele genomes but nothing wires it to a fitness
function. The honest fitness signal is **paper P&L on replayed
cycles** (Wave C of round 1 promoted the snapshot store, so we have
the raw material).

Hook the GA up to a fitness loop that:
1. Takes the last 200 replay snapshots (DB query, ~ms).
2. Runs each genome's prompt against each snapshot via the strategy's
   replay harness (parallel, bounded concurrency).
3. Scores by Sharpe over the resulting trades.
4. Mutates / crosses over the top-K, repeats for N generations.
5. Records the best genome to a `prompt_genomes` table (new) with
   its fitness, lineage, and cycle count.

Run it nightly via `apscheduler` already in the bot. The dashboard
gets a "candidate prompts" page where the operator sees which slot
swaps improved fitness and can promote a genome to live with one
click — which writes to `RuntimeConfig` and the next cycle picks it
up.

**Scope**:
- `crypto/prompts.py` is already structured around named slots; add
  `AllelePool` content for each slot (3-5 phrasings each).
- `core/llm/prompt_evo_runner.py` — wires GA to ReplayStore + a
  fitness function that backtests against snapshots.
- New SQLModel table `PromptGenome` with `id`, `genome_json`,
  `fitness`, `n_cycles`, `parent_ids`, `promoted_at`.
- New web route `/api/prompts/candidates` + dashboard page.
- New CLI: `halal-trader prompts evolve --generations 8` for ad-hoc
  runs.

**Acceptance**:
- Running the nightly job produces a `prompt_genomes` row count > 0
  the next morning.
- One-click "promote" path works end-to-end.

**Estimated effort**: 2 days.

---

### Wave G — Replay-fitted slippage + execution model

**Why**: today the executor records `paper_slippage_pct` and
`live_slippage_pct` but nothing closes the loop. Backtests assume a
fixed slippage. Live runs occasionally hit unexpected fills the
strategy didn't budget for. We have months of paired
(intended-price, filled-price, kline-context) data sitting in
`crypto_trades` — a regression model can learn the slippage function
and feed it back into the *backtester*, which currently lies about
what live execution will do.

Train a tiny gradient-boost model:
- features: order size relative to 1m volume, spread bps, ATR,
  RSI, kline volatility, hour of day.
- target: `(filled_price - intent_price) / intent_price`.

Save it to `models/slippage_v1.json` (XGBoost JSON format — small,
versioned, deployable). Backtester reads it; the executor optionally
*displays* the predicted slippage in the prompt so the LLM knows the
expected execution cost.

**Scope**:
- `ml/slippage.py` — train + predict.
- `crypto/backtest.py` — replace the constant `_SLIPPAGE_BPS` with
  the model's prediction.
- `crypto/executor.py` — add `predicted_slippage_pct` to the
  `crypto_trades` row when filling.
- A nightly retrain job (uses `RetrainingScheduler` already in the
  bot).

**Acceptance**:
- Backtest Sharpe and live Sharpe converge within ±0.3 over a
  one-month sample (today they often diverge by 0.6-1.0).
- Predicted slippage shows up in the prompt context block.

**Estimated effort**: 1 day for the model, 1 for the wiring.

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

### Wave J — Per-stage Prometheus histograms + Grafana dashboard

**Why**: `web/prometheus.py` exposes ~10 gauges (no histograms, no
labels). With Wave B's cycle pipeline, each stage has a cleanly
defined start/stop, so a histogram of stage latency is one decorator
away. The same trick applies to broker calls, LLM calls, and DB
writes.

The dashboard already has a `/metrics` endpoint; just add the
histograms and ship a Grafana dashboard JSON in `infra/grafana/`.

**Scope**:
- `core/metrics.py` — wrap `prometheus_client` with consistent
  naming.
- `crypto/cycle/stages/*.py` already wrapped via the
  `StageInstrumentation` from Wave B; add a histogram emit.
- `infra/grafana/halal-trader.json` — 8 panels: cycle p50/p95
  latency by stage, LLM call cost rate, broker error rate, kill-
  switch state, daily P&L, drift state, regime distribution, halal
  cache hit rate.

**Acceptance**:
- `curl localhost:8082/metrics | grep _bucket` shows histograms.
- Importing `infra/grafana/halal-trader.json` in a fresh Grafana
  produces a working dashboard against the bot's `/metrics`.

**Estimated effort**: 1 day.

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
