# Operator context & hard-won lessons

Durable knowledge for any LLM continuing this work — the things that are **not
derivable from the code**: how the operator wants work done, the trading intent
behind the design, open issues that aren't code-fixable, and lessons already
paid for once (don't re-learn them expensively). Distilled from prior working
sessions. Keep it current; delete what goes stale.

---

## Working agreement

Solo home project, "dev mode — no one to review anything." The bottleneck is
throughput, not review.

- **Work directly on `main`.** Commit + push at clean, tested checkpoints
  without asking each time. No PR-review gate.
- **Be autonomous.** "Always keep going." Keep building through the roadmap on
  your own judgment; make the required decisions yourself.
- **The test gate is the quality bar**: `just lint` (ruff), `just typecheck`
  (mypy strict on `core/` + `domain/` + all of `src/halabot`), `just test`
  (pytest, real Postgres on :5433). All green before every commit.
- **Surface a decision only when it's genuinely operator-only**: spending real
  money, choosing a model/provider, or a destructive/irreversible op. Otherwise
  proceed.
- **Hard invariants that override all of the above** (never violate for
  throughput): paper/testnet only — never real money; halal compliance is
  non-negotiable (long-only, no short/interest/leverage/derivatives); never
  destabilize the live bot; the `src/halabot` engine never trades (see below).

## Trading strategy direction (stocks): **fast in, slow out**

Operator decision (2026-05-22), explicitly NOT "fast in, fast out." Motivated by
watching whipsaw bleed a session to ~$565 slippage on $42 net P&L — symmetric
churn kills this bot.

- **Entries can be aggressive**: react to fresh news + a *confirming* price move
  within seconds. The whipsaw guard on entries is the **price-confirmation
  requirement** (must be moving up, not news alone).
- **Exits must be patient**: default trailing stops WIDE (≈2–3% activation,
  1.5–2% trail). Don't tighten on small profits — that exits winners early.
- **Momentum positions are LLM-untouchable.** A position entered on a reactor
  momentum signal (`entry_type='reactor_momentum'`) must not be closed by the
  LLM — only the monitor's rule-based exit (high-water-mark trail or N-bar trend
  break) closes it. The LLM is bad at holding; rule-based monitor exits are good
  at it.
- **Size into conviction**: a fully-confirmed signal (high news score + strong
  price + volume) may size larger than a scheduled-cycle entry.
- Intraday only (must close by EOD), but **within the day, hold as long as the
  trend works** — not day-trading in the fast-out sense.
- **The single failure mode to design against**: selling a winner because the
  next 1-min candle is red. Exits require a *structural* break (e.g. close below
  VWAP/20-EMA for N consecutive candles, or drawdown >X% from post-entry high) —
  never any pullback.

## LLM provider: GLM-5.2 (and don't undo it)

Sole provider since the 2026-07-01 cutover (commit `d42c5aa`) — OpenAI /
Anthropic / Ollama were removed entirely. `core/llm/glm.py:GLMLLM` speaks any
OpenAI-compatible endpoint; **OpenRouter is the default** and the deliberate
choice (verified research verdict, 2026-07-01):

- GLM-5.2 is at parity with the top OpenAI model on independent evals at ~1/5
  the price. But the real reason for OpenRouter over Z.ai-direct is **multi-host
  failover**: MIT-weight GLM is served by many hosts, so `FallbackLLM` can chain
  a second GLM endpoint — a structural fix for single-provider outages that an
  OpenAI-exclusive model can't offer. Z.ai's own API had near-total 429 outages
  (2026-06-15/17); the GLM Coding Plan's ToS forbids SDK/bot use. **Do not
  "simplify" back to a single vendor.** (Exclude SiliconFlow + Databricks hosts
  — missing JSON mode / function calling.)
- **The bot won't start without `GLM_API_KEY`** in `.env` (an OpenRouter key by
  default). Load-bearing compat points on the hot path: forced
  `tool_choice=submit_decisions` and `response_format=json_object`. Failure mode
  is **silent no-action cycles**, not crashes — smoke-test the first live cycle
  after any provider/env change. GLM thinking is OFF by default (latency/cost).
- If 429s ever return: an OpenAI-style `insufficient_quota` / OpenRouter 402
  "Insufficient credits" is a **billing** problem (top up), not a rate limit —
  switching providers won't fix it. The strategy LLM now fires a rate-limited
  Telegram alert on credit exhaustion (`llm.quota_exhausted`).

## Open operator-gated issues (you cannot fix these in code)

1. **Halal screening runs in Zoya SANDBOX** (`ZOYA_USE_SANDBOX=true`) — verdicts
   are *randomised*, not real Shariah screening. The universe is tiny (~3 of 20
   "halal": AAPL/ADBE/INTU are arbitrary sandbox output), which is the root cause
   of the stock bot's **symbol fixation** — the LLM funnels every entry into
   those names because they're the *only* ones `get_halal_symbols()` returns.
   Ironic corollary: with **no** Zoya key, `halal/cache.py` seeds the full
   20-symbol AAOIFI `DEFAULT_HALAL_SYMBOLS` — a *better* universe than the
   sandbox's 3. Fix is operator-only: a paid Zoya **production** key +
   `ZOYA_USE_SANDBOX=false`. Until then, universe breadth is capped by the
   screener, not the LLM — don't chase "symbol diversity" prompt tweaks; they
   can't help.
2. **Chronic ~100% stock reconcile drift** is **ledger hygiene, not a trading
   bug** — the cycle and the monitor both decide off *broker truth*, not the DB
   ledger. Root cause was inconsistent exit recording; it's fixed *forward* (the
   monitor now writes a `filled` SELL row on every exit and clamps sells to the
   broker-held long qty, so it can't go short even on a corrupted ledger). A
   historical backlog of phantom rows remains. Clearing it needs
   `halal-trader reconcile fix-drift --apply`, which is **DESTRUCTIVE** (writes
   synthetic RECONCILE-ADJ rows) and **operator-gated — do NOT auto-run**;
   dry-run (no `--apply`) is safe to preview. **Do not unilaterally change
   `core/reconcile.py:_aggregate_stocks_positions`** — it's shared with the
   destructive fix-drift tool and an "open-buys-only" detection model breaks
   fix-drift's reduce path. See git history around commits e685886/6b35d69 for
   why a clean-looking refactor was reverted.
3. **OpenRouter credits** — the GLM-era equivalent of the old OpenAI quota. Watch
   for 402 "Insufficient credits"; that's a top-up, not a code fix.

## The `src/halabot` shadow engine — safety + lessons

A strangler-fig rebuild (`src/halabot/`, sibling to legacy `halal_trader/`,
shared Postgres via isolated `hb_` tables outside the Alembic chain). Spec:
`docs/REARCHITECTURE.md`. An always-on **Sense→Understand→Convict→Act** engine
around a persistent per-asset `BeliefState`.

**Safety invariant — do not violate:** the engine is **shadow-only / read-only**;
the `execution/` layer is **DORMANT** (`app.build_engine` never imports it — a
dormancy test enforces this). Live trading only arms via `ENGINE_LIVE` + a dated
`ENGINE_LIVE_TOKEN`, and only after the Phase-3 significance gate passes. **Keep
`ENGINE_LIVE` unset.** Halal is a hard gate on entry AND holds. Quality bar:
`ruff`/`mypy`(strict)/`pytest tests/halabot/` all green (needs Postgres :5433).

**Hard-won engineering lessons (paid for once — don't repeat):**

- **Validate every edge change with `halabot backtest` before shipping it
  default-on.** The bar-cache (`--cache-write`/`--cache-read`) + disjoint
  out-of-sample splits (`--oos-splits N`) are the methodological backbone —
  they killed re-fetch drift between A/B arms (comparing across separate live
  invocations is INVALID — each re-fetches different bars) and exposed
  nested/overlapping windows masquerading as independent evidence.
- **Nested/overlapping windows oversell.** Always OOS on *disjoint* windows
  before trusting a default-on change.
- **Market-relative signals pay off; per-asset technicals don't.** What SHIPPED
  ON and survived disjoint-OOS: the **market-regime gate** ("don't fight the
  tape" — block buys while SPY < its 50-bar SMA) and **relative-strength vs
  SPY**. What was **NO-GO**: a structural/breakout regime signal (Donchian
  breakout entries *lose* in 4/5 windows), and the Appendix-H exit ladder (a
  trailing stop fights the conviction-decay slow-out and exits winners early —
  conviction-decay IS the slow-out).
- **Conviction is already near-optimal; the edge is better INPUTS, not
  reshaping conviction.** Conviction predicts wins (sharp threshold ~0.40) but
  every *mechanical* lever to exploit it fails its A/B (convex sizing, raising
  the entry band, forecaster reweighting). Don't chase post-hoc conviction
  reshaping; add better signals instead.
- **Chronos foundation-model forecaster** (`forecaster_enabled`, `[ml]` extra)
  shipped ON — cleanest OOS profile of any edge (helps in aggregate, never
  hurts a disjoint window); loads lazily and degrades to a silent no-op without
  `[ml]`/weights, so default-on can't crash the engine.
- **Known limitation**: `EvidenceRegimeClassifier` regime is *circular at entry*
  — `signed>0 → TRENDING_UP` is derived from the same weighted evidence sum that
  drives conviction, so every entry is `trending_up` (BREAKOUT never emitted).
  This is why market-relative + structural signals were explored as
  non-circular inputs.
- **The merge-dedup bug**: evidence merge must key on `(source, event_id)`, not
  `event_id` alone — the latter silently dropped all-but-the-first interpreter's
  evidence per bar (the engine ran on ~1 signal). Don't reintroduce it.
- **Finding dead code here**: run static import-reachability from the entrypoints
  (`halal_trader.cli`, `halabot.cli`/`app`, the FastAPI `create_app`), NOT grep —
  a large removed "wave" dead-code layer cross-referenced itself, so grep shows
  false "live". (That cleanup already happened; `halal_trader` is the actual live
  single-user paper bot only.)
- The LLM everywhere in halabot is now **GLM-5.2** (via `create_llm`), not the
  OpenAI backend the older code comments mention. The sparse LLM touches (thesis
  writer, news scorer) degrade to no-ops if the LLM can't init (INV-1: an
  LLM-down run still updates beliefs).

## Advisory features — these NEVER trade

- **Daily "stock of the day" recommendation** (`recommendation/engine.py`, CLI
  `halal-trader recommend`, `/api/recommendation`, dashboard page, 09:05 ET job):
  an LLM picks the single most-promising halal stock from the curated AAOIFI-20
  universe (deliberately decoupled from the randomised sandbox Zoya screener so
  "most promising" spans a real opportunity set). Advisory only — kept out of the
  execution path entirely.
- **Belief Board** (`/beliefs` on the :8082 dashboard): renders the shadow
  engine's live per-asset beliefs + decision stream. Advisory only.
