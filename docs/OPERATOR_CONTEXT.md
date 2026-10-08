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
  throughput): paper only until a strategy passes the written capital gates
  (below); halal compliance is non-negotiable (long-only, no short/interest/
  leverage/derivatives); never destabilize the live bot; the `src/halabot`
  engine never trades (see below).

## Direction (decided 2026-10-01)

- **Halal stock trading is the product and is being developed full-time**;
  the refactor aims to *improve* it (edge, universe, execution), not only
  harden it. The roadmap and the decisions behind it are in the local-only
  `docs/MODERNIZATION_PLAN.md` (§9); see CLAUDE.md for why it isn't committed.
- **Crypto trading is abandoned**; its code is deleted and, since 2026-10-08,
  its tables dropped.
- **Real capital only through staged, pre-signed gates** (paper → small live →
  scale; thresholds in the plan). Until the first gate is signed, everything
  stays paper. A live account would be a cash account (no margin).
- **Free or cheap services first**, paid upgrades only when evidence shows
  they are the binding constraint.
- **Single developer, no PR process**: commit straight to `main`; the
  pre-commit hook and CI are the gate.

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
- ~~Intraday only (must close by EOD)~~ **Superseded 2026-10-02 for event
  trades:** event-driven positions (the reactor, and S2 when it lands) hold
  for **days to weeks**, as long as the trend works. The research on event
  drift points there, and a live account is a cash account, where quick
  round trips on unsettled funds draw broker violations. The scheduled LLM
  day-trader keeps its intraday flatten as the control strategy.
- **The single failure mode to design against**: selling a winner because the
  next 1-min candle is red. Exits require a *structural* break (e.g. close below
  VWAP/20-EMA for N consecutive candles, or drawdown >X% from post-entry high) —
  never any pullback.

## Operator decisions of 2026-10-02

- **Halal screen: the strict option.** The in-house screen applies the
  stricter of AAOIFI and S&P Shariah on every axis (business activities,
  ratios), and treats an index Shariah board's exclusion as a veto. False
  exclusions are acceptable; false passes are not.
- **LLM budget ($50/month OpenRouter key limit):** $25 live (day-trader,
  reactor, shadow), $15 research, $10 headroom.
- **News:** live news moves to Alpaca (Benzinga), the feed the research
  is backtested on; Finnhub stays as a fallback.

## Decisions taken under the operator's delegation (2026-10-04)

The operator asked the agent to take the open decisions itself and to
automate as much as possible. Recorded here so they can be revisited:

- **LLM budget enforced** (`LLM_BUDGET_ENFORCE=true`): a week of observe mode
  peaked at $0.58/day for the live pool against its $3 daily cap; pools are
  $25 live / $15 research per month.
- **The LLM day-trader is retired** (`DAY_TRADER_ENABLED=false`), not halted:
  +2.5%/yr vs SPUS 18.6%. Its exits and the EOD flatten still run. The
  kill-switch stays for emergencies and stops every strategy. The core under
  a halt buys nothing but still sells what the screen no longer holds halal
  (alerted as `core.halted_sells`); `halt --close-all core` liquidates it.
- **The core's live-money gate** (portfolio/readiness.py, checked every
  evening): >= 20 trading days on paper, a run on each of the last 20, a
  monthly rebalance that included sells, tracking error vs its forward book
  < 3%/yr and a cumulative gap < 1 point (both from pre-trade equity), no
  refused, unfilled or partly filled order and no halted run in that window.
  A `core.ready` alert asks for the live keys. The switch (live keys +
  `CORE_PAPER=false` + today's `CORE_LIVE_CONFIRMATION`) is the operator's
  step; the bot and `core run` refuse it unless the paper gate passed, the
  account is a cash account (multiplier 1, no shorting) and its keys are not
  the day-trader's. The bot reads the dated token once, at start: a restart
  on a later day leaves the live core refusing (alerted as `core.refused`)
  until the token is renewed. The first live stage holds at most
  `CORE_LIVE_MAX_NOTIONAL` invested. Live history is recorded as account
  `core-live`, apart from paper's `core`.
- **Paper purification is a rehearsal**: shown, but the compliance status
  only turns to "attention" for an account trading real money.
- **No key rotation needed**: the only tokens in old local logs are retired
  ones; the current Finnhub and FRED keys never reached a log.

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

1. ~~Halal screening runs in Zoya SANDBOX~~ **Resolved 2026-10-06:** every
   halal gate reads the strict in-house screen (`halal_screen_results`, weekly
   in the evening research run): the core's order boundary, and since then the
   day-trader's and reactor's (`halal/strict.py` via `HalalScreener`), whose
   universe is the screen's `HALAL_UNIVERSE_SIZE` largest halal names. A stale
   (>10 days) or missing screen makes nothing halal. Sandbox Zoya verdicts are
   ignored; a production key could only veto. The old curated 20-name list
   (`DEFAULT_HALAL_SYMBOLS`) is research-only now: seven of its names fail the
   strict screen.
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
  an LLM picks the single most-promising halal stock from the 30 largest names
  the strict in-house screen passes (no pick when the screen is stale). A failed
  run retries once after 15 minutes. Advisory only — kept out of the execution
  path entirely.
- **Belief Board** (`/beliefs` on the :8082 dashboard): renders the shadow
  engine's live per-asset beliefs + decision stream. Advisory only.
