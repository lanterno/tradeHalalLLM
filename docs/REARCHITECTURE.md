# Halal Trader — Re-Architecture & Implementable Spec

> **Status:** Proposal / design — not yet implemented.
> **Scope:** Replace the per-cycle transactional trading loop with a continuously-maintained
> *market-understanding engine*. Trades become a consequence of conviction, not of cycle cadence.
> **Constraints (unchanged):** paper/testnet only — never real money; halal compliance is non-negotiable.
> **Decisions:** the four forks are **locked** (defaults adopted 2026-05-28) — see Part V.

This document has five parts:

1. **Vision & rationale** — why, and the current-state assessment.
2. **Cross-cutting contracts** — the event model, the canonical `BeliefState`, config, and data model that every layer depends on.
3. **Layer-by-layer implementable specs** — `platform → perception → belief → cognition → conviction → policy → execution → risk → learning → api`, each with types, algorithms, persistence, reuse map, invariants, and tests.
4. **Migration plan** — strangler-fig, paper-safe, with per-phase acceptance criteria.
5. **Risks, forks, open questions.**

---

# Part I — Vision & Rationale

## 1. The reframe

Today the system is a **transactional decision engine**: every cycle it rebuilds a large prompt and asks an LLM *"what should I buy or sell right now?"* The overhaul inverts the dependency:

> **Maintain a persistent, evolving model of the market (a `BeliefState`). Trades fall out as the *delta* between the portfolio we hold and the portfolio our conviction implies. Understanding is the product; trades are a side-effect.**

This directly serves the operator's stated shift: *"a bot that is always on, checks the news, and triggers based on what it learns … the focus shouldn't be to buy and sell but to understand the market."*

## 2. Current-state assessment (grounded)

| Signal | Number | Implication |
|---|---|---|
| Source code | 118.7k LOC, 468 files | Past "lean and composable" |
| Test code | 139k LOC, 474 files | Heavy to change |
| `web/`+`marketplace/`+`community/`+`education/` | ~26k LOC | A half-built SaaS platform bolted on |
| `halal/` | 18.6k LOC | Screening sprawled into a compliance *platform* |
| Stocks vs crypto | 2× everything | Duplicated scheduler/cycle/executor/monitor/strategy/portfolio |
| Live result (2026-05-27) | −$292, 40% win, 33 trades | Per-cycle framing *causes* churn |

**Four structural problems the overhaul must solve:**

1. **Transactional framing → churn.** A stateless "re-decide from scratch every 15 min" loop has no memory of *why* it holds a position, so it whipsaws. The min-hold gate, recent-close cooldown, and stop-loss-reentry gate added this week are all band-aids over the absence of a persistent thesis.
2. **Understanding exists but is subordinate.** Regime memory, RAG over rationales, calibration, drift, multi-timeframe, forecaster, catalysts, sentiment velocity are computed *only to stuff a prompt and then discarded*. There is no durable market model.
3. **Fragility from accretion.** Single-LLM-provider outage = bot dark; halal-cache poisoning; reconcile-drift accumulation; blank error logs; test→prod log pollution; an SL/TP monitor that was never wired in; $0 EOD closes.
4. **Scope sprawl drowns the edge.** The investing intelligence is ~30k of 118k LOC.

## 3. Organizing principle: **Sense → Understand → Convict → Act**

```
                         ┌───────────────────────────────────────────┐
   EVENTS (durable bus)  │            BELIEF STATE (world model)       │
   price ticks ───────►  │  per-asset: regime, direction, conviction,  │
   news items  ───────►  │  thesis, key levels, pending catalysts,     │  ◄── the heart
   macro events ──────►  │  evidence trail, horizon, confidence, ver   │
   fills / pnl ───────►  │  portfolio: heat, drawdown, correlation      │
                         └───────────────────────────────────────────┘
        ▲                      ▲              │                 │
        │                      │              ▼                 ▼
   PERCEPTION            COGNITION        CONVICTION          POLICY+ACTION
   ingest →              update beliefs   beliefs →           target weights →
   Observations          from evidence    calibrated score    deltas → exec
                              ▲                                      │
                              └────────── LEARNING ◄─────────────────┘
                                outcomes recalibrate belief→conviction
```

**Always on, event-driven.** There is no fixed cycle. Information *arrives* (a price break, a news item, a macro release), updates the relevant belief, and *only if* conviction crosses a threshold does a trade result. A periodic "tick" still exists, but only as a low-priority heartbeat for time-decay and reconciliation — not as the decision trigger.

## 4. Design invariants (the resilience lessons, encoded as rules)

These are non-negotiable properties every layer must uphold. Each maps to a real incident from operations.

- **INV-1 — Degrade, don't die.** Deterministic cognition keeps beliefs current when the LLM is unavailable. An LLM outage stales the *narrative*, never the *beliefs* or risk management. *(Fixes: LLM-down-bot-dark.)*
- **INV-2 — Transient failures never mutate persistent state.** A screening/API error is a "no-verdict," never a verdict. It cannot poison a cache or flip a belief. *(Fixes: halal-cache poisoning.)*
- **INV-3 — The broker is the source of truth for positions.** The DB reconciles *to* the broker, never the reverse; reconciliation runs continuously, not once a day. *(Fixes: reconcile-drift accretion, $0 EOD closes.)*
- **INV-4 — Every external error is logged with its type.** No bare `str(e)`; always `repr`/type + payload. *(Fixes: blank Zoya/LLM error logs.)*
- **INV-5 — Replayability.** All inputs are durable events; beliefs and decisions are reconstructable from the log. *(Enables learning + debugging.)*
- **INV-6 — Tests never touch production state** (logs, DB, broker). *(Fixes: pytest→prod-log pollution.)*
- **INV-7 — Halal compliance is a hard gate, on entry AND on holds.** No order is placeable without a positive, fresh compliance verdict linked by FK; and a *held* position whose re-screen returns a real (non-transient) `not_halal`/`doubtful` is force-exited regardless of conviction (Appendix D, H). Profitable closes on impure-revenue names accrue purification. *(Unchanged in spirit; the holds + purification cases are made explicit — fix R-05/R-06.)*
- **INV-8 — Position provenance.** Every open position links to the belief version that opened it; exits are driven by *thesis invalidation*, not cycle re-rolls. *(Fixes: churn.)*
- **INV-9 — Real money requires an explicit, dated confirmation token + un-loosenable hard floors.** The engine refuses to arm live trading unless `LIVE_MODE_CONFIRMATION=I-UNDERSTAND-REAL-MONEY-<UTC-date>` is present, and in live mode a `LiveModeChecker` enforces max-account / max-single-order / daily-loss floors that config cannot relax. Flipping the paper flag off must never be a single silent env change. *(Preserves `safeguards.py`; fix R-04.)*
- **INV-10 — No implicit leverage.** Total target gross exposure is capped (`≤ max_gross_exposure ≤ 1.0` long-only); the per-asset target vector is normalized before deltas so simultaneous convictions can never sum past the cap. *(Fix R-03.)*

---

# Part II — Cross-cutting contracts

## 5. The event model + durable log

Everything that enters or moves through the system is an immutable, append-only **Event**. The event log is the system's spine (INV-5).

```python
# platform/events.py
EventType = Literal[
    # perception
    "observation.price", "observation.bar", "observation.news",
    "observation.macro", "observation.onchain", "observation.sentiment",
    # belief
    "belief.updated", "belief.thesis_refreshed", "belief.invalidated",
    # conviction / policy
    "conviction.scored", "policy.target_changed", "policy.trade_proposed",
    # execution
    "order.submitted", "order.filled", "order.rejected", "position.reconciled",
    # risk / ops
    "risk.state", "risk.halt", "compliance.verdict", "system.heartbeat",
]

@dataclass(frozen=True)
class Event:
    id: UUID
    type: EventType
    asset: str | None              # None for portfolio-level
    ts: datetime                   # event time (UTC)
    source: str                    # producing component
    payload: dict[str, Any]        # typed per EventType (see schemas/)
    causation_id: UUID | None      # the event that caused this one
    correlation_id: UUID           # groups a causal chain (e.g. a news → belief → trade flow)
    schema_version: int
```

**Bus contract** (`platform/bus.py`):

```python
class EventBus(Protocol):
    async def publish(self, event: Event) -> None: ...
    def subscribe(self, types: set[EventType], handler: Handler) -> Subscription: ...
    async def replay(self, *, since: datetime, types: set[EventType] | None = None) -> AsyncIterator[Event]: ...
```

- **Default impl:** in-process async fan-out **+** synchronous append to a Postgres `event_log` table before handlers run (durability first, then dispatch). This keeps the home-bot single-node simple while guaranteeing replay.
- **Upgrade path:** swap the impl for Redis Streams / NATS JetStream without touching subscribers (it's a Protocol). Defer until multi-process is actually needed.
- **Ordering:** per-`asset` ordering is guaranteed (single writer per asset partition); cross-asset ordering is best-effort.
- **Backpressure:** handlers that fall behind are measured (lag metric); the bus never blocks perception — slow handlers get a bounded queue and shed `observation.price` first (it's the most replaceable).

`event_log` table:

| column | type | notes |
|---|---|---|
| `id` | uuid pk | |
| `type` | text, indexed | |
| `asset` | text, indexed (nullable) | |
| `ts` | timestamptz, indexed | event time |
| `ingested_at` | timestamptz default now() | |
| `source` | text | |
| `payload` | jsonb | |
| `causation_id` / `correlation_id` | uuid | |
| `schema_version` | int | |

Retention: partition by month; keep raw `observation.price` 14 days, everything else 1 year (beliefs/decisions/outcomes are the training corpus).

## 6. The canonical `BeliefState`

```python
# belief/schema.py
class Regime(StrEnum):
    TRENDING_UP = "trending_up"; TRENDING_DOWN = "trending_down"
    RANGING = "ranging"; VOLATILE = "volatile"; BREAKOUT = "breakout"

class Direction(StrEnum):
    LONG_BIAS = "long_bias"; NEUTRAL = "neutral"   # long-only universe (halal)

class Horizon(StrEnum):
    INTRADAY = "intraday"; SWING = "swing"; POSITION = "position"

@dataclass
class Levels:
    support: float | None; resistance: float | None
    stop: float | None; invalidation: float | None   # thesis dies if breached

@dataclass
class EvidenceItem:
    source: str                 # "regime" | "forecaster" | "news" | "macro" | "sentiment" | "rag" | "thesis"
    direction: float            # signed contribution in [-1, +1]
    weight: float               # 0..1 reliability/recency weight
    detail: str                 # human-readable
    ts: datetime
    event_id: UUID | None       # provenance into event_log

@dataclass
class BeliefState:
    asset: str
    regime: Regime
    regime_confidence: float            # 0..1
    direction: Direction
    conviction: float                   # 0..1, CALIBRATED — used by policy sizing (see conviction/)
    conviction_raw: float               # 0..1, PRE-calibration — used by material_shift's raw-vs-raw band test (fix R-11)
    horizon: Horizon
    thesis: str                         # LLM narrative; refreshed only on material shift
    levels: Levels
    catalysts_pending: list[Catalyst]   # earnings/FOMC/etc. with timing + expected impact
    evidence: list[EvidenceItem]        # decaying window; the "why"
    halal: ComplianceVerdict            # cached, transient-error-safe (INV-2); re-screened on holds (INV-7)
    opened_trade_ids: list[int]         # provenance (INV-8)
    last_updated: datetime
    last_thesis_refresh: datetime | None   # None until the first thesis is written (material_shift guards on this)
    version: int                        # bumped on every persisted mutation
```

**Lifecycle (state machine):**

```
            new evidence (cheap models)            conviction ↑ over entry band
NEUTRAL ───────────────────────────────►  FORMING ──────────────────────────►  CONVICTION_LONG
   ▲                                          │  thesis refresh (LLM) on material shift  │
   │ conviction decays / invalidation         ▼                                          │
   └───────────────────  INVALIDATED  ◄──── price breaks `levels.invalidation` ──────────┘
                              │ emits belief.invalidated → policy forces exit
```

Beliefs are **versioned rows**, not mutated in place — `belief_state` keeps `(asset, version)` history so we can reconstruct "what did we believe at T and why" (INV-5) and link positions to the exact belief that opened them (INV-8).

## 7. Configuration surface (new, narrowed)

The new engine has a small, typed settings tree (Pydantic Settings, reusing `config.py` patterns). Illustrative top-level groups:

```
ENGINE_         universe, tick_interval_s, max_open_positions, heartbeat_interval_s
BELIEF_         evidence_decay_halflife_min, evidence_decay_trading_time (bool, default true),
                conviction_entry_band, conviction_exit_band, bootstrap_window_min,
                thesis_max_age_h, catalyst_impact_threshold
COGNITION_      enabled_models (regime,forecaster,anomaly,...), llm_thesis_enabled
CONVICTION_     calibration_model, min_samples_to_calibrate, win_threshold_pct
POLICY_         max_weight_per_asset, max_gross_exposure (≤1.0), max_sector_pct,
                min_hold_minutes, stop_loss_reentry_cooldown_minutes, target_rebalance_threshold
RISK_           max_portfolio_heat_pct, max_drawdown_pct, daily_loss_limit, correlation_threshold
EXEC_           venue (alpaca|binance), min_notional_usd, reconcile_interval_s,
                per_asset_breaker_threshold, per_asset_breaker_cooldown_s
LLM_            provider, fallback_providers, daily_usd_cap, news_daily_classify_cap,
                circuit_breaker_threshold   (cost cap + quota/circuit breaker — fix R-15)
HALAL_          screener (zoya|coingecko), cache_ttl_h
SAFEGUARD_      live_mode (default paper), live_max_account_usd, live_max_order_usd,
                live_daily_loss_floor   (un-loosenable in live mode — INV-9, fix R-04)
```

New vs the first draft (each closes a review finding): `max_gross_exposure` (R-03), `RISK_daily_loss_limit` (R-10), `evidence_decay_trading_time` (R-09), `heartbeat_interval_s` (R-08), `LLM_daily_usd_cap`/`news_daily_classify_cap`/`circuit_breaker_threshold` (R-15), `EXEC_per_asset_breaker_*` (per-asset breaker), `CONVICTION_win_threshold_pct` (label), `SAFEGUARD_*` (R-04).

Every field documented in `.env.example`; settings-parity test enforced (kept from today).

## 8. New data model (additive over today's schema)

| table | purpose | key columns |
|---|---|---|
| `event_log` | the spine (§5) | see above |
| `belief_state` | versioned world model | `asset, version, regime, conviction, direction, horizon, thesis, levels(jsonb), halal_verdict, updated_at` |
| `belief_evidence` | evidence trail per belief version | `belief_id fk, source, direction, weight, detail, event_id` |
| `belief_state` | (cont.) also stores `conviction_raw` so `material_shift` compares raw-to-raw (fix R-11) | |
| `conviction_score` | calibrated score history (telemetry, **not** calibration input — fix R, leakage) | `asset, ts, raw_score, calibrated, features(jsonb), belief_version` |
| `target_weight` | policy output history | `asset, ts, target_weight, current_weight, reason, belief_version` |
| `outcome` | closed-position outcome linked to belief | `trade_id, asset, belief_version, entry_belief(jsonb), return_pct, realized_pnl_usd, impure_ratio, hold_seconds, exit_reason, label(int NOT NULL)` |

Reused as-is: `trades`/`crypto_trades` (now carry a `belief_version` FK **and** an `engine_owner` tag for migration-time reconcile scoping — fix R-02), `halal_screenings`, `regime_memory`, `rationales` (pgvector), `daily_pnl`. Kept (compliance, non-negotiable): `sharia_exceptions`, `purification_entries` — the purification ledger stays in the trading repo (fix R-06). Moved with their code (to `halabot-platform`): the SaaS/marketplace/education tables only.

---

# Part III — Layer-by-layer implementable specs

Each layer is a package with a single responsibility, a Protocol-defined boundary, and no upward dependencies (perception knows nothing of policy, etc.). Layers communicate **only via the bus + the belief store**.

---

## L0 — `platform/` (foundation)

**Purpose:** the bus, durable event log, task supervision, config, observability, and DB session management. Everything else depends on this and nothing else.

**Key interfaces:**

```python
class Clock(Protocol):          # injectable — tests use a fake (INV-6)
    def now(self) -> datetime: ...

class Supervisor(Protocol):     # structured concurrency (reuse core/supervisor.py / anyio)
    async def spawn(self, name: str, coro: Awaitable) -> None: ...
    async def shutdown(self) -> None: ...
```

**Responsibilities & reuse:**
- Bus + `event_log` (new; §5).
- Supervisor ← `core/supervisor.py` (anyio task groups).
- Config ← `config.py` (narrowed tree, §7).
- Observability ← `core/observability.py` (ContextVars: `correlation_id` auto-attached to logs from the current event), `core/tracing.py` (OTel spans per event-handler), `core/metrics.py` (Prometheus).
- DB ← `db/` (engine, Alembic; `RepoBundle` pattern retained).

**Invariants:** INV-4 (typed error logging is a logging filter here), INV-5 (durable-append-before-dispatch), INV-6 (injectable `Clock`, `LOG_DIR` redirect in test conftest — already done).

**Tests:** bus delivers to subscribers; durable-append happens before handler runs; replay returns events in `ts` order; a slow handler doesn't block `publish`; supervisor cancels cleanly.

---

## L1 — `perception/` (ingest → Observations)

**Purpose:** turn the outside world into typed `observation.*` events. **No interpretation** — perception reports facts, cognition interprets them.

**Boundary:**

```python
class Source(Protocol):
    name: str
    async def run(self, emit: Callable[[Event], Awaitable[None]]) -> None: ...
    # long-lived; emits observation.* events; supervised; restart-safe
```

**Sources (each a thin adapter; all existing collectors fold in):**

| Source | Emits | Reuse from |
|---|---|---|
| `market_data/bars` | `observation.bar` | `trading/bars.py`, `crypto/websocket.py`, `mcp/client.get_stock_snapshot` |
| `market_data/ticks` | `observation.price` | Alpaca/Binance streams |
| `news/finnhub` | `observation.news` | `sentiment/stocks_events.py` (the reactor's fetch+classify) |
| `news/cryptopanic` | `observation.news` | `sentiment/cryptopanic.py` |
| `news/edgar`,`fed`,`reddit` | `observation.news` / `.sentiment` | `trading/edgar_catalysts.py`, `trading/fed_speak.py`, `sentiment/reddit*.py`, `sentiment/velocity.py` |
| `macro/fred` | `observation.macro` | `trading/fred_catalysts.py`, `trading/options_iv.py` |
| `onchain` | `observation.onchain` | crypto whale/basis collectors |

**Observation payload contracts** (`schemas/observations.py`), e.g.:

```python
class NewsObservation(TypedDict):
    asset: str; headline: str; summary: str; url: str; published_at: str
    source: str; raw_sentiment: float | None  # lexicon score; LLM scoring is COGNITION, not here
```

**Key design change vs today:** the news *reactor's LLM classification moves OUT of perception into cognition*. Perception just emits the raw headline; cognition decides what it means and how much it should move a belief. This decouples "we saw news" (cheap, always works) from "we understood it" (LLM, may be down — INV-1).

**Invariants:** INV-2 (a source error emits nothing and self-restarts; it never emits a degraded/placeholder observation that downstream could mistake for signal), INV-4 (typed error logs), dedup by `(source, asset, url|hash)` persisted to survive restarts (the reactor-state persistence we built generalizes here).

**Tests:** each adapter maps a fixture payload → correct Observation; a source exception is caught, logged with type, and the source restarts; dedup survives a simulated restart.

---

## L3 — `belief/` (the world model — store + updater) ★

> Specced before cognition/conviction because everything reads/writes it.

**Purpose:** own the `BeliefState` per asset: persist it (versioned), apply incremental updates from evidence, and emit `belief.*` events on material change.

**Store:**

```python
class BeliefStore(Protocol):
    async def get(self, asset: str) -> BeliefState | None: ...
    async def get_version(self, asset: str, version: int) -> BeliefState | None: ...
    async def put(self, belief: BeliefState) -> int: ...   # writes a new version, returns it
    async def all_active(self) -> list[BeliefState]: ...
```

**Updater (the core algorithm):** evidence-driven, incremental, deterministic-first.

```python
# `now` is injected (platform.Clock) so replay/bootstrap can pass event-time, not wall-time (Appendix F).
async def apply_evidence(asset: str, items: list[EvidenceItem], now: datetime) -> None:
    b = await store.get(asset) or BeliefState.neutral(asset)
    prev = deepcopy(b)                                # ★ snapshot BEFORE mutation for material_shift (fix R-11)
    # 1. decay existing evidence by *trading-time* recency (half-life from config).
    #    decay() uses a trading-clock (market-closed gaps frozen) so an overnight/weekend
    #    gap doesn't annihilate evidence and force a Monday-open mass exit (fix R-09).
    b.evidence = decay(b.evidence, now, halflife=cfg.evidence_decay_halflife_min)
    b.evidence = merge(b.evidence, items)            # dedups by event_id, bounds per-source (fix R, idempotency)
    # 2. recompute deterministic fields from the weighted evidence vector
    b.regime, b.regime_confidence = regime_from_evidence(b.evidence)
    b.direction = LONG_BIAS if weighted_sum(b.evidence) > cfg.long_threshold else NEUTRAL
    b.levels = level_engine.update(asset, b.levels)   # support/resistance/invalidation from bars
    # 3. raw conviction (pre-calibration), LLM-free — ONE source of truth, the evidence list.
    #    drift/anomaly flags flow from the L2 interpreters' latest evidence so their
    #    down-weighting actually reaches conviction (fix R-12).
    raw = conviction_raw(b.evidence, b.regime_confidence,
                         drift_flag=has_flag(b.evidence, "drift"),
                         anomaly_flag=has_flag(b.evidence, "anomaly"))
    b.conviction = await calibrator.calibrate(asset, raw, features=feature_vec(b))  # L4
    # 4. thesis refresh ONLY when material (cost control + INV-1). Compares the PREVIOUS
    #    persisted belief to the new RAW score on a single (raw) scale (fix R-11).
    if (cfg.llm_thesis_enabled and llm.available() and not llm.breaker_open()   # quota/circuit breaker (fix R-15)
            and material_shift(prev=prev, new_raw=raw, new_regime=b.regime,
                               now=now, has_open_position=has_position(asset))):
        b.thesis = await thesis_writer.write(b)       # L2 cognition (LLM, sparse)
        b.last_thesis_refresh = now
    # 5. invalidation check (only against a *live* price, never a replayed historical one — Appendix F)
    px = last_price(asset)
    if not is_replay(now) and b.levels.invalidation and px is not None and px < b.levels.invalidation:
        await bus.publish(Event("belief.invalidated", asset, ...))
    v = await store.put(b)                            # new version
    await bus.publish(Event("belief.updated", asset, payload=summary(b), ...))
```

**`material_shift`** (governs LLM spend, full signature in Appendix B.3): true when regime flips, the **raw** conviction crosses a band edge relative to the **previous persisted belief's raw band** (the previous raw is persisted alongside the calibrated value so the comparison is raw-vs-raw — fix R-11), a high-impact catalyst lands, or it's been > N hours since the last refresh while a position is open. Otherwise the thesis is *not* re-written — beliefs still update via the cheap path.

**Persistence:** `belief_state` + `belief_evidence` (§8). Every `put` is a new version row; `belief_state` stores **both** `conviction` (calibrated) and `conviction_raw` (so `material_shift` compares raw-to-raw on the next update — fix R-11).

**Invariants:** INV-1 (steps 1–3 are LLM-free; step 4 is the only LLM touch and is triple-guarded: `llm_thesis_enabled`, `llm.available()`, and `not breaker_open()`), INV-5 (versioned + evidence carries `event_id`; replay-safe via Appendix F), INV-8 (`opened_trade_ids` maintained by execution callbacks).

**Tests:** evidence decay reduces stale weight; conflicting evidence nets correctly; conviction monotonic in aligned evidence; thesis NOT refreshed below the material-shift threshold; `belief.invalidated` fires when price breaks invalidation; version increments on each put; an LLM-down run still produces full belief updates minus thesis.

---

## L2 — `cognition/` (interpret observations → evidence)

**Purpose:** subscribe to `observation.*`, produce `EvidenceItem`s, and hand them to the belief updater. This is where *understanding* is manufactured. Split into **cheap/continuous** (deterministic) and **expensive/sparse** (LLM).

**Cheap, continuous interpreters** (run on every relevant observation, no LLM):

| Interpreter | Consumes | Produces evidence | Reuse |
|---|---|---|---|
| `indicators` | `observation.bar` | momentum/RSI/MACD/BB/ATR-derived direction | `crypto/indicators.py` |
| `regime` | bars + indicators | regime + confidence; updates `regime_memory` | `RegimeDetector`, `ml/regime_memory.py`, `ml/causal_regime.py` |
| `multiframe` | multi-tf bars | trend-alignment score | `trading/timeframes.py` |
| `forecaster` | bars | expected move + confidence | `ml/forecaster.py` (Chronos) |
| `anomaly` | indicator vector | anomaly flag (down-weights conviction) | `ml/anomaly.py` |
| `drift` | feature stream | concept-drift flag (widens uncertainty) | `ml/drift.py` |
| `news_lexicon` | `observation.news` | quick polarity (pre-LLM) | `sentiment/headline_polarity.py`, `sentiment/scoring.py` |

**Expensive, sparse interpreters** (LLM; invoked by the belief updater's `material_shift`, not per-observation):

| Interpreter | Role | Reuse |
|---|---|---|
| `thesis_writer` | synthesize the evidence into a narrative + set `horizon`, refine `levels.invalidation` | `core/llm/*` (ensemble/adversarial as quality options), prompt registry |
| `news_analyst` | score a *high-impact* headline's belief impact (only headlines the lexicon flags as potentially material) | the LLM classifier from `sentiment/stocks_events.py`, now belief-aware |
| `rag_grounding` | pull analogous past situations to weight conviction | `core/llm/rag_db.py`, `rationales` pgvector |

**Boundary:**

```python
class Interpreter(Protocol):
    consumes: set[EventType]
    async def interpret(self, obs: Event) -> list[EvidenceItem]: ...
```

**Key design change vs today:** the LLM is demoted from *decider* to *synthesizer*. ~95% of cognition is deterministic and free; the LLM adds narrative + horizon + invalidation level on material shifts only. This is the cost reduction (~10×) and the resilience win (INV-1).

**Invariants:** INV-1 (cheap interpreters carry the system when LLM is down), INV-4. A failed interpreter yields zero evidence (never a fabricated signal).

**Tests:** each interpreter maps fixtures → expected evidence sign/weight; the LLM interpreters are *not* called for immaterial observations; the chain produces a complete evidence set with the LLM mocked as unavailable.

---

## L4 — `conviction/` (belief → calibrated score)

**Purpose:** turn the belief's raw signal into a **calibrated probability of a favorable move over the belief's horizon**, so sizing is principled rather than vibes.

```python
class Calibrator(Protocol):
    async def calibrate(self, asset: str, raw: float, *, features: dict) -> float: ...
    async def fit(self, samples: list[CalibrationSample]) -> None: ...  # from learning/
```

- **Model:** start with **Platt/isotonic calibration** (reuse `ml/calibration.py`) mapping `raw_conviction → P(win | features)` fit on `outcome` rows. Until enough samples (`CONVICTION_min_samples_to_calibrate`), fall back to identity (raw == calibrated) so the system runs cold-start.
- **Features:** regime, regime_confidence, evidence composition, drift flag, anomaly flag, time-of-day, recent realized vol. Logged to `conviction_score` for every scoring (INV-5) so the learning loop has a training set.
- **Output:** `conviction ∈ [0,1]` written back onto the belief; also emits `conviction.scored`.

**Invariants:** cold-start safe (identity fallback); calibration only *re-weights*, never flips direction; a calibration model load failure degrades to identity + logs (INV-1/4).

**Tests:** identity fallback below min-samples; calibrated output monotonic in raw; a degenerate all-wins/all-losses training set doesn't produce NaNs; features persisted on every score.

---

## L5 — `policy/` (conviction → target weights → trade deltas) ★ anti-churn

**Purpose:** the deterministic bridge from understanding to action. Maps the set of beliefs to a **target portfolio** and emits trades *only as the delta* from the current portfolio. **This is what kills churn.**

```python
@dataclass
class TargetWeight: asset: str; weight: float; reason: str; belief_version: int

class Policy(Protocol):
    async def targets(self, beliefs: list[BeliefState], portfolio: PortfolioState,
                      risk: RiskState) -> list[TargetWeight]: ...
    async def deltas(self, targets: list[TargetWeight], portfolio: PortfolioState) -> list[TradeProposal]: ...
```

**Per-asset target weight** — there is exactly **one** definition (Appendix B.5), with hysteresis (entry band > exit band). The L5 view and B.5 are the same function; B.5 is canonical. Summary: `0` if not long-biased or below the hysteresis threshold (`exit_band` when already held, else `enter_band`), else `scale(conviction)` × correlation × volatility multipliers, capped at `max_weight_per_asset`. (Fix R-13: the earlier two-definition contradiction is resolved — B.5 only.)

**Portfolio normalization (gross-exposure cap — fix R-03):** the raw per-asset targets are a *vector*; before deltas, the portfolio target is normalized so total invested weight ≤ `cfg.max_gross_exposure` (≤ 1.0 for cash-only/long-only; no implicit leverage). Cash is modeled as the residual `1 - Σ weights`.

```python
def targets(beliefs, portfolio, risk) -> list[TargetWeight]:
    if risk.halted: return [TargetWeight(b.asset, 0.0, f"risk halt: {risk.reason}", b.version) for b in beliefs]
    raw = {b.asset: target_weight(b, risk, held=portfolio.holds(b.asset)) for b in beliefs}  # B.5
    gross = sum(raw.values())
    # If convictions collectively want more than max_gross_exposure, scale the WHOLE vector
    # down proportionally — never let Σ weights exceed the cap (fix R-03: no 6×0.20=1.20 leverage).
    if gross > cfg.max_gross_exposure:
        raw = {a: w * cfg.max_gross_exposure / gross for a, w in raw.items()}
    return [TargetWeight(a, w, "conviction", beliefs_by_asset[a].version) for a, w in raw.items()]
```

**Delta → trade (the churn killer), pending-order aware (fix R-14):**

```python
def deltas(targets, portfolio):
    for t in targets:
        # current_effective = filled weight + in-flight (open/partially-filled) order weight,
        # so a partial fill or a still-working order does NOT trigger a duplicate order (fix R-14).
        cur = portfolio.effective_weight(t.asset)        # filled + pending notional / equity
        gap = t.weight - cur
        if abs(gap) < cfg.target_rebalance_threshold:    # e.g. 0.25 of max_weight (and ≥ min hysteresis step)
            continue                                      # ← NO TRADE: belief didn't move the target enough
        if portfolio.has_open_order(t.asset):
            continue                                      # one working order per asset at a time; reconcile next tick
        side = "buy" if gap > 0 else "sell"
        yield TradeProposal(t.asset, side, notional=abs(gap)*equity, reason=t.reason, belief_version=t.belief_version)
```

Consequences, by construction:
- **No thesis change → no trade.** Yesterday's 33-trade churn is impossible: a held winner whose conviction is stable produces `gap ≈ 0`.
- **Fast in:** a conviction jump (e.g. a news-driven belief update) raises the target → immediate buy delta. The reactor generalizes into the core.
- **Slow out:** you trim/exit only when conviction *decays* or the belief is **invalidated** (`belief.invalidated` forces target → 0). No fixed take-profit yanking winners.
- **Falling-knife fix is free:** a stopped-out asset has a bearish/low-conviction belief → target 0 → the policy won't re-buy it until the belief itself turns. No separate stop-loss-reentry timer needed.

**Gates (`policy/gates.py`)** — the hard pre-trade checks, retained from today but now centralized (full ordered chain in Appendix B.6):
- **kill-switch** (`core/halt`) checked first.
- **risk halt** — portfolio heat / drawdown (L7).
- **daily-loss-limit** — realized intraday-loss floor (`|today realized P&L| / starting_equity ≥ cfg.daily_loss_limit` → block new entries; exits still allowed). This is a *realized*-loss gate distinct from unrealized heat/drawdown — without it a day of repeated stop-outs bleeds uncapped (fix R-10).
- **Halal** (INV-7): no buy without a fresh positive `ComplianceVerdict` + FK; and a held position whose re-screen returns a real `not_halal`/`doubtful` is force-exited (Appendix D, H — fix R-05).
- **min-hold**, **recent-close cooldown**, **stop-loss-reentry** — kept as *belt-and-suspenders* (the belief mechanism makes them rarely bind, but they're cheap safety).
- **market-close lockout**, **max positions**, **sector cap**.
- **venue feasibility** — min-notional **and** lot-size / step-size / min-qty / max-qty alignment from the venue's exchange filters, validated locally before submit so a non-conforming order is caught here, not bounced as a venue `-1013` (fix R, lot-size).
- **LLM cost cap** is enforced separately (not a per-trade gate): a cumulative-UTC-day spend tracker halts new LLM thesis/news calls when `LLM_daily_usd_cap` is crossed (fix R-15) — independent of the per-call `material_shift` throttle.

**Invariants:** INV-7, INV-8 (proposals carry `belief_version`), deterministic (no LLM in the action path — INV-1: you can always act on existing beliefs even if cognition's LLM is down).

**Tests:** stable conviction → zero deltas; conviction jump → buy delta sized correctly; `belief.invalidated` → full exit delta; gates reject appropriately; rebalance threshold suppresses sub-threshold churn; correlation/vol multipliers applied.

---

## L6 — `execution/` (venue-agnostic orders, fills, reconcile)

**Purpose:** execute `TradeProposal`s against a venue, confirm fills, keep the DB reconciled to broker truth, and manage open positions (SL/TP/trailing/trend-break). Merges the two executors + two monitors into one.

```python
class Venue(Protocol):                      # ports already exist
    async def place(self, order: Order) -> OrderResult: ...
    async def positions(self) -> list[Position]: ...
    async def close(self, asset: str) -> OrderResult: ...
    async def snapshot(self, asset: str) -> Quote: ...

class PositionManager:                       # one impl, both asset classes
    async def on_target_changed(self, proposal: TradeProposal) -> None: ...
    async def monitor_loop(self) -> None: ...     # SL/TP/trailing/trend-break (reuse trading/monitor.py logic)
```

**Reuse:** `trading/executor.py` + `crypto/executor.py` → one `orders.py` (the BaseExecutor sells-first orchestration, fill confirmation `core/fills.py`, slippage recording survive). `mcp/client.py`, `crypto/exchange.py` become `venues/`. `trading/monitor.py` + `crypto/monitor.py` → one monitor (trailing + the trend-break exit we added). `core/reconcile.py` (the broker-truth + reverse-orphan import + `fix-drift` we built this session) becomes the continuous reconciler.

**Key design changes:**
- **Continuous reconciliation** (INV-3), **engine-scoped**: a `reconcile` loop runs on `EXEC_reconcile_interval_s` *and* on every `order.filled`. **During migration (Phases 2–4) two engines share one broker account**, so reconcile is scoped by an **`engine_owner` tag** on every trade row (`"legacy"` | `"belief"`): the belief engine reconciles only positions it opened and treats `engine_owner="legacy"` broker positions as out-of-scope (and vice-versa). Without this, the broker-truth reconciler would see the *other* engine's position as a phantom and import/adjust it, and the two engines would fight over the same shares (fix R-02). Post-Phase-6 (single engine) the tag is dropped and reconcile reverts to whole-account.
- **Exit ownership:** the monitor is the *only* exit authority for thesis-driven holds (the policy emits target-0 on invalidation; the monitor executes SL/TP/trailing). No "LLM SELL" path — exits are belief- or rule-driven, eliminating the LLM/monitor conflict that caused the reactor-lockout hack.
- **Position ↔ belief link** (INV-8): every fill writes `belief_version` onto the trade row and back-references into `opened_trade_ids`.
- **Per-asset circuit breaker** (carried from today): N consecutive *unexpected* order errors on an asset within a window opens a per-asset breaker for a cooldown, so a single malfunctioning symbol (bad filter, venue glitch) is quarantined instead of retried indefinitely by the continuous target loop (fix R, per-asset breaker). `-1013`/`-2010` are rejections (not breaker trips); `-1003` triggers rate-limit backoff; quote/order fetches stay throttled (5 concurrent) — all carried from the current executors.

**Invariants:** INV-2 (a venue/quote error never fabricates a $0 close — the `_eod_exit_price` fix generalizes: skip rather than invent), INV-3, INV-4, INV-7 (gate checked in the order path too, as defense in depth).

**Tests:** fill confirmation populates submitted/filled; reconcile imports broker-only positions and neutralizes phantom DB nets; reconcile ignores the *other* engine's tagged positions during migration; no $0 synthetic exits; monitor fires SL/TP/trailing/trend-break; per-asset breaker opens after N unexpected errors and blocks retries until cooldown; concurrent buy/sell on one asset is serialized.

---

## L7 — `risk/` (portfolio-level)

**Purpose:** portfolio-wide guardrails that the policy consults and that can halt the system. Mostly a lift-and-shift of the existing engine.

```python
class RiskEngine(Protocol):
    async def evaluate(self, portfolio: PortfolioState, beliefs: list[BeliefState]) -> RiskState: ...

@dataclass
class RiskState:
    portfolio_heat_pct: float        # unrealized loss on open positions
    drawdown_pct: float              # peak-to-trough equity
    realized_loss_today_pct: float   # realized intraday P&L floor (fix R-10)
    gross_exposure: float            # Σ position weights (feeds the policy normalization, fix R-03)
    correlation: dict[str, float]; halted: bool; reason: str | None
    def correlation_multiplier(self, asset) -> float: ...
    def volatility_multiplier(self, asset) -> float: ...
```

**Three independent halt conditions** (any one halts new entries; exits always allowed):
1. **portfolio heat** > `max_portfolio_heat_pct` — *unrealized* loss on open positions.
2. **drawdown** > `max_drawdown_pct` — peak-to-trough equity.
3. **daily loss limit** — `realized_loss_today_pct ≥ daily_loss_limit` (fix R-10). This is the *realized* floor a day of repeated stop-outs trips even when heat and equity-drawdown stay low; it must be a first-class halt, not folded into heat.

**Reuse:** `crypto/risk.py` (correlation/heat/drawdown/vol scaling) + `trading/portfolio.py` (`should_halt_trading` realized-loss check) + `core/halt.py`. Emits `risk.state` / `risk.halt` events; the policy multiplies target weights by its multipliers, normalizes against `max_gross_exposure`, and refuses new targets when `halted`.

**Invariants:** halt is checked first in the action path; all three halts are belief-independent (a risk halt overrides any conviction). INV-3 (uses broker-truth equity/positions).

**Tests:** heat > limit halts new targets but still allows exits; drawdown halts all; **realized daily-loss floor halts even when heat/drawdown are under limit** (the repeated-stop-out scenario); correlation/vol multipliers shrink targets; gross-exposure feeds policy normalization; halt clears on recovery.

---

## L8 — `learning/` (close the loop)

**Purpose:** turn realized outcomes into improvements of the belief→conviction mapping and the thesis quality. This is *"based on what it learns."*

**Pipeline:**

```
position closed ──► outcome row (return, hold, exit_reason, label, ENTRY belief snapshot)
                         │
                         ├──► calibrator.fit()         # recalibrate conviction (L4)
                         ├──► rationale store           # RAG corpus for grounding (pgvector)
                         ├──► regime_memory update      # which regimes paid off
                         ├──► purification ledger        # impure-revenue accrual on profitable closes (INV-7)
                         └──► experiment ledger         # A/B of policies/prompts
```

**`outcome.label` — defined (fix R, label-undefined):** the calibration target is binary `win = 1 if net_return_pct > cfg.win_threshold_pct else 0`, where `win_threshold_pct` defaults to a small positive band that covers round-trip cost (~0.2%), so a breakeven scrape isn't labeled a win. Stored as an int on the `outcome` row; `NOT NULL` is satisfiable because every close computes it.

**No look-ahead in calibration (fix R, leakage):** the calibrator is fit **only** on `outcome.entry_belief` features — the belief snapshot *at entry*, persisted with the opening fill. The per-scoring `conviction_score` rows (which keep updating during the hold) are telemetry, **not** training inputs — using them would leak mid-trade information correlated with the result. Held-out evaluation is walk-forward by close date.

**Purification (INV-7, fix R-06):** on every profitable close of a name with a non-zero impure-revenue fraction, `core/post_close.record_round_trip`'s purification accrual runs off the `outcome` row (which now carries `realized_pnl_usd` and the symbol's `impure_ratio` — see §8/Appendix C), writing the owed amount to the purification ledger. Compliance accrual is not optional and not dropped.

**Reuse:** `ml/retrainer.py`, `ml/calibration.py`, `core/post_close.py` (drift/thesis/regret/RAG/**purification** fan-out), `core/shadow_runner.py` + prompt-evolution GA → the **experiment framework**.

**New first-class concept — outcome attribution:** every closed trade is scored against the belief that opened it (`outcome.entry_belief`), so we learn *which kinds of evidence and which regimes actually predict wins* — and the calibrator down-weights the rest. This is the compounding edge.

**Invariants:** learning never touches live trading directly — it proposes new model versions promoted via the experiment gate (shadow-must-beat-live, with the **statistical** gate of Part IV, not n=5). INV-5 (everything trains off the durable log); INV-7 (purification accrues on every qualifying close).

**Tests:** an outcome row with a non-null `label` and entry belief is written on every close; calibrator refit uses entry-only features (no mid-trade leakage) and improves walk-forward log-loss; a profitable close on an impure-revenue name accrues a purification entry; a shadow policy that underperforms (per the significance test) is not promoted.

---

## L9 — `api/` + dashboard (understanding-first)

**Purpose:** the operator surface. The headline change: the primary view is **the bot's understanding**, not an order blotter.

- **Belief board:** per-asset card — regime, conviction (with calibration), direction, thesis, key levels, top evidence (the "why"), pending catalysts, and the position held under it. One glance answers "what does the bot think and why."
- **Decision stream:** the causal chain `news → belief.updated → conviction.scored → policy.target_changed → order.filled`, reconstructable from `correlation_id` (INV-5). You can replay any decision.
- **Risk + health:** heat/drawdown, LLM/provider health (the `/api/system/status` classifier-health surface we built generalizes to cognition health), reconcile status.
- **Controls:** kill-switch, per-asset belief pin/override, halt, force-reconcile.

**Reuse:** keep a *thin* slice of `web/` (FastAPI app, the React SPA shell, `/api/system/status`, risk routes, auth). **Cut** everything non-operational (marketplace, KYC, SOC2, advisor, conference, newsletter, billing, robo-advisor) — move to a separate repo per the scope decision.

**Tests:** belief board renders from store; decision stream reconstructs a known correlation chain; controls gate on confirmation headers (kept from today).

---

# Part IV — Migration plan (strangler-fig, paper-safe)

Each phase ships independently, keeps the live bot trading, and is reversible **per its documented downgrade** (not `git` alone — fix R-07). The **Phase 3 shadow A/B is the gate**: we do not flip to conviction-driven execution until the shadow *significantly* beats the churning cycle on real sessions.

| Phase | Goal | Builds | Acceptance criteria | Reversible? |
|---|---|---|---|---|
| **0a. Dependency audit** | Know the seams | Grep the import graph: which `web`/`marketplace`/`community`/`education`/`halal` modules the trading path (`trading`,`crypto`,`core`,`markets`) actually imports; produce the precise **gate-vs-platform split of `halal/`** (keep: `cache`,`zoya`,`sector_limits`,`round_trip_purification` + whatever the trading path imports; move: certification/scholar/jurisdiction *platform* modules ONLY if unused by trading) and the **test partition** map (the 83 mixed-import test files) | Import graph published; every trading-path import classified keep/move; no "move" item is imported by the trading path | Yes (analysis only) |
| **0b. Carve & freeze** | Legible core | Move ONLY the audited-safe SaaS/platform sprawl to `halabot-platform`; keep the operational API slice, the screening gate, and the purification ledger; stand up `halabot/platform/` (bus, event_log, supervisor) | Trading repo **imports clean** (no ModuleNotFoundError) and tests green with sprawl removed; `event_log` table live; bus unit-tested | Downgrade: revert the move commit (no migration yet) |
| **1. Belief store (read-only shadow)** | A world model exists | `belief/` store+schema; populate from the *existing* cycle's computed data | Beliefs persisted & versioned; belief board renders; **no trading change** | Downgrade: `alembic downgrade` belief tables, then revert |
| **2. Event bus + perception** | Always-on understanding | `perception/` adapters wrap existing collectors → `observation.*`; `cognition/` cheap interpreters → `belief.updated` continuously. **Read-only**: Phase-2 components NEVER place orders, so the `engine_owner` distinction is moot until Phase 4 | Beliefs update from live events without the cycle; **LLM-down run still updates beliefs** (INV-1 proven); zero orders from the new path | Reversible (no execution) |
| **3. Conviction + policy (shadow)** | Prove anti-churn | `conviction/` + `policy/` compute targets/deltas, **log-only, no execution**; publish the raw-score + re-size histograms (used to set the entry band + rebalance threshold — B.2/B.5) | Over a window large enough for a **significance test** (effect size + variance on trade-count and realized P&L vs the live cycle; not a fixed n=5), shadow shows significantly lower churn at ≥ live P&L; report published | Reversible (no execution) |
| **4. Flip execution (stocks, flagged)** | Conviction trades live | `execution/` unified; route stocks through policy behind `ENGINE_LIVE=stocks`; old cycle = fallback. **`engine_owner` tag active**: each engine reconciles only its own positions (L6) so the two never fight over shares | Live stocks trades come only from target deltas; reconcile clean & engine-scoped; kill-switch + all gates honored; rollback flag flips back to the old cycle with positions intact | Yes (flag); the flag is the rollback — no migration to undo |
| **5. Unify + learning** | One engine, it learns | Collapse the dual schedulers → one supervisor + venue adapters (`binance` dormant per fork 2); wire `learning/` (outcome→calibration) | Single engine drives stocks; **dormant crypto adapter's data/screener/monitor path stays green in CI** (fork 2 — it's unwired, not unmaintained); calibrator improving on walk-forward; experiment gate operational | Partial — collapsing the crypto stack is structurally hard to undo; gated on the dormant adapter passing its suite first |
| **6. Decommission** | Remove the old | Delete the transactional pipeline + quarantined sprawl from the trading repo | Old `trading/cycle`, `crypto/cycle`, dual schedulers removed; LOC down ~60–70%; all tests green | No (point of no return) |

**Shared-DB / Alembic ownership across the repo split (fix R, repo-split):** there is **one** Postgres and **one** Alembic head, owned by the **trading repo**. `halabot-platform` does not generate migrations against the shared chain. If the platform surface needs mutable tables after the split, it gets its **own** database + its own Alembic history (the SaaS tables are not referenced by the trading engine, so they can live in a separate schema/DB cleanly). This avoids the divergent-heads failure where two repos race the single `alembic_version` row.

**Migration runbook (per phase that ships a migration):** forward = `alembic upgrade head` (the bot's head-check then passes); rollback = `alembic downgrade <prev_rev>` **then** `git checkout <prev_commit>` — in that order, because `init_db` refuses to start when the DB is ahead of the code's bundled head (the reversibility caveat the review caught).

**Reuse map (where today's modules land):**

```
trading/bars, crypto/websocket, mcp/client        → perception/market_data
sentiment/*, trading/{edgar,fed_speak,fred,options_iv}_catalysts → perception/news + perception/macro
crypto/indicators, trading/timeframes             → cognition/indicators, cognition/multiframe
ml/{regime_memory,causal_regime}, RegimeDetector  → cognition/regime
ml/{forecaster,anomaly,drift,calibration}         → cognition/* + conviction/
core/llm/* (ensemble/adversarial/agentic), prompts → cognition/thesis (sparse)
core/llm/rag_db, rationales pgvector              → cognition/rag_grounding + learning/
crypto/risk, trading/portfolio, core/halt         → risk/
trading/executor + crypto/executor, core/fills    → execution/orders
trading/monitor + crypto/monitor                  → execution/position_manager
core/reconcile (incl. this week's fix-drift)      → execution/reconcile
ml/retrainer, core/post_close, shadow_runner, GA  → learning/
web/ (operational slice only)                      → api/
core/{supervisor,observability,tracing,metrics,event_bus} → platform/
db/, RepoBundle, Alembic                          → platform/db (+ new tables §8)
```

---

# Part V — Risks & locked decisions

**Risks & mitigations:**
- *Belief modeling is the hard part.* Mitigation: Phases 1–3 validate beliefs and the anti-churn policy in **shadow** before any live flip; identity-calibration cold-start keeps it honest.
- *Two systems during migration.* Mitigation: strangler-fig with the bus as the seam; the old cycle stays as fallback through Phase 4.
- *Under-trading risk* (conviction never crosses entry). Mitigation: the entry band and rebalance threshold are config; tune from the Phase-3 shadow data, not guesses.
- *Test-suite mass* (139k LOC). Mitigation: tests for cut modules go with them; new layers are small and Protocol-bounded → cheaper to test.

## Decisions (LOCKED — defaults adopted 2026-05-28)

The four forks are locked with the recommended defaults below. They can be revisited, but Phase-0 work proceeds on these.

| # | Fork | **Decision** | Rationale | Revisit if… |
|---|---|---|---|---|
| 1 | Scope of the cut | **Quarantine the SaaS/platform surface to a separate repo** (`halabot-platform`); keep only the *operational* slice in the trading repo (trading API, `/api/system/status`, risk routes, auth, kill-switch). **Move, don't delete** — preserve history. Quarantined: non-operational `web/`, `marketplace/`, `community/`, `education/`, advisor-registration/KYC/SOC2, and the compliance-*platform* parts of `halal/` (keep the screening *gate*). | The investing edge is ~30k of 118k LOC; "focus on better investing decisions" is impossible while ~75% is platform. A separate repo lets that surface live on (or be revived) without weighing down the engine. | A platform feature turns out to be load-bearing for trading (none identified). |
| 2 | Stocks-only vs unified | **Asset-agnostic engine, build & validate stocks-first.** Crypto stays as a **dormant venue adapter** behind the `Venue` port — not a parallel stack. No second scheduler/cycle/executor/monitor is ever rebuilt. | Kills the dual-stack duplication (the biggest source of today's 2× code) while keeping crypto a config-flip away, not a rewrite. Stocks is the only live market today. | Operator re-enables crypto for live trading → exercise the dormant adapter (no architectural change needed). |
| 3 | Bus impl | **In-process async fan-out + Postgres `event_log`** (durable append *before* dispatch), behind the `EventBus` Protocol. | A single-node home bot doesn't warrant Redis/NATS ops burden; durability + replay come from Postgres; the Protocol makes a later swap a non-event. | Engine needs to scale to multiple processes/nodes → swap impl behind the Protocol. |
| 4 | LLM role | **Demote to sparse thesis synthesizer.** Deterministic cognition is primary and continuous; the LLM is invoked only on `material_shift` (thesis/horizon/invalidation) and on high-impact news scoring. **Never in the action path.** | This is the philosophical core of the shift: ~10× cost reduction, and understanding/risk survive an LLM outage (INV-1, proven painful twice this week). | A specific decision is shown to need LLM judgement the deterministic layer can't supply → add a narrow, guarded LLM interpreter (still off the hot path). |

**Consequences now baked into the plan:**
- Phase 0 = create `halabot-platform` repo, move the quarantined surface there, leave the operational API slice; stand up `halabot/platform/` (bus + `event_log`).
- The engine targets **one** code path with `venues/{alpaca,binance}`; `binance` ships dormant.
- All "per the scope/fork decision" references elsewhere in this doc resolve to the above.

**Recommended first concrete step:** Phase 0 + Phase 1 — carve the sprawl into `halabot-platform`, stand up `platform/` (bus + `event_log`) and `belief/` (store + schema), and render a read-only belief board. That delivers the world-model spine with zero trading-behavior risk, and makes everything after it incremental.

---

# Part VI — Implementation appendices (concrete detail)

Part III gives each layer's shape; this part fills the gaps an implementer hits on day one: full payload schemas, the actual decision algorithms, DDL, the compliance contract, the degradation matrix, the concurrency/bootstrap model, the file layout, and the unified exit rules.

## Appendix A — Event payload schemas

All payloads are `TypedDict`s in `schemas/`, validated on `publish` (fail-closed: a malformed payload is logged with type and dropped, never dispatched — INV-4). `ts` and envelope fields live on `Event` (§5); payloads carry only the body.

```python
# schemas/observations.py
class PriceObservation(TypedDict):
    asset: str; price: float; bid: float | None; ask: float | None

class BarObservation(TypedDict):
    asset: str; tf: str            # "1Min" | "1Hour" | "1Day"
    o: float; h: float; l: float; c: float; v: float; bar_ts: str

class NewsObservation(TypedDict):
    asset: str; headline: str; summary: str; url: str
    published_at: str; source: str
    lexicon_polarity: float | None  # cheap pre-LLM score; None if lexicon abstained

class MacroObservation(TypedDict):
    kind: str                       # "CPI" | "FOMC" | "NFP" | "GDP" | "earnings"
    asset: str | None               # None = market-wide
    scheduled_for: str; expected_impact: float  # 0..1
    actual: float | None; consensus: float | None

class SentimentObservation(TypedDict):
    asset: str; mention_velocity: float; novelty: float; net_polarity: float; window_min: int

class OnchainObservation(TypedDict):
    asset: str; signal: str         # "whale_inflow" | "whale_outflow" | "basis"
    magnitude: float; detail: str

# schemas/belief.py
class BeliefUpdatedPayload(TypedDict):
    version: int; regime: str; regime_confidence: float
    direction: str; conviction: float; horizon: str
    top_evidence: list[dict]        # [{source, direction, weight, detail}] (truncated)
    invalidation: float | None; thesis_refreshed: bool

class BeliefInvalidatedPayload(TypedDict):
    version: int; reason: str       # "price_break" | "conviction_decay" | "catalyst_flip"
    invalidation_level: float | None; last_price: float

# schemas/decision.py
class ConvictionScoredPayload(TypedDict):
    raw: float; calibrated: float; belief_version: int; features: dict

class TargetChangedPayload(TypedDict):
    target_weight: float; current_weight: float; belief_version: int; reason: str

class TradeProposedPayload(TypedDict):
    side: str; notional_usd: float; target_weight: float
    current_weight: float; belief_version: int; reason: str

# schemas/execution.py
class OrderFilledPayload(TypedDict):
    side: str; quantity: float; filled_price: float; filled_quantity: float
    order_id: str; trade_id: int; belief_version: int
    engine_owner: str               # "legacy" | "belief" — reconcile scoping during migration (fix R-02)
    slippage_pct: float | None

class PositionReconciledPayload(TypedDict):
    db_net: float; broker_qty: float; engine_owner: str
    action: str                     # "none" | "adjustment" | "import" | "skip_other_engine"
    adjustment_qty: float

# schemas/risk.py
class RiskStatePayload(TypedDict):
    portfolio_heat_pct: float; drawdown_pct: float
    realized_loss_today_pct: float; gross_exposure: float   # fix R-10 / R-03
    halted: bool; reason: str | None
```

## Appendix B — Core algorithms (concrete)

### B.1 Evidence decay + merge (`belief/evidence.py`)

```python
def decay(items: list[EvidenceItem], now: datetime, halflife_min: float) -> list[EvidenceItem]:
    out = []
    for it in items:
        # TRADING-TIME age, not wall-clock: minutes the market was OPEN between it.ts and now,
        # so an overnight/weekend gap doesn't annihilate evidence and force a Monday-open mass
        # exit (fix R-09). For 24/7 venues (crypto) trading_minutes == wall-clock minutes.
        age = trading_minutes_between(it.ts, now) if cfg.evidence_decay_trading_time \
              else (now - it.ts).total_seconds() / 60.0
        factor = 0.5 ** (age / halflife_min)          # exponential half-life
        if it.weight * factor < EPS_PRUNE:             # drop fully-decayed evidence
            continue
        out.append(replace(it, weight=it.weight * factor))
    return out

def merge(existing: list[EvidenceItem], fresh: list[EvidenceItem], cap_per_source: int = 3) -> list[EvidenceItem]:
    # Idempotency (INV-5): drop any fresh item whose event_id we already hold, so an
    # at-least-once redelivery / bootstrap-replay overlap can't double-count evidence (fix R, idempotency).
    seen = {it.event_id for it in existing if it.event_id is not None}
    fresh = [it for it in fresh if it.event_id is None or it.event_id not in seen]
    by_source: dict[str, list[EvidenceItem]] = defaultdict(list)
    for it in existing + fresh:                        # fresh appended last (newer)
        by_source[it.source].append(it)
    out = []
    for src, items in by_source.items():
        items.sort(key=lambda x: x.ts, reverse=True)   # newest first
        out.extend(items[:cap_per_source])             # bound memory per source
    return out
```

> **Prose/semantics note (fix R, merge):** `cap_per_source` keeps up to N *distinct* recent items per source (after event_id dedup), and all retained items contribute to `weighted_sum` — decay is what fades old ones, not replacement. The earlier "newest replaces stale" phrasing was misleading; the model is decay-and-cap, not single-replace.

### B.2 Raw conviction (`conviction/raw.py`) — deterministic, LLM-free (INV-1)

```python
def conviction_raw(evidence: list[EvidenceItem], regime_conf: float,
                   drift_flag: bool, anomaly_flag: bool) -> float:
    if not evidence:
        return 0.0
    w = sum(e.weight for e in evidence) or 1.0
    signed = sum(e.direction * e.weight for e in evidence) / w   # ∈ [-1, +1]
    if signed <= 0:                                              # long-only: no bullish net → no conviction
        return 0.0
    agreement = fraction_same_sign(evidence)                    # 0..1, dispersion penalty
    raw = signed * (0.5 + 0.5 * agreement) * regime_conf        # regime + agreement scaling
    if drift_flag:   raw *= 0.7                                  # concept drift → widen uncertainty
    if anomaly_flag: raw *= 0.6                                  # anomalous tape → trust less
    return clamp(raw, 0.0, 1.0)

def weighted_sum(evidence: list[EvidenceItem]) -> float:        # the SAME normalized `signed` used above
    if not evidence: return 0.0
    w = sum(e.weight for e in evidence) or 1.0
    return sum(e.direction * e.weight for e in evidence) / w
```

`direction` (in `apply_evidence`) and `conviction_raw` both derive `signed` from `weighted_sum` — one formula, so direction and conviction can never disagree on the same evidence (fix R, consistency). `calibrated = calibrator.calibrate(asset, raw, features)` (L4); the belief stores **both** `conviction` (calibrated) and `conviction_raw` (= this `raw`, for `material_shift`'s raw-vs-raw band compare).

**Cold-start note (fix R, calibration ceiling):** because `raw = signed·(0.5+0.5·agreement)·regime_conf` is a product of ≤1 factors, raw rarely reaches a high band under identity calibration (the cold-start fallback). So `conviction_entry_band` is **tuned from the Phase-3 shadow distribution of raw scores**, not set to an aspirational 0.60 — otherwise the bot under-trades cold, never generating the outcomes the calibrator needs (chicken-and-egg). The shadow phase reports the raw-score histogram precisely to set this.

### B.3 `material_shift` (`belief/updater.py`) — governs LLM spend

```python
def material_shift(prev: BeliefState, new_raw: float, new_regime: Regime, now: datetime,
                   has_open_position: bool) -> bool:
    # `prev` is the PREVIOUS PERSISTED belief (snapshotted before mutation in apply_evidence) — NOT
    # the in-progress one, so these comparisons are real deltas, not a value against itself (fix R-11).
    if prev.regime != new_regime:                                   return True   # regime flip
    # Compare RAW-to-RAW (prev.conviction_raw, not prev.conviction which is calibrated) so the
    # band-edge test is on one scale (fix R-11).
    if band_index(prev.conviction_raw) != band_index(new_raw):      return True   # crossed a conviction band edge
    if any(c.is_imminent(now) and c.expected_impact >= cfg.catalyst_impact_threshold
           for c in prev.catalysts_pending):                        return True   # high-impact catalyst landing
    if has_open_position and prev.last_thesis_refresh is not None \
            and (now - prev.last_thesis_refresh) > MAX_THESIS_AGE:   return True   # stale narrative on a live position
    if has_open_position and prev.last_thesis_refresh is None:       return True   # never written a thesis for a held position
    return False                                                    # else: cheap update only, no LLM
```

`band_index` buckets the **raw** score into e.g. `[0, .3, .55, .75, 1.0]` so small wiggles don't trigger LLM calls. (The `target_weight` bands operate on the *calibrated* conviction and are a separate, intentionally-different grid; `material_shift`'s job is throttling LLM spend on raw-signal moves, not sizing.)

### B.4 Level engine (`belief/levels.py`)

```python
def update_levels(asset: str, bars: list[Bar], atr: float, prev: Levels) -> Levels:
    highs, lows = swing_points(bars, lookback=cfg.swing_lookback)
    support     = nearest_below(last_price(bars), lows)
    resistance  = nearest_above(last_price(bars), highs)
    # invalidation = the structural level that, if lost, kills the long thesis:
    #   max(most-recent swing low, entry - k*ATR). Ratchets UP only (slow-out).
    structural  = lows[-1] if lows else None
    px = last_price(bars)
    atr_floor   = (px - cfg.atr_stop_mult * atr) if (px is not None and atr is not None and atr > 0) else None
    candidates  = [x for x in (structural, atr_floor, prev.invalidation) if x is not None]
    invalidation = max(candidates) if candidates else None   # cold-start (no swings, no ATR yet) → None, not a crash (fix R, all-None max)
    stop = invalidation                                  # monitor uses this as the hard stop (None ⇒ no hard stop yet; entry deferred until set)
    return Levels(support, resistance, stop, invalidation)
```

The ratchet-up-only invalidation is the structural "slow out": it tightens as price rises, never loosens. A `None` invalidation (cold-start asset with no swing structure and no ATR yet) is valid — the policy simply won't open until the level engine has enough bars to set one, so no position is ever held without a computable stop.

### B.5 Target weight + hysteresis (`policy/sizing.py`)

```python
# THE single canonical target_weight (fix R-13: the earlier L5 variant is removed; this is it).
def target_weight(b: BeliefState, risk: RiskState, held: bool) -> float:
    if b.direction != Direction.LONG_BIAS:                       return 0.0
    enter_band, exit_band = cfg.conviction_entry_band, cfg.conviction_exit_band  # e.g. 0.60 / 0.45
    assert 0.0 <= exit_band < enter_band < 1.0                   # config invariant: guards the denominator below
    threshold = exit_band if held else enter_band                # HYSTERESIS: easier to hold than to enter
    if b.conviction < threshold:                                 return 0.0
    scale = clamp((b.conviction - exit_band) / (1.0 - exit_band), 0.0, 1.0)  # 1-exit_band > 0 by the assert
    raw = scale * cfg.max_weight_per_asset
    raw *= risk.correlation_multiplier(b.asset)
    raw *= risk.volatility_multiplier(b.asset)
    return min(raw, cfg.max_weight_per_asset)
```

`held` is `currently_held` — `portfolio.holds(asset)` at the call site. The config invariant `0 ≤ exit_band < enter_band < 1` is validated at settings load, so `1.0 - exit_band` is never zero (fix R-13 denominator).

**Two anti-churn mechanisms, both required:**
1. **Hysteresis** (entry band > exit band): a position clears a *higher* bar to open than to stay open, so conviction noise around the threshold can't flip-flop you in/out (the 0-vs-nonzero churn).
2. **Rebalance threshold** (`deltas`, L5): a *held* position whose calibrated conviction wanders **between** the bands still re-sizes continuously unless the resulting weight change exceeds `target_rebalance_threshold` — which suppresses the in-band re-sizing churn that hysteresis alone does not (the review's "churn relocated into rebalancing" point). Set `target_rebalance_threshold` from the Phase-3 shadow re-size histogram.

### B.6 Gate ordering (`policy/gates.py`) — cheap, short-circuiting, ordered

```python
GATES = [                       # evaluated in order; first rejection wins; cheapest/most-decisive first
    halt_gate,                  # kill-switch (DB read)            — INV first-check
    risk_halt_gate,             # heat / drawdown / DAILY-LOSS-LIMIT halt (all three, fix R-10)
    direction_gate,             # belief not long → no buy
    rebalance_threshold_gate,   # |target-effective| < threshold → no trade  (the churn killer)
    open_order_gate,            # an order already working for this asset → wait (fix R-14, no duplicate)
    market_close_lockout_gate,  # no new BUYs in last N min pre-close
    max_positions_gate,
    sector_cap_gate,
    min_hold_gate,              # SELL side: belt-and-suspenders
    recent_close_cooldown_gate, # belt-and-suspenders
    stop_loss_reentry_gate,     # belt-and-suspenders (belief usually already blocks this)
    halal_gate,                 # INV-7: fresh positive verdict + FK; LAST so it's only paid on otherwise-valid trades
    venue_feasibility_gate,     # min-notional AND lot/step/min-qty/max-qty alignment (fix R, lot-size)
    buying_power_gate,
]
# Enforced OUTSIDE this per-trade chain: LLM_daily_usd_cap (cumulative-spend halt of LLM calls, fix R-15)
# and the per-asset circuit breaker (L6) — neither is a per-proposal check.
```

Note: most band-aid gates (min-hold, cooldown, reentry) now rarely bind because the belief/policy layer already produces the right behavior — they remain as defense in depth, not primary control.

## Appendix C — DDL (new tables; additive)

```sql
-- the spine (partition by month on ts)
CREATE TABLE event_log (
  id uuid PRIMARY KEY, type text NOT NULL, asset text, ts timestamptz NOT NULL,
  ingested_at timestamptz NOT NULL DEFAULT now(), source text NOT NULL,
  payload jsonb NOT NULL, causation_id uuid, correlation_id uuid NOT NULL, schema_version int NOT NULL
);
CREATE INDEX ix_event_type_ts ON event_log (type, ts);
CREATE INDEX ix_event_asset_ts ON event_log (asset, ts);
CREATE INDEX ix_event_corr ON event_log (correlation_id);

CREATE TABLE belief_state (
  id bigserial PRIMARY KEY, asset text NOT NULL, version int NOT NULL,
  regime text NOT NULL, regime_confidence double precision NOT NULL,
  direction text NOT NULL, conviction double precision NOT NULL,
  conviction_raw double precision NOT NULL,            -- pre-calibration, for material_shift (fix R-11)
  horizon text NOT NULL,
  thesis text, levels jsonb NOT NULL, catalysts jsonb NOT NULL DEFAULT '[]',
  halal_verdict jsonb NOT NULL, opened_trade_ids jsonb NOT NULL DEFAULT '[]',
  last_thesis_refresh timestamptz, updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (asset, version)
);
CREATE INDEX ix_belief_asset_ver ON belief_state (asset, version DESC);

CREATE TABLE belief_evidence (
  id bigserial PRIMARY KEY, belief_id bigint NOT NULL REFERENCES belief_state(id),
  source text NOT NULL, direction double precision NOT NULL, weight double precision NOT NULL,
  detail text, event_id uuid, ts timestamptz NOT NULL
);
CREATE INDEX ix_evidence_belief ON belief_evidence (belief_id);

CREATE TABLE conviction_score (
  id bigserial PRIMARY KEY, asset text NOT NULL, ts timestamptz NOT NULL DEFAULT now(),
  raw_score double precision NOT NULL, calibrated double precision NOT NULL,
  features jsonb NOT NULL, belief_version int NOT NULL
);
CREATE INDEX ix_conv_asset_ts ON conviction_score (asset, ts);

CREATE TABLE target_weight (
  id bigserial PRIMARY KEY, asset text NOT NULL, ts timestamptz NOT NULL DEFAULT now(),
  target_weight double precision NOT NULL, current_weight double precision NOT NULL,
  reason text NOT NULL, belief_version int NOT NULL
);

CREATE TABLE outcome (
  id bigserial PRIMARY KEY, trade_id bigint NOT NULL, asset text NOT NULL,
  belief_version int NOT NULL, entry_belief jsonb NOT NULL,    -- ENTRY snapshot only (no leakage, fix R)
  return_pct double precision NOT NULL,
  realized_pnl_usd double precision NOT NULL,                  -- purification input (fix R-06)
  impure_ratio double precision NOT NULL DEFAULT 0,            -- symbol's impure-revenue fraction (fix R-06)
  hold_seconds int NOT NULL, exit_reason text NOT NULL,
  label int NOT NULL,                                          -- win=1 if return_pct>win_threshold else 0 (fix R, label)
  closed_at timestamptz NOT NULL
);
CREATE INDEX ix_outcome_asset ON outcome (asset);

-- link existing positions to the belief that opened them (INV-8) + reconcile scoping (fix R-02)
ALTER TABLE trades        ADD COLUMN belief_version int, ADD COLUMN engine_owner text NOT NULL DEFAULT 'legacy';
ALTER TABLE crypto_trades ADD COLUMN belief_version int, ADD COLUMN engine_owner text NOT NULL DEFAULT 'legacy';
```

Each ships as an Alembic revision; `init_db` head-check unchanged. `payload`/`features`/`levels` are JSONB for schema flexibility on the fast-moving fields, with typed accessors in code.

**Reversibility (fix R-07):** every revision ships a tested `downgrade()`. Because `init_db` refuses to start when the DB revision ≠ the code's bundled Alembic head, **reverting code alone is not enough** — a rollback is `alembic downgrade <prev>` *then* `git checkout`. The migration runbook (Part IV) pairs each phase's forward revision with its downgrade command; "reversible" means *reversible via the documented downgrade*, not via `git` alone.

## Appendix D — Compliance verdict + transient-safe cache (INV-2, INV-7)

```python
@dataclass(frozen=True)
class ComplianceVerdict:
    asset: str
    status: Literal["halal", "not_halal", "doubtful"]
    detail: str
    screened_at: datetime
    screening_id: int                 # FK target for the trade row
    transient_error: bool = False     # True ⇒ this is a NO-VERDICT, never persisted

class ComplianceGate:
    async def verdict(self, asset: str) -> ComplianceVerdict: ...   # cache → screener
    def is_tradeable(self, v: ComplianceVerdict, now: datetime) -> bool:
        return (v.status == "halal"
                and not v.transient_error
                and (now - v.screened_at) <= cfg.cache_ttl)
```

**Cache write rule (the poisoning fix, generalized):** the screener returns `transient_error=True` on any API/transport failure; the cache layer **skips writing** transient errors, preserving the prior good verdict. A symbol is only ever cached as `not_halal`/`doubtful` from a *real* verdict, never from an outage. The gate treats `transient_error` and stale verdicts as "not tradeable" (fail-closed for trading) while leaving the belief intact (a screening outage must not flip a belief).

**Lapsed compliance on HELD positions (fix R-05):** the belief worker re-screens each held asset on the `cache_ttl` cadence. When a re-screen returns a **real** (non-transient) `not_halal`/`doubtful` verdict on a position currently held, the worker emits `belief.invalidated` with reason `"compliance_lapsed"`, which the policy turns into target → 0 and the monitor force-exits — **regardless of conviction or P&L**. A *transient* screening error never triggers this (INV-2). This closes the hole where a name that turned non-compliant mid-hold (e.g. new financials cross the debt screen) would otherwise be held indefinitely because its price-conviction stayed high. The forced exit is recorded with `exit_reason="compliance_lapsed"` and still accrues any owed purification on the realized P&L (INV-7).

## Appendix E — Degradation matrix (what each layer does when a dependency is down)

| Dependency down | Perception | Cognition | Belief | Conviction | Policy | Execution |
|---|---|---|---|---|---|---|
| **LLM** | unaffected | cheap interpreters run; thesis/news-LLM skipped | beliefs update; `thesis` stales | identity-or-prior calibrator | unaffected (deterministic) | unaffected |
| **Market data feed** | source restarts; emits nothing | no new bar evidence | **the heartbeat tick drives decay** (see below) so conviction decays even with no new data | uses last features | targets drift toward 0 → graceful de-risk | reconcile continues |
| **News API** | source restarts | no news evidence | unaffected | unaffected | unaffected | unaffected |
| **Compliance API** | n/a | n/a | belief unchanged (INV-2); held-position re-screen deferred, not failed | unaffected | halal gate fails-closed → no new buys; exits allowed | exits allowed |
| **Broker/venue** | n/a | n/a | n/a | n/a | proposals queue | retries; per-asset breaker; reconcile flags; no $0 invents (INV-2) |
| **DB** | buffer in-memory, shed price ticks first | — | **read-only from in-memory cache; cannot persist new beliefs** | — | **no NEW entries** | **monitor uses last-known beliefs/levels held in memory to keep enforcing exits** |

**The two corrected rows:**
- **Market-data down (fix R-08):** decay must NOT depend on new evidence arriving (it previously only ran inside `apply_evidence`). The `system.heartbeat` tick (every `ENGINE_heartbeat_interval_s`) calls `apply_evidence(asset, items=[], now)` for every active belief — i.e. a decay-only update with no new items — so conviction *does* fade and targets de-risk during a data blackout. The heartbeat is the wired mechanism behind the "decays over time" claim.
- **DB down (fix R, deadlock):** the first-draft "durable-append-before-dispatch" would deadlock here (can't append → nothing dispatches → can't even monitor). Resolved by **two bus tiers**: *control/exit* events (`order.*`, `belief.invalidated`, `risk.halt`, `system.heartbeat`) dispatch **best-effort in-memory even if the durable append fails** (logged, reconciled later), so the monitor keeps closing risk; *durable* events (observations, belief versions, outcomes) require the append and are the ones blocked when the DB is down. The monitor reads the last-known beliefs/levels from the in-memory belief cache, so SL/TP/invalidation enforcement survives a DB outage. New entries are refused (can't persist provenance), but **risk-reducing exits never block on the DB.**

The throughline (INV-1): **understanding and risk-management never depend on the LLM or any single external service.** The worst case for any single outage is "stop opening new risk; keep managing what's open" — and that holds for the DB outage too, by the two-tier bus.

## Appendix F — Concurrency, partitioning, bootstrap

- **Per-asset single-writer.** Belief mutations for an asset are serialized through a per-asset async worker (an actor keyed by `asset`), so the updater never races itself. Cross-asset workers run concurrently under the supervisor. This is why per-asset event ordering is guaranteed (§5).
- **Supervisor topology** (`platform/supervisor.py`, anyio task group):
  ```
  root
  ├── bus dispatcher
  ├── heartbeat               (emits system.heartbeat every ENGINE_heartbeat_interval_s → drives decay even with no new data, fix R-08)
  ├── perception/*            (one task per source)
  ├── belief workers          (one per active asset; spawned lazily; subscribe to observation.* AND system.heartbeat)
  ├── execution.monitor_loop  (SL/TP/trailing/trend-break)
  ├── execution.reconcile_loop
  ├── risk.evaluate_loop
  ├── learning.train_loop     (off the hot path)
  └── api server
  ```
- **Heartbeat-driven decay (fix R-08):** each belief worker handles `system.heartbeat` by calling `apply_evidence(asset, items=[], now)` — a decay-only pass. So conviction fades on the passage of (trading) time alone, with zero new observations. This is what makes the degradation matrix's "market-data down → decays → de-risk" actually true.
- **Idempotency:** handlers key off `Event.id`; replays and at-least-once delivery are safe because `merge` drops any evidence item whose `event_id` is already held (Appendix B.1 — the dedup is by `event_id`, fixing the earlier incorrect "merge dedups by source" claim).
- **Per-asset ordering vs application order (fix R, ordering):** the per-asset worker serializes *writes*, but multiple interpreters can deliver evidence for one asset out of `ts` order. The worker therefore **orders by `event.ts` within a small coalescing window** (drains the asset's queue, sorts by `ts`, applies as one batch) before each `apply_evidence`, so decay/merge see monotonic time and a strict-`ts` replay reproduces the same belief version (preserving INV-5). Cross-asset ordering remains best-effort (independent workers); only per-asset determinism is guaranteed, which is all the belief model needs.
- **Cold-start / bootstrap (fix R, time-base):** on launch the belief store seeds each universe asset to `BeliefState.neutral(asset)`, then **replays** the last `BELIEF_bootstrap_window` of `observation.*` from `event_log` through cognition **with the injected `Clock` set to each event's `ts` (event-time, not wall-time)** so decay ages each observation relative to *then*, and a final decay-to-`now` pass brings the warmed belief to the present. During replay `is_replay(now)` is true, which **suppresses `belief.invalidated` emission and all order/exit side-effects** (replay warms beliefs; it must never fire trades against historical prices). To avoid double-counting the overlap between the replay window and live events already buffered, replay runs to completion before the worker subscribes to the live stream, and `merge`'s `event_id` dedup absorbs any residual overlap. If no history, beliefs start neutral and warm up live. Open positions are re-linked to their `belief_version` (and `engine_owner`) from the trade rows.
- **Shutdown:** drain the bus, flush belief workers, persist final versions; the monitor's in-memory trailing high-water marks are persisted per-position so a restart doesn't reset slow-out stops (fixes the "restart resets trailing" caveat from operations).

## Appendix G — Package & file layout (what to create)

```
halabot/
├── platform/
│   ├── bus.py              # EventBus protocol + InProcessPgBus impl
│   ├── events.py           # Event, EventType
│   ├── event_log.py        # durable append + replay
│   ├── supervisor.py       # anyio task group wrapper  (← core/supervisor.py)
│   ├── clock.py            # Clock protocol + system/fake
│   ├── config.py           # narrowed Settings tree   (← config.py)
│   ├── observability.py    # ctx vars, log filter (typed errors)  (← core/observability.py)
│   └── db/                 # engine, RepoBundle, Alembic           (← db/)
├── schemas/                # TypedDict payloads (Appendix A)
├── perception/
│   ├── base.py             # Source protocol
│   ├── market_data/{bars,ticks}.py
│   ├── news/{finnhub,cryptopanic,edgar,fed,reddit}.py
│   ├── macro/{fred,options_iv}.py
│   └── onchain/{whale,basis}.py
├── cognition/
│   ├── base.py             # Interpreter protocol
│   ├── indicators.py multiframe.py regime.py forecaster.py anomaly.py drift.py
│   ├── news_lexicon.py
│   └── llm/{thesis_writer,news_analyst,rag_grounding}.py   (← core/llm/*)
├── belief/
│   ├── schema.py           # BeliefState, Levels, EvidenceItem, enums
│   ├── store.py            # BeliefStore protocol + Pg impl (versioned)
│   ├── updater.py          # apply_evidence, material_shift
│   ├── evidence.py         # decay, merge
│   └── levels.py           # level engine
├── conviction/
│   ├── raw.py              # conviction_raw
│   └── calibrator.py       # Platt/isotonic + identity fallback  (← ml/calibration.py)
├── policy/
│   ├── policy.py           # targets() + deltas()
│   ├── sizing.py           # target_weight + hysteresis
│   └── gates.py            # ordered gate chain
├── execution/
│   ├── venues/{alpaca,binance}.py   (← mcp/client.py, crypto/exchange.py)
│   ├── orders.py           # place/confirm/record  (← trading+crypto executors, core/fills.py)
│   ├── position_manager.py # monitor loop          (← trading+crypto monitors)
│   └── reconcile.py        # continuous broker-truth reconcile  (← core/reconcile.py)
├── risk/
│   └── engine.py           # RiskEngine             (← crypto/risk.py, trading/portfolio.py, core/halt.py)
├── learning/
│   ├── outcomes.py         # close → outcome row + label
│   ├── retrain.py          # calibrator/model refit (← ml/retrainer.py)
│   └── experiments.py      # shadow A/B + promotion (← core/shadow_runner.py, prompt GA)
└── api/
    ├── app.py              # FastAPI            (← thin slice of web/app.py)
    └── routes/{beliefs,decisions,risk,system,controls}.py
```

## Appendix H — Unified exit authority (`execution/position_manager.py`)

A single monitor owns all exits. Precedence (first match wins), evaluated each monitor tick + on relevant events:

```
1. risk halt / kill-switch        → flatten (risk-reducing always allowed, even halted)
2. compliance lapsed (real not_halal/doubtful on a held name) → exit, ANY P&L  (compliance_lapsed, fix R-05)
3. belief.invalidated             → exit (thesis dead: price broke invalidation)
4. hard stop (price <= levels.stop)→ exit  (stop_loss)
5. trend-break (close < SMA, winner)→ exit (trend_break)  [reuse this week's logic]
6. trailing-stop ratchet           → tighten stop (no exit; slow-out)
7. policy target == 0 (conviction decayed) → exit (target_zero)
   else: hold
```

Compliance sits at rung 2 — above every P&L/structural consideration — because halal compliance is non-negotiable (INV-7): a held name that turns non-compliant is exited regardless of how well the trade is going. The exit accrues any owed purification on the realized P&L (Appendix D, L8).

There is **no LLM-initiated exit** and **no fixed take-profit** — exits are compliance-, belief-, or rule-driven. This removes the LLM-vs-monitor conflict (and the `reactor_momentum` permanent-lockout hack it forced). "Let winners run" is the absence of a TP rung: a winner is only cut on compliance (2), a *structural* break (3, 4, 5), or genuine conviction decay (7), never on hitting an arbitrary profit target.

---

# Part VII — Revision log (from the max-effort code review)

This revision (2026-05-28, Opus 4.8) folds in a 5-angle max-effort review of the v1 spec. Each finding (`R-NN` = the review's rank) is closed in-place above; this table is the index.

| ID | Finding | Fix (where) |
|---|---|---|
| R-01 | Phase 0 carve breaks the live bot via shared `halal/` imports | Part IV Phase 0 now does a dependency audit + precise gate-vs-platform split; purification ledger stays (Part IV, §8) |
| R-02 | Old + new engines fight over one broker account during migration | `engine_owner` tag on trades; reconcile scoped per engine (L6, §8, Appendix C, schemas) |
| R-03 | No gross-exposure cap → implicit leverage when many assets convict | `max_gross_exposure` + portfolio normalization in `targets()`; INV-10; `RiskState.gross_exposure` (L5, L7, §7) |
| R-04 | Live-mode confirmation-token safeguard dropped | INV-9 + `SAFEGUARD_*` config; un-loosenable live floors (§4, §7) |
| R-05 | Lapsed compliance on HELD positions not handled | belief re-screen → `belief.invalidated(compliance_lapsed)` → exit rung 2 (INV-7, Appendix D, H) |
| R-06 | Purification ledger trigger dropped | `outcome.realized_pnl_usd`/`impure_ratio`; purification fan-out kept; tables stay in repo (L8, §8, Appendix C) |
| R-07 | "Reversible via git" false once a migration lands | every revision ships tested `downgrade()`; rollback runbook = downgrade then checkout (Appendix C, Part IV) |
| R-08 | Decay never runs on a data outage (only on new evidence) | `system.heartbeat` drives a decay-only `apply_evidence`; degradation matrix corrected (Appendix E, F) |
| R-09 | Wall-clock decay → Monday-open mass exit | trading-time decay (`evidence_decay_trading_time`); `trading_minutes_between` (Appendix B.1, §7) |
| R-10 | Daily realized-loss limit has no home | first-class halt in `RiskState` + gate chain + `RISK_daily_loss_limit` (L7, L5, B.6, §7) |
| R-11 | `material_shift` call-site signature wrong + calibrated-vs-raw compare + mutated `prev` | snapshot `prev` before mutation; store `conviction_raw`; raw-vs-raw band test; full signature (L3, B.3, §6, Appendix C) |
| R-12 | `conviction_raw` call-site wrong; drift/anomaly flags never wired | call site passes the evidence list + drift/anomaly flags; one `weighted_sum` for direction & conviction (L3, B.2) |
| R-13 | Two contradictory `target_weight` definitions; unguarded denominator | single canonical B.5 with config-invariant assert; L5 references it (L5, B.5) |
| R-14 | Partial-fill / open-order → duplicate stacked orders | `effective_weight` (filled+pending) + `open_order_gate` in deltas/gates (L5, B.6) |
| R-15 | No LLM cost cap / quota / circuit breaker in cognition path | `breaker_open()` guard on thesis call; `LLM_daily_usd_cap` + `news_daily_classify_cap` + `circuit_breaker_threshold` (L3, §7, B.6) |
| — | `update_levels` `max()` ValueError on all-None | guard empty candidate list → `None` invalidation, defer entry (B.4) |
| — | `outcome.label` had no producer | defined `win = return_pct > win_threshold` (L8, Appendix C) |
| — | merge "dedups by source" claim false | dedup by `event_id` in `merge` (B.1, Appendix F) |
| — | DB-down deadlock (durable-append-before-dispatch) | two-tier bus: control/exit events best-effort in-memory (Appendix E) |
| — | Bootstrap replay time-base / look-ahead | event-time Clock during replay; `is_replay` suppresses invalidation+orders (L3, Appendix F) |
| — | Cold-start identity-calibration can't reach a high entry band | entry band tuned from the Phase-3 raw-score histogram, not aspirationally (B.2) |
| — | Repo split shares one Postgres + one Alembic chain | addressed in Part IV: one DB, one Alembic head owned by the trading repo; platform repo gets its own schema/DB if it needs mutable tables |
| — | "tests go with cut modules" assumes clean partition (83 mixed files) | Part IV Phase 0: test split is a dependency-audited step, not assumed 1:1 |
| — | Dormant crypto adapter not "free" (needs ws/screener/ws-monitor) | Part IV / fork 2: dormant means *unwired but maintained*; its data/screener/monitor path is explicitly in scope to keep green |
| — | Phase-3 "≥5 sessions" statistically meaningless | replaced with a significance gate (Part IV) — effect size + variance, not a fixed tiny n |

*End of spec. Layers L0–L9 (Part III) + Appendices A–H (Part VI) + the review fixes (Part VII) constitute the implementable surface. The four forks in Part V are locked; Phase 0 + Phase 1 (platform bus/log + belief store/board) is the recommended first build.*
