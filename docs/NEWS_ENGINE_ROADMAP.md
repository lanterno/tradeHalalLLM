# News decision engine: roadmap

*Started 2026-10-10. Owner: the operator. Status: proposal, nothing built yet.*

The news reactor today does one thing: a positively scored headline plus a
rising price buys half a position. The goal is an engine that reasons about a
news event over time, the way a discretionary trader would:

> Bad news hits. Wait for the dip to finish. When the selling is exhausted
> and a rebound is likely, buy. As it recovers, sell before the downtrend
> resumes.

That is a **multi-step plan per event** (a playbook), not a single score. This
roadmap says how to build one that can be trusted with money, in the order
that keeps us from building the expensive parts before the cheap evidence
says they can work.

## 1. What we already know (the evidence this starts from)

Six pre-registered tests, all on this repository's research harness (net of
costs, against SPY, train/holdout splits), 2026-10-09/10:

| Test | Result |
|---|---|
| Reactor as built: buy 60 s after a positively scored headline | −0.30% same day (t −12) |
| Earnings beats / beat-and-raise at the first tradable price (2016-21) | top decile 5d −0.39% / −0.22% |
| Industry peers after a leader's surprise (2016-21) | tech −0.48% at 5d; all industries IC −0.03 |
| Selling 60 s after strongly negative news | −0.10% / −0.11% gross, next-day recovery |
| A tech-industry expert reading with point-in-time context (Jul-Oct 2026) | 5d IC −0.113, no better than a generic reading (−0.106) |
| Buying after the worst news days, lexicon (2016-21) | worst decile 5d −0.29% (t −6.0) |

What this means for the design:

- **A headline's direction is priced within a minute.** Neither speed nor a
  better reading of *which way* the news points is an edge for this bot.
- **What is left is the path.** How far a stock falls, when the selling
  exhausts, whether a rebound holds or fades: these unfold over minutes to
  days, depend on *what kind* of news it was, and are where a patient,
  well-informed participant can still act. That is exactly the playbook in
  the example, and it is untested so far.
- **The LLM's job changes.** Not "positive or negative?" (no edge), but
  "what kind of event is this, is the damage permanent, and what path do
  events like it usually take?" (classification and routing), plus checks
  at each decision point. Whether that adds value is a hypothesis to measure,
  not an assumption.
- **Halal, long-only.** The engine can buy a rebound and exit before a fade;
  it can never short the fade, use options or leverage. Every order passes
  the halal gate at the order boundary.

## 2. Principles (best practice in event-driven and LLM trading systems)

1. **The LLM interprets; deterministic code decides and acts.** The LLM turns
   text into a structured event card and answers narrow questions at
   transitions. Entry and exit rules, sizing, risk and every gate are code,
   testable and replayable. An LLM outage means "no new playbooks", never
   "a position without a plan" (halabot's INV-1).
2. **Structured, versioned, recorded.** LLM outputs are JSON against a schema,
   the prompt and model are versioned in the record, raw output is kept, and
   every decision can be replayed from the event log.
3. **No look-ahead, anywhere.** Context is point-in-time. An LLM is evaluated
   only on news after its training cutoff (GLM-5.2: ~2025-09; scored from
   2025-12). Each event is judged alone or with earlier items only (the
   2026-10-10 batch reading leaked later headlines). Backtests enter at the
   first price the engine could really have had.
4. **Measure before building, pre-register before measuring.** Every edge is
   stated with its rule before the data is seen; train 2016-21, validate
   2022-24, hold out 2025-26; trials go in the ledger (`quant_trials`) with a
   deflated Sharpe for the number tried. A failed test stops the line of work.
5. **Realistic simulation.** Minute-level, with latency, spreads and impact by
   liquidity (study.COST_BPS), halts, gaps and partial fills. A playbook that
   only works at the close of the minute it was triggered does not work.
6. **Every playbook has an exit before it has an entry.** A stop below the
   capitulation low, a profit target, a failure condition, a time stop, and an
   abort on new information. No open-ended positions.
7. **Shadow, then paper, then money.** Shadow live until the forward record
   matches the backtest; paper at small size; real money only through the
   written capital gates (the core's model).
8. **Risk is portfolio-level.** News days cluster (one macro headline moves
   fifty names): caps on concurrent playbooks, per-sector exposure, the daily
   loss limit, the kill-switch, and correlation of open playbooks.
9. **Observability.** Every state transition is an event (dashboard, Telegram,
   decision log), with its reason; the operator can see why the engine holds
   what it holds.
10. **Cost-aware.** LLM calls are budgeted per event and metered (the
    2026-10-09 credit outage is the lesson): no call is made that the playbook
    cannot use.

## 3. Architecture

The engine lives in **`src/halabot`**, which already has the event log, the
bus, perception (news, bars), beliefs with price levels (support, resistance,
invalidation), a conviction calibrator, policy gates, sizing, a risk engine,
dormant execution, and A/B and significance tooling. A new **playbooks**
layer sits between cognition and policy; it runs in shadow from day one.

```
 news ──► Story builder ──► Event interpreter (LLM) ──► Event card
          (cluster, dedup,     (type, materiality,         │
           entity resolution)   permanence, expected path) ▼
 bars/quotes ─────────────────────────────────────► Playbook engine
                                                     (one state machine
                                                      per live event)
                                                          │ proposals
                                                          ▼
                                     Policy gates · sizing · risk engine
                                     (halal, long-only, kill-switch, limits)
                                                          │
                                                          ▼
                                     Execution (shadow → paper → live)
                                                          │
                                                          ▼
                                     Decision log ─► outcomes ─► calibration
```

### 3.1 Story builder (deterministic)

Ten to thirty headlines about one event become **one story**: clustered by
symbol, time and text similarity; repackaged headlines folded in; the
company resolved correctly (the 2026-10-10 reading found "AKAM" tags on an
Alkami story). A story has a start time, its headlines, its first tradable
moment and its market context at that moment.

### 3.2 Event interpreter (LLM)

One call per new story (and per material update), returning an **event card**:

```json
{
  "event_type": "guidance_cut",          // taxonomy below
  "materiality": 0.8,                    // how much it changes value
  "surprise": "worse",                   // vs what the context said was expected
  "permanence": "transient",             // transient | structural | unclear
  "scope": "company",                    // company | peers | sector | market
  "expected_path": "overreaction_bounce",// one of the playbook archetypes
  "key_uncertainty": "whether the cut is one-off demand timing",
  "invalidation": "a second cut, or peers guiding down",
  "confidence": 0.6
}
```

Taxonomy (first version): earnings result, guidance raise/cut, M&A
(target/acquirer), legal/regulatory, product/contract, management change,
analyst action, offering/dilution, accounting/short report, operational
incident, macro/sector, routine/noise.

Best practice: a fixed schema with validation; the same prompt for
evaluation and production; several samples with agreement as confidence
(self-consistency); calibration of `confidence` against outcomes; a rules
fallback (8-K item codes, earnings facts, keywords) so the engine is
measurable on 2016-24 history, where an LLM would leak.

### 3.3 Path model

For each (event type, move size, liquidity, market regime), the empirical
distribution of what happens next: how deep and how long the dip, how often a
rebound comes, how far it goes, how often it fades. Built from history (rules
classification over 2016-24, the LLM's on post-cutoff data). This is what
turns "wait for the dip" into numbers: wait until X, expect Y, give up at Z.

### 3.4 Playbook engine (deterministic state machines)

One instance per live story, for example **overreaction bounce**:

| State | Enters when | Leaves to |
|---|---|---|
| **Detected** | a negative card with permanence transient/unclear and materiality above a floor | Watching, or Dismissed |
| **Watching** | the drop exceeds k × ATR from the pre-news price | Armed when selling exhausts; Expired at its time limit; Aborted on new negative news or a market-wide break |
| **Armed** | exhaustion signals: no new low for n minutes, a volume climax then fading volume, reclaim of VWAP or the opening range high; optionally an LLM check ("anything new that makes the damage permanent?") | Entered on a limit order; Disarmed if a new low prints |
| **Entered** | filled | Managing |
| **Managing** | target: a fraction of the gap retraced; trailing stop under the higher lows; stop under the capitulation low | Exited on target, stop, failure (rejection at the pre-news level or VWAP, momentum stall), time stop, or abort |
| **Exited / Expired / Aborted** | terminal; outcome recorded | — |

Other archetypes, each measured before it is built: **gap-and-go** (material
positive surprise, enter on the first pullback that holds), **sympathy
laggard** (a peer moves, this name has not yet), **post-earnings drift**
(confirmed beat-and-raise, held for days), **noise fade** (a sharp move on
routine news).

### 3.5 Execution and risk

Proposals go through halabot's policy gates, sizing and risk engine, then the
same halal-at-the-order-boundary gate as every other order. Limit orders by
default; position size from the playbook's stop distance (risk per trade as a
fraction of equity), capped per name and sector; the daily loss limit and
kill-switch apply to playbooks like any entry.

### 3.6 Decision log and learning

Every transition, with its inputs (card, prices, signals) and reason, goes to
the event log; outcomes are labelled when a playbook ends. The reactor's
`reactor_decisions` table is the start of this record. Learning is offline
and gated: parameters move only through a new pre-registered test.

## 4. Phases

Each phase ends with a gate. A failed gate stops the line of work, or sends it
back to an earlier phase with a new hypothesis.

### Phase 0: foundations (about 2 weeks; no LLM spend)

- Minute-bar history for liquid halal tech names on their news days, 2016 to
  now (Alpaca SIP; ~40k symbol-days, ~4-5 h at 150 req/min, ~2 GB).
- Story builder: clustering, dedup, entity-resolution checks.
- Rules taxonomy from 8-K items, earnings facts, insider filings and keywords,
  so event types exist for all history.
- Point-in-time context store (filings, earnings facts, prior moves, levels).
- An event-driven, minute-level simulator: latency, costs by liquidity, halts,
  partial fills; replays a playbook on one story at a time.
- **Gate:** the simulator reproduces known effects (the SUE drift of
  2016-19, the reactor's −0.30%) within their confidence intervals.

### Phase 1: the path atlas (about 1-2 weeks; no LLM spend)

- For each event type and move size: dip depth and timing, rebound
  probability and size, fade probability, by regime and liquidity.
- Pre-registered H1: *after a large drop on non-structural negative news, a
  long entered on exhaustion signals and exited on retrace targets or failure
  earns a positive expectancy net of cost.* Train 2016-21, validate 2022-24.
- **Gate:** H1 passes on both windows for at least one event family with
  enough events to trade (≥ 200 a year across the universe). If nothing
  passes, the playbook idea is shelved before any LLM work.

### Phase 2: the deterministic playbook (about 2-3 weeks)

- Overreaction bounce as a state machine in `src/halabot/playbooks`, driven
  by the simulator and by live bars in the same code.
- Few parameters, fitted walk-forward; every variant in the trial ledger;
  deflated Sharpe for the number of variants.
- **Gate:** validation-window expectancy positive with t ≥ 2 after costs and
  multiple-testing correction; drawdown and concurrency within risk limits.

### Phase 3: the LLM interpreter (about 2 weeks; ~$ few per 1k events)

- Event cards for post-cutoff stories (one story per call, no batching
  leaks), schema-validated, self-consistent, versioned.
- A/B on the same stories: playbooks routed by LLM cards vs by rules.
  Separately: does an LLM check at Armed and Managing improve exits?
- **Gate:** LLM routing improves net expectancy significantly on held-out
  post-cutoff stories, at a cost per event the budget can carry. If not, the
  engine runs on rules and the LLM writes only the human explanation.

### Phase 4: shadow live (4-8 weeks)

- The engine runs inside the shadow process: real stories, real bars, state
  machines advancing, simulated fills at realistic prices; Belief Board and
  Telegram show live playbooks; nothing is placed.
- **Gate:** at least 50 completed playbooks; realised shadow expectancy within
  the backtest's confidence interval; no operational faults (stuck states,
  missed exits).

### Phase 5: paper, then capital

- Paper orders at a quarter of the intended size through the executor's
  gates; then full paper size; then the written capital gates.
- The reactor's current entries stay in shadow until this phase passes.

## 5. What could go wrong

| Risk | Mitigation |
|---|---|
| The bounce effect is real but too small after costs | Phase 1 measures it before anything is built; limit orders; liquid names only |
| Overfitting many parameters to few events | few parameters; walk-forward; trial ledger; deflated Sharpe; untouched holdout |
| LLM leakage makes history look better than live | LLM evaluated only after its cutoff; one story per call; rules baseline for history |
| A "transient" drop is actually structural (falling knife) | hard stop under the capitulation low; abort on new negative news; materiality floor; time stop |
| Clustered news days (macro) open many correlated playbooks | concurrency and sector caps; market-wide breaks abort |
| LLM or data outage mid-playbook | exits are deterministic and need no LLM; an outage blocks new playbooks only |
| Credit or budget exhaustion | per-event budget, account-balance alert (already live), fallback to rules |
| Halal drift (a name stops passing mid-playbook) | the halal gate at every order; a failing name exits at the next check |

## 6. Decisions for the operator

1. **Universe:** halal tech only (consistent with the tech focus), or every
   liquid halal name for more events.
2. **Holding period:** intraday only (closed by the end of day, like today's
   day-trader) or up to a few days (more room for rebounds, overnight risk).
3. **Budget:** the LLM cost per event for Phase 3 and live operation, given
   the OpenRouter balance.
4. **Where it runs live:** inside halabot (recommended: event log, beliefs,
   shadow-first already exist) rather than extending the legacy reactor.
