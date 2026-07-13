# Quantitative prediction roadmap — "how high / how low"

Working backlog for grounding the stock predictions (the daily stock-of-the-day
recommendation and the live stock cycle's target/stop levels) in best-in-class
mathematical and financial forecasting. Today the levels are LLM guesses with
only crude repairs: the recommendation engine enforces ordering with fixed
fallbacks (`recommendation/engine.py:_validate` — bad stop → entry×0.95, bad
target → entry×1.08), and the live path's guard sits in the executor
(`trading/executor.py:_sanitize_long_risk_levels` — invalid stop clamped to 5 %
below fill, invalid target *dropped to None*). This roadmap replaces guesses
with a quantitative **PriceOutlook** per symbol — expected high/low bands with
measured coverage, a deterministic level map, and options-implied ranges —
while the LLM keeps the roles it is good at: narrative synthesis, catalyst
weighing, and choosing among quantitatively pre-ranked candidates.

Same working agreement as `docs/ROADMAP.md`: pick the next unchecked item,
build a tested+committed slice, check it off. Advisory surfaces first; nothing
touches live sizing/execution without offline validation on disjoint OOS
windows.

---

## Honest foundations (read before building anything)

The research base rates that shape every design choice below:

- **Volatility and range are the predictable quantities.** HAR-family models
  reach 20–60 % out-of-sample R² on next-day realized vol/range. Daily
  *returns* are near-white-noise: honest OOS R² ≈ 0.4 % **per month** at the
  stock level (Gu–Kelly–Xiu), direction accuracy ceiling ~52–55 %, good
  cross-sectional IC = 0.02–0.05. Architect everything as **predictable scale
  around a near-zero, unpredictable center** — never sell the band midpoint
  as a price target.
- **The horizon high/low is a path extreme, not a terminal price.** For a
  driftless walk, P(max > b) ≈ 2·P(close > b) (reflection principle): a band
  that contains the *close* 90 % of the time contains the *high/low* only
  ~80 % of the time. Bands must either model the running max/min directly
  (Monte Carlo path extremes, quantile regression on max/min targets) or
  carry an empirically calibrated widening factor. A second correction stacks
  on top: daily-step models see only daily extremes — the true intraday
  high/low is systematically more extreme (fix via an empirical
  high-vs-close ratio or Brownian-bridge adjustment, or let the conformal
  layer absorb it — but decide and say which).
- **Coverage must be measured, not assumed.** Three biases compound (path
  extremes, fat tails, low-biased range estimators); the fix for all three is
  one empirical calibration step of the band multiplier on out-of-sample
  coverage. Metrics: pinball loss per quantile, PICP (interval coverage) vs
  nominal — also *conditional* on vol regime — Winkler score so width can't
  be gamed, and Kupiec/Christoffersen tests on the outer band.
- **The overfitting graveyard is the default outcome.** With a ~20-symbol
  universe and a few years of dailies, the expected max in-sample Sharpe of
  pure noise exceeds 1.0 after only **~10–20 tried variants** — the trap
  engages before it feels like many attempts. Every candidate signal needs: a
  trials ledger (`core/sharpe_stats.py` PSR/DSR already exists — feed it an
  honest trial count), disjoint-OOS validation, and a **placebo control** for
  level-type signals (distance/width-matched random levels — Fibonacci
  famously fails exactly this test).
- **Repo lessons still bind** (`docs/OPERATOR_CONTEXT.md`): per-asset
  technicals historically did NOT survive disjoint-OOS in halabot (Donchian
  breakout lost in 4/5 windows) while market-relative signals did; conviction
  is near-optimal and the edge is *better inputs*. This roadmap is exactly a
  better-inputs program — but treat the literature's level hold rates as
  upper bounds, expect **most level families to fail placebo**, and
  pre-register each family's success criterion in the trials ledger before
  running it. That is the plan working, not failing: the survivors are the
  product.
- **Exit-safety principle (fast in, slow out — binding).** Quant levels may
  **widen or floor stops, never raise them**, and in the live path targets
  are **advisory/flag-first, never auto-tightened**: the stock monitor
  hard-exits the instant price touches `target_price`, so snapping targets
  down to a "realistic" range would systematically exit winners early — the
  single failure mode the operator designed against. An over-ambitious TP is
  *aligned* with slow-out (the wide trailing stop becomes the exit).
  Positions with `entry_type='reactor_momentum'` are untouchable: no quant
  level ever modifies their targets, stops, or exits.

**What this system should claim when done:** next-day/next-week per-symbol
range width and high/low bands with measured coverage (the crown jewel);
risk-environment classification (vol regime, VIX term structure) good enough
to size and gate; deterministic level maps with placebo-beating hold rates; a
small direction tilt (IC ≈ 0.02–0.05) from regime-gated reversal/momentum
features. **What it must never claim:** point forecasts of tomorrow's price,
which of high/low gets touched first intraday, or >55 % daily direction
accuracy (treat any backtest showing that as leakage until proven otherwise).

---

## Architecture: the PriceOutlook engine

New package `src/halal_trader/quant/` — pure-numpy deterministic core (house
style: no pandas/scipy **in the deterministic layer**; the `[ml]` extra may
pull them transitively), heavy models behind `[ml]` with graceful degradation.
One entry point:

```
quant/outlook.py:build_outlook(symbol, klines, *, horizon_days=(1, 5), ...)
  -> PriceOutlook
```

`PriceOutlook` (frozen dataclass) per symbol:

- `bands`: per-horizon expected **path** high/low at chosen quantiles
  (q10/q25/q50/q75/q90 of the running max and min), with the estimator that
  produced them (`atr` | `har_yz` | `garch_fhs` | `chronos` | `ensemble`) and
  its calibration version.
- `levels`: deterministic level map — prior day/week/month H/L, anchored
  VWAPs, swing-cluster zones with touch counts, round-number snaps, pivots —
  each tagged with its measured OOS hold rate (or `unvalidated`).
- `vol`: Yang-Zhang / EWMA vol state, ATR, vol-percentile vs own 1-y history
  with a vol-cone shrinkage flag, expected 1d range in $ and %.
- `implied` (optional, Phase 3): options expected move, ATM IV, IV percentile,
  25Δ skew flag.
- `regime` (portfolio-level, computed once per cycle): VIX term-structure
  state, SPY-vs-SMA trend gate.

Consumers (in wiring order):

| Consumer | How |
|---|---|
| `recommendation/engine.py:_build_candidates` | outlook fields join the per-symbol summary → prompt table + `candidates` JSONB (zero API change) |
| `recommendation/engine.py:_validate` | **asymmetric** grounding: floor the stop at the calibrated lower band (widen-only), flag targets outside the upper band; store raw-LLM and quant levels side by side for A/B scoring |
| `recommendation/scorecard.py` | labels realized high/low; scores coverage, pinball, hit rates |
| `trading/cycle.py` stage list | new `BuildOutlookStage` → one prompt block with expected ranges + level map per symbol |
| `trading/strategy.py` post-LLM | advisory feasibility flag on unrealistic targets (hard snapping only after offline A/B evidence, and never on reactor positions) |
| dashboard / CLI | bands + levels on the recommendation page and per-symbol views |

Housekeeping that comes with the package: calibration state (band-z
multipliers, conformal residual windows) is a **persisted, versioned
artifact** — an `ml_artefacts`-style row or Alembic table with a
`calibration_version` stamped into every PriceOutlook, plus staleness
detection (age + sample-count guards). Walk-forward calibration code lives in
`halal_trader/quant/` (not in `src/halabot/` — the halabot harness is reused
for *replay validation only*, keeping the two package trees from growing
duplicate learners). New failure modes get `core/events.py` constants and
`AlertSink.notify` call sites per house convention; new CLI commands follow
the lazy-import pattern.

---

## Phase 0 — Measure first (the loop is currently blind)

The scorecard discards highs/lows (`scorecard.py:_closes_by_date` keeps only
closes), so nobody knows whether today's LLM targets/stops are any good. No
forecasting work ships before the measurement loop can score it.

- [x] **Keep OHLC in outcome labeling** — extend `_closes_by_date` →
  `(date, high, low, close)`; in `backfill_outcomes()` compute per-horizon
  realized max-high / min-low, MFE/MAE, `target_hit` / `stop_hit` /
  `first_hit` (touch sequencing), time-to-hit. New outcome columns on
  `DailyRecommendation` via Alembic (`realized_high_5d`, `realized_low_5d`,
  `mfe_pct`, `mae_pct`, `target_hit`, `stop_hit`, `first_hit`). They flow
  through `update_recommendation_outcome`'s hasattr-guarded setattr loop —
  the model change alone is sufficient, no repo change.
- [x] **Fix silent mislabeling of stale picks** — the real bug is worse than
  "never scored": `_forward_returns` anchors on the *first bar ≥ rec date*,
  so a pick older than the 90-day fetch window scores against the window's
  first bar — wrong entry, plausible-looking returns, `scored` status.
  Guard: require the bar window to contain the rec date, else write
  `outcome_status='skipped'` (a documented value on the plain-string column —
  nothing sets it today, no migration needed). Include a **one-time audit /
  re-label of already-scored rows** whose rec date predates their bar window;
  the baseline scorecard below is untrustworthy until this runs.
  Done 2026-07-13: guard + `skipped` status + `halal-trader recommend
  --audit-outcomes` (operator should run it once).
- [x] **Anchor outcomes to the plan, not the close** — forward returns are
  currently anchored to the rec-date close, not `suggested_entry` or the
  open, and `whatif_equity_curve` ignores stops/targets entirely. Add
  open-anchored (and, where marketable, entry-anchored) variants plus a
  bracket-aware what-if (entry → first of target/stop/time-exit) so the
  scorecard scores the *stated plan*. Done 2026-07-13 (migration
  d4b8f0a2c6e5): `entry_open` + `_plan_bracket` simulation (gap-through
  exits at the open; same-bar ties resolve to the stop — pessimistic),
  `plan_return_5d`/`plan_exit` columns, plan equity curve in the
  what-if + `plan_exit_counts`/`avg_plan_return_5d` in the scorecard.
- [x] **Baseline scorecard for the LLM's own levels** — target-hit rate,
  stop-hit rate, which-first stats, implied-R vs realized-R, all
  `SampleGate`-gated. This is the baseline every quant method must beat.
  Done 2026-07-13 (target/stop hit rates, first-hit counts, MFE/MAE in
  `compute_scorecard` + CLI panel); implied-R vs realized-R still open
  with the plan-anchoring item above.
- [x] **Forecast-evaluation primitives** — `quant/eval.py` (pure numpy):
  pinball loss, interval coverage (PICP), Winkler score, Kupiec/
  Christoffersen coverage tests, coverage-by-regime splits. Unit-tested
  against known values; shared by scorecard and backtests. 2026-07-13.
- [x] **Trials ledger** — a small Postgres table recording every strategy /
  forecast variant evaluated (name, config hash, window, metrics,
  pre-registered success criterion), so `deflated_sharpe_ratio` gets an
  honest trial count. CLI: `halal-trader quant trials`. Done
  2026-07-13 (`quant_trials` table, migration c7a2e9f4d1b3;
  `db/repos/quant_trials.py`; `quant calibrate`/`validate-levels`
  self-record with pre-registered criteria — first rows: zcal pass,
  swing_zones fail, prior_extremes/round_numbers inconclusive).
  Scorecard now also reports `band_coverage_5d` from the bands stored
  in candidates JSONB (live coverage feed, no recompute leakage).
- [ ] **Overnight-vs-intraday decomposition diagnostic** — trivial from daily
  OHLC (`open/prev_close − 1` vs `close/open − 1`): reports how much drift
  our intraday-only stock strategy structurally forfeits (most equity drift
  accrues overnight) and contextualizes the bot's own fills. Diagnostic
  panel only — expectations-setting for the operator, not a signal.
- [x] **Label all candidates, not just the pick** — outcome-label the whole
  `candidates` JSONB universe per rec (open-anchored forward returns +
  realized high/low). Gives counterfactual evaluation ("did the LLM pick the
  best candidate?") and the pooled training panel Phase 2 needs. Mind
  `get_recommendations_to_score`'s newest-first `limit=500` cap once
  candidate labeling multiplies pending work. Done 2026-07-13: outcomes
  written into `candidates[sym]["outcome"]` JSONB (no migration), one
  cached bar fetch per symbol per run; scorecard gains
  `candidate_band_coverage_5d` (~20× the coverage sample) and
  `avg_pick_percentile_5d` (counterfactual: 0.5 = random picking).
- [ ] **Data hygiene: adjusted bars** — split/dividend-unadjusted OHLC
  corrupts overnight terms in vol estimators, shifts historical levels, and
  distorts return labels. Establish what adjustment the Alpaca MCP bars
  carry, request adjusted bars if available, and record the convention in
  the bar cache; add a corporate-action sanity check (gap > x % with no
  news) to labeling.

## Phase 1 — Deterministic range + levels engine (pure math, no new deps)

Everything here is 10–40 lines of numpy, offline-testable, and consumable by
both bots. Ship behind the outlook API with per-piece validation. Build the
measurement harness *before* the level families it measures.

- [x] **Range-based vol estimators** — `quant/volatility.py`: Parkinson,
  Garman-Klass, Rogers-Satchell, **Yang-Zhang** (the default: minimum-variance
  OHLC estimator, handles the overnight gaps that dominate large-cap risk),
  plus EWMA (λ=0.94) on the YZ series. Requires adjusted bars (Phase 0
  hygiene item). Unit-test against published values. 2026-07-13
  (`quant/volatility.py`: close-to-close baseline + Parkinson, GK, RS,
  Yang-Zhang, EWMA; daily units, NaN warm-ups).
- [x] **HAR range/vol forecaster** — HAR(1, 5, 22) OLS on log YZ-vol, one
  direct model per horizon (h=1, 5). Pure-numpy least squares; ~30 lines.
  Captures *most of* the documented 35–40 % HAR-RV error reduction over
  GARCH at 1–5 d horizons (the full figure needs intraday realized vol; the
  daily-OHLC range proxy retains most of the gain). Fit in logs with the
  half-variance bias correction on exponentiation; direct multi-step models
  need non-overlapping evaluation windows. 2026-07-13 (`quant/bands.py:
  fit_har` — direct per-horizon, log-space, half-variance correction,
  refuses <60 rows).
- [x] **Band conversion with empirical calibration** — `quant/bands.py`:
  `close·exp(±z·σ̂·√h)` with **z calibrated per horizon on walk-forward
  realized max-high/min-low coverage** — this one calibration step absorbs
  the path-extreme, fat-tail, and estimator-bias corrections. The research
  says calibrate per symbol (coverage multiples genuinely vary by symbol and
  regime); with a ~20-symbol universe that is data-starved, so: **start
  pooled across the universe, add per-symbol shrinkage as coverage samples
  accrue, and track per-symbol coverage residuals in the scorecard** to know
  when pooling is costing accuracy. Also expose expected range
  `E[range] ≈ 1.596·σ̂·√h` and the ATR-multiple band as the naive baseline
  every model must beat. Primitives done 2026-07-13 (`quant/bands.py:
  price_bands`, `atr_band`, `calibrate_z` — binding-z quantile over
  realized path extremes). Completed 2026-07-13: `quant/calibration.py`
  (expanding-window walk-forward HAR refit, pooled across the AAOIFI-20,
  versioned JSON artifact + mtime-cached loader the outlook consumes) +
  `halal-trader quant calibrate` (writes `data/analytics/
  band_calibration.json`, caches bars under `data/bar_cache/`). First
  real run zcal-20260713-c80: z(1d)=1.65, z(5d)=1.84 at 80 % path
  coverage (n≈3.2k pooled; per-symbol residuals 73–89 %) — vs textbook
  1.28, confirming the path-extreme/fat-tail under-coverage. NOTE the
  artifact is host-local runtime data (gitignored): the containerized
  daily job needs `quant calibrate` run in-container (or a shared
  volume) to pick it up. Per-symbol shrinkage stays a future item.
- [ ] **Volatility cone guardrail** — per-symbol percentile cone of rolling
  YZ vol over window lengths [5, 10, 21, 42, 63] (2+ years of dailies,
  Hodges-Tompkins overlapping-window correction); when current vol sits at a
  percentile extreme, shrink σ̂ toward the cone median (vol mean reversion)
  and flag the band as regime-extreme. ~20 lines; a shrinkage prior, not a
  forecaster.
- [ ] **Cornish-Fisher tail bound** — reuse `ml/bayesian_var.py`
  (skew/kurtosis-adjusted downside quantile; **no default-on callers** —
  reachable only via the opt-in crypto agentic `compute_var_95` tool, so
  keep `bayesian_var`/`render_result` signatures stable) at α=0.05/0.95 on
  daily returns as the fat-tail-honest "how low can it go" cross-check on
  the lower band; surface disagreement with the Gaussian band as a tail-risk
  flag.
- [x] **Touch-and-hold validation harness (build FIRST in the levels track)**
  — the shared honesty tool for every level family: touch = enter level ±
  0.25·ATR; outcomes = reject (≥1·ATR reversal) / break-hold / break-fail;
  always condition on touch; always compare against distance/width-matched
  placebo levels. A level family enters the prompt only if its OOS hold rate
  beats placebo. Honest *intraday* touch/hold measurement needs the intraday
  bars enabling fix (below) — daily-bar approximations are a stopgap and
  must be labeled as such. Results feed the trials ledger with
  pre-registered criteria. Done 2026-07-13 (`quant/level_eval.py` +
  `halal-trader quant validate-levels`, daily-bar approximation,
  same-side distance-band placebo, seeded).
- [~] **Level map: prior-extreme family first** — `quant/levels.py`: prior
  day/week/month high/low (zero parameters; Osler documents the
  stop-clustering *mechanism*; the 55–70 % first-touch respect rates are
  practitioner-measured intraday numbers — treat as upper bounds pending our
  own harness) + round-number snapping (modifier only). Then anchored VWAP
  from *mechanical* anchors (last major swing, largest gap ≥ 4 %, month/YTD
  start — rules fixed ex ante). Then swing-point detection (rolling-window
  extrema with ATR-scaled prominence — mind the confirmation-lag lookahead
  bug) + clustering of touches into zones ranked by touch count.
  Classic/Camarilla pivots as free extra features. **Skip Fibonacci**
  (peer-reviewed: no better than width-matched random zones).
  Families built 2026-07-13 (`quant/levels.py`: prior extremes, round
  numbers, swing zones + `level_map` with round-snap; AVWAP still open).
  **First validation verdict (400d × 20 symbols, daily approximation):
  swing_zones FAIL placebo (−1.8pp — the halabot per-asset-structure
  lesson repeats); prior_extremes +2.1pp and round_numbers +2.8pp are
  positive but ~1σ — insufficient. NO family is prompt-wired.** Next:
  re-test on disjoint OOS windows and with intraday bars before any
  wiring; treat the current uplifts as a screen, not evidence.
- [ ] **Direction tilt (the only return-side item, and it is small)** —
  regime-gated short-term reversal: RSI(2)-style oversold entries in liquid
  large caps, gated by SPY-vs-200d-SMA *and* a vol-regime cap (the gate is
  what removes buying-dips-into-crashes). Best-evidenced short-horizon
  return edge (~30–50 bps/week net in careful large-cap studies, hit rate
  53–56 %). Plus the one-line 52-week-high proximity feature
  (`close / rolling_max(close, 252)`). Both validated via
  `core/signal_eval.py` IC with t-stat > 3 on disjoint windows before
  entering any prompt; framed as tilts, never as forecasts.
- [~] **Wire into the recommendation engine — advisory slice** — outlook
  fields into the candidate table (expected 1d/5d band, nearest support
  below / resistance above with hold rates, vol percentile) + a
  "quantitative context" prompt block; persist per-candidate bands in
  `candidates` JSONB and pick-level typed columns
  (`predicted_low_5d`/`predicted_high_5d` + model tag + calibration
  version). No level authority yet. Bands half done 2026-07-13
  (`quant/outlook.py:build_outlook` + engine wiring: band5d/rng1d/vpct
  in the prompt table with an UNCALIBRATED label, full per-horizon
  bands in `candidates` JSONB, 200-day fetch for HAR). Remaining:
  level map fields (needs `quant/levels.py`) and pick-level typed
  columns once the calibration artifact exists.
- [ ] **Recommendation-level grounding — authority slice (separate, later)**
  — after the advisory slice has accrued Phase 0 metrics: floor the LLM stop
  at the calibrated lower band (**widen-only**), flag targets outside the
  upper band. Keep raw LLM levels in a column for permanent A/B. This is the
  advisory product only — live-path authority is Phase 4 and gated harder.
- [ ] **Target feasibility metric** — per-symbol empirical CDF of
  (high−open)/ATR; compute "target beyond the ~65th-percentile move" as a
  *flag + prompt fact* (and a scorecard dimension), not an auto-snap. See
  the exit-safety principle: in the live path the monitor hard-exits at
  `target_price`, so lowering targets mechanically = exiting winners early.

## Phase 2 — Probabilistic forecasting (`[ml]`-gated, new deps)

Only starts once Phase 0 metrics exist, because every model here ships only if
it beats the Phase 1 ATR/HAR baseline on pinball + Winkler + coverage on
disjoint OOS windows.

- [ ] **GARCH-FHS Monte Carlo of path extremes** — add `arch>=8.0` to `[ml]`
  (cp314 wheels reported by two of three research passes — verify at
  `uv lock`; pure-Python `ARCH_NO_BINARY=1` fallback exists; note arch pulls
  pandas/scipy/statsmodels transitively into `[ml]`, acceptable under extras
  gating). GJR-GARCH(1,1) with skew-t per symbol,
  `forecast(horizon=5, method='bootstrap', simulations=10_000)`, reconstruct
  price paths, take per-path running max/min, report empirical q05–q95.
  One of **two** methods here that model the extreme directly (the other is
  the quantile GBM on max/min targets below). Gotchas the research flags:
  scale returns ×100 for the optimizer, refit on a rolling ~750-day window
  (never fit once), expect fit explosions on short/gappy series, drift
  pinned to zero (5-day drift is unforecastable), and apply the
  intraday-extreme correction (daily-step sim highs are max-of-closes —
  correct with an empirical high-vs-close ratio, or document that the
  conformal layer absorbs it). Constant-vol GBM Monte Carlo (20 lines numpy,
  no deps) as the sanity baseline.
- [ ] **Conformal calibration layer** — rolling-window split conformal per
  horizon on whichever band source serves (Phase 1 HAR bands included);
  MAPIE (`mapie>=1.4`, pure Python, BSD) or a ~15-line hand-rolled adaptive
  conformal (ACI) update in house style. Verify coverage *conditional on vol
  regime* — marginal coverage hides exactly the failure that matters.
- [ ] **Coverage-drift action hook** — don't reproduce the repo's
  DriftRiskPolicy open loop (fitted-but-unconsumed): define the trigger now.
  Rolling coverage breach (e.g. PICP > ±5 pts off nominal over the trailing
  window) → widen bands / bump ACI γ / suppress the band source, plus an
  `AlertSink.notify` and a `core/events` constant. Ship with the conformal
  layer, not after it.
- [ ] **Chronos for stocks (a wrapper rewrite, not a version bump)** — the
  repo has *two* Chronos wrappers: `ml/forecaster.py` (hardcodes
  `chronos-t5-small`, sample-path `predict()`, quantiles discarded after
  text rendering) and halabot's `cognition/chronos_forecaster.py` (already
  calls `predict_quantiles`, then collapses q10/q90 to a scalar vote). Work:
  (a) upgrade `chronos-forecasting` 2.2.2 → 2.3.x (Chronos-2 / Bolt,
  Apache-2.0, CPU-capable) **and port `ml/forecaster.py` to the 2.x
  quantile API**, returning numeric bands to callers; (b) give the rec
  engine its own lazy construction behind a **named flag**
  (`REC_FORECASTER_ENABLED`, default on with lazy degrade-to-None — a
  deliberate, visible setting rather than a silent bypass of the crypto
  `ML_ENABLED` gate); (c) decide the torch-cp314 failure mode up front: if
  the pinned torch has no cp314 wheel at upgrade time, the item *waits* (no
  vendoring); (d) the "daily bars too sparse" exclusion comment in
  `core/cycle_stages.py` is stale — `forecast()` needs ~20 closes and the
  rec engine already fetches 60 daily bars (extend to ~120 for context).
  Published finance evidence says zero-shot TSFMs *lose to GBMs on returns*
  and only *match* HAR on vol — Chronos enters as one conformalized ensemble
  member, capped effort, not the centerpiece.
- [ ] **Quantile GBM on path-extreme targets** — train directly on
  `y_high = max(High[t+1..t+k])/Close_t − 1` and `y_low = min(Low…)` at
  q10/q50/q90, pooled cross-sectionally across the whole universe
  (per-symbol data is far too thin), vol-normalized features and targets.
  Use sklearn `HistGradientBoostingRegressor(loss="quantile")` or the
  already-pinned xgboost `reg:quantileerror` — **not LightGBM** (no cp314
  wheels, stalled release line). Purged/embargoed walk-forward CV only;
  monotonize crossing quantiles; conformalize the output. Features largely
  exist (indicators, YZ vols, gap stats; + day-of-week once `bars.py` stops
  discarding real timestamps — see enabling fixes).
- [ ] **Vol-forecast ensemble** — simple average of HAR and GARCH σ̂ (hard to
  beat), with Chronos/GBM members admitted only on OOS evidence; per-member
  and ensemble pinball/coverage tracked continuously in the scorecard.
- [ ] **Volume profile levels (after intraday bars land)** — the level
  family most defensible from first principles (real order flow happened
  there): ~40 lines of numpy — ATR-scaled price bins, per-bar volume spread
  across its range, POC = argmax, Value Area = greedy 70 % expansion,
  HVN/LVN via histogram peaks; naked-POC revisit stats; the "80 % rule" at
  its honestly measured ~67 %. Needs the intraday-bars enabling fix;
  validate through the same touch-and-hold + placebo harness (including
  volume-shuffled placebo profiles).
- [ ] **Earnings/event awareness** — none of these models see the calendar;
  widen or suppress bands when an earnings date falls inside the horizon.
  Decision to record first: the source is Finnhub's earnings-calendar
  endpoint (already keyed for news; note its rate limits) — Alpaca's
  `get_calendar` is the market-session calendar, and
  `catalysts.py:EarningsCalendarSource` is dormant scaffolding with a
  signature mismatch (fix or replace against the chosen source). Until
  then, at minimum flag band-unreliable symbols in the prompt.

## Phase 3 — Options-implied and market-regime layer (read-only market data)

Halal framing: consuming options *market data* as a signal is reading public
information — the bot never holds, writes, or trades a derivative. Long stock
only, as always. **Prerequisite for the prompt surfaces here:** the
`docs/ROADMAP.md` cross-cutting "prohibited-instrument refusal contract" —
feeding IV/skew/expected-move context into prompts raises the odds the LLM
*proposes* an options position, so the output-side guard (every LLM surface
refuses non-equity instruments) ships before or with the first options-data
prompt block.

- [ ] **VIX term-structure regime gate** — VIX9D/VIX and VIX/VIX3M ratios →
  3-state regime (risk-on / caution / risk-off) with 2-day hysteresis; free
  index history (CBOE CSV primary, yfinance fallback). Fetch runs
  out-of-cycle on the scheduler with cached last-good values — a fetch
  failure degrades the prompt block to "regime unavailable", never blocks or
  delays a cycle. Caveat carried from research: a VIX9D inversion just
  before a scheduled event (CPI/FOMC) is *expectation, not stress* — gate on
  the slow ratio, use the fast ratio as color. Cheapest, most robust
  options-derived input, and exactly the market-relative signal class that
  survived halabot OOS.
- [ ] **Revive the options-IV feed on Alpaca's indicative feed** — the
  retired `trading/options_iv.py` + `options_catalyst_adapter.py`
  scaffolding is intact (prompt formatting, Catalyst kinds,
  CatalystRiskPolicy 0.5× pre-event sizing all reactivate on re-wiring);
  only the dead Yahoo fetcher needs replacing. Alpaca's **free/paper tier
  includes option chains with pre-computed greeks + IV** (indicative feed;
  snapshot-only, no history; delayed/modified quotes). Route: alpaca-py
  `OptionHistoricalDataClient` directly, or a new MCP tool wrapper if the
  installed `alpaca-mcp-server` exposes chains — check its tool list at
  connect, and remember this integration's history of breaking on MCP
  tool/arg renames (`mcp/client.py` comments). Liquidity filters mandatory
  (max spread, min OI).
- [ ] **IV history persistence — ship with the fetcher, before any consumer**
  — Alpaca gives no historical IV, so every day not persisted is signal
  lost, and a *silent* fetch/persist failure is irrecoverable data loss:
  the daily job gets a `core/events` constant and an `AlertSink.notify` on
  failure from day one. One row per symbol per day: ATM IV, expected move,
  25Δ skew, PCR, OI. Alembic table + daily job slot in the existing
  scheduler. Seed the IV-percentile cold start with 20-day realized vol as
  a proxy until ~60 d of real IV history accrues.
- [ ] **Options expected move → `PriceOutlook.implied`** — EM = S·σ_ATM·√T
  (0.85×straddle cross-check; pick ONE day-count convention — calendar vs
  trading days differ by ~√(365/252) ≈ 20 % — and record it). **Never serve
  the front-expiry EM as a normal-day band when earnings fall before
  expiry** — the event jump is embedded; skip the week or use the
  post-earnings expiry. Render next to the statistical band in prompt +
  dashboard; implied-vs-statistical disagreement is itself signal (event
  premium). IV over-forecasts realized vol ~10–20 % *in calm markets* (vol
  risk premium — the bias shrinks or inverts under stress), so treat it as
  a conservative envelope and validate coverage like any other band source.
- [ ] **IV-derived context bundle** (same fetch, marginal cost ~zero): IV
  percentile (size/stop-width modulator once history accrues), 25Δ put skew
  (entry *veto* on names pricing crash risk — cross-sectional rank,
  weeks-horizon), put-call-parity IV spread (mild directional lean; decayed
  Cremers-Weinbaum effect; restrict to short-dated, non-dividend-imminent
  names — dividends/early exercise create mechanical parity deviations that
  are not informed flow), PCR z-score vs own 20-day history (prompt feature
  only; 1–2 d signal that reverses; and note the sign flip: single-stock
  call-heavy flow is bullish, while *market-wide* PCR extremes read
  contrarian — never mix the two interpretations). All prompt/veto features
  — none gate sizing until OOS-validated.
- [ ] **Index-level GEX — research note only (not a pickable slice yet)** —
  net dealer gamma sign on SPY as a third regime feature is attractive
  (positive → mean-reverting tape, negative → trend-amplifying), but it
  needs a data-source decision first (full SPY chain OI daily: pagination
  over Alpaca's 1000-contract pages vs the unofficial CBOE CDN JSON), and
  the known caveats are structural (dealer-positioning assumption
  unverifiable; 0DTE invisible in overnight OI). Write the sourcing note,
  then decide whether it graduates. Per-symbol GEX walls and max pain stay
  **rejected**.

## Phase 4 — Live stock cycle integration + sizing (gated hardest)

Everything above ships advisory-first. This phase moves validated pieces into
the live 15-min cycle's decision path — each item individually
offline-validated per the house rule before default-on. Two standing
constraints repeat because they are binding: **reactor_momentum positions are
untouchable**, and the live universe is currently the ~3-symbol Zoya-sandbox
trio, so per-symbol empirical CDFs and hit-rate stats will be sample-starved
until the operator flips the production key (pool across the AAOIFI-20
universe for anything statistical).

- [ ] **`BuildOutlookStage` in the stock cycle** — per-symbol expected-range
  + levels block in the live prompt (the dormant `sentiment_text` slot or a
  new block in `strategy.py`). Prompt-only first: measure whether LLM levels
  get saner (Phase 0 metrics) before any hard authority.
- [ ] **Post-LLM stop floor in the live path** — stops floored at the
  Cornish-Fisher/GARCH lower band rather than an arbitrary % (**widen-only**:
  a stop wider than the quant floor is left alone; never raised). Targets:
  advisory feasibility flag only — any hard target snapping requires the
  same offline A/B evidence bar as a sizing change, because the monitor
  hard-exits at `target_price` and lowering targets mechanically exits
  winners early. Never touches reactor positions.
- [ ] **Vol-aware sizing** — position size scaled by forecast σ̂ (constant
  dollar-vol targeting) — this is `docs/ROADMAP.md` Phase 1's sizing loop
  fed by this roadmap's vol engine; coordinate, don't duplicate. Half-Kelly
  primitive (`core/sizing.py`) gets its win/payoff stats from the Phase 0
  hit-rate columns. Offline evidence first, per the standing rule.
- [ ] **Close the calibration loop while here** — `ml/calibration.py` curves
  are fit-and-forget today (`apply_calibration` has zero call sites;
  `InsightsHub.calibration` stays identity). Load the fitted curve at
  composition time and report calibrated confidence alongside raw in
  prompts/scorecard. (`docs/ROADMAP.md` Phase 1 owns wiring calibrated
  confidence into *sizing*; this item only closes the load-and-display
  loop.)
- [ ] **Stocks retraining loop construction** — `trading/monitor.py` already
  calls `retrainer.on_trade_closed` but the scheduler never constructs a
  stocks-namespace `RetrainingScheduler`; construct it (and the
  anomaly/classifier pair if `ML_ENABLED`) so stock outcomes start labeling
  snapshots — the supervised loop any learned forecaster rides.

## Phase 5 — Product surfaces

- [ ] **Recommendation page: bands + levels** — predicted high/low band
  (statistical + implied) vs realized path per pick; hit/coverage badges in
  the history table (it already renders suggested vs fwd returns side by
  side); scorecard tiles for coverage %, pinball trend, target/stop hit
  rates.
- [ ] **Per-symbol research view** — chart with level map (prior extremes,
  AVWAPs, swing zones with hold rates), current bands per horizon, vol
  percentile, IV context. **This roadmap owns the item**; `docs/ROADMAP.md`
  Phase 4's research-page entry should point here when picked up.
- [ ] **CLI** — `halal-trader quant outlook SYMBOL` (print the full
  PriceOutlook), `halal-trader quant coverage` (rolling coverage report).
  Lazy-import heavy deps inside the command functions per house convention.
- [ ] **Telegram digest hook** — daily pick message gains "expected range /
  levels / implied move" lines. Depends on the (unbuilt) advisory digest
  item in `docs/ROADMAP.md` Phase 4 — build that first or bundle.

## Cross-cutting enabling fixes (small, unblock multiple phases)

- [ ] **Preserve real bar timestamps** — `trading/bars.py:bars_to_klines`
  replaces real times with synthetic monotonic ms, blocking day-of-week /
  session features and correct multi-day intraday alignment. Parse the real
  `t` field (audit downstream consumers first — they claim to use ordering
  only).
- [ ] **Intraday bars fetch path** (prerequisite for the Phase 1
  touch-and-hold harness's honest mode and the Phase 2 volume profile) —
  `get_stock_bars` passes the timeframe straight through; `15Min`/`1Hour`
  bars enable true realized-vol estimates, opening-range features, intraday
  touch/hold measurement, and Chronos context without protocol changes. Add
  a small throttled fetch path + cache.
- [ ] **Daily-bar research cache** — reproducible OOS validation needs
  same-bars replays (the halabot lesson: comparing across separate live
  fetches is invalid). Either reuse halabot's `--cache-write/--cache-read`
  JSON caches or add a `quant/` bar-cache keyed by (symbol, timeframe,
  window, adjustment convention).
- [ ] **Show the LLM more than 5 bars** — `strategy.py:_format_bars` renders
  only the last 5 daily bars of the 60 fetched; once outlook blocks land
  this may be moot (scalars beat raw bars), but decide deliberately rather
  than leaving the LLM range-blind by accident.
- [ ] **Forecast-accuracy metrics in the halabot backtest `_Book`**
  (optional but high-leverage): record active [q10,q90] per proposal and
  score realized coverage/pinball alongside P&L, bucketed by the existing
  RegimeStats splits — makes `halabot backtest --oos-splits` the one harness
  that validates both P&L edges *and* forecast quality.
- [ ] **Expose raw series to quant/** — `compute_all` returns rounded point
  scalars only (2–6 dp); quant needs the raw arrays. Compute in `quant/`
  from klines directly (preferred — no churn in shared code).

## Dependency plan (Python 3.14, uv)

| Package | Status | Where | Note |
|---|---|---|---|
| numpy (2.4.2) | already core | `quant/` core | house style: pure numpy in the deterministic layer |
| `arch >= 8.0` | **add** | `[ml]` | GARCH/GJR/HARX, bootstrap path sims; cp314 wheels reported (verify at `uv lock`; `ARCH_NO_BINARY=1` source fallback); NCSA license; pulls pandas/scipy/statsmodels transitively — acceptable inside `[ml]` only |
| `mapie >= 1.4` | **add** | `[ml]` | conformal intervals; pure Python, BSD-3; v1.x API (ignore v0.x tutorials) — or hand-roll ~15-line ACI and skip the dep |
| scikit-learn (1.8) | already `[ml]` | quantile GBM | `HistGradientBoostingRegressor(loss="quantile")` |
| xgboost (3.2) | already `[ml]` | quantile GBM alt | `reg:quantileerror` fits multiple quantiles in one model |
| `chronos-forecasting` 2.2.2 → **2.3.x** | upgrade + port | `[ml]` | a wrapper rewrite of `ml/forecaster.py` (t5-small `predict()` → Chronos-2/Bolt `predict_quantiles`), Apache-2.0; blocked if the torch pin has no cp314 wheel — wait, don't vendor |
| statsmodels | transitive via arch | HAR diagnostics | cp314 wheels; use directly only if numpy OLS proves insufficient |
| scoringrules | optional | eval | or hand-roll pinball/Winkler/Kupiec in `quant/eval.py` (preferred, ~60 lines) |

**Explicitly rejected:** LightGBM (no cp314 wheels, release line stalled —
sklearn HistGB instead); pandas-ta / TA-Lib (everything needed is a few lines
of numpy); `ta` (dead since 2023); Lag-Llama (unmaintained, superseded);
Moirai 2.0 (CC-BY-NC weights); yfinance anywhere near the live path (Yahoo
blocking; VIX-index fallback only); mlfinlab (commercial license — reimplement
DSR/PBO from the papers); hmmlearn (no cp314; vol-percentile rule or
statsmodels MarkovRegression if regime-switching is ever needed); a dedicated
PEAD engine (contested post-2010s, data-starved at 4 events/symbol/year);
Fibonacci levels (fails width-matched placebo in peer review); per-symbol
GEX/max-pain trading levels (unverifiable dealer assumptions, 0DTE-blind).

## Validation gates (what "done" means for any forecast)

1. **Beats the naive baseline** (ATR-multiple band / constant-vol GBM MC) on
   pinball + Winkler + coverage, on **disjoint OOS windows**, same-bars
   replay (cached bars), per the halabot methodology.
2. **Coverage holds conditionally** — PICP within ±5 pts of nominal in both
   calm and high-vol regime buckets, not just marginally — and the
   coverage-drift action hook (widen / suppress / alert) is wired, not
   aspirational.
3. **Levels beat placebo** — touch-conditioned hold rate exceeds
   distance/width-matched random levels OOS, against a success criterion
   pre-registered in the trials ledger.
4. **Trials-ledger honest** — DSR computed with the true variant count; PBO
   check for anything tuned.
5. **Advisory before authority** — a signal appears in prompts/dashboard ≥
   a few weeks and its Phase 0 metrics accrue before it grounds levels;
   level authority is always asymmetric (stops widen-only, targets
   flag-only in the live path); sizing changes additionally follow the
   `docs/ROADMAP.md` Phase 1 offline-first rule; reactor positions are
   never touched.
6. **Degrades gracefully** — every `[ml]` component lazy-loads and
   no-ops to the deterministic layer when unavailable (the
   `LazyChronosForecaster` pattern); the recommendation must still generate
   with zero optional deps installed.

## Operator dependencies

- [ ] **Options data tier decision** — free Alpaca indicative feed is the
  default (delayed/modified quotes; fine for ATM IV/EM on liquid names). If
  its noise proves too high on our universe: Algo Trader Plus (~$99/mo, real
  OPRA) is an operator-only spend decision.
- [ ] **`ML_ENABLED` flip** — the crypto-side Chronos/anomaly/classifier
  stack stays dormant until the operator flips `ML_ENABLED=true`; the rec
  engine's forecaster gets its own visible `REC_FORECASTER_ENABLED` flag so
  the daily pick doesn't wait on (or silently bypass) that decision.
- [ ] **Zoya production key** (inherited from `docs/ROADMAP.md`) — universe
  breadth caps cross-sectional training, factor ranks, and Phase 4's
  per-symbol statistics (the live cycle sees ~3 sandbox symbols today).

## Sequencing note

Phases 0→1 are pure-numpy + measurement and can proceed immediately in
autonomous slices — but inside Phase 0, the mislabeling fix and audit come
first (the baseline everything calibrates against is currently suspect), and
inside Phase 1 the touch-and-hold harness precedes any level family. Phase 2
items are independent of each other but all gated on Phase 0 metrics. Phase
3's IV-history table should ship as early as possible (history accrual is
time-gated even though its consumers aren't built). Phase 4 is deliberately
last: it touches the live path and inherits every validation gate. Within any
phase, prefer the item that unblocks measurement over the item that adds a
model.
