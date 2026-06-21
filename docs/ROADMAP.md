# halabot roadmap

Working backlog for the autonomous build loop. **Pick the next unchecked item, build a tested+committed slice, check it off.** Thesis: halal long-only strips the short leg from every factor premium, so the highest-leverage work is (1) validation rigor, (2) risk-adjusted sizing, (3) orthogonal alt-data, (4) honest outcome tracking — not more indicators.

Constraints: long-only / no-leverage / no-derivatives (halal, non-negotiable). Never change live trading/sizing paths without offline validation. Advisory + infra + validation work is preferred for autonomous slices.

## Phase 0 — Foundations & honesty (do first)
- [x] **Recommendation Scorecard + forward-return labeling service** — outcome columns on `DailyRecommendation`, leakage-safe forward-return backfill (the shared foundation), track record vs a halal benchmark (SPUS). CLI/web/scheduler/dashboard surfaces. 2026-06-20.
- [ ] Deflated & Probabilistic Sharpe gate on every backtest, walk-forward fold, and the prompt-evolution GA. **← next**
- [ ] Fix the walk-forward leakage bug (`test_start == train_end`).
- [ ] Unify the no-short invariant into one enforced gate (today two executor clamps) + halal-screening freshness/expiry gate.
- [ ] Minimum-track-record / sample-size guard before any learned stat (Kelly/calibration/IC) may act.
- [ ] Retire the dead Yahoo options-IV feed (permanent 401s).

## Phase 1 — Close the risk/sizing loop (offline-validate before any live wiring)
- [ ] Wire the confidence calibrator into live position sizing (all 3 critics' #1).
- [ ] Half-Kelly sizing from per-bucket win-rate/payoff, hard-clamped at f*≥0.
- [ ] CPPI-style continuous drawdown throttle on the sizing path.
- [ ] CVaR (Expected Shortfall) tail-risk budget gate + prompt surface.
- [ ] Persisted realized-vol / covariance estimator across cycles + turnover/churn budget.
- [ ] In-cycle daily-loss-halt enforced before order submission.

## Phase 2 — Real orthogonal edge
- [ ] Cross-sectional factor core (momentum + low-vol + trend-quality), one rank module / two consumers (live tilt + advisory basket).
- [ ] Rank-IC / ICIR keep-kill signal-evaluation harness (alpha-decay pruning).
- [ ] FinBERT/FinGPT transformer sentiment behind the `HeadlineClassifier` Protocol (gives stocks sentiment).
- [ ] SEC EDGAR Form 4 insider-buy clustering catalyst (buy-side only).
- [ ] OSS sentence-transformer embeddings to replace the hashing embedder behind the `Embedder` Protocol.

## Phase 3 — Validation & research infrastructure
- [ ] Purged + embargoed cross-validation in the ML retrainer.
- [ ] PBO via combinatorial-purged CV as a research-job promotion gate.
- [ ] Frozen-shadow paper replica wired to the live AlertSink (live-decay sensor).
- [ ] Performance attribution by regime / signal-source / conviction.
- [ ] Realistic stock transaction-cost & slippage sweep in backtests.

## Phase 4 — Product / UX depth (parallelizable)
- [ ] Per-symbol research page (chart + indicators + halal screen + headline sentiment).
- [ ] What-if simulator: equity curve of taking every stock-of-the-day pick.
- [ ] "What changed since yesterday" diff + computed signal-attribution badges (not LLM self-weights).
- [ ] Advisory notification digest (Telegram): daily pick + held-position level alerts.

## Cross-cutting (build once, early)
- [ ] Forward-return labeling service *(folded into the Phase-0 scorecard item)*.
- [ ] Factor-rank core (Phase 2) shared by live tilt + advisory basket.
- [ ] Prohibited-instrument refusal contract shared by every LLM surface.

## Operator dependency
- [ ] **Zoya production key + `USE_SANDBOX=false`** — gates the *real* quality of Phases 2–4 (factor/recommendation universe is sandbox-randomized today).

## Done
- [x] Daily halal recommendation feature (engine + DB + CLI + web API + dashboard + 09:05 ET job). 2026-06-20.
- [x] Flaky anomaly-state persistence test fixed (awaitable save). 2026-06-20.
