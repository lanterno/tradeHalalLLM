# halabot roadmap

Working backlog for the autonomous build loop. **Pick the next unchecked item, build a tested+committed slice, check it off.** Thesis: halal long-only strips the short leg from every factor premium, so the highest-leverage work is (1) validation rigor, (2) risk-adjusted sizing, (3) orthogonal alt-data, (4) honest outcome tracking — not more indicators.

Constraints: long-only / no-leverage / no-derivatives (halal, non-negotiable). Never change live trading/sizing paths without offline validation. Advisory + infra + validation work is preferred for autonomous slices.

## Phase 0 — Foundations & honesty (do first)
- [x] **Recommendation Scorecard + forward-return labeling service** — outcome columns on `DailyRecommendation`, leakage-safe forward-return backfill (the shared foundation), track record vs a halal benchmark (SPUS). CLI/web/scheduler/dashboard surfaces. 2026-06-20.
- [x] Deflated & Probabilistic Sharpe — PSR/DSR stats module + backtest `psr` + walk-forward `avg_psr` + CLI display. 2026-06-21.
- [ ] GA Sharpe gate: make prompt-evo fitness carry a per-genome return series, then gate promotion on DSR (deferred from the PSR item — needs the fitness refactor).
- [x] Fix the walk-forward leakage bug — folds now feed `warmup`(=engine window_size) context bars then the test window, so each fold is pure OOS. 2026-06-22.
- [x] Unify the no-short invariant into one enforced gate — `core/long_only.py:clamp_sell_to_long`, both executors use it (behavior-preserving). 2026-06-22.
- [ ] Halal-screening freshness/expiry gate so factor/sentiment signals never act on a stale or sandbox screen (split from the no-short item).
- [x] Minimum-track-record / sample-size guard — `core/sample_guard.py` (SampleGate + gate_stat); scorecard now reports `sufficient`/`min_samples`. 2026-06-23.
- [x] Retire the dead Yahoo options-IV feed — unwired from catalyst sources; module kept dormant. 2026-06-23.

**Phase 0 complete** except two split-out items: the GA Sharpe gate (needs the prompt-evo fitness refactor) and the halal-screening freshness gate (below). Next: screening-freshness gate, then Phase 1 (sizing loop, offline-validated).

## Phase 1 — Close the risk/sizing loop (offline-validate before any live wiring)
- [ ] Wire the confidence calibrator into live position sizing (all 3 critics' #1).
- [~] Half-Kelly sizing — pure primitive done (`core/sizing.py:half_kelly_fraction`, clamped ≥0, sample-gated). 2026-06-23. Remaining: live wiring (needs per-bucket win/payoff stats + offline backtest evidence).
- [~] CPPI drawdown throttle — primitive done + wired into the backtest as opt-in (`drawdown_throttle_budget`) for offline evidence. 2026-06-23. Remaining: live sizing-path wiring.
- [~] CVaR (Expected Shortfall) — metric done (`core/risk_metrics.py`, backtest `cvar_5pct` + walk-forward `avg_cvar_5pct` + CLI). 2026-06-23. Remaining: the live tail-risk *budget gate* that shapes sizing (deferred — touches live sizing).
- [ ] Persisted realized-vol / covariance estimator across cycles + turnover/churn budget.
- [ ] In-cycle daily-loss-halt enforced before order submission.

## Phase 2 — Real orthogonal edge
- [x] Cross-sectional factor core (momentum + low-vol + trend-quality) — `core/factors.py:rank_factors`; consumer #1 = advisory recommendation (factor leaders in the prompt + scores stored). 2026-06-25. Consumer #2 (live top-N tilt) is the deferred live-sizing work.
- [x] Rank-IC / ICIR keep-kill harness — `core/signal_eval.py` (information_coefficient + icir); first consumer = scorecard `conviction_ic`. 2026-06-25.
- [x] FinBERT transformer sentiment behind the `HeadlineClassifier` Protocol — `sentiment/finbert_classifier.py`, config-gated (`stocks.headline_classifier=finbert`, default llm). Local/free/LLM-outage-resilient. 2026-07-01.
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
- [x] What-if simulator: equity curve of taking every pick — `scorecard.whatif_equity_curve` + `/api/recommendation/whatif` + CLI line. 2026-07-01. Follow-up: dashboard chart.
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
