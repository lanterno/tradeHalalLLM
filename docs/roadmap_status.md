# Roadmap status

This is a snapshot of which roadmap waves are landed (with commit
hashes), partial, blocked, or still pending. Maintain alongside
`docs/ARCHITECTURE.md` when changes ship.

The roadmap proper lives in `~/.claude/plans/i-want-you-to-swift-pine.md`.

## Wave 1 — Close the gaps

| Item | Status | Commit / notes |
|---|---|---|
| 1.1 Microstructure → prompt | ✅ landed | pre-existing WIP (uncommitted at session start) |
| 1.2 Multi-timeframe confluence | ✅ landed | `crypto/timeframes.py` (WIP) |
| 1.3 Stocks-side parity | ⏳ partial | analytics modules built; live wiring on `trading/*` deferred (Phase 4 of finishing plan) |
| 1.4 Catalyst calendar | ✅ landed | `eed276d` (tests) + WIP module + extras (Static, Earnings, RiskPolicy, next_window) |
| 1.5 Thesis-tag P&L attribution | ✅ landed | `44ed326` (module) + post-close hook in monitor |

## Wave 2 — Decision quality

| Item | Status | Commit |
|---|---|---|
| 2.1 Ensemble judge | ✅ landed | `ed9f1ed` (module) + opt-in seam in `CryptoTradingStrategy` |
| 2.2 Platt confidence calibration | ✅ landed | `af7553b` (module + metrics); CLI joins LlmDecision deferred |
| 2.3 Counter-factual regret | ✅ landed | `2414b0d` (module) + sidecar persistence on close |
| 2.4 RAG over reasoning traces | 🟡 blocked | needs vector DB choice (pgvector vs LanceDB) and embedding model — design decision |
| 2.5 Adversarial stress harness | ✅ landed | `7652d45` |

## Wave 3 — Alpha sources

| Item | Status | Notes |
|---|---|---|
| 3.1 On-chain flows | 🔴 blocked | external API key + integration |
| 3.2 SEC EDGAR 8-K | 🔴 blocked | external API key + integration |
| 3.3 FOMC sentiment | 🔴 blocked | external API key |
| 3.4 Options IV surface | 🔴 blocked | external data feed |
| 3.5 Spot-perp basis | ✅ landed | `ad5f157` + cycle wiring via `BasisTracker` in hub |
| 3.6 Mention velocity | 🟡 partial | `7255706` (module + hub field); live wiring deferred (sentiment manager doesn't expose raw timestamped mentions yet) |

## Wave 4 — Frontier / novel

| Item | Status | Notes |
|---|---|---|
| 4.1 Prompt-evolution GA | ✅ scaffold | `3f5a6f9` — fitness function pluggable, no live runs yet |
| 4.2 LoRA self-distillation | 🔴 blocked | needs ~500 closed real trades + GPU + Sharia review of fine-tuned model use |
| 4.3 Embedding-based regime memory | ✅ landed | `fc6f0ca` + cycle wiring (daily snapshot + prompt query) |
| 4.4 Online concept-drift detection | ✅ landed | `5932bd5` + close-hook observer |
| 4.5 Adversarial co-bot | ✅ landed | `67cae10` + opt-in seam in strategy |
| 4.6 Live shadow-bot | 🟡 partial | ledger + alert wiring landed (`6d05e39`), parallel run loop deferred |

## Wave 5 — Halal differentiation + ops

| Item | Status | Notes |
|---|---|---|
| 5.1 Sukuk treasury | 🟡 partial | `ae49a0e` policy module + cycle log; broker integration deferred |
| 5.2 Automated purification | ✅ landed | `5db26c9` + close-hook recorder + sidecar ledger |
| 5.3 Sharia exception queue | 🔴 blocked | UI work; better with operator in loop |
| 5.4 Postgres migration | 🔴 blocked | schema migration touching every persistence call site |
| 5.5 OpenTelemetry tracing | ✅ landed | `081d96f` (module) + spans on LLM/cycle/orderbook/account/execute stages |
| 5.6 Bit-perfect replay harness | ✅ landed | `09fa3ba` + per-cycle snapshot in cycle |

## Cross-cutting

* **Insights hub** (`f51bcbc`) — single in-process container threaded into web `app_state["insights"]` so every analytic is reachable from the dashboard.
* **Insights web routes** (`95209bb`) — `/api/insights/{regret,thesis,drift,stress,shadow,calibration,regime,basis,treasury,purification,replay}`.
* **Insights CLI** (`94b258c` + later) — 9 subcommands surfacing the analytics at the terminal.
* **Post-close fan-out** — `core/post_close.py` wires drift / thesis / regret / purification at the monitor close site; one call, four recorders.

## Legend
✅ landed · 🟡 partial · 🔴 blocked (external dep / decision needed)
