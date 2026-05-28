# Phase 0a — Carve Dependency Audit

> Evidence-backed map of what can move to `halabot-platform` without breaking the
> live trading bot. De-risks REARCHITECTURE finding **R-01**. All references are
> `file:line` against `main` at audit time (2026-05-28). Re-run before executing the
> Phase 0b carve — the import graph drifts.

## Summary

The trading path imports only **10 of `halal/`'s 58 modules** directly (+3 transitive).
The carve is feasible, but **6 blockers** must be resolved first — none fatal, all
mechanical. The single real R-01 break is `cli/dashboard.py` importing `web.app`
(lazy + `ImportError`-guarded, so imports survive; only the `dashboard` command breaks).

## `halal/` KEEP-list (stays with the trading core)

- **Screening gate:** `cache.py`, `zoya.py`, `sector_limits.py`
- **Purification ledger (compliance, non-negotiable):** `purification.py`,
  `round_trip_purification.py`, `purification_schedule.py`, `time_weighted_purification.py`
- **Trading-path / core deps:** `zakat.py`, `jurisdiction_router.py`,
  `aaoifi_standard_17.py`, `audit.py`, `signing.py` (transitive via `audit`),
  `explainer.py`, `exception_queue.py`

Evidence for the load-bearing imports:

| Module | Imported by |
|---|---|
| `cache.py`, `zoya.py` | `trading/scheduler.py:17-18` |
| `sector_limits.py` | `trading/executor.py:1302`, `trading/strategy.py:356` |
| `round_trip_purification.py` | `core/post_close.py:192`, `crypto/components.py:240` |
| `zakat.py` | `core/tax_gcc_zakat.py:25` |
| `jurisdiction_router.py` (`Jurisdiction` enum) | `core/yearend_checklist.py:26` |
| `aaoifi_standard_17.py` | `markets/sukuk_*`, `ml/halal_optimizer.py:44` |
| `audit.py`, `explainer.py`, `exception_queue.py` | `cli/halal.py`, `cli/insights.py` |

## `halal/` MOVE-list (to `halabot-platform`)

All scholar/consensus/jurisdiction-platform and instrument-structuring modules with
**no trading-path importer** (verified empty): `gate`, `continuous_screen`,
`bridge_screen`, `riba_detector`, `gharar_detector`, `maysir_screen`, `corroborate`,
`startup_db`, `regulator_index`, `commodities_screener`, `sukuk_screener`,
`reit_screener`, `aaoifi_summary`, `aaoifi_seed`, `aaoifi_standard_21/30`, `consensus`,
`multi_school_consensus`, `regional_scholars`, `scholar_profiles/review/calendar`,
`certification_readiness`, `compliance_attestation`, `ssb_governance`, `nft_screen`,
`audit_nft`, `spac_screen`, `esg_alignment`, `climate_finance`, `charity_receipts`,
`disbursement_reconciler`, `zakat_receipt`, and the Islamic-finance instrument set
(`arboun`, `mudarabah*`, `musharakah*`, `wakalah*`, `salam_*`, `sukuk_onchain`,
`waad`, `halal_put`, `fx_hedge`, `duration_hedge`, `hedge_basket`,
`portfolio_insurance`, `pls_strategy`, `lp_gp_fund`, `convertible_musharakah`,
`stablecoin_gateway`, `vault_custodian`, …).

Whole packages that move cleanly (no trading-path import): `marketplace/`,
`community/`, `education/`, `i18n/`, and the non-operational `web/` routes.

## Blockers (resolve before Phase 0b)

| # | Blocker | Resolution |
|---|---|---|
| **B-1** | `cli/dashboard.py:19` imports `web.app` (the R-01 break). Lazy + `ImportError`-guarded so imports survive, but `halal-trader dashboard` breaks at runtime. | Move the `dashboard` command to the platform repo, or let the existing `except ImportError` degrade it gracefully. |
| **B-2** | `jurisdiction_router.py` is a move-candidate by theme but `core/yearend_checklist.py:26` imports its `Jurisdiction` enum. | KEEP the module (recommended — self-contained enum), or extract `Jurisdiction` to `domain/`. |
| **B-3** | `aaoifi_standard_17.py` is dual-use: KEEP for `markets/`+`ml/`, but platform `sukuk_onchain`/`hybrid_optimizer` also import it. | Publish it as a shared module both repos depend on (not exclusively owned). |
| **B-4** | `web/routes/insights.py:17,44` imports `halal_trader.cli.insights` — moved web depends on kept `cli/`. | Platform repo declares the trading-core package as a dependency, or that route loses function. |
| **B-5** | Shared DB/Alembic: `web` reads/writes the same `db/models.py` tables (`Trade`, `CryptoTrade`, `LlmDecision`, `IndicatorSnapshot`, `HalalScreening`, `RoundTripPurificationRow`, `ShariaExceptionRow`). | Trading-core owns `db/` + Alembic head; platform repo consumes it as a dependency (REARCHITECTURE Part IV shared-DB rule). |
| **B-6** | ~14 of 86 candidate test files also import trading-path code / bind the Postgres conftest fixtures (`database_url`, `engine`) — can't lift-and-shift. | Split each mixed test, or have the platform repo depend on trading-core + replicate the `:5433` conftest harness. Mixed files: `test_web_app`, `test_web_audit`, `test_web_insights`, `test_admin_trades`, `test_prometheus`, `test_research_api`, `test_ws_cycle`, `test_bot_context_wiring`, … |

## Carve seam edges (the files to touch)

`cli/dashboard.py:19`, `cli/__init__.py:13,66`, `core/yearend_checklist.py:26`,
`web/routes/insights.py:17,44`, and the shared schema authority `db/models.py`.
