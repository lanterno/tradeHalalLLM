# Round 4 Roadmap — "Best Halal Stock Trading App, Ever"

> **Vision**: Beyond a personal paper-trading bot. This roadmap aims at a
> production-grade, multi-user, multi-asset, multi-strategy halal trading
> platform with verifiable Shariah compliance, top-tier ML/LLM infrastructure,
> rich operator UX, and quantitative-research-firm-grade analytics. Every
> waved feature is independently shippable; the platform stays usable
> throughout.

The previous three rounds focused on a single-user paper-trading bot
with mechanical cleanup. Round 4 is the **transformation round**: from
"a halal trading bot that runs on my laptop" to "the platform every
halal trader in the world wants to use".

---

## Guiding Principles

1. **Halal-first, always.** Every new feature must preserve verifiable
   Shariah compliance. New asset classes need scholar review before
   shipping (commodities, REITs, etc.). The audit trail is non-optional.
2. **Operator empathy.** Every feature ships with an operator-facing
   surface (dashboard tile, CLI command, alert) — not just a backend
   capability.
3. **Test-driven by default.** Every wave lands with both unit + integration
   tests + a one-line acceptance test that an operator can run.
4. **Backwards-compatible by default.** Existing single-user paper mode
   keeps working at every stage; multi-user, real-money, etc. are all
   opt-in via clear feature flags.
5. **Observable everything.** Prometheus + Grafana + structured JSON logs
   exist — extend, don't bypass. Every new background task / cycle stage /
   model emits standard events.

---

## Wave 0 — Cleanup before scale-up (1-2 weeks)

Land before any new features. Pays off across every subsequent wave.

### 0.A — Drop the python-3.14 multi-except syntax  ✅ landed

`crypto/exchange.py:257` previously used the Python 3.14 tuple-form
`except A, B, C, D:` syntax. Replaced with the parenthesised form
`except (A, B, C, D):` for cross-version readability and consistency
with the rest of the codebase.

### 0.B — DRY `_wrap_existing` between `crypto/strategy.py` + `trading/strategy.py`  ✅ landed

Both modules previously had an identical private `_wrap_existing(plan)`
async helper (a passthrough that lets `run_ensemble` consume the
primary plan as a variant without re-calling the LLM). Moved to
`core/llm/ensemble.py:wrap_existing` (made public since it's a shared
API). Both call sites updated to import from the new location;
no leftover references. 5 new tests in
`tests/test_ensemble_wrap_existing.py` pin identity-preserving
passthrough across primitives, dataclasses, None, and dicts.

### 0.C — Move stocks-side ATR baseline + risk knobs to `Settings.stocks`  ✅ landed

`trading/risk.py` previously pulled `atr_baseline /
max_portfolio_heat_pct / max_drawdown_pct / high_correlation_threshold /
correlation_reduction_factor` off `settings.crypto.*` (a residual from
when stocks borrowed crypto's knobs). Round-4 wave 0.C added the same
five fields onto `StockSettings`, updated `trading/risk.py` to read
from `settings.stocks.*`, refreshed `.env.example` with the new
unprefixed env names (`MAX_PORTFOLIO_HEAT_PCT` etc.), and fixed
`tests/test_trading_risk.py` to construct `StockSettings` explicitly
(the old `Settings(crypto_max_portfolio_heat_pct=...)` kwargs were
silently swallowed by `extra="ignore"` so the test was using defaults
by accident — caught and corrected). 8 trading-risk tests + 99
settings-parity tests stay green.

### 0.D — Drop the legacy `Repository` shim once all call sites use `RepoBundle`  ✅ landed

`db/repos/__init__.py:from_repository` was a one-liner that delegated
to `from_engine(repo._engine)` — kept around during the round-2
Wave D migration. Audited the codebase: zero call sites remained
across `src/` and `tests/`. Dropped the shim + the now-unused
`Repository` TYPE_CHECKING import. Updated module docstring to
reflect that every consumer now goes through `from_engine`. 96
settings-parity + the touched test surface stay green.

### 0.E — Settings parity sweep  ✅ landed

`tests/test_settings_parity.py` is the long-standing parity
enforcer between `.env.example` and the `Settings` model. The 96
parity tests stayed green throughout Wave 0; in 0.C the new
unprefixed stocks-side risk knobs (`MAX_PORTFOLIO_HEAT_PCT` etc.)
were added to `.env.example` to match the freshly-promoted
`StockSettings` fields. No drift detected.

**Acceptance**: `just precommit` clean, no commented-out env vars,
`uv run pytest tests/test_settings_parity.py -q` green.

---

## Wave 1 — Multi-broker + multi-asset (4-6 weeks)

Right now: stocks via Alpaca paper-trade, crypto via Binance testnet.
Goal: pluggable brokers per asset class so operators can choose.

### 1.A — Broker plugin framework  ✅ landed

New package `src/halal_trader/brokers/` with two public APIs:
`register_stock_broker(name, factory)` /
`register_crypto_broker(name, factory)` and the matching
`get_*_broker_factory(name)` resolvers. Factories are
`Callable[[Settings], Broker | CryptoBroker]` — lazy, so adding a
new adapter (e.g. IBKR via `ib_insync`) doesn't pull its SDK at
package-import time. Default registrations (`alpaca` for stocks,
`binance` for crypto) preserve existing operator workflows.
`StockSettings.broker` / `CryptoSettings.broker` env-driven selectors
default to those names; an unknown name raises a precise
`BrokerNotConfiguredError` listing the registered alternatives so
operators can fix typos without grepping the source. 14 new tests
pin: case-insensitive lookup, lazy factory invocation, custom
test-stub registrations don't pollute production, register-over-existing
swap, defensive-copy snapshot of `KNOWN_*_BROKERS()`. The existing
adapter classes (`AlpacaMCPClient`, `BinanceClient`) are unchanged
— they're now wrapped by factories rather than instantiated
directly. Subsequent 1.* sub-tasks (IBKR, Tradier, Coinbase, Saxo)
will register against this framework.

### 1.A.1 — SDK-free paper-trading brokers  ✅ landed

A high-leverage detour along the way: shipped `brokers/paper.py`
with `PaperStockBroker` + `PaperCryptoBroker` — in-memory matching
engines that satisfy the `Broker` / `CryptoBroker` Protocols
*without any external SDK*. Registered under the name ``paper`` in
both registries. Use cases:

* New-user onboarding — get a bot running end-to-end before
  signing up for Alpaca / Binance.
* CI integration tests — exercise the full broker contract with
  zero network (wave 8.A's integration lane gets a free
  prerequisite).
* Demoing the dashboard to scholars / auditors / investors
  without burning real capital.
* Backtesting + walk-forward (wave 4.F) get the same broker
  contract as live trading — no shape drift.

Halal compliance still runs end-to-end: only the matching is
synthetic, so paper sessions remain proof of compliance. 35 new
tests in `tests/test_paper_brokers.py` cover the full surface
(account, positions, snapshot, bars, klines, orderbook, ticker,
buy/sell/close, slippage, weighted-avg cost basis, dust-position
cleanup, registry hookup, cross-protocol smoke). Lint + format
clean.

### 1.B — Interactive Brokers (IBKR) stocks adapter

IBKR is what most serious US retail equity traders use. Implement via
their TWS API (`ib_insync`). Required surface: account, positions,
place_order, get_clock, get_stock_snapshot, get_stock_bars,
close_position. Mirror Alpaca's shape.

### 1.C — Tradier stocks adapter

Cheaper than IBKR for low-volume traders, has a clean REST API.
Lower priority than IBKR but a valid second adapter.

### 1.D — Coinbase Advanced Trade adapter

Binance access is restricted in the US. Coinbase is the canonical
fallback. Implement the same surface as `BinanceClient` against
Coinbase's REST API. Halal-screen integration: many CB-listed coins
overlap with the existing CoinGecko categories so the screener works.

### 1.E — Saxo Bank adapter (international stocks)

Halal traders outside the US care about LSE / TSE / DIFC stocks.
Saxo is one of the few brokers that gives REST API access to
non-US markets.

### 1.B–1.E aux ✅ landed (`markets/broker_registry.py`) — capability registry

**Implementation**: `src/halal_trader/markets/broker_registry.py`
ships the pure-Python broker capability matrix that the deferred
broker SDK adapters consult once they wire up. `Broker` enum has
12 closed-set entries (paper + live for Alpaca, Binance, IBKR,
Tradier, Coinbase, Saxo). `AssetClass` (EQUITY / CRYPTO / OPTION
/ ETF / FX / COMMODITY) and `OrderType` (MARKET / LIMIT / STOP /
STOP_LIMIT / TRAILING_STOP / BRACKET / OCO) closed enums. `BrokerProfile`
carries display_name + is_paper + asset_classes + order_types +
supported_exchanges + rate_limit_per_min. Validation enforces
non-empty fields + positive rate limit + the load-bearing **MARKET
order required** invariant (a broker that can't market-order can't
execute the halt's emergency-exit flow). The 12 profiles populate
the canonical registry: Alpaca paper/live (NYSE+NASDAQ, equity+ETF,
6 order types incl. BRACKET, 200/min), Binance testnet/live (crypto
only, 4 order types incl. OCO, 1200/min), IBKR paper/live (full
asset set, 7 order types incl. OCO+BRACKET, 5 exchanges, 50/min
conservative), Tradier sandbox/live (US equity+option, 4 order
types, 120/min), Coinbase sandbox/live (crypto, 3 order types,
600/min), Saxo sim/live (full asset set, 7 international exchanges
including TADAWUL+DIFC for the OIC zone, 5 order types, 120/min).
`is_paper(broker)` is the load-bearing safety gate. `assert_can_execute`
raises `BrokerCannotExecuteError` if asset_class / order_type /
exchange not supported; the exchange check skips for crypto/FX
since those don't trade on MIC exchanges. Helper queries
brokers_supporting_{asset_class,exchange,order_type} return
deterministic-order tuples. Tests in `tests/test_broker_registry.py`
(49 cases): all 12 broker + 6 asset class + 7 order type enum
string-value pins; profile validation including MARKET-required
structural pin; registry coverage (every Broker has profile;
canonical order); paper/live partition is complete + disjoint;
is_paper accuracy across all 6 paper/live pairs; capability matrix
pins (Alpaca = equity+ETF only; Binance = crypto only; IBKR = full
asset set; Saxo = international exchanges TADAWUL+DIFC pinned;
crypto brokers = Binance+Coinbase only; option brokers = IBKR/
Tradier/Saxo not Alpaca; FX = IBKR+Saxo only; TADAWUL = Saxo only;
NYSE = Alpaca/IBKR/Tradier not Binance); order type pins (OCO =
Binance+IBKR not Alpaca; BRACKET = Alpaca+IBKR not Binance);
assert_can_execute happy path + asset-class / order-type / exchange
mismatch rejections; crypto skips exchange check; optional exchange
arg; error carries broker + reason; render output with paper marker
📝/💰 + asset list + order list + rate limit + no-secret regression
across all 12 brokers; e2e flows (US trader picks Alpaca; Saudi
trader picks Saxo for TADAWUL; crypto trader picks Binance/Coinbase;
paper-safety-gate pin across all 6 pairs).

### 1.F — Multi-broker portfolio aggregator  ✅ landed

Shipped `brokers/aggregator.py:PortfolioAggregator` — read-only
fan-out over a set of stock + crypto brokers. Single
`snapshot()` call returns:

* `total_equity_usd`, `stocks_equity_usd`, `crypto_equity_usd`
* `stock_positions` (unified across all stock brokers — useful
  during a staged Alpaca → IBKR migration)
* `crypto_balances_by_asset` (same asset on Binance + Coinbase
  sums correctly)
* `per_broker_equity` (attribution)
* `per_broker_health` list (per-broker `available=True/False` +
  error message)

**Failure-isolation guarantee**: a per-broker timeout / API
failure does NOT abort the snapshot. The operator sees what they
have AND what's broken, with explicit `has_failures` +
`healthy_broker_names` accessors. 5-second per-broker timeout by
default (matches the reconcile loop). 15 new tests in
`tests/test_portfolio_aggregator.py` cover: empty / single-broker /
multi-broker happy paths, position unification across stock
brokers, balance summation across crypto brokers, isolated stock /
crypto / timeout failures, all-broken-graceful-degradation, name
override on re-add, frozen-dataclass invariants. mypy strict
passes on the package.

The aggregator builds on the existing wave 1.A.1 paper brokers,
so the test suite exercises a real broker contract without any
SDK or network. Re-exported from `halal_trader.brokers`
(`AggregatedPortfolio`, `BrokerHealth`, `PortfolioAggregator`).
The dashboard's risk-state push (round-3 Wave 1.3) and the mobile
summary will wire this in next iteration.

### 1.G — Halal commodities (gold/silver via ETF or spot) ✅ landed (screener)

Strict Shariah requires physical possession for spot commodities,
which is complex. The interim: gold/silver-backed ETFs (PHYS, SLV
caveats notwithstanding — needs scholar review). Add as a third
asset class with its own `commodities/` module.

**Landed (screener)** as `halal/commodities_screener.py` — pure-
Python ETF-specific Shariah screen for commodity vehicles. Picked
a focused module rather than wedging the rules into the general
stock screener (Zoya / Wahed) because the failure modes are
commodity-specific: the question isn't "does this ETF's company
have ≤33% debt", it's "is the underlying gold actually allocated
bullion in the unitholder's name or a synthetic swap pretending
to be gold". Eight `CommodityType`s (gold / silver / platinum /
palladium / copper / oil / natural_gas / agricultural) split into
three classes: ribawi (gold + silver — AAOIFI Standard 57 +
classical fiqh — require allocated-physical + segregated
storage); debated (oil / gas / agricultural — sector-scholar
disagreement, DOUBTFUL even with clean backing); industrial
non-ribawi (platinum / palladium / copper — clean backing →
HALAL). `BackingMode` enum names the five financing structures:
ALLOCATED_PHYSICAL (strongest — unitholder claim on specific bars
in named account), UNALLOCATED_PHYSICAL (pooled vault), FUTURES_
BACKED (rolled futures, gharar concerns), SWAP_BACKED (synthetic
derivative, unconditional NOT_HALAL), UNKNOWN (filings didn't
disclose, INSUFFICIENT_DATA). `StorageLocation` enum (SEGREGATED /
COMMINGLED / PAPER) layers on top — PAPER is unconditional
NOT_HALAL. `CommodityETFFinancials` carries symbol / name /
commodity / backing_mode / storage_location / leverage_factor /
physical_holdings_pct / has_audited_holdings. `screen_commodity_etf
(financials, *, thresholds)` returns a `CommodityScreenResult`
with verdict, contributing numbers, and per-rule failure / warning
lists. **Pinned semantics**: (1) HALAL requires every check pass —
allocated-physical + segregated + ≥95% physical (default) +
leverage=1.0 + audited holdings; any single failure flips to
NOT_HALAL or DOUBTFUL, never silent HALAL; (2) NOT_HALAL is
unconditional for SWAP_BACKED, PAPER storage, leverage > 1.0,
inverse leverage (negative factor — short ETFs require derivatives
+ interest-bearing financing), and physical holdings below
threshold; (3) DOUBTFUL is the operator-decides bucket — un
allocated-but-physical, debated commodity types, missing audit;
(4) INSUFFICIENT_DATA when backing UNKNOWN — filings undisclosed,
never silent HALAL because nothing rejected; (5) float boundaries
inclusive (95% physical → HALAL, 94.99% → NOT_HALAL; 1.0× leverage
→ HALAL, 1.01× → NOT_HALAL — both directions tested). Ribawi
commodities take additional rules: gold / silver with unallocated
backing trip a ribawi-specific warning even when clean otherwise;
gold / silver with commingled storage trip a ribawi-storage
warning. Threshold customisation: stricter operators bump
`min_physical_holdings_pct` to 98%; the customisation flips
borderline verdicts. Halal alignment: read-only screening; never
opens a position; mirrors the conservative-default pattern of
Wave 1.I REIT screen + Wave 2.G regulator index (DOUBTFUL on
missing data, never silent HALAL); pure-Python (`dataclasses` +
`enum`); no DB / network / async; frozen dataclasses on every
output. 53 tests cover threshold validation (zero / negative /
above-100 rejections), CommodityETFFinancials field validation
(empty symbol / name; physical_holdings_pct out of [0,100]
rejected), every hard-rejection path (SWAP_BACKED → NOT_HALAL with
specific message; PAPER storage → NOT_HALAL; leverage > 1.0 →
NOT_HALAL; inverse / short → NOT_HALAL with "inverse" / "short"
message; physical < threshold → NOT_HALAL), threshold inclusivity
in BOTH directions (95% physical → HALAL, 94.99% fails; 1.0×
leverage → HALAL, 1.01× fails — the symmetry pin matters),
INSUFFICIENT_DATA on UNKNOWN backing (even with all other flags
clean), HALAL happy paths (PHYS-shaped clean gold; PSLV silver;
PPLT platinum), DOUBTFUL paths (unallocated gold; unallocated
copper; futures-backed-with-segregated; commingled storage; no
audited holdings), the ribawi-stricter-rules pin (gold +
unallocated + segregated → DOUBTFUL with ribawi-allocated
warning; silver + allocated + commingled → DOUBTFUL with ribawi-
segregated warning; non-ribawi copper + allocated + commingled →
DOUBTFUL with no ribawi-specific warning), the debated-commodities
pin (oil / gas / agricultural with clean allocated-segregated
still → DOUBTFUL; platinum / palladium → HALAL because not in the
debated set), multiple-failure aggregation (a SWAP + PAPER + 2× +
0% case lists ≥3 distinct failures), threshold customisation
(stricter 98% physical flips a 96% verdict NOT_HALAL), frozen-
dataclass immutability across all four output dataclasses, all
enum string values pinned for JSON / DB stability, render output
across HALAL / NOT_HALAL / DOUBTFUL / INSUFFICIENT_DATA with the
right emoji + section headers + physical % + leverage formatting,
result-shape sanity, and two real-world end-to-end cases (GLD-
shaped unallocated-commingled gold → DOUBTFUL; UCO-shaped 2× swap-
financed oil → NOT_HALAL with multiple distinct failures).
Cycle-side wiring (delegate from `halal/screener.py` for symbols
classified as commodity ETFs, surface verdict in dashboard's halal-
status tile, route DOUBTFUL into the scholar-review queue from
Wave 2.F, plumb a CSV / Yahoo-Finance fundamentals fetcher for the
backing_mode / storage_location / leverage / physical_holdings_pct
inputs) deferred to follow-ups — screener verified in isolation
first.

### 1.H — Halal sukuk (Islamic bonds) integration ✅ landed (screener)

Sukuk are the halal alternative to bonds. Major issuers: GCC sovereigns,
Indonesia, Malaysia. Integrate via Bloomberg or Refinitiv API (will
need a paid feed). Defer to Wave 1.J because of vendor cost.

**Landed (screener)** as `halal/sukuk_screener.py` — pure-Python
screening logic for sukuk issues (the live ingest path / paid feed
remains a separate follow-up). Picked a focused module rather than
the general stock screener because the failure modes are sukuk-
specific: not "does the issuer have ≤33% conventional debt", but
"is this specific issue structured under one of the AAOIFI-approved
contracts, backed by halal underlying assets, with a profit-share
or rental coupon (not interest), and certified by a recognised
shariah board". Eight `SukukStructure`s split into broadly-
accepted (IJARAH / MUSHARAKA / MUDARABA / WAKALA / SALAM / ISTISNA
/ HYBRID) vs contested-on-tradability (MURABAHA — debt-like once
issued; some scholars dispute). `AssetClass` enum: seven halal
underlyings (REAL_ESTATE / INFRASTRUCTURE / EQUIPMENT / AIRCRAFT /
VEHICLES / POWER_PLANTS / UTILITIES); two unconditional rejects
(FINANCIAL_RECEIVABLES — conventional debt portfolio dressed as
sukuk; PROHIBITED_OPERATIONS — alcohol / gambling / banking-ops /
pork / arms); UNKNOWN → INSUFFICIENT_DATA. `IssuerType` enum:
SOVEREIGN + SUPRANATIONAL + CORPORATE_HALAL pass clean;
CORPORATE_MIXED + CONVENTIONAL_BANK trip DOUBTFUL warnings.
`CouponType` enum: PROFIT_SHARE + RENTAL_INCOME + FIXED_PROFIT_RATE
all pass (the FIXED_PROFIT_RATE pin is load-bearing — AAOIFI
permits a contractually-fixed ijarah rental that *uses* a
benchmark like LIBOR for transparency without making the rental
itself an interest payment); INTEREST is the riba red line and
categorically NOT_HALAL. `SukukIssue` carries isin / issuer_name /
issuer_type / structure / asset_class / coupon_type /
is_aaoifi_certified / is_asset_backed / maturity_days /
expected_yield_pct. `screen_sukuk(issue, *, thresholds)` returns
a `SukukScreenResult` with the verdict, contributing flags, and
per-rule failure / warning lists. **Pinned semantics**: (1) HALAL
requires every check pass — AAOIFI-approved structure + AAOIFI-
certified + halal asset class + non-interest coupon + permissible
issuer + asset-backed (preferred); any single failure flips
verdict; (2) NOT_HALAL is unconditional for INTEREST coupon /
PROHIBITED_OPERATIONS / FINANCIAL_RECEIVABLES — even if everything
else is clean (the pin is regression-tested with the "INTEREST
overrides every other clean flag" case); (3) DOUBTFUL covers
not-AAOIFI-certified, MURABAHA under strict mode, conventional-
bank issuer, asset-based under strict mode, very short tenor
(money-market-shaped); (4) INSUFFICIENT_DATA when asset_class is
UNKNOWN — never silent HALAL just because nothing rejected;
mirrors Wave 1.G commodities + 1.I REIT + 2.G regulator-index
patterns; (5) FIXED_PROFIT_RATE is HALAL by default — guards
against a sloppy implementation conflating "uses LIBOR" with "is
interest" and rejecting every modern sukuk. Threshold customisation:
`require_aaoifi_certified` (default True; sovereign-only operators
can relax), `accept_murabaha` (default True; strict-Hanafi
operators set False), `accept_asset_based` (default True; most
modern GCC sukuk are asset-based not asset-backed; strict operators
set False to require true-sale), `min_maturity_days` (default 1d;
filters out money-market-shaped issues). Halal alignment: read-
only screening; never opens a position; pure-Python (`dataclasses`
+ `enum`); no DB / network / async; frozen dataclasses on every
output. 50 tests cover threshold validation (negative min_maturity
rejected; zero accepted as "any tenor"), SukukIssue field
validation (empty isin / issuer_name; negative maturity / yield),
every hard-rejection path (interest coupon → NOT_HALAL with riba
message; prohibited operations → NOT_HALAL with "prohibited"
message; financial-receivables → NOT_HALAL with "conventional
debt" message; interest overrides everything-else-clean), the
INSUFFICIENT_DATA pin (UNKNOWN asset class alone; UNKNOWN even
with all other flags clean), HALAL happy paths (clean ijarah-
sovereign-real-estate-rental; clean musharaka-supranational-
infrastructure-profit-share; clean wakala-corporate-halal;
fixed_profit_rate clean → HALAL — the LIBOR-isn't-interest pin;
istisna-aircraft; salam-utilities; hybrid; every halal asset
class passes with clean flags), DOUBTFUL paths (not-AAOIFI-
certified; conventional-bank issuer; corporate-mixed; short
maturity; murabaha + strict mode; asset-based + strict mode),
the murabaha-default-mode-passes pin and asset-based-default-
mode-passes pin, AAOIFI-uncertified-with-override → HALAL,
multiple-warning aggregation (≥4 distinct warnings on a
DOUBTFUL aggregate), multiple-failure aggregation on hard
rejections, threshold customisation flow (strict thresholds flip
marginal verdicts), frozen-dataclass immutability, all enum
string values pinned for JSON / DB stability, render output
across all four verdicts with structure / asset_class / coupon /
maturity / aaoifi-certified / asset-backed sections, and three
real-world end-to-end cases (Saudi sovereign 5y ijarah → HALAL;
Dubai-Islamic-Bank-shaped wakala → DOUBTFUL because of
CONVENTIONAL_BANK issuer registration; synthetic credit-linked
note over conventional loans → NOT_HALAL because of
financial-receivables underlying). Live ingest (Bloomberg /
Refinitiv vendor feed adapter behind the existing screener
interface), CSV-import path for operators with their own data
sources, and the cycle-side wiring to surface the verdict in the
dashboard's halal-status tile + route DOUBTFUL into the scholar-
review queue from Wave 2.F deferred to follow-ups.

### 1.I — Halal REIT screening ✅ landed (screener core)

Real estate is generally halal but REITs that lever up using interest
debt are not. Need a sector-specific compliance screen (debt ratio,
business activities). Wire into the existing screener interface.

**Landed (screener core)** as `halal/reit_screener.py` — a pure-
Python REIT-specific Shariah screen. The general-purpose Zoya /
sector-limit screener catches the broad-strokes failures (alcohol,
gambling, pork) but slips on REITs whose own SIC code is "real
estate" while a mall-anchor tenant is a conventional bank or wine
retailer; this layer applies the AAOIFI Standard 21 + Mufti Faraz
Adam REIT framework: debt-to-marketcap ≤ 33%, NPI from forbidden-
tenant rents ≤ 5%, property type cannot be inherently non-
permissible. `screen_reit(financials, *, thresholds)` returns a
`REITScreenResult` with status (HALAL / NOT_HALAL / DOUBTFUL /
INSUFFICIENT_DATA), debt %, NPI %, liquid-assets %, purification %
(equals NPI when HALAL), and per-rule failure / warning lists for
the audit trail. Three property-type classes pinned: **inherently
halal** (RESIDENTIAL / OFFICE / INDUSTRIAL / HEALTHCARE / DATA_CENTER
/ SELF_STORAGE) pass without tenant detail; **tenant-dependent**
(RETAIL_MALL / DIVERSIFIED) need the breakdown — absent it, DOUBTFUL
not HALAL; **inherently doubtful** (HOTEL / SPECIALTY) need scholar
review even with clean tenant data because the underlying
hospitality/specialty activities aren't tenant-list-captured.
Pinned semantics: HALAL requires both checks pass *and* data
present (missing tenant breakdown on retail mall → DOUBTFUL not
silent HALAL); zero market cap → INSUFFICIENT_DATA (can't compute
ratios); float-comparison boundaries inclusive (33.0% debt is HALAL,
33.0001% is NOT_HALAL — both directions tested); tenant-pct sum
above 100% rejected at construction (data-entry guard); below-100%
sum accepted (implicit halal-other-tenants, the documented case
when only suspicious tenants are listed); purification = NPI when
HALAL with marginal NPI so the operator gets the actionable
dividend-purification fraction directly. Thresholds configurable
via `REITThresholds` (a 30% / 3% strict profile flips marginal
verdicts the other way — pinned in tests). Halal alignment: the
screener mirrors the conservative-tiebreak philosophy of Wave 2.B
halal consensus (DOUBTFUL on missing data, never silent HALAL); no
DB / network / async; frozen dataclasses on every output. 56 tests
cover threshold validation (zero / negative / >100 rejections),
TenantContribution + REITFinancials field validation (every
non-negative invariant fires; tenant-pct sum > 100% rejected),
every property-type class outcome (inherently halal passes, tenant-
dependent without breakdown → DOUBTFUL, inherently doubtful
warning), debt threshold inclusive boundary (33.0% pass, 33.0001%
fail), liquid-assets > 70% mis-classification rejection, NPI
boundary inclusive (5% pass with purification, 8% fail), marginal
NPI HALAL+purification path (purification_pct equals npi_pct),
multiple-failure aggregation (3+ distinct rule failures listed),
threshold customisation (strict 30% / 3% flips marginal verdicts),
frozen-dataclass immutability across all four output dataclasses,
status / property-type string-value pins for JSON serialisation,
render output with emoji per status (✅/❌/⚠️/❓), and an
end-to-end real-world Simon-property-trust-shaped retail mall with
mixed tenants summing to 8% NPI → NOT_HALAL. Wiring into the
existing screener (delegate from `halal/screener.py` when the
candidate symbol's SIC code or property classification flags a
REIT) deferred to a follow-up — the screener verified in isolation
first so the integration sees a stable contract.

### 1.J — International equities (LSE, TSE, DIFC) via Saxo ✅ landed (`markets/international_registry.py`) — registry core

Once Saxo is in (1.E), enable cross-listed halal stocks: BABA on HKSE,
TCS on NSE, Aramco on TADAWUL. The strategy code is asset-class-agnostic;
mainly a screener + exchange-rules update.

**Implementation**: `src/halal_trader/markets/international_registry.py`
ships the pure-Python catalogue + cross-listing resolver that the
broker plugin (deferred until Saxo / IBKR land) consults. `Exchange`
enum with 13 ISO 10383 MIC values (NYSE/NASDAQ/OTC + LSE/TSE/HKSE/
NSE/BSE/TADAWUL/DIFC/EGX/KLSE/IDX/PSX). `Jurisdiction` enum (11
values). `HalalInfrastructure` enum (BUILT_IN / THIRD_PARTY /
OUR_SCREENER_ONLY) tagging which exchanges have native shariah
indices. `TradingHours` validated: close > open, trading_days
non-empty + within [0,6], tz_offset_minutes in [-720, 840].
`ExchangeProfile` carries display_name + jurisdiction + hours +
halal infra + settlement_days (T+N, capped at 5). Module-level
`_EXCHANGE_REGISTRY` populated at module load. `is_market_open(exchange,
*, now)` converts `now` to exchange-local time and checks day-of-week
+ open/close window (open inclusive, close exclusive — pinned both
directions). `CrossListing` carries issuer_id + exchange-specific
symbol + is_primary_listing flag. `resolve_home_market(listings, *,
issuer_id, operator_jurisdiction)` returns the preferred listing:
prefers operator_jurisdiction match (Saudi operator → TADAWUL for
Aramco), falls back to is_primary_listing, then None. Render helper
with halal-infra emoji (🕌/🔍/🛠️). Tests in
`tests/test_international_registry.py` (55 cases): all 13 MIC + 11
jurisdiction + 3 halal-infra string-value pins; TradingHours
validation including tz_offset bounds + close-before-open + empty
trading_days; ExchangeProfile validation + immutability; registry
coverage (every Exchange enum has a profile); jurisdiction pins
(NYSE/NASDAQ/OTC = US; TADAWUL = SAUDI_ARABIA + BUILT_IN; DIFC =
UAE + BUILT_IN; PSX/KLSE/IDX = BUILT_IN halal); is_market_open
across timezones (NYSE during/before/after session, weekend closed;
TADAWUL Sun-Thu pin with Friday closed; TSE during session; EGX
Friday closed; open boundary inclusive, close boundary exclusive
both directions; naive now rejected); CrossListing validation;
resolve_home_market (returns primary when no jurisdiction; returns
local when jurisdiction matches — Saudi gets TADAWUL Aramco, US gets
OTC Aramco; HK gets HKSE BABA, US gets NYSE BABA; unknown issuer
returns None; no-jurisdiction-match falls back to primary; no
primary + no match returns None); render output with correct halal
emoji per infra + settlement T+N + no-secret leak; e2e flows (Saudi
operator routing Aramco to TADAWUL with BUILT_IN; LSE-NYSE overlap
window where LSE open + NYSE not yet).

**Acceptance**: `just stocks --broker ibkr --once` runs a clean cycle
on IBKR paper. `just commodities --once` works for gold ETFs. The
dashboard shows unified-portfolio numbers across all wired brokers.

---

## Wave 2 — Verifiable halal compliance (3-4 weeks)

The current screener pulls Zoya / CoinGecko / sector-limit checks. For
"the best halal app ever" we need provable end-to-end traceability.

### 2.A — Per-trade halal receipt with cryptographic signature  ✅ landed

Shipped `halal/signing.py` with Ed25519 signing on top of the existing
`halal/audit.py:Receipt` shape:

* **Ed25519** (deterministic, 32-byte sigs, fast verify) so two
  signatures of the same payload are bit-identical → auditors can
  reproduce. Detached signature: existing JSON consumers see no
  change unless they opt in.
* **Canonical JSON** for the signed bytes — sorted keys, no
  whitespace, `default=str` for datetime / Decimal — so any
  scholar on any platform can reproduce the exact bytes.
* **Base64-URL** (no padding) for keys + signatures — pasteable
  into URLs, CLI args, email.
* **Key persistence**: `halal/signing.get_or_create_signer(data_dir)`
  generates the keypair on first call and writes
  `halal_signing.key` (PEM, 0600) + `halal_signing.pub` (PEM, 0644).
  Idempotent on subsequent calls — same keypair returned.
* **Audit module integration**: `export_receipt(...,
  sign=True, data_dir=...)` opt-in returns a `SignedReceipt`
  instead of the bare receipt; default off so existing CLI / web
  callers see no behaviour change.

27 new tests in `tests/test_halal_signing.py` pin: round-trip
sign/verify, deterministic-signature property, payload-tamper /
signature-tamper / key-swap / unknown-algorithm rejection, garbage
base64 swallowed (no crash), key persistence + 0600 perms (POSIX),
on-disk format stable across "process restart", corrupt private-key
file raises *loud* error (operator must NOT silently regenerate
and lose continuity), URL-safe encoding (no `+` / `/` / `=`),
fixed sizes (43 chars public key, 86 chars signature). mypy clean.

### 2.B — Multi-source halal screener (consensus across providers)  ✅ landed (aggregator)

Right now: Zoya only for stocks. Wire IdealRatings, Musaffa, and
Wahed Invest's screening API as alternates. Apply a consensus rule:
if any provider says NON_COMPLIANT, treat as not_halal. If
they disagree (one halal, one doubtful), default to doubtful.

**Landed (aggregator):** `halal/consensus.py` ships
`consensus(opinions, *, policy=STRICT)` →
`ConsensusDecision(decision, policy, opinions, reason, dissenters)`.
Generalises the existing two-source `halal/corroborate.py` (binary,
N=2) to N providers and the three-state `halal | doubtful |
not_halal` decision the existing `HalalScreening` audit row already
uses. Three resolution policies: `STRICT` (default — any
`not_halal` rejects, any `doubtful` without a `not_halal` yields
`doubtful`, only unanimous `halal` yields `halal`); `MAJORITY` (most
common decision wins); `WEIGHTED` (per-opinion weight sums). All
three resolve ties by **most conservative wins**
(`not_halal > doubtful > halal`) — pinned so a refactor of the rank
lookup can't silently flip the direction. Two safety semantics
baked in: empty input returns `DOUBTFUL` (no-opinions = unattested
= refuse); negative weights clamp to 0 so a buggy config can't
silently invert a contribution. `ScreeningOpinion` accepts the
decision either as the enum *or* as a string (so a JSON-loaded
audit row feeds in directly), and an unknown string raises with a
clear message rather than degrading silently. Result is a frozen
dataclass safe to cache; preserves input order; carries the
operator-readable `reason` string suitable for the dashboard
compliance tile or notifier payload. 23 tests cover every policy
branch (STRICT precedence ladder, MAJORITY tiebreak, WEIGHTED
clamp-and-sum, three-way tie → most conservative), the empty-input
contract, dissenter tracking, the string / enum coercion path
(including the typo-rejection case), the immutability invariant,
and the reason-string content. Concrete provider adapters
(IdealRatings, Musaffa, Wahed) deferred to a follow-up — aggregator
is verified in isolation first so the wiring sees a stable contract.

### 2.C — Per-scholar policy profiles  ✅ landed (registry)

Different scholars hold different positions on edge cases (e.g.
gambling-adjacent revenue thresholds, debt ratios). Let the operator
pick a policy profile (AAOIFI default, Mufti Taqi Usmani, Sheikh
Yusuf Talal DeLorenzo) that adjusts the consensus rule. Each profile
ships with a markdown doc explaining its rulings.

**Landed (registry):** `halal/scholar_profiles.py` ships three
named profiles plus the registry / lookup / serialisation
machinery. `AAOIFI_DEFAULT` (debt 30% / non-permissible income 5% /
cash-and-receivables 33%, STRICT consensus); `TAQI_USMANI`
(stricter — debt 25% / non-permissible income 3%, STRICT consensus,
weights musaffa + idealratings 1.5×); `DELORENZO_DJIM` (DJIM-era —
debt 33%, MAJORITY consensus). `ScholarProfile` bundles three
concerns: per-provider WEIGHTED-policy weights, threshold overrides
for the three financial ratios, and the default consensus policy
the operator's profile choice should pick. `evaluate_thresholds`
applies the profile's caps to a set of ratios and returns
`(passed, violations)` — missing inputs are *skipped* (treated as
"not measured" rather than "passes") so a partial filing can't
silently approve. `apply_profile_weights` re-weights opinions
using the profile's trust map without mutating the input;
`consensus_with_profile` is the one-shot convenience that picks the
profile's default policy unless the caller overrides it, and only
re-weights under WEIGHTED (avoids wasted work under STRICT /
MAJORITY which ignore weights). `get_profile(name)` is
case-insensitive and raises with the candidate list on a typo —
pin so a misconfigured `.env` surfaces immediately rather than
silently degrading. `profile_to_dict` produces an audit-trail-ready
JSON-friendly dict (policy as string not enum, weights as plain
dict). `register_profile` lets operators ship custom profiles
without forking. 26 tests cover every default value (AAOIFI
30/5/33), the relative ordering invariants (Taqi Usmani strictly
stricter than AAOIFI, DeLorenzo uses MAJORITY), the threshold
evaluator's missing-input skip semantic, the no-mutation invariant
on the re-weight helper, the one-shot helper's policy override and
re-weight-only-when-WEIGHTED behaviour, registry case-insensitivity
+ typo rejection + dynamic registration, and the JSON-friendly
serialisation shape. CLI flag (`halal-trader --scholar=…`) and
audit-row wiring deferred to a follow-up — registry verified in
isolation first.

### 2.D — Quarterly purification ledger automation  ✅ landed (scheduler + receipts)

The existing `halal/round_trip_purification.py` tracks per-trade
purification. Round it out: monthly + quarterly auto-disbursement
emails to a configured charity address, with a signed PDF receipt
attached. Integrate with major Islamic charities (Islamic Relief,
Penny Appeal, Zakat Foundation) for direct disbursement.

**Landed (scheduler + receipts):**
`halal/purification_schedule.py` ships
`schedule_disbursements(entries, *, period, include_paid, charity)`
→ `list[DisbursementReceipt]`. Three cadences: `MONTHLY`
(`"2026-03"`), `QUARTERLY` (`"2026-Q1"` — default; matches AAOIFI
guidance for periodic settlement), `YEARLY` (`"2026"`). Each
receipt carries: label slug (audit-trail key + future PDF
filename), UTC half-open `[starts_at, ends_at)` bounds, total USD,
entry count, per-symbol `SymbolBreakdown` sorted by descending USD
(operator's eye lands on concentration first), the immutable list
of underlying entries, and a pre-rendered markdown body suitable
for emailing the operator. `upcoming_due(entries, *, now, period)`
returns the receipt for the *current* period — drives a dashboard
tile that says "this quarter so far you owe $X across N holdings".
Three semantic pins: `include_paid` defaults to **False** (operator
wants to see what's still due, not what's settled); naive
`received_at` timestamps are coerced to UTC rather than crashing
(legacy ledger rows might be tz-naive); the scheduler **never
auto-marks entries paid** — that's a one-way audit-trail commitment
that needs explicit human consent, and the markdown receipt
explicitly tells the operator to mark them in the ledger after
disbursement. 35 tests cover: every cadence's slug format, the
`(month-1)//3 + 1` quarter math, multi-period bucketing,
chronological ordering of receipts, naive-tz tolerance, the paid /
outstanding filter contract (default + opt-in), USD totals,
per-symbol aggregation, the descending-USD breakdown sort, the
half-open period bounds (April 1 → July 1 for Q2; Dec 1 →
next-year Jan 1; Q4 wraps the year), the no-mutation invariant on
the ledger, the markdown body's label / total / charity-line /
breakdown-table / audit-reminder content, the `is_empty` property
on the receipt, the `upcoming_due` returning None outside the
active period and the current period's receipt inside it, the
default-`now` fallback, the immutability invariant, and the
unknown-period rejection. PDF rendering, email pipeline, and
charity-API integration deferred to a follow-up — scheduler +
receipts verified in isolation first so the email layer sees a
stable contract.

### 2.F + 11.B aux — Scholar consultation calendar ✅ landed (`halal/scholar_calendar.py`)

**Implementation**: `src/halal_trader/halal/scholar_calendar.py`
is the cadence-meeting complement to Wave 2.F scholar review +
Wave 11.B SSB governance. Wave 2.F handles ad-hoc verdicts; Wave
11.B tracks the structured board meetings; this module schedules
the regular cadence: annual halal compliance audit, quarterly
portfolio review, ad-hoc consultations. `ConsultationKind` enum
(ANNUAL_AUDIT / QUARTERLY_REVIEW / AD_HOC). `ConsultationStatus`
enum (SCHEDULED → CONFIRMED → COMPLETED + terminal CANCELLED).
`CalendarPolicy` ships **30d / 7d / 1d reminder ladder** + **60d
annual confirmation lead time** defaults; validation enforces
descending lead-time order + positive values. `Consultation`
carries id + kind + scholar_handle + scheduled_at + status +
topic + optional minutes_url. **COMPLETED status requires
minutes_url** (audit trail) + non-COMPLETED must NOT have it
(structural pin). Lifecycle: `schedule_consultation` rejects past
dates + naive datetimes; `confirm_consultation`,
`complete_consultation` (requires minutes_url), `cancel_consultation`
(any pre-COMPLETED state, terminal). `is_due_for_reminder` walks
the threshold ladder: returns True if any threshold has been
crossed AND last_reminder_at predates that threshold; only
SCHEDULED + CONFIRMED states fire reminders.
`annual_audit_overdue_for_confirmation` flags ANNUAL_AUDIT still
SCHEDULED within 60d of scheduled date — boundary inclusive at
60d. `filter_due_for_reminder` + `upcoming(horizon=90d)` for the
dashboard. Render with kind emoji 📋/📅/💬 + status emoji
🗓️/✅/📝/❌. **No-secret pin**: dataclass deliberately doesn't
carry scholar contact emails or meeting URLs (operator-side
state). Tests in `tests/test_scholar_calendar.py` (60 cases): all
enum string-value pins; **30d/7d/1d ladder + 60d annual default
pins**; **descending-order pin** (7d before 30d rejected); zero-
lead rejected; Consultation validation including **COMPLETED-
requires-minutes-url + non-COMPLETED-must-not-have-it both
directions pinned**; immutable; schedule rejects past + now-equal
dates; lifecycle SCHEDULED → CONFIRMED → COMPLETED happy path;
cancel from any pre-COMPLETED; **revert from COMPLETED rejected
pin**; reminder ladder pins (due at 30d threshold; not due at 31d;
due at 7d if 30d already fired; **not-due if last_reminder AT
active threshold**; due at 1d boundary); COMPLETED + CANCELLED
never fire reminders; **annual-audit overdue at 60d boundary
inclusive pin**; only ANNUAL_AUDIT triggers the check; CONFIRMED
audits not overdue; filter_due returns sorted by scheduled_at
ascending; **last_reminders dict suppresses already-fired
reminders pin**; upcoming within 90d horizon excludes COMPLETED +
CANCELLED + sorted ascending; custom horizon; render with kind +
status emojis + minutes when COMPLETED + omitted otherwise + no-
secret-leak pin (no `@` no `zoom.us` no `meet.google`); e2e flows
(annual audit full lifecycle Q1 schedule → 30d-out confirm →
day-of complete with minutes; **overdue annual-audit at 50d
caught pin**; reminder ladder fires at 30d then again at 7d
threshold; replay consistency).

### 2.D aux — Charity disbursement reconciler ✅ landed (`halal/disbursement_reconciler.py`)

**Implementation**: `src/halal_trader/halal/disbursement_reconciler.py`
is the paid-side complement to Wave 2.D's owed-side scheduler.
Wave 2.D ships the "you owe $X this quarter" generator; this
module ships the "did you actually pay it?" auditor.
`ReconciliationStatus` enum (UNRECONCILED / UNDERPAID / RECONCILED
/ OVERPAID). `ReconcilerPolicy` defaults: $0.01 cents tolerance +
90-day overdue threshold; rejects negative tolerance + tolerance
above $1 (anything more permissive than $1 means we'd allow
underpayment of more than a dollar — too lax for a halal-compliance
audit). `OwedPeriod` mirrors Wave 2.D receipt shape with minimum
fields. `DisbursementReceipt` carries the bank-wire confirmation;
`wire_reference` field is operator audit metadata kept out of
render (no-secret pin). `reconcile_period(period, receipts, *, now,
policy)` filters receipts by period_label, sums paid amount,
computes shortfall, classifies status, flags overdue.
`reconcile_all` returns sorted-by-start results. `total_outstanding`
sums positive shortfalls only (overpayments NOT netted —
load-bearing audit pin: a $50 overpayment in Q1 doesn't reduce a
$200 underpayment in Q2 without explicit operator decision).
`overdue_periods` filters the audit-tile candidates. Render with
status emoji (❓⚠️✅💚) + OVERDUE marker; no-secret regression.
Tests in `tests/test_disbursement_reconciler.py` (49 cases):
ReconciliationStatus enum string-value pin; policy validation
including $1-tolerance-cap pin; OwedPeriod + DisbursementReceipt
field validation; **status classification both directions** (no
receipts → UNRECONCILED; exact match → RECONCILED; within $0.005
cents tolerance → RECONCILED; $1.50 short → UNDERPAID; over →
OVERPAID); multiple receipts sum together; receipts filter by
period_label; **overdue boundary at 90 days inclusive** (90d past
end → overdue, 89d → not overdue); reconciled never overdue;
overpaid never overdue; custom 30-day strict overdue threshold;
reconcile_all sorted by start; **total_outstanding excludes
overpayments load-bearing pin** (no netting); overdue_periods
filter; render with period_label + status emoji + OVERDUE marker
+ amounts visible; render no-secret regression (no `WIRE-`,
no `CONFIDENTIAL`, no `@`, no `routing`/`iban`); render summary
counts per status + total outstanding; e2e flow (full year audit:
4 quarters Q1=reconciled / Q2=underpaid+overdue / Q3=overpaid /
Q4=unreconciled+overdue with correct status + outstanding sum);
replay consistency.

### 2.E — Live AAOIFI compliance dashboard tile  ✅ landed (backend)

Shipped `halal/aaoifi_summary.py` with the `compute_aaoifi_summary`
aggregator + `AAOIFISummary` dataclass that powers the tile.
SQL-only aggregations (no per-row Python math) so the tile stays
fast on 100k+ row audit logs. Computes:

* Trade counts (today / this month / this quarter, stocks + crypto
  combined).
* Per-decision screening counts (halal / doubtful / not_halal) for
  the quarter — the most useful audit slice.
* **Non-halal fills count** for the quarter (the red-alert metric):
  joins trades to their screening rows; any fill *without* a
  ``decision='halal'`` screening (including unattested legacy
  trades) gets surfaced rather than hidden.
* Purification (combined dividend + capital-gains sides):
  ``accrued_usd``, ``disbursed_usd``, ``outstanding_usd``.
* `status` property: ``violation`` (any non-halal fill) →
  ``attention`` (outstanding purification ≥ $0.01) → ``compliant``.
  Sub-cent residuals treated as compliant — operator's tile
  doesn't go amber over rounding noise.

New route `GET /api/halal/compliance` registered in
`web/routes/halal_compliance.py` returns the summary in a
JSON shape the dashboard renders. Wired into `register_all`.

22 in-memory helper tests in `tests/test_aaoifi_summary_helpers.py`
pin: quarter-start boundary across all four quarters with year
preservation, month-start / today-start drop-time invariants,
tz-aware UTC return values (load-bearing for SQL comparisons
against tz-aware DB columns), `is_compliant`/`status` priority
ordering (violation > attention > compliant), `outstanding`
floors at zero on over-disbursement, sub-cent rounding-noise
threshold, frozen-dataclass invariant. The DB-aggregating
`compute_aaoifi_summary` is integration-tested separately.

The dashboard frontend tile is the last 30 lines of work and
will land alongside the next `dashboard/` PR (out of scope for
this iteration's pure-Python sweep).

### 2.F — Scholar review workflow for new asset classes ✅ landed

When the bot encounters a symbol with insufficient screening data
(e.g. an IPO Zoya hasn't classified yet), it queues a "scholar review
required" item via the existing `halal.exception_queue`. Build a
scholar-facing web view that lets a configured email list approve or
reject queued items, with the verdict feeding back into the cache.

**Landed** as `halal/scholar_review.py` — a pure-Python workflow on
top of `halal/exception_queue`. `render_review_packet(...)` produces
a markdown brief (header / Why this review / Symbol context /
Financial screen inputs / Notes / References / How to respond) from
a `ReviewContext` (sector, market_cap, recent_revenue_breakdown,
debt-to-marketcap%, non-permissible-income%, notes, references) plus
the queued entry's reasoning. `ScholarVerdict` is a frozen dataclass
with `__post_init__` validation: `entry_id` non-empty, `decided_by`
non-empty (audit-trail attribution), and rationale required for
APPROVED/REJECTED (whitespace-only also rejected). DEFERRED and
WITHDRAWN allow empty rationale because they intentionally punt the
decision. `apply_verdict(...)` enforces a status guard: raises
`AlreadyDecidedError` if `pending_entry_status != "pending"` —
prevents one scholar's verdict from overriding another's already-
recorded decision. `render_recorded_verdict(...)` formats the audit
row for ops + Telegram with kind-specific emoji (✅/❌/⏳/↩️).
Pinned no-PII contract: `ReviewContext` has no operator-identifying
fields by design, and a regression test (`test_packet_does_not_leak_
operator_pii`) asserts the rendered markdown never contains
"account"/"balance"/"operator id"/etc. — scholars get the
shariah-relevant facts, not the operator's portfolio. 34 tests
covering packet rendering, verdict validation (rationale required
both directions), apply_verdict status guard, and recorded-verdict
rendering. Cycle wiring (web view + email list) deferred to a
follow-up; the module ships pure-Python so the workflow can be
exercised in a notebook today.

### 2.G — Pakistani SECP / Saudi CMA filings ingest ✅ landed (matcher)

Local-market regulators publish halal-compliance screening results
for their listed equities (Saudi: Tadawul Halal Index; Pakistan:
KMI-30 index). Wire these as authoritative sources for the
respective markets.

**Landed (matcher)** as `halal/regulator_index.py` — the pure-
Python ingestion + matching layer. `RegulatorSource` enum names
four indices the bot can ingest (TADAWUL / CMA_HALAL for Saudi;
KMI30 / SECP_HALAL for Pakistan). `Market` enum (SAUDI / PAKISTAN
/ OTHER) gates which regulators apply to which symbols — the
authority registry maps every regulator to exactly one market and
the screener silently drops cross-market rows. `IndexListing` is
the per-symbol row (symbol / verdict / listed_at / notes), validated
to require timezone-aware datetimes (load-bearing for staleness
arithmetic). `RegulatorIndex` carries the source, a fetch
timestamp, and the listings tuple; rejects duplicate symbols at
construction (case-insensitive normalisation) so a fetcher that
double-ingests a row surfaces the bug at construction. `lookup` is
case-insensitive + whitespace-stripped so Saudi 4-digit numerics
(`1010`) and Pakistani alphanumerics (`HBL`) coexist. The screener
`screen_with_regulator(*, symbol, market, indices, now, thresholds)`
returns a `RegulatorScreenResult` carrying the combined verdict,
the contributing-source list, the oldest listing age, and stale /
expired flags. Three pinned semantics: (1) **authority is per-
market** — querying TADAWUL with `market=Market.OTHER` returns
UNKNOWN with no source, *not* a silent NOT_HALAL (cross-market
authority is a category error, not a failed screen); (2) **absence
≠ NOT_HALAL** — a symbol not present in any covering regulator's
index is UNKNOWN, not NOT_HALAL (the index is a positive list,
not a negative list — mistaking absence for forbidden would
silently disqualify every newly-listed Saudi equity until the next
quarterly index update); (3) **staleness ladder**: listings within
`stale_days` (default 90) are fresh; between `stale_days` and
`expired_days` (default 365) carry a stale warning but verdict
applies; older than `expired_days` are demoted to UNKNOWN. Both
threshold boundaries inclusive (90d → stale; 365d → expired)
pinned via tests. Multiple covering indices combine via
conservative-tiebreak: NOT_HALAL > UNKNOWN > HALAL is the override
order so any single regulator saying NOT_HALAL is enough to
disqualify (mirrors Wave 2.B consensus + Wave 4.J committee).
A "one fresh + one expired listing" cohort uses the fresh
verdict (the expired source contributes UNKNOWN; the fresh source
HALAL; combined → HALAL) — pinned via test. Render helper produces
emoji-prefixed (✅/❌/❓) ops display with stale / expired markers.
Halal alignment: ingestion is read-only; never opens a position;
the matcher never silently downgrades a verdict; pure-Python
(stdlib `dataclasses` + `enum` + `datetime`); no DB / network /
async — the live fetcher (httpx client + cron + CSV import) is a
follow-up, the matcher verified in isolation first so the fetcher
sees a stable contract. 58 tests cover authority registry (every
regulator → its market mapping pinned), IndexListing + RegulatorIndex
+ RegulatorThresholds field validation (timezone-aware datetimes
required, naive rejected, duplicate-symbol rejection at construction
with case-insensitive normalisation, threshold-ladder ordering),
case-insensitive + whitespace-stripped lookup (HBL / Hbl / hbl /
"  hbl  " all match), basic outcomes (Saudi listing in TADAWUL →
HALAL; Pakistani listing in KMI30 → HALAL; NOT_HALAL flows through),
the cross-market authority guard pinned in BOTH directions (US
symbol with TADAWUL index → UNKNOWN; Pakistani symbol with only
Saudi indices → UNKNOWN), the absence ≠ NOT_HALAL pin (symbol
missing from a covering index → UNKNOWN with the right warning;
empty-indices flows to UNKNOWN), Market.OTHER → UNKNOWN with the
"no regional regulator" warning, multiple-index combination (two
HALAL → HALAL with both sources; one HALAL one NOT_HALAL → NOT_HALAL
conservative tiebreak; one HALAL one UNKNOWN → HALAL), staleness
ladder (fresh / stale / expired transitions with both 90d / 365d
boundaries inclusive), the mixed-staleness-across-sources pin (one
fresh + one expired → fresh source's verdict wins, oldest_age
reflects expired), custom thresholds flowing through (200d listing
+ strict 30/180 thresholds → expired), screener input validation
(naive `now` rejected, empty symbol rejected), convenience helpers
(`listing_age_days`, `newest_index`), frozen-dataclass immutability
across all four output dataclasses, all enum string values pinned
for JSON / DB serialisation (`tadawul` / `cma_halal` / `kmi_30` /
`secp_halal` / `saudi` / `pakistan` / `other` / `halal` /
`not_halal` / `unknown`), and render output across HALAL /
NOT_HALAL / UNKNOWN with stale + expired markers. Live fetcher
+ Wave 2.B consensus integration (high-priority source for the
fold-in) deferred to follow-ups.

**Acceptance**: Every trade row has an attached signed receipt. The
operator can run `halal-trader audit verify --since 2026-04-01` and
get a clean cryptographic proof of compliance.

---

## Wave 3 — Multi-user platform (6-8 weeks)

Right now: single-user laptop bot. Round 4 transforms it into a
hosted multi-user platform. This is the highest-impact wave.

### 3.A — User accounts + auth ✅ landed (primitives core)

Add a `User` table, OAuth2 sign-in (Google + Apple), JWT bearer
auth on the FastAPI app. Each user has isolated bots / portfolios /
purification ledgers. Existing operator-mode keeps working when
no auth is configured.

**Landed (primitives core)** as `web/auth.py` — pure-Python auth
primitives that the FastAPI route layer + OAuth callback handlers
compose with. Stdlib `hashlib.scrypt` for password hashing (RFC
7914 with n=16384, r=8, p=1; per-password 16-byte random salt);
constant-time verification via `hmac.compare_digest`; session
lifecycle with TTL bounds [5min, 24h]; login-rate-limit policy
(default 5 failures in 15 min). Five `AuthOutcome` values pinned
(SUCCESS / INVALID_CREDENTIALS / RATE_LIMITED / SESSION_EXPIRED /
SESSION_NOT_FOUND). `PasswordPolicy` defaults: ≥12 chars, digit
required, symbol required, NIST SP 800-63B floor 8 chars enforced
at construction. `PasswordHash` carries algorithm + scrypt params
(n/r/p) + salt_b64 + hash_b64. `hash_password(plaintext, *,
policy)` validates policy then produces a fresh-salt scrypt hash.
`verify_password(plaintext, hashed)` returns False on any error
(unknown algorithm, malformed salt, mismatched password) — never
raises into the auth path. `Session` carries session_id (32-byte
URL-safe base64 token, ≥16 chars enforced) + user_id +
issued_at + expires_at (timezone-aware). `issue_session(*,
user_id, now, ttl_minutes=60)` rejects out-of-bounds TTL.
`is_session_valid(session, *, now)` returns boolean — boundary
inclusive at issued_at, exclusive at expires_at. `LoginAttempt`
records per-attempt for rate-limit bookkeeping. `RateLimitPolicy`
defaults max_failures=5, window_minutes=15. `evaluate_rate_limit
(*, user_id, history, now, policy)` walks recent attempts and
returns False when N consecutive failures hit the threshold; a
successful login resets the counter. `authenticate(*, user_id,
plaintext_password, stored_hash, history, now, ...)` composes
rate-limit gate + password verify + session issue. **Five
pinned semantics**: (1) password ≥12 chars + digit + symbol;
(2) constant-time verify via `hmac.compare_digest` —
timing-attack-resistant pin; (3) session TTL bounded [5min,
24h]; (4) failed-attempt rate limiter with success-resets-counter
semantic; (5) render output never includes password hash bytes /
salt / session_id (mirrors no-secret patterns of Wave 3.B vault +
Wave 8.D OTLP + Wave 12.G co-pilot). Halal alignment: auth
primitives are read-only with respect to operator portfolio;
never opens a position; pure-Python (stdlib `hashlib` + `hmac` +
`secrets` + `base64` + `dataclasses` + `enum` + `datetime`); no
DB / network / async / HTTP. Frozen dataclasses on every output.
74 tests cover password policy validation; password hashing
(unique salt; rejects too-short / no-digit / no-symbol /
non-string); custom + relaxed policies; verification (correct;
wrong; empty; non-string; unknown algorithm; malformed salt all
return False); PasswordHash field validation; **session
lifecycle pins** (default 60min TTL; custom TTL; high-entropy ID;
unique IDs across calls; rejects empty user_id / naive now;
boundary-inclusive 5min minimum; boundary-inclusive 24h max);
Session field validation (empty / short session_id; naive
issued_at; expires-before-issued); is_session_valid (within
window True; after expiry False; **boundary-inclusive at
issued_at**; **boundary-exclusive at expires_at**); RateLimitPolicy
+ LoginAttempt validation; **rate-limit pins** (no-history
allowed; under-threshold allowed; **at-threshold blocked**;
success-resets-counter; old-failures-outside-window-don't-count;
other-users-don't-count — the per-user-isolation pin; custom
policy flow); authenticate composition (success → SESSION;
invalid creds → INVALID_CREDENTIALS; rate-limited blocks even
with correct password); frozen-dataclass immutability across
PasswordHash + Session + LoginAttempt + PasswordPolicy +
RateLimitPolicy; AuthOutcome string values pinned for JSON / DB
stability; render output (success + invalid creds + rate-limited
emoji per outcome ✅🔑🚫⏰❓); **render-no-secret pins** in three
flavours (session_id never in render; plaintext password never;
password hash bytes / salt never); session expiry rendered
(non-secret); cryptographic-determinism pins (two hashes of same
password differ; round-trip hash → verify works); end-to-end
realistic flows in three flavours (full login lifecycle 2h
later → still valid at 30min → expired at 61min; **brute-force
attack blocked** — 5 wrong guesses lock out the 6th attempt
even with correct password; **per-user isolation pin** —
attacker locked-out on user-A doesn't block user-B login).
OAuth provider integration (Google + Apple authorisation
servers, JWT issuer keys via httpx + pyjwt), Postgres `users` +
`sessions` + `login_attempts` tables wired to `db/repository.py`,
the FastAPI middleware that extracts the session token from the
bearer header + composes with Wave 3.D tenant isolation guard,
the dashboard's auth-event audit log, and the email-verification
flow deferred to follow-ups — primitives core verified in
isolation first.

### 3.B — Per-user encrypted secrets vault ✅ landed (crypto core)

Today: API keys live in the operator's `.env`. Multi-user requires a
per-user encrypted vault. Use `cryptography.fernet` with a server-side
KEK derived from `SECRET_KEY` env var; user supplies their broker keys
through a settings UI.

**Landed (crypto core)** as `web/secrets_vault.py` — pure-Python
cryptographic core of the per-user vault. A server-side master KEK
derives a per-user Fernet DEK via HKDF-SHA256 (salt=owner_id,
info=`halal-trader-vault-v{version}`); secrets ship encrypted at
rest as `EncryptedSecret` rows; decryption requires presenting the
right `owner_id` (different owner_id → different derived DEK →
Fernet `InvalidToken`); audit metadata tracks `created_at` /
`last_rotated_at` / `last_accessed_at` / `key_version` for the
rotation cadence + access trail. `SecretKind` enum names seven
storage categories (broker_api_key / broker_api_secret /
llm_api_key / news_api_key / cryptopanic_key / reddit_client_secret
/ screener_api_key) with stable string values for DB / JSON
migration safety. `SecretVault.store(...)` encrypts + returns the
row; `reveal(secret, *, owner_id)` decrypts + returns the
plaintext bytes plus an access-time-updated row the caller
persists; `rotate(...)` re-encrypts under a new master KEK +
version; `needs_rotation(secret)` flags rows past the cadence.
**Five pinned safety semantics**: (1) plaintext is never stored —
the vault holds `EncryptedSecret` rows only and the frozen-
dataclass shape means a future field addition can't leak plaintext
alongside the ciphertext; (2) wrong owner_id → InvalidToken — the
DEK derives from `(master_kek, owner_id, key_version)` so a leaked
ciphertext alone is insufficient to recover plaintext, the
cryptographic boundary is not bypassable by an application-layer
"permission denied" check; (3) tampered ciphertext → InvalidToken
— Fernet's auth-tag catches any single-byte mutation, vault re-
raises as `SecretIntegrityError` so the operator's exception
handler doesn't depend on `cryptography` package internals; (4)
master KEK ≥ 32 bytes — HKDF input minimum, shorter keys rejected
at construction so a misconfigured `SECRET_KEY` env var fails fast;
(5) render output never contains plaintext / ciphertext / any
derived key — `render_secret_metadata()` shows owner / kind /
label / dates / key_version only, audit display can't accidentally
leak the secret it's auditing. Rotation flagging is age-based on
`last_rotated_at` (not `last_accessed_at`) — operators rotate on
cadence, not usage; pinned via test that frequently-accessed-but-
never-rotated secrets still trip `needs_rotation`. KEK rotation
flow: decrypt under current KEK → re-encrypt under new KEK + new
version → return new row; if current-KEK decrypt fails (operator
already lost the current KEK), `rotate()` surfaces `Secret
IntegrityError` rather than silently re-encrypting garbage. 52
tests cover every concern: KEK length validation (short rejected,
32-byte accepted, longer accepted), `VaultPolicy` field
validation (rotation_days / max_label_length / min_plaintext_bytes
all reject zero/negative), happy path round-trip (str + bytes
plaintext), `last_accessed_at` updates on reveal but the input
row stays frozen (caller persists the new row), cross-user
isolation pins (wrong owner_id → SecretIntegrityError; identical
plaintext under different owner_ids produces different ciphertext;
forged secret claiming user-A's ownership of user-B's ciphertext →
InvalidToken at the cryptographic boundary), tamper detection
(single-bit flip rejected; truncation rejected), cross-KEK
isolation (old-KEK ciphertext under new-KEK vault → fails cleanly),
storage validation (empty owner / empty label / too-short
plaintext / too-long label / custom min_plaintext flows through),
EncryptedSecret field validation (every non-empty / non-naive-
datetime / positive-version invariant fires), KEK rotation
(re-encrypts under new KEK + version; preserves owner / kind /
label / created_at; rejects short new KEK; rejects zero version;
fails when current KEK already changed), `needs_rotation`
(fresh→False, old→True, threshold inclusive at exactly 90d, uses
last_rotated not last_accessed pin, custom rotation_days flow),
render output's no-secret-leak contract (plaintext absent;
ciphertext bytes absent in raw + hex form; owner / kind / label /
dates / key_version present; "accessed: never" for unaccessed,
formatted timestamp when accessed), frozen-dataclass immutability,
SecretKind string values pinned for DB stability, and two
cryptographic-property pins (Fernet random IV → two encrypts of
identical plaintext produce different ciphertexts; round-trip
preserves Unicode plaintext via UTF-8 encoding). Postgres
persistence (an `encrypted_secrets` table + repository), the
FastAPI route adapter (`/api/users/{user_id}/secrets`), the
operator settings UI for per-user key entry, and the cycle-side
hook (every broker / LLM client load uses the vault to decrypt
the user's keys at startup) deferred to follow-ups — vault
verified in isolation first.

### 3.C — Per-user resource quotas ✅ landed (engine)

Compute / LLM / API call quotas per user so one user can't burn the
whole instance's daily LLM budget. Tier system (free / pro / enterprise)
with per-tier knobs. Wire into `core/llm/budget.py`.

**Landed (engine)** as `web/quotas.py` — pure-Python per-user quota
accounting layer. Three-tier ladder (FREE / PRO / ENTERPRISE)
defined in `Tier`; five resource categories in `ResourceKind`
(LLM_USD / LLM_TOKENS / BROKER_API_CALLS / SCREENER_API_CALLS /
CYCLE_RUNS). `TierLimits` carries per-resource daily caps with
positive-or-zero invariants on every field — zero is valid (FREE
tier disables broker_api_calls so free-tier users only get paper
trading). `DEFAULT_TIER_LIMITS` ships the documented best-guesses:
FREE=$0.50/200k tokens/0 broker/200 screener/24 cycles, PRO=$10/2M
tokens/10k broker/2k screener/288 cycles, ENTERPRISE=$100/20M
tokens/100k broker/20k screener/1440 cycles. `ResourceUsage` is
the per-user rolling-window accounting row carrying `user_id` /
`tier` / `window_started_at` / per-resource counters; every counter
validates non-negative at construction. `QuotaTracker` is the
stateless engine: `check(usage, *, resource, requested=0)` returns
a read-only `QuotaCheckResult` snapshot (used / limit / remaining /
pct_used / state / window_started_at / warnings); `consume(usage,
*, resource, amount)` returns a new usage row with the amount
added or raises `QuotaExceededError` carrying the user_id /
resource / tier / used / limit context for the operator's
exception handler. **Five pinned safety semantics**: (1) **rolling
24-hour windows** rather than calendar days — calendar-day reset
at the operator's UTC midnight could wipe out a US-East user's
morning usage in a way that feels random; rolling window means
each user's budget refreshes 24h after they started spending
(pinned via test that 25h triggers reset, 23h59m doesn't, exactly
24h is inclusive); (2) **WARNING band starts at 80%, EXCEEDED at
100%** — both thresholds inclusive (8.0 of 10.0 is WARNING;
exactly 10.0 is EXCEEDED); zero-limit resource (FREE tier
broker_api_calls) with zero use lands OK, zero-limit with any use
lands EXCEEDED rather than divide-by-zero; (3) **negative
consumption rejected** — refunds aren't a quota concern, prevents
an integer-overflow / accidental-credit class of bug; (4)
**consuming up to but not past limit succeeds** — operator-friendly
choice that lets a user spend their last cent rather than rejecting
the call that brings them to exactly cap (the next consume of
positive amount fails); (5) **fractional token consumption rounds
down** via `int()` — half a token consumes zero tokens (user
gets benefit-of-the-doubt for partial counts). The tracker is
stateless: usage rows flow through (caller persists; tracker
computes); a future Postgres adapter just passes ResourceUsage
rows through the same surface. `check()` does not mutate the input
usage row even when it auto-rolls the window in the returned
snapshot — pinned via test. `remaining()` clamps at zero when
over (no negative remaining). Render helper produces emoji-prefixed
(✅/⚠️/🚫) one-user lines visually consistent with the rest of
the ops-display surfaces; USD renders with `$` prefix and 2/4
decimal precision, token counts as plain integers. Halal alignment:
quota engine is purely operational — never opens a position; the
budget gate sits in front of LLM / screener / broker calls so a
runaway user can't burn the whole instance's spend; pure-Python
(stdlib `dataclasses` + `enum` + `datetime`); no DB / network /
async / HTTP. Frozen dataclasses on every output. 60 tests cover:
default-tier-limits sanity (every tier present; FREE disables
broker_api_calls; PRO > FREE on every dimension; ENTERPRISE > PRO),
TierLimits + ResourceUsage field validation (negative-rejection on
every counter; zero accepted as the disable case), `for_resource`
+ `used_for` selectors return the right field for every resource,
QuotaTracker construction (default limits used when None passed;
partial map rejected; custom limits flow through), the three-band
classification ladder pinned at the boundary in both directions
(below 80% → OK; exactly 80% → WARNING; 95% → WARNING; exactly
100% → EXCEEDED; 150% → EXCEEDED), the zero-limit edge cases
(zero limit + zero use → OK; zero limit + any use → EXCEEDED with
no divide-by-zero), `check()` with `requested>0` simulates consume
without mutating input, negative-requested rejected, `remaining`
clamps at zero, consume lifecycle (adds to existing; returns new
frozen row; raises QuotaExceededError on overage with full context;
at-exact-limit succeeds; rejects negative amount; zero-amount is
no-op; fractional token count floors via int), the rolling 24h
window pin (25h elapsed → reset, 23h59m → no reset, exactly 24h
inclusive), check() also rolls the window in returned snapshot
without mutating input, QuotaCheckResult shape carries every
field, warnings empty when OK, render output across every state
with emoji + state name (USD format vs integer format pin),
window-start in render output, frozen-dataclass immutability across
TierLimits / ResourceUsage / QuotaCheckResult, all enum string
values pinned for JSON / DB stability (free / pro / enterprise;
llm_usd / llm_tokens / broker_api_calls / screener_api_calls /
cycle_runs; ok / warning / exceeded), cross-tier scenarios (FREE
user blocked immediately on broker_api_calls; ENTERPRISE has 200×
the LLM headroom of FREE), an end-to-end realistic PRO-user flow
(across-the-day spending hits WARNING at 80%, rejects $2.01 over,
next-day window reset starts fresh), and the QuotaExceededError
context regression (carries user_id / resource / tier / used /
limit; str(exc) mentions all four). Postgres persistence (a
`resource_usage` table + repository wired to `db/repository.py`),
the LLM-router gate (every `core/llm/factory.py` call site checks
quota before the LLM call), the broker-side gate (every
`crypto/exchange.py` request checks quota), and the dashboard
quota tile rendering per-user state deferred to follow-ups —
engine verified in isolation first.

### 3.D — Multi-tenant database isolation ✅ landed (scope guards)

Either: (a) row-level user_id on every table + Postgres RLS policies,
or (b) per-user schemas. Pick (a) for simplicity. Update every repo
to filter by user_id from the request context.

**Landed (scope guards)** as `db/tenant_isolation.py` — pure-
Python tenant-scope guard layer that runs *before* the SQL query,
so a forgotten WHERE clause in a repository method can't silently
leak one tenant's data to another. ContextVar-based active scope
with two-mode `ScopeKind` (USER / ADMIN). `TenantContext` frozen
dataclass carries user_id + scope; rejects empty/whitespace
user_id at construction. `enter_user_scope(user_id)` /
`enter_admin_scope(admin_user_id)` are the explicit context
managers — admin scope is opt-in only, never implicit. The
`require_tenant()` helper raises `TenantViolationError` when
called without an active scope; `assert_row_scope(*, row_user_id,
operation)` is the read-side guard repositories call after
fetching a row; `assert_payload_scope(*, payload_user_id,
operation)` is the write-side guard. **Five pinned semantics**:
(1) **TenantContext required for every scoped operation** —
`require_tenant()` raises explicitly rather than silently
allowing fallthrough; (2) **ADMIN scope opt-in only** —
admin scope can't be entered implicitly; the explicit
`enter_admin_scope` makes admin-side queries traceable in audit
logs; pinned via test that admin scope rejects empty user_id;
(3) **empty user_id rejected** at TenantContext construction —
mirrors validation patterns of Wave 11.C KYC + Wave 11.D privacy;
(4) **row-validation helper raises on cross-tenant** — when a
fetched row's user_id doesn't match the active scope,
`TenantViolationError` raised; SQL bugs (missing JOIN, wrong
WHERE) caught at the application layer; admin scope bypasses but
still requires the call (ops audit logs prove the check ran);
(5) **render output redacts other tenants' user_ids** —
`TenantViolationError.message` references the active scope's
user_id but the would-be-leaked `target_user_id` is rendered as
`<other-tenant>` so error logs / Slack alerts can be safely
shipped to ops channels. The target_user_id stays on the
exception object for debugging in dev environments. Mirrors
no-PII patterns of Wave 11.D + 11.C + 3.B. Picked an explicit
guard module over Postgres-only RLS because the bot's tests run
against the same SQLAlchemy + asyncpg layer the production
deploys to, and a Python-side guard is testable without spinning
up Postgres in unit tests; RLS at the DB layer remains
recommended as defence-in-depth — this module is the operator's
first defensive boundary, not the only one. Halal alignment:
isolation guard is operator-side privacy; never opens a position;
pure-Python (stdlib `contextvars` + `contextlib` + `dataclasses`
+ `enum`); no DB / network / async. Frozen dataclass on
TenantContext. 44 tests cover TenantContext validation
(empty/whitespace user_id rejected; default scope is USER; admin
scope accepted); default-no-scope state (current_tenant is None;
require_tenant raises; is_admin_scope False); enter_user_scope
lifecycle (sets current_tenant; restores previous on exit;
nested scopes; rejects empty user_id; is_admin_scope still False);
**enter_admin_scope explicit-opt-in pin** (sets ADMIN kind;
is_admin_scope True; restores previous on exit; rejects empty
admin_id); assert_row_scope (passes for matching row; raises for
mismatch with active+target user_ids; admin bypasses; raises
without scope; rejects empty row_user_id; operation label flows
through); assert_payload_scope (matching pass; mismatch raises;
admin bypasses; raises without scope; rejects empty payload_user_id);
**TenantViolationError no-PII pin** (str() shows `<other-tenant>`
not the actual target_user_id; target_user_id on exception
object for debug; default operation is "query"); frozen-dataclass
immutability; ScopeKind string values pinned; render output
across no-scope (🔓) / user-scope (👤) / admin-scope (🛡️) with
emoji per kind; render shows active user_id only; end-to-end
realistic flows (repository.get_trade simulated SQL bug → blocked
by row-scope check; correct row passes; admin compliance review
across tenants permitted; write with payload-user-id mismatch
blocked; nested admin-into-user-scope restores correctly);
ContextVar-isolation regression (scope doesn't leak after exit;
multiple sequential scopes); operation label flows through to
the violation error. Postgres RLS policies (defence-in-depth
layer at the DB), Wave 3.A user-auth integration (extracts user_id
from JWT, enters user_scope per request via FastAPI middleware),
the operator admin console that enters admin_scope explicitly
with audit-log capture, and per-repo updates to call
`assert_row_scope` / `assert_payload_scope` at every read / write
entry deferred to follow-ups — guards verified in isolation
first.

### 3.E — Self-service onboarding flow ✅ landed (`web/onboarding.py`)

New user lands → Google sign-in → "connect a broker" → optional broker
keys → "pick strategy" → optional model choice → first cycle runs in
paper mode automatically. Should be < 5 minutes from sign-in to first
trade simulated.

**Implementation**: `src/halal_trader/web/onboarding.py` ships the
pure-Python state machine the onboarding route consumes. `OnboardingStep`
enum (SIGNED_IN / BROKER_CHOSEN / BROKER_KEYS_STORED / STRATEGY_CHOSEN
/ MODEL_CHOSEN / FIRST_CYCLE_RUN) with pinned string values + canonical
`_STEP_ORDER`. `_OPTIONAL_STEPS` frozenset names the two optional steps
(BROKER_KEYS_STORED, MODEL_CHOSEN) — required steps cannot be skipped.
`StepStatus` enum (PENDING / COMPLETED / SKIPPED). `StepOutOfOrderError`
+ `StepNotSkippableError` exceptions carry the offending step + the
missing prerequisite for a clean operator-facing error. `OnboardingPolicy`
ships the 5-minute SLA threshold (operator-tunable; rejects zero /
negative). `StepRecord` (audit row) rejects naive `decided_at` and
PENDING status (a record is a *decision*, not a not-yet). `OnboardingState`
is the immutable per-user state — operations (`complete_step`, `skip_step`)
return new state rather than mutating, pinned for replay-ability.
`start_onboarding` creates initial state with SIGNED_IN already
completed. `complete_step` enforces canonical order — completing
STRATEGY_CHOSEN before BROKER_CHOSEN raises `StepOutOfOrderError`;
optional steps still pending don't block (a user with no broker keys
can still pick a strategy). `skip_step` only accepts optional steps —
required steps raise `StepNotSkippableError`. `time_to_first_trade`
returns the elapsed time from SIGNED_IN to FIRST_CYCLE_RUN (None if
incomplete). `flag_sla_breach` is `time_to_first_trade > sla_threshold`
— exclusive boundary so exactly 5 minutes is NOT a breach (pinned both
directions). In-progress flow doesn't breach yet (False until complete).
`progress_pct` returns fraction of decided steps (skipped counts as
decided). `render_onboarding_state` shows progress emoji per step
(✅ completed / ⏭️ skipped / ⬜ pending), next-step hint, completion
marker with elapsed-seconds. No-secret-leak regression: never includes
api_key / cus_ / sub_ / bearer / session_ substrings. Tests in
`tests/test_onboarding.py` (60 cases): step + status enum string-value
pins; policy validation (5min default, zero/negative rejected, frozen);
StepRecord validation (naive decided_at rejected, PENDING status
rejected, frozen); OnboardingState validation; start_onboarding
(initial SIGNED_IN, naive-now rejected, empty user_id rejected,
1/6 progress); complete_step (advances state, full sequential happy
path, out-of-order rejected, optional skipping allowed,
already-decided rejected, naive-now rejected, returns new state
not mutates, records timestamp); skip_step (marks SKIPPED, rejects
required steps SIGNED_IN/BROKER_CHOSEN/STRATEGY_CHOSEN/FIRST_CYCLE_RUN,
optional MODEL_CHOSEN skippable, already-decided rejected); next_step
(first pending in order, skips decided optional, None when complete);
is_complete (false when first_cycle pending, true when completed);
time_to_first_trade (None when incomplete, elapsed when complete);
flag_sla_breach (boundary exclusive at 5min; >5min breaches; in-progress
doesn't; custom policy flows through); progress_pct (initial 1/6, after
complete 2/6, after skip 3/6 — skipped counts as decided, full 1.0);
render output (user_id + progress %, status emoji, next-step hint,
completion marker with elapsed-seconds, no-secret-leak regression);
e2e flows (happy path under SLA, skip-optionals real-world, slow user
breaches SLA, replay consistency).

### 3.F — Stripe billing + tier upgrades ✅ landed (`web/billing_state.py`)

Pro tier: $19/mo, real-money trading + larger universe + GPT-4
access. Enterprise: custom pricing, multi-broker, multi-strategy,
SLA support. Stripe webhooks update user tier.

**Implementation**: `src/halal_trader/web/billing_state.py` ships the
pure-Python billing state machine the Stripe webhook handler composes
with. `Tier` is re-used from Wave 3.C (quotas) so billing + quota gate
speak the same vocabulary. Six-state lifecycle (`TRIALING / ACTIVE /
PAST_DUE / GRACE_PERIOD / CANCELLED / EXPIRED`) with pinned string
values for JSON/DB stability. Seven event kinds (`SUBSCRIPTION_CREATED
/ INVOICE_PAID / INVOICE_PAYMENT_FAILED / TIER_UPGRADED / TIER_DOWNGRADED
/ SUBSCRIPTION_CANCELLED / TRIAL_ENDED`). `BillingPolicy` validates
trial in `[7, 30]` days (default 14) + positive grace period (default
7d). `Subscription` enforces tz-aware datetimes + period-end-after-start
invariant; `BillingEvent` requires `target_tier` for `TIER_UPGRADED /
TIER_DOWNGRADED`. `create_trial` rejects FREE (no trial path).
`apply_event` is deterministic + idempotent on `SUBSCRIPTION_CREATED`
(operators replay event history to audit historical tier-at-moment).
Pinned semantics: invoice paid extends period + clears grace; invoice
failed enters 7-day grace; upgrades take effect immediately; downgrades
+ cancellations stage at period end via `cancel_at_period_end`; trial
end without conversion → `EXPIRED`. The load-bearing
`compute_effective_tier` is what the Wave 3.C quota gate keys on:
covers every status × `cancel_at_period_end` × time combination
(EXPIRED→FREE; CANCELLED past period end→FREE; GRACE past grace
end→FREE inclusive boundary; TRIALING past trial end→FREE inclusive
boundary; ACTIVE-with-cancel-at-period-end past period end→FREE).
`render_subscription` shows status emoji (🆓✅⏰⚠️🛑❌) + user_id +
tier + effective tier + period + trial/grace/cancel markers; pinned
no-secret regression: never includes Stripe customer IDs / invoice
amounts / dollar signs (mirrors Wave 3.B vault + Wave 12.G co-pilot).
Tests in `tests/test_billing_state.py` (76 cases): policy validation
+ boundary cases (7d/30d trial; 0/-1 grace); subscription tz-aware
+ period-end-after-start invariants; event target_tier requirement;
enum string-value pins; `create_trial` happy paths + FREE rejection
+ naive-now rejection; every event-kind transition; effective-tier
across every status × time combination including boundary-inclusive
grace expiry; no-secret-leak render contract; e2e flows (trial →
conversion; invoice failure → grace → recovery; invoice failure →
grace expires to FREE; cancellation lifecycle; immediate upgrade;
trial expires; downgrade-takes-effect-at-period-end); determinism
regression pins.

### 3.G — Admin console ✅ landed (`web/admin_console.py`)

Hosted-tenant admin: see all users, their LLM spend, their portfolio
P&L, halt-switch any user's bot, billing override. Restricted to
configured admin emails.

**Implementation**: `src/halal_trader/web/admin_console.py` ships the
pure-Python admin view-model + action audit layer that composes the
multi-tenant primitives (3.A auth, 3.C quotas, 3.F billing) into a
single view-model the admin route consumes. `AdminEmailList` is a
frozen-set of admin emails normalised to lowercase + whitespace-strip
at construction (so `Admin@Foo.com` and `admin@foo.com` are the same
admin); rejects empty list / empty email / email without `@`.
`UserSummary` frozen dataclass carries the cross-cutting triage facts
(user_id, email, effective_tier, subscription_status, LLM USD spend
today, LLM USD limit today, halt_active, last_active_at, joined_at)
and *deliberately* does NOT carry: password hash, scrypt salt,
session tokens, broker API keys, Stripe customer ID, invoice amounts
— the no-secret-leak contract is structural, not just a render-time
filter. `AdminAction` enum (six values: HALT_USER / RESUME_USER /
OVERRIDE_TIER / REVOKE_SESSION / SUSPEND_USER / INSPECT_USER) with
pinned string values for JSON / DB stability. `AdminActionRequest`
validates: empty action_id / target_user_id / performed_by rejected;
performed_by must contain `@`; performed_at tz-aware; reason required
for the four destructive actions (HALT_USER / OVERRIDE_TIER /
SUSPEND_USER / REVOKE_SESSION) — RESUME_USER and INSPECT_USER do
not require reason; target_tier required for OVERRIDE_TIER.
`audit_admin_action(request, admin_emails)` is the load-bearing
authorization boundary — non-admin caller raises
`AdminAuthorizationError` carrying the offending email; admin caller
gets `AdminActionRecord` with the email normalised on the audit row.
`build_admin_view(users, *, now, active_window=24h)` aggregates a
user iterable into `AdminView` with: total_users, active_users_24h
(boundary inclusive: exactly 24h ago is still active), halt_active
count, total_llm_usd_today, tier_breakdown (every Tier appears even
with zero users), users sorted by joined_at ascending (deterministic
ordering across renders). `render_user_summary` / `render_admin_view`
/ `render_action_record` produce ops display with tier emoji
(🆓💼🏢) + status emoji from billing_state (🆓✅⏰⚠️🛑❌) + halt
marker; pinned no-secret regression: never includes `cus_` / `sub_`
Stripe IDs, `password` / `hash` / `salt`, `session_` / `bearer`
tokens. Tests in `tests/test_admin_console.py` (76 cases): admin-list
case-insensitive normalisation; non-admin rejection at audit
boundary; UserSummary tz-aware + non-negative-USD invariants;
llm_usage_fraction including zero-limit divide-by-zero guard;
AdminActionRequest reason-required-for-destructive-action pin
(both directions: HALT/OVERRIDE/SUSPEND/REVOKE require, RESUME/INSPECT
don't); target_tier required for OVERRIDE_TIER; whitespace-only
reason rejected; build_admin_view aggregation pins (active-window
boundary inclusive; tier_breakdown includes zero-count tiers; sorted
by joined_at; deterministic for same input); custom active_window
flows through; render no-secret-leak regression on user/view/record;
end-to-end flows (admin halts then resumes; non-admin attempt
blocked at boundary; full multi-user view render).

### 3.H — Multi-user dashboard with strategy-leaderboard ✅ landed (`web/leaderboard.py`)

Aggregated (anonymised) leaderboard: top 10 strategies by Sharpe this
quarter, by win rate this month. Lets users discover what's working
and clone a public strategy template. Privacy: opt-in only.

**Implementation**: `src/halal_trader/web/leaderboard.py` ships the
pure-Python ranking + anonymisation engine. `LeaderboardMetric` enum
(SHARPE / WIN_RATE / TOTAL_RETURN_PCT) and `LeaderboardWindow` enum
(MONTHLY=30d / QUARTERLY=90d / YEARLY=365d / ALL_TIME) with pinned
string values. `LeaderboardPolicy` (default
min_entries_to_publish=5, top_n=10, min_sample_size=10) — `min_entries`
enforces the k-anonymity-like floor (below 5, individual entries are
re-identifiable by elimination). `StrategyEntry` carries user_id +
strategy_id + display_handle + strategy_kind + opt_in flag + dates +
metrics + sample_size; rejects empty fields, naive datetimes, win_rate
outside [0,1], negative sample_size. `LeaderboardRow` (the rendered
output) intentionally has NO user_id field — anonymisation is
structural, not just a render-time filter. `build_leaderboard`
pipeline: opt-in filter → time-window filter → min-sample-size filter
→ k-anonymity floor (suppressed if below) → sort by metric desc with
older-strategy-first tiebreak → top-N → strip user_id at the
boundary. `auto_handle(user_id)` produces `strategist_{sha256[:8]}`
stable handles that don't leak user_id (tested: email-shaped user_ids
don't leak components). `StrategyTemplate` rejects forbidden config
keys (user_id, email, broker_api_key, stripe_id — case-insensitive).
Render helpers (`render_leaderboard`, `render_template`) carry
no-leak regression: no email-shaped substrings, no Stripe IDs, no `$`
amounts. Tests in `tests/test_leaderboard.py` (67 cases): policy
validation; entry validation including win_rate boundary [0,1] both
directions; opt-in filter (opt-out entries with high sharpe excluded);
k-anonymity boundary inclusive at 5 (4 → suppressed, 5 → published);
window filter (30d boundary inclusive); ALL_TIME window includes old
entries; min_sample_size filter; ranking by each metric;
older-strategy-first tiebreak pin; top_n cap; rank starts at 1; row
has no user_id field structural pin; auto_handle stability +
no-user_id-leak; template forbidden-key check (user_id / email /
broker_api_key / stripe_id) case-insensitive; render no-leak
regression with email-pattern regex; e2e mixed opt-in/out flow.

### 3.I — Cloud deployment templates ✅ landed (`ops/deployment_manifest.py`) — manifest validator

Dockerfile + docker-compose + Terraform module + Helm chart for
self-hosted multi-user. Documented one-command deploy to AWS / GCP / Fly.io.

**Implementation**: `src/halal_trader/ops/deployment_manifest.py`
ships the pure-Python manifest spec + validator that the actual
deployment artifacts (Dockerfile / docker-compose.yml / Helm / TF)
will land separately and consume as their source of truth.
`DeploymentTarget` enum (DOCKER_COMPOSE / K8S_HELM / TERRAFORM_AWS
/ TERRAFORM_GCP / FLY_IO), `ServiceKind` (POSTGRES / BOT / DASHBOARD
/ AUX). `EnvVar` enforces secret-vs-inline mutual exclusion: secret
env vars must have `secret_ref` and no inline `value`; non-secret
env vars must have `value` and no `secret_ref` (the load-bearing
leaked-secret pin). `ResourceLimits` enforces memory in [128MB, 32GB]
and CPU in [0.1, 16.0] cores. `ServiceSpec` rejects duplicate env
names within a service + invalid port ranges + empty fields.
`DeploymentManifest` rejects duplicate service names + empty
services. `validate_manifest` enforces: required kinds (postgres +
bot + dashboard) present; postgres exposes 5432 or None (production
uses canonical port, not the test 5433); bot service has `DATABASE_URL`
as a secret (load-bearing pin against accidentally inlining the
postgres password into a docker-compose.yml); only one BOT service.
`collect_secret_refs` returns deduplicated sorted secret refs (the
deployment-time vault-resolution checklist). Render helpers with
target emoji (🐳 docker / ☸️ k8s / 🟧 AWS / 🔷 GCP / 🪂 fly) + service
emoji (🗄️ postgres / 🤖 bot / 📊 dashboard / 🧰 aux). No-secret regression:
secret env vars render as `<secret:ref>` placeholder, never inline.
Tests in `tests/test_deployment_manifest.py` (56 cases): 5 target +
4 kind enum string-value pins; EnvVar inline + secret happy paths +
4-way mutual exclusion (inline-with-secret rejected, secret-without-ref
rejected, secret-with-empty-ref rejected, non-secret-with-ref rejected,
non-secret-without-value rejected); ResourceLimits boundary pins (128MB
inclusive lower; 64GB rejected; 0.1 CPU inclusive lower; 32 CPU
rejected); ServiceSpec validation; DeploymentManifest duplicate
detection; immutability across all dataclasses; validate_manifest
happy path; rejects missing each of postgres/bot/dashboard; postgres
exposes-wrong-port rejected; postgres internal-only accepted; bot
without DATABASE_URL rejected; **bot with inline DATABASE_URL rejected
(load-bearing leaked-secret regression pin)**; multiple bots
rejected; AUX-only manifest rejected as missing required kinds;
ManifestViolationError carries manifest_name + reason; collect_secret_refs
deduplicates + sorts; total_resource_request math; render service /
manifest with emoji + secret placeholder + no inline-password leak;
e2e flows (full deployment validates; target swap docker→helm preserves
validation; **inline DATABASE_URL with "LeakedPassword123" caught by
validation BEFORE reaching any Dockerfile** — the load-bearing
regression test).

### 3.J — Mobile app ✅ landed (`web/mobile_push.py`) — push policy + delivery state

Native iOS/Android app for monitoring + halt-switch + push notifications.
The existing `/api/mobile/summary` is the contract. Wire the operator's
phone to receive trade fills + risk halts + daily summary as native
push.

**Implementation**: `src/halal_trader/web/mobile_push.py` ships
the pure-Python push notification policy + delivery state engine.
The actual RN/Flutter app + APNs/FCM SDK integration are deferred,
but the policy + delivery state are regression-pinned. `NotificationKind`
(TRADE_FILL / RISK_HALT / DAILY_SUMMARY) → `Priority` ladder
(NORMAL / CRITICAL / LOW). `Platform` (IOS / ANDROID). `DeliveryStatus`
(PENDING / SENT / DELIVERED / FAILED) state machine. `NotificationPolicy`
defaults: 22:00-07:00 quiet hours wraparound + 30 non-critical
notifications/hour rate limit. `DeviceRegistration` validates iOS
token must be hex (APNs format) + tz offset in [-720, 840] minutes
+ tz-aware datetimes. `NotificationRequest` enforces APNs limits
(title ≤ 100 chars, body ≤ 1000 chars). `evaluate_gate(request, *,
registration, recent_send_count, policy)` returns GateOutcome (SEND
/ HOLD_QUIET_HOURS / HOLD_RATE_LIMIT / HOLD_NO_DEVICE). Pinned
semantics: CRITICAL bypasses BOTH quiet hours AND rate limits;
NORMAL/LOW respect both. Quiet hours boundaries: 22:00 inclusive
(quiet), 07:00 exclusive (out of quiet). Rate limit at exactly 30
holds, 29 sends. Per-user-timezone conversion via `_user_local_now`
so a NY user at 14:00 UTC = 09:00 NY (sends) vs 04:00 UTC = 23:00
NY (holds). Delivery state machine enforces forward order: PENDING
→ SENT → DELIVERED; cannot skip. FAILED can come from any pre-FAILED
state with required `failure_reason`; FAILED is terminal.
`count_recent_sends` excludes PENDING and FAILED — only SENT +
DELIVERED count toward rate limit (a failed delivery shouldn't
penalize the user). Render helpers with kind emoji (💸 trade /
🚨 halt / 📊 summary), priority emoji (🔴/🟡/🔵), status emoji
(⏳/📤/✅/❌). No-secret regression: render never includes
device_token / FCM / APNs / key. Tests in `tests/test_mobile_push.py`
(71 cases): all 5 enum string-value pins; priority ladder pinned;
NotificationPolicy validation including quiet-start-equals-end
rejection; DeviceRegistration including iOS hex format pin; APNs
title 100-char + body 1000-char limits; evaluate_gate across no
device / revoked device / business hours / quiet hours / quiet hours
boundaries (22:00 inclusive, 07:00 exclusive, 06:00 quiet); CRITICAL
bypasses quiet hours AND rate limit (1000 sends still SEND);
30-recent-sends-rate-limit pin; 29 sends still SEND; per-user-tz
conversion (NY 09:00 SEND vs 23:00 HOLD); delivery state machine
(forward only, cannot skip PENDING→DELIVERED, FAILED from any pre-
FAILED state, requires reason, FAILED terminal); state immutability;
count_recent_sends includes SENT + DELIVERED, excludes PENDING +
FAILED, filters by user_ids, 60min boundary inclusive; render with
kind+priority+status emojis + no-secret regression; e2e flows (RISK_HALT
at 3am delivers; DAILY_SUMMARY at 3am held; noisy strategy rate-
limited but RISK_HALT still sends; replay consistency).

**Acceptance**: A new user can sign up at app.halal-trader.dev and
have a paper-trading bot running within 5 minutes, never touching
the codebase.

---

## Wave 4 — Best-in-class strategy quality (6-10 weeks)

Move from "GPT picks trades" to a quantitative-research-firm grade
strategy stack.

### 4.A — Reinforcement learning policy training ✅ landed (`ml/rl_training.py`) — config + episode tracker

Beyond LLM-only: train a PPO/SAC policy on historical replay data
(the existing `core/replay.py` snapshots are perfect training data).
The RL policy + LLM rationale form an ensemble; the LLM provides
explainability while the RL policy provides rigour.

**Implementation**: `src/halal_trader/ml/rl_training.py` ships the
pure-Python config + episode tracker + reward-shaping engine that
the actual training step (Stable-Baselines3 / Ray RLlib / CleanRL —
operator-side) consumes. `RLAlgorithm` enum (PPO / SAC) + `ActionSpace`
enum (DISCRETE / CONTINUOUS) with pinned string values. `TrainingConfig`
validates hyperparameters: learning_rate in (0, 0.1] inclusive at
upper bound, gamma in (0, 1) exclusive at both ends (1.0 means
infinite horizon — explicitly rejected), batch_size positive,
clip_range in (0, 1), total_timesteps positive. Algorithm/action-
space compatibility enforced: PPO requires DISCRETE, SAC requires
CONTINUOUS — pinned both directions. `RewardShapingPolicy` ships
default weights (return=1.0, sharpe=0.5, drawdown=2.0 — drawdown
weighted 2x to enforce avoid-drawdown roadmap pin); rejects negative
weights so reward direction can't accidentally flip but accepts zero
weights (operator can disable a component). `shaped_reward(*, raw_return,
sharpe, drawdown_pct, policy)` computes return_weight * raw_return +
sharpe_weight * sharpe - drawdown_weight * drawdown_pct; rejects
negative drawdown_pct. `TrajectoryStep` carries timestep + action_id
+ raw_return + sharpe + drawdown_pct + shaped_reward; validates
non-negative timestep + non-empty action_id + non-negative drawdown.
`Episode` is immutable; `start_episode` / `append_step` /
`terminate_episode` return new states. Properties: step_count,
total_reward, total_raw_return, max_drawdown_pct (zero when empty).
Step ordering enforced (step.timestep must equal current step_count).
`TrainingResult` summary + `aggregate_results` for end-of-run
aggregation. Render helpers with algorithm emoji (🎯 PPO / 🌊 SAC),
no-secret regression on weights/optimizer/checkpoint. Tests in
`tests/test_rl_training.py` (63 cases): RLAlgorithm + ActionSpace
enum string-value pins; default config sanity; learning_rate at upper
boundary 0.1 inclusive, 0.11 rejected; gamma at 0.0 / 1.0 both
rejected; PPO+CONTINUOUS rejected, SAC+DISCRETE rejected, SAC+CONTINUOUS
accepted; immutability across all dataclasses; reward shaping default
weights pinned; negative weights rejected per-component; zero weight
accepted; shaped_reward arithmetic exact pin (1.0*0.01 + 0.5*1.0 -
2.0*0.05 = 0.41); zero drawdown case; negative-return-with-drawdown
double-negative pin; negative drawdown_pct rejected; custom policy
disabling drawdown component flows through; trajectory step validation;
episode validation + immutability; start_episode + append_step +
terminate_episode lifecycle; step_count / total_reward / max_drawdown
properties; ordering enforcement (step.timestep must match);
TrainingResult validation; aggregate_results across realistic 100-
episode 5-step run with mean_episode_reward = 2.55 exact (5 steps ×
0.51); render config + result with algorithm emoji + no-secret
regression; e2e flows (10-step episode lifecycle; 100-episode
training run aggregate; replay consistency).

### 4.B — Genetic strategy generator  ✅ landed (engine)

The existing prompt-evo GA evolves prompts. Extend to evolve
*strategies* — combinations of indicator signals, regime conditions,
risk filters. Auto-promote winners after 100+ paper trades cross
a Sharpe threshold.

**Landed (engine):** `ml/strategy_ga.py` ships `evolve(*,
fitness_fn, population_size, generations, …)` →
`list[GenerationReport]`. Genome (`StrategyGenome`) carries
indicator filters (rsi_buy_max / rsi_sell_min / macd_required /
bb_buy_below / min_volume_ratio), the optional regime gate, and
risk caps (max_position_pct / max_simultaneous_positions /
min_confidence). Search-space bounds (`GenomeBounds`) are
operator-tweakable so the GA can be locked to a sub-search (e.g.
"only evolve indicator filters, fix regime_gate=uptrend").
Operations: `random_genome` (uniform within bounds),
`crossover` (uniform two-parent), `mutate` (per-gene Gaussian
nudges + bool flips + regime resamples at `rate`), `tournament_select`
(picked over fitness-proportionate roulette because Sharpe can
legitimately go negative and roulette degenerates), `elitism`
(top-K survival pinned to keep the best-found solution).
Three invariants pinned across all operations: bounds-respect
(every genome stays inside the search space), `rsi_buy_max <
rsi_sell_min` band ordering (a genome where buy_max ≥ sell_min
would never produce sells; helpers swap), and seed-determinism
(every operation takes a `random.Random`; same seed + same
fitness fn → same final genome). Fitness is fully caller-supplied
— the engine is mechanic, not application; doesn't know what BUY
looks like, doesn't call brokers, doesn't read backtests. The
caller hands over `fitness_fn(StrategyGenome) → float` and the
GA does the rest. `seed_population` lets operators inject
hand-tuned genomes (e.g. the live prompt's current settings) so
the GA builds off a known-good start; padding with randoms when
short. Halal alignment: bounds default to long-only,
no-leverage (max_position_pct ≤ 0.30; no negative quantities).
Pure-Python (no NumPy / SciPy / DB / async); the fitness call is
the only blocking point — caller is responsible for parallel
backtests if they want them. 27 tests cover bounds-respect on
random + mutate (100-iteration stress), the band-ordering
invariant after crossover from cross-violating parents,
seed-determinism on every primitive, the `mutate(rate=0)` no-op
pin, regime-gate lockdown via custom bounds, tournament selection
+ population-smaller-than-tournament-size + empty-population /
zero-size validation, elitism descending order + zero-K, the
`evolve` driver's N-generations-→-N+1-reports off-by-one pin,
the monotonic-non-decreasing-best invariant under elitism, an
end-to-end convergence test against a peaked synthetic fitness
landscape (final-gen best > initial-gen best, lands within 0.5
of the optimum), seeded-population injection + padding,
input-validation rejections, the `to_dict` round-trip, and the
genome immutability. Auto-promotion gate (run the new
`PromotionThresholds` from Wave 4.F over the GA's best after
100+ paper trades) and dashboard live-evolution view deferred
to follow-ups; engine verified in isolation first.

### 4.C — Causal regime detection ✅ landed (Bayesian network)

Replace the rule-based regime detector with a causal model: given
the current macro state (rates, VIX, dollar index, sector breadth),
which trading regime are we in? Train on 20 years of SPY data; outputs
a probability distribution over regimes that gates strategy selection.

**Landed (Bayesian network)** as `ml/causal_regime.py` — pure-
Python discrete Bayesian network covering five macro variables
(`RatesLevel` LOW/NORMAL/HIGH; `RatesChange` EASING/HOLDING/
TIGHTENING; `VIXState` CALM/ELEVATED/CRISIS; `DXYState` WEAK/
NORMAL/STRONG; `BreadthState` NEGATIVE/MIXED/POSITIVE) → Regime
target (RISK_ON/NEUTRAL/RISK_OFF/CRISIS). The graph: rates_change
→ vix; rates_level → dxy; vix → breadth; (vix, dxy, breadth) →
regime. Wave 4.D landed correlation-based fusion; this wave ships
the **causal** counterpart with **do-calculus interventions**
distinct from observations. The distinction matters operationally:
observation `vix=CRISIS` propagates upstream via Bayes' rule;
intervention `do(vix=CRISIS)` does NOT update upstream beliefs
because we *forced* vix to that state. Picked discrete Bayesian
network with brute-force enumeration over sample-based methods
because (a) graph has 5 variables × 3 states (3^5 = 243 cells) so
brute force is fast and exact, (b) operators can read the CPTs in
source, (c) sample-based methods need burn-in + convergence checks
operationally tedious for a per-cycle hot path. Six pinned
semantics: (1) **closed variable + state set** — adding a
variable / state is code review; (2) **CPTs module-level frozen**
— runtime config drift can't change priors; (3) **intervention
overrides observation** when both supplied for same field
(warning emitted); (4) **partial evidence supported** — engine
marginalises out missing fields; empty evidence returns the prior;
(5) **probability distributions sum to 1.0** within tolerance —
output renormalised to prevent float drift; (6) **render output
shows regime marginal as percentages** plus most-likely
+ confidence + bar chart, doesn't dump raw CPT cells. Halal
alignment: read-only inference; never opens a position; pure-
Python (stdlib `dataclasses` + `enum`); no NumPy / scipy / DB /
network / async; frozen dataclasses. 43 tests cover empty-evidence
prior; distribution-sums-to-1 invariant; single-variable
observation effects (CALM VIX skews RISK_ON; CRISIS VIX skews
CRISIS+RISK_OFF; negative/positive breadth correctly skews);
full-evidence scenarios (full RISK_ON config → RISK_ON; full
CRISIS config → CRISIS or RISK_OFF; NEUTRAL); confidence
properties (equals most-likely probability; in [0, 1]; full
evidence locks to the specific CPT cell); **do-calculus
intervention pins** in five flavours (single-variable
intervention forces state regardless of upstream evidence;
intervention does not propagate upstream; multi-variable
intervention; **intervention-overrides-evidence with warning** —
the load-bearing pin); evidence_used + intervention_used
preserved through to result; frozen-dataclass immutability; all
six enum string values pinned for JSON / DB stability; render
output (most-likely + confidence + distribution bar chart;
do-clause when intervention set; warnings shown; emoji per
regime); end-to-end realistic scenarios (2008 GFC → CRISIS+RISK_
OFF combined > 70%; late-2020 recovery → RISK_ON > 70%; what-if
VIX-doubles intervention shifts allocation); CPT validation
(every cell sums to 1.0; complete coverage of (vix, dxy, breadth)
combinations — the 27-cell enumeration regression-pin). Live
macro-data ingest (FRED for rates; Yahoo / CBOE for VIX; ICE for
DXY; computed sector breadth from constituent ETFs), Postgres
`macro_observations` + `regime_inferences` tables wired to `db/
repository.py`, the per-cycle hook that infers the regime + feeds
it into `crypto/strategy.py` for regime-gated entries, and the
dashboard's regime tile rendering the distribution + most-likely
arrow deferred to follow-ups — engine verified in isolation
first.

### 4.D — Cross-asset signal fusion  ✅ landed (engine)

The current cycle treats each pair independently (modulo correlation
in risk). Add cross-asset signals: "BTC breaking out + DXY weakening +
TIPS rallying = high-conviction risk-on"; "VIX > 30 + gold breakout +
yield curve inverting = regime-shift, raise cash". This needs a
**market context engine** that pulls macro indicators every cycle.

**Landed (engine):** `core/cross_asset_signal.py` ships
`fuse(MacroContextSnapshot, *, thresholds)` → `MacroRegimeSignal`.
Five per-factor scorers (VIX / DXY / yield curve + 10y change /
gold / sector breadth) each emit a `MacroSignalReason` with a
signed score (positive = risk-off; negative = risk-on); the fusion
sums them, classifies into `RISK_ON` / `NEUTRAL` / `RISK_OFF`
against tunable thresholds, and computes a `risk_bias ∈ [-1, 1]`
the strategy multiplies into entry sizing. Picked a hand-rolled
rule rather than an LLM call because the macro signal must be
≤1ms in the cycle hot path *and* deterministic (operators want
regression-tested mappings from a known macro snapshot to a known
signal); LLM here adds latency, cost, and non-determinism for no
edge over the published practitioner rules. Confidence is the
geometric mean of coverage (fraction of factors measured) and
agreement (how unanimous the measured factors are) — pinned so a
2-of-5 partial feed lands at ~0.45 confidence even with full
agreement; `risk_bias` is scaled by confidence so a thin feed
exerts less pull on the strategy. Cold-start safety: empty
snapshot → NEUTRAL with zero confidence and zero risk_bias —
strategies that interpret neutral as "no macro tilt" stay
correct. Halal alignment baked into the docstring: signal can
downsize a high-conviction LLM buy or raise the entry bar; never
opens a new position by itself, never shorts. Pure-Python; no
NumPy / scipy / DB / async. Frozen dataclasses safe to cache.
37 tests cover every scorer's threshold logic (VIX extreme /
high / spike / calm; DXY strong both directions; curve inverted
+ 10y move; gold rally + dump asymmetry; breadth strong / weak /
mid), the fusion's classic risk-off and risk-on combos, mixed
signals → NEUTRAL, threshold customisation, the confidence
formula across coverage and agreement axes, the risk_bias
clamp invariant, the no-mutation immutability, and the
`render_signal` emoji + factor list output. Concrete data feed
(FRED for yields, Yahoo for VIX/DXY/gold, computed sector
breadth) and the per-cycle hook deferred to a follow-up; engine
is verified in isolation first.

### 4.E — Order book / microstructure ML  ✅ landed (feature extractor)

Train a model on tick data (when we have it) that predicts short-term
price movement from order book imbalance + recent prints. Filter the
existing LLM picks: "the LLM says BUY but the OB model says next 5
minutes is bearish, downsize 50%".

**Landed (feature extractor):** `ml/microstructure.py` ships
`extract(snapshot, *, levels=10)` → `MicrostructureFeatures`.
The bot doesn't yet ingest L2 tick data, but when it does
(Coinbase Advanced Trade adapter / Binance L2 stream), the
strategy needs a small set of proven microstructure features
to filter or downsize LLM picks; this module is the feature
layer. Five features computed per snapshot, all bounded so a
glitched feed can't explode the downstream signal: **imbalance**
(top-N volume imbalance, ∈ [-1, 1]); **micro-price**
(volume-weighted "true mid" from Cartea / Jaimungal — leads the
simple mid by a few seconds; the weighting puts more weight on
the side with *less* size, the side about to move toward;
deviation from mid in basis points clamped to ±500bp);
**spread** (absolute and bp); **depth-decay** (slope of
log(volume) vs level index across both sides — flat book ≈ 0,
thinning book negative; clamped to [-5, 5]); **top-of-book log-
skew** (`log(bid_size / ask_size)` at the touch — clamped to
[-5, 5]; returns 0 when either side is zero rather than -inf).
Three pinned input contracts: bids must be descending and asks
ascending (extractor doesn't sort — pinned so a future adapter
that breaks the order surfaces immediately rather than silently
mis-computing); negative size and zero / negative price rejected
at construction; crossed books (best_bid ≥ best_ask) rejected
at the snapshot level rather than silently producing a negative
spread. Three pinned numerical safeties: zero-volume both sides
returns 0 imbalance (no information); zero-volume in one side at
the touch returns 0 log-skew (log(0) = -inf would corrupt
downstream); below 3 valid (non-zero) levels per side, depth-
decay slope is 0 (polyfit on 2 points has no useful information).
Picked a feature module rather than a model directly because the
features are stable across model iterations — pin them once
here so the future model layer has a stable input contract.
Operators can also feed the features into the LLM prompt as raw
values for explainability without an in-house model. Halal
alignment: read-only signal computation; never opens a position.
Pure-numpy (uses np.polyfit for the slope only); no DB / async.
39 tests cover input validation (level price + size, snapshot
sort order both directions, empty sides, crossed book), every
feature's correctness on a balanced book + asymmetric books +
extreme cases (clamping fires correctly), depth-decay edges
(flat → 0, thinning → negative, < 3 valid levels → 0,
zero-size levels skipped), top-of-book skew including the
log(0) → 0 fallback and clamping, the levels_used record on
both deeper-than-book and shallower-than-book inputs, and
render output. Tick-data wiring (Binance L2 WebSocket adapter
+ per-cycle feature snapshot + model layer) deferred to a
follow-up — extractor verified in isolation first.

### 4.F — Walk-forward + out-of-sample validation harness  ✅ landed (gate)

`crypto/walkforward.py` exists. Promote it to a first-class operator
tool: every new strategy must pass walk-forward + out-of-sample tests
before promote-to-live. Wire into the dashboard.

**Landed (gate):** `core/promotion_gate.py` ships
`evaluate_promotion(walk_forward, *, thresholds, monte_carlo,
ab_comparison)` → `PromotionVerdict(passed, checks, failures,
warnings)`. Composes three already-shipped sources — the existing
`WalkForwardReport`, the existing `MonteCarloReport`, and the
Round-4 Wave 5.B `ABComparison` — into one pass/fail decision.
Six walk-forward checks always run: out-of-sample average return ≥
floor, Sharpe ≥ 0.5, win rate ≥ 0.40, fold count ≥ 5, **worst-fold
max drawdown** ≤ 20% (pin: not the *mean* drawdown — a single fold
with a 30% trough must kill the promotion even if averages look
fine), total trades across folds ≥ 50. Two optional Monte Carlo
checks layer on when supplied: 95th-percentile drawdown ≤ 30%
(catches Sharpe-by-luck cases), runs ≥ 100 (the percentile is
only stable with enough resamples). One opt-in A/B check rejects
promotion on a *significant-but-worse* delta vs the live baseline
(pin: a strategy that's significantly worse must be rejected even
though the difference is statistically real — operator wants no
regressions). Defaults are deliberately demanding; thresholds can
be tightened or loosened, with both directions documented in the
audit trail. The gate is **additive** — adding Monte Carlo or A/B
inputs can only ever produce *more* failure conditions, never
soften an existing failure (a regression test pins this). Each
failed check carries a one-line `remediation` hint so the operator
knows *what to fix*, not just that it failed (e.g. "Sharpe is too
low for live trading. Tighten entry filters or reduce
holding-period noise"). Soft warning ladder for marginal fold
counts (fold_count just above floor → warn but pass).
`render_verdict` produces a CLI / Slack-ready text payload
visually consistent with `crypto/stress.render_report`. 33 tests
cover every check's pass / fail / unmeasured paths, the
worst-fold-not-mean-drawdown invariant, the additive-composition
contract on both directions (good MC keeps a passing WF passing;
bad MC turns a passing WF into a failing one), the A/B opt-in
semantics including the four edge cases (skipped by default,
required-but-missing fails, significant-and-better passes,
significant-but-worse rejects, non-significant rejects, None
p-value rejects), threshold customisation in both directions, the
`actual / threshold` round trip on `CheckResult`, immutability,
and the render helper's PASS / FAIL / remediation / warnings /
n/a-on-unmeasured output. CLI command (`halal-trader strategy
promotion-check`) and dashboard tile deferred to a follow-up;
gate is verified in isolation first.

### 4.F aux — Strategy backtest comparator ✅ landed (`ml/backtest_comparator.py`)

**Implementation**: `src/halal_trader/ml/backtest_comparator.py`
is the A-vs-B complement to Wave 4.F's single-strategy walk-forward
gate. Wave 4.F gates one strategy against historical baselines;
this module compares TWO strategies with significance testing.
`ComparisonVerdict` enum (A_WINS / B_WINS / TIE / INCONCLUSIVE).
`ComparisonPolicy` defaults: alpha=0.05, min_samples=50,
min_effect_size=0.1 (Cohen's d). `BacktestResult` carries strategy_id
+ summary stats per metric (mean + std + sample size), no raw
return series. Pure-Python Welch's t-test approximation using
`math.erf` for the normal-approximation p-value (avoids scipy
dependency). Cohen's d via pooled std. **Drawdown is special-cased
in `_LOWER_IS_BETTER` frozenset** — for max_drawdown_pct, lower
value WINS (drawdown is bad), pinned both directions. `compare_metric`
returns per-metric `MetricComparison` with t-stat + p-value +
cohens_d + verdict. `compare_backtests` returns `StrategyComparison`
with per-metric results + `overall_verdict` property aggregating:
**any INCONCLUSIVE → overall INCONCLUSIVE pin** (3 wins + 1
inconclusive ≠ winner — operators must wait for more data); A
needs majority + zero clear losses for A_WINS; otherwise TIE. Tests
in `tests/test_backtest_comparator.py` (44 cases): ComparisonVerdict
+ ComparisonPolicy validation; BacktestResult validation including
**negative drawdown rejected** (positive magnitude pin); Welch
t-test against published value (mean diff=2, std=1, n=100 → t≈14.14);
extreme p-values; identical-means → p=1.0; Cohen's d sign + magnitude;
**inconclusive on low samples** + **inconclusive on small effect
size**; A_WINS / B_WINS / TIE classifications; **drawdown-inverted
pin** (5% beats 15%); zero-variance edge case; custom strict policy;
StrategyComparison overall verdict (clear winner with majority +
no losses; **any inconclusive metric → overall inconclusive pin**;
2-2 split → TIE; all tie → TIE); compare_backtests rejects same
strategy_id; render with verdict emoji 🅰️/🅱️/🟰/❓ + no-secret
regression; e2e flows (momentum_v2 clearly beats momentum on all
4 metrics; 30-trade pilot returns INCONCLUSIVE; replay consistency).

### 4.G — Bayesian portfolio optimisation  ✅ landed (HRP core)

Replace fixed `max_position_pct` with a Bayesian-optimal allocation
across the active strategies + open positions. Use Black-Litterman
or hierarchical risk parity, accepting Sharia constraints (no shorts,
no leverage, halal universe).

**Landed (HRP core):** `ml/hrp.py` ships a pure-NumPy Hierarchical
Risk Parity allocator following López de Prado (JoPM 2016). Pipeline:
correlation → distance ``√(½(1−ρ))`` → single-linkage clustering →
quasi-diagonalisation → recursive bisection with inverse-variance
allocation factor. Halal constraints baked in: weights are always
non-negative (no shorts) and sum to ≤ 1.0 (no leverage). Zero-variance
assets are dropped silently (caller's universe can change cycle to
cycle). 26 tests cover: weight non-negativity, sum=1, lower-vol →
higher-weight, equal-vol → equal-weights, correlated cluster gets
diversification penalty, single-asset / empty / mismatched-shape /
short-history / invalid-buffer edges, plus direct unit tests on
every helper (`_correlation_to_distance`, `_inverse_variance_weights`,
`_cluster_variance`, `_single_linkage_order`, `_recursive_bisection`).
Wiring into the cycle (replacing the flat `max_position_pct` cap)
is intentionally deferred — the algorithm is verified in isolation
first; wiring is a separate, reversible follow-up.

### 4.G aux — Portfolio risk aggregator + pre-trade gate ✅ landed (`ml/risk_aggregator.py`)

**Implementation**: `src/halal_trader/ml/risk_aggregator.py`
complements Wave 4.G HRP weights + the crypto monitor's per-position
SL/TP enforcement with the **account-level risk view + pre-trade
gate**. `Position` carries symbol + notional_usd + entry_price +
stop_loss_price; long-only invariant pinned (SL strictly below
entry; short positions deferred). `at_risk_usd(position)` computes
notional × (entry − SL) / entry — pinned via test ($1000 × 5/100
= $50). `aggregate_risk(positions, *, account_value_usd)` returns
`RiskSnapshot` with totals + largest-position attribution.
`RiskPolicy` defaults: max_total_at_risk_pct=6%, max_single_position
=2%, max_position_count=20 (conventional retail risk caps);
**single-position cap can't exceed total cap** (validation pin).
`evaluate_pre_trade_gate(*, open_positions, new_position,
account_value_usd, policy)` is **forward-looking** — simulates
"after this new buy, are we within all 3 caps?" rather than checking
current state. Three rejection outcomes: REJECTED_POSITION_COUNT
(checked first) / REJECTED_SINGLE_POSITION / REJECTED_TOTAL_RISK.
All boundaries inclusive (exactly at cap → APPROVED). Render with
gate-outcome emoji (✅/🚫/⚠️/📊); no-secret regression. Tests in
`tests/test_risk_aggregator.py` (46 cases): RiskGateOutcome enum
string-value pin; **6%/2%/20 default policy pin**; policy validation
including single>total rejection pin; Position validation including
**SL-must-be-below-entry long-only pin both directions** (SL=entry
+ SL>entry); at_risk_usd math (pinned $50/$10/$200 known results +
proportional-to-notional); aggregate_risk empty + single + multi +
largest-identified; deterministic; RiskSnapshot validation +
immutable; pre-trade gate APPROVED on clean book; **REJECTED_SINGLE_POSITION
when 5% > 2% cap**; **2% boundary inclusive APPROVED pin**;
**REJECTED_TOTAL_RISK when 5% existing + 2% new = 7% > 6% cap**;
**6% total boundary inclusive APPROVED pin**;
**REJECTED_POSITION_COUNT 21 > 20 pin**; **20 boundary inclusive**;
**position-count-checked-first priority pin** (count breach +
single breach → POSITION_COUNT outcome); custom strict policy
flows through; render snapshot with $-amounts + percentages +
no-secret leak; render decision per outcome with distinct emoji;
e2e flows (realistic $100k account 5×1% existing + 0.5% new =
5.5% → APPROVED; over-concentrated 4% single rejected; replay
consistency).

### 4.H — Sentiment-from-real-data (not just headlines)  ✅ landed (filing classifier)

Beyond Reddit/CryptoPanic: ingest 8-K / 10-Q filings via SEC EDGAR
(already partially wired), conference call transcripts, analyst
revisions, insider trading filings. Build a per-symbol sentiment
score that's more grounded than headline aggregation.

**Landed (filing classifier):** `sentiment/filing_classifier.py`
ships `classify(text)` → `SentimentResult` and
`aggregate_for_symbol(*, symbol, filings)` → `SymbolSentiment`.
Pure rule-based: a small set of operator-readable lexicons
(20-phrase positive list, 26-phrase negative list, 16
strong-signal weight overrides at 1.5×–2.5×, 11-phrase negation
hedge list) with sliding-window matching and a bounded
aggregate. Picked rules over a trained model because (a)
operators need to read the rule and challenge a verdict — opaque
ML on a compliance-adjacent surface is the wrong default; (b)
the classifier stays under 100 lines of audit-able dataclass +
regex; (c) re-training requires labelled SEC text we don't
have. Three pinned semantics: **longest-phrase-match-first**
("record revenue and record earnings" produces two hits not
three; the longer phrase wins at any starting position to avoid
overlap double-counting); **negation window of 4 tokens**
("not raised guidance" flips bullish → bearish; "not just a
great quarter, the company posted record revenue" with too
many tokens between hedge and phrase doesn't flip — natural-
language negation is local); **score normalisation** via
`(positive - negative) / (positive + negative + 1)` — the +1
damper prevents single-hit saturation, keeps the result
bounded in [-1, 1], and produces near-zero for documents with
both strong positives AND strong negatives ("mixed signal" is
honest output, not picking a side). Threshold bands at ±0.15
keep the classifier from declaring a definite tilt on sparse
signal. The aggregator weights filings by hit count so a
filing with 4 strong-negative phrases dominates a filing with
1 mild-positive phrase — matches the practitioner intuition
that "going concern + material weakness + fraud + bankruptcy"
in one filing should not be diluted by a separate filing
mentioning "operating leverage". Halal alignment: read-only
signal generation; never opens a position; lexicons are
publicly checkable English phrases. Pure-Python (re + dataclass);
lexicons are module-level frozensets so a runtime mutation
can't tilt the classifier. 46 tests cover empty / whitespace /
no-match neutral pins, every lexicon class (positive / negative /
strong-weighted), the negation-window flip in both directions
(bullish → bearish AND bearish → bullish, so the symmetry is
pinned), the negation window's locality (4-token cap regression-
tested with a "far-away hedge doesn't fire" case), longest-
match-first invariant, the bounded-score guarantees under
extreme repeats, balanced-text-near-zero pin, threshold band
edges, every PhraseHit attribution field (phrase / polarity /
token_index / negated flag), document-order hit list, score-
formula sanity (`0.5` for one default-weight hit), the
aggregator's confidence weighting + per-label counts +
empty-input safety, frozen-dataclass immutability across all
three output types, render output across the three labels, and
tokeniser pinning (case-fold matching, punctuation stripped,
apostrophes preserved). EDGAR adapter (subscribe to the existing
EDGAR feed in `trading/scheduler.py`, classify the 8-K / 10-Q
text body, surface as a per-symbol prompt-context line) deferred
to a follow-up — classifier verified in isolation first.

### 4.I — Anomaly detection on equity curve  ✅ landed (detector core)

Existing `ml/anomaly.py` detects market anomalies. Add a *reverse*
anomaly: detect when *our own* P&L is anomalous (sudden drawdown,
sudden hot streak). Auto-halt + alert; "you're either getting
unlucky or your edge has eroded — investigate".

**Landed (detector core):** `ml/equity_anomaly.py` exposes
`detect_return_anomaly(returns)` and `detect_drawdown_anomaly(curve)`,
both returning an `EquityAnomalyReport` with z-score, severity
(`normal` / `warn` / `alert` at |z| ≥ 2 / 3), direction (`drawdown`
/ `hot` / `normal`), window stats, and a one-line operator-readable
recommendation. Per-trade detector: z-scores `returns[-1]` against
the prior window. Drawdown detector: z-scores the current peak-to-
trough drawdown against the historical drawdown distribution on the
same curve (drawdown is path-dependent, so it surfaces a different
class of anomaly than per-trade returns — a string of small losses
won't trip the return detector but can drive a multi-week trough
the operator must see). Hot-streak alerts on the drawdown side are
suppressed (drawdown is bounded above by 0). Cold-start guard
returns `severity = "normal"` below `min_window` so a fresh bot
can't false-fire on its first trade. NaN / inf entries stripped;
non-positive equity values fall back to `normal` rather than NaN-ing
out. `equity_curve_from_returns(rs, starting=…)` is a helper for
building curves from per-trade returns. Pure NumPy + math; no scipy
needed. 26 tests cover the cold-start contract, every severity band,
both directions, the bounded-drawdown semantic, the empty / short /
zero-std / non-positive-equity / threshold-validation edges, plus
an end-to-end "simulated blow-up" round trip. Wiring (auto-halt on
`alert` + Telegram / Slack / Discord notification) deferred to a
follow-up; detector verified in isolation first.

### 4.J — Live-shadow runner with model voting  ✅ landed (committee)

The existing `ShadowRunner` runs an alternate strategy in parallel
without trading. Extend: each strategy variant (GPT-4o, Claude,
Llama, RL policy, GA-evolved) runs in shadow simultaneously. The
"committee vote" decides the live trade. Operators see per-variant
attribution in the dashboard.

**Landed (committee):** `core/committee.py` ships `vote(decisions,
*, policy, weights)` taking a list of per-pair `VariantDecision`s
and returning a `CommitteeVerdict` with the chosen action +
per-variant attribution. Three voting policies share the
conservative-tiebreak philosophy used elsewhere (Wave 2.B / 4.F):
**MAJORITY** (most-common action; ties → most conservative);
**WEIGHTED** (per-variant weight sums tied to rolling Sharpe;
ties → conservative; negative weights clamp to 0 so a buggy
config can't invert); **UNANIMOUS** (every variant must agree on
a non-HOLD action, else HOLD — strictest mode for highest-
conviction trades). Tiebreak ordering pinned: HOLD > SELL > BUY
(refusing to open new risk on a split is the safer default).
Two asymmetric aggregations matter: BUY confidence = **MIN**
across agreeing variants (a 3-of-5 BUY where the lowest agreeing
variant says 0.4 should size off 0.4, not 0.9 — opening risk
needs the strictest test); SELL/HOLD confidence = **MEAN** across
agreeing variants (closing risk benefits from averaging). BUY
quantity = **MEDIAN** across agreeing variants (robust to one
outsized-bet outlier); SELL/HOLD quantity = 0 (executor handles
the close based on the open position). Empty decisions list →
HOLD with zero confidence (a no-variants cycle where every model
failed must not promote a blank vote into BUY/SELL). `VariantAttribution`
on each verdict surfaces per-variant action + confidence + assigned
weight + the `agreed_with_committee` flag so the dashboard can
render "this trade was 4-for-1 BUY (GPT-4o + Claude + Llama + GA
voted BUY; RL voted HOLD)" and operators can spot reliably-aligned
vs contrarian variants over time. `render_verdict` produces an
emoji-prefixed (🟢 / 🔴 / 🟡) text payload visually consistent
with the other Round-4 render helpers. Halal alignment baked in:
voting is pure aggregation; never opens a position; halal screener
still gates every candidate before the committee sees it. Pure-
Python (no NumPy / DB / async / LLM); operates on plain dataclasses
so the committee can be tested without the full strategy stack.
33 tests cover input validation (confidence bounds, negative
quantity rejection, JSON-string action coercion, unknown-action
rejection), empty-input HOLD safety, all three voting policies
across pass / tiebreak / unanimous-agree / unanimous-disagree-
holds, weighted-default-1.0 + negative-clamp + zero-total-HOLD
fallback, the BUY-min-confidence + SELL-mean-confidence asymmetry,
median-quantity-robustness vs outliers, SELL-quantity=0 pin,
attribution per variant + agreement marking + carries-weight, the
`agreement_count` and `total_variants` properties, immutability,
and render output across every action type. Cycle-side wiring
(parallel ShadowRunners per variant + per-cycle vote + executor
gates on the verdict + per-variant attribution row in the audit
trail) deferred to a follow-up — committee verified in isolation
first.

**Acceptance**: An out-of-sample backtest over 2025-Q4 shows the
new ensemble beats the round-3 LLM-only baseline by ≥30% on Sharpe.

---

## Wave 5 — Operator UX excellence (4-6 weeks)

Transform the dashboard from "useful" to "delightful".

### 5.A — Real-time live-cycle replay timeline  ✅ landed (aggregator)

The existing `/ws/cycle` streams stage events. Build a
visual timeline: "stage start → broker call (240ms) → strategy.analyze
(2.1s) → execute (180ms) → done". Operators can replay any historical
cycle from the replay store with millisecond-resolution stage timings.

**Landed (aggregator):** `core/cycle_timeline.py` ships
`build_timeline(events)` → `CycleTimeline`. Takes a flat list of
`StageEvent`s (start/end pairs from the existing
`cycle.stage.start` / `cycle.stage.end` bus topics) and returns
the structured view with: per-stage `StageRun`s
(start/end/duration/status/error/attrs), top-N bottlenecks ranked
by descending duration, total wall-clock duration, error and open-
stage counts, plus a pre-rendered markdown body suitable for the
dashboard's "click any historical cycle" tile and CLI output.
Three semantic pins: (1) duration prefers the END event's
`elapsed_ms` (the pipeline measures it with `time.monotonic()`)
over wall-clock derivation (which can be skewed by bus-publish
delay); falls back to wall-clock for legacy events without
`elapsed_ms` rather than reporting None; (2) a START without a
matching END is reported as `status="open"` rather than dropped —
operator wants to see *that* the cycle was killed mid-stage X, not
a timeline that silently omits the killer; (3) an orphan END (live
bus dropped the START under backpressure) synthesises a zero-
wall-clock run with the elapsed_ms duration so it's not lost
either. Bottleneck ranking excludes open stages (unknown duration
shouldn't fake a 0ms entry). `StageEvent.from_bus_payload(topic,
payload, *, at)` adapts the loose-dict bus payload to the typed
event; an unrecognised topic raises immediately so a wiring bug
surfaces rather than silently producing an empty timeline. Pure-
Python; no DB / async / NumPy. Frozen dataclasses safe to cache.
28 tests cover the start/end pairing, the duration source
preference (elapsed_ms → wall-clock), the open-stage and
orphan-end recovery paths, the duplicate-start defensive flush,
the error-status propagation, the bottleneck ranking + top-5 cap
+ open-stage exclusion + pct-of-total ≤ 100% sanity, the
`from_bus_payload` adapter (start/end topic detection, attr
stripping, unknown-topic rejection), the markdown sections
(total / health / bottleneck table / per-stage error and open
markers), and the immutability invariant. WebSocket plumbing and
dashboard tile deferred to a follow-up — aggregator is verified
in isolation first.

### 5.B — Strategy A/B comparison view  ✅ landed (backend math)

Side-by-side performance comparison of two strategies (or two prompt
versions) over the same time window. Sharpe, win rate, max drawdown,
return. Statistical significance test on the difference.

**Landed (backend math):** `core/ab_compare.py` exposes
`cohort_stats(returns)` → `CohortStats` (n_trades, win_rate, mean /
median / std return, Sharpe, max drawdown, total compound return,
profit factor) and `compare(a, b)` → `ABComparison` (per-cohort
stats + Welch's t-statistic + Welch-Satterthwaite degrees of
freedom + two-tailed p-value + `significant_at_05` boolean +
`mean_diff` direction). Uses `scipy.stats.t.sf` when available for a
precise p-value; degrades to the standard-normal SF when df ≥ 30
without scipy; returns `p_value = None` for the small-df + no-scipy
combination so the dashboard renders "unknown" rather than a wrong
answer. Pure-NumPy core; no `[ml]` extra needed. NaN / inf entries
are stripped silently (closed-trade rows occasionally have a missing
`return_pct`). 30 tests cover every helper and the integration paths
including a deterministic "doesn't false-flag iid samples" run and a
detect-significance run on cleanly-separated distributions. SQL-side
wiring (a `/api/strategies/compare?a=...&b=...` route that pulls
returns from `LLMDecision.prompt_version` cohorts) is deferred to a
follow-up — math first, plumbing second.

### 5.C — Trade rationale explorer  ✅ landed (search index)

Click any trade → see the full LLM reasoning, the indicator vector,
the regime label, the RAG hits, the adversarial review, the ensemble
votes. Search by reasoning ("show me trades that mentioned MACD
divergence").

**Landed (search index):** `core/rationale_search.py` ships
`RationaleIndex` — an in-memory TF-IDF index over LLM rationales
with sub-millisecond search letting the dashboard scrub queries
live without a DB round-trip per keystroke. Picked a hand-rolled
TF-IDF over Postgres `tsvector` because the corpus is small
(≤ 100k decisions over a year), the dashboard's explainability
requirement ("why did this rank first?") needs per-token
contributions exposed (easier when we own the scoring path), and
pure-Python keeps the index testable without a database. Three
layers: (1) **Tokenisation** — case-folded, word-boundary split,
drops short tokens (< 2 chars) and a deliberately tiny stop-word
list (only the most common contentless tokens — operators search
for "the market" or "is overbought" and removing too many stop
words kills phrase recall); (2) **TF-IDF scoring** — sub-linear
TF dampening `1 + log(tf)` (a rationale that mentions MACD 10×
doesn't score 10× a rationale that mentions it once — pinned
ratio < 5×), smoothed IDF `log((N+1)/(df+1)) + 1` so universal
terms still contribute a positive score (zero IDF would make
those invisible); (3) **per-query-term contributions** exposed on
each `SearchHit` via `TermContribution(term, tf, idf,
contribution)` so the dashboard renders a stacked bar
explanation. `search(query, *, limit, require_all)` supports
both OR-style (default) and AND-style filtering. Three pinned
behaviours: query terms are de-duplicated (a typo'd `"MACD
MACD"` doesn't double-score); stable tie-break preserves input
order so older trades appear first when scores tie; an all-stop-
word query returns no hits rather than every doc at score zero.
Snippet builder centres on the first match position so operators
see the matching context, not the start of every doc; truncates
long text at ~200 chars; falls back to a head snippet when the
literal substring doesn't appear (the tokeniser may have stripped
it). Append-only invariant pinned: duplicate `doc_id` raises
rather than overwriting (the audit trail's `LlmDecision` row,
once written, is the record); operators rebuild via `clear()`.
Halal alignment: pure read-only search — never opens a position,
never bypasses the screener. Pure-Python (no numpy / scipy /
DB / async). 39 tests cover tokenisation (case-fold, stop-word
drop, length filter, punctuation split, alphanumeric tokens
preserved), index lifecycle (add / clear / get_doc / duplicate
rejection), basic search (matching, no-match, empty index, all-
stop-word query rejection, limit validation), ranking (higher TF
ranks above lower; rare-term IDF dominates over common terms;
sub-linear TF cap; per-term contributions match TF; query
de-duplication; stable tie-break; limit respected), `require_all`
filter (filters docs missing a term + default-False OR semantics),
snippet generation across match-at-start / match-at-end / centred
match / long-text truncation / empty-doc edge, metadata pass-
through, output structure, render output, and numerical sanity
(IDF smoothing keeps universal terms positive; TF dampening
matches the documented `1 + log(tf)` formula directly via the
exposed contribution field). Dashboard wiring (subscribe to
`LlmDecision` rows, build the index on dashboard mount, expose a
`/api/rationale/search?q=...` endpoint) deferred to a follow-up —
index verified in isolation first.

### 5.D — Per-trade lessons-learned card  ✅ landed (renderer)

When a trade closes (especially a loss), auto-generate a "lessons
learned" card: what the LLM said, what actually happened, the
indicators at exit, what the regret model says about an alternative
sizing. Operators can star cards for review.

**Landed (renderer):** `core/lessons_card.py` exposes
`render(LessonCardInput)` → `LessonCard` — a deterministic, pure-
Python post-mortem that runs without an LLM call so the dashboard
isn't bottlenecked behind the per-trade-close queue. Heuristic
classifier produces six verdicts: `winner_thesis_intact`,
`winner_lucky` (entered at RSI ≥ 70 or BB upper), `winner` (no exit
snapshot), `loser_thesis_invalidated` (large indicator delta
between entry and exit), `loser_noise` (small delta — random walk),
`loser` (no exit snapshot), plus `unknown` for not-yet-closed rows.
Six lesson heuristics emit 0–3 one-sentence operator nudges
(capped — pinned so future additions can't produce a wall of text):
high-confidence loss with no regime change → review conviction
signal; extended-entry winner → size down for over-extended
entries; stop-loss exit on RSI ≥ 70 entry → filter buys above
RSI 70; trailing-stop winner → don't loosen the trailing distance;
low-volume loser → gate entries on volume ≥ 0.8×; high-volume
winner → confirm volume remains a useful filter. Output includes
indicator dicts (entry + optional exit + per-key delta), the
verdict, the lessons list, and a pre-rendered markdown payload
suitable for the Slack / Discord notifiers (so they don't have to
re-walk the structure). 26 tests cover every classifier branch,
each lesson heuristic firing in isolation, the lessons-≤-3 cap, the
markdown emoji / pair / verdict / rationale-truncation contract,
and the partial-input degradation cases (legacy trades without
snapshots still produce a useful card). SQL adapter + LLM
"expert commentary" layer deferred to a follow-up.

### 5.E — Voice control via local LLM ✅ landed (`web/voice_intent.py`) — intent classifier

Operator says "stop the crypto bot" / "what's my drawdown today" /
"halt and close all positions" — local Whisper + gpt-oss handles it.
Operator-only feature, off by default. A surprise-and-delight feature
that pays off during incident response.

**Implementation**: `src/halal_trader/web/voice_intent.py` ships
the pure-Python intent classifier. Whisper STT and gpt-oss
disambiguation are operator-side; this module ships the
deterministic grammar over the transcribed text. `VoiceIntent`
enum (HALT / RESUME / STATUS / DRAWDOWN_QUERY / HALT_AND_CLOSE_ALL
/ UNKNOWN). Module-level `_INTENT_PATTERNS` tuple is the
priority-ordered grammar — HALT_AND_CLOSE_ALL checked BEFORE HALT
so "halt and close all positions" doesn't accidentally match the
simpler HALT intent (the load-bearing priority pin). Token-set
matching after lowercase + punctuation-strip; "stop the bot",
"kill the bot", "pause the bot", "halt trading" all match HALT.
"halt" alone (without bot/trading target word) returns UNKNOWN —
guards against accidental triggering on a fragment. `classify_intent`
returns UNKNOWN for empty/whitespace input rather than guessing.
`_DESTRUCTIVE_INTENTS` frozenset ({HALT, HALT_AND_CLOSE_ALL}) drives
the confirmation gate. `IntentRecognition` carries
recognition_id + intent + raw_text + recognized_at + expires_at
(default 10s window) + optional confirmed_at. `recognize` builds
a recognition; `confirm_recognition` requires destructive intent +
unconfirmed + within window (boundary inclusive at expires_at);
non-destructive raises NotDestructiveError; expired raises
ConfirmationExpiredError. `is_executable(recognition, *, now)`:
UNKNOWN never; non-destructive immediately; destructive requires
confirmed_at + within window. Render with intent emoji
(🛑/▶️/📊/📉/🚨/❓) + DESTRUCTIVE marker for unconfirmed +
confirmed marker after. No-secret regression: no audio paths /
device IDs in render. Tests in `tests/test_voice_intent.py` (73
cases): VoiceIntent enum string-value pins; classify HALT across
6 phrasings (halt/stop/kill/pause + bot or trading target);
HALT_AND_CLOSE_ALL across 4 phrasings including "emergency exit"
+ "liquidate everything"; **HALT_AND_CLOSE_ALL takes priority
over HALT pin** ("halt and close all" classifies as the more
destructive intent, not the simpler one); RESUME/STATUS/DRAWDOWN_QUERY
synonym sets; UNKNOWN for empty/whitespace/random/fragment;
case-insensitive matching; punctuation-strip ("halt!" matches);
is_destructive across all 6 intents (HALT + HALT_AND_CLOSE_ALL
yes; RESUME/STATUS/DRAWDOWN_QUERY/UNKNOWN no — RESUME explicitly
non-destructive since recovery actions don't need confirmation);
IntentRecognition validation (empty fields rejected; naive datetimes
rejected; expires-before-recognized rejected; immutable);
recognize default 10s + custom window flow; confirm_recognition
within window; **boundary inclusive at expires_at**; past-window
raises ConfirmationExpiredError carrying recognition_id;
non-destructive raises NotDestructiveError; already-confirmed
rejected; is_executable across UNKNOWN false / non-destructive
true / destructive-unconfirmed false / destructive-confirmed-
within-window true / destructive-confirmed-past-window false;
render with intent emoji + DESTRUCTIVE marker for unconfirmed +
confirmed marker; render no-secret regression; e2e flows (status
runs immediately; HALT requires confirmation then executes after
3s; emergency exit classified as HALT_AND_CLOSE_ALL; misheard
"what time is it" returns UNKNOWN; replay consistency).

### 5.F — Scenario simulator  ✅ landed (core)

"What if the Fed surprises hawkish tomorrow?" — operator picks a
canned scenario from `crypto/stress.py`, the simulator runs it
against current positions and shows projected drawdown / fills. Lets
operators manually de-risk before known events.

**Landed (core):** `core/scenario_sim.py` exposes
`simulate(positions, klines)` → `ScenarioReport` with per-position
`PositionProjection`s (filled / fill_reason / fill_bar_index /
fill_price / start / end / min / max equity) plus aggregate
portfolio P&L and worst-case drawdown. Path-aware fills: SL/TP
checked against the bar's high/low (not just close), matching the
real position monitor's wick-fill semantic. Pinned tiebreak: when
both SL and TP could fire in the same bar (volatile gap bars), SL
wins — worst-case execution is the safer projection for an operator
deciding whether to de-risk. Trailing-stop ratchet matches
`core/sl_tp.py`: trail observed at end of bar, fired next bar
(prevents the "phantom same-bar trailing fill" bug where a high
ratchets the SL inside the bar that the SL itself recovers from).
Operates on `SimulatedPosition` dataclass (pair / qty /
entry_price / stop_loss / take_profit / trailing_stop_pct) so the
simulator is free of DB / domain imports — caller maps live
`CryptoTrade` rows or `BrokerPosition` records into the shape.
`render_report` produces a CLI / Slack-ready text payload visually
consistent with `crypto/stress.render_report`. 20 tests cover: empty
positions / empty klines edges, SL fires on wick + on subsequent
bar, TP fires on wick, SL-wins-on-tie invariant, trailing ratchet
fires next bar / doesn't fire if bar holds above trail / combines
correctly with TP, min/max equity tracking, portfolio aggregation
arithmetic, end-to-end runs against the existing `flash_crash_klines`
+ `gap_down_klines` generators (verifying the simulator + stress
modules interop), and frozen-dataclass immutability. Dashboard
button + CLI command (`halal-trader scenario run --scenario
flash_crash`) deferred to a follow-up.

### 5.G — Slack / Discord integration  ✅ landed

Beyond Telegram: enable trade alerts + daily summary delivery to
Slack workspaces or Discord servers. Per-channel routing (production
fills → #trading, errors → #alerts).

**Landed:** `notifications/webhooks.py` ships `SlackNotifier` +
`DiscordNotifier` mirroring the `TelegramNotifier` surface
(`enabled`, `send`, `notify_trade`, `notify_sl_tp`,
`notify_daily_summary`, `notify_buzz`). Both speak JSON-over-HTTPS
incoming-webhook protocols — operator pastes a URL into
`SLACK_WEBHOOK_URL` / `DISCORD_WEBHOOK_URL`, no SDK or OAuth.
Placeholder / non-https URLs fail the `enabled` check so a
half-configured `.env` cannot silently send to a fake endpoint.
36 tests with `httpx.MockTransport`. `SlackSettings` + `DiscordSettings`
under the existing pydantic-settings convention.

### 5.H — Per-position "story view"  ✅ landed (aggregator)

Hold AAPL → see a single page with the entry rationale, every
indicator since entry, every news event for AAPL, the current
unrealized P&L tracked over time, the SL/TP levels. Tells the
story of the trade.

**Landed (aggregator):** `core/position_story.py` ships
`build_story(PositionStoryInput)` → `PositionStory`. Input bundles
the trade row + entry indicator snapshot + indicator timeline +
price timeline + news events; output is the structured story plus
a pre-rendered markdown narrative. Computes the unrealized-P&L
curve from each price observation (pct + USD), the per-indicator
delta between entry and latest snapshot (skips fields measured on
neither side; emits a partial row with `delta=None` when one side
is missing — fewer rows beats a wall of n/a, but "we measured this
once" must still be visible), and bullish / bearish news counts
(score-sign-based; neutral score=0 deliberately doesn't double-count).
Markdown sections: header with green / red emoji, entry / current /
risk levels (SL / TP / trailing) line, LLM rationale (truncated at
400 chars), indicator-drift table, news section ordered most-
recent-first capped at 5 with bullish ▲ / bearish ▼ markers and
optional URL link, P&L track section calling out trough / peak /
latest. Defensive pins: empty price timeline → no current P&L
(dashboard renders a "no data yet" state cleanly); zero / negative
entry price → empty P&L curve (corrupt row doesn't NaN-out the
dashboard); legacy positions without `llm_reasoning` skip the "Why
we entered" section without crashing. Pure-Python; no NumPy / DB /
async. Frozen dataclasses safe to cache. 32 tests cover P&L curve
arithmetic and edges (empty / negative / zero-entry), indicator-
delta rules (entry-only, latest-only, both, neither), news
classification including the neutral-score pin, every markdown
section's content, the long-position smoke test (168 hourly ticks),
and the immutability invariant. SQL fan-in (subscribe to
`crypto_trades` + `indicator_snapshots` + news + kline ticks +
`llm_decisions` for a given pair) deferred to a follow-up; the
aggregator is verified in isolation first so the wiring layer sees
a stable contract.

### 5.I — End-of-day operator email summary  ✅ landed (renderer)

Shipped `notifications/email.py` with the pure-Python renderer
that powers the daily / weekly digest. `render_email(input)`
returns a `RenderedEmail` (subject + HTML body + plaintext twin)
the SMTP layer hands to SES / SendGrid / direct SMTP (sender
extension TBD; renderer is already the bulk of the work and is
testable with no I/O).

Sections: P&L (today / month / quarter), trades roster (recent N
fills with side / qty / price / status), halal-compliance banner
(driven by the same `AAOIFISummary` the dashboard tile uses —
single source of truth), purification status, risk snapshot
(drawdown + heat + halt banner), notable-events list. Inline
styles only (every email client interprets `<style>` differently;
inline is the only thing reliable across Gmail / Outlook / Apple
Mail).

**Inbox-triage discipline**: the subject line emoji (🟢 / 🟠 / 🔴)
mirrors the AAOIFI status so a violation is visible from the
inbox preview, not just inside the email. Pinned by the test
suite — any future status that doesn't get a corresponding emoji
falls back to ⚪ rather than silently rendering as compliant.

Adapter `summary_from_aaoifi(...)` builds an `EmailSummaryInput`
straight from the AAOIFI summary + per-call overrides — keeps the
halal slice (status / non-halal fills / purification numbers) in
sync with the dashboard tile.

38 new tests in `tests/test_notifications_email.py` pin: subject
emoji per status, negative-P&L sign rendering, every-section
present in HTML + plaintext, P&L colour green/red on positive/
negative, halt banner only when halted (with `(no reason given)`
fallback when the reason was lost), trades table with both
`symbol` (stocks) and `pair` (crypto) keys, purification amounts
visible, risk-section omitted gracefully when no data, notable
events list, operator-name greeting, thousands-separator money
format, defensive fallback for an unknown `compliance_status`,
plaintext line-by-line trade rendering, frozen `RenderedEmail`,
adapter passes through `trades_today` / period label / purification.
mypy clean.

### 5.J — Dark mode + accessibility audit ✅ landed (`web/accessibility_audit.py`)

Existing dashboard is light-themed only. Add dark mode + run a full
WCAG AA audit (alt text, color contrast, keyboard nav). Halal traders
work odd hours; dark mode is non-negotiable.

**Acceptance**: An operator can identify the cause of a losing trade
in under 30 seconds via the dashboard, without grepping JSON logs.

**Implementation**: `src/halal_trader/web/accessibility_audit.py`
ships the pure-Python WCAG AA audit engine. Five `WcagCriterion`
literals pinned (1.1.1 alt text / 1.4.3 contrast / 1.4.11 non-text
contrast / 2.1.1 keyboard / 2.4.7 focus visible). `Severity` (ERROR
/ WARN) and `Theme` (LIGHT / DARK) enums. `relative_luminance`
implements the WCAG sRGB luminance formula (white → 1.0, black →
0.0); `contrast_ratio` returns the 1.0-21.0 ratio. `TextComponent`
carries `light_foreground`/`light_background`/`dark_foreground`/
`dark_background` plus `is_large_text` and `is_ui_component` flags;
hex color validation rejects `black` / `#FFF` / unprefixed hex.
`ImageComponent` with `decorative=True` skips the alt-text check.
`InteractiveComponent` requires both `keyboard_reachable=True` and
`focus_indicator_visible=True` to pass. `audit(*, text_components,
image_components, interactive_components)` returns `AuditReport`
with `violations` sorted by `(criterion, component_id)` for
deterministic ordering; the `passed` and `has_errors` properties
let callers gate CI. Boundaries: 4.5:1 inclusive for normal text
(#767676 on white passes; #888888 on white fails), 3.0 for large
text, 3.0 for UI components per 1.4.11. Both LIGHT and DARK themes
audited from the same component manifest — a component bad in
either theme surfaces both violations. Tests in
`tests/test_accessibility_audit.py` (55 cases): WCAG criterion +
severity + theme enum string-value pins; luminance pure-white=1
pure-black=0 + bad-format rejection (`black`, `#FFF`, missing `#`)
+ lowercase-hex acceptance; contrast white-on-black=21
same-color=1 symmetry + known-value smoke check; component
validation across all three component types; Violation field
validation; contrast threshold pins both directions (4.5 inclusive,
just below fails); large text uses 3:1; UI component routes to
1.4.11; dark theme failures surface; both-theme failures surface
twice; alt-text required for non-decorative images including
whitespace-only rejection; decorative=True bypasses; keyboard +
focus pins independent + combined; report aggregates audit_counts;
deterministic ordering; render output (passed report shows ✅,
failed shows ❌, severity emoji, theme markers, audit count); e2e
realistic clean dashboard passes; e2e dashboard with multiple
violation types surfaces all three criteria together.

---

## Wave 6 — ML infrastructure (3-5 weeks)

Move from "we run some ML models" to a proper MLOps stack.

### 6.A — Model registry with versioning  ✅ landed (policy)

`db.ml_artefacts` is a start. Add semantic versioning (v1.2.3),
model lineage (which training run produced this model), automatic
A/B between versions before promote. Hooks into the existing
`prompt_evo` pattern.

**Landed (policy):** `ml/registry.py` ships `Semver` (parse from
`v1.2.3` or `1.2.3`, total ordering via `dataclass(order=True)`,
`bump_major / minor / patch`, `is_compatible_with` checking
major-version parity), `ModelLineage` (parent_version,
training_run_id, fitness_score, fitness_metric_name, notes),
`RegistryRecord` (name + version + lineage + payload_kind +
size_bytes + is_active flag), `lineage_chain(records, *, target)`
that walks parent pointers root→target with cycle and
missing-parent rejections (pin: a corrupt registry must surface
immediately, not silently truncate), and `should_promote(*,
candidate, incumbent, policy)` that composes three checks:
compatibility (major-version parity, blocked by default — operator
opts in for major-version cutovers), lineage continuity (candidate's
parent must equal incumbent's version — stops a sideloaded model
short-circuiting the chain), and fitness uplift (candidate must
exceed incumbent by `min_fitness_uplift`, default 0.05 Sharpe).
Cold-start is handled: when incumbent is `None`, any candidate
with a fitness_score promotes; partial-bootstrap (incumbent has
no fitness) promotes on candidate's score alone. `render_lineage`
produces a CLI-friendly chain summary with ★ marker for the
active record. Picked a separate module rather than extending
`ml_artefacts.py` because that's persistence (SQL/pickle/JSONB)
while this is policy (semver/lineage/promotion) — keeping them
apart means a future SQL refactor (e.g. switch to S3 + manifest)
doesn't ripple into the policy contract; the registry is also
pure-Python so its tests run without Postgres. 39 tests cover:
semver parse with/without v-prefix + whitespace tolerance + hard
rejection of malformed strings (no silent (0,0,0) rounding), tuple-
ordering comparisons, all three bump methods preserving the
frozen-dataclass invariant, compatibility predicate, RegistryRecord
input validation (payload_kind / negative size_bytes), lineage_chain
walks (root, multi-step, name-filtering, missing-parent reject,
cycle detection, duplicate-version rejection), should_promote across
every check (cold-start with/without fitness; major-bump blocked
by default + opt-in via policy; lineage-continuity reject + opt-out;
fitness-uplift below/above/negative; partial-bootstrap incumbent
without fitness; missing candidate fitness), and render_lineage
(empty chain, active marker, missing fitness fallback). SQL adapter
that converts existing `ml_artefacts` rows to `RegistryRecord`s and
the dashboard's lineage tile deferred to a follow-up — policy
verified in isolation first so the persistence layer sees a stable
contract.

### 6.B — Feature store  ✅ landed (schema + migration)

Materialised feature pipelines (RSI / MACD / etc.) versioned alongside
models. When a feature is renamed or its computation changes, the
feature store handles the migration without breaking inference.

**Landed (schema + migration):** `ml/feature_store.py` ships
`FeatureSpec(name, dtype, description, producer, required)` and
`FeatureSchema(name, version: Semver, features)` (versioning reuses
the Wave 6.A `Semver` so operators only learn one rule set), plus
four declarative migration rules — `Rename(from_name, to_name)`,
`Drop(name)`, `AddDefault(name, default_value)`, `Cast(name,
target_dtype)` — composed via `MigrationPlan(schema_name,
from_version, to_version, rules)`. `migrate(payload, *, from_schema,
to_schema, plans)` finds the matching plan and applies its rules in
order. `validate(payload, schema)` checks required-feature presence
and dtype compatibility. Three asymmetric-but-pinned dtype rules:
FLOAT accepts int (auto-promotion is a common numpy interop case);
INT rejects bool (a True passed where INT was expected is almost
certainly a bug); BOOL rejects int (5 silently becoming True is a
real bug). Cast rule pins three "no silent failures" semantics:
float→int rejects lossy values (1.7 won't silently become 1),
to-bool rejects ambiguous values (only exact 0/1/True/False
accepted; "yes" / 5 surface immediately), every cast wraps the
underlying error rather than swallowing. Rename rule rejects
no-ops (from == to is a config bug) and collisions (target already
present would silently overwrite). Drop is idempotent (no-op if
feature absent — chains compose). AddDefault never overwrites.
`migrate` rejects schema-name mismatches (cross-schema migration
is nonsense), missing plans (no silent inference), and ambiguous
plans (two plans for the same versions = config bug). 41 tests
cover dtype matching across the bool/int/float asymmetry, schema
duplicate-name rejection, every migration rule's pass + edge +
rejection paths, MigrationPlan rule-order pin, the migrate
driver's plan-lookup contract including same-version copy + no-mut
invariant + name-mismatch + missing + ambiguous rejections,
validate across required + optional + dtype-mismatch + extras,
and render_validation output. SQL-side persistence (a
`feature_schemas` table mirroring `ml_artefacts`) and the
cycle-side wiring (validate before inference, migrate on schema
drift) deferred to a follow-up.

### 6.C — Online learning for high-frequency signals  ✅ landed (engine)

Some signals (orderbook imbalance, basis) have too short a half-life
for batch retraining. Add an online-learning path: stream-fit a
linear model on the last N samples, reweight every minute.

**Landed (engine):** `ml/online_learner.py` ships `OnlineLearner`
with `update(features, target)` + `predict(features)` + `reset()`.
Closed-form weighted ridge regression on a rolling buffer (deque-
backed, default window=200) re-fitted on every update — the hot
path is O(N × K²) and stays sub-millisecond for K ≤ 20 features.
Picked closed-form refit over an online RLS update because the
implementation stays one-screen-of-numpy auditable; with the small
windows the cycle uses, the constant-factor difference is
invisible. Two operating modes: **plain ridge** (decay=1.0) and
**exponentially-weighted ridge** (decay < 1, samples weighted by
`decay**(age)`). Three pinned safeties: feature_clip + target_clip
bound every input before fitting (a glitched feed sending 1e9
can't dominate the fit — clipping rather than dropping samples
keeps the buffer aligned with the time axis); a tanh squash on
`predict` bounds the output to [-1, 1] so a runaway coefficient
can't cause a size explosion downstream; `min_samples_for_predict`
returns 0.0 below the cold-start threshold so strategies treating
0 as "no edge" stay correct. Validation rejects non-finite
features / target, wrong feature count, non-positive
ridge_lambda / window / clip, and decay outside (0, 1]. `reset()`
clears the buffer + zeros the coefs (use on a regime-change
detection event when the learner should start over rather than
slowly forget). `LearnerSnapshot` exposes coef + intercept + RMSE
+ sample_count + last_prediction + last_target for the dashboard
tile. Halal alignment: the learner emits a *signal* (the
prediction), never opens a position. Pure-numpy; no scipy / sklearn
/ DB / async. 29 tests cover config validation across every
parameter, the cold-start zero-prediction guard + above-threshold
behaviour, ridge fit recovering a known linear relationship and a
non-zero intercept, rolling-window eviction, end-to-end
regime-flip response (positive slope window → negative slope →
coef sign flips), exponential-decay speeding up regime response
vs plain ridge, decay=1.0 matching plain ridge exactly,
feature/target clipping bounding glitched-feed contributions
(test pins RMSE < 1.0 even when target was 1e6), input-validation
rejections, the tanh-bounded output guarantee under runaway coefs,
the predict([0]) → tanh(intercept) sanity, reset clearing buffer
and coefs (predict returns 0 again afterward), the
last_target/last_prediction snapshot updates, and frozen-dataclass
immutability on `LearnerSnapshot`. Cycle-side wiring (instantiate
per fast-half-life signal, feed orderbook imbalance / basis /
sentiment-velocity samples, multiply the LLM plan's confidence by
the learner's signal) deferred to a follow-up — engine verified
in isolation first.

### 6.D — Continuous integration for ML  ✅ landed (orchestrator)

Every model commit triggers: walk-forward backtest, Sharpe regression
check, drift comparison with prod model. Block promote-to-live if any
metric regresses by > 10%.

**Landed (orchestrator):** `ml/ci_pipeline.py` ships `run_ci(...)`
composing four gates against an incumbent baseline: (1)
**walk-forward** via Wave 4.F's `evaluate_promotion`; (2) **Sharpe
regression** — candidate Sharpe must be ≥ 90% of incumbent (the
roadmap's "no >10% regression"); (3) **win-rate regression** —
symmetric guard at 95% (tighter floor because win-rate is more
sensitive to small-sample variability); (4) **distribution
drift** — KS-style max-absolute-CDF-distance between the
candidate's and incumbent's per-trade-return distributions, capped
at 0.30. The KS distance is computed without scipy: at each
unique observation, evaluate both empirical CDFs and take the
max absolute difference; the bot's heavy-tailed returns don't fit
a parametric distribution well, so non-parametric is right. Pin:
**skipped checks count as PASS** — when there's no incumbent
(cold start), the regression checks return passed=True with a
`skipped: …` remediation note that the dashboard surfaces with
a `—` marker; operators running a fresh model don't get gated.
Same pattern for sample-size-too-small drift (KS on 10 points is
mostly noise) and non-positive incumbent Sharpe (no real baseline
to compare against). Aggregate `passed` is True iff every gate
passed (skip counts as pass via the `is_skipped` property which
is only True when passed=True AND the remediation starts with
`skipped`). Picked a separate orchestrator from
`promotion_gate.py` because the live-trading gate runs against
absolute thresholds while CI runs against the incumbent — they
share threshold values but compose differently. The Wave 4.F
`PromotionVerdict` is exposed on the report for operator drill-
in. 35 tests cover threshold validation, the KS-distance helper
(zero-on-identical, one-on-disjoint, empty handling, [0, 1]
bound), per-gate checks (pass/fail/skip across each), aggregate
behaviour (every-gate-passes → PASS, single-fail → FAIL,
3-skip-1-pass → PASS with summary mention, MC layered through),
the `is_skipped` precedence pin (failure with "skipped" substring
isn't a skip — only passed=True with the prefix), output
immutability, and render output (PASS / FAIL / ✘ / — for skipped
/ remediation arrow). CLI command (`halal-trader ml ci run`) and
dashboard tile deferred to a follow-up — orchestrator verified
in isolation first.

### 6.E — Deterministic replay with model fingerprints  ✅ landed (engine)

Every cycle's replay snapshot already records inputs. Add the model
fingerprint (hash of the weights) so a replay can rerun a cycle's
decision against the *exact* model that produced it. Critical for
debugging "why did this happen 3 weeks ago".

**Landed (engine):** `ml/fingerprint.py` ships `fingerprint_bytes`
(SHA-256 default + BLAKE2b-256 alternative for short payloads) and
`fingerprint_json` (canonicalises via sorted keys / no whitespace /
unicode-preserving UTF-8 before hashing — pin so a refactor that
changes dict insertion order or whitespace doesn't break
fingerprint equality between runs that ought to be identical).
`ModelFingerprint(digest_hex, algorithm, byte_count, captured_at,
short_tag)` — short_tag is the first 12 hex chars, prefix-aligned
with the full hex so the dashboard can correlate without
re-deriving. `matches(other)` checks all three key fields
(digest + algorithm + byte_count) so a freak partial-write
collision (matching hash but different size) surfaces as a
mismatch. `verify(actual, expected)` returns a
`VerificationOutcome` with a four-bucket `VerificationStatus`
(MATCH / HASH_MISMATCH / ALGORITHM_MISMATCH / BYTE_COUNT_MISMATCH)
diagnosed in precedence order — algorithm first (two artefacts
produced by different hashers can't be compared), byte count next
(diagnoses truncated/oversize cases distinctly), hash last (same
size, different bytes = blob change in the middle). The reason
string includes both fingerprints' short tags so a Slack alert is
self-contained. Pin: an unsupported hash algorithm raises rather
than silently using a default (a caller passing `"md5"` for
compatibility surfaces immediately). Pure-Python (stdlib `hashlib`
+ `json`); no NumPy / DB / network. The hash digest is computed
from the bytes / JSON the caller hands in, so the module never
opens a file by itself — every IO surface in the bot is owned by
its module; fingerprinting must not introduce a new one. 31 tests
cover: byte fingerprint stability (same input → same output across
calls), the SHA-256 known-constant pin (fingerprint of `b"x"`
matches the published hex), BLAKE2b alternative producing different
digests, byte_count recording, short_tag length + prefix invariant,
bytes-like inputs (`bytes` / `bytearray` / `memoryview`), non-bytes
input rejection, empty-bytes producing the published SHA-256
constant, the supplied `captured_at` flowing through, the
unsupported-algorithm rejection path, canonical_json key-sort +
whitespace-strip + unicode-preserve + nesting invariance,
fingerprint_json end-to-end dict-order invariance, every
VerificationStatus path including the precedence order
(algorithm → byte_count → hash), the hash-mismatch + byte-count-
mismatch diagnostic separation, the freak collision path (matching
hash + algorithm but different byte_count rejected by `matches`),
the render helper output, and immutability on both
`ModelFingerprint` and `VerificationOutcome`. Cycle-side wiring
(record fingerprint in the replay snapshot at decision time;
verify on replay against the registry) deferred to a follow-up —
engine verified in isolation first.

### 6.F — Real-time inference latency budget  ✅ landed (tracker)

Set a 200ms budget for the full ML inference path (forecaster +
anomaly + signal classifier). Profile + optimise; consider TorchScript
/ ONNX export. Slow inference makes the cycle late.

**Landed (tracker):** `ml/latency_budget.py` ships
`LatencyBudgetTracker` with per-stage `StageBudget(name, budget_ms,
soft_pct, min_samples)` declarations and a rolling sample window
(default 100) per stage tracking p50 / p95 / p99 (nearest-rank,
not interpolated — pin so the dashboard reads "the 95th-worst
sample we actually saw" rather than a synthetic point). Each
`observe()` returns a `StageObservation(name, budget_ms, status,
current_ms, p50_ms, p95_ms, p99_ms, sample_count, headroom_pct)`
with a four-way classification: cold-start (<min_samples) checks
only `current` vs hard budget so first-sample noise doesn't
alarm; steady-state flips RED if either current OR p95 crosses
the hard budget, AMBER if either crosses `soft_pct × budget`
(default 80%), GREEN otherwise. Pin: `record()` for an unknown
stage raises rather than silently creating one — a typo'd stage
name would otherwise hide samples in a never-checked bucket.
`aggregate(observations)` produces a `BudgetReport` with summed
budgets / current / p95 across the inference path plus an
`overall_status` taking the worst per-stage status (RED beats
AMBER beats GREEN — pin so a single RED stage flips the whole
report). `render_report` emits an emoji-prefixed text payload
visually consistent with the other Round-4 render helpers.
Halal alignment: the tracker is observability + load-shedding
only; a RED status causes the cycle to skip optional stages
(agentic tools, RAG) and fall back to the canonical signal —
never to *force* a trade through. Pure-Python (no NumPy /
DB / network); frozen output dataclasses + a mutable ring buffer
for the sample window. Hot-path `record()` is O(1) (deque
append + dirty bit); percentiles are computed lazily on `observe()`
read with a cache that invalidates on the next record. 34 tests
cover budget validation (positive budget, soft_pct strictly in
(0, 1), positive min_samples), cold-start GREEN (no samples) +
cold-start AMBER (single sample at 90% of budget) + cold-start
RED (single sample over budget), steady-state GREEN under
budget, AMBER when p95 crosses soft, RED when p95 crosses hard
even if current is fast (the consistently-slow stage case), RED
when current breaches even if p95 is fine, percentile invariance
on uniform windows + nearest-rank pin (p95 of 1..20 = 19), the
ring-buffer eviction caps history at the window size, headroom
positive under budget + negative over, unknown-stage rejection
on both record and observe, tracker construction validation
(non-positive window), the stages property listing declared
names, negative-latency rejection, observe_all parity with
declared stages, aggregate sums + worst-status-wins (GREEN+RED
→ RED, GREEN+AMBER → AMBER), None-current-ms handled by aggregate,
empty-aggregate → GREEN, render emoji + stage lines + empty
fallback + RED-stage marker, the `is_breaching` property aligns
with status, and frozen-dataclass immutability on both the
observation and the report. Cycle-side wiring (record per stage
during `_run_cycle_impl`, gate optional enrichment stages on
status) and dashboard tile deferred to a follow-up — tracker
verified in isolation first.

### 6.G — GPU support (optional)

When the ML stack grows, CPU inference is the bottleneck. Add CUDA
support via `device="cuda"` in `ModelHub`. Document the cost/benefit;
keep CPU as default.

### 6.H — Active learning loop  ✅ landed (selector)

Hard cases (LLM gave a confident BUY that became a 2σ loss) get
flagged for human review. Operator labels via the dashboard; labelled
examples flow into the next training run. Closes the loop on the
"trade post-mortem" data.

**Landed (selector):** `ml/active_learning.py` ships `score_case`
(per-trade priority calculator) and `select_top_n` (full triage
queue) over `TradeCase(trade_id, pair, predicted_return,
actual_return, confidence, age_seconds, indicator_outlier_score,
rationale)`. Four scoring components combine via tunable
`ScorerWeights`: **confidence × error** (high-confidence misses
score high; capped at 2.0 so a freak outlier doesn't dominate);
**sign disagreement** (predicted +5%, actual -3% is the canonical
"embarrassing wrong-direction" — fires only when both sides are
non-zero so a "no opinion" zero can't be flagged); **outlier**
(passes through the operator-supplied indicator outlier score
clamped to [0, 1] so un-normalised z-scores can't dominate);
**recency** (exponential decay with operator-tunable half-life,
default 7 days). Per-component contributions are exposed on the
`Priority` so the dashboard can render the *why* — "this case
bubbled up because it was a high-confidence sign disagreement",
not "because the algorithm said so". `_explain` picks the
dominant non-zero component and surfaces a one-line operator-
readable reason. `select_top_n` sorts descending with a stable
tie-break (older cases reviewed first when scores tie). Pin: the
selector triages, never auto-labels — operator review is the only
authority that produces a label flowing into training. Halal
alignment: the queue is informational; never opens a position.
Pure-Python; no numpy / scipy / DB / async. 34 tests cover
TradeCase + ScorerWeights validation (confidence in [0,1], age
≥ 0, weights non-negative, half-life positive), each scorer
component (confidence × error scales with confidence and
magnitude, capped; sign-disagreement zero-on-match + zero-when-
either-side-zero + scales with magnitude; outlier clamping above
1 and below 0; recency decays + half-life pin where t=HL → 0.5×
fresh score), the weighted total, the dominant-component
explanation including the no-signal-fallback, `select_top_n`
ranking + n > input-size + zero-n rejection + stable ties + empty
input, and the render queue (each case + score format + reason
arrow + empty fallback). Dashboard wiring (subscribe to closed-
trade events, push the top-N into a review panel, persist
operator labels back to the IndicatorSnapshot table for the next
retrainer cycle) deferred to a follow-up — selector verified in
isolation first.

### 6.I — Distillation: teach a small fast model from the LLM ensemble ✅ landed (`ml/distillation.py`)

Once the ensemble has emitted 10k+ decisions, train a compact local
model (DistilBERT-sized) to mimic its decisions. Trades 5% of accuracy
for 100x faster inference + zero LLM cost. Pin this as a long-term
cost-reduction lever.

**Implementation**: `src/halal_trader/ml/distillation.py` ships the
pure-Python distillation policy + deployment-gate engine. The actual
HuggingFace `transformers` distillation training loop is operator-side
and consumes this module's policy output as configuration. Four
enums: `DistillationDecision` (SKIP_INSUFFICIENT_DECISIONS /
SKIP_INSUFFICIENT_DIVERSITY / SKIP_RECENT_RETRAIN / LAUNCH),
`DeploymentStage` (NOT_DEPLOYED / SHADOW / CANARY / PRODUCTION;
terminal REJECTED / RETIRED), `GateOutcome` (PASSED / FAILED).
`DistillationPolicy` ships the roadmap-pinned thresholds:
`min_decisions=10_000` (the roadmap pin), `min_class_balance=0.05`
(rare-class signal floor), `min_retrain_interval=14d`,
`accuracy_floor=0.95` (the roadmap-pinned 5% accuracy trade),
`latency_speedup_floor=10.0` (the gate floor; 100x is the
aspiration). Validation rejects invalid policy values including
the "speedup ≤ 1.0 = no point distilling" pin. `DecisionCohort`
carries BUY/SELL/HOLD counts + total + earliest/latest dates;
validates sum-equals-total invariant + tz-aware datetimes; the
`class_balance()` method returns the smallest class fraction.
`trigger_decision(cohort, *, last_retrain_at, now)` runs the gate
chain in priority order: insufficient decisions → diversity →
recent retrain → LAUNCH. Boundaries pinned both directions: 9999
decisions skips, 10000 launches; <0.05 class fraction skips, =0.05
launches; <14 days since retrain skips, =14 days launches.
`StudentValidation` carries agreement_rate + latency p99 +
teacher_latency_p99 + holdout_size; `latency_speedup` property
is teacher/student. `evaluate_gates` runs accuracy ≥95% and
latency ≥10x speedup gates with boundary-inclusive comparisons
both directions. `StudentDeployment` state machine: `start_deployment`
at NOT_DEPLOYED → `promote(SHADOW)` → `promote(CANARY)` →
`promote(PRODUCTION)`; skip raises `DeploymentOrderError`; `reject`
moves to terminal REJECTED; `retire` moves to terminal RETIRED;
neither REJECTED nor RETIRED can be promoted from. Render helpers
for validation report + deployment state with no-secret-leak
regression (no raw decisions / training_data / weights). Tests in
`tests/test_distillation.py` (76 cases): all four enum string-value
pins; policy validation including roadmap-threshold pin (10k /
0.95 / 10.0); cohort sum-invariant + class_balance; trigger
decision priority order + boundaries inclusive both directions
(9999/10000 + 0.05 + 14d); naive datetime rejections; validation
field validation + speedup property; gate boundary pins (95%
inclusive, just-below fails; 10x inclusive, just-below fails);
custom policy flows through; deployment state machine (one-step
forward; full path to PRODUCTION; skip rejected; cannot promote()
to REJECTED/RETIRED; cannot promote from REJECTED/RETIRED;
already-at-stage rejected; immutability); reject + retire idempotent
checks; render no-secret regression; e2e full lifecycle from
trigger through PRODUCTION; failed-student rejection; replay
consistency.

**Acceptance**: `halal-trader ml status` shows model versions, last
retrain timestamp, drift metrics, latency budget compliance for every
production model.

---

## Wave 7 — Backtest + research toolkit (3-4 weeks)

Make the platform a research environment, not just a runtime.

### 7.A — Notebook integration  ✅ landed (3 starter notebooks)

Pre-built Jupyter / Marimo notebooks for: "explore the replay store",
"fit a custom regime detector", "test a new prompt template",
"correlate sentiment with returns". Operators run these locally
against their own data.

**Landed (3 starter notebooks):** `notebooks/` directory with a
README index, three valid `.ipynb` files (each verified to
parse cleanly via `json.load`), and the contributor / setup
documentation.

**`explore-replay-store.ipynb`** — six-section research notebook
that pulls recent decisions / closed trades / cost summaries
from Postgres, plots the equity curve, and ends with an explicit
"read-only invariant check" cell that prints the row counts of
the three tables read so the operator can verify the notebook
hasn't modified anything. Tunable `LOOKBACK_DAYS` parameter at
the top.

**`test-custom-prompt.ipynb`** — operators paste a candidate
system prompt into the cell, the notebook samples N historical
`LlmDecision` rows, runs the candidate against each via
`core.llm.factory:create_llm` (so it goes through the same
provider chain + cost cap + circuit breaker the live cycle
uses), and reports per-pair-action agreement vs the recorded
decision. Pin: registering the candidate via `core.llm.prompts`
is **explicitly NOT done in the notebook** — the notebook is
ad-hoc testing; promoting requires the operator to follow the
five-step checklist (move into `core/llm/prompts/`, register
with version bump, run Wave 5.B A/B comparator after 100 live
trades, run Wave 4.F promotion gate, run Wave 6.D ML CI
pipeline) the notebook documents in its closing markdown cell.

**`sentiment-vs-returns.ipynb`** — joins closed trades against
their `IndicatorSnapshot` rows, buckets by entry sentiment, and
runs Wave 5.B's `core.ab_compare.compare` between bullish-entry
and bearish-entry returns to surface a Welch's t-test verdict.
Three interpretation paths documented in the closing cell:
significant + bullish-better-than-bearish (signal is informative,
consider increasing weight); not significant (signal isn't
producing actionable edge); significant + bearish-better
(contrarian indicator). Cross-references Wave 7.C's
`core.counterfactual` for the "what-if" follow-up.

`notebooks/README.md` documents setup (`uv sync --extra
dashboard && uv pip install jupyterlab && uv run jupyter lab
notebooks/`), the three available notebooks with one-line
descriptions, the halal alignment note (every notebook is
research-only and holds itself to a `commit=False` discipline
so the asyncpg transaction never commits), the conventions
("What this does / What it doesn't do" markdown header on every
notebook, parameterised SQL only — no f-string injection,
anonymised aggregates by default), and a contribution path
(operator PR → maintainer runs end-to-end → confirms read-only
invariant via no INSERT / UPDATE / DELETE → adds to index).
Pure docs / notebooks — no code change.

### 7.B — Synthetic data generators for stress tests  ✅ landed

The existing `crypto/stress.py` generates flash crashes / pumps /
gaps. Extend to: regime shifts (slow → fast volatility), correlation
breakdowns, liquidity crises (wide spreads). Battery of canned
scenarios every strategy must pass.

**Landed:** `crypto/stress.py` gains four new generators that map
to the failure modes the existing five didn't cover.
`regime_shift_klines` walks calmly for `n_calm` bars then jumps to
~15× volatility for `n_turbulent` bars (catches strategies that
keep their fixed sizing through a vol regime change).
`volatility_explosion_klines` produces a sustained high-vol burst
that mean-reverts gently (catches strategies that confuse range for
trend). `liquidity_crunch_klines` collapses volume to ~5% of normal
and inflates each bar's H-L range to ~3× its body — the kline-level
approximation of a wide bid-ask spread (Kline has no spread field).
`correlated_pair_klines` returns two synced kline streams whose
correlation drops at a configurable break-point — lets multi-pair
strategies be tested against correlation breakdown without real
data. Three new graders (`_grade_regime_shift`,
`_grade_volatility_explosion`, `_grade_liquidity_crunch`) wired into
`_GRADERS` and three new entries in `standard_scenarios()` so the
default suite now covers eight failure modes. 23 tests cover: kline
counts, the structural promises (vol jump, near-zero drift, wide
H-L, collapsed volume), seed determinism, the correlation rho
ladder pre vs post breakdown, predicate validation, grader pass/fail
on no-op vs high-confidence plans, and an invariant that every
scenario in the standard suite has a registered grader (so a future
addition can't silently "pass" by missing one).

### 7.C — Counterfactual reasoning ("what if I hadn't traded BTCUSDT this week")  ✅ landed (analyzer core)

Given a closed trade history, compute the counterfactual: "what
would equity have been if you'd skipped trades matching pattern X?"
Surfaces hidden costs (e.g. stop-out clusters in a specific regime).

**Landed (analyzer core):** `core/counterfactual.py` ships
`analyze_counterfactual(trades, skip_predicate, *, starting_equity=1.0)`
returning a `CounterfactualReport` with the actual + counterfactual
`CohortStats` (reusing the A/B comparator's `cohort_stats` so the
Sharpe / drawdown / profit-factor formulas stay consistent across
the dashboard), per-trade equity curves for both paths (length-
matched — the counterfactual flatlines across skipped trades so
the time axis is preserved for dashboard plotting), `skipped_count`,
and a signed `return_uplift`. Three predicate factories ship:
`by_symbol(s)` (handles both `Trade.symbol` and `CryptoTrade.pair`,
case-insensitive), `by_regime(r)`, and `by_loss_streak(n)` (a
stateful closure that starts skipping after `n` consecutive losses
— pinned to be independent per factory call so concurrent analyses
don't share streak state). Tolerant to dict-shaped or attribute-
shaped rows; legacy rows missing `return_pct` are silently dropped;
predicate exceptions are treated as "do not skip" so a buggy
predicate can't crash the operator's research session. 31 tests
cover the row-extractor tolerance, the curve-flatlines-on-skip
plotting convention, the kept-trades-only stats invariant (so
held-flat zeros don't distort win-rate), the uplift sign matches
direction (winner-skip → negative, loser-skip → positive), every
predicate factory's edge cases (case-insensitivity, missing fields,
attribute fallback, streak reset on win, factory independence),
and an end-to-end "drop DOGE makes the portfolio better" run on
attribute-shaped rows. SQL plumbing (a `/api/research/counterfactual`
route that runs a predicate against the live trade ledger) is
deferred to a follow-up — analyzer is verified in isolation first.

### 7.D — Public research API  ✅ landed (auth + rate limiter)

Read-only API for academic / external researchers to query the
anonymised aggregate trade history. Gated by API key; rate-limited.
Lets the broader halal-finance research community build on the platform.

**Landed (auth + rate limiter):** `web/research_api_keys.py` ships
the auth + rate-limit gate that sits in front of the future
research-API route handlers. Three composed layers: (1)
**`ApiKey`** dataclass with opaque `key_id` (public identifier
researchers quote in `X-Api-Key`), `secret_hash` (SHA-256 — never
stores or logs plaintext; `verify_secret` uses
`hmac.compare_digest` for constant-time comparison so timing
attacks can't extract secrets), `frozenset[Scope]` for
permissions, and `Tier`. (2) **Token-bucket rate limiter**
per-(key_id, endpoint_class) — a slow query on `trades.list`
doesn't burn budget for fast queries on `halal.summary`. Tiers:
ANONYMOUS (10/min, burst 20), RESEARCHER (60/min, burst 120),
PARTNER (600/min, burst 1200). Tier limits hard-coded in source
not Settings — a config typo can't silently 10× a tier. (3)
**Permission gate** with five scopes (`read:trades / halal /
regime / rationale / aggregate`) — additive, no implicit
hierarchy. `authenticate(...)` returns one of five
`AuthOutcome`s: `UNKNOWN_KEY`, `INVALID_SECRET`, `MISSING_SCOPE`,
`RATE_LIMITED`, `ALLOWED`. Pinned check order: auth → scope →
rate limit. Two security pins regression-tested: an attacker
spamming a victim's key_id with wrong secrets does NOT consume
the victim's bucket; same for wrong-scope spam — the bucket
consume happens after auth + scope checks pass. `issue_secret`
generates URL-safe 32-byte (256-bit) secrets via
`secrets.token_urlsafe`; `make_api_key` is the single point of
entry that hashes the secret + builds the immutable `ApiKey`.
`register` rejects duplicate `key_id` (silent overwrites mask
key leaks); `revoke` is idempotent. Halal alignment: the
registry never authorises a trade — it gates *read* access to
anonymised data; no operator identity / position-level data
flows through. Pure-Python (stdlib hashlib + hmac + secrets +
time); no DB / network / async. 37 tests cover hash determinism
+ rejection of empty plaintext, secret verification on match /
mismatch / empty, URL-safe character set + 10-uniqueness pin,
low-entropy issuance rejection (< 16 bytes), `make_api_key`
plaintext-vs-hash invariant + scope frozenset immutability + ApiKey
immutability + has_scope additivity, the documented tier-limit
values + relative ordering pin, registry register / revoke /
get + duplicate-rejection + idempotent-revoke, all five auth
outcomes, the auth-before-bucket-consume pin (twice — once for
invalid secret, once for missing scope), rate-limit kicks-in
after burst + refill after time + per-endpoint-isolation +
partner-tier high-burst + retry_after math (1 / refill_per_sec
for cost=1), revoked key denial, and frozen-dataclass
immutability across `AuthResult`. FastAPI route adapter (mount
the registry on `/api/research/*` with `Depends(authenticate)`),
SQL persistence (load `ApiKey` rows from a `research_api_keys`
table at startup), and the dashboard's key-management page
deferred to follow-ups — gate verified in isolation first.

### 7.E — Strategy marketplace ✅ landed (`web/marketplace.py`) — interface spec

Operators can publish their strategy as a template; other operators
can subscribe (with revenue share). Defer the legal / regulatory
work to a future round; just spec the interface here.

**Implementation**: `src/halal_trader/web/marketplace.py` ships
the pure-Python contract — listing shape, subscription lifecycle,
revenue-share math — that the future legal/payments review can
sign off on as a stable target. Four enums: `LicenseTerm`
(PERSONAL_USE / COMMERCIAL_USE / NON_COMMERCIAL_USE /
RESEARCH_ONLY), `ListingStatus` (DRAFT / PUBLISHED / UNLISTED /
TAKEN_DOWN), `SubscriptionStatus` (TRIAL / ACTIVE / PAUSED /
CANCELLED), `HalalCertLevel` (BASIC / MODERATE / STRICT /
SCHOLAR_REVIEWED). `MarketplacePolicy` ships the 7-day default
trial, 90/10 author/platform revenue share, $1-$999 monthly
pricing band; validation rejects author_share at 0 or 1, ceiling
below floor. `MarketplaceListing` carries listing_id +
author_anonymous_handle (mirrors Wave 10.A gallery anonymisation —
no user_id leaks) + name + description + strategy_kind +
halal_cert_level + license_term + monthly_price_usd + status +
optional published_at. `validate_listing` enforces price band
(both boundaries inclusive), PII denylist on name + description
(email / SSN / IP / phone regex), DRAFT-or-UNLISTED status (only
those can be re-validated for publishing). `publish_listing`
applies the gate and flips status. `take_down_listing` moves to
TAKEN_DOWN terminal. `Subscription` carries subscription_id +
listing_id + subscriber_anonymous_handle + status + dates;
validates trial_end_at >= started_at. `start_subscription`
enforces PUBLISHED listing status. `convert_to_active` requires
TRIAL + now >= trial_end_at. `pause_subscription` only from
ACTIVE. `resume_subscription` only from PAUSED. `cancel_subscription`
from any non-CANCELLED state, sets cancelled_at. `compute_split`
returns RevenueSplit with author + platform amounts summing to
revenue (within $0.01 rounding tolerance). Render no-secret
regression: no Stripe customer IDs, card data, payout accounts.
Tests in `tests/test_marketplace.py` (74 cases): all four enum
string-value pins; policy validation including author_share
(0,1) bounds + ceiling > floor + trial range [1,30]; listing
field validation; validate_listing across price-band boundaries
both directions ($1 inclusive, $0.50 below; $999 inclusive,
$1500 above), PII patterns rejected for email/phone/SSN/IP in
description and in name; only DRAFT or UNLISTED can be validated;
published flow sets published_at + immutable original; publish
runs validation; take_down sets terminal; subscription field
validation including trial_end_at >= started_at; start_subscription
rejects unpublished/taken-down listings; convert_to_active
requires now >= trial_end_at + only from TRIAL; pause/resume only
from ACTIVE/PAUSED respectively; cancel from any non-CANCELLED;
compute_split 90/10 default + zero revenue + custom share +
rounding consistency for awkward $19.99; RevenueSplit rejects
inconsistent sum + negative amounts; render listing with
$-formatted price + no-secret regression on Stripe IDs / payout /
card / ach; render subscription + render split; e2e
publish→subscribe→convert→cancel flow with correct revenue split
after 3 months; replay consistency.

**Acceptance**: A researcher with a fresh laptop can clone the
notebook templates, point them at the public dataset, and reproduce
the headline backtest within 30 minutes.

---

## Wave 8 — Reliability + production readiness (3-5 weeks)

What we need before letting real users with real money use this.

### 8.A — End-to-end integration test suite ✅ landed (`ops/e2e_scenarios.py`) — catalogue

Today's integration tests stub the broker. Add a nightly CI lane that
hits real testnet endpoints (Binance, Alpaca), runs full cycles, asserts
no halt-trips, no rejected orders, no missed fills. Catches integration
drift the unit tests can't.

**Implementation**: `src/halal_trader/ops/e2e_scenarios.py` ships
the pure-Python scenario catalogue + last-run tracker the nightly
CI lane consults. `ScenarioKind` enum (CYCLE / ORDER / HALT /
RECONCILE / WEBSOCKET / FAILOVER), `RequiredBroker` (NONE /
BINANCE_TESTNET / ALPACA_PAPER), `RunOutcome` (PASSED / FAILED /
SKIPPED — SKIPPED for broker-unreachable doesn't reset the
freshness clock), `FreshnessLevel` (FRESH / STALE / CRITICAL /
NEVER_RUN). `ScenarioPolicy` ladders fresh=7d / stale=14d /
critical=28d (operator-tunable; rejects out-of-order ladder).
`Scenario` carries scenario_id + kind + required_broker +
description + expected_outcome (pinned non-empty so contributors
must document success criteria). 8 seed scenarios cover the
roadmap's "real Binance testnet, real Alpaca paper" pin:
binance_full_cycle, binance_order_lifecycle,
binance_websocket_stream, alpaca_full_cycle,
alpaca_order_lifecycle, halt_then_resume, broker_5xx_failover,
reconciliation_drift. `RunRecord` is the audit row.
`last_passed_run` returns the most recent PASSED record (ignores
FAILED + SKIPPED — the structural pin that keeps the staleness
clock honest). `freshness_for(last_passed_at, *, now, policy)`
classifies via the inclusive ladder (7d → STALE; 28d → CRITICAL).
`build_status` aggregates per-freshness counts. Render helpers
with kind emoji (🔄 cycle / 📋 order / 🛑 halt / ⚖️ reconcile /
📡 websocket / 🔁 failover) + freshness emoji (✅⚠️🔴❓).
No-secret regression. Tests in `tests/test_e2e_scenarios.py`
(56 cases): all four enum string-value pins; policy validation
including out-of-order ladder rejection (stale_below_fresh,
critical_below_stale); Scenario validation; RunRecord validation;
catalogue coverage (every ScenarioKind has at least one seed;
both Binance + Alpaca have scenarios; full_cycle for both
brokers; halt_then_resume present; reconciliation_drift present;
canonical sorted order); freshness ladder boundaries inclusive
(7d boundary → STALE; just-below 7d → FRESH; 28d boundary →
CRITICAL; just-below 28d → STALE); custom policy flows through;
naive datetime rejected on now and last_passed_at;
last_passed_run ignores FAILED and SKIPPED (the load-bearing
pin: SKIPPED runs don't help freshness because the test didn't
actually validate); aggregate status across mixed cohort;
empty + all-never-run cohorts; render scenario with kind emoji
+ broker + description + expected; render no-secret regression
across all seed scenarios; e2e nightly run lifecycle (PASS day 1
→ FRESH; day 8 → STALE; day 30 → CRITICAL); SKIPPED-for-broker-
down doesn't reset clock (load-bearing pin).

### 8.B — Chaos engineering harness  ✅ landed (engine)

Random fault injection: kill a process mid-cycle, drop a WebSocket,
return malformed JSON from the broker, return a 5xx, exhaust a rate
limit. Every fault must result in either a successful recovery or a
clean halt — never a hang or silent data loss.

**Landed (engine):** `core/chaos.py` ships eight `FaultKind`s (broker
timeout / 5xx / malformed JSON / rate-limit, websocket drop, DB
connection drop, LLM timeout / 500), four predicates (`always()`,
`never()`, `on_call_number(n)`, `for_label(label)`) plus
`combine_and()` composition, and `chaos_call` / `chaos_call_async`
wrappers that swap the target's call for a fault when the predicate
matches. The fault library uses sentinel exceptions all descending
from `HarnessError` so target callables can catch the base class for
generic recovery; `BrokerRateLimitError` extends `BrokerHttpError`
so a recovery handler catching the parent automatically handles 429s.
`evaluate(scenario, target, *, deadline_seconds, expected_exceptions)`
classifies into four buckets: `RECOVER` (returned normally),
`CLEAN_HALT` (raised an expected exception type), `HANG` (exceeded
deadline), `CRASH` (raised an unexpected type). Pin: the operator
**declares** which exception types are acceptable rather than letting
pytest's blanket "any raise is a fail" mask the distinction; default
expected_exceptions is `(HarnessError,)` so every harness fault
counts as a clean halt by default. `evaluate_all(scenarios,
target_factory)` runs every scenario via a fresh target — pinned so
state can't leak between iterations. `standard_scenarios()` covers
every FaultKind (regression test pinned that the suite covers
`set(FaultKind)` exactly so adding a kind without a scenario fails
loudly). `render_verdicts` produces a CLI / Slack-ready text payload
visually consistent with the existing `crypto/stress.render_report`.
Pure-Python; no DB / network / async-loop ownership. 34 tests cover
each predicate (always / never / on-Nth / by-label / combine_and
including the empty-composition `→ always` fallback), the fault
library (every kind raises its sentinel; rate-limit is BrokerHttpError
subclass with status_code=429; HTTP error carries status_code),
the sync + async wrappers (passthrough on never, raise on always,
predicate sees label/args/kwargs), the evaluator's four-bucket
classification (recover / clean_halt with expected exception /
crash on unexpected MemoryError / hang on deadline overrun), the
default `expected_exceptions=(HarnessError,)` pin, the elapsed_ms
recording, `evaluate_all`'s factory invocation pattern (one call per
scenario, fresh target each time), `standard_scenarios()` covers
every FaultKind + unique names + descriptions, and the verdict
`passed` property + scenario immutability. CLI command (`halal-trader
chaos run`) and integration with the cycle's actual broker / DB /
LLM call sites deferred to a follow-up; engine verified in isolation
first.

### 8.C — Disaster recovery drills ✅ landed (`ops/dr_drills.py`)

Restore from backup, replay the last 24 hours of cycles, reconcile
broker state. Documented runbook + monthly automated drill.

**Implementation**: `src/halal_trader/ops/dr_drills.py` ships the
pure-Python drill state machine + scoring layer. `DrillKind` enum
(BACKUP_RESTORE / CYCLE_REPLAY / BROKER_RECONCILE) with pinned
string values; each kind has its own canonical step sequence
pinned in code (`_BACKUP_RESTORE_STEPS` 6 steps, `_CYCLE_REPLAY_STEPS`
5 steps, `_BROKER_RECONCILE_STEPS` 4 steps). `StepStatus` (PENDING /
PASSED / FAILED) and `DrillStatus` (IN_PROGRESS / PASSED / FAILED)
enums; pinned three-way state because PASS/FAIL is the actionable
signal but PENDING is needed for in-progress accounting. `DrillPolicy`
ships the 30-day cadence default (operator-tunable; rejects zero/
negative). `StepRecord` (audit row) with optional `notes` for
operator commentary. `DrillRun` is the immutable per-drill state;
operations return new state. `start_drill` creates a fresh run with
empty records. `record_step` enforces canonical-order prerequisite
(out-of-order step raises `StepOutOfOrderError` with both step +
missing prereq), unknown step (not in kind's sequence) raises
`UnknownStepError`, FAILED drill blocks subsequent steps via
`DrillAlreadyFailedError`, already-decided step rejected, naive now
rejected, PENDING-as-input rejected. `aggregate_status` returns
FAILED if any step failed; PASSED if all passed; else IN_PROGRESS.
`next_step` returns first pending step (None if drill is FAILED or
PASSED). `completed_at` returns max decided_at when done. `is_overdue`
true when never-run or `now - last_passed_at > cadence` (exclusive
boundary at exactly 30d — pinned both directions). `days_overdue`
returns excess days past cadence; -1 sentinel for never-run.
`render_drill` shows kind + operator + per-step emoji (✅ passed /
❌ failed / ⬜ pending) + step name + notes + aggregate status +
next-step hint + completion timestamp. No-secret-leak regression:
no `$` / `USD` / `cus_` / `sub_` / `api_key` / `bearer` substrings.
Tests in `tests/test_dr_drills.py` (60 cases): drill kind / step
status / drill status enum string-value pins; canonical step
sequences pinned per kind; policy validation (30d default, zero/
negative cadence rejected, frozen); StepRecord validation (empty
step rejected, naive decided_at rejected, PENDING status rejected,
default empty notes, frozen); DrillRun validation; start_drill
(empty drill_id / operator / naive now rejected); record_step
(passes first step, full pass-all-steps happy path, out-of-order
rejected with prereq carried, unknown step rejected, already-decided
rejected, PENDING-input rejected, naive now rejected, returns new
state without mutating, records notes); failure short-circuit
(failed step → FAILED aggregate; subsequent record_step raises
DrillAlreadyFailedError; failed drill's next_step returns None);
aggregate_status (in-progress at start, in-progress with partial
records, passed when all passed); completed_at (None when
in-progress, max decided_at when done); is_overdue (never-run is
overdue, recent is not, exactly 30d boundary is NOT overdue, past-30d
is overdue, custom cadence flows through, naive now/last_passed_at
rejected); days_overdue (zero when not overdue, excess days
returned, -1 sentinel for never-run); render output (drill_id +
kind + operator visible; step emoji per status; failure emoji;
notes shown when present; next-step hint; completion timestamp;
no-secret-leak regression); e2e flows (monthly drill passes →
overdue check at +10d false, at +45d true; failure mid-flow blocks
subsequent steps; replay consistency).

### 8.D — Distributed tracing via OpenTelemetry  ✅ landed (translator)

`core/tracing.py` is in-house. Migrate to OTel (the contract is the
same). Wire to a free Tempo / Jaeger instance. Operators can grep
across services + brokers + LLM calls in one view.

**Landed (translator):** `core/otel_translator.py` ships
`build_trace(*, cycle_started_at, cycle_ended_at, stage_events,
trace_id, cycle_attributes)` returning a `TraceBundle` ready to
ship to any OTLP/HTTP collector — without pulling in
`opentelemetry-sdk` as a hard dependency. Hand-rolled translator
because (a) the SDK has 30+ transitive deps + non-trivial
startup cost the per-cycle hot path can't afford, (b) the OTLP
v1 wire format is well-specified and < 100 lines of dataclass-
to-dict, (c) pure-Python keeps the translator testable without
a collector. Three concerns: ID generation (`new_trace_id` →
32-char lowercase hex; `new_span_id` → 16-char; both
cryptographically random per OTLP spec — uppercase / wrong
length is silently dropped by collectors so the validator
catches it eagerly); span hierarchy (cycle is a root span;
every stage is a child via `parent_span_id`; nested stages
just chain another link, the translator already supports); OTLP
JSON encoding (camelCase field names, nano timestamps as
strings to handle int64 precision in lossy JSON parsers,
attributes as the `[{key, value: {stringValue: …}}]` list-of-
records shape per the protobuf-derived JSON spec). Three pinned
safeties: open-stage spans (no `ended_at`) get `UNSET` status
+ "stage in progress" message + zero-duration end_nanos — pin
so an incomplete cycle still produces a valid trace the
collector renders as in-flight; the **redacted-attribute
filter** drops `prompt / rationale / raw_response / thinking /
api_key / secret / operator_id / broker_key / llm_token` from
attrs before exporting (LLM rationales contain operator strategy
IP and shouldn't ship to a third-party APM); attribute values
capped at 256 chars so a runaway value can't bloat the export
payload. The redacted-keys list is a `frozenset` so runtime
mutation can't add a non-redacted key by mistake — operators
extend it via code + review. Float-precision pin: timestamp
conversion uses integer microseconds × 1000 rather than
`timestamp() * 1e9` (the float path drifts ~128ns at 2026-era
unix times). `to_otlp_payload(*, service_name)` wraps in the
required `resourceSpans[0].scopeSpans[0]` nesting; `to_json()`
returns a JSON string operators ship via httpx / requests /
urllib without further transformation. Halal alignment:
tracing is observability-only; never opens a position; the
redacted-attribute filter is the explicit operator-IP /
PII guard. Pure-Python (stdlib `secrets` + `json`); no
opentelemetry-sdk / DB / network. 47 tests cover ID generation
(length + alphabet + uniqueness), span validation (every
ID-validation rule + start/end ordering + zero-duration
acceptance + parent-id-empty-vs-set), redaction filter
(drops every key in the denylist + caps long values + passes
short values + frozenset immutability of the denylist),
`span_from_event` across completed / errored / open paths,
the OTLP/JSON shape (camelCase + nano-as-string + list-of-
records attributes + sorted-keys for deterministic output +
status code + message), `build_trace` for complete-cycle +
incomplete-with-completed-stages + incomplete-with-no-stages +
custom-trace-id paths, the `to_otlp_payload` resourceSpans
wrapper + service.name attribute + custom-service-name
override, the JSON-roundtrip via `json.loads`, frozen
immutability across all three output dataclasses, and the
microsecond-precision pin via a 42ms-duration assertion.
Cycle-side wiring (subscribe to the bus, build a TraceBundle on
`cycle.complete`, ship via httpx to the operator's collector
endpoint) and the `Settings.otlp_endpoint` config deferred to a
follow-up — translator verified in isolation first.

### 8.E — On-call runbook + alerting  ✅ landed (router + seed runbooks)

Per-error-type runbook in `docs/runbooks/`. Pager alerts for:
chain-backoff, halt-trip, drift > threshold, snapshot-store failure,
broker-API > 5% error rate.

**Landed (router + seed runbooks):** `core/alert_router.py` ships
`AlertSpec(type, severity, summary, runbook_url, context)` typed
payload, `Severity` IntEnum (INFO/WARN/PAGE — IntEnum so `severity
>= warn` reads naturally), `Channel` Protocol matching the existing
Telegram / Slack / Discord notifier shape, `AlertRoute(severity_min,
channel)` rules, and `AlertRouter` composing them. Three semantic
pins: per-(type, channel) dedup so a Slack-suppressed alert can
still reach Telegram (operator sees each alert once *per channel*,
not once globally — distinct from the existing `AlertSink` which
dedups globally per type); a per-channel send failure does NOT
abort dispatch to other channels (operator wants Telegram delivered
even if Slack is down); a channel `enabled=False` is skipped
silently rather than counted as suppressed or failed (operator
deliberately turned it off). `send` returning False is classified
as a failure and does NOT burn the cooldown — the next attempt
retries instead of being silently suppressed. Multiple routes
targeting the same channel (e.g. WARN + PAGE both → Slack)
de-duplicate to one delivery per alert. `render_message` produces
an emoji-prefixed (ℹ️ / ⚠️ / 🚨) text payload that flags
missing `runbook_url` with `(none yet — write one in
docs/runbooks/)` so the gap is visible in every alert until a
runbook is filed. Pure-Python; no DB / network. The clock is
injectable via `now_fn` so dedup tests are deterministic.

**Seed runbooks:** `docs/runbooks/` ships a `README.md` with the
filename convention + index, a `_template.md` with the five-section
template (likely causes / diagnose / mitigate / escalate /
postmortem), and four runbooks for the alerts the bot already
raises: `halt-engaged.md` (PAGE — kill-switch trip causes +
mitigation across operator-engaged / spend-cap / circuit-breaker /
drift paths), `chain-backoff.md` (WARN — LLM provider chain in
exponential backoff), `drift-breach.md` (WARN — Wave 4.I detector
tripping), `broker-api-error-rate.md` (PAGE — broker outage / rate-
limit / auth / network).

28 tests cover severity routing (≥ min), multi-route same-channel
dedup, multi-channel distinct dispatch, cooldown semantics
(repeat-suppressed, repeat-after-cooldown-passes, per-type
isolation, per-channel isolation), failure isolation (one channel
raising doesn't block others; send-returning-False is failure not
suppression and doesn't burn cooldown), disabled-channel silent
skip, `reached_any` property, `matching_channels` filter +
deduplication, every render-message section (emoji / type / severity
name / runbook URL present + missing-runbook flag / sorted context
k/v / context-omission when empty), and frozen-dataclass
immutability on AlertSpec + DispatchResult. Concrete-channel
adapters wrapping Telegram / Slack / Discord / email and the cycle-
side wiring (replace `AlertSink.notify` callsites with
`AlertRouter.dispatch`) deferred to a follow-up — router verified
in isolation first.

### 10.F aux — Feature flag rollout engine ✅ landed (`web/feature_flags.py`)

**Implementation**: `src/halal_trader/web/feature_flags.py` is the
runtime-rollout complement to Wave 10.F's edition gating. Edition
gating answers "is feature X in this build?"; this module answers
"is feature X enabled for THIS user RIGHT NOW during gradual
rollout?". `RolloutKind` enum (OFF / ON / PERCENTAGE /
COHORT_ALLOWLIST). `FeatureFlag` carries flag_id + description +
kind + percentage + cohort_user_ids; validation enforces consistency
(percentage in [0,100]; **PERCENTAGE kind with 0% rejected — use OFF;
PERCENTAGE with 100% rejected — use ON**; COHORT_ALLOWLIST requires
non-empty user_ids; non-cohort kinds must not have cohort_user_ids).
`is_enabled(flag, *, user_id)` is pure: SHA-256 of (flag_id:user_id)
salted by flag_id for correlation-free 50/50 splits across two
flags. `FlagRegistry` rejects duplicate flag_ids. `enabled_count`
for "did 10% rollout actually enable ~10%?" verification without
leaking individual user_ids. Render shows cohort SIZE not user_ids;
no-secret regression. Tests in `tests/test_feature_flags.py` (44
cases): RolloutKind enum string-value pin; FeatureFlag validation
including **PERCENTAGE-with-0-or-100% rejected pins** + **non-cohort-
must-not-have-cohort_user_ids pin**; immutable; OFF always-false,
ON always-true; **deterministic per-user evaluation pin** (no
flicker across reloads); 50% rollout enables ~half of 1000 users
(empirical 400-600 range); 10% rollout enables ~100 (70-130 range);
**correlation-free pin** (two 50% flags overlap ~25% not 50%);
COHORT_ALLOWLIST explicit + not hash-based; FlagRegistry
duplicate-flag-id rejection; lookup + is_enabled_in; all_flags
sorted; enabled_count; render shows cohort SIZE not individual
user_ids (load-bearing privacy pin); render no-secret regression
(no `@`, no individual user_ids); registry render with summary
counts; e2e flows (**monotonic gradual rollout 0→10→50→100% pin**:
each user enabled at stage N stays enabled at stage N+1, no
flicker); cohort-then-percentage transition; replay consistency.

### Deprecation policy engine ✅ landed (`ops/deprecation.py`)

**Implementation**: `src/halal_trader/ops/deprecation.py` is the
sunset-timeline complement to Wave 9.F API reference. The reference
docs surface what's currently public; this engine tracks how
public APIs move from announced → deprecated → removed.
`DeprecationStage` enum (ANNOUNCED / DEPRECATED / REMOVED)
forward-only. `DeprecationPolicy` defaults: 60-day announce + 90-day
deprecated = 150-day total sunset. **Validation enforces minimum
30-day announce + 60-day deprecated** (users need migration time;
shorter windows would break library users); pinned via test against
silently-shorter sunsets. `DeprecatedSymbol` carries symbol +
announced_at + stage + optional replacement + migration_url +
reason. `announce_deprecation` creates ANNOUNCED record.
`advance_stage` enforces forward-only (cannot skip ahead, cannot
revert, REMOVED terminal). `scheduled_deprecated_at` /
`scheduled_removal_at` compute timeline dates from policy.
`is_overdue_for_advancement` flags records past their scheduled
date — boundary inclusive (≥) at 60 / 150 days. `emit_warning_if_needed`
emits DeprecationWarning ONLY in DEPRECATED stage (no warning in
ANNOUNCED grace period; no warning in REMOVED — callers raise
SymbolRemovedError instead). `assert_not_removed` raises
`SymbolRemovedError` (inherits RuntimeError so generic handlers
catch) carrying replacement + migration_url for actionable error
messages. Render with stage emoji 📣/⚠️/🗑️ + scheduled date for
non-terminal stages + replacement / migration_url / reason when
present. Tests in `tests/test_deprecation.py` (48 cases): stage enum
string-value pin; **60d announce + 90d deprecated default pin**;
**minimum 30d announce / 60d deprecated rejected pin**; immutable;
DeprecatedSymbol validation; lifecycle ANNOUNCED → DEPRECATED →
REMOVED; **REMOVED-is-terminal rejection pin**; metadata carries
forward; scheduled_*_at math + custom policy flow; **overdue boundary
inclusive at 60d / 150d both directions pinned** (60d overdue, 59d
not); REMOVED never overdue; **no warning in ANNOUNCED stage**
(grace period silent); **DeprecationWarning in DEPRECATED stage
includes replacement + migration_url**; **no warning in REMOVED**
(use assert_not_removed); assert_not_removed passes ANNOUNCED/
DEPRECATED, raises SymbolRemovedError on REMOVED with carried
fields; **SymbolRemovedError inherits RuntimeError** for generic
handlers; works without replacement; filter_overdue returns only
overdue + sorted by announced_at ascending (oldest first); render
shows scheduled date for ANNOUNCED + DEPRECATED but not REMOVED;
includes replacement / migration / reason when set; omits when
empty; no-secret leak in render; e2e flows (full sunset Q1 →
DEPRECATED Q2 → REMOVED Q3 with assertion at each stage; warning
emission with migration link visible during DEPRECATED window;
replay consistency).

### Token bucket rate limiter ✅ landed (`core/rate_limiter.py`)

**Implementation**: `src/halal_trader/core/rate_limiter.py` ships
the public reusable token bucket. Broker / screener / LLM-provider
adapters need to stay BELOW the upstream's rate limit to avoid
getting throttled in the first place — a proactive complement to
the `core/circuit_breaker.py` reactive layer (breaker reacts after
a call fails; limiter prevents the call before it would). Distinct
from the private `_Bucket` in `web/research_api_keys.py` (stateful,
tied to research API tier registry); this module ships the public
reusable primitive. `ConsumeOutcome` enum (ALLOWED /
DENIED_INSUFFICIENT / DENIED_OVERSIZED) pinned string values.
`BucketPolicy` carries `capacity` (max tokens / burst size) +
`refill_rate_per_sec` (steady-state replenishment); validation
rejects zero/negative + non-finite (inf/NaN) on both fields.
`BucketSnapshot` carries `tokens` (>=0, finite) + `last_refill_at`
(tz-aware). `full_bucket(*, now, policy)` constructs a fresh
full bucket. `refill(snapshot, *, now, policy)` advances elapsed-
time refill capped at capacity; **rejects backwards-clock**
(now < last_refill_at). `try_consume(snapshot, n, *, now, policy)`
is **atomic refill+spend**: refills first, then spends if
sufficient; returns (new_snapshot, ConsumeOutcome). Even on
DENIED_INSUFFICIENT, last_refill_at advances so subsequent
`time_until_available` is meaningful. **n > capacity →
DENIED_OVERSIZED** (programming error; never allowed regardless
of refill). Zero/negative/non-finite n rejected. Float costs
supported. `time_until_available(snapshot, n, *, now, policy)`
returns timedelta until n tokens available (0 if already; max
sentinel for n > capacity). `fill_ratio` for dashboard tile.
Render with ratio-based emoji (🟢 ≥50% / 🟡 ≥20% / 🔴 <20%) +
tokens/capacity + refill rate. **No-secret-leak pin**: snapshot
deliberately doesn't carry adapter call args; render shows only
bucket state. Tests in `tests/test_rate_limiter.py` (52 cases):
ConsumeOutcome enum string-value pin; BucketPolicy validation
(zero/negative/inf/NaN rejected on both fields); immutability;
BucketSnapshot validation (negative + non-finite tokens + naive
datetime rejected); full_bucket starts at capacity; **refill
proportional pin** (3 sec * 1/s = 3 tokens added); **caps at
capacity pin**; zero-elapsed no-op; **fractional rate pin** (0.5/s
sub-second cadence); **backwards-clock rejected pin**; advances
last_refill_at; try_consume ALLOWED when sufficient; **DENIED_
INSUFFICIENT no-spend pin** (snapshot reflects refill but tokens
unchanged on refill side); **atomic refill+spend pin** (zero-token
bucket + 5sec * 1/s = 5 tokens, then spend 3 → 2); **DENIED_
OVERSIZED pin** (n > capacity at full bucket); **n == capacity
allowed at full pin**; zero/negative/inf n rejected; fractional
costs supported; advances last_refill_at on both ALLOWED + DENIED;
time_until_available 0 when sufficient; proportional to deficit;
with partial refill; **oversized returns timedelta.max pin**;
fractional rate; fill_ratio full=1.0 + empty=0.0 + after-refill;
render emoji per band; includes refill rate; **no-secret-leak pin**
(no `sk_live`/`Authorization`/`Bearer` substrings); e2e flows
(steady drain and recover: burst 5 calls drains bucket; 6th denied;
1s later 1 token allows next; 100s pause refills to capacity);
**oversized-permanently-impossible pin** (very fast 1000/s refill
+ 5 capacity, n=10 still DENIED_OVERSIZED at T+100s + time_until
returns max); **caller-uses-time-until-to-back-off pin** (deny →
consult time_until → wait → retry succeeds with exact deficit
math); replay consistency.

### Idempotency key store ✅ landed (`core/idempotency.py`)

**Implementation**: `src/halal_trader/core/idempotency.py` ships
the local-ledger half of retry-safe broker submission. When the
bot submits an order to a broker and the network call times out,
the bot needs to retry — but retrying naively could double-submit
the order if the original request actually reached the broker.
This module ships deterministic key generation from (operation,
payload) + a forward state machine PENDING → {SUCCEEDED | FAILED}
+ a pending-timeout reclaim path for genuinely stuck requests.
`IdempotencyState` enum (PENDING / SUCCEEDED / FAILED) pinned
string values. `IdempotencyAction` enum (PROCEED_NEW /
PROCEED_RETRY / REPLAY / IN_FLIGHT_REJECT). `IdempotencyPolicy`
defaults: 300s pending timeout + 86400s (24h) TTL; **TTL >=
pending_timeout** structural pin (otherwise pending entries get
evicted before they can be reclaimed). `IdempotencyEntry` carries
key + state + result (None for PENDING/FAILED, required for
SUCCEEDED) + attempts (>=1) + first_seen_at + last_attempt_at +
terminal_at (None for PENDING, required for terminal). Five
structural invariants pinned: PENDING-must-not-have-result,
PENDING-must-not-have-terminal_at, terminal-requires-terminal_at,
SUCCEEDED-requires-result, FAILED-must-not-have-result.
`make_idempotency_key(operation, payload)` produces canonical
SHA-256 hex (64 chars); dict insertion order doesn't matter
(`json.dumps(sort_keys=True)`). `decide(entries, key, *, now,
policy)` is pure read returning the action: no entry → PROCEED_NEW;
SUCCEEDED → REPLAY; FAILED → PROCEED_RETRY; PENDING + within
timeout → IN_FLIGHT_REJECT; PENDING + past timeout → PROCEED_RETRY
(stuck-caller reclaim, boundary inclusive at 300s). `claim` marks
key PENDING (creates new entry with attempts=1, or increments
existing entry's attempts and resets state to PENDING preserving
first_seen_at). `record_success(entries, key, result, *, now)`
and `record_failure` mark terminal — both reject calls on already-
terminal entries (forward-only state machine pin); both raise
KeyError on unknown key. `replay_result` returns the cached
SUCCEEDED result; raises on PENDING/FAILED. `evict_expired`
drops entries past TTL; **TTL computed from first_seen_at not
terminal_at** so long-lived stuck PENDING gets cleaned up.
Render with state emoji 🕐/✅/❌ + truncated key prefix + state +
attempt count + first_seen_at; **no-secret-leak pin**: render
NEVER includes the result payload (could contain order_id /
fill_price / API tokens). Tests in `tests/test_idempotency.py`
(69 cases): enum string-value pins; policy validation including
**TTL >= pending_timeout structural pin**; key determinism +
**dict-order-independent pin** + 64-char-hex pin + supports all
PayloadValue types (str/int/float/bool/None); empty operation
rejected; entry validation pinned in all 5 directions (PENDING-
no-result, PENDING-no-terminal_at, terminal-requires-terminal_at,
SUCCEEDED-requires-result, FAILED-no-result); **last_attempt_at
>= first_seen_at pin**; naive datetime rejected; decide returns
PROCEED_NEW for missing key; REPLAY for SUCCEEDED; PROCEED_RETRY
for FAILED; **IN_FLIGHT_REJECT for fresh PENDING pin**;
**PROCEED_RETRY at pending-timeout boundary inclusive pin** (300s
exact reclaims); just-below-timeout still rejects; far-past
proceeds; custom timeout both directions; claim creates new entry
+ increments existing + preserves first_seen_at + clears terminal
state on retry; record_success marks terminal; **forward-only
pin** (cannot record_success on SUCCEEDED + cannot record_failure
on terminal); empty result rejected; record_failure mirrors;
unknown-key KeyError both functions; replay returns cached;
**replay only on SUCCEEDED pin** (PENDING + FAILED both rejected);
evict drops old + keeps fresh; **TTL boundary inclusive pin** (at
86400s exact); **TTL uses first_seen_at not terminal_at pin** (a
PENDING that's been pending for >24h gets evicted); custom TTL;
render no-secret-leak pin (sk_live + order_id + fill_price not
in output); e2e flows (first-call-succeeds happy path with
replay returning cached id; failure-retries-to-success path with
attempts=2; **stuck-pending-reclaimed pin** original caller
crashed at T0 + 6 minutes later second caller proceeds with
attempts=2 + first_seen_at preserved; **concurrent-callers-blocked
pin** fresh PENDING blocks second caller within timeout; replay
consistency same operations → equal final ledger).

### Adapter circuit breaker ✅ landed (`core/circuit_breaker.py`)

**Implementation**: `src/halal_trader/core/circuit_breaker.py`
ships the classic three-state circuit breaker pattern as a pure-
Python snapshot-based engine. Broker / screener / LLM-provider
adapters call out over the network; when an upstream is failing
(rate-limited, down, slow), the bot needs to stop hammering it
after N consecutive failures and let it recover before retrying.
`BreakerState` enum (CLOSED / OPEN / HALF_OPEN) with pinned string
values. `CallOutcome` enum (SUCCESS / FAILURE). `BreakerPolicy`
defaults: 5 failures → OPEN; 60s cooldown; 2 HALF_OPEN successes
→ CLOSED. Validation rejects zero/negative thresholds + cooldowns
+ probe counts. `BreakerSnapshot` is the persistable state:
state + consecutive_failures + opened_at + half_open_successes;
**OPEN-requires-opened_at** structural pin; rejects naive datetimes.
`is_call_allowed(snapshot)` is the load-bearing gate (True for
CLOSED + HALF_OPEN, False for OPEN). `record_outcome(snapshot,
outcome, *, now, policy)` updates per the state machine: CLOSED
+ SUCCESS resets failures; CLOSED + FAILURE increments and trips
to OPEN at threshold; OPEN is identity (defensive against stray
outcomes); HALF_OPEN + SUCCESS increments and closes at probe
count; HALF_OPEN + FAILURE re-opens with **fresh cooldown**.
`tick(snapshot, *, now, policy)` drives the OPEN → HALF_OPEN
transition once cooldown elapses (boundary inclusive at
`now - opened_at >= cooldown`). `time_until_retry` returns
remaining cooldown for the dashboard ETA display. Render with
state emoji 🟢/🔴/🟡 + retry ETA when OPEN + probe progress when
HALF_OPEN. **No-secret-leak pin**: snapshot deliberately doesn't
carry adapter call args / responses; render shows only state +
counters; the adapter-side logger handles raw API call details.
Tests in `tests/test_circuit_breaker.py` (49 cases): enum string-
value pins; default policy + validation pins (zero/negative
rejected); snapshot validation (negative counts + OPEN-without-
opened_at + naive datetime all rejected); is_call_allowed across
all three states; **CLOSED + SUCCESS resets pin**; **CLOSED at-
threshold trips pin** (5th failure with default); **CLOSED below-
threshold pin** (4th failure stays CLOSED); custom threshold;
**OPEN identity pin** (stray outcome no-op); HALF_OPEN + SUCCESS
increments; **HALF_OPEN at-probe-count closes pin** (2nd success
→ CLOSED with reset); **HALF_OPEN + FAILURE reopens with fresh
cooldown pin**; custom probe count both directions; **naive now
rejected pin**; tick CLOSED/HALF_OPEN no-ops; **tick OPEN at
cooldown boundary inclusive pin** (60s exact triggers); tick
before cooldown stays OPEN; custom cooldown; time_until_retry
returns 0 outside OPEN + remaining when OPEN + 0 after expiry;
render closed/open/half_open with appropriate emojis + retry ETA
+ probe progress; render no-secret-leak pin (no `sk_live` /
`Bearer` / `password` substrings); e2e flows (full recovery cycle
5-fail → OPEN → cooldown → HALF_OPEN → 2-success → CLOSED;
**failed-recovery pin** HALF_OPEN failure re-opens with fresh
cooldown timed from second open; **intermittent-failures-don't-
trip pin** 3 failures + 1 success + 3 failures stays CLOSED at
3; replay consistency).

### Pre-flight check engine ✅ landed (`ops/preflight.py`)

**Implementation**: `src/halal_trader/ops/preflight.py` complements
the bot's startup sequence. Before the first cycle runs, the bot
needs to verify every critical dependency is wired (secrets vault,
DB schema revision, broker auth, halal screener, alert router); a
single inspectable report ("✅ all critical checks passed") is far
more reliable than reading 30 lines of startup log scrolling past.
`CheckSeverity` enum (INFO < WARN < CRITICAL) pinned string values.
`CheckOutcome` enum (PASSED / WARNED / FAILED). `CheckSpec` carries
name + description + severity (rejects empty name/description).
`CheckResult` carries spec + passed bool + non-empty message;
`outcome` property maps (passed, severity) → CheckOutcome:
passed=True → PASSED regardless of severity, failed INFO → PASSED
(info checks never fail), failed WARN → WARNED, failed CRITICAL →
FAILED. `PreflightReport` aggregates results tuple + counts +
validates counts equal results length. **`is_safe_to_start`
property returns failed_count == 0** — load-bearing safety gate
the cycle's startup sequence consults; WARN-level issues don't
block startup, only CRITICAL does. `run_checks(checks, *,
fail_fast=False)` runs each (spec, runner) tuple, captures runner
exceptions as `passed=False, message="check runner raised: <exc!r>"`,
sorts results by spec name for deterministic ordering;
fail_fast=True stops on first CRITICAL failure. `critical_failures`
+ `warnings_only` filters for kill-switch + dashboard surfaces.
Render with outcome emoji (✅/⚠️/❌) + per-section ordering
(failures → warnings → passed). **No-secret-leak pin**: render
shows spec name + operator-facing message; raw stack traces / API
responses are operator-side debug logs, not the report. Tests in
`tests/test_preflight.py` (38 cases): all enum string-value pins;
CheckSpec validation (empty name + description rejected); **outcome
mapping all four combinations** (passed/INFO=PASSED, failed/INFO=
PASSED, failed/WARN=WARNED, failed/CRITICAL=FAILED); frozen
dataclass invariants; **count-consistency invariant pin**
(passed+warned+failed must equal len(results)); **is_safe_to_start
gates only on CRITICAL pin** (warnings don't block); run_checks
empty / all-passing / one-failure / warning / info-failure;
**runner exception captured as failure pin** (no crash); sorted-by-
name determinism; **fail_fast stops on CRITICAL but not WARN pin**;
default fail_fast=False runs every check; critical_failures +
warnings_only filters; render top-line ALL GO / GO with warnings /
NO-GO; render lists failures-warnings-passed in that order; **no-
secret-leak pin** (raw API token from message stays in operator
debug log, only high-level summary in render); e2e realistic
6-check startup sequence (4 pass / 1 warn / 1 fail) with NO-GO
verdict; replay consistency.

### Settings drift detector ✅ landed (`ops/settings_drift.py`)

**Implementation**: `src/halal_trader/ops/settings_drift.py` ships
the operator's "what knobs have I tuned away from defaults?" audit.
`DriftMagnitude` enum (NONE / TINY / MODERATE / SIGNIFICANT /
EXTREME) tiers drift relative to default; for booleans any flip is
at least MODERATE; for numerics, bands are 10%/50%/100%.
`NumericBounds` lets settings declare `[min, max]` ranges; out-of-
bounds values flag as EXTREME plus surface a warning marker. `SettingSpec`
catalogues each setting with default + description + optional
bounds + `is_secret` tag. `detect_drift_for_setting` handles
boolean / string / numeric types with type-mismatch raising
TypeError. `build_drift_report` requires every catalogued setting
in current_values (operators must explicitly provide every value;
no silent defaults), but extra operator-defined env vars are
silently ignored (closed-catalogue pin). `filter_drifted` returns
entries at or above a magnitude minimum; sorted by name for
deterministic display. Render with magnitude emoji
(✅/🟢/🟡/🟠/🔴) + out-of-bounds marker; **secrets render as
`<secret>` placeholder, never the actual value**. Long strings
truncated to 37 chars + `...`. Tests in
`tests/test_settings_drift.py` (55 cases): DriftMagnitude enum
string-value pin; NumericBounds boundary inclusive both directions;
**bounds-on-string rejected pin**; **default-out-of-its-own-bounds
rejected pin**; boolean drift (no-drift / flip = MODERATE / non-bool
TypeError); string drift (no-drift / change = MODERATE /
**empty-to-set or set-to-empty = SIGNIFICANT for credentials pin**);
numeric drift across all 5 magnitude bands (5%=TINY, 30%=MODERATE,
75%=SIGNIFICANT, 200%=EXTREME); negative-direction symmetric;
**default=0 special-cased to absolute-magnitude bands**; int default
works; bool not treated as int 0/1; **out-of-bounds always EXTREME
pin**; bounds boundary inclusive; DriftEntry + DriftReport validation
including out_of_bounds-as-subset-of-drifted invariant; build_report
with no drift / some drift / **missing setting raises KeyError pin**;
**extra keys silently ignored pin**; entries sorted; deterministic;
filter_drifted defaults include TINY+; minimum=SIGNIFICANT excludes
TINY+MODERATE; render entry with emoji + secret placeholder + out-
of-bounds marker + long-string truncation; render report no-drift +
with-drift; **render no-secret-leak pin** (sk_live key never in
output, only `<secret>`); e2e flows (realistic 5-setting audit with
secrets correctly masked + `max_position_pct=5.0` typo caught as
EXTREME+out-of-bounds; replay consistency).

### 8.E aux — Incident response state machine ✅ landed (`ops/incident_response.py`)

**Implementation**: `src/halal_trader/ops/incident_response.py` is
the incident-lifecycle complement to Wave 8.E's alert router.
`Severity` enum (SEV1 / SEV2 / SEV3 / SEV4) with pinned ack SLAs:
SEV1=5min, SEV2=30min, SEV3=4h, SEV4=24h. `IncidentStatus` enum
(OPEN → ACKNOWLEDGED → MITIGATED → RESOLVED → POSTMORTEM_PUBLISHED)
forward-only with module-level `_STATUS_ORDER` tuple. `IncidentPolicy`
ships defaults (operator-tunable); rejects zero ack SLAs + missing
severity entries + zero postmortem deadline. `Incident` dataclass
enforces per-status attribution requirements: OPEN must NOT have
acker/resolver/author; ACKNOWLEDGED requires acker; RESOLVED requires
acker+resolver; POSTMORTEM_PUBLISHED requires all three (the
load-bearing audit-trail pin). Lifecycle functions (`acknowledge`,
`mitigate`, `resolve`, `publish_postmortem`) enforce forward-only
one-step progression. **Pinned: postmortem required for SEV1+SEV2
only**; `publish_postmortem` raises `PostmortemNotRequiredError` for
SEV3+SEV4 (operators can't accidentally bloat the postmortem queue
with low-severity items). `is_ack_overdue` enforces severity-driven
SLAs both directions (5min boundary inclusive: 5min not overdue,
6min overdue); ACKNOWLEDGED+ never ack-overdue. `is_postmortem_overdue`
fires only for sev1/sev2 RESOLVED past 7-day deadline (boundary
inclusive); already-published never overdue; non-RESOLVED states
never overdue. `filter_overdue` combines both flavors + sorts SEV1
first. Tests in `tests/test_incident_response.py` (65 cases): 4
Severity + 5 IncidentStatus enum string-value pins; **roadmap-pinned
SLA defaults** (5min/30min/4h/24h + 7d postmortem); policy
validation (zero SLAs + missing severities + zero postmortem all
rejected); severity_outranks strict; Incident validation including
**per-status attribution pin both directions** (OPEN must not have
attribution; ACKNOWLEDGED requires acker; RESOLVED requires acker
+ resolver; POSTMORTEM_PUBLISHED requires all three); open_incident
+ acknowledge + mitigate + resolve forward-only with skip rejected;
publish_postmortem SEV1+SEV2 supported, **SEV3+SEV4 raise
PostmortemNotRequiredError carrying severity**; ack-overdue
boundaries inclusive both directions per SLA per severity (5min,
30min, 4h, 24h); ACKNOWLEDGED+ never ack-overdue; postmortem-overdue
sev1+sev2 past 7d, sev3+sev4 never overdue, published never overdue,
non-RESOLVED never overdue; filter_overdue combines both flavors +
sorts SEV1-first; render with severity emoji 🔴/🟠/🟡/🔵 + status
emoji ⚠️/👀/🛡️/✅/📋 + attribution shown when set; no-secret
regression (no stack/traceback/api_key/bearer); e2e flows (full
SEV1 lifecycle 2min ack → 30min mitigate → 1hr resolve → 3-day
postmortem; SEV1 unacked at 6min triggers ack-overdue; SEV3 no
postmortem required raises if attempted; replay consistency).

### 8.F — Backups (DB + replay store) with point-in-time recovery  ✅ landed (runbook)

Postgres WAL archiving + base backups + restore drills. RPO < 5 min,
RTO < 30 min.

**Landed (runbook):** `docs/runbooks/backups-and-pitr.md` is the
on-call's restore-drill procedure with **RPO ≤ 5 min** / **RTO ≤
30 min** targets. Six sections: what's backed up (the audit-
trail tables in the single Postgres database — every per-trade /
per-decision / per-screening row); what's NOT backed up (`.env`
secrets and broker keys are operator-local; broker positions are
authoritative on the broker side; HuggingFace model cache is
re-downloadable); architecture (WAL archiving via
`archive_command` + daily base backups via `pg_basebackup` or
pgBackRest, both shipped to remote storage nightly); setup walks
the operator through enabling `wal_level=replica` +
`archive_mode=on` in the Postgres container, taking the first
base backup, scheduling daily backups via host cron, and
syncing both archives off-machine via rclone; restore covers
both full disaster recovery (stop bot → wipe data dir → extract
latest base → recovery.signal + restore_command → restart →
verify max(timestamp)) and point-in-time recovery (same flow but
with `recovery_target_time` set explicitly); restore drill
procedure runs quarterly via a side container so the operator
verifies the chain on a cadence with `docs/postmortems/drills/`
records of every drill outcome. Targets table pins RPO / RTO /
retention values + drill cadence; drill failure modes section
covers the three most-common drill failures (empty rows from
broken WAL chain; Postgres won't start from missing WAL
segments; max-timestamp earlier than expected from archive lag).
Halal-specific considerations call out two edge cases — restoring
past a halal-screening update can revert a `not_halal` to `halal`
unless the operator re-runs `halal-trader crypto screen`, and
restoring past a purification `paid_at` can make settled entries
look outstanding. Post-restore checklist is the seven-item
"before resuming" pin: db migrate head check, halal screen
refresh, position reconcile (already runs on startup), halt
status, model fingerprint match, and audit-record-keeper
notification. README runbooks index gains a new "Operations
playbooks" section so the doc is discoverable from the
runbooks/README.md alongside the alert-triggered runbooks. Pure
docs — no code change.

### 8.G — Status page (`status.halal-trader.dev`)  ✅ landed (snapshot aggregator)

Public uptime + incident history. Drives operator + user trust.

**Landed (snapshot aggregator):** `core/status_snapshot.py` ships
`build_snapshot(*, halts, cycle_events, now, thresholds)` →
`StatusSnapshot` ready to publish. Two append-only input streams
(`HaltRecord`s + `CycleEventRecord`s — both already produced by
the bot) compose into a four-level traffic light: OPERATIONAL
(no halt + ≥99% success), DEGRADED (95-99% OR a recent resolved
incident), PARTIAL_OUTAGE (80-95% OR a recent halt > 5min),
MAJOR_OUTAGE (currently halted OR <80% success). Pinned
classification decision tree: ongoing halt → MAJOR_OUTAGE
unconditionally; <80% success → MAJOR_OUTAGE; long-halt-in-window
flips to ≥ PARTIAL_OUTAGE regardless of success rate; the
success-rate ladder otherwise. The thresholds dataclass validates
its own ladder ordering (a config that put degraded below
partial would silently classify everything as PARTIAL_OUTAGE).
Pinned safety semantics: empty stream → OPERATIONAL with
success_rate=1.0 (no-data is "no incidents", not "no signal");
ongoing halt is the headline incident, NOT also in the historical
list (avoids double-counting in the public render); halt reasons
pass through `filter_reason` which fully redacts to "halt for
operational reasons" on any of five sensitive substrings
(`api_key / secret / token / password / operator` —
case-insensitive, full replacement rather than partial because
key tails leak via length under partial redaction). 80-char cap
on long reasons with `…` suffix. Halal alignment baked in:
exposes only anonymised operational metadata; no per-trade data,
operator identifiers, broker keys, or LLM prompts; the denylist
mirrors `otel_translator.py` so a future expansion stays
consistent. Pure-Python; no DB / network. `render_snapshot`
produces a text payload with the emoji-prefixed status, an
"⚠️ Ongoing" line if active, and a "Recent incidents" list (or
"No incidents in the window" on a clean record). 43 tests cover
threshold validation (misordered ladder + negative-recent-halt-
minutes + zero-window-days rejections), HaltRecord / CycleEvent
input validation, the `filter_reason` denylist (every sensitive
substring + clean-text passthrough + length cap + empty-input
+ case-insensitive match), the empty-streams → OPERATIONAL
contract, ongoing-halt-yields-MAJOR-OUTAGE + ongoing-halt-not-in-
historical-list pins, the success-rate ladder across all four
bands (99% / 97% / 90% / 70%), the long-halt-flips-up rules
(short halt → DEGRADED, long halt → PARTIAL_OUTAGE, long halt
+ low success rate → MAJOR_OUTAGE), window-inclusion (halts and
cycles outside the 7d window excluded; custom window changes
inclusion), incidents-sorted-most-recent-first, the reason filter
applied to incident summaries, summary line content (emoji,
halt mention when active, cycle count, "no cycle data"
fallback), the documented string values for `StatusLevel`,
frozen-dataclass immutability, and render output (summary
emoji + ongoing-line + incident list + clean-window message).
Cycle-side wiring (a route handler that pulls the latest halts
+ cycle events from Postgres and serves the snapshot at
`/api/status/public`, plus the dashboard / `status.halal-trader.dev`
page that renders it) deferred to a follow-up — aggregator
verified in isolation first.

### 8.H — Security audit + responsible disclosure policy  ✅ landed (policy + threat model)

External security audit (one-time). Bug bounty program via HackerOne
or HackerOne-alternative.

**Landed (policy + threat model):** `SECURITY.md` ships at the
repo root with seven sections: TL;DR, threat model (assets the
bot defends + explicit "what we're NOT defending" — multi-tenant
isolation, side-channel on shared cloud, rationale confidentiality,
DoS), secure defaults table (every reasonable two-choice default
is the safer one — paper / testnet / empty WEB_API_TOKEN /
require-confirmation / STRICT halal consensus / kill-switch
engages on first error chain), disclosure section with the
maintainer email + 72h-acknowledge / 7d-triage / 30d-fix-for-high /
90d-otherwise SLA, in-scope vs out-of-scope itemised list (in:
secret extraction / halal screener bypass / halt-switch bypass /
audit-trail tampering / cross-broker secret leakage / paper-to-
live escalation / pre-auth code execution; out: brute force of
operator-set passwords / social engineering / dependency CVEs <
30d / upstream-service issues / rate-limit-only / on-host attacks /
cloud side-channel), security-relevant features by Round-4 wave
(cross-links the existing Wave 2.A signing / 2.B consensus / 2.C
profiles / 4.I anomaly / 4.F promotion / 8.B chaos / 8.E alerts /
6.E fingerprint / 6.B feature schema), an operator-hardening
checklist (encrypted disk for `.env`, 0600 perms, localhost
Postgres, firewall, WEB_API_TOKEN, LLM cap, alert routing,
signing-key backup, no-shared-VM), halal-specific threats
(spoofed verdicts → HTTPS-only + STRICT consensus + audit;
cache poisoning → TTL + refresh + audit; override misuse →
audit log), and a "deprecated / not implemented" section with
honest pointers at the gaps (per-secret encryption at rest, CSRF
on mutation endpoints, 2FA — all routed to Wave 3 multi-user
work). README gains a SECURITY.md pointer in the "In a hurry?"
section. No code change beyond the doc + README link — the
existing safeguards (kill-switch, paper defaults, testnet, audit
trail, screening) implement what the policy describes.

**Acceptance**: A 99.9% uptime SLA is achievable with the existing
codebase + ops processes; the runbooks cover every failure mode
encountered in chaos testing.

---

## Wave 9 — Documentation excellence (2-3 weeks)

The best app needs the best docs.

### 9.A — Quickstart that gets a new user trading in < 10 minutes  ✅ landed

Replace the existing README with a guided quickstart. Real-time
console output checked into the docs site for credibility.

**Landed:** `docs/QUICKSTART.md` — a step-by-step "fresh clone → first
paper trade in 10 minutes" guide using Binance testnet (free, no KYC,
fake money). Covers prerequisites with install commands, a 4-line
`.env` minimum (one of three LLM backends), `pg-up` + `db migrate`,
the single-cycle `--once` path with sample console output, the
dashboard browser walk-through, and a troubleshooting table for the
top failure modes (`SchemaError`, port-5433 not running, Ollama
refused, halal-screener rate-limit, Binance `-1013`, halt-engaged).
README links to it in a new "In a hurry?" callout above
"Prerequisites".

### 9.B — Strategy author's guide  ✅ landed

How to build a custom strategy. Walk through the abstract `BaseStrategy`
class, the prompt-version registry, the test patterns. Every code
example is a working file in `examples/`.

**Landed:** `docs/STRATEGY_AUTHORING.md` is a single-page operator
guide structured around the contract — strategy chooses, executor
enforces — then the `BaseStrategy` orchestration (the override
points are `_build_prompts` + `_validate_plan`), the schema the
LLM must return, the prompt-version registry's stable-SHA invariant
(every byte change → version change → audit trail catches the
edit), the per-symbol halal-compliance boundary (strategy receives
already-screened pairs; never call a screener / broker from inside
`analyze`), the four test harnesses (unit-mock-LLM / stress-8-
scenarios / scenario simulator / A/B comparator with a Welch's
p < 0.05 promotion gate), a fully-worked pure-Python RSI
mean-reversion example with strategy class + five unit tests +
how to wire it into the stress harness + backtest CLI invocation,
a 7-item promotion checklist, and a "common pitfalls" section
(no disk reads in `analyze` — blocks the loop; don't catch LLM
exceptions — base class already does the retry-then-fail-soft;
don't mutate input lists; don't introduce un-seeded randomness;
don't flip `BINANCE_TESTNET` / `ALPACA_PAPER_TRADE` — the bot is
paper only by design). README's "In a hurry?" callout gains a
parallel pointer so first-time contributors find the guide
immediately. The doc cross-links every Round-4 testing wave that
strategy authors will use (5.B / 5.F / 7.B) so the guide acts as
a tour of the recently-landed research toolkit.

### 9.C — Halal compliance handbook  ✅ landed (handbook)

`docs/halal_jurisprudence.md` exists. Expand to a full handbook:
the rulings the platform follows, the AAOIFI standards referenced,
the scholar-profile diff. Reviewed quarterly by a paid Shariah
advisor.

**Landed (handbook):** `docs/halal_jurisprudence.md` expanded from
58 → 498 lines into a nine-section handbook. **Section 1**
(foundational prohibitions) covers Riba / Maysir / Gharar /
prohibited industries with concrete bot-implementation pointers
(`crypto/screener.py` rules + Zoya GICS sectors + the spot-only /
testnet-only mechanical guarantee against leverage Maysir).
**Section 2** (AAOIFI ratios) walks the three financial cuts the
default profile applies — debt / non-permissible-income / cash-
and-receivables — with the rationale for each (why
backward-looking averages, why 5% income tolerance, why
cash-and-receivables is informational by default). **Section 3**
(asset-class rulings) covers equities + crypto (using Mufti
Faraz Adam's 4-pillar framework as the explicit source) +
commodities + sukuk + REITs + international equities, with
forward references to the upcoming Waves 1.G / 1.H / 1.I / 1.J
that wire each integration. **Section 4** (decision states)
formalises `halal` / `doubtful` / `not_halal` with the bot
behaviour and the conservative-wins-on-ties rule.
**Section 5** (purification) cross-links to Wave 2.D's scheduler
and the per-trade / per-dividend purification helpers, with the
"never auto-mark paid" pin that Wave 2.D enforces. **Section 6**
(scholar profiles) is the new Wave 2.C surface: aaoifi_default /
taqi_usmani / delorenzo_djim with the rationale for each, the
three consensus policies (STRICT / MAJORITY / WEIGHTED), and the
empty-input-defaults-to-doubtful safety. **Section 7** (exception
queue) documents the Approve / Reject / Defer triage and the
"approval doesn't auto-promote" pin. **Section 8** (audit trail)
documents every field a future scholar reviewer can use to
replay a decision. **Section 9** (limits + disclaimers) makes the
handbook honest about what it is — a methodology, not a personal
fatwa, with explicit pointers at the kill-switch's role as
risk-control rather than Sharia-compliance and the paper-only
default. References section cites the source standards (AAOIFI 21,
AAOIFI 30, Mufti Taqi Usmani's *Introduction to Islamic Finance*,
Mufti Faraz Adam's framework) plus a pointer to register custom
scholar profiles. The "Last reviewed" footer is set with a note
pending external scholar sign-off.

### 9.D — Operator runbooks (per Wave 8.E)  ✅ landed (5 more runbooks)

Every alert has a one-page runbook: symptoms, root causes, immediate
mitigations, long-term fixes.

**Landed (5 more runbooks):** Building on the four seed runbooks
from Wave 8.E (halt-engaged / chain-backoff / drift-breach /
broker-api-error-rate), added five more covering the remaining
alert classes the bot can raise: **cycle-stuck.md** (PAGE — cycle
exceeded `asyncio.wait_for(interval × 2)` watchdog; diagnose via
the last START-without-END stage in `cycle.stage.*` events;
mitigate via halt + restart, Postgres `pg_terminate_backend` for
stuck transactions, Ollama restart for LLM-backend hang); **llm-
circuit-breaker.md** (PAGE — `FallbackLLM` chain in extended
backoff; covers all-providers-down / fallback mis-config / cost-
cap-hit / quota-throttling / local-Ollama-died; mitigate via halt
+ provider status checks + cost-cap reset path); **db-connection-
lost.md** (PAGE — repeated OperationalError; covers container-
stopped / network-partition / connection-exhausted / disk-full /
auth-broken; mitigate via container restart + connection
termination + WAL log cleanup); **halal-screener-stale.md** (WARN
— halal cache age past `HALAL_CACHE_MAX_AGE_HOURS`; covers
screener-API-down / auth-broken / rate-limited / refresher-
crashed; mitigate paths plus the explicit pin that "stale cache
is safer than no cache" because STRICT consensus + doubtful-on-
unknown-symbol keep the operator safe); **snapshot-store-
failure.md** (WARN — replay snapshot persist failed; covers
Postgres-write-failure / JSON-serialisation-error / schema-drift
/ FK-violation; pinned the audit-trail "missing snapshot is a
gap not a corruption — don't try to backfill" invariant). The
runbooks/README.md index updated to list all nine entries with
their severities. Each runbook follows the five-section template
(likely causes / diagnose / mitigate / escalate / postmortem)
with concrete `bash` / `psql` / `curl` commands the on-call can
copy-paste, plus an "Acknowledgement window" + "Last reviewed"
footer; PAGE-severity runbooks include a postmortem-filing
requirement. Pure docs — no code change. The set is now
substantial enough that the alert router (Wave 8.E) can route
every alert with a `runbook_url` field that resolves to one of
these instead of the placeholder "(none yet — write one in
docs/runbooks/)".

### 9.E — Architecture deep-dive videos ✅ landed (`web/video_catalogue.py`) — topic catalogue + production state

Record 5-10 minute videos walking through key parts of the architecture:
the cycle pipeline, the halal screener, the LLM ensemble, the replay
store. Posted publicly.

**Implementation**: `src/halal_trader/web/video_catalogue.py` ships
the pure-Python topic registry + production state machine. The
recording / editing / hosting pipeline is operator-side; this
module ships the planning + lifecycle. `TopicArea` enum (8 closed-
set areas covering the four roadmap-named topics plus PURIFICATION_LEDGER,
KILL_SWITCH, BROKER_PLUGIN, OBSERVABILITY for natural expansion).
`ProductionStatus` enum (DRAFTED → RECORDED → EDITED → PUBLISHED)
with module-level `_PRODUCTION_ORDER` tuple for forward-only
progression. **5-10 minute duration band pinned per roadmap** —
4min59s and 11min both rejected at construction. `VideoTopic`
carries topic_id + title + area + estimated_duration + prerequisites
frozenset + status + drafted_at + last_status_at; rejects self-prereq
(trivial cycle) at construction. `draft_topic` creates DRAFTED state.
`advance_topic` enforces strict forward-only one-step-at-a-time;
cannot skip DRAFTED → EDITED, cannot skip to PUBLISHED, cannot
backtrack from RECORDED → DRAFTED, **PUBLISHED is terminal** (content
revisions require new topic record so audit trail of "what was
published when" stays clean). `is_publishable` returns True only for
PUBLISHED. `assert_no_prereq_cycle(topics)` walks the prereq graph
detecting cycles (handles 2-topic A→B→A and 3-topic A→B→C→A cases;
external prereqs not in the batch are skipped, not raised — operators
iteratively add topics). `watch_path(target_id, *, topics)` returns
topological-order watch path that ends at target; handles diamond
DAG (A→B, A→C, B+C→D includes A exactly once); detects cycles
during traversal. `total_runtime` sums durations; `published_topics`
returns sorted-by-id PUBLISHED only. Render with area emoji
(🔄/✅/🧠/💾/🤲/🛑/🔌/📊) + status emoji (📝/🎥/✂️/🚀). No-secret
regression. Tests in `tests/test_video_catalogue.py` (49 cases): 8
TopicArea + 4 ProductionStatus enum string-value pins; **5-min and
10-min duration boundaries inclusive both directions** (load-bearing
roadmap pin); 4:59 and 11:00 rejected; **self-prereq trivial cycle
caught at construction**; lifecycle DRAFTED→RECORDED→EDITED→PUBLISHED
with skip + backtrack + post-PUBLISHED-terminal all rejected;
is_publishable across all 4 statuses pinned; cycle detection
(linear chain + diamond DAG OK; 2-topic + 3-topic cycles caught);
external-prereq skipped pin; watch_path for no-prereqs / linear /
diamond DAG (A appears once, before B/C/D); unknown-topic raises;
external-prereq skipped; cycle in watch_path raises;
total_runtime + published_topics sorted; render with area + status
emojis + duration in min (7.5 min); no-prereqs renders as `—`;
no-secret regression (no api_key / youtube / vimeo / transcript /
bearer); e2e flows (full lifecycle for cycle pipeline video; **4-
topic catalogue with roadmap-named topics builds DAG correctly with
total runtime 30 min**; replay consistency).

### 9.F — API reference (auto-generated) ✅ landed (introspection core)

Sphinx + autoapi over `domain/`, `core/`, every public module.
Versioned with the codebase.

**Landed (introspection core)** as `core/api_reference.py` —
pure-Python module-introspection-based API documentation
generator. Picked module-introspection over Sphinx-autoapi for
the core because (a) the operator's docs build pipeline can plug
in any renderer (markdown, ReST, JSON, OpenAPI), (b) the
introspection result is testable + regression-pinnable in a way
that Sphinx output isn't, (c) the existing wave-pattern is "ship
the data structure, defer the rendering" — Sphinx integration is
an operator-side build script. Five-value `SymbolKind` enum
(CLASS / DATACLASS / FUNCTION / ENUM / CONSTANT). `ApiSymbol`
frozen dataclass carries qualified_name + short_name + kind +
summary + description + signature. `ApiModule` aggregates per-
module symbols. `ApiReference` is the top-level package surface.
`extract_module_reference(module, *, include_private=False)`
walks a module via `inspect`, classifies attributes, extracts
docstrings via `inspect.cleandoc`, and emits the structured
result. `extract_api_reference(*, package_name, modules)` covers
multiple modules with deterministic alphabetical sorting.
Five pinned semantics: (1) **public-by-default** — symbols whose
names start with `_` are skipped unless `include_private=True`;
(2) **deterministic ordering** — symbols sorted alphabetically
within a module, modules sorted by qualified name (operators
diffing two reference outputs see real changes, not iteration-
order noise); (3) **`__all__` honoured when present** — a module
declaring `__all__` exports only the listed symbols; (4)
**imported symbols not re-documented** — `_is_module_local`
filters out symbols whose `__module__` differs from the current
module (regression-pinned with the kyc.py module which doesn't
re-document `dataclass` even though it imports it); (5) **render
output never includes operator secret values** — constants whose
name matches the secret-name denylist (`password` / `api_key` /
`secret` / `token` / `private_key` / `session_id`) render as
`<redacted>` in markdown. Mirrors no-secret patterns of Wave 8.D
OTLP + Wave 3.B vault + Wave 12.G co-pilot. Signature extraction
per kind: function → `func_name(params)` via `inspect.signature`;
dataclass → `Name(field: type, ...)` via `dataclasses.fields`;
enum → `Name(MEMBER_A, MEMBER_B)` via `__members__`; class →
empty (intentional simplicity for non-dataclass non-enum);
constant → empty (the value is rendered separately, redacted if
the name matches the denylist). Halal alignment: read-only
introspection; never opens a position; pure-Python (stdlib
`inspect` + `dataclasses` + `enum` + `types`); no DB / network /
async. 41 tests cover symbol classification (function / class /
dataclass / enum / constant); public-by-default + include_private
flag flow; **imported-symbol filtering pin** via the real kyc.py
module; **`__all__`-honoured pin** via synthetic module with
`__all__ = ["fn_a"]`; docstring extraction (module summary; first-
line summary; full description; no-docstring → empty); signature
shapes (function with params; dataclass with field list; enum
with members; class empty); deterministic ordering pins (symbols
alphabetically; modules by qualified_name); ApiReference +
ApiModule + ApiSymbol field validation; frozen-dataclass
immutability; SymbolKind string values pinned for JSON / DB
stability; render output (module header; kind groups; signature
in code block; summary; **secret-constant redacted pin** asserts
`<redacted>` substituted and the original `sk-secret-leaked`
value never appears; empty reference; module with no symbols;
function signature renders correctly); end-to-end extraction of
real modules (web/kyc; web/privacy multi-module reference; module
summary + symbols have descriptions). Sphinx / MkDocs build
script that consumes `extract_api_reference` and writes the
rendered output to `docs/api/`, the CI step that emits a warning
when a public symbol gains/loses a docstring (regression-pinning
docstring coverage), and the `just docs-build` recipe that
regenerates the full API reference deferred to follow-ups —
introspection core verified in isolation first.

**Acceptance**: A motivated new contributor can ship their first PR
within their first day, using only the docs.

---

## Wave 10 — Community + ecosystem (4-6 weeks)

Turn the platform into a movement.

### 10.A — Public strategy gallery ✅ landed (curation engine)

Operators voluntarily publish their strategies + performance. Sortable
by Sharpe, by halal-compliance strictness, by simplicity. Forkable
into the operator's own account.

**Landed (curation engine)** as `web/strategy_gallery.py` —
pure-Python publication-curation engine with anonymised author
tokens, halal-strictness rating, simplicity score, fork-lineage
tracking, and pre-publication safety gates. Three-tier
`StrategyVisibility` (PRIVATE default / PUBLIC_UNLISTED /
PUBLIC_LISTED); four-tier `HalalStrictnessLevel` (BASIC /
MODERATE / STRICT / MAX_STRICT) — closed enum the gallery sorts
on; `PublicMetrics` carries Sharpe, win rate, max drawdown, total
trades, time period (every metric required for sortability).
`StrategyEntry` carries strategy_id + anonymous_author +
name/version/summary + halal_strictness + simplicity_score +
visibility + optional metrics + optional parent_fork_id +
opt_in_publication flag. `validate_for_publication(entry)` is
the publication gate that raises `GalleryViolationError` on
unsafe entries. `compute_simplicity_score(*, lines_of_code,
symbol_list_size, reasoning_depth)` produces a 0-100 score:
50 LOC + 5 symbols + depth 1 → 100; 100 LOC + 5 symbols + depth
2 → ~92; 500 LOC + 50 symbols + depth 5 → ~32; 1000+ LOC +
100+ symbols + depth 10+ → 0; clamped at [0, 100].
`assemble_lineage(entries, *, target_id)` walks the fork chain
back to its root with cycle detection. `hash_author(user_id, *,
salt)` produces deterministic salt-hashed author tokens
(`anon-{16hex}`) sharing the construction with Wave 10.B open-
dataset anonymisation. Six pinned semantics: (1) **PRIVATE
default visibility** — `StrategyEntry.__init__` defaults to
PRIVATE; publication gate refuses non-PRIVATE without
`opt_in_publication=True`; (2) **anonymous author token** — same
HMAC-SHA256 salt-hash as Wave 10.B; same salt produces same
token, different salt produces different token (re-publication
anti-linking pin); (3) **PII denylist on summary + name** — same
five regex patterns as Wave 10.B (email / SSN / IP / phone /
ETH-address); summary/name with PII REJECTED at validation
(rather than auto-redacted) so the operator notices and rewrites;
(4) **halal-strictness closed enum** — operator declares
strictness level; the gallery sorts on it; (5) **simplicity
score** — heuristic-based load-bearing pin; simpler strategies
sort to top so new operators see approachable examples first;
(6) **render output never includes operator's raw user_id /
portfolio / wallet addresses** — only the anonymous_author token;
mirrors no-PII patterns of Wave 11.D + 11.C + 3.B + 10.B. Cycle
detection in fork lineage raises `GalleryViolationError` rather
than infinite-loop. Halal alignment: read-only curation; never
opens a position; pure-Python (stdlib `hmac` + `hashlib` + `re`
+ `dataclasses` + `enum`); no DB / network / async. Frozen
dataclasses on every output. 65 tests cover PublicMetrics field
validation; StrategyEntry field validation; **default
visibility = PRIVATE pin**; **default opt_in_publication =
False pin**; hash_author semantics (5 pins covering deterministic-
within-salt + different-users + different-salts + short-salt
rejection + empty-user-id rejection); simplicity-score scoring
(boundary cases at 50/100/500/1000+ LOC; clamping at 0 and 100;
input validation for negative LOC / symbols / depth);
**publication gates** (PRIVATE passes silently; PUBLIC without
opt-in raises; PUBLIC with opt-in passes; PII in summary
rejected for email + SSN + ETH-address; PII in name rejected;
clean summary passes; PUBLIC_LISTED without metrics raises;
PUBLIC_UNLISTED without metrics passes — the asymmetry pin);
fork lineage walking (root only; 2-level; multi-level chain;
target-not-found raises KeyError; **detached parent stops walk**;
**cycle raises GalleryViolationError** — not infinite-loop);
frozen-dataclass immutability; all enum string values pinned for
JSON / DB stability; render output (anonymous_author shown;
visibility emoji per kind 🔒🔗🌍; strictness emoji per level
🟢🟡🟠🔴; metrics shown when present, omitted when None;
parent_fork_id shown when set, omitted when None; **render-no-
PII pin** asserts `user_id` and `@` never in render output;
simplicity score formatted to 1 decimal); end-to-end realistic
flows (typical publication lifecycle: hash + score + entry +
validate; full 3-deep fork lineage walk; publication blocked for
PII in summary — operator must rewrite; simplicity ordering for
gallery sort regression-pinned). Postgres `gallery_strategies` +
`gallery_audit_log` tables wired to `db/repository.py`, the
FastAPI route adapters (`/api/gallery` listing public strategies
with sort + filter; `/api/gallery/{strategy_id}` returning entry
+ lineage; `/api/gallery/publish` for operator's opt-in flow with
`validate_for_publication` invoked before persist), the dashboard
gallery tile rendering top-N by Sharpe / strictness / simplicity,
the fork-flow that copies a published strategy into the
operator's account preserving lineage, and the per-strategy
discussion threads / star count / fork count features deferred
to follow-ups — engine verified in isolation first.

### 10.B — Open dataset of halal trading decisions ✅ landed (anonymisation engine)

The aggregated, anonymised LLM decision history becomes a public
dataset. Academic research community downloads it, cites the platform.
Halal-finance researchers get a goldmine of empirical data.

**Landed (anonymisation engine)** as
`web/dataset_anonymisation.py` — pure-Python anonymisation engine
for the public research dataset. `RawDecision` carries the
operator's raw row (decision_id / user_id / timestamp / symbol /
sector / regime / action / notional_usd / rationale).
`anonymise_decision(raw, *, salt, policy)` produces an
`AnonymisedDecision` with operator-identifying fields stripped.
`anonymise_dataset(decisions, *, salt, policy)` runs the full
pipeline + k-anonymity filter. Six pinned semantics: (1) **salt
required (≥16 bytes)** — anonymisation refuses short / empty
salt; without it the user_id hash is reversible; (2) **deterministic
hash within a single salt** — researchers correlate same-anon-
user decisions within an export; across exports with different
salts the mapping changes (re-exports can't be cross-referenced);
(3) **PII denylist on free-form text** — five regex patterns
(email, SSN, IP, phone, ETH-address-shaped) replaced with
`<redacted-{kind}>`; pinned via test for each pattern + the
clean-rationale-unchanged invariant; can be disabled via
policy for already-clean data; (4) **k-anonymity floor (default
k=5)** — rows whose quasi-identifier (anonymous_user, sector,
regime) tuple appears in <k other rows are dropped. k=1 disables
the check (operator opt-out for non-public internal exports);
strict k=10 drops rows that survive k=5; (5) **timestamp rounded
to hour** — 14:35:12 → 14:00:00; reduces timing-correlation
attack surface; (6) **USD bucketed into five tiers** (MICRO <
$100; SMALL $100-1k; MEDIUM $1k-10k; LARGE $10k-100k; WHALE >
$100k); boundary $100 inclusive into SMALL; $99.99 in MICRO.
HMAC-SHA256(user_id keyed by salt) → first 16 hex chars (64-bit
collision-resistant). Halal alignment: read-only export; never
opens a position; pure-Python (stdlib `hmac` + `hashlib` + `re`
+ `collections.Counter`); no DB / network / async; frozen
dataclasses on every output. Render summary shows count + bucket
distribution only — never raw user_ids / rationales / notionals;
**no-raw-data render pin** asserts `alice-secret-id` and
`SECRET-OPERATOR-TEXT-XYZ` never appear in render. 53 tests
cover policy validation; RawDecision field validation
(empty-rejected; naive-tz rejected; negative notional rejected);
salt requirement (short rejected on both helpers; 16-byte
accepted); user-id hash semantics (token starts with `anon-`;
contains no original ID; deterministic within salt; different
users → different tokens; **different salts → different tokens**
— the no-cross-export-correlation pin); timestamp rounding
(14:35:12 → 14:00:00; top-of-hour unchanged); USD bucketing
across all five tiers + boundary inclusivity in both directions;
**PII redaction in five patterns** (email; SSN; IP;
ETH-address; clean-rationale-unchanged); PII redaction
disable-via-policy flow; symbol+sector+regime+action preserved;
**k-anonymity filter** (5 same-QID rows kept under k=5; 4
dropped; k=1 keeps all; QID grouping by (anon-user, sector,
regime) — different sectors form different groups; strict k=10
flips); empty-dataset returns empty result; frozen-dataclass
immutability across all four output types; USDBucket string
values pinned for JSON / DB stability; render-summary
(count / dropped / bucket distribution); **no-raw-data render
pins** for user_id + rationale; full-dataset realistic flow
(100 decisions across 3 users in 2 sectors → all 30 kept under
k=5); end-to-end PII-scrubbing pipeline (email + IP + SSN +
ETH-address all redacted simultaneously). Postgres
`research_dataset_exports` audit table, the FastAPI route
adapter (`/api/research/dataset/{export_id}.csv` returning the
anonymised CSV with the salt + k-floor metadata in headers),
the operator console for triggering exports with the policy
parameters, and the public-dataset cite-able DOI registration
flow deferred to follow-ups — engine verified in isolation
first.

### 10.C — Discord / Slack community ✅ landed (`web/moderation.py`) — moderation engine

Active community space for halal traders. Operators share strategies,
discuss compliance edge cases, request features. Moderated by the
core team.

**Implementation**: `src/halal_trader/web/moderation.py` ships the
pure-Python content classifier + review state engine the Discord/
Slack bot integrations consume. `ContentClassification` enum
(CLEAN / SPAM / HARASSMENT / FINANCIAL_ADVICE / PII_LEAK) +
`ReviewStatus` enum (PENDING / AUTO_APPROVED / FLAGGED / ESCALATED
/ REMOVED). `ModerationPolicy` ships 3-message/60s spam threshold +
`detect_pii=True` default. `classify(text, *, recent_identical_count,
policy)` returns `ClassificationResult` with classification +
flagged_phrases + severity_score [0, 1]. **Pinned priority order:
PII_LEAK > HARASSMENT > FINANCIAL_ADVICE > SPAM > CLEAN.** PII is
the load-bearing short-circuit — detected via 5 regex patterns
(email / SSN / IP / phone / 40+-char alphanumeric API-key shape) +
auto-removed without moderator review (the channel never sees the
leak). Harassment routes to ESCALATED for human review. Financial
advice → FLAGGED but visible (transparent moderation: stays in
channel with disclaimer rather than silently deleted). Spam → REMOVED
auto. `auto_decide` runs the classification → status mapping; only
fires from PENDING. `moderator_remove` and `moderator_approve` are
the human-review escape valves; both require non-empty moderator
name and only accept FLAGGED or ESCALATED inputs (cannot reach in
to PENDING). `is_visible_to_channel` returns True for AUTO_APPROVED
+ FLAGGED only — PENDING is hidden until auto_decide runs;
ESCALATED + REMOVED are hidden. Render with classification emoji
(✅/📧/⚠️/💼/🔓) + status emoji (⏳/✅/🚩/📣/🗑️). No-secret
regression: render shows flagged_phrases (the labels) but never the
original message — operators reviewing the moderation log see the
pattern that triggered without re-reading every PII leak. Tests in
`tests/test_moderation.py` (69 cases): 5 ContentClassification + 5
ReviewStatus enum string-value pins; ModerationPolicy validation
(spam_threshold below 2 + zero spam_window rejected; immutable);
ClassificationResult validation (severity_score [0,1] range
inclusive both directions; immutable); classify CLEAN for empty +
normal + strategy discussion; classify PII for each of 5 patterns
(email / SSN / IP / phone / API-key-shape with realistic 58-char
"Binance" decoy); **PII takes priority over harassment + financial
advice** (load-bearing short-circuit pin); detect_pii=False allows
fallthrough to financial advice; classify HARASSMENT across 3
phrasings; harassment-priority-over-financial-advice pin; classify
FINANCIAL_ADVICE across 6 phrasings (you should buy/sell, guaranteed,
risk free, definitely, recommend); SPAM at 3-message threshold
inclusive boundary, 2 still CLEAN; financial-advice-priority-over-
spam pin (a spammed advice message stays as advice); custom strict
2-threshold; MessageReview validation including escalated/removed-
require-moderator pin; auto-approved no-moderator-required;
immutable; initial_review starts PENDING; auto_decide CLEAN→AUTO,
PII→REMOVED+auto-moderator (load-bearing), HARASSMENT→ESCALATED+auto,
**FINANCIAL_ADVICE→FLAGGED (visible+transparent pin)**, SPAM→REMOVED+auto;
already-decided rejected; moderator_remove from FLAGGED + ESCALATED
+ rejects PENDING + requires moderator name; moderator_approve
escape valve from FLAGGED + ESCALATED; is_visible_to_channel returns
true for AUTO_APPROVED + FLAGGED only, false for PENDING + ESCALATED
+ REMOVED; render with emojis + flagged phrases visible + raw text
absent (no `@` no `123-45` from masked PII labels); e2e flows
(API-key paste blocked immediately; financial advice stays visible
but flagged; human moderator removes flagged; replay consistency).

### 10.D — Quarterly halal-trading newsletter ✅ landed (`web/newsletter.py`) — digest builder

Highlights from the platform's aggregate data: what worked, what
didn't, regulatory changes, scholar updates. Goes to an opt-in mailing
list.

**Implementation**: `src/halal_trader/web/newsletter.py` ships the
pure-Python digest builder + opt-in subscriber state machine. The
actual mailing-list send (Mailgun/Postmark/SES) is operator-side.
`SectionKind` enum (TOP_PERFORMERS / REGULATORY / SCHOLAR_UPDATES /
WHAT_DIDNT_WORK) covers the four roadmap-named content domains.
`Section` validates non-empty fields + 120-char title cap + 4000-char
body cap. `validate_section` runs PII denylist (5 patterns: email /
SSN / IP / phone / API-key-shape) + handle denylist (@username).
`Digest` validates duplicate section IDs + tz-aware published_at.
`validate_digest` runs validation across every section before
publication (the load-bearing pre-send pin: a contributor that
pasted "alice.smith@halal-trader.dev" into a section gets caught
before the digest reaches 1000 subscribers). `Subscription` carries
subscription_id + subscriber_anonymous_handle + status +
subscribed_at + optional unsubscribed_at; UNSUBSCRIBED status
requires unsubscribed_at; SUBSCRIBED must not have it; unsubscribed
≥ subscribed. The dataclass deliberately doesn't carry the email
address — that's operator-side state, so render output is
structurally email-leak-free. `subscribe` / `unsubscribe` /
`active_subscribers` are the state machine; unsubscribe one-way
(re-subscribing means a fresh subscription record). `render_digest`
emits markdown with sections in canonical order (TOP →
REGULATORY → SCHOLAR → WHAT_DIDNT_WORK regardless of input order)
+ kind emoji headings (🏆/⚖️/📚/🔍). Tests in
`tests/test_newsletter.py` (57 cases): all enum string-value pins;
Section validation (empty fields + 120/4000 char caps + frozen);
validate_section across each PII pattern + each location (title +
body) + handle pattern (Twitter/Discord @username); clean
strategy-discussion + Sharpe quote passes; DigestViolationError
carries section_id + reason; Digest validation (duplicate section
IDs rejected; naive published_at rejected; empty sections rejected;
frozen); validate_digest across all sections; sections_by_kind
filters; **Subscription validation pin: UNSUBSCRIBED requires
unsubscribed_at + SUBSCRIBED must not have it both directions
pinned**; unsubscribed-before-subscribed rejected; subscribe basic
+ unsubscribe records timestamp + double-unsubscribe rejected;
state immutability; active_subscribers excludes unsubscribed; render
section + digest with canonical kind ordering pinned (TOP < REG <
SCH < NOPE indices); render no-secret regression (no email shape +
no `@` + no API key); e2e flows (full Q1 digest with all 4 kinds;
subscription lifecycle Q1 subscribe → Q3 unsubscribe → already-
unsubscribed raises; **PII-caught-before-send load-bearing pin** —
"alice.smith@halal-trader.dev" in section body raises during
validate_digest); replay consistency.

### 10.E — Annual halal fintech conference ✅ landed (`web/conference.py`) — planning engine

Bring together halal fintech founders, operators, scholars, academics.
Platform sponsors / hosts. Cements brand position.

**Implementation**: `src/halal_trader/web/conference.py` ships the
pure-Python conference planning state machine. `SpeakerStatus`
enum (INVITED → ACCEPTED/DECLINED → CONFIRMED → WITHDRAWN);
`SpeakerKind` (SCHOLAR / FOUNDER / OPERATOR / ACADEMIC / REGULATOR
— the five kinds the roadmap names); `SponsorTier` ladder
(PLATINUM > GOLD > SILVER > BRONZE). `Speaker` dataclass deliberately
excludes contact email/phone (operator-side state); validation
enforces INVITED has no decided_at AND non-INVITED requires it (both
directions pinned). Lifecycle transitions: `invite_speaker` →
`accept_invitation` (INVITED only) → `confirm_speaker` (ACCEPTED only;
**CONFIRMED is what the printed program keys on — load-bearing
print-safety pin**); `decline_invitation` from INVITED only;
`withdraw_speaker` from CONFIRMED only (post-print pull-out audit
trail). `is_print_safe(speaker)` returns True only for CONFIRMED.
`Session` carries session_id + title + room + tz-aware time bounds +
non-empty speaker_ids. `_windows_overlap` uses half-open [start, end)
intervals so back-to-back sessions (a.end == b.start) DON'T conflict.
`assert_no_conflict` detects: duplicate session_id; speaker
double-booking (same speaker_id in overlapping windows); room
double-booking (same room in overlapping windows); panel sessions
catch ANY shared-speaker collision (not just primary). `Sponsor` +
`tier_outranks` (strict >) + `sponsors_at_or_above` (filters +
returns sorted highest-tier-first). `speaker_kind_balance` returns
counts per SpeakerKind for **CONFIRMED speakers only** (operators
checking "did we book enough scholars?" want actual speaking
roster, not intentions). Render with status emoji 📧/✅/❌/🎤/🚫,
kind emoji 📚/🚀/⚙️/🎓/⚖️, tier emoji 💎/🥇/🥈/🥉. No-secret
regression: render never includes `@` (no email-shape) or invoice
amounts. Tests in `tests/test_conference.py` (61 cases): 3 enum
string-value pins; Speaker validation including INVITED-must-not-
have-decided-at and non-INVITED-requires-decided-at both directions;
decided-before-invited rejected; invite_speaker basic + naive-now
rejection; accept transitions + non-INVITED-source rejected;
decline INVITED only; **confirm ACCEPTED only with skip-from-INVITED
rejection (load-bearing pin)**; **withdraw CONFIRMED only**; is_print_safe
across all 5 statuses pinned; Session field validation; **back-to-
back sessions same room don't conflict (a.end == b.start)**;
speaker double-booking caught across rooms; room double-booking
caught across speakers; panel speaker collision pin (any of 3
panel speakers double-booked → conflict); duplicate session_id
rejected; Sponsor validation + immutability; tier_outranks strict
(same tier returns False); sponsors_at_or_above filters + sorts
descending; speaker_kind_balance counts CONFIRMED only +
zero-counted kinds present; render with emojis + no-secret
regression (no @ no $); e2e flows (full speaker lifecycle invitation
→ accept → confirm → print-safe; speaker pulls-out post-confirmation
must be removed; 4-session schedule with back-to-back validation;
double-booking caught at planning time; replay consistency).

### 10.F — Open-source core, paid hosting model ✅ landed (`web/feature_gate.py`)

The core trading engine + screener stays open source (MIT). Hosted
multi-user is the paid product. Aligns incentives — community
contributes to a free core; commercial users fund development.

**Implementation**: `src/halal_trader/web/feature_gate.py` ships
the pure-Python feature-gating engine. `Edition` enum (OSS / HOSTED)
+ `Feature` enum (18 features across core trading + multi-user +
tier-gated value-adds) with pinned string values. `FeatureSpec`
frozen dataclass carries `oss_available` + `min_tier` + description.
Module-level `_FEATURE_REGISTRY` is the canonical matrix; `_TIER_ORDER`
maps FREE/PRO/ENTERPRISE to ordering ints for ≥ comparison.
`is_feature_available(feature, *, edition, tier=None)` returns True
based on the spec; HOSTED requires tier (raises ValueError if
omitted), OSS ignores tier. `require_feature` raises
`FeatureNotAvailableError` (carrying feature + edition + tier) on
unavailable. `features_available` returns the deterministic-order
list of available specs. **Pinned regression**: core trading
features (CYCLE_RUN, STOCK_TRADING, CRYPTO_TRADING, HALAL_SCREENER,
LOCAL_DASHBOARD, LOCAL_BACKTEST, PURIFICATION_LEDGER) are
*regression-pinned available in OSS* — a future PR walling the core
engine behind a paywall fails CI rather than ships. **OSS-strict-
subset pin**: every OSS-available feature has `min_tier=Tier.FREE`
in HOSTED; pinned via test that iterates the registry. `render_matrix`
emits the marketing-page table with ✅/— per (feature × OSS / Free /
Pro / Enterprise) cell. Tests in `tests/test_feature_gate.py` (49
cases): edition + feature enum string-value pins (all 18 features);
FeatureSpec validation; registry coverage (one spec per Feature
enum value, canonical order); core-features-stay-OSS regression
pin; hosted-only features unavailable in OSS pin; OSS-strict-subset
across HOSTED ENTERPRISE; tier ordering (FREE blocked from PRO,
PRO blocked from ENTERPRISE, ENTERPRISE has all PRO features);
HOSTED-without-tier raises; OSS ignores tier; require_feature
silent on available + raises with carried context on unavailable;
features_available correct sets per (edition, tier); deterministic;
error message includes feature + edition + tier; render_matrix
includes headers + every feature + correct ✅/— counts per row
(cycle_run row has 4 ✅, billing row has 1 — and 3 ✅,
public_research_api row has 1 ✅); render no-secret regression;
render_context_summary; e2e flows (OSS user runs full trading loop;
OSS user blocked from billing; HOSTED FREE user has basic but not
LLM cycles; HOSTED ENTERPRISE has everything; PRO→ENTERPRISE
upgrade unlocks research API).

### 10.G — Halal-fintech partnership integrations ✅ landed (`web/partnership_directory.py`)

Strategic integrations with: Wahed Invest (managed accounts), Aghaz
(robo-advisor), Amana Mutual Funds. They send users; we provide
high-frequency / active-management capability they don't.

**Implementation**: `src/halal_trader/web/partnership_directory.py`
ships the pure-Python partner directory + integration-readiness
aggregator. `Capability` enum (11 values: managed_portfolios,
robo_advisor, mutual_funds, active_management, hft, halal_screening,
purification_ledger, llm_reasoning, backtesting, broker_api,
user_base) with pinned string values. `OUR_CAPABILITIES` frozenset
names what halal-trader brings (active_management, HFT,
halal_screening, purification_ledger, llm_reasoning, backtesting).
`HalalCertLevel` ladder (NONE / SELF_DECLARED / THIRD_PARTY_AUDITED
/ SCHOLAR_REVIEWED / SHARIAH_BOARD_CERTIFIED) with `_CERT_ORDER`
ints for ≥ comparison via `cert_meets_minimum`. `IntegrationStage`
funnel (INITIAL_OUTREACH → MUTUAL_INTEREST → SCOPE_ALIGNED →
LEGAL_REVIEW → INTEGRATION_BUILD → LIVE; off-funnel PAUSED)
with hard-pinned `_STAGE_ORDER`. `StageTransition` audit row.
`Partner` frozen dataclass carries only public-facing fields
(partner_id, display_name, public_url, capabilities, halal_cert_level,
current_stage, transitions, active); validates http(s):// URL prefix;
*deliberately* excludes internal contact emails, revenue, NDA docs.
`create_partner` constructs at INITIAL_OUTREACH with the first
audit transition. `advance_stage` enforces one-step-at-a-time
forward progression (skip raises `StageOutOfOrderError`); PAUSED
can be entered from any stage; resuming from PAUSED accepts any
non-paused stage. `deactivate` flips active=False while preserving
the full audit trail. `complementarity_score(partner)` returns
`1 - |intersection| / |union|` of partner ∪ our capabilities — 1.0
means perfectly disjoint (high-value partnership), 0.0 means
redundant. `cert_meets_minimum` for due-diligence sort order.
`build_funnel` aggregates per-stage counts (active partners only).
Render no-secret regression: no email-shaped substrings, no `$` /
`USD` / `ARR` / `MRR`, no `internal` / `nda`. Tests in
`tests/test_partnership_directory.py` (49 cases): enum string-value
pins; OUR_CAPABILITIES inventory pin (we have HFT + active mgmt +
halal screening; we don't have managed portfolios / mutual funds
/ user base — pinned both directions); StageTransition validation;
Partner construction (rejects empty fields, URLs without scheme;
accepts both https + http); starts at INITIAL_OUTREACH; immutable;
advance_stage one-step-forward; skip rejected with both stages on
the exception; full funnel to LIVE happy path; PAUSED accepted
from anywhere; resume from PAUSED accepts any prior stage;
same-stage rejected; naive-now rejected; returns new state without
mutating; records notes; deactivate preserves transitions;
complementarity (perfect-disjoint=1.0, perfect-overlap=0.0,
Wahed > 60%, empty partner = 1.0, custom our_capabilities flows
through); cert ladder (self-meets-self, higher meets lower, lower
fails higher, NONE below everything); filter_active +
filter_at_stage; build_funnel counts per stage + excludes inactive +
total_live count; render partner (display_name + url visible;
stage emoji 📨; cert emoji 🕌; complementarity %; inactive
marker); no-secret-leak regression (no email pattern, no `$`,
no `MRR`, no `internal`/`nda`); render funnel (active count,
per-stage non-zero counts shown, zero-count stages omitted);
e2e flows (Wahed full funnel through LIVE; paused-then-revived
preserves audit trail; replay consistency).

**Acceptance**: ≥1000 active users, ≥10 community-contributed
strategies, ≥3 strategic partnerships announced.

---

## Wave 11 — Regulatory + legal (long-running)

Required if we're going to take real-money users at scale.

### 11.A — Investment-advisor registration (US: SEC RIA, UK: FCA) ✅ landed (readiness aggregator)

Once we serve real-money users, we're an investment advisor. File
the relevant registrations. ~6 months lead time.

**Landed (readiness aggregator)** as
`web/advisor_registration_readiness.py` — pure-Python
deployment-readiness aggregator across major investment-advisor
regulators (SEC RIA / FCA UK / Saudi CMA / UAE VARA / Singapore
MAS / Australia ASIC). Mirrors design of Wave 11.E (SOC 2) +
Wave 11.F (halal certification): closed authority set,
module-level frozen requirement sets, per-jurisdiction control
catalogue, 12-month staleness horizon for annual filings, no-PII
render contract. Six `RegulatorAuthority` values; nine
`RegistrationCategory` labels (FORM_FILING / BACKGROUND_CHECKS /
EXAMS_LICENSURE / SURETY_BOND / AML / COMPLIANCE_PROGRAM /
RECORDKEEPING / CLIENT_MONEY / DISPUTE_RESOLUTION); two
`RequirementSeverity` levels; 22 `EvidenceKind` artifact labels
covering form filings (FORM_ADV_PART_1A/2A/2B / ANNUAL_FORM_ADV_
AMENDMENT / FCA_SUP_FORM / FCA_GABRIEL_RETURN), identifiers
(CRD_NUMBER / IARD_REGISTRATION / FCA_FIRM_REFERENCE_NUMBER /
SAUDI_CMA_LICENCE / SINGAPORE_CMS_LICENCE / AUSTRALIA_AFSL),
personal qualifications (SERIES_65_PASSED / SERIES_66_PASSED /
SMCR_CERTIFIED_PERSONS / EXEC_BACKGROUND_CHECK), compliance
program (SURETY_BOND / AML_PROGRAM / COMPLIANCE_MANUAL /
RECORDKEEPING_PROCEDURES), UK/Singapore specifics (CLIENT_MONEY
_RULES_DOCUMENTED / FOS_MEMBERSHIP / DISPUTE_RESOLUTION_
PROCEDURES). Module-level frozen requirement sets: 10 SEC RIA
controls (Form ADV Parts 1A/2A/2B; Series 65; exec background;
surety bond; AML program; Rule 206(4)-7 compliance manual;
Rule 204-2 recordkeeping; warning-severity annual amendment);
6 FCA controls (SUP form; SMCR-certified persons + DBS check;
CASS-7 client money rules; FOS membership; SYSC-6.3 MLR-2017
AML; warning-severity annual Gabriel/RegData return). Saudi
CMA / UAE VARA / Singapore MAS / Australia ASIC start empty for
operator-side extension. Five pinned semantics: (1) closed
regulator-authority set; (2) per-authority requirements
module-level frozen; (3) critical-requirement gap → BLOCKING;
(4) 12-month staleness horizon (default 365d) — annual filings
past horizon become stale + WARNING; boundary at exactly 365d
inclusive; strict 180d policy catches 300d artifacts; (5)
render output never includes operator-identifying detail —
references abstract evidence-kind labels not raw firm name /
CRD number / executive PII / IARD codes (mirrors no-PII patterns
of Wave 11.D + 11.C + 3.B + 11.E + 11.F). Same readiness ladder
as 11.E + 11.F (READY / NEARLY_READY / GAPS / NOT_READY).
Halal alignment: read-only aggregation; never opens a position;
pure-Python; no DB / network / async; frozen dataclasses. 45
tests cover policy validation; per-authority requirement-set
queries (SEC RIA 10 controls; FCA 6 controls; Saudi CMA / UAE
VARA / Singapore MAS / Australia ASIC empty by design);
RegistrationRequirement + EvidenceArtifact field validation;
**READY** happy paths (SEC RIA full evidence; FCA full evidence);
**NOT_READY** (no evidence; ≥3 blocking); **GAPS** (1 blocking
gap); **NEARLY_READY** (only warning unmet — drop ANNUAL_FORM_
ADV_AMENDMENT); **stale-evidence pins** (>365d → stale +
nearly_ready; exactly-365d boundary inclusive; no last_updated →
stale; strict 180d catches 300d artifact); empty-authority →
NOT_READY with note; input validation; per-requirement
assessments preserve count + carry notes; frozen-dataclass
immutability; all enum string values pinned; render output
across all four levels with emoji; **render-no-PII pin**
(asserts firm_name + actual CRD numbers never in render); end-
to-end realistic flows (typical pre-filing SEC RIA with annual
amendment gap → NEARLY_READY; UK FCA filing missing SMCR-
certified persons → GAPS; assessment count matches; met_pct
aggregates to 100% on full evidence). Postgres
`registration_evidence` + `registration_audit_log` tables wired
to `db/repository.py`, the FastAPI route adapters
(`/api/registration/{authority}/readiness` and
`/api/registration/gaps`), the dashboard's registration tile
rendering per-authority progress with the actionable-gaps list,
and the per-day CI step that emits a readiness warning when any
required evidence's `last_updated` falls past horizon deferred
to follow-ups — aggregator verified in isolation first.

### 11.B — Shariah Supervisory Board ✅ landed (governance engine)

Three-member SSB (one Hanafi, one Shafi'i, one Maliki at minimum)
that reviews the platform quarterly. Public register of their rulings.

**Landed (governance engine)** as `halal/ssb_governance.py` —
pure-Python state engine for the SSB. Four `FiqhSchool`s (HANAFI /
SHAFII / MALIKI / HANBALI). Four `RulingScope`s (PRODUCT /
STRATEGY / POLICY / CERTIFICATION). Four `RulingOutcome`s
(PERMISSIBLE / IMPERMISSIBLE / PERMISSIBLE_WITH_CONDITIONS /
DEFERRED). `ScholarMember` carries name + school + appointed_at +
expires_at + bio_url with `is_active(now)` checking term currency.
`Vote` requires non-empty member_name and rationale for
IMPERMISSIBLE / PERMISSIBLE_WITH_CONDITIONS (silent rejections
without justification rejected at construction). `Ruling` carries
ruling_id + scope + subject + description + issued_at + votes +
conditions; validates conditions non-empty when any vote is
PERMISSIBLE_WITH_CONDITIONS; rejects duplicate-voter ballots.
**Six pinned semantics**: (1) **minimum 3 members across ≥ 3
schools** — three Hanafis fails, two Hanafis + one Shafi'i fails;
diversity prevents single-madhhab ruling being misread as
universal consensus; (2) **any IMPERMISSIBLE → IMPERMISSIBLE
outcome** — the conservative-tiebreak rule shared with Wave 2.B
halal consensus + Wave 4.J committee + Wave 1.G commodities + 1.I
REIT + 2.G regulator-index; (3) **2/3 supermajority required for
PERMISSIBLE** — below threshold → DEFERRED, not silent
PERMISSIBLE; default 0.667; operators can bump to 0.75 for
stricter platforms; (4) **PERMISSIBLE_WITH_CONDITIONS requires
explicit conditions** — a "yes but…" without the caveats is
meaningless; (5) **term expiry tracked** — expired-term members
silently dropped from active count + surface warnings; (6)
**quarterly review cadence (90d default) enforced** — board with
no recent rulings flagged via `needs_quarterly_review`. Pinned:
empty rulings list returns True (overdue from incorporation);
boundary at exactly 90d is NOT overdue (strict greater-than). The
`_consensus_outcome` rule is the load-bearing voting algorithm:
any IMPERMISSIBLE wins; supermajority of pass votes → PERMISSIBLE
(or PERMISSIBLE_WITH_CONDITIONS if any vote was conditional);
otherwise DEFERRED. `validate_board(members, *, now, policy)`
returns `BoardCompositionResult` with active count + school count
+ schools represented + failures + warnings — operators publish
the result to their public register so external auditors can
confirm composition compliance. Halal alignment: SSB engine is
the ultimate compliance authority for the platform; pure-Python
(`dataclasses` + `enum` + `datetime`); no DB / network / async;
frozen dataclasses on every output. Render output is the public
register format (operators publish at `halal-trader.dev/ssb/
rulings/<id>`); regression-pinned no-PII contract — the ruling
references the *product* not the operator's user / portfolio /
account. 71 tests cover policy validation (every guard fires),
ScholarMember validation (timezone-aware datetimes; expires-after-
appointed; is_active in/out/before-term), board composition
(diverse 3-school passes; three-Hanafis fails; 2-school fails;
2-member fails minimum; expired-member excluded with warning;
expired-majority fails minimum; duplicate name rejected; 4-school
strict policy flips a 3-school result), Vote validation (empty
name; rationale required for IMPERMISSIBLE + PERMISSIBLE_WITH_
CONDITIONS in both empty and whitespace-only forms; PERMISSIBLE +
DEFERRED allow empty rationale), the **consensus rules in eight
flavours** (unanimous PERMISSIBLE; any-IMPERMISSIBLE-overrides;
2-of-3 below supermajority threshold → DEFERRED; 2-of-3 at
threshold → PERMISSIBLE; mixed pass-with-conditional →
PERMISSIBLE_WITH_CONDITIONS; all DEFERRED → DEFERRED; strict
0.75 supermajority flips marginal verdicts; empty votes →
DEFERRED), Ruling validation (every field invariant; conditional
without conditions rejected; duplicate-voter rejected),
needs_quarterly_review (empty list True; recent ruling False; old
ruling True; exactly-90d boundary False; just-past-90d True; uses
most-recent issued_at across multiple rulings; custom 30d cycle
flow; rejects naive now), frozen-dataclass immutability across
ScholarMember + Vote + Ruling + SSBPolicy + BoardCompositionResult,
all enum string values pinned for public-register stability,
render output across all four outcomes with emoji (✅/❌/⚠️/⏳) +
section headers + vote-attribution table + conditions list + the
**no-operator-PII pin** (asserts neither "user_id" nor "portfolio"
nor "balance" appears in the rendered ruling), board-composition
render with VALID/INVALID + active count + schools, and two end-
to-end realistic scenarios (typical quarterly meeting lifecycle:
validate board → issue PERMISSIBLE_WITH_CONDITIONS ruling on the
Wave 1.G commodity screener → confirm review trigger fires at
91d; dissenting-minority-blocks-pass: 2 PERMISSIBLE + 1
IMPERMISSIBLE-with-AAOIFI-citation → IMPERMISSIBLE consensus).
Postgres `ssb_members` + `ssb_rulings` tables wired to `db/
repository.py`, the FastAPI route adapters (`/api/ssb/board`,
`/api/ssb/rulings`, `/api/ssb/rulings/{ruling_id}`), the public-
register page that renders rulings + the dashboard's compliance
tile showing next review-due date deferred to follow-ups —
governance engine verified in isolation first.

### 11.C — KYC/AML for hosted users ✅ landed (state engine)

Required by most jurisdictions for financial platforms. Integrate
Persona / Sumsub / Onfido. Per-user KYC level gates real-money
trading.

**Landed (state engine)** as `web/kyc.py` — pure-Python KYC + AML
state machine with five-tier `KYCLevel` ladder (NONE / EMAIL_
VERIFIED / IDENTITY_VERIFIED / ADDRESS_VERIFIED / ENHANCED_DUE_
DILIGENCE), six `KYCStatus` workflow states (NOT_STARTED /
IN_PROGRESS / VERIFIED / EXPIRED / REJECTED / UNDER_REVIEW), three
`RiskLevel`s for AML scoring (LOW / MEDIUM / HIGH), three
`SanctionsOutcome`s (CLEAR / MATCH / FALSE_POSITIVE), six
`Activity` types the engine gates (SIGNUP / DEMO_TRADING /
PAPER_TRADING / REAL_MONEY_DEPOSIT / REAL_MONEY_TRADING /
WITHDRAW), and per-jurisdiction `JurisdictionRequirement`
overrides. `permits(state, *, activity, now, policy)` returns a
deterministic `GateDecision` carrying allowed bool + reason +
required/actual level. **Six pinned semantics**: (1) **default =
paper-trading-only** — a user with no KYC can SIGNUP / DEMO /
PAPER_TRADING but is rejected for real-money activity, so a forgot-
to-wire-up-the-gate failure produces the safe default not the
unsafe one; (2) **real-money trading requires IDENTITY_VERIFIED
minimum** with jurisdiction overrides bumping EU / UAE / SA / GB
to ADDRESS_VERIFIED per MiFID II / VARA; (3) **sanctions MATCH
blocks every activity except SIGNUP** — even WITHDRAW is blocked
under MATCH (funds stay pending operator compliance review per
the law); FALSE_POSITIVE (set by compliance ops after review) is
treated as CLEAR for gating; (4) **expired KYC blocks deposits /
new trades but permits WITHDRAW** — trapping a user's funds during
re-verification is operationally awful and legally questionable;
(5) **HIGH risk score requires ENHANCED_DUE_DILIGENCE** for real-
money inflow per FATF guidance; HIGH risk doesn't block WITHDRAW
(the engine never traps funds for risk-score reasons); (6)
**unregistered jurisdictions block real-money** — operator must
explicitly add a JurisdictionRequirement per supported country, so
a forgot-to-configure failure surfaces immediately. Default
expiry 365d (FATF cap); operators in stricter jurisdictions bump
to 180d; expiry boundary inclusive at exactly 365d. Halal
alignment: KYC engine is the boundary the operator's compliance
program runs against; never opens a position; mirrors the
no-PII-in-render contract of Wave 11.D privacy + Wave 8.D OTLP +
Wave 3.B vault (render summarises level + status + jurisdiction +
risk + sanctions + verified-at, never the underlying ID document
data — passport / license / SSN never appear in the render
output, regression-pinned). Pure-Python (`dataclasses` + `enum` +
`datetime`); no DB / network / async / HTTP. Frozen dataclasses
on every output. 63 tests cover: level int_value ordering pin
(NONE < EMAIL < IDENTITY < ADDRESS < ENHANCED for jurisdiction-
minimum gates); default-policy seven supported jurisdictions; UAE
/ EU bump to ADDRESS_VERIFIED; US / PK / MY at IDENTITY_VERIFIED;
expiry default 365d; policy validation (zero / negative expiry
rejected); JurisdictionRequirement validation (empty jurisdiction
rejected); UserKYCState field validation (empty user_id /
jurisdiction rejected; naive verified_at rejected when set; None
verified_at accepted); is_expired ladder (False for never-
verified; False within horizon; True past horizon; True at
exactly 365d boundary inclusive; False at 364d23h; rejects naive
now; custom expiry flows through); the **sanctions-MATCH gate
pin** (blocks REAL_MONEY_TRADING, PAPER_TRADING, WITHDRAW; permits
only SIGNUP); FALSE_POSITIVE-treated-as-CLEAR pin; KYC-free
activities pass under no-KYC (SIGNUP / DEMO / PAPER); real-money
gates (NOT_STARTED / IN_PROGRESS / REJECTED / UNDER_REVIEW all
block); jurisdiction-minimum pin (US identity-verified passes; EU
identity-verified blocked; EU address-verified passes); unregistered
jurisdiction blocks real-money; deposit follows same rules as
trading; the **expired-KYC pin** (blocks REAL_MONEY_TRADING +
DEPOSIT; permits WITHDRAW with the "trapping funds" reason);
status=EXPIRED triggers same path as elapsed-time-expired;
**HIGH-risk-requires-EDD pin** (blocks REAL_MONEY_TRADING without
ENHANCED_DUE_DILIGENCE; passes with EDD; MEDIUM follows standard
ladder; HIGH doesn't block WITHDRAW); permits() rejects naive now;
frozen-dataclass immutability across UserKYCState / GateDecision /
KYCPolicy / JurisdictionRequirement; all enum string values pinned
for JSON / DB stability; render_user_state with emoji per status
(✅⏰❌🔍⚪🟡); the **no-ID-document pin** (passport / license /
ssn never appear in render); render_decision with
ALLOWED/BLOCKED/level-comparison; never-verified-renders-"never";
and three end-to-end realistic flows (US user paper→KYC→real-trade
journey; EU user needs ADDRESS_VERIFIED not just IDENTITY; full
sanctions-MATCH-then-FALSE_POSITIVE compliance review cycle).
Vendor adapter layer (Persona / Sumsub / Onfido / Trulioo HTTP
clients behind a `KYCProvider` Protocol — vendor returns level +
status + risk + sanctions; the engine consumes the result), the
Postgres `user_kyc_states` + `kyc_audit_log` tables wired to
`db/repository.py`, the FastAPI route adapters (`/api/users/me/
kyc/start` triggering vendor flow; webhook receiver for vendor
async results), and the cycle-side hook (every executor /
withdraw call site checks `permits()`) deferred to follow-ups —
state engine verified in isolation first.

### 11.D — Privacy + GDPR/CCPA compliance ✅ landed (engine)

Privacy policy, data deletion flow, data export flow. Required by
law in EU + California; good practice everywhere.

**Landed (engine)** as `web/privacy.py` — pure-Python data-subject-
rights engine covering GDPR Article 15 (Right of Access / Export),
Article 17 (Right to Erasure / Deletion), and CCPA §1798.105.
Twelve `DataCategory`s the bot holds (PII / auth_credentials /
broker_keys / trading_history / halal_audit / purification_ledger /
llm_prompts / llm_responses / usage_telemetry / marketing_analytics
/ device_fingerprint / support_tickets) each carry a default
`LegalBasis` per GDPR Article 6 (CONSENT / CONTRACT / LEGITIMATE_
INTEREST / LEGAL_OBLIGATION / VITAL_INTEREST / PUBLIC_TASK) and a
retention horizon. `RetentionPolicy` validates that every category
has both a basis and a positive retention at construction (no
silent gaps). `DeletionAction` enum (HARD_DELETE / ANONYMISE /
DENIED) names the three outcomes per category. `ConsentRecord`
tracks `granted_at` + optional `revoked_at`; `is_active` returns
False once revoked; `revoke_consent(consent, *, now)` returns a
new revoked record (re-revoking already-revoked is a no-op so the
revoked timestamp doesn't keep advancing). `build_export_plan(*,
user_id, holdings, policy, requested_at)` returns an `ExportPlan`
covering every category with `count > 0` (zero-count omitted —
nothing to export); `build_deletion_plan(...)` returns a
`DeletionPlan` with per-category `CategoryDeletionEntry` rows
including the action + reason. **Five pinned safety semantics**:
(1) **legal-obligation categories cannot be hard-deleted** —
trading_history / halal_audit / purification_ledger return
ANONYMISE by default (PII fields scrubbed; financial / shariah
audit trail preserved per FINRA / SEC / AAOIFI retention); if
operator policy disables anonymisation, the action is DENIED with
the legal basis explained; (2) **consent-basis data MUST hard-
delete on Article 17 request** — even if a consent record was
previously revoked, the request still hard-deletes the rows; (3)
**legitimate-interest is overridden by Article 17** — usage_
telemetry + device_fingerprint hard-delete because user's right to
erasure outweighs the legitimate-interest balancing test (per
ICO guidance); (4) **export plan covers every populated category**
— zero-count categories omitted; partial export is a regulatory
failure mode; (5) **render output never contains raw personal
data** — receipts summarise action + category + count, never
field values; the receipt can be pasted in Slack / Telegram audit
channels safely. Defaults are conservative: TRADING_HISTORY +
HALAL_AUDIT retained 7 years (FINRA / shariah audit cycles);
MARKETING_ANALYTICS 90 days; USAGE_TELEMETRY 30 days. `is_overdue`
boundary inclusive (at exactly retention_days, the row is overdue
— pinned both directions). Halal alignment: privacy engine is the
boundary the operator's controller-side compliance program runs
against; never opens a position; pure-Python (`dataclasses` +
`enum` + `datetime`); no DB / network / async; frozen dataclasses
on every output. 55 tests cover: default-policy coverage (every
category has retention + basis; trading / halal / purification are
LEGAL_OBLIGATION; LLM / marketing are CONSENT; the retention
asymmetry of 7y vs 90d vs 30d), policy validation (partial
retention map rejected, partial basis map rejected, zero retention
rejected), is_overdue ladder (past-horizon True; at-exactly-30d
inclusive True; just-inside False; rejects naive datetimes),
CategoryHolding + ConsentRecord field validation (negative count
rejected; empty user_id rejected; naive datetimes rejected;
revoked-before-granted rejected), revoke_consent flow (returns new
record with revoked_at set; input unchanged; re-revoke is no-op),
ExportPlan (every populated category included; zero-count omitted;
empty holdings → empty entries; legal_basis carried per category;
recorded_at timestamps preserved), the **legal-obligation-anonymise
pin** (trading_history → ANONYMISE; with operator_allows_anonymise=
False → DENIED; halal_audit + purification_ledger same path), the
**consent-hard-delete pin** (LLM_PROMPTS + MARKETING_ANALYTICS →
HARD_DELETE), the **contract-hard-delete pin** (PII + AUTH_
CREDENTIALS + BROKER_KEYS → HARD_DELETE on account closure), the
**legitimate-interest-overridden-by-Article-17 pin** (USAGE_
TELEMETRY + DEVICE_FINGERPRINT → HARD_DELETE with "Article 17"
in the reason), zero-count omitted from deletion plan, mixed-
category aggregate counts (deleted_count + anonymised_count +
denied_count partition correctly across PII + trading + LLM
combination), revoked-consent-still-hard-deletes pin, consents-
filtered-by-user-id pin (other users' consents ignored), an end-
to-end "typical user" lifecycle (mixed holdings across 10
categories aggregate to 476 hard-deleted + 316 anonymised + 0
denied), frozen-dataclass immutability across ExportPlan +
DeletionPlan + ConsentRecord, all enum string values pinned for
JSON / DB stability, render output (export plan with 📦 + total
rows; deletion plan with 🗑️/🫥/🔒 emojis per action; empty
plans render "no data on file"; the no-PII pin asserts neither
"@" nor "phone" nor "address" appears in the rendered receipt).
FastAPI route adapters (`/api/users/me/export` and `/api/users/me/
deletion-request`), Postgres `data_holdings` + `consent_records`
tables wired to `db/repository.py`, the cycle-side scrub job that
iterates the deletion plan and runs the actual SQL, and the
operator dashboard showing pending DSR requests deferred to
follow-ups — engine verified in isolation first.

### 11.E — SOC 2 Type II audit ✅ landed (readiness aggregator)

Once we're at scale, SOC 2 is what enterprise customers (hedge funds,
family offices) ask for. ~12 months of evidence collection.

**Landed (readiness aggregator)** as `web/soc2_readiness.py` —
parallel to Wave 11.F's halal-certification readiness aggregator
but for SOC 2. Five `TrustServiceCategory` values per AICPA TSP
100-2017 (SECURITY / AVAILABILITY / PROCESSING_INTEGRITY /
CONFIDENTIALITY / PRIVACY); ten `ControlCategory` labels
(ACCESS_CONTROL / CHANGE_MANAGEMENT / SYSTEM_OPERATIONS /
RISK_ASSESSMENT / INCIDENT_RESPONSE / LOGICAL_SECURITY /
MONITORING / BUSINESS_CONTINUITY / VENDOR_MANAGEMENT /
DATA_CLASSIFICATION); two `ControlSeverity` levels
(BLOCKING / WARNING); 19 `EvidenceKind` artifact labels covering
the operator-side audit trail (access logs / MFA / PR review
logs / backup records / incident reports / DR drill reports /
vulnerability scans / pen-test reports / employee onboarding-
offboarding / security training / vendor SOC 2 reports / risk
register / data classification policy / encryption at-rest +
in-transit / uptime monitoring / status page); two
`AuditType` (TYPE_I point-in-time vs TYPE_II 12-month observation
window); four-level `ReadinessLevel` ladder.
Module-level frozen control sets: 10 Security controls
(CC6.1 logical access; CC6.2 MFA; CC6.3 onboarding; CC7.1 vuln
scans; CC7.4 incident response; CC8.1 PR review; CC9.1 vendor
SOC 2 — warning; CC3.2 risk register; CC2.3 security training —
warning; CC6.7 pen testing — warning); 4 Availability controls
(A1.1 uptime; A1.2 backups; A1.3 DR drills; A1.4 status page —
warning); 3 Confidentiality controls (C1.1 data classification;
C1.2 encryption at rest; C1.3 encryption in transit). Processing
Integrity + Privacy ship empty for operator-side extension. Five
pinned semantics: (1) **closed Trust Services Category set** per
AICPA; (2) **control catalogue module-level frozen** — runtime
config drift can't weaken floor; (3) **critical-control gap →
BLOCKING** (missing MFA → BLOCKING; missing status page →
WARNING); (4) **Type II default 12-month evidence horizon** with
Type I 30-day; boundary-inclusive at exactly horizon-day; (5)
**render output never includes operator-identifying detail** —
references abstract evidence-kind labels, never user IDs / IP
addresses / audit-trail contents. Mirrors no-PII patterns of
Wave 11.D + 11.C + 3.B + 11.F. Multi-service overall verdict =
strictest (worst) per-service level — operator picks
their target service set; the report flags the weakest. Halal
alignment: read-only aggregation; never opens a position; pure-
Python; no DB / network / async; frozen dataclasses. 47 tests
cover policy validation; per-service control-set queries; control
+ artifact field validation; **READY** happy paths (Security only;
Security+Availability; three-service combo); NOT_READY (no
evidence; ≥3 blocking gaps); GAPS (1 blocking gap); NEARLY_READY
(only warnings unmet); **stale-evidence pins** (>365d Type II
horizon; Type I 30-day catches 60d artifact; no last_updated →
stale; exactly-365d boundary inclusive); empty-PROCESSING_
INTEGRITY → NOT_READY with note; **multi-service overall verdict
pin** (one ready + one not-ready → overall not_ready; one ready
+ one gaps → gaps); overall_met_pct aggregates correctly; input
validation (naive now rejected; empty trust_services rejected);
frozen-dataclass immutability; all enum string values pinned;
render output across all four levels with emoji + per-service
breakdown; **render-no-user-PII pin** (asserts user_id /
ip_address / 10.0.0 / 192.168 never in render); end-to-end
realistic flows (typical pre-audit Security-only with warnings
unmet → NEARLY_READY; full three-service Security+Availability+
Confidentiality → READY; partial-evidence flow with Availability
gap → GAPS overall). Postgres `soc2_evidence` + `soc2_audit_log`
tables wired to `db/repository.py`, the FastAPI route adapter
(`/api/soc2/{trust_service}/readiness` and `/api/soc2/readiness`
for the multi-service overview), the dashboard's SOC 2 tile
rendering per-service progress with the actionable-gaps list,
and the per-day CI step that emits a readiness warning when any
required evidence's `last_updated` falls past the operator's
horizon deferred to follow-ups — aggregator verified in isolation
first.

### 11.F — Halal certification from a recognised authority ✅ landed (readiness aggregator)

Apply for AAOIFI certification, Saudi Tadawul approval, Malaysian
SAC recognition. These are the badges that unlock institutional
adoption in Muslim-majority markets.

**Landed (readiness aggregator)** as
`halal/certification_readiness.py` — pure-Python deployment-
readiness aggregator that scores the operator's current state
against each body's requirement set BEFORE the application is
submitted. Picked a focused readiness aggregator over an
"auto-certify" flow because (a) certification bodies require
human auditor sign-off, (b) the audit-trail evidence is the
operator's existing artifacts (Wave 2.A signed receipts, Wave 2.D
purification ledger, Wave 11.B SSB rulings, Wave 11.C KYC, Wave
11.D privacy), and (c) the readiness report's actionable output
lets the operator focus pre-application work on actual gaps. Five
`CertificationBody` values (AAOIFI / SAUDI_TADAWUL /
MALAYSIAN_SAC / BAHRAIN_CBB / INDONESIA_DSN_MUI). Six
`RequirementCategory` labels (SCREENING / AUDIT / PURIFICATION /
SSB_GOVERNANCE / KYC_AML / DOCUMENTATION). Two
`RequirementSeverity` levels (BLOCKING / WARNING). Ten
`EvidenceArtifactKind` labels matching the existing Round-4
artifacts (HALAL_SCREENER_DECISIONS / SIGNED_TRADE_RECEIPTS /
PURIFICATION_LEDGER / SSB_RULINGS / SSB_QUARTERLY_REVIEWS /
KYC_VERIFIED_USERS / AML_SANCTIONS_SCREENING /
ANNUAL_AUDIT_REPORT / PUBLIC_PRIVACY_POLICY /
SHARIAH_AUDIT_REPORT). Four `ReadinessLevel`s (READY /
NEARLY_READY / GAPS / NOT_READY). Per-body requirement sets
pinned via module-level frozen tuples; AAOIFI has 7 requirements
across S21 / GS / S17 standards; Tadawul has 4 (with KYC + AML);
SAC has 3; Bahrain CBB + Indonesia DSN-MUI start empty for
operator extension. **Five pinned semantics**: (1) **closed body
set** — operators add via code review with regression-test
coverage; (2) **per-body requirements module-level frozen** —
runtime config drift can't silently weaken the floor; (3)
**critical-requirement gap → BLOCKING** — missing SSB ruling for
AAOIFI is BLOCKING (operator can't apply); missing privacy policy
is WARNING (apply but flagged); (4) **stale evidence (> 180d
default) → WARNING** — boundary at exactly 180d is stale (pinned
both directions); evidence-without-timestamp also stale; custom
90d strict policy catches 100d-old artifacts; (5) **render output
never includes operator-identifying detail** — references
abstract artifact-kind labels, never raw rule_id contents /
member names / KYC secrets. Mirrors no-PII patterns of Wave 11.D
+ 11.C + 3.B. The ReadinessLevel ladder: 0 blocking + 0 warning +
0 stale → READY; 0 blocking + (warnings or stale) → NEARLY_READY;
1-2 blocking → GAPS; ≥3 blocking → NOT_READY. Halal alignment:
read-only aggregation; never opens a position; pure-Python; no
DB / network / async; frozen dataclasses on every output. 47
tests cover policy validation; per-body requirement-set queries
(AAOIFI / Tadawul / SAC populated; Bahrain / Indonesia empty by
design); requirement validation; artifact validation (timezone-
aware datetimes; None accepted for missing timestamp); **READY**
happy path (all AAOIFI artifacts present + recent → READY with
100% met_pct); **NOT_READY** (no evidence; ≥3 blocking gaps);
**GAPS** in two flavours (1 blocking gap; 2 blocking gaps —
the boundary pin); **NEARLY_READY** (only warnings unmet — drop
PUBLIC_PRIVACY_POLICY which is the warning-severity AAOIFI req);
**stale-artifact pins** (artifact > 180d → stale; exactly 180d
boundary inclusive; 179d fresh; no last_updated → stale; strict
90d policy flips a 100d artifact); **empty-body pin** (Bahrain /
Indonesia → NOT_READY with "requirement set is empty" note);
evaluate validation (naive now rejected); per-requirement
assessments preserve count + carry notes (missing → "missing
artifact (BLOCKING)"; stale → "stale" with day count); frozen-
dataclass immutability; all enum string values pinned for JSON /
DB stability; render output across all four levels with emoji
(✅/🟢/⚠️/❌); per-requirement breakdown surfaces spec_id +
category + severity; **the no-operator-PII pin** asserts
"user_id" / "ssn" / "passport" never in render output; and three
end-to-end realistic flows (typical pre-application AAOIFI →
GAPS with 2 blocking gaps; Tadawul-readiness with KYC + AML +
audit + screening → READY; Malaysian SAC with the warning-
severity purification gap only → NEARLY_READY). Postgres
`certification_evidence` + `readiness_history` tables wired to
`db/repository.py`, the FastAPI route adapters
(`/api/certification/{body}/readiness`), the dashboard's
certification-readiness tile rendering per-body progress with
the actionable-gaps list, and a CI step that emits a readiness
warning when any required artifact's `last_updated` falls past
the operator's staleness horizon deferred to follow-ups —
aggregator verified in isolation first.

**Acceptance**: A US-based real-money user can sign up, complete KYC,
fund their account, and trade halal stocks / crypto / commodities
with full SEC + Shariah compliance — all in-platform.

---

## Wave 12 — Long-term vision (years)

Speculative but worth charting now.

### 12.A — Robo-advisor mode (target-date halal portfolio) ✅ landed (engine)

For users who don't want to actively trade: a managed-portfolio mode
with halal target-date funds, automatic rebalancing, tax-loss
harvesting (US-only).

**Landed (engine)** as `web/robo_advisor.py` — pure-Python target-
date halal portfolio engine. Three `RiskProfile`s (CONSERVATIVE /
MODERATE / AGGRESSIVE) and five `HalalAssetClass` categories
(HALAL_EQUITY / SUKUK / HALAL_COMMODITIES / HALAL_REIT / CASH).
The closed-set asset enum is the load-bearing pin: conventional
bonds, leveraged ETFs, and any non-halal vehicle are categorically
absent — the engine *cannot* allocate to them because the type
system rejects it (regression-pinned via the "no `bonds` /
`conventional_bonds` / `fixed_income` in enum values" test).
`compute_target_allocation(profile, years_to_target)` interpolates
linearly between far-horizon (≥ 30 years) and near-horizon (≤ 1
year) anchor points; far-horizon AGGRESSIVE = 80% equity / 10%
sukuk; near-horizon CONSERVATIVE = 10% equity / 50% cash.
`compute_rebalance(*, user_id, current, target, threshold_pct=5.0)`
returns a `RebalancePlan` with per-asset trades or a no-op when
every asset's drift is below threshold. **Six pinned semantics**:
(1) **halal-only by construction** — closed enum set, structural
friction prevents accidentally allocating to non-halal categories;
(2) **weights sum to 1.0** with float-tolerance window [0.999,
1.001] — neither too tight (rejecting clean inputs to rounding
noise) nor too loose (silently approving 5% off-target weights);
(3) **no leverage / no shorts** — every weight ∈ [0, 1] enforced
at construction; (4) **rebalance threshold prevents churn** —
default 5% drift before any trade fires; pinned at boundary
(4.99% → no-op; 5.00% → triggers); (5) **glide path monotonic
toward conservative** — equity weight strictly decreases as years
shrink, cash weight strictly increases (pinned via the strict-
inequality regression on three time horizons); (6) **no USD in
render** — every receipt shows weight % only, never $ amounts —
the no-PII / no-balance-leak pattern of Wave 11.D + 11.C + 3.B.
Glide-path interpolation tested at midpoint (15.5y MODERATE
equity = midpoint of 60% and 20%); both anchor boundaries
inclusive (30y exactly → far anchor; 1y exactly → near anchor;
0y clamps to near). Profile asymmetry pinned across horizons
(AGGRESSIVE > MODERATE > CONSERVATIVE for equity at every
horizon). Halal alignment: closed-set asset enum is the structural
guarantee; pure-Python (`dataclasses` + `enum`); no DB / network /
async; frozen dataclasses on every output. 50 tests cover
TargetAllocation validation (missing class / negative / above-1 /
sum below-tolerance / sum above-tolerance / tiny float drift
accepted), Holding + CurrentAllocation validation (negative
weight / above-1 weight / duplicate asset / missing asset / sum
out of tolerance), the **glide-path semantics** (far-horizon
anchor at ≥ 30y; near-horizon at ≤ 1y; zero-years clamps to
near; exact-30y at far anchor; exact-1y at near anchor; midpoint
linear interpolation; equity-decrease monotonic across horizons
strict inequality; cash-increase monotonic; AGGRESSIVE >
MODERATE > CONSERVATIVE equity at every horizon), negative-years
rejected, weights-always-sum-to-1 across all profile × horizon
combinations regression test, the **rebalance threshold pin in
both directions** (no drift → no-op; 4% drift below 5% threshold
→ no-op; exactly-5% triggers boundary inclusive; 10% drift
triggers; deltas correctly computed; custom 3% threshold catches
4% drift the default would miss), threshold validation (zero /
negative rejected; empty user_id rejected), RebalanceTrade
buy/sell/neutral classification pinned, max_drift_pct correctly
computed across asset classes, the **closed-set enum guarantee**
(exactly 5 asset classes; no conventional-bond / fixed-income /
bonds value), frozen-dataclass immutability across all four
output types, all enum string values pinned, render output (target
allocation with profile emoji 🛡️/⚖️/🚀; rebalance plan with
↑/↓ arrows; no-op render path; **no-USD pin** asserts neither
"$" nor "USD" appears in any render output), and three end-to-
end realistic flows (30y-out aggressive user → 80% equity / 10%
sukuk; 5y-out conservative user → sukuk + cash > equity; full
100%-cash → diversified rebalance lifecycle). Postgres
`portfolio_accounts` + `rebalance_audit_log` tables wired to
`db/repository.py`, automatic-rebalance scheduler (cron-driven
quarterly check + threshold-driven immediate rebalance), per-
profile glide-path versioning (operators can publish v2 glide
path without affecting users on v1), tax-loss-harvesting US-only
mode (operator opt-in; offsets US-source capital gains within
the same calendar year), and the dashboard's robo tile rendering
the user's allocation pie chart deferred to follow-ups — engine
verified in isolation first.

### 12.B — Halal options strategies ✅ landed (screener)

Conservative options strategies (covered calls, cash-secured puts)
are debated in halal finance — some scholars permit, most don't.
Once SSB rules in our favour, build a halal options module.

**Landed (screener)** as `web/halal_options.py` — pure-Python
options-strategy halal screener that composes with Wave 11.B
SSB governance. Thirteen `OptionStrategy` values across four
classes: single-leg longs (LONG_CALL / LONG_PUT — debated);
naked shorts (NAKED_CALL / NAKED_PUT — unconditional gharar);
covered structures (COVERED_CALL / CASH_SECURED_PUT — debated;
PROTECTIVE_PUT — permitted as insurance); multi-leg spreads
(BULL_CALL_SPREAD / BEAR_PUT_SPREAD / IRON_CONDOR / BUTTERFLY /
STRADDLE / STRANGLE — embed short legs, all unconditional gharar);
plus OTHER catchall → UNKNOWN. **Five pinned semantics**: (1)
**closed strategy enum** — non-listed combinations route to OTHER
→ UNKNOWN; the safe-block default protects against exotic
combinations (calendar spreads, jade lizards) the engine doesn't
recognise; (2) **NAKED_CALL / NAKED_PUT unconditionally
NOT_HALAL** — no scholar ruling reverses these (classical +
modern consensus); the policy-construction guard refuses to
accept them in `permitted_with_ssb_ruling` even if operator
tries; (3) **multi-leg spreads with any short leg unconditionally
NOT_HALAL** — bull-call spread embeds a short call; iron condor /
butterfly / straddle / strangle likewise; the long-leg of the
spread does not redeem the gharar of the short. Same construction
guard refuses to accept them in operator override; (4)
**COVERED_CALL / CASH_SECURED_PUT / LONG_CALL / LONG_PUT
DOUBTFUL by default** — operator's Wave 11.B SSB must issue a
ruling; the policy `permitted_with_ssb_ruling` set + `ssb_ruling
_id` cite the ruling and escalate the verdict to HALAL_WITH_
CONDITIONS with the ruling_id surfaced in the audit trail.
Policy validation pinned: SSB-approved strategies require a
ruling_id (no auditable approval without citation); (5)
**PROTECTIVE_PUT HALAL** — buying insurance is the cleanest
case (defined premium for defined coverage, no gharar);
**underlying-must-be-halal-screened gate** — even PROTECTIVE_PUT
on a non-halal-screened underlying blocks (the strategy
inherits the underlying's haram status). Render output never
includes contract symbols / strike prices / expiration dates —
mirrors no-PII patterns of every other Round-4 wave. Halal
alignment: read-only screening; never opens a position; pure-
Python; no DB / network / async; frozen dataclasses. 51 tests
cover policy validation (ssb_ruling_id required when permitting;
naked/iron-condor/butterfly explicitly rejected from
permitted set — the construction guard pin); request
validation; **underlying-gate pin** in three flavours
(non-halal underlying blocks PROTECTIVE_PUT + COVERED_CALL +
LONG_CALL); **naked strategies pin** in three flavours (NAKED_
CALL / NAKED_PUT each with gharar message; even with halal
underlying + SSB approval of debated, naked still NOT_HALAL —
the override-everything pin); **spread strategies pin** in six
flavours (BULL_CALL_SPREAD / BEAR_PUT_SPREAD / IRON_CONDOR /
BUTTERFLY / STRADDLE / STRANGLE all categorically NOT_HALAL);
PROTECTIVE_PUT HALAL with no SSB ruling needed; **debated
strategies default DOUBTFUL** in four flavours (COVERED_CALL /
CASH_SECURED_PUT / LONG_CALL / LONG_PUT); SSB-approval
escalation pin (COVERED_CALL with operator's Wave 11.B ruling →
HALAL_WITH_CONDITIONS with cited ruling_id; selective approval
— operator approves COVERED_CALL but not CASH_SECURED_PUT, the
latter remains DOUBTFUL); OTHER strategy → UNKNOWN; frozen-
dataclass immutability; all enum string values pinned for
JSON / DB stability; render output across all five verdicts
with emoji (✅/❌/⚠️/📋/❓); **the no-strike-or-expiry pin**
asserts neither "strike" nor "expiry" nor "expires" appears in
render output; closed-set guarantee for naked + spread + debated
sets via category-coverage tests; and four end-to-end realistic
flows (typical PROTECTIVE_PUT journey for downside hedge → HALAL;
COVERED_CALL with SSB ruling SSB-2026-Q1-001 + multiple
permitted strategies → HALAL_WITH_CONDITIONS; attempted
IRON_CONDOR via policy construction → blocked at policy
construction time; result preserves strategy + underlying for
audit). Live options-chain ingest (Tradier / IBKR options API
adapter), strategy-construction layer (compose multi-leg orders
from individual legs), exit management (profit-target /
stop-loss + delta-hedging), Postgres `options_strategies` table,
the FastAPI route adapter (`/api/options/screen`), and the
dashboard's options tile rendering open positions with verdict
emoji deferred to follow-ups — screener verified in isolation
first.

### 12.C — Decentralised on-chain settlement (where halal) ✅ landed (route screener)

Ethereum / Solana-based settlement for crypto trades. Eliminates
exchange counterparty risk. Halal scrutiny for the underlying tokens
+ smart-contract audit.

**Landed (route screener)** as `web/settlement_route.py` — pure-
Python halal screener for proposed DEX swap routes. Six chains
(ETHEREUM_MAINNET / OPTIMISM / ARBITRUM / BASE / POLYGON /
SOLANA_MAINNET); nine DEX protocols (UNISWAP_V3 / V4 / SUSHISWAP /
CURVE / BALANCER / 1INCH aggregator / JUPITER / RAYDIUM / OTHER);
six liquidity models with `_HALAL_LIQUIDITY_MODELS` frozenset
(CONSTANT_PRODUCT / CONCENTRATED_LP / STABLE_CURVE /
ORDER_BOOK_ON_CHAIN HALAL; ORDER_BOOK_OFF_CHAIN DOUBTFUL with
gharar warning); four MEV protection levels (NATIVE_PROTECTION /
FLASHBOTS / MEV_BLOCKER pass; NONE → DOUBTFUL); five intermediate-
token statuses with the riba-via-back-door guard (FORBIDDEN →
NOT_HALAL; DEBATED_STABLECOIN like USDT → DOUBTFUL); reuses
Wave 12.E `SmartContractAudit` ladder. **Five pinned semantics**:
(1) **slippage > max threshold (default 5%) is NOT_HALAL** —
gharar on price uncertainty; > 10% policy max rejected at
construction (above gharar threshold by definition);
(2) **FORBIDDEN intermediate token is unconditionally NOT_HALAL**
— routing through interest-bearing wrapper / synthetic asset
inherits the riba; even with all-other-clean still rejected
(regression-pinned); (3) **DOUBTFUL** for ORDER_BOOK_OFF_CHAIN
(matched-price-not-fixed-at-submission gharar; off-chain relayer
shariah status unclear), UNAUDITED / SELF_AUDITED contracts,
MEV_PROTECTION=NONE (sandwich-attack exposure), DEBATED stablecoin
intermediate, cross-chain bridge use (extra custody + validator-
honesty assumption), > 3 hops (compounding gharar), slippage
between 2% soft warning and 5% max; (4) **INSUFFICIENT_DATA**
when liquidity_model UNKNOWN; (5) **render output never includes
wallet addresses, contract addresses, or transaction hashes** —
mirrors no-PII pattern of Wave 11.D + 11.C + 3.B + the no-message-
echo of Wave 12.G + the no-address pattern of Wave 12.E. Halal
alignment: read-only screening; never opens a position; pure-
Python; no DB / network / async; frozen dataclasses. 58 tests
cover policy validation (every guard); route validation; the
**slippage hard-rejection pin** (6% above 5% default → NOT_HALAL;
exactly 5% boundary inclusive falls to DOUBTFUL via soft-threshold;
strict 1%/0.5% policy flips a 2% verdict NOT_HALAL); the
**FORBIDDEN-intermediate pin** in two flavours (alone with riba
message; even with everything else clean still NOT_HALAL); the
**INSUFFICIENT_DATA pin** for UNKNOWN liquidity model; HALAL
happy paths (Uniswap V3; Curve stablecoin; Jupiter Solana;
**dYdX V4 ORDER_BOOK_ON_CHAIN passes** — the pin matters because
on-chain order books are structurally different from off-chain);
**DOUBTFUL pins** for off-chain orderbook (with operator-override
to disable flagging); UNAUDITED / SELF_AUDITED contracts;
AUDITED_INDIE passes (not just Big Four); MEV protection NONE
with native / Flashbots / MEV_BLOCKER all passing; DEBATED
stablecoin; bridge use; > 3 hops boundary inclusive at exactly 3;
slippage above soft threshold but below hard threshold; soft
threshold boundary at exactly 2% does not trigger; multiple-
warning aggregation (≥5 distinct flags surface together);
frozen-dataclass immutability across SettlementRoute +
SettlementScreenResult + SettlementRoutePolicy; all enum string
values pinned for JSON / DB stability; render output across all
four verdicts with emoji + chain + protocol + slippage %; the
**no-address-or-tx-hash pin** asserts neither `0x` nor `tx hash`
appear; result-shape sanity carrying chain + protocol metadata
through; and five end-to-end realistic flows (Uniswap V4 on
Optimism with Flashbots → HALAL — the best-case ETH-side path;
Jupiter Solana through USDC intermediate → HALAL; ETH→Polygon
bridge → DOUBTFUL with bridge warning; **worst-case-doubtful
multi-flag aggregate** — UNAUDITED + NONE MEV + 4.5% slippage +
DEBATED stablecoin + bridge + 4 hops surfaces ≥6 distinct
warnings; excessive-slippage NOT_HALAL even with all-other-clean
— the override-everything pin). Live router integration (httpx
client to 1inch / 0x / Jupiter quote APIs; per-cycle gas-strategy
estimator; the cycle-side hook that wraps every executor swap
with the gate before submitting on-chain), Postgres
`settlement_routes` + `settlement_audit_log` tables wired to
`db/repository.py`, the FastAPI route adapter (`/api/settlement/
screen`), and the dashboard's settlement-route tile rendering
historical swaps with their verdicts deferred to follow-ups —
screener verified in isolation first.

### 12.D — Halal venture capital allocation ✅ landed (gate)

Connect operator portfolios to halal-screened private market
opportunities (Wahed Ventures, etc.). Long-term "complete halal
wealth platform" play.

**Landed (gate)** as `web/halal_vc.py` — pure-Python deal-screen
+ allocation gate. Twelve `HalalSector` values (HEALTHCARE /
EDUCATION / AGRITECH / CLEAN_ENERGY / HALAL_FINTECH / SAAS_B2B /
LOGISTICS / HALAL_FOOD / MODEST_FASHION / PROPTECH / DEVELOPER_
TOOLS / BIOTECH) — closed enum, structural friction prevents
allocating to non-halal sectors. Three scrutiny sectors flagged
(HALAL_FINTECH adjacent to conventional banking; HALAL_FOOD
adjacent to alcohol/pork; MODEST_FASHION adjacent to mainstream
apparel). Seven `DealStage`s (PRE_SEED / SEED / SERIES_A / B / C
/ GROWTH / PRE_IPO) with PRE_SEED + SEED flagged as
DOUBTFUL_PIVOT. Seven `UseOfProceeds` values with **RETIRE_DEBT
unconditionally NOT_HALAL** (the riba-via-back-door pattern: a
halal business that uses raised equity to pay down conventional
debt is funding riba); UNDISCLOSED → INSUFFICIENT_DATA. Three
`FounderShariahCompliance` tiers (SCHOLAR_BOARD_BACKED — strongest;
SELF_DECLARED_HALAL — DOUBTFUL warning; UNKNOWN — DOUBTFUL with
"verify before allocating"). Five `VCDealVerdict` values
including new DOUBTFUL_PIVOT for early-stage. `VCAllocationPolicy`
defaults: max_per_deal_pct=10% (operators bump to 25% for high-
conviction; engine refuses > 50% at construction — category
error of putting half a portfolio in one illiquid 7-year-lockup
deal). `screen_deal(deal, *, policy)` returns `VCDealScreenResult`
with verdict + per-rule failures + warnings. `evaluate_allocation
(request, *, deal, policy)` combines deal screen with concentration
check and returns `VCAllocationDecision` carrying allowed +
requested_pct + cap_pct + reason + deal_verdict. **Six pinned
semantics**: (1) **closed sector enum** — the structural halal
guarantee shared with Wave 12.A robo-advisor; alcohol / gambling
/ tobacco / weapons / adult / pork / conventional_banking
categorically absent — regression test asserts none of those
strings appear in HalalSector values; (2) **DOUBTFUL_PIVOT for
PRE_SEED + SEED** — pre-product startups frequently pivot,
operators consult Wave 2.F scholar review for these; require_
scholar_board_for_pre_product=True default emits a stricter
warning when no review; (3) **UseOfProceeds=UNDISCLOSED →
INSUFFICIENT_DATA** — never silent HALAL on undisclosed proceeds
because operators need to verify proceeds aren't going to retire
conventional debt or finance haram operations; (4) **per-deal cap
inclusive at boundary** — exactly 10% allowed; > 10% blocked;
(5) **NOT_HALAL deal blocks regardless of pct** — even a 1%
allocation to a riba-back-door deal is rejected; (6) **render
output never includes USD** — minimum check size and user
portfolio totals never in render — pinned via test that "$"
absent. Halal alignment: structural sector closed-set; deal
screen is read-only; never opens a position; pure-Python; no DB
/ network / async; frozen dataclasses on every output. The
accredited-investor check delegates to Wave 11.C KYC engine
(HIGH risk + EDD layered on top per FATF private-market rules
— not re-implemented here). 60 tests cover policy validation
(zero / above-50 / 25 high-conviction accepted; negative lockup
disclosure rejected); VCDeal validation (every field invariant);
**hard rejection pins** (RETIRE_DEBT → NOT_HALAL with riba
message; even with everything else clean → still NOT_HALAL);
**INSUFFICIENT_DATA pin** for UNDISCLOSED in two flavours; HALAL
happy paths (Series A SaaS; growth-stage healthcare; Series B
education); **DOUBTFUL_PIVOT pins** (PRE_SEED with scholar board
still DOUBTFUL_PIVOT; SEED without scholar board with strict
warning; SERIES_A doesn't trigger pivot warning); **DOUBTFUL
pins** (scrutiny sector; UNKNOWN founder; self-declared founder;
no scholar review; multiple-signal aggregation); **closed sector
set** regression (no `alcohol` / `gambling` / `tobacco` /
`weapons` / `adult` / `pork` / `conventional_banking` values);
allocation evaluation (within-cap allowed; at-cap inclusive
allowed; above-cap blocked; NOT_HALAL blocks regardless of pct
— even tiny 1% allocations rejected; INSUFFICIENT_DATA blocks;
DOUBTFUL within cap allowed with verdict carried in decision;
DOUBTFUL_PIVOT within cap allowed; mismatched deal_id rejected;
strict 5% cap blocks 8% request); request validation; frozen-
dataclass immutability across all five output types; all enum
string values pinned; render output across all five verdicts
with emoji (✅/❌/⚠️/🌱/❓); render-no-USD pin; render-allocation-
decision with ALLOWED/BLOCKED + cap %; and three end-to-end
realistic flows (typical halal-fintech seed journey → DOUBTFUL_
PIVOT but allocation under 10% allowed; riba-back-door pattern
blocked even at 2%; concentration-protection lifecycle — 30%
blocked under default 10% cap; 30% blocked even under high-
conviction 25% cap; exactly-25% allowed under 25% cap). Wahed
Ventures / Curate Capital partner adapter (deal-flow API client),
Postgres `vc_deals` + `vc_allocations` tables wired to `db/
repository.py`, the FastAPI route adapters (`/api/vc/deals`
listing screened deals; `/api/vc/allocations` for the user's
allocation flow with the Wave 11.C KYC accredited-investor
check), and the dashboard's VC tile rendering deal flow with the
verdict emoji deferred to follow-ups — gate verified in isolation
first.

### 12.E — Halal real estate REITs (tokenised) ✅ landed (screener)

Once tokenised real estate platforms (RealT, Lofty) achieve halal
certification, integrate them. Diversification beyond public markets.

**Landed (screener)** as `web/tokenised_reit.py` — pure-Python
tokenised-real-estate halal screener. Re-uses Wave 1.I REIT
property-type taxonomy for the underlying physical asset; layers
on six tokenisation-specific concerns: TokenStandard (ERC20 /
721 / 1155 / SPL / NATIVE_OTHER / UNKNOWN); RegulatorRegistration
(SEC Reg A+ / D / MiCA Article 16 / OTHER / NONE); CustodyModel
(DIRECT_OWNERSHIP / SPV_OWNERSHIP / DERIVATIVE_RIGHTS);
SmartContractAudit (AUDITED_BIG_FOUR / AUDITED_INDIE /
SELF_AUDITED / UNAUDITED); YieldDenomination (RENT_DIRECT_FIAT /
USDC_STABLECOIN / USDT_STABLECOIN / OTHER_STABLECOIN /
NATIVE_CRYPTO / NONE); DeFiIntegration (STANDALONE /
LENDING_ENABLED / BORROWING_ENABLED / BOTH_ENABLED).
**Five pinned semantics**: (1) **DERIVATIVE_RIGHTS custody is
unconditionally NOT_HALAL** — the holder owns no physical asset
(gharar); (2) **LENDING_ENABLED / BORROWING_ENABLED / BOTH_ENABLED
is unconditionally NOT_HALAL** — even if the underlying property
is halal, the protocol enables riba. The asymmetric pin: a halal
warehouse REIT that the operator can't borrow against is HALAL;
the same REIT with on-chain borrowing enabled is NOT_HALAL because
the *protocol* enables riba even if the operator doesn't
personally borrow; (3) **DOUBTFUL** for SPV_OWNERSHIP (legal-
title transferability depends on jurisdiction), unregistered
offerings, unaudited / self-audited contracts, USDT / OTHER /
NATIVE_CRYPTO yield, hotel / specialty property type;
(4) **INSUFFICIENT_DATA when token_standard is UNKNOWN** —
operator must verify before allocating; (5) **render output
never includes wallet addresses or token contract IDs** — mirrors
no-PII pattern of Wave 11.D + 11.C + 3.B + the no-message-echo
pattern of Wave 12.G. `_HALAL_YIELD_DENOMINATIONS` is a closed
frozenset (RENT_DIRECT_FIAT / USDC_STABLECOIN / NONE) — operators
extend via code review, not runtime config. Halal alignment:
read-only screening; never opens a position; pure-Python; no DB
/ network / async; frozen dataclasses on every output. 44 tests
cover deal validation; the **DERIVATIVE_RIGHTS pin** in two
flavours (alone with gharar message; even with everything else
clean still NOT_HALAL); the **DeFi-riba pins** in four flavours
(LENDING_ENABLED → riba message; BORROWING_ENABLED; BOTH_ENABLED;
LENDING_ENABLED overrides clean flags — the asymmetric pin);
multiple-failure aggregation (DERIVATIVE_RIGHTS + LENDING_ENABLED
→ exactly 2 failures); INSUFFICIENT_DATA for UNKNOWN token
standard in two flavours; HALAL happy paths (best-case
direct-ownership-Reg-A+-Big-Four-USDC-standalone; direct-fiat-
yield variant; capital-appreciation-only NONE yield variant);
**DOUBTFUL pins** for SPV ownership (the typical RealT model);
no regulator; UNAUDITED contract; SELF_AUDITED; USDT yield;
OTHER stablecoin; NATIVE_CRYPTO yield; hotel / specialty
property; **multiple-warning aggregation** (SPV + no-reg +
unaudited + USDT → ≥4 distinct warnings); the **AUDITED_INDIE-
passes pin** (Trail-of-Bits-style indie audit accepted as HALAL —
not just Big Four); frozen-dataclass immutability; all enum
string values pinned for JSON / DB stability across all six
enums + verdict; render output across all four verdicts with
emoji (✅ / ❌ / ⚠️ / ❓); the **render-no-address contract**;
and three end-to-end realistic scenarios (RealT-shaped Detroit
residential SPV → DOUBTFUL via SPV warning even when otherwise
clean; Aave-style collateral-token Lofty warehouse with halal
property + direct ownership + Big Four audit but BORROWING_ENABLED
→ NOT_HALAL with riba message; unregulated DAO platform with
SPV + unaudited + USDT → DOUBTFUL with ≥4 warnings — the
multiple-flag-aggregation case). RealT / Lofty / Propy partner
adapter (deal-flow API client behind a `TokenisedREITProvider`
Protocol), Postgres `tokenised_reit_deals` table wired to `db/
repository.py`, the FastAPI route adapter (`/api/tokenised-reit/
screen`), and the dashboard's tokenised-RE tile rendering deals
with verdict emoji deferred to follow-ups — screener verified in
isolation first.

### 12.F — Localisation: Arabic + Urdu + Bahasa Malay UI ✅ landed (engine)

The biggest halal-trading audiences are in MENA, Pakistan, and
Indonesia. Full UI localisation.

**Landed (engine)** as `web/i18n.py` — pure-Python i18n core
covering the seven major halal-trading locales: EN (fallback) /
AR (MENA) / UR (Pakistan) / MS (Malaysia) / ID (Indonesia) / FR
(Maghreb) / TR (Turkey). `Locale` enum with stable BCP-47 string
values. `LocaleProfile` carries direction flag (is_rtl=True for
AR / UR; False for the rest), 3-letter ISO currency code,
currency symbol, currency-prefix-vs-suffix flag (USD prefix; SAR
+ EUR suffix), decimal + thousands separators (English uses `.`
+ `,`; ID + FR + TR use `,` + `.` / ` `; etc.), and percent
symbol. `MessageCatalog` is one locale's key→string dictionary
validated to reject empty keys + non-string values.
`translate(*, key, locale, catalogs, fallback_locale=EN, **placeholders)`
is the single entry point — returns the localised string with
placeholders substituted via `str.format`. **Six pinned semantics**:
(1) **missing translation falls back to English** — never empty
string; if EN also missing, returns the key itself with a
`UserWarning` so the operator's CI / log sweep catches
untranslated strings; (2) **RTL flagged on profile** — UI mirroring
is the frontend's job; the engine surfaces the directionality;
(3) **placeholder substitution refuses sensitive keys** — the
forbidden set (`password` / `api_key` / `secret` / `token` /
`private_key` / `session_id`) mirrors the no-secret denylist of
Wave 8.D OTLP + Wave 3.B vault + Wave 11.C / 11.D — guards
against the i18n surface becoming a secret-leakage vector via a
"log this template with values" debug feature; (4) **missing
placeholder raises** rather than leaving the brace literal in the
output (partial substitution renders badly in production);
(5) **locale-aware number/currency/percent/date formatting** —
`format_number(1234567.89)` → `"1,234,567.89"` (EN) /
`"1.234.567,89"` (ID) / `"1 234 567,89"` (FR); `format_currency`
respects prefix vs suffix per locale; AR / UR dates use DD-MM-YYYY,
others use ISO YYYY-MM-DD; (6) **`is_translation_complete`
audit helper** returns (is_done, missing_keys) so operators sweep
incomplete catalogs in CI before shipping. Halal alignment: i18n
engine never opens a position; pure-Python (`dataclasses` +
`enum` + `datetime` + `warnings`); no DB / network / async; frozen
dataclasses on every output. 66 tests cover Locale string values
+ RTL flags pinned per locale; default profiles for every locale
(AR has SAR suffix; EN has USD prefix; ID uses European decimal
convention; FR uses space thousands); LocaleProfile validation
(currency_code length; empty currency_symbol / decimal_separator
rejected); MessageCatalog validation (empty key + non-string
value rejected); translate happy paths (EN; AR; with placeholders;
AR with placeholders); **fallback semantics in four flavours**
(missing-in-target falls back to EN with warning; missing locale
entirely falls back to EN with warning; missing in both returns
key with "returning key as-is" warning; missing-translation never
returns empty string — the load-bearing pin); EN target with
fallback also EN does NOT warn (no-noise pin); **forbidden-
placeholder pins** for every value in the denylist (password /
api_key / secret / token); missing-placeholder raises KeyError;
empty-key rejected at translate entry; format_number across all
locales with separator swapping (EN; ID swap; FR space); negative
+ zero values handled; format_currency with prefix EN + suffix AR
+ suffix FR + prefix ID with European separators; format_percent
with locale decimal + negative + above-100 cases; format_date
with EN/FR ISO + AR/UR DD-MM-YYYY + naive-datetime rejected;
is_translation_complete returns (True, ()) when all keys present
and (False, missing) when any missing; frozen-dataclass
immutability across LocaleProfile + MessageCatalog; render_locale_
profile shows direction + currency + format example; and three
end-to-end realistic flows (Saudi user sees AR welcome + RTL flag
+ SAR suffix currency + DD-MM-YYYY date; Indonesian user sees IDR
prefix + European decimal convention; partially-translated FR
catalog falls back per-key with mixed-language UI). Operator-side
translation workflow (Crowdin / Lokalise / messages.json) +
Postgres `i18n_messages` table wired to `db/repository.py` + the
FastAPI middleware that infers user locale from `Accept-Language`
header + the dashboard locale-switcher widget deferred to follow-
ups — engine verified in isolation first.

### 12.G — AI co-pilot mode (24/7 conversational) ✅ landed (intent gate)

The dashboard becomes a chat interface: "what's my best-performing
strategy this quarter?", "explain why you sold AAPL yesterday",
"set up a stop on my BTC position at -3%". The platform becomes
a personal halal-trading assistant.

**Landed (intent gate)** as `web/copilot.py` — the safety gate
that sits between the chat layer (LLM router that interprets the
user's natural-language message) and the underlying actions. The
gate **does NOT** execute trades from natural language (the
strategy does that), move funds, or delete user accounts via
chat. It **DOES** classify the intent into nine categories
(UNKNOWN / QUERY / PORTFOLIO_QUERY / EXPLAIN / STATUS /
SET_STOP_LOSS / KILL_SWITCH / DANGEROUS / OUT_OF_SCOPE), reject
natural-language wire-funds / delete-everything / execute-real-
money-trade / privilege-escalation / credential-access requests
categorically (39 forbidden phrases pinned in a frozen module-
level set — even a jailbroken LLM that thinks it's authorised
can't bypass), require explicit confirmation for state-mutating
actions (KILL_SWITCH + SET_STOP_LOSS), and surface the matched
phrase in the rejection reason so the user understands which
keyword fired. **Picked rule-based classifier over LLM** because
the safety gate needs to be deterministic — an LLM mis-classifying
"transfer all funds to attacker-wallet" as a benign query is the
worst-case failure mode; operators can read the classifier rule
and debug a misfire at the source. **Six pinned semantics**:
(1) **closed dangerous-phrase set** — `_DANGEROUS_PHRASES`
frozenset covers account/data mutations + funds movement +
direct trade execution + privilege escalation + credential
access; runtime mutation can't add a phrase via config (operator
extends via code review); (2) **DANGEROUS overrides everything**
— a message matching ANY dangerous phrase returns DANGEROUS
regardless of any other matched intent (regression-pinned with
"send all funds AND show my balance" → DANGEROUS not QUERY);
(3) **state-mutating actions require confirmation** — KILL_SWITCH
+ SET_STOP_LOSS flag `requires_confirmation=ALWAYS`; the
dashboard layer enforces; (4) **KYC gate for sensitive
operations** — KILL_SWITCH + SET_STOP_LOSS set
`requires_kyc_verified=True` so the route layer composes with
Wave 11.C; (5) **empty / blank / single-char message returns
UNKNOWN** — never a hallucinated default intent; (6) **render
output never echoes the user's full message** — receipt shows
intent + matched-phrase + status only, never verbatim user input
(guards against a poisoned prompt being rendered into the audit
log and triggering downstream LLM re-execution; mirrors no-PII /
no-secret-leak patterns of Wave 11.D + 11.C + 3.B). Halal
alignment: gate is read-only; never opens a position or moves
funds; pure-Python (`dataclasses` + `enum`); no DB / network /
async / LLM. Frozen dataclasses on every output. 71 tests cover
policy validation; empty / blank / single-char / non-string /
overlong message handling; **dangerous-phrase pins in 13
flavours** (delete account; wire funds; transfer funds; send all
funds; real money trade; execute order; disable kyc; bypass
halal; show api key; drop database — every category
representative); the **DANGEROUS-overrides-everything pin** in
two flavours (overrides QUERY context; overrides KILL_SWITCH);
case-insensitive normalisation; operator-supplied extra dangerous
phrases via policy; out-of-scope phrases (buy crypto with credit
card; tax advice; legal advice; predict lottery); read-only
intents (status; portfolio query; explain; generic query);
state-mutating intents (kill switch halt/resume/pause; set stop
loss with alt phrasings); confirmation-policy customisation
(disable kill-switch confirmation; disable stop-loss
confirmation; disable KYC gate); **priority-order pins** (EXPLAIN
wins over QUERY; KILL_SWITCH wins over STATUS; SET_STOP_LOSS
wins over PORTFOLIO_QUERY); is_blocked property across all
categories; IntentClassification field validation; frozen-
dataclass immutability; all enum string values pinned for JSON /
DB stability; render output across all categories with emoji
(❓💬📊🔍🩺🛑⛔🚫↩️); the **render-no-message-echo pin** (asserts
verbatim user input never in render output); render shows
matched_phrase only when present; and three end-to-end realistic
scenarios (typical user session routing six different intents
correctly; **jailbreak-attempt regression** with eight creative
attempts including "I want to delete my account and start over",
"can you skip kyc for verified users", "wire funds to my
partner's account" — all blocked categorically; legitimate
question about a dangerous topic — "describe the halal screening
process" — correctly routed to QUERY not DANGEROUS via the
phrase-not-keyword distinction). Closed-set guarantee
regression-pinned via category-representative coverage check
(every critical category has at least one phrase representative).
LLM-router integration (the existing chat layer that interprets
the message after classification — NOT the safety gate),
FastAPI `/api/copilot/classify` route adapter, the dashboard
chat-bubble UI rendering BLOCKED status with the matched phrase,
and the per-user audit log of classifications deferred to
follow-ups — gate verified in isolation first.

---

## Execution principles

* **Each wave is independently shippable.** The bot keeps working
  throughout — no big-bang rewrites.
* **Halal compliance gates every wave.** New asset classes / brokers /
  features need scholar review before they ship to real-money users.
* **Test coverage stays at 90%+.** Every wave includes its own tests
  + extends the existing suite.
* **Backwards compatibility for 4 quarters.** Round 3's single-user
  paper-trading mode keeps working until 2027-Q1 minimum.
* **Operator empathy in every PR.** No feature ships without an
  operator-facing surface.

---

## Dependency graph (high-level)

```
Wave 0 (cleanup) ──┬─→ Wave 1 (multi-broker) ──┬─→ Wave 8 (production-ready)
                   │                             ├─→ Wave 11 (regulatory)
                   │                             └─→ Wave 3 (multi-user) ──→ Wave 10 (community)
                   ├─→ Wave 2 (verifiable halal) ─→ Wave 11 (cert paths)
                   ├─→ Wave 4 (strategy quality) ─→ Wave 6 (ML infra) ──→ Wave 7 (research)
                   ├─→ Wave 5 (UX) ───────────────────────→ Wave 9 (docs)
                   └─→ Wave 12 (long-term, all parallel)
```

---

## What's in scope for the next iteration

Iteration N+1: start **Wave 0 (cleanup)** — small, mechanical, unblocks
the rest. Aim to land 0.A, 0.B, 0.C in the first sprint. Each task lands
with tests + docs + a one-line acceptance test.
