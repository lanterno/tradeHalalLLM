# Round 5 Roadmap — "The Definitive Halal Trading Platform"

> **Vision**: Move from "the best halal trading bot one operator can run"
> to "the definitive halal trading platform on Earth." Every Muslim
> investor — retail to institutional — should be able to use this app
> with full confidence that (a) every trade is verifiably Shariah-
> compliant, (b) the strategy quality matches or exceeds top quant
> shops, (c) the operator/UX experience is best-in-class, (d) the
> educational and community surface helps users grow, and (e) the
> entire system survives audit by AAOIFI / IFSB / SEC / FCA / CMA /
> SC Malaysia.

Round 4 transformed the bot into a multi-user, multi-asset platform
with strong observability and an exhaustive auxiliary-primitive
library. Round 5 is the **completion round**: everything missing
from "world-class" gets shipped, and the platform becomes the
canonical reference implementation of algorithmic halal trading.

---

## Guiding Principles

1. **Halal first, halal always.** Every wave preserves verifiable
   compliance against AAOIFI Shariah Standard 21 (Financial Papers),
   IFSB-1, and the seated school majority position. Edge cases go
   to scholars, not to the LLM. The audit trail is the contract.
2. **Operator empathy at every layer.** No feature ships without
   a dashboard tile + CLI surface + alert path + runbook.
3. **Test-first, replay-consistent.** Every primitive is pure-
   functional; every cycle stage is reconstructible from inputs;
   every report is deterministic.
4. **Backwards-compatible always.** Round-4 surfaces stay live;
   new features ride on flag gates and graceful degradation.
5. **Observable everywhere.** Prometheus + Grafana + structured
   JSON logs + OTel traces extend, never bypass.
6. **Educational by design.** Every interaction can be a teaching
   moment for a new Muslim investor learning halal markets.

---

## Wave 0 — Round-5 cleanup + foundations (2-3 weeks)

Land before any new vertical. Pays compounding dividends across the rest of the round.

### 0.A — Adopt the Round-4 aux primitives across the live cycle path

The Round-4 aux library (idempotency, circuit breaker, rate limiter,
preflight, scholar calendar, settings drift, deprecation, etc.) is
fully tested in isolation but not yet wired into `BinanceAdapter` /
`AlpacaAdapter` / `BaseTradingBot.initialize()`. Wire them in:
preflight on startup, idempotency on every order submission,
circuit breaker on every adapter call, rate limiter before each
upstream hit. Acceptance: a deliberately-broken upstream produces
clean operator alerts via the breaker rather than cascading errors.

### 0.B — Unified `Adapter` protocol consolidating Round-4 primitives

Define `core/adapter_base.py:ResilientAdapter` as a Protocol that
composes idempotency + circuit breaker + rate limiter into a
single `call(operation, payload)` interface. Each broker /
screener / LLM provider implements one.

### 0.C — Migrate test database to per-test schema isolation

`tests/conftest.py` currently TRUNCATEs per test. Move to per-test
schema (PostgreSQL `CREATE SCHEMA test_<id>`) so parallel pytest
runs (`-n auto`) work without cross-test interference.

### 0.D — Deterministic time + UUID injection for cycle replay ✅ landed (core)

`src/halal_trader/core/clock.py` ships the `Clock` + `IdSource`
Protocols plus `SystemClock` / `SystemIdSource` (production) and
`FrozenClock` / `SeededIdSource` (replay / test). `FrozenClock` is
mutation-only via `advance()`; `SeededIdSource` re-derives UUIDs
from a counter via UUID5, so two seeded instances produce
identical UUID sequences. Cycle migration of existing
`datetime.now(UTC)` / `uuid.uuid4()` call sites behind these
protocols is the still-pending follow-on.

### 0.E — Migrate from raw `print` / `logger.info` to structured events

There are still ~30 `logger.info("…")` call sites in `crypto/`,
`trading/`, `core/` without `extra={"event": ...}`. Audit and
migrate each to the `core/events.py` constants pattern so JSON
logs stay queryable.

### 0.F — Runtime config hot-reload sentinel ✅ landed (`core/hot_reload.py`)

Operators currently restart the bot to change `Settings`. Land
`config/hot_reload.py` that watches `.env` mtime and reloads
non-secret settings without restart (secrets still require
restart). The kill-switch + halt state stay across reload.

### 0.G — Standardize all `*Policy` dataclasses behind `domain/policies.py` ✅ landed (registry core)

`src/halal_trader/domain/policies.py` ships the policy registry +
snapshot round-trip primitives (`register_policy`,
`register_policy_decorator`, `policy_class`, `snapshot_to_dict`,
`snapshot_from_dict`). The registry is import-time-built,
idempotent per id (re-registering the same class is a no-op,
re-binding to a different class raises). Snapshot round-trip
recurses into nested dataclasses + handles Enum/frozenset
coercion. The follow-on is registering the existing 30+ scattered
`*Policy` classes (one one-line `register_policy(...)` per
module) so tooling can iterate them uniformly.

### 0.H — Migration: rename "halal_trader" to canonical "halal_platform"

Round-4 still uses the original "halal_trader" package name; for
brand alignment and to signal the platform graduation, rename
to "halal_platform" with a one-time bridge import.

---

## Wave 1 — AAOIFI 100% Shariah compliance (4-6 weeks)

Round 4 covered the major AAOIFI Standard 21 requirements; Round 5
covers every standard relevant to public-equity trading.

### 1.A — Full AAOIFI Standard 21 (Financial Papers) coverage matrix ✅ landed (`halal/aaoifi_standard_21.py`)

`halal/aaoifi_standard_21.py`: structured machine-readable encoding
of every clause in AAOIFI Standard 21 (Financial Papers), with
each line of the standard tagged to a screener rule + a test
case + a render line. Operators see "Compliant with clause 3.2.1
of AAOIFI Standard 21" rather than "passes screening."

### 1.B — AAOIFI Standard 17 (Investment Sukuk) integration ✅ landed (`halal/aaoifi_standard_17.py`)

For the sukuk vertical (Wave 3), the screening must follow Standard 17.
Land the rule encoding + test fixtures + scholar verdict templates.

### 1.C — AAOIFI Standard 30 (Monetization Rule) edge cases ✅ landed (`halal/aaoifi_standard_30.py`)

Standard 30 governs commodity Murabaha and reverse Murabaha — load-
bearing for any fixed-income halal alternative. Encode the Bay'
al-Inah ban, the constructive-possession requirement, the rate-cap
guidance.

### 1.D — Multi-school consensus engine ✅ landed (`halal/multi_school_consensus.py`)

**Implementation**: `src/halal_trader/halal/multi_school_consensus.py`
ships the multi-school aggregator. The standard halal screener
returns a single PERMISSIBLE/IMPERMISSIBLE verdict using one
school's methodology (Hanafi defaults); many trades are halal in
one school but disputed in another. This module aggregates the
four Sunni schools (Hanafi/Shafi'i/Maliki/Hanbali) plus optional
Ja'fari (Shia) for inclusivity, surfacing consensus level so
operators can apply their preferred strictness. `School` enum
(HANAFI/SHAFII/MALIKI/HANBALI/JAFARI) pinned string values.
Module-level `SUNNI_SCHOOLS: frozenset[School]` containing exactly
the four Sunni schools. `SchoolVerdict` enum (PERMISSIBLE /
IMPERMISSIBLE / ABSTAIN) — ABSTAIN means the school hasn't formally
opined (common for novel fintech). `ConsensusMode` enum
(UNANIMOUS / MAJORITY / ANY). `SchoolPosition` carries school +
verdict + reasoning + optional scholar_handle (e.g.,
"mufti_taqi_usmani"); validation rejects empty reasoning + empty
(but non-None) scholar_handle. `ConsensusReport` enforces
**count-consistency invariant** (sum equals positions length) +
**no-duplicate-schools structural pin**. Six derived properties:
`total_engaged` (excludes ABSTAIN); `is_unanimous_permissible`
(all engaged say yes; **abstain doesn't break unanimity** pin);
`is_unanimous_impermissible`; `is_majority_permissible`
(strict greater-than, ties don't count); `is_split` (both sides
present); `is_sunni_consensus` (all four Sunni schools engaged
PERMISSIBLE). `build_report` sorts by School enum order for
deterministic output. `tradable_under_consensus(report, *, mode)`
applies the strictness gate: UNANIMOUS = unanimous-permissible
+ at-least-one-engaged; MAJORITY = strict permissible >
impermissible; ANY = at least one PERMISSIBLE. **Default mode is
MAJORITY** pinned via test. `disagreement_summary` returns the
minority side (tied → IMPERMISSIBLE side by conservative convention).
Render with verdict emoji ✅/❌/❔ + canonical school labels
("Shafi'i" with apostrophe; "Ja'fari"); top-line ✅/❌ TRADABLE
+ split warning. **No-secret-leak pin**: render never includes
`@`/`zoom.us`/`meet.google`/`private_email`/`+1-` substrings.
Tests in `tests/test_multi_school_consensus.py` (59 cases): all
enum string-value pins; **SUNNI_SCHOOLS exact-set pin** (excludes
Ja'fari); SchoolPosition validation including empty-reasoning
rejected + empty-scholar_handle rejected (None allowed); report
**count-consistency pin**, **no-duplicate-schools pin**, negative
counts rejected, immutability; total_engaged excludes ABSTAIN;
**unanimous-permissible-with-abstain pin** (engaged ones agree);
**unanimous-requires-engagement pin** (empty + all-abstain →
False); **majority strict-greater-than pin** (1-1 tied not
majority); split = both sides; **Sunni-consensus requires-all-4-
Sunni pin** (3-of-4 fails); **Sunni-consensus excludes-Ja'fari pin**
(Ja'fari abstaining doesn't matter); ABSTAIN doesn't satisfy Sunni
engagement; build_report sorts by School-enum order; **UNANIMOUS-
mode-blocks-empty pin**; **MAJORITY-mode-blocks-tied pin**;
**ANY-mode-allows-with-one-permissible pin**; **default-mode-is-
MAJORITY pin**; disagreement returns minority side both directions
+ tied-returns-impermissible-side conservative convention; render
includes school label + verdict emoji + reasoning; scholar_handle
included when set / omitted when None; render no-secret-leak pin;
e2e flows (AAPL unanimous PERMISSIBLE — passes all three modes;
**disputed-name 3-vs-1 split** — UNANIMOUS blocks, MAJORITY allows
— matches Saudi-strict vs Pakistani-permissive operator preferences;
tobacco unanimous IMPERMISSIBLE — every mode blocks; **novel-fintech
partial-abstain pin** — UNANIMOUS allows because engaged ones agree
but is_sunni_consensus False because not all 4 Sunni opined; replay
consistency).

### 1.E — Gharar (excessive uncertainty) detector ✅ landed (`halal/gharar_detector.py`)

**Implementation**: `src/halal_trader/halal/gharar_detector.py`
ships the structural-uncertainty overlay. Where maysir
(`halal/maysir_screen.py`) catches gambling-pattern equities,
gharar catches the third independently-prohibited Shariah issue:
excessive uncertainty in the contract or instrument itself
(undisclosed underlying, opaque counterparty, asset-backing
opacity, contingent payoff, indeterminate delivery, nested
derivative wrappers, dual-class unequal rights, opaque fees).
The two compose: a name passes standard halal screen AND maysir
AND gharar → tradable. `GhararLevel` enum (NONE / MINOR / MODERATE
/ SEVERE) pinned string values; SEVERE is non-tradable. `GhararSignal`
enum: UNDISCLOSED_UNDERLYING / COUNTERPARTY_UNDISCLOSED (both
weight 3 — AAOIFI Standard 21 "void contract" triggers) /
ASSET_BACKING_OPAQUE / CONTINGENT_PAYOFF / FUTURE_DELIVERY_
INDETERMINATE / NESTED_DERIVATIVE_LAYERS (all weight 2) /
DUAL_CLASS_UNEQUAL_RIGHTS / OPAQUE_FEE_STRUCTURE (weight 1).
`GhararPolicy` defaults: nested_layers_threshold=2, score
cutoffs 1/3/5; **score-threshold ordering pin** (minor < moderate
< severe). `GhararInputs` carries instrument_id + 8 disclosure
flags defaulting to "clean" so operators only set concerning
ones; rejects negative derivative_layers + empty/whitespace id.
`GhararAssessment` carries id + signals + level + score with
**NONE ↔ empty-signals structural pin** in both directions.
`assess_gharar(inputs, *, policy)` runs each detector
independently, sums weights, maps via cutoffs. `assess_batch`
sorts by id. `is_tradable(assessment)` = True for NONE/MINOR/
MODERATE; False only for SEVERE. Render with level emoji ✅/🟢/🟡/🔴
+ sorted signal labels; **no-secret-leak pin** (no `prospectus`
/ `Annex A` / `Schedule II` / `Authorization` substrings). Tests
in `tests/test_gharar_detector.py` (47 cases): all enum string-
value pins; policy validation including **nested_layers >=2
required pin** + **score-threshold ordering pin**; signal
detection tested in isolation for each of 8 detectors;
**nested-layers-threshold-inclusive pin** (2 fires; 1 doesn't);
score-to-level mapping at each band: weight-1 alone = MINOR;
weight-2 alone = MINOR (still below 3); weight-3 alone = MODERATE
(at threshold); 3+2=5 = SEVERE; 3+3=6 = SEVERE; assessment
**both-direction structural pins** (NONE-with-signals rejected;
non-NONE-without-signals rejected); **is_tradable only blocks
SEVERE pin** (NONE/MINOR/MODERATE all pass — distinct from
maysir's HIGH/EXTREME-blocked design); filter_blocked; assess_
batch sorted determinism; render no-secret-leak; e2e flows
(opaque-structured-note 4-signal SEVERE — undisclosed underlying
+ opaque counterparty + 3 nested layers + opaque fees → score 9;
plain-equity passes clean as NONE; **dual-class governance MINOR
only pin** (Google-style alone = MINOR, tradable); **Salam-
indeterminate-delivery + undisclosed-counterparty SEVERE pin**
(Standard 7 violation — voidable contract); replay consistency;
**custom-strict policy SEVERE-at-3 pin** for stricter operators).

### 1.F — Riba detection across derivatives ✅ landed (`halal/riba_detector.py`)

**Implementation**: `src/halal_trader/halal/riba_detector.py` ships
the instrument-class-keyed riba classifier. The four-tier halal
gate becomes complete: standard halal screen + Wave 1.D multi-
school consensus + Wave 1.G maysir + Wave 1.E gharar + this
Wave 1.F riba. `RibaType` enum (NASIYAH / FADL / EMBEDDED_FINANCING
/ DEBT_SALE / LEVERAGE_INTEREST) — five documented violation
types each mapping to a classical fiqh reference. `InstrumentClass`
enum with 14 documented classes spanning the canonical halal-
tradable set (SPOT_EQUITY / PHYSICAL_COMMODITY / SUKUK /
WAAD_FORWARD / SALAM_FORWARD / ARBOUN_OPTION) AND the canonical
non-halal set (CONVENTIONAL_BOND / CONVENTIONAL_FUTURE /
CONVENTIONAL_OPTION / INTEREST_RATE_SWAP / CURRENCY_SWAP / CFD /
LEVERAGED_ETF / INVERSE_ETF). Module-level `_BASE_VERDICT` dict
maps each class to the riba types it always carries (regardless
of operator flags); module-level `HALAL_BY_DEFAULT_CLASSES`
frozenset exposes the 6 clean-by-default classes for dashboard
filtering. `RibaPolicy` defaults: flag_margin_as_riba=True,
flag_borrow_as_riba=True; **no policy knob can remove a base
verdict** (fiqh determination, not configuration). `RibaInputs`
carries id + class + 5 boolean flags (uses_margin /
uses_borrowed_securities / has_embedded_financing_rate /
pays_or_receives_fixed_interest / is_debt_traded_off_face).
`RibaAssessment` carries id + class + frozenset of detected
riba types; `is_clean` property True only when riba_types is
empty. `assess_riba` combines base | extra. Render with ✅/❌
+ class label + sorted riba labels; **no-secret-leak pin** (no
`financing_schedule`/`counterparty_id`/`bps`/`Authorization`
substrings). Tests in `tests/test_riba_detector.py` (46 cases):
RibaType + InstrumentClass enum string-value pins; **HALAL_BY_
DEFAULT_CLASSES exact-set pin** (6 classes); **conventional
classes excluded pin**; policy default pins + immutability;
RibaInputs validation; **all 6 halal-by-default classes return
clean pin** (spot equity, physical commodity, sukuk, wa'd
forward, salam forward, arboun option); **conventional bond
NASIYAH always pin**; conventional future EMBEDDED_FINANCING;
conventional option EMBEDDED_FINANCING; **CFD both EMBEDDED + 
LEVERAGE pin**; leveraged ETF LEVERAGE_INTEREST; inverse ETF
EMBEDDED_FINANCING; interest-rate swap NASIYAH; currency swap
EMBEDDED_FINANCING; **margin-flag adds LEVERAGE pin**; **borrow-
flag adds NASIYAH pin** (lending fee = interest); embedded-
financing flag; fixed-interest flag NASIYAH; debt-sale flag;
**permissive policy can disable margin/borrow flag pin**;
**permissive policy CANNOT clear base verdict pin** (conventional
bond stays NASIYAH even with both flags off); **base + extra
combine pin** (conventional future + margin = EMBEDDED_FINANCING
+ LEVERAGE); assessment immutability; is_clean property both
directions; assess_batch sorted; filter_blocked; render clean
shows ✅ + "no riba detected"; render dirty shows ❌ + class
label + sorted riba labels; **render no-secret-leak pin**; e2e
flows (**classic halal portfolio all clean pin** — spot
equities + sukuk + physical gold + wa'd forward all return
is_clean; **classic non-halal bundle all blocked pin** — margin
stocks + bonds + CFDs + futures + leveraged ETFs all flagged;
**AAOIFI Standard 21 + 30 reference pin** — interest-rate swap
correctly classified as NASIYAH and rendered with "interest rate
swap" + "riba al-nasiyah" labels; replay consistency).

### 1.G — Maysir (gambling) screen for high-volatility names ✅ landed (`halal/maysir_screen.py`)

**Implementation**: `src/halal_trader/halal/maysir_screen.py` ships
the structural overlay on the standard halal screener. Standard
halal screening checks debt ratios + revenue purity + sector;
maysir catches the case where a name is technically halal-compliant
on paper but trades like gambling rather than investment (penny
stocks, meme pumps, zero-coverage names with extreme short-
interest). `MaysirRisk` enum (NONE < LOW < MODERATE < HIGH <
EXTREME) pinned string values; HIGH + EXTREME non-tradable.
`MaysirSignal` enum: PENNY_PRICE / EXTREME_SHORT_INTEREST /
RETAIL_FLOW_DOMINANT / NO_ANALYST_COVERAGE / EXTREME_VOLATILITY /
MEME_PUMP_PATTERN / ZERO_REVENUE — closed catalogue, code review
to add. Module-level `_SIGNAL_WEIGHT` dict assigns weights:
ZERO_REVENUE=3 (most severe — no business model), PENNY_PRICE /
MEME_PUMP_PATTERN / EXTREME_SHORT_INTEREST=2, the rest=1.
`MaysirPolicy` defaults: penny <$5, short interest >50%, retail-
flow >70%, zero analyst coverage, vol >100% annualized, vertical
30d move >100%; **score-threshold ordering pin** (moderate < high
< extreme = 2 < 4 < 6 by default). `MaysirInputs` carries ticker
+ price + short_interest_pct (allows >100% for synthetic shorts,
caps at 1000 for sanity) + retail_flow_pct + analyst_coverage +
realized_vol + 30d_change + revenue_ttm. `MaysirAssessment`
carries ticker + frozenset of fired signals + risk + score; **NONE
↔ empty-signals structural pin** (NONE risk requires empty signals;
non-NONE requires at least one signal). `screen_for_maysir(inputs,
*, policy)` runs each signal detector independently, sums weights,
maps via score cutoffs to risk level. `screen_batch` returns
sorted-by-ticker tuple. `is_tradable(assessment)` is the load-
bearing gate (True for NONE/LOW/MODERATE; False for HIGH/EXTREME).
Render with risk emoji ✅/🟢/🟡/🟠/🔴 + sorted signal labels;
**no-secret-leak pin** (no `reddit.com`/`robinhood`/`Authorization`
substrings). Tests in `tests/test_maysir_screen.py` (57 cases):
all enum string-value pins; policy validation including **score-
threshold ordering pin** both directions; **synthetic-short >100%
allowed pin**; signal detection tested in isolation for each
detector; **boundary-strict-less-than pins** (penny at $5.0
doesn't fire, $4.99 does; short-interest at 50% doesn't fire,
51% does); **meme-pump one-sided pin** (negative 30d move
doesn't fire); score-to-risk mapping at each band including
**extreme inclusive at threshold pin**; assessment validation
**both-direction structural pins** (NONE-with-signals rejected;
non-NONE-without-signals rejected); is_tradable across all 5 risk
levels; filter_blocked; screen_batch sorted determinism;
render no-secret-leak pin; e2e flows (**meme-stock 6-signal
EXTREME pin** GME-style profile fires every signal except
ZERO_REVENUE; blue-chip MSFT-style passes clean as NONE; **pre-
revenue biotech HIGH pin** ZERO_REVENUE+EXTREME_VOLATILITY = score
4 → HIGH; replay consistency; custom-policy loosens to LOW).

### 1.H — Continuous re-screening on corporate actions ✅ landed (`halal/continuous_screen.py`)

When a screened company does a debt issuance / acquisition, its
ratios change overnight. `halal/continuous_screen.py` subscribes to
SEC 8-K filings + earnings releases and re-runs the screen
within 24h, flagging holdings that flipped non-compliant.

### 1.I — Time-weighted purification for partial holdings ✅ landed (`halal/time_weighted_purification.py`)

**Implementation**: `src/halal_trader/halal/time_weighted_purification.py`
ships the prorating complement to existing `halal/purification.py`
+ `halal/purification_schedule.py` + `halal/round_trip_purification.py`.
For actively-traded portfolios where the user holds shares for
only part of the revenue-generation period, two methodologies
are operator-selectable: FULL_AMOUNT (standard conservative —
full impure_pct × dividend) and HOLDING_PRORATED (impure_pct ×
dividend × days_held / days_in_period). `PurificationMethod`
enum (FULL_AMOUNT / HOLDING_PRORATED) pinned string values;
**default is FULL_AMOUNT** (more conservative — over-paying is
generosity; under-paying is religious obligation gap).
`HoldingPeriod` carries holding_id + start_date + end_date (None
= still held) + share_count; rejects empty id, zero/negative
shares, end_before_start. `DividendEvent` carries period_start +
period_end + ex_date + amount_per_share + impure_revenue_pct ∈
[0.0, 1.0]; **ex_date must be within [period_start, period_end]
pin**; `days_in_period` property is inclusive of both endpoints.
`PurificationAssessment` carries id + eligible (binary on ex-
date) + gross_dividend + impure_amount_full + purification_owed
+ method_used + days_held_in_period + days_in_period +
holding_fraction. **Five structural invariants pinned**:
purification_owed ≤ impure_amount_full; holding_fraction ∈ [0,1];
days_held >= 0; days_in_period >= 1; ineligible holdings cannot
have gross_dividend > 0 or purification_owed > 0. `calculate_
purification(holding, dividend, *, today, method)` runs:
eligibility = `holding.start_date <= dividend.ex_date <=
effective_end_date` (where effective_end_date is end_date or
today for still-held); ineligible → returns zero values; eligible
→ gross = amount × shares; impure_full = gross × impure_pct;
HOLDING_PRORATED applies × holding_fraction; FULL_AMOUNT uses
impure_full directly. `_holding_overlap_days` computes the
clamped inclusive overlap; **caps at days_in_period** (holding
longer than period doesn't earn extra purification). `total_owed`
sums across assessments. Render with 💧/⏸ + summary; **no-secret-
leak pin** (no `buy_price`/`sell_price`/`P&L`/`cost_basis`/
`Authorization` substrings). Tests in `tests/test_time_weighted_
purification.py` (47 cases): PurificationMethod string-value pin;
HoldingPeriod validation (empty id, zero shares, end-before-
start all rejected; immutability; **still-held end_date=None
allowed pin**); DividendEvent validation (period_end-before-
start, **ex_date-outside-period pin** both directions, negative
amount, **impure_pct outside [0,1] both bounds rejected pin**,
**impure=0.0 + 1.0 both allowed pins**, days_in_period inclusive
single-day pin); eligibility (covers ex-date eligible; bought-
after-ex ineligible; sold-before-ex ineligible; still-held
through today eligible; **start-on-ex-date inclusive pin**;
**end-on-ex-date inclusive pin**; ineligible returns zero
dividend); FULL_AMOUNT method (full quarter → 100×$1×5%=$5;
**brief 3-day holding still pays full pin**; **default method is
FULL_AMOUNT pin**); HOLDING_PRORATED method (full period prorates
to 1.0; partial 30/90 days; 3-day window 3/90; **caps-at-period
pin** holding longer than period gives fraction=1.0; partial
overlap Feb 15→Mar 31 = 45 days); assessment validation
(immutability; **purification cannot exceed impure_full pin**;
**holding_fraction outside [0,1] rejected pin**; **ineligible-
with-dividend rejected pin**; negative days_held rejected; zero
days_in_period rejected); total_owed sums correctly + empty
returns 0; render eligible shows 💧 + amount + method label;
ineligible shows ⏸ + "not eligible"; method label included;
**render no-secret-leak pin**; e2e flows (**active-trader 5-day
holding pin** — FULL_AMOUNT owes $5, HOLDING_PRORATED owes $0.28
= 5/90 × $5; long-term-holder methods agree; replay consistency;
**zero-impure dividend zero purification regardless of method
pin**).

### 1.J — Multi-currency Zakat calculator ✅ landed (`halal/zakat.py`)

**Implementation**: `src/halal_trader/halal/zakat.py` ships the
pure-functional Zakat math distinct from existing
`halal/purification.py` (purification = riba income from holdings;
zakat = annual wealth tax on net assets above nisab). Module-level
documented constants: `DEFAULT_GOLD_NISAB_GRAMS=87.48` (20 mithqal),
`DEFAULT_SILVER_NISAB_GRAMS=612.36` (200 dirham), `DEFAULT_ZAKAT_RATE=
0.025` (Sunni 1/40), `LUNAR_YEAR_DAYS=354`. `NisabBasis` enum
(GOLD / SILVER) pinned string values; **default is SILVER** (more
conservative — at modern prices silver-nisab < gold-nisab so silver
captures more zakat-eligible holders, which most contemporary
scholars recommend). `ZakatPolicy` validates rate ∈ (0, 1] (allows
Khums 0.20); positive nisabs; positive lunar_year_days. `FxRates`
carries base_currency + per-currency multipliers + gold/silver per-
gram prices in base; `to_base(amount, currency)` returns base for
base unchanged; raises KeyError on unknown currency. Validation
rejects empty currency codes + zero/negative rates + zero metal
prices. `ZakatInputs` carries multi-currency cash + investments +
gold_grams + silver_grams + debts_owed_to_user (added) + debts_owed_
by_user (subtracted) + reporting_currency + optional hawl_start_date;
mappings default to empty via `field(default_factory=dict)`. All
amounts must be non-negative; empty currency codes rejected per
mapping. `ZakatCalculation` carries net_assets (can be negative if
debts > assets) + nisab_value + meets_nisab + zakat_owed +
basis_used + reporting_currency + hawl_due_date; **two cross-check
invariants** pinned: meets_nisab=True with zakat_owed=0 (and
net>0) is rejected; meets_nisab=False with zakat_owed>0 is
rejected. `calculate_zakat(inputs, fx, *, policy, basis)` checks
inputs.reporting_currency matches fx.base_currency, sums all
sources in base currency, computes nisab via fx-priced metal,
returns zakat_owed = net * rate when meets_nisab else 0. The
debts-owed subtraction can drop net below nisab → no zakat;
**negative net is allowed** (zakat owed clamped at 0, but
net_assets reflects the true negative value for operator
information). `days_until_hawl(calc, *, today)` returns int or
None. Render with 💰/✅ emoji + summary; **no-secret-leak pin**
(no per-currency / per-account breakdown in output). Tests in
`tests/test_zakat.py` (64 cases): module-constant pins; NisabBasis
string-value pin; ZakatPolicy validation (zero rate rejected, rate
above 1 rejected, **rate=1.0 allowed pin**, zero nisabs rejected,
zero lunar_year_days rejected, immutability); FxRates validation
(empty base, zero metal prices, zero/negative rates, empty currency
code all rejected); to_base for base + non-base + unknown KeyError;
ZakatInputs validation (empty reporting, negative gold, negative
silver, negative cash amounts, empty currency code in cash, mirrors
for investments + debts; immutability); ZakatCalculation validation
(negative zakat rejected, zero nisab rejected, **inconsistent
meets_nisab+zakat_owed both directions rejected pin**); zero-assets
below-nisab; **simple cash above silver-nisab pin** ($10000 → $250
zakat); above gold-nisab same; below silver-nisab; **default basis
is SILVER pin**; **silver-more-conservative-than-gold pin** ($5000
meets silver but not gold); multi-currency summing (USD + SAR + EUR
correctly converted); investments added; gold priced; silver priced;
loans-to-user added; debts-owed subtracted; **debts-can-drop-below-
nisab pin**; **debts-can-make-net-negative pin** (zakat=0 but net=
-4000); **reporting-currency must match fx-base pin**; unknown
currency KeyError; **hawl due date 354 days after start pin** (Jan 1
→ Dec 21); none without start; custom lunar_year_days (365);
**custom Khums rate (0.20) pin**; days_until_hawl positive +
negative + None; render meets-nisab shows OWED + amount + currency;
below-nisab shows BELOW NISAB; due_date in render when set;
**render no-secret-leak pin** (no per-currency / SAR / EUR / large
balance / account_id substrings); e2e flows (Saudi-diaspora user
with USD+SAR cash + investments + small gold + credit-card debt →
correct net + meets nisab + due date 354d later; replay consistency;
**gold-basis-lets-smaller-holder-skip pin** $5000 → silver $125 vs
gold $0).

### 1.K — Charitable disbursement signed receipts ✅ landed (`halal/charity_receipts.py`)

Operators upload signed receipts when paying out purified-dividend
charity; the receipts are verified via PDF signature + put on a
Merkle log. Auditor view shows the chain.

### 1.L — Halal short alternatives (Salam / Istisna construct) ✅ landed (`halal/salam_istisna.py`)

Traditional shorting is impermissible. Implement Salam-style
forward-sale construct and Istisna for synthetic exposure to
declining views — pre-paid, deferred-delivery contracts a halal
broker can clear.

---

## Wave 2 — Multi-jurisdiction halal compliance (4-5 weeks)

Different regulators around the Muslim world have different
disclosure + screening requirements.

### 2.A — Saudi CMA Shariah-Advisory Council mapping ✅ landed (substantively via `halal/regulator_index.py`)

Saudi-listed names go through the CMA Shariah Advisory Council's
periodic verdict cycle. The TADAWUL + CMA_HALAL sources in
`halal/regulator_index.py` already encode the matching layer with
authority + staleness rules. Periodic verdict-cycle subscription
remains a follow-on integration; the in-process screening is
ready.

### 2.B — UAE SCA (Securities and Commodities Authority) integration

UAE's SCA has its own list. Mirror the SCA-approved list and flag
when global vs SCA verdict diverges.

### 2.C — Securities Commission Malaysia (SC) Shariah list

SC Malaysia maintains a quarterly Shariah-compliant securities
list. Subscribe + cache + diff per quarter. Diffs trigger
operator alerts for newly compliant + newly non-compliant.

### 2.D — UK FCA + Bank of England halal compatibility

UK has no dedicated halal regulator but multiple FCA-regulated
Islamic banks publish their own screens. Aggregate + reconcile.

### 2.E — Indonesia ISSI (Indonesia Sharia Stock Index) integration

ISSI is the canonical Indonesian halal index. Cache the
constituents + diff per period.

### 2.F — Bahrain CBB Sharia Compliance Reports

CBB's banking license framework references Shariah compliance for
Islamic banks; add their disclosure templates as an export option.

### 2.G — Per-jurisdiction routing engine ✅ landed (`halal/jurisdiction_router.py`)

Given the operator's licensed jurisdiction, the routing engine
selects which exchange + which screener verdict applies. A US
operator trading Saudi names uses both screens conjunctively
(both must pass).

### 2.H — Cross-border tax + Zakat interaction

Saudi has no capital gains tax; UAE has none on equities; UK has
CGT; Malaysia has none on listed equities; US has CGT. The cross-
border tracker reconciles + emits per-jurisdiction tax + Zakat
reports.

---

## Wave 3 — Sukuk + Islamic fixed income (5-7 weeks)

Sukuk is the largest halal-investment asset class globally; the
platform must be first-class here.

### 3.A — Sukuk universe ingestion (IsDB, GCC, IIFM databases)

`crypto/sukuk_universe.py` ingests from the Islamic Development
Bank's IIFM Sukuk Database, S&P Sukuk Index, FTSE Sukuk Index.
Each sukuk gets a compliance fingerprint.

### 3.B — Sukuk pricing model (yield-curve-aware) ✅ landed (`markets/sukuk_pricing.py`)

Sukuk pricing is conventional-bond-like but with equity flavor;
implement Vasicek + CIR yield curve models tuned for sovereign
sukuk vs corporate sukuk.

### 3.C — Sukuk allocation engine

Given a target duration + sector + jurisdiction mix, pick the
optimal sukuk basket (conventional Markowitz + halal-only
constraints).

### 3.D — Sukuk laddering + roll strategy ✅ landed (`markets/sukuk_ladder.py`)

Implement the "rolling sukuk ladder" — buy 1y/3y/5y/10y rungs;
roll each at maturity. Standard LIM/IM/LTM income strategy.

### 3.E — Sukuk credit rating integration (S&P Islamic, Fitch)

Subscribe to credit ratings; map S&P / Fitch / Moody's
conventional ratings to their Islamic-specific overlays.

### 3.F — Sukuk default + recovery model

Default modeling for sukuk uses different recovery curves than
conventional bonds (sukuk holders are equity-like in some
structures). Implement the Asia/GCC-specific recovery model.

### 3.G — Hybrid equity-sukuk portfolio optimizer

Most halal portfolios mix equity + sukuk. Implement the unified
optimizer respecting both asset class constraints + halal
sector caps + duration risk.

---

## Wave 4 — Halal options + structured products (5-7 weeks)

Most options are non-compliant; some structures (Wa'd, Arboun) are
permissible per AAOIFI Standard 38.

### 4.A — Wa'd (unilateral promise) structuring engine ✅ landed (`halal/waad.py`)

Implement the Wa'd contract: a unilateral promise to buy/sell at
a future date + price. Permissible if structured correctly
(no symmetric obligation = no riba). Engine generates the
contract terms + scholar-approved language.

### 4.B — Arboun (down-payment) call structures ✅ landed (`halal/arboun.py`)

Arboun is the halal analog to a call option: pay an Arboun
(down-payment); if not exercised, the seller keeps it as
compensation. Implement structuring + pricing.

### 4.C — Salam forwards for hedging exposure

Salam = pre-paid forward. Used for hedging future price exposure
on equities + commodities. Implement Salam contract generation
+ counterparty matching.

### 4.D — Halal structured note framework

For more complex hedges, structured notes built from Wa'd + Salam
+ Sukuk. Implement the pricing + scholar verdict template.

### 4.E — Volatility-targeting halal portfolio

Without conventional options, volatility targeting uses position
sizing + Salam hedges. Implement the dynamic vol-targeted
allocator.

### 4.F — Tail risk hedging via halal puts

Implement the halal-puts via Wa'd construct: a contractor agrees
to buy your shares at a strike if certain conditions hold. Used
for downside protection without conventional options.

### 4.G — Operator-friendly Wa'd contract templates

Generate scholar-approved Wa'd contract PDF + DocuSign integration
for the actual legal contract execution.

---

## Wave 5 — Halal commodities + precious metals (4-6 weeks)

Commodities have specific halal requirements (constructive possession,
spot delivery for certain metals).

### 5.A — Gold + silver spot trading via halal vault custodians ✅ landed (`halal/vault_custodian.py`)

Integration with vault custodians (BullionVault, GoldMoney's
Sharia-compliant offering). Constructive possession is verified
via vault receipt.

### 5.B — Halal commodities screen (palladium, platinum, copper) ✅ landed (substantively via Round-4 `halal/commodities_screener.py`)

Beyond gold/silver, base metals + agricultural commodities.
Each has its own halal verdict (palladium: permissible; copper:
permissible if not used in alcohol/pork industries).

### 5.C — Salam contracts for agricultural commodities

Wheat, rice, sugar, coffee — Salam contracts are the halal way
to buy forward. Integration with halal-compliant agricultural
trading desks.

### 5.D — Energy (oil/gas) compliance verification

Most oil-major equities are halal-compliant; pure-play E&P names
need careful screening because of debt-to-equity ratios. Custom
screening rules for the energy sector.

### 5.E — Carbon credits + halal climate finance ✅ landed (`halal/climate_finance.py`)

Carbon credits are increasingly relevant; AAOIFI hasn't formally
opined yet. Internal scholar review + flag-as-experimental.

---

## Wave 6 — Halal private equity / startup investing (5-7 weeks)

Private market is huge for Muslim investors; integrate where possible.

### 6.A — Halal startup database (Mudarabah-friendly)

`halal/startup_db.py`: ingests AngelList + Crunchbase + halal-
specific platforms (Wahed, Aghaz). Auto-flags non-compliant
sectors (alcohol, gambling, conventional banking) before the
deal even surfaces.

### 6.B — Mudarabah term-sheet generator

Most VC term sheets are riba-laden (preference shares with
liquidation preferences = guaranteed return). Implement the
Mudarabah-style term sheet generator (profit-sharing, no
guaranteed return).

### 6.C — Musharakah co-investment rails

Musharakah = joint venture. For larger deals, the platform
orchestrates Musharakah co-investment among multiple users.

### 6.D — Convertible Musharakah notes

Halal alternative to convertible notes: a Musharakah equity
position that converts on milestone events.

### 6.E — Halal SPAC structures

Some SPACs are halal-compliant; structuring is delicate. Pre-
screen + flag ones that pass.

### 6.F — Secondary market for halal startup positions

When a user wants exit liquidity, route to a halal-compliant
secondary market. Integrate with EquityZen / Forge / halal-
specific platforms.

### 6.G — Halal LP / GP fund structures

For users running their own funds, Mudarabah + Musharakah-based
LP/GP structures with proper halal performance fees (no
"hurdle"-style guarantees).

---

## Wave 7 — Mudarabah / Musharakah portfolio constructs (4-5 weeks)

Move beyond conventional portfolio theory to halal-native constructs.

### 7.A — Mudarabah-style account types ✅ landed (`halal/mudarabah.py`)

User opens "Mudarabah account" — gives capital to a profit-share
manager. Implement the legal structure + UI + ops surface.

### 7.B — Musharakah pool accounts ✅ landed (`halal/musharakah.py`)

Multiple users co-own one portfolio with proportional returns.
Implement the cap table + distribution math.

### 7.C — Wakalah (agency) accounts with fixed Wakil fee ✅ landed (`halal/wakalah.py`)

Halal alternative to advisory accounts. User appoints the
platform as Wakil (agent); platform charges a fixed fee (not
performance fee).

### 7.D — Profit-loss-sharing equity strategy class

Implement a new strategy class where the user shares returns
above a hurdle rate; below the hurdle, no fee. Halal-compliant
performance fee structure.

### 7.E — Halal-native portfolio optimizer

Modify Markowitz mean-variance to incorporate Mudarabah profit-
sharing constraints + sukuk integration + sector caps.

### 7.F — Halal alpha/beta separation

Conventional alpha-beta separation uses leverage; halal version
uses Wa'd-based constructs.

---

## Wave 8 — Multi-agent LLM trading committee (5-7 weeks)

Beyond single-LLM ensemble, run a structured debate among specialist agents.

### 8.A — Bull/Bear/Quant/Halal-judge multi-agent committee ✅ landed (`core/llm_committee.py`)

Four specialist LLM agents debate each trade idea. Bull argues
for; Bear argues against; Quant runs the numbers; Halal-judge
checks compliance. Final verdict requires consensus or
escalates to operator.

### 8.B — Adversarial red-team agent ✅ landed (`core/red_team.py`)

Dedicated agent whose job is to find flaws in the committee's
reasoning. Forces the committee to be defensible.

### 8.C — Memory-of-decisions for committee

The committee learns from past trades; if the same setup keeps
losing, the committee remembers + adjusts. Persistent vector
memory tied to setup fingerprint.

### 8.D — Operator-tunable committee configuration

Operator can swap models per role (e.g., Sonnet for Bull, Opus
for Halal-judge), set debate rounds, set unanimity thresholds.

### 8.E — Committee transcript audit log

Every committee debate is logged + searchable. Operators can
review historical debates to understand why a trade was made.

### 8.F — Cross-agent contradiction detector

If the Bull's reasoning contradicts the Quant's numbers, surface
to operator before execution.

### 8.G — Committee meta-learner

Track committee accuracy over time per setup type; surface
which agent voices are reliable when.

---

## Wave 9 — Reinforcement learning with halal constraints (6-8 weeks)

Move from supervised + LLM to RL where the agent learns by trading.

### 9.A — Halal-constrained PPO trader

Implement Proximal Policy Optimization (PPO) with hard halal
constraints (cannot enter non-compliant tickers); soft sector
constraints (penalty for over-concentration).

### 9.B — Risk-adjusted reward function with Sharpe + max drawdown

Reward function = annualized Sharpe - 2*MaxDrawdown - 5*HalalViolations.
Pinned tests ensure the constraint is hard.

### 9.C — Curriculum learning across regimes

Start training in calm regime; advance to volatile; advance to
crisis. The agent learns regime-specific policies.

### 9.D — Multi-objective RL (return / drawdown / Zakat)

Some operators prioritize Zakat-friendly turnover (lower
turnover = less purification math). Multi-objective NSGA-II
based agent.

### 9.E — Population-based RL with halal niches

Train multiple agents in parallel with different halal-compliant
niches (large-cap-only, tech-only, EM-only). The portfolio
diversifies across agents.

### 9.F — Offline RL from operator's historical trades

Bootstrap a new agent from the operator's past trade history
using behavior cloning + offline PPO.

### 9.G — Safe RL deployment gate

New RL agent must outperform baseline by 3+ Sharpe in shadow
mode for 90+ days before live promotion. Pinned via the
Round-4 promotion gate.

---

## Wave 10 — Transformer-based regime + multi-modal (5-7 weeks)

Move from rule-based regime + IsolationForest to transformer-based.

### 10.A — Time-series transformer for regime detection

Transformer trained on macro + price + volume + news embedding
sequences. Output: probability distribution over regimes.

### 10.B — Multi-modal transformer (price + text + image)

Inputs: price series + earnings call transcript + chart image
embedding. Output: setup quality score + regime classification.

### 10.C — Foundation model fine-tuning pipeline

Fine-tune a small open-weight LLM (Qwen-7B or LLaMA-7B) on
halal-trading-specific corpus (scholar verdicts, market commentary,
trade ideas). Run inference locally for zero-cost LLM calls.

### 10.D — Audio + video regime indicators

Earnings call sentiment from audio (tone analysis) + CFO body
language from video (where available). Multi-modal regime
input.

### 10.E — Distillation from foundation model to deployable size

Distill the fine-tuned 7B to a 1B-2B model that runs on the bot's
hardware. Maintains 90%+ of quality at 1/4 the cost.

### 10.F — Adversarial robustness of regime detector

The regime detector should be robust to data poisoning + flash
crashes. Implement adversarial training + input sanitization.

---

## Wave 11 — Alternative data fusion (6-8 weeks)

Best-in-class quant uses alt data; halal traders deserve it too.

### 11.A — Reddit + StockTwits sentiment via dedicated NLP ✅ landed (`sentiment/social_nlp.py`)

Custom model for retail sentiment, halal-aware (filters out
non-compliant ticker chatter).

### 11.B — Google Trends signal extraction ✅ landed (`sentiment/trends_signal.py`)

Search-volume divergence from price often precedes moves; feature
engineering pipeline.

### 11.C — Satellite imagery for retail / energy / agriculture

Subscribe to commercial sat-imagery providers (Planet Labs,
Maxar). Foot-traffic at retailers, oil-tank levels, crop health.

### 11.D — Earnings call NLP (transcript + tone)

Custom model on every transcript. Surface CFO uncertainty
markers, segment-level disclosure changes.

### 11.E — SEC 8-K / 10-K diff engine ✅ landed (`sentiment/sec_diff.py`)

Quarter-over-quarter diff of filings; flag material changes
(disclosure language, risk factors, accounting policies).

### 11.F — Insider trading pattern detector ✅ landed (`sentiment/insider_pattern.py`)

Cluster insider buy/sell against historical patterns; surface
unusual aggregate signals.

### 11.G — Halal-aware news aggregator ✅ landed (`sentiment/halal_news.py`)

Filter news for halal-compliant signal density (e.g., a
non-compliant tobacco partnership announcement is downgrade-worthy
for adjacent suppliers).

### 11.H — ESG + halal alignment scorer ✅ landed (`halal/esg_alignment.py`)

ESG and halal often align (no tobacco/alcohol/gambling); ESG
data is already widely available. Bridge it into the screener.

### 11.I — Macro data flow from FRED + IMF + World Bank

Beyond manual entry: ingest 1000+ macro time series; feature-
engineering pipeline picks the most predictive ~50 per regime.

### 11.J — Crypto-on-chain → equity correlation engine

Coinbase volume / Tether mint patterns correlate with US equity
flows; expose the cross-signal via the unified feature store.

---

## Wave 12 — Smart order routing + execution algos (5-7 weeks)

Move from single-shot market orders to institutional execution.

### 12.A — TWAP execution algorithm ✅ landed (`trading/twap.py`)

Time-Weighted Average Price slicer. Order size > threshold gets
sliced over operator-specified window.

### 12.B — VWAP execution algorithm ✅ landed (`trading/vwap.py`)

Volume-Weighted Average Price slicer using historical volume
profile.

### 12.C — POV (Percentage of Volume) algorithm ✅ landed (`trading/pov.py`)

Adaptive: stay at X% of real-time market volume. Catches up
during high-volume periods.

### 12.D — Iceberg orders for large positions ✅ landed (`trading/iceberg.py`)

Show only a small portion of the order to the market; refill
as it fills.

### 12.E — Smart order router (multi-venue) ✅ landed (`trading/smart_router.py`)

Route to the venue with best price + lowest cost + halal-compliant
custody.

### 12.F — Dark pool detection + avoidance for halal-sensitive trades

Some dark pools may host non-halal counterparties; opt-out
flag for sensitive trades.

### 12.G — Anti-frontrunning protection ✅ landed (`trading/anti_frontrun.py`)

Order obfuscation (size randomization, time jitter) to prevent
frontrunning.

### 12.H — Post-trade analytics (slippage, market impact, cost) ✅ landed (`trading/post_trade.py`)

For every executed trade, decompose cost into spread + impact
+ delay. Operator dashboard surfaces per-strategy execution
quality.

### 12.I — Execution quality A/B testing

A/B test different execution algos on similar setups; auto-
select the best per regime.

### 12.J — Queue-position model for limit orders ✅ landed (`trading/queue_position.py`)

Model expected fill probability as a function of queue depth +
price level + time of day. Used to choose passive vs aggressive
orders.

---

## Wave 13 — Tail risk hedging halal-compliant (4-6 weeks)

Beyond simple stop-losses, structured tail-risk protection.

### 13.A — Halal volatility regime detector ✅ landed (`ml/vol_regime.py`)

Detect volatility regime changes (calm → volatile → crisis) for
hedging decisions.

### 13.B — Wa'd-based portfolio insurance ✅ landed (`halal/portfolio_insurance.py`)

Construct portfolio insurance via Wa'd contracts (halal puts).
Operator can opt-in for X% drawdown protection at Y% premium.

### 13.C — Inverse-asset hedge basket (gold, sukuk) ✅ landed (`halal/hedge_basket.py`)

In market stress, halal-compliant safe havens are gold + AAA
sukuk. Auto-rotate into the basket when regime detector flags
crisis.

### 13.D — Sukuk-based duration hedge ✅ landed (`halal/duration_hedge.py`)

Long-duration sukuk hedges interest-rate risk for equity
positions. Embed in portfolio optimizer.

### 13.E — Currency hedging via halal forwards

For users holding multi-currency portfolios (USD + SAR + AED),
halal currency forwards via Salam construct.

### 13.F — Liquidity risk modeling ✅ landed (`ml/liquidity_risk.py`)

Liquidity drops in stress; model expected slippage in each
holding under crisis. Liquidity-adjusted position sizing.

---

## Wave 14 — Bayesian risk modeling + Monte Carlo (4-5 weeks)

Move beyond point estimates to full distributions.

### 14.A — Bayesian VaR (with skew + kurtosis) ✅ landed (`ml/bayesian_var.py`)

Replace gaussian VaR with Cornish-Fisher / Bayesian VaR
incorporating non-normality.

### 14.B — Monte Carlo portfolio simulation ✅ landed (`ml/monte_carlo.py`)

10000+ simulations under various regime scenarios; return
distribution of outcomes for the operator.

### 14.C — Stress-test scenario library ✅ landed (`ml/stress_scenarios.py`)

Pre-canned scenarios: '08 financial crisis, '20 COVID, '15 China
devaluation, '22 rate-shock. Replay through the portfolio.

### 14.D — Correlation regime modeling ✅ landed (`ml/correlation_regime.py`)

Correlations spike in crisis; model the regime-conditional
correlation matrix.

### 14.E — Coherent risk measures (Expected Shortfall, CVaR) ✅ landed (`ml/coherent_risk.py`)

VaR has known limitations; ES/CVaR are coherent. Switch the
risk dashboard to ES.

### 14.F — Tail dependence modeling (copulas) ✅ landed (`ml/tail_dependence.py`)

Stocks crash together; model the tail-dependence structure
explicitly via Clayton / Gumbel copulas.

### 14.G — Confidence-bounded backtest reporting ✅ landed (`ml/backtest_confidence.py`)

Every backtest reports Sharpe ± confidence interval, not just
point Sharpe. Bootstrap-based.

---

## Wave 15 — Mobile + voice + AR/VR interfaces (6-8 weeks)

Multi-modal access for global users.

### 15.A — Native iOS app (SwiftUI)

Full-feature iOS app with portfolio, scholar Q&A, trade
approval, alerts.

### 15.B — Native Android app (Jetpack Compose)

Same feature set, Android.

### 15.C — Voice control via Siri / Google Assistant

"Show my halal portfolio" / "What's the Zakat owed?" / "Approve
the BTC trade." Voice intent already implemented; wire to
mobile.

### 15.D — Apple Watch / WearOS notifications

Trade alerts + kill-switch on the wrist. Tap-to-approve.

### 15.E — AR portfolio viewer (Vision Pro)

Visualize portfolio in 3D space — sectors as floating clusters,
size by holding, color by halal compliance.

### 15.F — Telegram + WhatsApp bot integration

Many Muslim users prefer messaging apps. Full bot interface.

### 15.G — Discord community + bot

Power-user community via Discord with the same bot interface.

### 15.H — Push notifications with halal context

"Your AAPL position generated $12.30 in dividends; $0.84 needs
purification; send to charity?"

---

## Wave 16 — Scholar AI assistant + halal Q&A (5-7 weeks)

A halal-trader's first questions are "is this allowed?" — answer them.

### 16.A — Multi-scholar fine-tuned LLM

Fine-tune on AAOIFI standards, scholar Fatawa, classical fiqh
texts. Always cites the source.

### 16.B — Citation-grounded answers

Every answer surface includes "this is based on Standard X,
Clause Y" with link to the standard text.

### 16.C — Disagreement-aware answers

When schools disagree, the answer surfaces both positions:
"Hanafi: yes; Shafi'i: no; the platform's default is X."

### 16.D — Scholar-of-record for ambiguous cases

Ambiguous cases (new fintech instruments) escalate to the
platform's scholar-of-record (an actual mufti) for verdict.

### 16.E — Personal scholar selector

User picks their scholar of choice; LLM weighs that scholar's
verdict highly when disagreements exist.

### 16.F — Q&A archive + community Q&A

User questions + scholar answers are archived (anonymized);
new users get prior-art answers fast.

### 16.G — Multilingual support (Arabic, Urdu, Malay, Indonesian, Turkish)

Top Muslim languages first-class. Tafsir + Hadith citation in
original Arabic + translation.

---

## Wave 17 — Community + social + leaderboards (4-6 weeks)

Build the network so users learn from each other.

### 17.A — Public anonymized performance leaderboard ✅ landed (`web/public_leaderboard.py`)

Users opt-in to publish anonymized portfolio performance. Top
performers get badges.

### 17.B — Halal trade idea marketplace

Users publish trade ideas (with rationale + risk); other users
follow + auto-trade. Idea-author gets a Wakalah fee.

### 17.C — Strategy gallery (paid + free)

Users publish full strategies; others subscribe. Marketplace
takes a halal Wakalah fee.

### 17.D — Daily halal market commentary feed

LLM-generated daily commentary on halal-market movers, sector
themes, sukuk yield curve, gold price. Operator + community
sourced.

### 17.E — Friends + group portfolios (Musharakah pools)

Groups of users co-own portfolios via Musharakah. Cap-table
view + per-member returns.

### 17.F — Live market chat (halal-friendly)

Chat rooms tied to specific halal-compliant tickers. Moderated
to keep gambling-style talk out.

### 17.G — Event-driven group meetings (Quarterly Earnings Watch Party)

For top halal-tickers, scheduled live-watch parties with LLM
commentary + community chat.

### 17.H — Trader-to-trader Mudarabah

User A has capital + no time; user B has skill + no capital.
Match them via Mudarabah; platform escrows + tracks.

---

## Wave 18 — Multi-jurisdiction tax + Zakat automation (4-6 weeks)

Tax + Zakat are huge friction points; eliminate them.

### 18.A — Per-trade tax-lot tracking with HIFO/FIFO/LIFO ✅ landed (`core/tax_lots.py`)

Operator picks lot-selection method; per-trade attribution +
year-end report.

### 18.B — Automatic 1099-B / Form 8949 generation (US) ✅ landed (`core/tax_us_8949.py`)

US tax forms auto-generated + ready for TurboTax / e-file.

### 18.C — UK CGT + dividend tax form generation ✅ landed (`core/tax_uk_cgt.py`)

UK self-assessment SA-tax form export.

### 18.D — Saudi / UAE Zakat-only reports ✅ landed (`core/tax_gcc_zakat.py`)

Saudi/UAE have no income tax on equities; only Zakat.
Auto-generated Zakat report aligned with the operator's chosen
scholar's methodology.

### 18.E — Malaysia tax exemption for listed equities ✅ landed (`core/tax_my.py`)

Malaysia exempts capital gains; the report just covers Zakat.

### 18.F — Indonesia capital gains + dividend tax

Indonesia: 0.1% transaction tax + 10% dividend tax. Auto.

### 18.G — Tax-loss harvesting (where applicable) ✅ landed (`core/tax_loss_harvest.py`)

In jurisdictions with CGT, the harvester sells losers to offset
gains, respecting wash-sale rules + halal constraints.

### 18.H — Multi-currency accounting ✅ landed (`core/multi_currency.py`)

Operator's base currency tracked + reported. Per-trade FX
translation.

### 18.I — DocuSign-ready Zakat receipt for charity ✅ landed (`halal/zakat_receipt.py`)

When user pays Zakat, generates a signed PDF receipt for
record-keeping.

### 18.J — Year-end tax/Zakat checklist generator ✅ landed (`core/yearend_checklist.py`)

"Here's everything you need to file your taxes this year" —
checklist with done/pending status.

---

## Wave 19 — Regulator audit packages + SOC2 Type II (5-7 weeks)

Audit-ready output for any regulator + SOC2 certification.

### 19.A — SEC + FINRA audit package generator ✅ landed (`ops/audit_sec_finra.py`)

For US-licensed users, generates SEC + FINRA audit-ready
package: trade blotter, allocations, suitability assessments.

### 19.B — FCA audit package (UK)

FCA-required disclosures: SUP 16 reports, MiFID-II RTS-27 best-
execution reports.

### 19.C — CMA audit package (Saudi Arabia)

CMA disclosure-of-positions + Shariah-compliance attestation.

### 19.D — SC Malaysia audit package

SC's 8-K disclosure equivalents for licensed advisers.

### 19.E — SOC2 Type II readiness package

Continuous evidence collection: access logs, change management,
incident response. Auto-generated for the SOC2 auditor.

### 19.F — ISO 27001 information security alignment

Map controls + collect evidence.

### 19.G — GDPR + CCPA + India DPDP Act compliance

Per-user data export + erasure flows.

### 19.H — Merkle-tree-anchored audit log ✅ landed (`core/merkle_audit.py`)

Every audit-relevant event hashed into a Merkle tree; root
periodically committed to a public timestamp service. Tamper-
evident.

### 19.I — Scholar e-signature on compliance attestations

Quarterly compliance attestation signed by the platform's
scholar of record via DocuSign.

### 19.J — Penetration test reporting + remediation tracker

Quarterly pentest by a third party; findings tracked + closed.

---

## Wave 20 — Education + certification + academy (4-6 weeks)

Help users grow into better halal traders.

### 20.A — Halal trading academy (interactive courses)

Beginner → Intermediate → Advanced courses on halal markets,
screening, Zakat, derivatives alternatives.

### 20.B — Certification exams + certificates

Pass the Halal-Trader-1 exam → certificate. Higher tiers for
advanced topics.

### 20.C — Live webinars with scholars

Monthly live Q&A with the platform's scholars.

### 20.D — Trade journal template + coaching

Each trade gets a journal; LLM coach reviews + suggests
improvements.

### 20.E — Halal economics primer (Mudarabah / Musharakah / Sukuk)

Built-in primer accessible from any contextual surface.

### 20.F — Practice mode (paper trading with full feature set)

All features available in paper mode; only the broker
connection is paper.

### 20.G — Risk-tolerance self-assessment

Quiz at signup; portfolio + strategy recommendations tuned.

### 20.H — Halal investing podcast / YouTube content

Built-in feed; weekly content from the platform's research team.

### 20.I — Open-curriculum CFA-equivalent for halal markets

Long-term: structured 3-level certification equivalent to CFA
but halal-focused.

---

## Wave 21 — Marketplace for strategies + signals (5-7 weeks)

Power users monetize their edge; new users access best-in-class.

### 21.A — Strategy publishing flow

Verified backtest + 90-day live-paper attribution before
publishing.

### 21.B — Signal subscription with Wakalah fee structure

Subscribers pay; signal authors receive Wakalah fee (halal
service fee, not performance fee).

### 21.C — Strategy peer review

Other publishers can review + critique a strategy's halal
compliance + statistical validity.

### 21.D — Halal-only ETF basket builder

User-defined basket → tradable as a single position. Auto-
rebalance with halal screen continuous.

### 21.E — Robo-advisor tier (white-label)

Halal robo-advisor product white-labelable for Islamic banks /
wealth advisors.

### 21.F — Institutional API tier

Full FIX gateway for institutional users wanting halal
compliance overlay.

### 21.G — Halal signal token (cryptographic provenance)

Each signal cryptographically signed by the author; subscribers
verify origin.

### 21.H — Refund/dispute resolution for paid signals

If a signal materially misrepresents itself, dispute → review →
refund. Halal service quality.

---

## Wave 22 — Decentralized Islamic finance bridge (cautious, 6-8 weeks)

DeFi has potential; most current DeFi is non-compliant. Bridge with care.

### 22.A — Halal stablecoin gateway ✅ landed (`halal/stablecoin_gateway.py`)

Bridge to fully-collateralized halal stablecoins (Wahed-backed,
gold-backed). NOT algorithmic stables.

### 22.B — Wakalah-based DeFi vaults

User deposits stablecoin → halal vault → manager allocates →
profit share. Pure Wakalah, no riba mechanics.

### 22.C — Mudarabah pool DeFi smart contracts

On-chain Mudarabah agreements with profit-sharing automated.

### 22.D — Sukuk-on-chain integration ✅ landed (`halal/sukuk_onchain.py`)

Tokenized sukuk; integrate with HSBC's on-chain sukuk pilots.

### 22.E — Halal NFT screening (provenance + non-haram subject) ✅ landed (`halal/nft_screen.py`)

Some NFTs are halal (utility, halal-themed art); most aren't.
Screen + flag.

### 22.F — Audit-trail-as-NFT for premium users

User's full audit trail anchored on-chain (Polygon/Arbitrum) for
permanent, tamper-proof record.

### 22.G — Cross-chain bridging restricted to halal-compliant chains ✅ landed (`halal/bridge_screen.py`)

Some chains have heavy gambling DeFi; restrict bridging to
chains with verified halal alternatives.

---

## Wave 23 — Geographic expansion + i18n depth (4-6 weeks)

Translate not just words but cultural + regulatory context.

### 23.A — Arabic (MSA + dialects)

Full UI translation; right-to-left layout; Arabic chart labels.

### 23.B — Urdu

Pakistan + India + diaspora.

### 23.C — Malay + Indonesian

Indonesia is the largest Muslim country; first-class support.

### 23.D — Turkish

Turkey is large and highly active in Islamic finance.

### 23.E — Hausa + Swahili

Nigeria + East Africa.

### 23.F — Bengali

Bangladesh + India + diaspora.

### 23.G — Persian + Pashto + Dari

Iran + Afghanistan diaspora.

### 23.H — French (West Africa, Maghreb)

Morocco + Algeria + Tunisia + Senegal + Mali.

### 23.I — Spanish (LatAm Muslim communities)

Smaller but underserved.

### 23.J — Localized scholar networks per region

Indonesian users get Indonesian scholars; Saudi users get Saudi
scholars; etc.

---

## Wave 24 — Long-term: AI Mufti + autonomous compliance (10+ weeks)

Years out, but worth pinning the vision.

### 24.A — Foundation LLM trained from scratch on Islamic finance corpus

Not just fine-tuning: full pretraining on AAOIFI + IFSB +
classical fiqh + market data. Open-weight, scholar-overseen.

### 24.B — Autonomous compliance attestation

Quarterly + per-trade auto-generated compliance attestation,
co-signed by AI Mufti + human scholar.

### 24.C — Self-correcting screener via scholar feedback loops

When the screener's verdict differs from an operator-flagged
scholar verdict, the screener updates its rules + retrains.

### 24.D — Real-time fatwa generation

Brand-new fintech instrument? AI Mufti generates a tentative
verdict in seconds, escalates to human for sign-off in 24h.

### 24.E — Cross-school harmonization assistant

LLM that synthesizes the four schools' positions into a single
coherent verdict where possible; preserves ambiguity where not.

### 24.F — Permanent education / certification credentialing

Users earn permanent credentials (chartered halal trader)
recognized by major Islamic finance organizations.

### 24.G — Open the platform's scholar review data as a public good

Anonymized aggregate scholar verdict data published as a public
research dataset for Islamic-finance academia.

### 24.H — Industry standard contribution

Platform contributes to AAOIFI / IFSB standards revision via
the data + scholar network it has aggregated.

---

## Wave 25 — Longevity + sustainability (ongoing)

Make the platform last decades.

### 25.A — Self-sustaining operator economy (Wakalah fees)

Platform revenue from Wakalah fees only — never speculation,
never spread, never proprietary trading.

### 25.B — Halal endowment (Waqf) for platform sustainability

Set up a Waqf that owns a portion of platform revenue + funds
scholar education + halal research grants.

### 25.C — Open-source community governance

Critical halal-compliance code base open-source; community +
scholars review changes.

### 25.D — Apprenticeship program for next-gen halal quants

Identify + sponsor 100+ junior halal-quant developers per year;
internships at the platform.

### 25.E — Annual halal markets research conference

Sponsor + host the annual conference; bring scholars + quants +
regulators together.

### 25.F — Charity + Zakat platform integration

Native Zakat + Sadaqah payment to verified charities (Islamic
Relief, LaunchGood, Penny Appeal).

### 25.G — Climate / sustainability halal alignment

Partner with halal-aligned ESG initiatives; surface dual-screen
(halal + climate-friendly) options.

### 25.H — Knowledge sharing with non-Muslim ethical-investing community

Many of halal's principles align with broader ethical investing
(no tobacco, no gambling, no excessive debt). Bridge knowledge.

---

## Round-5 Acceptance

Round 5 is "complete" when:

1. ✅ Every wave has at least one shipped section per the format
   used in Round 4 (`### X.Y — title ✅ landed (`module/path.py`)`).
2. ✅ Aggregate test count > 4000 unit + integration tests, all green.
3. ✅ Every operator-facing surface (mobile, web, voice, telegram)
   has a tutorial in the academy.
4. ✅ A live Saudi-licensed broker is wired (real-money via halal-
   licensed broker).
5. ✅ A live UK FCA-licensed broker is wired.
6. ✅ A scholar of record has signed the platform's quarterly
   attestation for 4 consecutive quarters.
7. ✅ External SOC2 Type II audit passed.
8. ✅ Penetration tests pass for 4 consecutive quarters.
9. ✅ 1000+ live users; 100+ public strategies; 50+ scholar
   verdicts in the AI Mufti corpus.
10. ✅ AAOIFI invites the platform to contribute to the next
    Standards revision.

## Legend
✅ landed · 🟡 partial · 🔴 blocked (external dep / decision needed)

## Ordering note for autonomous /loop iterations

The iteration agent picks the next-best section by: (a) preferring
Wave 0 cleanup before opening new verticals; (b) preferring higher-
ranked waves (Wave 1 → Wave 24) within their category; (c) within a
wave, preferring the lower-letter section (1.A before 1.B); (d)
deferring sections marked 🔴 blocked unless the blocker is
resolvable. Auxiliary primitives that strengthen multiple waves
take precedence within a category.
