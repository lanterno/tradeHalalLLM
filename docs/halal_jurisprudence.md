# Halal Jurisprudence Handbook

This document is the project's rulebook for which financial products
are permissible (halal) under Sharia law and the criteria the
screeners use to decide. The bot's `halal/explainer.py` cites
section numbers from this file when it explains a trade decision —
operators (and a future scholar reviewer) can trace every
compliance ruling back to the section that produced it.

This handbook is not a substitute for a personal scholar. It encodes
**one** widely-followed methodology (AAOIFI default with optional
operator-selectable variants) and surfaces every disagreement as a
configuration choice the operator owns. Borderline cases route to
the exception queue for explicit human acknowledgement before the
bot trades.

> **Reviewing this handbook:** the project commits to a quarterly
> review by an independent scholar. The "Last reviewed" footer at the
> bottom records the most recent review date and the scholar's
> name (when public consent is given).

---

## Section 1: Foundational prohibitions

These are the hard floors under every methodology in this handbook.
A trade that violates any of them cannot be reached by any operator
configuration; the screeners refuse the symbol regardless of
scholar profile (Section 6).

### 1.1 Riba (interest)

Any product that pays or charges interest as a function of the
holding period itself is forbidden. This rules out:

- Conventional bonds and preferred shares paying interest coupons.
- Stablecoins that pay yield from interest-bearing reserves
  (most "savings" tokens that aren't a Mudarabah / Wakalah
  structure).
- Lending-protocol "deposit yield" where the underlying is
  conventional debt.
- Margin / leverage products that charge a periodic interest
  rate to hold the position.

The bot's `crypto/screener.py` rules engine flags interest-bearing
features in a token's economics; the stock screener catches
interest-bearing debt via the AAOIFI 30%-of-market-cap ratio
(Section 2.4).

### 1.2 Maysir (gambling)

Pure speculation with no underlying productive activity is
forbidden. The line between speculation and investment isn't
binary; the bot's screening catches the most-clear cases:

- Leveraged perpetual futures with no underlying delivery
  obligation.
- Prediction markets whose payouts depend on event outcomes
  unrelated to a real-world economic activity.
- Lottery tokens.

The bot **does not** trade leveraged products (the brokers it
talks to are configured paper-only and spot-only). This is the
strongest mechanical guarantee against Maysir.

### 1.3 Gharar (excessive uncertainty)

Contracts whose value depends on future events outside the
underlying — or where the underlying isn't deliverable — fall
under Gharar. Practical implications:

- Naked options and naked swaps are forbidden.
- Tokens whose redemption mechanism is opaque or
  counterparty-dependent (most algorithmic stablecoins after
  the 2022 collapses) are flagged `doubtful`.
- Synthetic assets whose backing is unverifiable.

### 1.4 Prohibited industries

Companies whose primary revenue is from forbidden activities:

- Alcohol production / distribution
- Conventional banking, insurance, and lending (interest-based
  finance)
- Pork production
- Adult entertainment
- Conventional gambling and casinos
- Tobacco
- Conventional weapons (the bot does not exempt
  defense-contractor primes)

The Zoya stock screener's sector-classification rejects these at
the GICS-sector level. The crypto screener's category rules
(`crypto/screener.py`) enforce the same with a token-category
table.

---

## Section 2: AAOIFI financial ratios (default profile)

The Accounting and Auditing Organization for Islamic Financial
Institutions (AAOIFI) publishes the most-cited international
standards for Sharia screening. Standards 21 and 30 set the
financial-ratio bars the bot's default profile uses.

### 2.1 Interest-bearing debt ≤ 30% of market cap

Companies whose interest-bearing debt exceeds 30% of trailing
12-month-average market capitalisation are **not halal**. Tightening
this further is a scholar's prerogative (see the Taqi Usmani profile
in Section 6.2).

The market-cap denominator is deliberately a backward-looking
average rather than spot price — it makes the ratio less
manipulable by a transient price spike on the screening day.

### 2.2 Non-permissible income ≤ 5% of total revenue

A small amount of non-Sharia-compliant revenue (e.g. an interest
sweep on operating cash, a sliver of revenue from a non-core
unrelated subsidiary) is tolerated *if* the operator commits to
purifying the proportionate share of any capital gain (Section 5).

### 2.3 Cash and receivables ≤ 33% of market cap

A company whose balance sheet is dominated by cash + receivables
(rather than productive assets) starts to look like a fund holding
debt rather than a business — outside the spirit of equity
investing. AAOIFI 21 caps the ratio.

The bot treats this check as **informational** by default — the
DeLorenzo profile (Section 6.3) flips it to a hard reject. Most
operators leave it informational because real businesses can
legitimately hold large cash balances during M&A or capital-
return phases.

### 2.4 Implementation in the bot

`halal/scholar_profiles.py:ScreeningThresholds` carries the three
ratios as configurable fields:

```python
ScreeningThresholds(
    debt_to_marketcap_max=0.30,         # Section 2.1
    non_permissible_income_max=0.05,    # Section 2.2
    cash_and_receivables_max=0.33,      # Section 2.3
)
```

`halal/scholar_profiles.py:evaluate_thresholds(...)` applies them
and returns `(passed, violations)`. Missing inputs are *skipped*
(treated as "not measured" rather than "passes") — pin so a
partial filing can't silently approve.

---

## Section 3: Asset-class rulings

### 3.1 Equities

Stock screening follows AAOIFI standards via Zoya:

- Section 1.4 sector exclusions.
- Section 2 financial ratios.

The `Settings.zoya_*` fields configure the API endpoint and
sandbox mode; cached compliance lives in
`db/repos/stock_halal_cache.py`.

### 3.2 Cryptocurrencies

A crypto asset is permissible when **all** of the following hold:

1. **Utility-bearing.** It represents a network utility (gas /
   fee token, governance token, storage credit, compute
   payment). Pure-speculation tokens with no utility (most
   "memecoins") fail.
2. **Non-interest-bearing.** No automatic yield from holding.
   Staking *with locked-up tokens used to validate the chain*
   is permissible (it's compensation for productive
   service); staking *that's pure interest on a deposit* is
   not.
3. **Issuer revenue from halal activities.** A privacy coin
   whose primary use is contraband payments fails; a
   payments coin with mixed legal/illegal use stays
   `doubtful` until the operator makes a call.
4. **Material liquidity.** Market cap above ~$1B is the
   default proxy for "the asset has counterparty solvency on
   the exchange and a real secondary market"; smaller caps
   route to the exception queue.

The bot's `crypto/screener.py` is inspired by **Mufti Faraz Adam's
Crypto Shariah Screening Framework** (4-pillar classification:
category / token type / legitimacy / utility).

### 3.3 Commodities (gold, silver)

Gold and silver are explicitly permissible for halal trade per
classical jurisprudence — the prophet's hadith about fair
weight-for-weight exchange is the foundational ruling.
Modern considerations:

- **Spot-physical-delivery** trades are halal. The bot does
  not currently trade physical commodities.
- **ETFs holding physical gold (GLD / SLV with allocated
  custody)** are permissible.
- **Synthetic / paper gold** (futures with no delivery) is
  Maysir-adjacent and the bot rejects.

Wave 1.G is the upcoming integration; until it lands, gold /
silver are not in the bot's tradable universe.

### 3.4 Sukuk (Islamic bonds)

Sukuk represent fractional ownership of a real asset, with
profit derived from rental / project cash-flow rather than
interest. They are permissible by construction (the structure
is the ruling).

The bot does not currently trade sukuk; Wave 1.H scopes the
integration.

### 3.5 REITs

Real Estate Investment Trusts are permissible if **and only if**:

- The underlying properties are not used for forbidden
  activities (Section 1.4) — no hotels with bars / casinos,
  no conventional bank office towers where the lessee is a
  conventional bank.
- The REIT's debt structure passes Section 2's ratios.

Wave 1.I scopes the integration. Until then, REITs are
treated like any other equity by the Zoya screener.

### 3.6 International equities

The framework extends naturally — local-currency company
disclosures feed the same ratio engine. The bot does not
currently support international equities (London / Tokyo /
DIFC); Wave 1.J scopes Saxo Bank integration.

---

## Section 4: Decision states

The screener emits one of three decisions for any symbol; the
audit row (`HalalScreening`) records which.

| Decision | Meaning | Bot behaviour |
|---|---|---|
| `halal` | Compliant under the active profile | Tradable |
| `doubtful` | Insufficient data, edge case, or borderline | Exception queue (Section 7) |
| `not_halal` | Fails one or more hard rules | Refused; never in the candidate set |

The conservative-default tiebreak is pinned across every consensus
policy in `halal/consensus.py`: when providers disagree,
`not_halal > doubtful > halal`. A single rejection by any provider
overrides any number of `halal` votes (Section 6.4).

---

## Section 5: Purification

Companies with non-zero `non_permissible_income_max` (Section 2.2)
have a sliver of revenue from non-Sharia-compliant sources —
typically interest on operating cash. Capital gains accrued on
those companies must be purified by donating the proportionate
share to charity.

### 5.1 Per-trade purification

When a position closes profitably, `halal/round_trip_purification.py`
computes:

```
purification_due = max(0, capital_gain) × non_permissible_income_pct
```

The result is recorded in `purification_entries` against the
original trade. **Negative gains do not produce a credit** — the
operator never owes themselves charity from a loss.

### 5.2 Periodic disbursement

Wave 2.D `halal/purification_schedule.py` groups outstanding
purification entries into monthly / quarterly / yearly
disbursement bundles (default quarterly). `schedule_disbursements`
returns one `DisbursementReceipt` per period with:

- Total USD owed
- Per-symbol breakdown (sorted by descending USD so the
  operator's eye lands on concentration first)
- Markdown receipt body suitable for emailing the operator
  / charity

The scheduler **never auto-marks entries paid** — that's a
one-way audit-trail commitment that needs explicit operator
acknowledgement after the disbursement actually settles.

### 5.3 Charity choice

The operator picks the disbursement target. The bot doesn't
prefer any particular charity but the recommended pattern is:

- A reputable Islamic charity (Islamic Relief, Penny Appeal,
  Zakat Foundation) with a public Sharia advisory board.
- A direct disbursement (not an investment vehicle) so the
  funds discharge the obligation cleanly.
- Records retained for tax purposes — purification is **not**
  Zakat, but jurisdictions vary on whether it's tax-deductible.

### 5.4 Dividend purification

Dividend purification (separate from capital-gain purification)
is computed per-dividend by `halal/purification.py`:

```
dividend_purification = dividend × non_permissible_revenue_pct
```

The same disbursement scheduler handles both kinds.

---

## Section 6: Scholar profiles (Wave 2.C)

Different scholars hold different positions on edge cases.
`halal/scholar_profiles.py` ships three named profiles. The
operator picks one via configuration; the audit row records
which profile gated each trade so a future scholar challenge has
the chain of accountability.

### 6.1 `aaoifi_default` (default)

The international standard. Debt 30% / non-permissible income
5% / cash-and-receivables 33% (informational). STRICT consensus
across screening providers.

```python
AAOIFI_DEFAULT.thresholds == ScreeningThresholds()
AAOIFI_DEFAULT.default_policy == ConsensusPolicy.STRICT
```

### 6.2 `taqi_usmani`

Stricter than the AAOIFI default. Debt cap 25%, non-permissible
income 3%. Under the WEIGHTED consensus path, weights Musaffa +
IdealRatings 1.5× (operators in this tradition often weight the
more conservative providers heavier on borderline tech-sector
cases).

The rationale: Mufti Taqi Usmani has at times argued the AAOIFI
thresholds were a starting concession for a market lacking
sufficient Sharia-compliant financing, not a target. Profiles
following this view tighten the cuts.

### 6.3 `delorenzo_djim`

Sheikh Yusuf Talal DeLorenzo's DJIM-era methodology. Debt cap
33% (slightly more permissive than AAOIFI default), MAJORITY
consensus policy. Useful for operators following the older Dow
Jones Islamic Market screening tradition.

### 6.4 Multi-source consensus

When multiple providers screen the same symbol, three policies
combine their opinions (`halal/consensus.py`):

| Policy | Rule |
|---|---|
| `STRICT` (default) | Any `not_halal` rejects. Any `doubtful` (without `not_halal`) yields `doubtful`. Only unanimous `halal` yields `halal`. |
| `MAJORITY` | Most-common decision wins. Ties resolve to most-conservative. |
| `WEIGHTED` | Per-provider weight sums; largest wins. Ties resolve to most-conservative. |

All three share the **conservative-wins-on-ties** rule:
`not_halal > doubtful > halal`. The default STRICT policy is the
safest interpretation when scholars themselves disagree. An
operator who explicitly opts into MAJORITY / WEIGHTED takes
responsibility for the looser stance and can record their
reasoning in the audit trail.

### 6.5 Empty input

A symbol with **no** opinions returns `doubtful` rather than
`halal` — pin: "no opinions = unattested = refuse to trade",
the safest fail-shut default.

---

## Section 7: Exception queue

Decisions tagged `doubtful` flow to the operator's exception
queue (`halal/exception_queue.py`). The operator can:

1. **Approve** — typical for newly-listed tokens with
   insufficient screener data but a halal sector (e.g. a new
   layer-1 with no DeFi features, where the bot's category
   rules can't yet classify the token but the sector is
   clearly utility).
2. **Reject** — for borderline cases the operator wants to
   wait on.
3. **Defer** — explicit "ask a scholar before acting". The
   queue records this distinctly so a follow-up review can
   filter to deferred-and-still-unresolved.

Approved overrides are logged with `decided_by` (the operator's
identifier) and a free-form `reason` so a future scholar
challenge has the audit trail. **Approving an override does not
auto-promote the symbol to `halal` for future cycles** — it
only authorises *this* cycle's trade. The next refresh re-runs
the screener and queues the symbol again if it's still
borderline.

---

## Section 8: Audit trail

Every trade carries:

- `halal_screening_id` → `HalalScreening` row recording the
  decision, the source(s), the criteria (JSONB blob with the
  ratios that produced the decision), the cache hit flag.
- The active scholar profile name at decision time.
- The active consensus policy at decision time.

For trades where the operator overrode an exception (Section 7),
the chain extends to the queue row and the operator's free-form
reason. A future scholar reviewer can replay any historical
trade and answer "why was this allowed?" without reading code.

The post-trade `halal/audit.py:export_receipt(...)` builds a
JSON receipt joining the trade row with its screening — used
for compliance reporting and the Wave 2.A signed-receipt
workflow.

---

## Section 9: Limits and disclaimers

### 9.1 Methodology, not a fatwa

This handbook encodes one widely-followed methodology. It is
**not** a personal legal opinion (fatwa) for any specific
operator or trade. The operator's personal scholar may
disagree with one or more rulings here; the configuration
options in Section 6 are the lever for adapting.

### 9.2 No auto-promotion of overrides

An operator approving a `doubtful` override (Section 7)
authorises *that single cycle's trade*, not the symbol's
ongoing status. The screener re-runs every cycle.

### 9.3 The kill-switch is not Sharia compliance

The bot's `core/halt.py` kill-switch (engage with `halal-trader
halt --reason "..."`) stops new entries immediately. This is a
*risk control*, not a Sharia-compliance mechanism — engaging
halt does not retroactively un-trade a non-compliant symbol.
That's why the screener runs *before* the strategy, not after.

### 9.4 Real money

The bot is paper-trade only by design (`ALPACA_PAPER_TRADE=true`,
`BINANCE_TESTNET=true` are pinned in `.env.example`; `crypto/`
and `trading/` modules abort if these flip). Fiqh rulings on
paper trading are softer than on live trading — but the screener
applies the same rules either way, so the operator can study
the ruleset's behaviour without a real-money commitment.

---

## References and further reading

- AAOIFI Sharia Standards 21 (Financial Papers, Shares and Bonds)
  and 30 (Financial Indices) — the international framework this
  handbook's default profile follows.
- *Introduction to Islamic Finance* — Mufti Taqi Usmani.
- *Islamic Capital Markets: Products and Strategies* — Kabir
  Hassan & Michael Mahlknecht (eds.).
- Mufti Faraz Adam's *Crypto Shariah Screening Framework* — the
  basis for `crypto/screener.py`'s 4-pillar classification.
- Shariah Review Bureau, IFG, Mufti Menk, and Mufti Ismail Menk
  have all published frameworks for crypto screening with
  significant overlap; the bot's screener is closest to the IFG
  / Faraz Adam model.

For scholarly disagreements not captured by the three Wave 2.C
profiles, the operator may register a custom profile via
`halal.scholar_profiles.register_profile(...)` — the audit row
records the custom profile's name like any built-in.

---

_Last reviewed: 2026-05-01 (project-internal review). Pending
external scholar sign-off; the handbook sections aim to be
ready for review without further engineering work._
