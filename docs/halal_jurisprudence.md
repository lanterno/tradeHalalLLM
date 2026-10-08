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
configuration; the screen refuses the symbol whatever the
configuration (Section 6).

### 1.1 Riba (interest)

Any product that pays or charges interest as a function of the
holding period itself is forbidden. This rules out:

- Conventional bonds and preferred shares paying interest coupons.
- Margin / leverage products that charge a periodic interest
  rate to hold the position.

The screen catches interest-bearing debt and interest income through
the financial ratios (Section 2).

### 1.2 Maysir (gambling)

Pure speculation with no underlying productive activity is
forbidden. The line between speculation and investment isn't
binary; the bot's screening catches the most-clear cases:

- Leveraged perpetual futures with no underlying delivery
  obligation.
- Prediction markets whose payouts depend on event outcomes
  unrelated to a real-world economic activity.

The bot **does not** trade leveraged products: it is long-only
(`core/long_only.py`), never shorts, and the core portfolio's live
gate requires a cash account with no margin. This is the strongest
mechanical guarantee against Maysir.

### 1.3 Gharar (excessive uncertainty)

Contracts whose value depends on future events outside the
underlying — or where the underlying isn't deliverable — fall
under Gharar. Practical implications:

- Naked options and naked swaps are forbidden.
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

The in-house screen (`compliance/aaoifi.py`) rejects these by the
company's SEC SIC code, conservatively (casinos *and* general hotels,
since SIC 7011 does not separate them), and every exclusion names its
reason.

---

## Section 2: AAOIFI financial ratios (default profile)

The Accounting and Auditing Organization for Islamic Financial
Institutions (AAOIFI) publishes the most-cited international
standards for Sharia screening. Standards 21 and 30 set the
financial-ratio bars the bot's screen builds on; where S&P Shariah
is stricter, the stricter bar applies (Section 6).

### 2.1 Interest-bearing debt ≤ 30% of market cap

Companies whose interest-bearing debt exceeds 30% of trailing
12-month-average market capitalisation are **not halal**. Tightening
this further is a scholar's prerogative; the bot keeps the 30% bar.

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

The bot's strict screen makes it a hard limit, and a stricter one:
cash and interest-bearing securities at most 30% of market cap, and
accounts receivable separately at most 49% (Section 6).

### 2.4 Implementation in the bot

`compliance/aaoifi.py` applies the ratios to SEC filings, taking the
stricter of AAOIFI and S&P Shariah on every axis (Section 6):

```python
DEBT_LIMIT = 0.30           # interest-bearing debt / market cap
CASH_LIMIT = 0.30           # cash + interest-bearing securities / market cap
IMPURE_INCOME_LIMIT = 0.05  # impermissible income / revenue
RECEIVABLES_LIMIT = 0.49    # accounts receivable / market cap (S&P)
```

Anything that cannot be computed makes the verdict `doubtful`, which
is not halal: the screen fails closed rather than letting a partial
filing approve a company.

---

## Section 3: Asset-class rulings

### 3.1 Equities

Stocks are screened by the in-house AAOIFI-style screen
(`compliance/aaoifi.py`, run by `compliance/runner.py`) from SEC
filings:

- Section 1.4 business-activity exclusions.
- Section 2 financial ratios.

Its verdicts are checked weekly against the holdings of the SPUS and
HLAL halal ETFs. A production Zoya key, if one is ever set, can only
veto on top of it.

### 3.2 Commodities (gold, silver)

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

Gold and silver are not in the bot's tradable universe.

### 3.3 Sukuk (Islamic bonds)

Sukuk represent fractional ownership of a real asset, with
profit derived from rental / project cash-flow rather than
interest. They are permissible by construction (the structure
is the ruling).

The bot does not trade sukuk.

### 3.4 REITs

Real Estate Investment Trusts are permissible if **and only if**:

- The underlying properties are not used for forbidden
  activities (Section 1.4) — no hotels with bars / casinos,
  no conventional bank office towers where the lessee is a
  conventional bank.
- The REIT's debt structure passes Section 2's ratios.

The in-house screen applies Section 2's ratios to a REIT like any
other company, and fails one whose assets are mostly loans as a
lender.

### 3.5 International equities

The framework extends naturally — local-currency company
disclosures feed the same ratio engine. The bot does not
currently support international equities.

---

## Section 4: Decision states

The screener emits one of three decisions for any symbol; the
audit row (`HalalScreening`) records which.

| Decision | Meaning | Bot behaviour |
|---|---|---|
| `halal` | Compliant under the strict screen | Tradable |
| `doubtful` | Insufficient data, edge case, or borderline | Exception queue (Section 7) |
| `not_halal` | Fails one or more hard rules | Refused; never in the candidate set |

The rule is conservative throughout: `doubtful` is not halal, and an
index board's exclusion, or a production Zoya key's `not_halal`,
vetoes a `halal` from the screen (Section 6).

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

### 5.2 Dividend purification ledger

`compliance/purification.py` keeps one ledger per account: for every
dividend on a held position it accrues shares held × dividend per
share × the company's impure-income ratio (5%, the most a passing
company may have, where the screen has none), once per dividend. The
operator records donations against it (`halal-trader purify`); nothing
is marked paid automatically, since that needs explicit acknowledgement
after the disbursement actually settles.

### 5.3 Charity choice

The operator picks the disbursement target. The bot doesn't
prefer any particular charity but the recommended pattern is:

- A reputable Islamic charity (Islamic Relief, Penny Appeal,
  Zakat Foundation) with a public Sharia advisory board.
- A direct disbursement (not an investment vehicle) so the
  funds discharge the obligation cleanly.
- Records retained for tax purposes — purification is **not**
  Zakat, but jurisdictions vary on whether it's tax-deductible.

---

## Section 6: The strict screen

Scholars and index providers differ on edge cases. On 2026-10-02 the
operator chose the strict option: wherever AAOIFI and S&P Shariah
differ, the stricter rule applies, and an index Shariah board's
exclusion of a company is a veto (`compliance/index_veto.py`). In
practice:

- Cash is held to 30% of market cap (S&P) rather than AAOIFI's 33%,
  and accounts receivable to 49% (S&P's fourth ratio).
- Market cap is shares outstanding times the 36-month average
  month-end price where that history exists (S&P's method), so a
  name near a limit does not flip verdict with every price swing.
- Interest income a company does not report is estimated (cash and
  securities at 5%) rather than taken as zero.
- A REIT whose assets are mostly loans fails as a lender.

Only the newest screen counts, and only while it is fresh: a stale
or missing screen makes nothing halal (`halal/strict.py`). Every
verdict is stored with its metrics, so each decision can be audited.

---

## Section 7: Exception queue

Decisions tagged `doubtful` flow to the operator's exception
queue (`halal/exception_queue.py`). The operator can:

1. **Approve** — typical for a newly listed company whose filings
   are too thin for the ratios but whose business is clearly
   permissible.
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
- For the core portfolio, the screen date each order relied on
  (`core_orders.screen_as_of`), whose verdicts stay in
  `halal_screen_results` per method.

For trades where the operator overrode an exception (Section 7),
the chain extends to the queue row and the operator's free-form
reason. A future scholar reviewer can replay any historical
trade and answer "why was this allowed?" without reading code.

The post-trade `halal/audit.py:export_receipt(...)` builds a
JSON receipt joining the trade row with its screening — used
for compliance reporting, which `halal/signing.py` can sign so an
auditor can verify it without trusting the code.

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

The bot trades paper by default (`ALPACA_PAPER_TRADE=true` and
`CORE_PAPER=true` in `.env.example`), and going live needs that day's
confirmation token (`core/safeguards.py`). Fiqh rulings on
paper trading are softer than on live trading — but the screener
applies the same rules either way, so the operator can study
the ruleset's behaviour without a real-money commitment.

---

## References and further reading

- AAOIFI Sharia Standards 21 (Financial Papers, Shares and Bonds)
  and 30 (Financial Indices) — the international framework this
  handbook's screen builds on.
- *Introduction to Islamic Finance* — Mufti Taqi Usmani.
- *Islamic Capital Markets: Products and Strategies* — Kabir
  Hassan & Michael Mahlknecht (eds.).
- S&P Dow Jones Indices, *Shariah Indices Methodology* — the source
  of the strict screen's market-cap averaging and receivables ratio.

---

_Last reviewed: 2026-10-08 (project-internal review). Pending
external scholar sign-off; the handbook sections aim to be
ready for review without further engineering work._
