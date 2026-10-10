# H1 pre-registration: the overreaction bounce

*Written 2026-10-10, before any post-event price of the H1 universe was read. The binding record is
the `quant_trials` row `name = 'research.news.h1', kind = 'preregistration'`: its `config` (PREREG in
`src/halal_trader/events/h1.py`) carries every constant and the pins (code and data hashes) below, and
its `criterion` is the pass rule quoted here. This file is the human-readable copy. The full design is
[PHASE01_SPEC.md](PHASE01_SPEC.md); where this file lists a deviation, the deviation is what runs.*

Roadmap: [NEWS_ENGINE_ROADMAP.md](../NEWS_ENGINE_ROADMAP.md), Phase 1.

## The hypothesis

After a large drop on non-structural negative news (an analyst downgrade or an earnings miss:
`NSN_CORE`), a long position bought once the selling is exhausted earns a positive mean return net of
costs, abnormal to SPY at the same bars. Exhaustion: no new low for 20 minutes, a close above the VWAP
anchored at the news, and at most a quarter of the drop retraced. Exits: a 50% retrace target, a bar
close below the capitulation low, a structural item (abort), a halal-screen lapse, or a time stop.

The prior runs against it: post-news drops tend to continue (Chan 2003; Savor 2012), and this
repository's lexicon reversal test (worst decile, 5 days, −0.29%) agreed. A failure, including
"fail: insufficient events", is an accepted answer.

## Operator decisions taken by default (2026-10-10)

The roadmap's four decisions were not answered, so these defaults were taken and can be overridden:

- **Universe:** every liquid halal name (PRIMARY: halal in the newest screen strictly before the
  session, liquidity rank < 1000, previous close ≥ $5, one share class per company). Technology is a
  reported subgroup: on its own it has too few events for the count rule.
- **Holding:** both intraday (ID: flat by the close) and up to three sessions (MD3), as two cells.
- **LLM budget:** none; H1 uses rules only. The LLM interpreter (Phase 3) waits for an H1 pass.
- **Where it runs:** `src/halabot/playbooks` (the playbook) and `src/halal_trader/events` (the research).

## Cells, windows, metric, statistics

- Cells: `(NSN_CORE, ID, 1)` and `(NSN_CORE, MD3, 3)`. Nothing else is a confirmatory test.
- Train 2016-10-03..2021-12-31, validation 2022-01-03..2024-12-31, holdout from 2025-01-02, untouched
  until a pass (2025-12-01 on is non-confirmatory: earlier studies observed outcomes there).
- Metric per trade: net abnormal return to SPY at the same bar starts, across sessions in
  session-S units (A-ratios), minus two one-way costs from `study.COST_BPS` by liquidity rank.
- Fills: market orders at the clamped VWAP of the first bar starting at or after the order is active
  (real-time SIP bar visible 65 s after it starts, orders active 3 s after the decision; the open after
  a 5-minute gap). News is usable 600 s after publication.
- Statistics: the mean with CR1 standard errors clustered by entry date (df G−1); MD3 also a
  calendar-time Newey-West (4 lags) t; Holm across eligible cells at one-sided α 0.05.

## Pass rule (the registered criterion, verbatim)

> H1 passes iff some cell (NSN_CORE, ID|MD3) that Stage A finds eligible (validation >= 200
> entries/yr and >= 100 entry dates; train >= 500 entries and >= 100 dates) has, on train
> 2016-10-03..2021-12-31: mean net abnormal return > 0 with date-clustered CR1 t >= 2.0 (MD3 also
> calendar-time Newey-West(4) t >= 2.0), Holm rejection at one-sided alpha 0.05 across eligible cells,
> mean > 0 at 1.5x costs, beta-adjusted mean > 0, and mean > 0 excluding entries
> 2020-02-20..2020-06-30; and on validation 2022-01-03..2024-12-31 (train passers only) the same except
> the 2020 exclusion, Holm across train passers. Stage-A data skips <= 2% and unresolved exits <= 0.5%
> per window, else inconclusive. Nothing is changed after Stage A counts or any result is seen.

Order of work (enforced by `events h1`): data preconditions D1-D9, Phase 0 gates (g1-lookahead,
g1-synthetic, g1-determinism, r0, r1, r2, s0, s1, s1-calib, s2, s3, each with a passing ledger row),
registration, Stage A (entry counts only), train (eligible cells), validation (train passers only),
verdict, then the implementability trial (delayed-feed fills), sensitivities and the path atlas
(descriptive, train window only). Every step checks that the code and data are the registered ones; a
change after registration is a logged amendment reported with the old result, never a silent rerun.

## Data fixed before registration

- **Market calendar 2016-2024** from the broker's calendar (and the 2025-01-09 closure).
- **Delisted companies:** 60 tickers mapped to their SEC filer by hand (`compliance/delisted.HAND_CIKS`)
  and re-screened; unmapped companies fell from up to 4.6% to under 0.7% of every 2016-2024 screen.
- **Renamed tickers:** 46 renames (FB→META, SQ→XYZ, PCLN→BKNG...): the old tickers' news is stored
  under the current symbol, and reused tickers' foreign news is dropped (`events/renames.owner`).
- **8-K times:** EDGAR's submissions feed stamps about a third of filings 4-5 hours late (exactly the
  UTC offset); every 8-K since 2016 is restamped from its filing header (`events filings fix-times`).
- **Earnings facts:** parser v4, label `benzinga-earnings-v4+<PARSER_SHA>`.

## Deviations from PHASE01_SPEC.md (decided during the build, before registration)

Each was found by a reviewer or an audit of headlines, payloads or code; none from an outcome.

**Story builder (`stories.py`).** EDGAR closures (days of mourning, Christmas Eve orders) are not
business days; filings accepted before 06:00 become public at 06:00. The analyst clause rule applies
only when some clause has an analyst slot (otherwise the whole headline is checked against the
company's aliases), and the slot also reads "On <Co>" and "Initiates <Co> With <rating>". Only
substantive items (not noise, law-firm or mover) supersede facts; `nsn_at` re-reads the card at every
item. Movers never set `detect_at` or a follower's first item. A build waits for the current
extractor's facts and for every 8-K's header time, replaces its range atomically and marks it with a
hash of its inputs; counts and the plan refuse ranges not completely built from today's inputs.

**Parser and taxonomy (`earnings_parse.py`, `taxonomy.py`).** Guidance figures are re-read: never a
fragment glued to another token, a change ("By $0.05"), or the replaced guidance (From/Prior/Previous
and old parentheticals unless "To" follows the old figure); "+/-" tolerances are not a level. A figure
and its estimate in different units (a suffix on one side, or a ratio above 50 except EPS under $0.10)
do not compare. "May Not Compare" flags only its statement, or the whole headline when the flag names
an estimate, consensus, forecast, guidance or outlook. Sales after an estimate-first EPS are read
(`SALES_EST_FIRST`); forecasts are never read as results. A guidance cut stated beside a result types
the item `guidance_cut`; an explicit cut beats "unparsed guidance". False positives fixed: an analyst
"Restates <rating>", "Amended and Restated", a "Delay Report", a bare "Portnoy", "Bankruptcy Court
Approv...", capex/cost cuts, "Raises Lower End", analyst "Target" in a guidance cut; "Lowers/Raises
Target To" are price-target actions; a denial ("Did Not Downgrade") is `analyst_other`. Noise and
law-firm items contribute no facts; facts are chosen per metric (adjusted basis first, then latest).

**Context (`context.py`).** Listed in the module docstring and pinned in PREREG from there
(`context.deviations()`): the required news time, BROAD admitting names with no `ticker_ciks` row, the
share class settled before bars, `no_daily` reading S's bar, absent names as `not_halal`, NaN
descriptives, the 380-day history load.

**Simulator (`src/halabot/playbooks`).** The run takes a context view; a fill-model hook carries the
gate-only legacy fills; one pin per gate unit set, checked against the plan; a spare session after each
path for close fallbacks; visible bars are read-only views of the bars already visible; the daily book
compounds each trade's stake; a SPY bar defect sets aside only the headlines at or after it.

**Playbook (`bounce.py`).** The entry cutoff is a timer at close − 60 min; an equal low also disarms;
at the start, already-visible bars only set the state; an "in" story with no stock bar before the news
takes both previous closes as its baselines; a structural item of **any** of the symbol's stories,
known before the entry, vetoes it (after the entry it aborts); a buy filled at the flatten goes
straight to the time stop.

**Units (`units.py`).** Paths are four sessions (three plus the spare); 8-K stories also fetch S−1
(a filing retime can move a story one session back); a story is fetched when either universe admits it
at either possible news time; validation selects by the fastest news lag of the sensitivities; the SUE
gate's complement drops events whose units meet H1's own.

**Gates (`sim_gate.py`).** A 90% coverage floor; S0 passes when its table computes and reports the
deltas to the recorded study; SUE statistics are clustered by entry session; determinism runs the real
`sim.run` serially against a 6-worker pool. R0's reference changes in two numbers: the 2026-10-09
study read its headlines in no fixed order, so its seeded sample of 1,500 controls (recorded at
−0.28%) and one same-timestamp first headline (7,932 strong headlines recorded) cannot be reproduced.
With the order fixed, the pinned set has 7,931 strong headlines and the controls (n = 1,483, as
recorded) are at −0.30%; the strong group (n = 7,922, −0.30%, t −12.1) and the negative group
(−0.40%), which take every headline, reproduce exactly.
