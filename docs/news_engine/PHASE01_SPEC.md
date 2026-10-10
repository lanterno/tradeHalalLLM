# News engine Phase 0/1: final specification (stories-v1, rules-v1, sim-1, H1)

*Frozen on 2026-10-10 before any post-event price was read; committed before the H1 registration. Where the code differs, the code and the deviations in [H1_PREREGISTRATION.md](H1_PREREGISTRATION.md) are what was registered.*

*Status: final design, frozen before anyone has seen a post-event price. To write it I read code, payload shapes, counts and the design panel's reports, plus one market-calendar session count. I computed no return, dip or rebound and read no `minute_bars` or `daily_bars` rows. No repo file was edited.*

Paths below are relative to `/Users/nourataha/lab/halabot`.

## 0. How the disagreements were resolved

| Topic | Decision | From |
|---|---|---|
| Entity check, roundups, noise, law firms, movers, near-duplicates, followers | Kept. Every label is time-indexed through `card_at(t)` and `follower_at(t)` | trader, plus the critique fixes |
| Unclear negatives | Veto **before** entry only. After entry, only a **structural** item aborts | critique of trader #11 |
| H1 family | **NSN_CORE = analyst downgrade ∪ earnings miss.** Price-target-only stories (reactive; template starts 2018) and guidance cuts (structural) go to the atlas only | critique of rigor #5 and its comparison table |
| News latency | 600 s, defined once in the story builder (the reactor gate overrides it to 60 s) | critique of micro A7 |
| Drop trigger | SPY-relative on bar closes, scaled by σ, measured from the **pre-news** price | rigor trigger plus trader P0 |
| Fills | Market orders only. Decision on bar k fills at the VWAP of bar ≥ k+2 (clamped). After a 5-minute gap the fill is the open. A whole order fills on one bar. Never dropped as stale | rigor and micro, critique of micro B2 |
| Exits | Close-based stop and target, filled by market order. Structural abort, compliance exit, flatten at close − 5 min | rigor X1-X4, micro flatten and compliance |
| Costs | `study.COST_BPS` per side. A pass also needs a positive mean at 1.5× cost. The opening surcharge and square-root impact model is a non-gating sensitivity | critiques of rigor #13 and micro B1 |
| Prices across sessions | A-ratios from raw and `all` daily bars (dividends and splits). Skip only stale or unreadable adjustments | critiques of micro B8 and rigor #7 |
| Implausible ratios and unresolved trades | No drop filter. A trade that cannot exit is marked at its last trade and flagged, never dropped | critique of rigor #2 |
| Feed | H1 is judged on real-time SIP. SIP delayed 16 minutes is a separately counted trial that gates Phase 4's feed | micro, plus critiques A6 and #3 |
| Statistics | CR1 standard errors clustered by entry date, G−1 degrees of freedom. MD3 also needs a calendar-time Newey-West t with 4 lags. Holm across cells. Stage A counts come before any return | rigor, plus critiques #9 and #14 |
| Count rule | ≥ 200 entries a year **in validation**. Train needs N ≥ 500 and G ≥ 100 | critique of rigor #11 |
| Regime | Train must be positive with the 2020-02-20..06-30 crash excluded (gating, sign only). Results before and after the screen break are reported, not gated | critiques of rigor #8 and trader #15, combined |
| Order of work | Freeze → data preconditions → Phase 0 gate → register → Stage A → train → validation (train passers only) → verdict → atlas | rigor, plus critiques of micro C3 and trader #7 |
| Universe data | Map unmapped and delisted tickers, re-screen, and re-backfill news for renamed tickers **before** registering | critiques of rigor #1, trader #1 and micro A8 |

---

## A. Story builder (`src/halal_trader/events/stories.py`, `aliases.py`, `renames.py`)

### A.1 Constants and types

```python
BUILDER_VERSION: Final = "stories-v1"
ROUNDUP_MAX_SYMBOLS: Final = 3
NEWS_LAG: Final = timedelta(seconds=600)          # every item usable at public time + 600 s
REACT_CUTOFF: Final = timedelta(minutes=90)       # before effective_close_time(day)
DUP_JACCARD: Final = 0.60                         # same item repackaged, inside a story
REPOST_JACCARD: Final = 0.50                      # follower text match against a parent item
FOLLOW_SESSIONS: Final = 3
SHINGLE_K: Final = 3
FILING_OPEN, FILING_CUTOFF: Final = time(6, 0), time(17, 30)
STORY_KINDS: Final = frozenset({"news", "8-k", "8-k/a"})   # insider, 10-Q and 10-K are context only

@dataclass(frozen=True, slots=True)
class RawItem:
    event_id: int; source_id: str; kind: str          # 'news' | '8-k' | '8-k/a'
    symbol: str; published_at: datetime; seen_at: datetime
    headline: str                                     # '' for filings
    n_symbols: int | None                             # len(payload.symbols); None for live rows
    items_8k: tuple[str, ...]
    facts: tuple[EarningsFacts, ...]                  # extractor earnings_parse.EXTRACTOR (benzinga-earnings-v4+PARSER_SHA)

@dataclass(frozen=True, slots=True)
class StoryItem:
    raw: RawItem
    at: datetime                 # public time: news published_at; filing filing_public_at(accepted)
    available_at: datetime       # at + NEWS_LAG (live, Phase 4: max(seen_at, at))
    itype: str                   # taxonomy.classify_item
    shingles: frozenset[str]
    dup_of: int | None           # event_id of the earlier item it repeats
    supersedes: tuple[int, ...]  # event_ids whose facts it replaces (corrections)

@dataclass(slots=True)
class Story:
    story_id: str                # f"{symbol}:{session.isoformat()}"
    symbol: str
    session: date                # S, the reaction session
    items: list[StoryItem]       # by available_at, then event_id
    parent: str | None           # the symbol's latest non-follower story in S-3..S-1 (if any)
    parent_type_close: str | None
    def card_at(self, t: datetime) -> StoryCard          # only items with available_at <= t
    def follower_at(self, t: datetime) -> bool
    def nsn_at(self, cutoff: datetime) -> datetime | None
    def start_case(self) -> Literal["in", "out"]
    def at_news(self) -> datetime | None                 # `at` of the item that first made it NSN

def edgar_business_day(d: date) -> bool   # weekday and not a US federal holiday (observed rules, Juneteenth from 2021)
def filing_public_at(accepted: datetime) -> datetime
def reaction_session(available_at: datetime) -> date
def shingles(headline: str, k: int = SHINGLE_K) -> frozenset[str]
def jaccard(a: frozenset[str], b: frozenset[str]) -> float
def build(items: Iterable[RawItem], aliases: Mapping[str, AliasMatcher],
          *, history: Mapping[str, Sequence[Story]] | None = None) -> list[Story]      # pure
async def load_items(engine: AsyncEngine, *, start: date, end: date, symbols: Collection[str],
                     extractor: str | None = None) -> AsyncIterator[RawItem]  # conn.stream; None: EXTRACTOR when called
async def build_range(engine: AsyncEngine, *, start: date, end: date) -> int    # builds, persists, returns count
async def persist(engine: AsyncEngine, stories: Sequence[Story]) -> int
```

`build_range` runs one symbol at a time, sessions in order, so a story's parent and follower status come from stories already built. It loads symbols in batches of 100, the `labels.py` pattern.

### A.2 Item admission, in order

Each step adds to a counter in the run log.

1. **Kind.** Keep `news`, `8-k` and `8-k/a`. Insider and 10-Q/10-K rows are not story items: insider rows are stamped 17:00 ET and stop at 2026-03-31.
2. **Roundup.** For news, drop the row if `n_symbols > 3`.
   - Live rows have no `payload.symbols`, so `n_symbols=None` counts as 1 and the row is flagged `nsym_unknown`.
   - Code change: store `symbols` in the live payload (`sentiment/stocks_events.py:548`).
3. **Entity check (news only).** Keep the (article, symbol) row only if `AliasMatcher.matches` passes. Dots are removed on both sides.
   - **Analyst headlines** (where `ANALYST_ACTION` matches, §B.4) are checked clause by clause:
     - `analyst_clause(headline, matcher)` splits on `[;,]\s*` and on `\s+and\s+(?=(?:Downgrades|Upgrades|Initiates|Assumes|Resumes|Reinstates|Maintains|Reiterates)\b)`.
     - It returns the first clause whose `ANALYST_SLOT` slot matches the alias, together with the clauses after it that have no slot of their own.
     - If no clause matches, the row fails. The returned clause group, not the whole headline, is what gets classified.
   - **Alias sources** (`aliases.py`, persisted in table `story_aliases`; `alias_sha` is pinned in the pre-registration):
     - **(a) Names.** `market_assets.name` plus Alpaca `inactive_assets()` names, cleaned with `SUFFIX` (from the design panel's prototype).
       - Aliases are the cleaned full name, plus the first word if it has ≥ 4 letters (or ≥ 3 in capitals) and is not in `GENERIC`.
       - Plus the first two words if together they are ≥ 6 characters and neither is a single letter.
     - **(b) Learned slots.** Benzinga's own company slot, learned from single-symbol articles published 2016-01-01..2026-10-09 through `ANALYST_SLOT`, `EARN_CO` and `GUIDE_CO`.
       - A slot is kept if it occurs ≥ 3 times and makes up ≥ 10% of the symbol's slots. Its first word is also kept, under the same first-word rule.
     - **(c) `ALIAS_OVERRIDES`:**
       - GOOGL, GOOG: Google, Alphabet, YouTube, Waymo
       - META: Facebook, Meta, Instagram, WhatsApp, FB
       - AMZN: Amazon, AWS, Whole Foods
       - MSFT: Microsoft, LinkedIn, Xbox, Azure
       - AAPL: Apple, iPhone
       - TSLA: Tesla
       - NVDA: Nvidia
       - AMD: AMD, Advanced Micro
     - **(d) Tickers.** The ticker itself and every old ticker in `TICKER_RENAMES`, matched case-sensitively, optionally with a leading `$`. Only tickers of 2+ characters are used.
     - Sources (b) and (d) apply to **every** symbol, including ones without a `market_assets` row.
   - The pattern is `(?<![A-Za-z])(?:alias|…|(?-i:\$?TICKER))(?![a-z])` with `re.I`, longest alias first.
4. **Time.**
   - News: `at = published_at`.
   - 8-K: `at = filing_public_at(accepted)`:
     - accepted on an EDGAR business day in [06:00, 17:30) ET → the acceptance time;
     - otherwise → 06:00 ET on the next EDGAR business day. EDGAR is closed on Columbus Day and Veterans Day, so a filing after 17:30 the day before becomes public on the following business day.
   - `available_at = at + NEWS_LAG` for both.

### A.3 Reaction session and grouping

- `reaction_session(available_at)`:
  - the trading day of `available_at` if that day is a session and `available_at ≤ effective_close_time(day) − 90 min` (14:30, or 11:30 on early closes);
  - otherwise the next session. Calendar: `market_hours`.
- **Story = every admitted item of one symbol sharing a reaction session S.**
- Items typed `noise`, `law_firm` or `mover` belong to the story for display. They never create a story on their own, never type it and never trigger it.
- A (symbol, S) whose items are all of those types is persisted as type `noise_only` and is not a candidate for anything.

### A.4 Duplicates and corrections

- **Shingles.** Normalise first:
  - lowercase;
  - strip the prefix `^\s*(?:CORRECTION|CORRECTED|UPDATE[D]?|REPORTED EARLIER|EARLIER|BREAKING)\s*[:,\-]\s*`;
  - tokens by `[a-z][a-z0-9&'\-]+|\d+(?:\.\d+)?`, numbers mapped to `#`;
  - remove the stopwords: the a an of to in on for and or with by at as is are from after vs est estimate inc corp co company shares stock says said.

  Then take 3-grams.
- **Duplicates.** An item whose Jaccard with any earlier item in the story is ≥ 0.60 is `dup_of` that item. `n_distinct` counts the others.
- **Corrections.** A headline matching `^\s*(?:CORRECTION|CORRECTED)\b` supersedes the earlier items in the story that share a fact key (kind, period, basis, metric), or that have Jaccard ≥ 0.50 after the prefix is stripped.
  - From the correction's `available_at` onward, `card_at(t)` drops the superseded facts. Before that moment it keeps them.
  - Repeated wires with identical facts collapse to the latest fact per key.

### A.5 Followers (time-indexed)

- `parent` is the symbol's most recent **non-follower** story with session in [S−3, S−1] sessions. Its follower status is taken as of its own close.
- `follower_at(t)` is true if a parent exists and any of these holds:
  - **(a)** `parent_type_close ∈ STRUCTURAL ∪ {guidance_cut}`;
  - **(b)** every non-noise item with `available_at ≤ t` is reactive, where `REACTIVE = {analyst_pt_cut, analyst_init_neg, analyst_pt_raise, analyst_other, mover, guidance_unparsed}`;
  - **(c)** the first non-noise item (by `available_at`) has Jaccard ≥ 0.50 with any parent item.
- Rule (b) can only turn from true to false as items arrive. Rules (a) and (c) are fixed once the first item is known.

### A.6 Detection times and start case

- `nsn_at(cutoff)` is the first item `available_at ≤ cutoff` at which `card_at(t).family == "NSN_CORE"` (§B.7). It is re-checked at each new item until `cutoff`. `at_news` is that item's `at`.
- `start_case`:
  - `"in"` if `at_news ∈ [09:30, effective_close)` of **S**;
  - `"out"` otherwise: pre-open, previous evening, weekend, holiday, or late in-session news that rolled to the next session.
- `detect_at` is the first non-noise item's `available_at`, used for the atlas card.

### A.7 Persistence

Table `news_stories`: Alembic migration plus model. It is derived and rebuildable, so it is left out of backups.

- **Columns:** `builder_version, story_id, symbol, session, start_case, detect_at, nsn_at (nullable), at_news (nullable), type_detect, type_close, family_ever, follower_close, parent, n_items, n_distinct, items jsonb, flags jsonb`.
  - `items` holds `[{event_id, at, available_at, itype, dup_of, supersedes, entity_ok}]`.
- **Key and indexes:** PK `(builder_version, story_id)`; indexes `(symbol, session)` and `(family_ever, session)`.

Table `story_aliases`: `(builder_version, symbol, alias, source)`, PK on all four. It is backed up (small, and not reproducible later because `market_assets` changes).

Playbooks never read the persisted `type_close`. They call `card_at(now)`.

### A.8 Renamed tickers (`renames.py`)

```python
TICKER_RENAMES: Final[dict[str, tuple[str, date]]]   # old -> (current, last session under old)
async def seed_candidates(engine: AsyncEngine) -> list[tuple[str, date, date]]  # (symbol, first halal screen, first news)
async def backfill_renamed_news(engine: AsyncEngine, market: AlpacaMarketData, *, rate_per_min: int = 100) -> int
```

- **Seeding.** Take every symbol whose first news is more than 365 days after its first `halal` screen; 26 names were measured. Each is mapped by hand: FB→META 2022-06-08, SQ→XYZ, PCLN→BKNG, UTX→RTX, J2→ZD, CREE→WOLF, FLT→CPAY, and so on.
- **Backfill.** Fetches `market.news([old…])` over [2016-01-01, last old session] and stores rows with `symbol=current`. `payload.symbols` stays as Benzinga sent it.
- **Resume.** Task `news-renamed`, unit `OLD:YYYY-MM`.

---

## B. Rules taxonomy v1 (`src/halal_trader/events/taxonomy.py`, `RULES_VERSION = "rules-v1"`)

```python
@dataclass(frozen=True, slots=True)
class TypeMeta: direction: Literal["neg","pos","neutral","none"]; permanence: Literal["structural","unclear","transient","none"]
@dataclass(frozen=True, slots=True)
class EarningsVerdict:
    eps: int | None; sales: int | None            # +1 beat, 0 inline, -1 miss, None unknown
    guide: Literal["down","up","inline","unknown"] | None
    type: str
@dataclass(frozen=True, slots=True)
class StoryCard:
    type: str; direction: str; permanence: str
    family: Literal["NSN_CORE"] | None
    vetoes: tuple[str, ...]                       # structural + unclear-negative item types present
    positives: tuple[str, ...]
    earnings: EarningsVerdict | None
    follower: bool
TYPES: Final[dict[str, TypeMeta]]
def classify_item(item: RawItem, clause: str | None) -> str        # clause = analyst clause group or None
def earnings_verdict(facts: Sequence[EarningsFacts], items: Sequence[StoryItem]) -> EarningsVerdict | None
def resolve(items: Sequence[StoryItem], t: datetime, *, follower: bool) -> StoryCard
TAXONOMY_SHA: Final[str]   # sha256[:12] of json.dumps({regex name: pattern, ITEM_TYPES, TYPES, ORDER lists, REACTIVE}, sort_keys=True)
```

### B.1 Types

| Class | Types, as (direction, permanence) |
|---|---|
| **Structural negative** (veto before entry, abort after) | restatement, insolvency, delisting, short_report, fraud_probe, dilution, guidance_cut, regulatory_block, ma_break, customer_loss, impairment: all (neg, structural) |
| **Unclear negative** (veto before entry only) | antitrust_regulatory, management_exit, legal_adverse, operational_incident, auditor_change, agreement_terminated, sympathy: all (neg, unclear) |
| **Transient negative** | analyst_downgrade, analyst_pt_cut, analyst_init_neg, earnings_miss: all (neg, transient) |
| **Earnings, story level** | earnings_beat (pos, transient), earnings_inline (neutral), earnings_miss_guide_up (neutral, unclear), earnings_miss_guide_unk (neg, unclear), guidance_raise (pos, unclear), guidance_inline (neutral), earnings_unparsed (none) |
| **Positive** | ma_target (pos, structural), analyst_upgrade, analyst_pt_raise, buyback, regulatory_win, contract_win, product (pos, transient or unclear per prototype) |
| **Neutral** | ma_acquirer, restructuring, dividend, agreement, ma_closed, management_change, debt_financing, analyst_other, filing_other, other |
| **Item-only** | noise, law_firm, mover, earnings_fact, earnings_8k, guidance_unparsed |

### B.2 Classifying one item (first match wins)

1. **8-K.** Map every item code through `ITEM_TYPES`. Structural beats unclear, which beats any other. `2.02` gives `earnings_8k`, and {5.03, 5.07, 7.01, 8.01, 9.01} give `filing_other`. If a 5.02 filing's headline… filings have no headline, so 5.02 stays `management_change`.
2. `LAW_FIRM` → `law_firm`.
3. `NOISE` → `noise`.
4. The item has v4 `result` or `guidance` facts → `guidance_cut` if `GUIDANCE_CUT` matches the headline ("SLM Q1 EPS $0.870 Misses $0.880 Estimate; Withdraws FY20 Guidance"), else `earnings_fact`. Its facts count in the earnings verdict whatever the type.
5. `GUIDE_UNPARSED`, when `GUIDANCE_CUT` does not match → `guidance_unparsed` (an explicit cut goes on to step 6).
6. Negative regexes in this order: restatement, insolvency, delisting, short_report, antitrust_regulatory, fraud_probe, dilution, guidance_cut, regulatory_block, ma_break, customer_loss, impairment.
7. `ANALYST_DENIAL` → `analyst_other`. A denial is no rating change ("CORRECTION: RBC Did Not Downgrade KB Home Today", "Himax Was Not Downgraded"). It reads the **whole headline**, not the clause group: the company's clause alone would miss a denial in another clause ("CORRECTION: Baird Upgrade Of Kohl's Was From Nov. 10th 2017, Firm Did Not Upgrade Kohl's Today").
8. Analyst. If `ANALYST_ACTION` matches, classify the **clause group**:
   - `DOWNGRADE` and not `UPGRADE` → `analyst_downgrade`;
   - `UPGRADE` → `analyst_upgrade`;
   - `INIT_NEG` → `analyst_init_neg`;
   - `PT_CUT` → `analyst_pt_cut`;
   - `PT_RAISE` → `analyst_pt_raise`;
   - otherwise → `analyst_other`.
9. `MOVER` → `mover`.
10. management_exit, legal_adverse, operational_incident, sympathy.
11. ma_target, guidance_raise, buyback, regulatory_win, contract_win, product, ma_acquirer, restructuring, dividend.
12. `other`.

### B.3 8-K items (`ITEM_TYPES`)

| Item | Type | Direction / permanence |
|---|---|---|
| 4.02 | restatement | neg / structural |
| 3.01 | delisting | neg / structural |
| 2.04, 1.03 | insolvency | neg / structural |
| 2.06 | impairment | neg / structural |
| 3.02 | dilution | neg / structural |
| 4.01 | auditor_change | neg / unclear |
| 1.02 | agreement_terminated | neg / unclear |
| 2.05 | restructuring | neutral / unclear |
| 2.01 | ma_closed | neutral / unclear |
| 1.01 | agreement | neutral / unclear |
| 5.02 | management_change | neutral / unclear |
| 2.03 | debt_financing | neutral / unclear |
| 2.02 | earnings_8k (resolved at story level) | — |
| 5.03, 5.07, 7.01, 8.01, 9.01 | filing_other | none |

### B.4 Regexes

All are `re.I` unless marked case-sensitive.

```python
LAW_FIRM = r"Law Firm|Rosen Law|Pomerantz|Levi & Korsinsky|Bragar|Faruqi|Kessler Topaz|Glancy|Bronstein|Schall Law|Robbins (?:Geller|LLP)|Gross Law|Kirby McInerney|Holzer|Portnoy|Block & Leviton|Hagens Berman|Bernstein Liebhard|Johnson Fistel|Halper Sadeh|Gainey McKenna|Kahn Swick|Monteverde|Ademi|Brodsky & Smith|Rigrodsky|Wohl & Fruchter|Frank R\.? Cruz|Howard G\.? Smith|Edelson Lechtzin|Investor Alert|Shareholder Alert|Investors? Who Lost|Lead Plaintiff|Class Action (?:Filed|Lawsuit Filed|Reminder)|Securities Class Action|Deadline Alert"
NOISE    = <taxonomy.NOISE: the pattern in src/halal_trader/events/taxonomy.py>
MOVER    = <taxonomy.MOVER>
ANALYST_ACTION = r"\b(?:Downgrades?|Upgrades?|Maintains|Reiterates|Initiates|Assumes|Resumes|Reinstates)\b"   # case-sensitive
ANALYST_SLOT = r"\b(?:Downgrades|Upgrades|Initiates Coverage On|Assumes|Resumes|Reinstates)\s+(?P<a>.+?)(?:\s+(?:to|To|With|with|At|at)\b|,|$)|\b(?:Maintains|Reiterates)\s+.{1,40}?\s+on\s+(?P<b>.+?)(?:,|$)"  # case-sensitive
DOWNGRADE = r"\bDowngrade[sd]?\b|\bCut To (?:Neutral|Hold|Sell|Underperform|Underweight|Market Perform|Equal-?Weight|Sector Perform|Reduce)\b"
UPGRADE   = r"\bUpgrade[sd]?\b|\bRaised To (?:Buy|Outperform|Overweight)\b"
PT_CUT    = r"\b(?:Lowers?|Lowered|Cuts?|Slashes|Trims|Reduces|Decreases)\b.{0,40}\b(?:Price Target|PT|Target Price)\b|\b(?:Price Target|PT)\b.{0,10}\b(?:Cut|Lowered|Reduced|Slashed|Trimmed)\b"
PT_RAISE  = r"\b(?:Raises?|Raised|Boosts?|Lifts?|Increases?|Hikes?|Bumps?)\b.{0,40}\b(?:Price Target|PT|Target Price)\b"
INIT_NEG  = r"\bInitiates Coverage On .{1,60} [Ww]ith (?:Sell|Underperform|Underweight|Reduce)\b"            # case-sensitive
ANALYST_DENIAL = r"\b(?:Did|Does|Was|Has)\s+Not\s+(?:Issue\s+An?\s+)?(?:Downgrad|Upgrad|Initiat|Rating\s+Change)"

RESTATEMENT = r"\bRestat(?:e|es|ed|ement)\b|Non-?Reliance|Material Weakness|Accounting (?:Irregularit|Review|Errors?|Issues?|Probe)|Delay(?:s|ed)? (?:Its |Filing|Of )?(?:Annual|Quarterly)? ?(?:Report|10-[KQ]|Filing)|Late Filing|NT 10-[KQ]|Unable To Timely File"
INSOLVENCY  = r"\bBankrupt|Chapter (?:11|7)\b|Going Concern|\bDefault(?:s|ed)? On\b|Missed (?:Interest|Coupon) Payment|Forbearance|Restructuring Support Agreement|Insolven|Covenant (?:Breach|Waiver)|Debt Restructuring"
DELISTING   = r"\bDelist(?:ing|ed)?\b|(?:Nasdaq|NYSE) (?:Notice|Deficiency)|Deficiency (?:Letter|Notice)|Minimum Bid Price|Non-?Compliance With (?:Nasdaq|NYSE)"
SHORT_REPORT= r"\b(?:Hindenburg|Muddy Waters|Citron|Spruce Point|Grizzly Research|Culper|Kerrisdale|Bonitas|Wolfpack|Fuzzy Panda|Iceberg Research|Blue Orca|Gotham City|Viceroy|J Capital|Scorpion Capital|Morpheus Research|Hunterbrook|Andrew Left)\b|Short[- ]Sell(?:er|ing) (?:Report|Attack|Alleg|Target)|Short Report|Bear Raid"
ANTITRUST   = r"\bAntitrust|Anti-?Competitive|Competition (?:Watchdog|Authority|Regulator|Probe|Commission)|European Commission|\bEU (?:Fine|Probe|Investigat|Regulators?|Antitrust|Commission)|Digital Markets Act|\bDMA\b|\bCMA\b|\bFTC\b|Monopol|\bBreak(?:ing)?[- ]?[Uu]p\b(?! Fee)"
FRAUD_PROBE = r"\bSEC (?:Probe|Investigat|Subpoena|Charges|Sues|Inquiry|Enforcement)|Wells Notice|\b(?:DOJ|Department Of Justice) (?:Probe|Investigat|Charges|Sues|Subpoena|Indict)|\bSubpoena|Criminal (?:Probe|Investigation|Charges|Complaint)|\bFBI (?:Probe|Investigat|Raid)|\b(?:FBI|Police|Agents|Authorities|Regulators) Raid|\bRaided\b|\bIndict|\bFraud\b(?! (?:Prevention|Detection|Protection|Solutions?|Platform|Management))|Whistleblower|(?:Opens?|Launch(?:es)?|Faces?|Under) (?:An? )?(?:Probe|Investigation)|Probe(?:s|d)? Into|Investigat(?:es|ing|ion) Into"
DILUTION    = r"(?:Public|Secondary|Proposed|Underwritten|Follow-On|Common Stock|Equity|Share|Stock|Unit|Registered Direct|Upsized|Overnight|Convertible(?: Senior)?(?: Notes)?|Exchangeable(?: Senior)?(?: Notes)?) Offering|Prices? (?:Upsized )?(?:\$?[\d.,]+[MBK]? )?(?:Shares|Offering|Public Offering)|Private Placement|At-The-Market|\bATM (?:Program|Offering|Facility)|Mixed (?:Securities )?Shelf|Shelf Registration|Files? For .{0,20}Shelf|Block Trade|Bought Deal|(?:Sells?|Selling) [\d.,]+[MK]? (?:Shares|Of Its Shares)|Stock Sale By|Secondary Sale|May (?:Offer|Sell|Issue)(?: And Sell)?\b.{0,40}\b(?:Shares|Common Stock)|Equity Distribution Agreement|Sales Agreement For .{0,30}Shares"
GUIDANCE_CUT= <taxonomy.GUIDANCE_CUT>
REG_BLOCK   = r"Complete Response Letter|\bCRL\b|FDA (?:Rejects|Declines|Refuses|Delays)|Clinical Hold|Fail(?:s|ed)? (?:To Meet )?(?:Its )?Primary|Did(?:n't| Not) Meet (?:Its )?(?:Primary|Endpoint)|Trial (?:Fail|Halt|Stopped)|Discontinu(?:e|es|ed) (?:Trial|Study|Development|Program)|Export (?:Ban|Restriction|Curb|Control|License)|Blocks? (?:Merger|Deal|Acquisition)|\b(?:Import|Export|Sales) Ban\b|\bBan(?:s|ned)? (?:On|Of) (?:Sales|Imports?|Exports?|Shipments?)\b|\bBanned From\b"
MA_BREAK    = r"Terminat(?:e|es|ed|ion) (?:Of )?(?:the )?(?:Merger|Acquisition|Deal|Agreement To)|Deal (?:Collapses|Falls Through|Called Off|Blocked|Terminated)|Walks Away|Abandons? (?:Deal|Bid|Merger|Acquisition)"
CUSTOMER_LOSS = r"Los(?:es|ing|t) (?:A |Its |Key |Major )?(?:Contract|Customer|Client|Account)|Contract (?:Cancel|Terminat|Loss)|Customer (?:Loss|Exit)"
IMPAIRMENT  = r"\bImpairment|Write-?(?:Down|Off)s?\b"

MGMT_EXIT   = r"\b(?:CEO|CFO|Chief Executive|Chief Financial|President|Chairman|Founder)\b.{0,60}\b(?:Steps? Down|Stepping Down|Resign|Depart|Ousted|Fired|Terminated|Exits?|Leaves|Leaving|To Leave|Abrupt|Unexpected|Sudden)"
LEGAL_ADVERSE = r"\bLawsuit (?:Filed )?Against\b|\bSued\b|\bFaces? (?:A )?(?:Lawsuit|Suit|Class Action)\b|\bLoses? (?:Lawsuit|Case|Appeal|Patent (?:Case|Suit|Fight|Trial))|\bJury (?:Verdict|Orders|Finds|Awards)|\b(?:Judge|Court) Rules? Against|\bInfring(?:es|ed|ement) (?:Verdict|Ruling)|\bInjunction Against|\bFined\b|\bFine Of\b|\bPenalty Of\b"
OP_INCIDENT = r"\bRecall(?:s|ed|ing)?\b|Outage|(?:Data|Security) Breach|Cyber-?attack|Hack(?:ed|ers)?\b|Ransomware|Fire At|Explosion|\b(?:Workers?|Union|Labor|Employees) (?:Go On |Begin |Launch )?Strike|\bOn Strike\b|\bStrike At\b|Walkout|Production (?:Halt|Issue|Problem|Delay)|Supply (?:Shortage|Disruption)|Grounded"
SYMPATHY    = r"\bSympathy\b"
# positive / neutral: ma_target, guidance_raise, buyback, regulatory_win, contract_win, product, ma_acquirer,
# restructuring, dividend: the patterns in taxonomy.py (the prototype's table T).
```

### B.5 Earnings: parser v4 and verdicts

- **Parser.** `earnings_parse.py` gains `SALES_ONLY`, `RESULT_EST_FIRST`, `GUIDE_V4`, `NOT_COMPARABLE` and `GUIDE_UNPARSED`, (see earnings_parse.py). `EXTRACTOR = f"benzinga-earnings-v4+{PARSER_SHA}"` (`EXTRACTOR_V4 = "benzinga-earnings-v4"` is the prefix): every parser change is a new label, so no reader mixes two parsers' facts. Extraction only adds rows; the evening refresh and `events extract --drop-superseded` delete every other v4 label's facts afterwards. v3 rows are kept.
  - **Parse order per segment:** `_GUIDE_RANGE`, `GUIDE_V4`, `_RESULT`, `RESULT_EST_FIRST`, `_INLINE`, `SALES_ONLY`.
  - `NOT_COMPARABLE` sets that fact's surprise to None.
  - `PARSER_SHA` = sha256[:12] of the pattern sources.
- **Fact selection.** From non-superseded `result` facts available by t, take adjusted basis if any exists, otherwise GAAP. Use the latest per metric.
- **Per-metric verdict** m ∈ {eps, sales}:
  - surprise known: +1 if `surprise ≥ +0.005`, −1 if `≤ −0.005`, else 0;
  - surprise None: from the verdict word (beat(s) → +1, miss(es) → −1, meets/in-line/inline → 0);
  - otherwise (up from / down from / vs with no estimate) → None.
- **Guidance flags:**
  - `guide_down`: a guidance fact with action ∈ {lowers, cuts, withdraws, suspends}, or with `surprise ≤ −0.01` (any action, any metric), or a `guidance_cut` item.
  - `guide_up`: not guide_down, and a guidance fact with action `raises` or `surprise ≥ +0.01`.
  - `guide_unknown`: a `guidance_unparsed` item and no guidance fact.
  - `inline`: guidance facts exist and none of the above holds.
- **Story earnings type**, first match:
  1. guide_down → `guidance_cut` (structural);
  2. any metric −1 and guide_up → `earnings_miss_guide_up`;
  3. any metric −1 and guide_unknown → `earnings_miss_guide_unk`;
  4. any metric −1 → `earnings_miss` (a mixed beat-and-miss print counts);
  5. any metric +1 → `earnings_beat`;
  6. result facts → `earnings_inline`;
  7. only guidance facts → `guidance_raise` or `guidance_inline`.

### B.6 Story resolution `resolve(items, t)`

Uses items with `available_at ≤ t`, superseded facts removed and noise/law_firm ignored. The first match sets the type:

1. Any structural item → the earliest structural item's type.
2. Earnings type (B.5), if any facts exist.
3. `earnings_8k` or `guidance_unparsed` with no facts → `earnings_unparsed`.
4. `analyst_downgrade`.
5. Unclear negatives in this order: antitrust_regulatory, management_exit, legal_adverse, operational_incident, auditor_change, agreement_terminated, sympathy.
6. `analyst_pt_cut` (including `analyst_init_neg`).
7. Positives in this order: ma_target, guidance_raise, analyst_upgrade, buyback, regulatory_win, contract_win, product, analyst_pt_raise.
8. The most frequent neutral type (ties go to the earliest).
9. `mover_only`, then `analyst_other`, then `noise_only`.

Fields: `vetoes` = all structural and unclear-negative types present, plus `guidance_cut`. `positives` = present items of {analyst_upgrade, guidance_raise, ma_target}.

### B.7 The H1 family (exact)

`card_at(t).family == "NSN_CORE"` iff all of these hold:

- `type ∈ {"analyst_downgrade", "earnings_miss"}`;
- `vetoes == ()`;
- `positives == ()`;
- `follower_at(t) is False`;
- the story's first non-noise item is not a `mover`.

**Excluded from H1 and described in the atlas:** price-target-only stories, guidance cuts, misses with guide-up or unparsed guidance, unparsed earnings, and every unclear or structural negative.

---

## C. Point-in-time context (`src/halal_trader/events/context.py`)

```python
@dataclass(frozen=True, slots=True)
class Eligibility:
    eligible: bool
    reason: Literal["ok","no_screen","not_halal","unmapped","rank","price","share_class","no_daily","no_sigma"]
    universe: Literal["primary","broad"]          # broad adds index-veto-only and unmapped-only rows (sensitivity)
    screen_as_of: date | None; verdict: str | None; cik: int | None
    sector: str | None; tech: bool
    liquidity_rank: int | None                    # 0-based index in universe_at(S, top_n=3000)
    cost_bps: float                               # study.cost_bps(rank)

@dataclass(frozen=True, slots=True)
class PreEvent:
    session: date; prev_session: date
    prev_close_s: float          # close_raw(S-1) * A(S-1) / A(S): session-S raw units
    spy_prev_close_s: float
    sigma: float; sigma_n: int   # sd(ddof=1) of daily abnormal returns, see C.2
    beta: float                  # OLS slope on SPY over the same sessions, clipped to [0.5, 2.0]
    atr_pct: float               # signals.indicators.atr (Wilder 14) on 'all' bars / last close (descriptive)
    adv20_usd: float             # mean raw close * raw volume, 20 sessions
    ret5_vs_spy: float; ret20_vs_spy: float
    hi20_s: float; lo20_s: float; hi252_s: float; lo252_s: float   # adjusted into S units (descriptive levels)

class PitContext:
    @classmethod
    async def load(cls, engine: AsyncEngine, *, symbols: Collection[str], start: date, end: date) -> PitContext
    def eligibility(self, symbol: str, session: date) -> Eligibility
    def pre_event(self, symbol: str, session: date, at_news: datetime) -> PreEvent | None
    def adj(self, symbol: str, day: date) -> float | None          # A(day) = close_all / close_raw
    def screen_verdict(self, symbol: str, day: date) -> str          # newest as_of < day; 'no_screen' if none
    def facts_before(self, symbol: str, at: datetime, *, lookback_days: int = 200) -> list[EarningsFacts]
    def stories_before(self, symbol: str, session: date, *, sessions: int = 20) -> list[StoryCard]  # close cards
```

### C.1 Strictly-before rules

| Field | Source | Rule |
|---|---|---|
| Screen, verdict, CIK, SIC | `halal_screen_current`, loaded once | newest `as_of < S`; `verdict == 'halal'`; `cik IS NOT NULL`. `strict.verdict` is never used (its freshness rule does not apply to history) |
| BROAD (sensitivity only) | same | adds rows whose only reason matches `excluded by %Shariah index%`, and rows with `cik IS NULL` whose `ticker_ciks.status != 'fund'` |
| Liquidity rank | `universe_at(S, top_n=3000)`, cached per month | uses months before S's month; rank < 1000 |
| Price | `prev_close_s ≥ 5.0` | known before the open |
| Share class | the screen row's `cik` | among eligible symbols with one CIK at S, keep the lowest rank; the others get `share_class` |
| Sector, tech | `cap_sector(symbol, sic_description)` | tech iff `== TECHNOLOGY` |
| σ, β | `daily_bars` `all`, symbol and SPY | the 60 sessions whose close is **strictly before `at_news`** (pre-open news: through S−1; after-close news on N: through N; late in-session news on N: through N−1); ≥ 40 valid, else `no_sigma` |
| `prev_close_s`, `spy_prev_close_s` | `daily_bars` raw and `all` | S−1, converted to S units with `A(S−1)/A(S)`. That ratio holds exactly the corporate actions effective at S's open, which are announced beforehand |
| ATR, ADV, ret5/20, levels | `daily_bars` | sessions ≤ S−1 |
| Facts, prior stories | `event_facts`, `news_stories` | `published_at < at`; prior stories by session < S |

- **Daily abnormal return:** `C_all(x)/C_all(x−1) − 1 − (SPY_all(x)/SPY_all(x−1) − 1)`.
- **Loading:**
  - screens: one query;
  - universe: one query per month (about 121);
  - daily bars: streamed in symbol batches of 100 over [start − 100 calendar days, end + 7].
- **Not point in time (stated):** SIC description and therefore sector, today's company names, and the restated SEC facts behind the screen.

---

## D. Minute simulator (`src/halabot/playbooks/`)

### D.1 Layout

```
types.py      Session, BarSeries, DailyPoint, PathData, PathSkip, inputs (BarIn, GapIn, NewsIn, SessionIn, FillIn,
              OrderClosedIn, ComplianceIn, TimerIn) and intents (Submit, Cancel, SetTimer, Transition, Finish)
clock.py      FeedProfile, SIP_RT, SIP_DELAYED, HARNESS; the event heap
playbook.py   Playbook, Ctx, MarketView, PositionView, StoryView Protocols
bounce.py     OverreactionBounce (the H1 state machine; §G.4)
exchange.py   Exchange: working orders, fill rules, costs (pure)
rules.py      admission: halal gate, long-only clamp, session windows, entry window
loader.py     MinuteBarLoader, SpyData, ComplianceTimeline, WindowGuard, WindowUnlock
sim.py        simulate_symbol(), run()
records.py    TradeRecord, StoryOutcome, PgOutcomeSink, daily legs
legacy.py     LegacyReactorFill, DailyBarSource (gate-only fill models)
tests/halabot/playbooks/  test_clock, test_exchange, test_bounce, test_lookahead, test_loader, test_determinism, test_known_answers
```

Output tables are created by `halabot/platform/db.py:bootstrap_schema` (the `hb_*` exception):

- `hb_playbook_run(run_id uuid PK, created_at, mode sim|shadow|paper|live, playbook, playbook_version, cell, feed, window, stop_at, config jsonb, config_hash, prereg_trial_id, code_sha)`
- `hb_playbook_story(run_id, story_id, PK both, symbol, session, nsn_at, terminal_state, reason, skip, triggered_at, armed_at, entry_decided_at)`
- `hb_playbook_trade(run_id, story_id, PK both, <TradeRecord fields>)`

**Backups** include only runs with `mode IN ('shadow','paper','live')`.

### D.2 Clock, feed and latency

```python
@dataclass(frozen=True, slots=True)
class FeedProfile: name: str; bar_lag: timedelta
SIP_RT      = FeedProfile("sip-rt",      timedelta(seconds=5))     # H1
SIP_DELAYED = FeedProfile("sip-delayed", timedelta(minutes=17))    # implementability trial
HARNESS     = FeedProfile("harness",     timedelta(0))             # reactor gate only
ORDER_LAG: Final = timedelta(seconds=3)
GAP: Final = timedelta(minutes=5)
PRE_OPEN: Final = time(9, 20)
```

- A bar with start `ts` is visible at `ts + 60 s + bar_lag`. News is visible at `available_at`, which the builder sets; the simulator has no news lag of its own.
- There is one heap per symbol. The key is `(time, priority, seq)` and ties go by insertion order, so runs are deterministic.

| Priority | Event |
|---|---|
| 0 | exchange tests working orders against bar i (eligible iff `ts_i ≥ active_at`) at `ts_i + 60 s` |
| 1 | `FillIn` at the end of the filling bar + 1 s |
| 2 | `SessionIn`: pre_open 09:20, open, entry_start (open + 20 min), entry_cutoff (close − 60 min), flatten (close − 5 min), close |
| 3 | SPY `BarIn` |
| 4 | symbol `GapIn` (when the gap before bar i is ≥ 5 min, or the first bar is ≥ open + 5 min, which sets the `late_open` flag), then `BarIn` |
| 5 | `NewsIn` |
| 6 | `TimerIn` |

**Worked example (SIP_RT).** Bar 10:14 is visible at 10:15:05. The decision makes the order active at 10:15:08. The fill is on the first bar with `ts ≥ 10:15:08`, which is the 10:16 bar, at its VWAP.

**Under SIP_DELAYED** the same bar is visible at 10:32:00, the order is active at 10:32:03, and the fill is the 10:33 bar's VWAP.

### D.3 The Playbook protocol

```python
class Playbook(Protocol):
    name: str; version: str
    path_sessions: int                 # 1 (ID) or 3 (MD3)
    def start(self, ctx: Ctx) -> list[Intent]
    def on(self, ev: Input, ctx: Ctx) -> list[Intent]
    def state(self) -> str

class MarketView(Protocol):
    def bars(self, symbol: str) -> BarSeries        # visible bars of the current path, ending at searchsorted(visible_at, now, 'right')
    def last(self, symbol: str) -> float | None
    def to_s_units(self, price: float, day: date) -> float   # price * A(day) / A(S)
```

The playbook never sees a `DailyPoint` of the current or a later session. It sees only `PreEvent`, and A-ratios through `to_s_units`.

### D.4 Orders and admission (`rules.py`)

- **`OrderKind`:** H1 uses only `MARKET` (time-in-force day). `LIMIT`, `MOO` and `MOC` are reserved; each later trial that uses one is a new registration, tagged `.port` or `.research`.
- **Buy admission:**
  - `ComplianceTimeline.verdict(symbol, S) == 'halal'`; otherwise `not_halal` or `no_screen` (fail closed, like `_execute_buy`);
  - submitted within [open + 20 min, close − 60 min] of S;
  - no position held.
- **Sell admission:** the quantity is clamped to the position (long-only); a sell with no position is rejected.
- **Session rules:** an order must be active within [open, close). Orders outside session hours are refused, not queued. Every order is a day order and expires at `close`.
- **Order ids:** deterministic, `f"{story_id}:{n}"`.

### D.5 Fill rules (`exchange.py`, pure)

```python
class Exchange:
    def __init__(self, cfg: SimConfig) -> None
    def submit(self, order: WorkingOrder) -> None
    def on_bar(self, i: int, bars: BarSeries, *, first_of_session: bool) -> list[Execution]
    def expire(self, day: date) -> list[WorkingOrder]
```

- **Market order.** Fills the whole order on the first bar i with `ts_i ≥ active_at` in the current session. There is no participation cap; the trade records `participation = (reference_notional / price) / v_i`.
- **Price:**
  - the open `o_i` if any of these holds (flag `gap_fill`):
    - `ts_i − active_at ≥ 5 min`;
    - `ts_i − ts_{i−1} ≥ 5 min`;
    - i is the session's first bar and `ts_i ≥ open + 5 min`;
  - otherwise `clamp(vw_i, l_i, h_i)`, or `o_i` if `vw` is null.
- **No eligible bar before close:**
  - entry → expires; the story ends `EXPIRED/entry_unfilled`;
  - exit on a non-deadline session (MD3) → resubmitted at the next session's open, active at open + 3 s, with the same rule;
  - exit on the deadline session, or ID → the official daily close of that session (`close_fallback`); if that session has no daily bar → the next session inside the window by the market rule (`no_market`); if none → `unresolved`, marked at the last traded price (last minute close or last daily close). The trade is kept.
- **SPY legs** use the identical rule at the identical bar starts, with no cost. Fallback legs use SPY's official close.
- **Bar sanity at load:** a row with a non-positive price, `h < max(o,c)` or `l > min(o,c)` is dropped and counted.

### D.6 Costs

`c = study.cost_bps(rank)` one-way bps: 7 for rank < 300, 15 for 300-999. The return carries `−2c/1e4`.

`SimConfig.cost ∈ {"study", "study_x1.5", "study_x2", "surcharge"}`. The last three are reporting modes:

- **"surcharge"** = half-spread (2 / 10 bps) × 2 for fills within 15 minutes after the session's first bar or 5 minutes after a gap, plus `max(5, σ20·1e4·√(10,000/ADV20$))` bps of impact.

### D.7 Sessions, halts, overnight and compliance

- **Path.** Sessions S..S+`path_sessions`−1. Only regular-session bars in [09:30, effective close) exist, so early closes are honoured. A missing minute is never a price.
- **Halts.** A halt cannot be told from an illiquid stretch. Both are handled by `GapIn` and the gap fill.
- **Overnight (MD3).** News arrives at `available_at`. Then `pre_open` (the `ComplianceIn` check), then the first bar.
  - An exit pending overnight is resubmitted at the open.
  - A close-based stop on the first bar is decided when that bar is visible and fills on a later bar.
- **Compliance.** At `pre_open` of S+1 and S+2, if `screen_verdict(symbol, day) != 'halal'` → a market sell at the open + 3 s rule (`exit_reason="compliance"`).
- **Flatten.** At close − 5 min of the deadline session, every order is cancelled and a market sell goes out (`exit_reason="time_stop"`). A `Finish` while holding gets the same treatment.

### D.8 Prices across sessions

- All levels and returns are in **session-S units**: `p_S = p_raw(d)·A(d)/A(S)`, with `A(d) = close_all(d)/close_raw(d)`.
  - Splits and dividends inside the path are handled exactly. A dividend accrues to the holder through A.
  - SPY gets the same treatment with its own A.
- **Data-defect skip** (`adjust_defect`): `compliance.runner.corporate_actions(engine, [sym, "SPY"], since=first σ session − 1, through=last path session)` reports a `.stale` date, or a split with `ratio is None`, inside that span. This is the only corporate-action skip. It is counted as a data skip.

### D.9 Data loading

```python
# src/halal_trader/data/minutes.py (addition: the one reader stays here)
@dataclass(frozen=True, slots=True)
class BarArrays:
    ts: NDArray[np.int64]; o: NDArray[np.float64]; h: NDArray[np.float64]; l: NDArray[np.float64]
    c: NDArray[np.float64]; v: NDArray[np.float64]; vw: NDArray[np.float64]   # vw NaN when null
async def read_windows(engine: AsyncEngine, units: Sequence[tuple[str, date]]) -> dict[tuple[str, date], BarArrays]
```

- **Query.** One per batch: `unnest(:i, :sym, :lo, :hi)` joined to `minute_bars` on `symbol` and `ts ∈ [lo, hi)` (`session_bounds`), packed with `string_agg(int8send/float8send … ORDER BY ts)`, decoded with `np.frombuffer(…, ">f8")`.
- **Batches.** 250 paths, month-local. One batch is prefetched, so the peak is about 60 MB.

```python
class Window(StrEnum): TRAIN = "train"; VALIDATION = "validation"; HOLDOUT = "holdout"; GATE = "gate"
@dataclass(frozen=True, slots=True)
class WindowUnlock:
    prereg_id: int | None = None                          # quant_trials kind='preregistration', config_hash == config_hash(PREREG)
    gate: Literal["g1","calib","reactor","sue"] | None = None
    holdout: bool = False
class WindowLocked(RuntimeError): ...
class MinuteBarLoader:
    def __init__(self, engine: AsyncEngine, *, window: Window, window_end: date, unlock: WindowUnlock,
                 batch_paths: int = 250) -> None
    async def spy(self) -> SpyData
    def paths(self, requests: Sequence[PathRequest]) -> AsyncIterator[PathData | PathSkip]
@dataclass(frozen=True, slots=True)
class PathRequest: story_id: str; symbol: str; session: date; n_sessions: int
@dataclass(frozen=True, slots=True)
class PathSkip:
    story_id: str
    reason: Literal["units_missing","spy_missing","adjust_defect","bad_bars","halted_all_day","no_daily"]
```

- **Coverage.** A path is simulated only if every (symbol, d) and ("SPY", d) unit is in `minutes.done_units` (zero items allowed).
  - A SPY session counts as complete if it has ≥ 300 bars (≥ 150 on early closes).
  - A done unit with 0 bars on S, together with no daily bar on S → `halted_all_day`. That is not a data skip.
  - A done unit with 0 bars on S while a daily bar exists → `units_missing` (a data error).
  - More than 5 bars dropped by sanity in a path → `bad_bars`.
- **Window guard:**
  - every bar read must be ≤ `window_end`, so a path never spills into the next window;
  - sessions in [2016-10-01, 2024-12-31] need `prereg_id` (checked against the database);
  - sessions ≥ 2025-01-01 need `holdout=True` and a `kind='verdict'` row with `verdict='pass'` under the same hash;
  - `gate=` unlocks admit only their registered unit set (its sha256 is checked): g1 and calib are 2016-01-04..2016-09-30; reactor is the R0 pinned headline set; sue is the Σ_c units.
- **Calendar.** At start the loader asserts that `market_hours` sessions over the window equal SPY's raw `daily_bars` sessions.

### D.10 Driver

```python
@dataclass(frozen=True, slots=True)
class SimConfig:
    feed: FeedProfile = SIP_RT
    order_lag: timedelta = ORDER_LAG
    gap: timedelta = GAP
    cost: Literal["study","study_x1.5","study_x2","surcharge"] = "study"
    reference_notional: float = 10_000.0
    allowed_kinds: frozenset[OrderKind] = frozenset({OrderKind.MARKET})

def simulate_symbol(symbol: str, stories: Sequence[Story], make_playbook: Callable[[Story], Playbook],
                    paths: Mapping[str, PathData], spy: SpyData, ctx: PitContext, cfg: SimConfig, *,
                    stop_at: Literal["entry","end"] = "end", assume_full_hold: bool = False,
                    keep_transitions: bool = False) -> list[StoryOutcome]
async def run(engine: AsyncEngine, stories: Sequence[Story], make_playbook: Callable[[Story], Playbook], *,
              window: Window, window_end: date, cfg: SimConfig, unlock: WindowUnlock, sink: OutcomeSink,
              workers: int = 6, stop_at: Literal["entry","end"] = "end", assume_full_hold: bool = False) -> RunSummary
```

- **Per-symbol order.** Each symbol's stories run in session order inside one worker, partitioned by `crc32(symbol) % workers`. Output is byte-identical for any worker count.
- **Overlap.** Only one playbook per symbol may be WATCHING, ARMED, ENTERING or ENTERED. A story whose `nsn_at` falls while one is live is `blocked_open`; its items are delivered to the live playbook as `NewsIn` and can only trigger the structural abort.
- **Stage A.** `stop_at="entry"` stops at the fill and evaluates no exit. `assume_full_hold=True` blocks the symbol through the deadline session, so MD3 counts are a lower bound that uses no exit information.

### D.11 Records

```python
@dataclass(frozen=True, slots=True)
class TradeRecord:
    run_id: str; story_id: str; symbol: str; family_type: str; cell: str; variant: str; feed: str
    session: date; exit_session: date; sessions_held: int; start_case: str
    at_news: datetime; nsn_at: datetime; anchor_ts: datetime
    entry_decided_at: datetime; entry_active_at: datetime; entry_bar_ts: datetime
    exit_decided_at: datetime; exit_active_at: datetime; exit_bar_ts: datetime | None
    p0: float; spy0: float; sigma: float; thr: float; low_star: float; target: float   # S units
    entry_px: float; exit_px: float; adj_entry: float; adj_exit: float                  # raw prices and A
    spy_entry_px: float; spy_exit_px: float; spy_adj_entry: float; spy_adj_exit: float
    cost_bps: float; rank: int; tech: bool; beta: float
    r_gross: float; r_spy: float; r_net_abn: float; r_beta_adj: float
    exit_reason: Literal["target","stop","abort","compliance","time_stop"]
    mae: float; mfe: float; hold_minutes: int; participation: float
    flags: tuple[str, ...]          # gap_fill, late_open, close_fallback, no_market, unresolved
@dataclass(frozen=True, slots=True)
class StoryOutcome:
    story_id: str; symbol: str; session: date; terminal_state: str; reason: str
    skip: str | None; triggered_at: datetime | None; armed_at: datetime | None
    trade: TradeRecord | None; transitions: tuple[tuple[datetime, str, str, str], ...]
```

### D.12 Daily book and daily legs (for the ledger and the Newey-West series)

- A trade's **daily leg** on each held session runs from `a` (entry fill, or the previous official close) to `z` (exit fill, or that session's official close), in S units.
- `leg = z/a − 1 − (Z/A − 1)`, where Z and A are SPY at the same bars or closes. Minus c on the entry day and minus c on the exit day.

```python
@dataclass(frozen=True, slots=True)
class DailyBook:
    days: list[date]; returns: NDArray[np.float64]; benchmark: NDArray[np.float64]
    exposure: NDArray[np.float64]; skipped_full: int; skipped_loss_limit: int
def daily_book(trades: Sequence[TradeRecord], legs: Mapping[str, Sequence[Leg]], sessions: Sequence[date], *,
               slots: int = 8, daily_loss_limit: float = 0.02) -> DailyBook
```

- **Stake.** NAV_{d−1}/8 per trade, first come first served by `entry_decided_at`.
- **Rejections.** A trade is rejected if 8 positions are open (`skipped_full`), or if the realised P&L of trades closed earlier that day is ≤ −2% of NAV_{d−1} (`skipped_loss_limit`; open positions' marks are not included, which is stated).
- **Series.** `returns` and `benchmark` are stake-weighted stock and exposure-matched SPY legs. Days without a position are 0. The days run from the window start to the last exit.
- Eight slots is halabot's `max_open_positions`; the legacy bot's cap is 5.

### D.13 Stated limitations

- There are no quotes, so spreads are modelled.
- Historical bars are final, while live bars can be revised.
- Halts look like illiquid gaps.
- No partial-fill model: the participation share is recorded and bounded by a sensitivity.
- No trading outside the regular session.
- Some `close_fallback` fills use the official close.

---

## E. Phase 0 gate

All seeds are `20261010`. Each gate writes `record_trial(name="research.news.sim-gate", kind="gate", config={"sim": SIM_CONSTANTS, "gate": id}, verdict=pass|fail)`. These rows carry no `active_sr_period`. Re-running a gate after a bug fix is not a trial, but every run is logged.

### E.0 Data preconditions (counts only; all before registration)

| Id | Requirement |
|---|---|
| D1 | `market_hours` sessions 2016-01-04..2026-10-09 equal SPY raw `daily_bars` sessions as sets |
| D2 | `delisted.map_unmapped` and `rescreen_mapped` have run. For every screen 2016-09-30..2024-12-31, rows with `cik IS NULL` and status ≠ `fund` are ≤ 3% of non-fund rows. The residual per quarter is recorded in the registration |
| D3 | `TICKER_RENAMES` is applied and renamed news is backfilled. For every quarter 2016Q4..2024Q4, PIT-halal rank<1000 names with zero admitted news items are ≤ 2%. The residual names are listed in the registration |
| D4 | Every unit in plan H is done: ≥ 99% of each part, SPY 100%, and SPY sessions complete as in D.9 |
| D5 | `read`, `read_sessions` and `read_windows` never return a bar outside [open, effective close): a unit test plus a full scan |
| D6 | v4 facts extracted. The parse rate on 2016-10..2021-12 headlines is reported: the share of 2.02 stories left `earnings_unparsed` |
| D7 | Headline edit rate: live rows from 2026-10-08 onward against an Alpaca refetch by id. Reported, not gating |
| D8 | IEX history probe: 20 units in 2024 with `feed=iex`. Availability is recorded, not gating |
| D9 | `events stories counts`: stories by close type and by `nsn_at`-NSN, × year × window, PRIMARY and Tech. Recorded in the registration (counts only) |

### E.1 G1: mechanics and look-ahead

**Known-answer tests** on synthetic paths, exact to 1e-12 on entry bar, exit bar, prices and r:

- target; stop; no reclaim, which expires at the cutoff; market break; structural abort; unclear veto before entry;
- re-detection when NSN arrives late; follower that turns non-follower;
- MD3 overnight gap through L*; ex-dividend on S (P0 via A); 2:1 split on S+1 (A-ratio); stale-adjustment skip;
- early close (12:00 cutoff, 12:55 flatten); 5-minute gap fill at the open; late open;
- entry unfilled at close; exit `close_fallback`; `no_market`; `unresolved`;
- compliance exit at S+1; `blocked_open`; SPY identity (buy at the 09:31 bar rule, sell at the flatten rule; this is a unit test, not a gate).

**Look-ahead invariance:**

- **Inputs:** 1,000 synthetic paths, plus 500 real stories with S ∈ [2016-01-04, 2016-09-30] (liquidity-only universe rank<1000, price ≥ 5; NSN_CORE prefix stories first, then other negative types run with the family check disabled).
- **Procedure:** for 5 random decision times T per story:
  - scale every bar visible after T by U(0.5, 1.5), one factor per bar for o/h/l/c/vw, with volume × U(0.5, 1.5);
  - delete every item with `available_at > T`;
  - replace with noise the `DailyPoint` open and close of sessions ≥ S (A-ratios kept);
  - replay.
- **Pass:** every intent and transition at or before T, and the set of stories started by T, are identical.
- **Asserted in code:** every fill bar's `ts ≥ active_at`.

**Determinism:** 1 worker and 6 workers give identical sha256 over sorted records.

### E.2 G2: reactor replication (reference: n = 7,922, −0.30%, t −12.1, iid)

- **R0, pin.** Add `scored_before` to `intraday.first_in_session` and use `2026-10-10T00:00:00Z`; this must give 30,614 headlines with 7,932 at score ≥ 0.4. Run the legacy `intraday.run` with a market stub that raises on any fetch.
  - Must reproduce: strong group n = 7,922, mean −0.30% (2 dp), t −12.1 (1 dp); controls −0.28%; score ≤ −0.4 group −0.40%.
  - A mismatch fails until it is explained.
- **R1, exactness.** The simulator with `HARNESS`, a 60 s news-lag override and `LegacyReactorFill`:
  - entry at the VWAP (or open) of the first bar with ts ≥ published + 60 s, with no clamp, no gap rule and no stale limit;
  - exit at the last regular bar's close;
  - the same cost (`universe_at(month, 3000)`, two sides) and the same `_PLAUSIBLE` filter;
  - must match every headline in all three groups to |Δr| ≤ 1e-10, with an identical dropped set.
- **R2, realistic fills.** SIP_RT, `ORDER_LAG`, the D.5 market rule, flatten at close − 5 min, news lag 60 s.
  - The strong group's mean must lie inside the legacy 95% date-clustered interval `m ± t_{0.975, G−1}·SE_CR1`.
  - The clustered t and a per-headline decomposition (entry rule Δ, exit rule Δ) are reported. The clustered t replaces the iid t in the docs.
  - A failure blocks registration until a bug is found through the decomposition. If none is found, the operator's decision is logged as a `gate` row.
- **Holdout note.** These bars are in 2025-12..2026-10, inside the holdout. That sub-period is declared non-confirmatory (§G.14).

### E.3 G3: SUE drift replication (2016-19)

**The SUE complement Σ_c.** SUE observations from 2016-01-04..2019-12-31 whose symbol at the reaction session has rank < 1000 and is **not in BROAD** (§C.1). Events before 2016-10 have no screen, so all of them are in the complement. H1's own universe is never read by this gate.

- **S0, reference.**
  - `events study sue --start 2016 --end 2019 --by bucket` reproduces the recorded table (ICs at 5 and 20 days per bucket to 3 dp; mid-cap 20-day D10−D1 +1.98%). If the data vintage changed, the new values are documented and become the record.
  - Then on Σ_c in daily mode: D10−D1 and IC at h ∈ {5, 20}, with 95% date-cluster bootstrap CIs (resample event dates, B = 2,000).
- **S1, daily-mode exactness on Σ_c.** `DailyBarSource` gives an 09:30 pseudo-bar at the adjusted open and a last pseudo-bar at the adjusted close, with clock `study.entry_point`.
  - Equal to `study.evaluate` within 1e-10 for every observation and h ∈ {1, 5, 20, 60}; identical n and DecileRows.
  - Events on early-close days published between the early close and 16:00 are excluded and counted; that is `entry_point`'s look-ahead.
- **Calibration.** 2,000 random non-event (symbol, session) pairs, rank < 1000, sessions 2016-01-04..2016-08-31. Compute d = minute − daily for a 09:30-open entry and an h=5 close exit, and take the 99th percentile of |d| as `p99_cal`.
- **S2, minute mode on the study's clock.** On a 4,000-event seeded sample Σ_s ⊂ Σ_c: entry at the open of the 09:30 bar or the close of the last bar, exit at the last bar's close, with A-ratios. The paired `d = r_minute − r_daily` must pass all of:
  - the 90% date-clustered CI of mean(d) lies within ±0.10% (TOST at α = 0.05);
  - median |d| ≤ 0.10%;
  - p99 |d| ≤ max(1.00%, 1.5 × p99_cal).
- **S3, realistic fills on Σ_s.** Entry by the D.5 market rule at the first bar with ts ≥ max(published + 600 s, 09:30); exit at the deadline-session flatten rule. At h ∈ {5, 20}, the D10−D1 and IC point estimates must lie inside the S0 daily-mode 95% CI recomputed on Σ_s, with the same sign.

---

## F. Path atlas (`src/halal_trader/events/atlas.py`; Phase 1, descriptive only)

```python
@dataclass(frozen=True, slots=True)
class AtlasRow: ...      # one per story, fields below
@dataclass(frozen=True, slots=True)
class AtlasCell: table: str; key: tuple[str, ...]; n: int; dates: int; per_year: float; stats: dict[str, float] | None
async def build_atlas(engine: AsyncEngine, *, start: date = date(2016, 10, 3), end: date = date(2021, 12, 23)) -> list[AtlasCell]
```

**When and where it runs:**

- Only after `research.news.h1.verdict` exists, or after Stage A has recorded `fail: insufficient events`.
- It asserts `end ≤ 2021-12-23`: that is the last session whose 5-session continuation ends by 2021-12-31. It reads no bar after 2021-12-31.
- It is never extended to validation or holdout.
- It writes `data/research/news_atlas-stories-v1.json` (git-ignored) and prints the tables.

**Unit.** Every PRIMARY story S ∈ [2016-10-03, 2021-12-23] that has an item type outside {noise, law_firm, mover, other, analyst_other}.

- Cells are keyed by **`card_at(detect_at).type`**. `type_close` and `family_ever` are columns.
- `P0`, the anchor and σ follow §G.4, using `detect_at` and its item's `at` when the story is not NSN.

**Per-story measures** (S units, abnormal to SPY at the same bars):

- `gap_σ = D(anchor bar)/σ`
- `low_σ = min D_t / σ` over S
- `t_low` = minutes from the anchor to the bar that set L
- `retrace_close = (C_last − L)/(P0 − L)`
- maximum retrace by the S+2 close
- `fade` = a new low below L by S+2 after a retrace ≥ 0.25
- `cont_h` = abnormal close-to-close from S's close for h ∈ {1, 3, 5}, from adjusted daily bars
- H1 machine fields: triggered, armed, entered, exit reason, `r_ID`, `r_MD3`. These are computed for every negative-direction type by running `OverreactionBounce` with the family check disabled.

**Tables:**

- **Base:** type × `low_σ` bucket. Buckets: (−∞, −5], (−5, −3], (−3, −2], (−2, −1], (−1, ∞).
- **Marginals,** each crossed with type only:
  - timing: pre-open, in-session, previous evening or weekend;
  - rank < 300 vs 300-999;
  - SPY 20-day realised-volatility tercile (edges from SPY 2016-10..2021-12);
  - SPY above or below its 200-day SMA;
  - screen regime before or after 2020-10-01;
  - analyst-coverage regime before or after 2018-01-01;
  - Tech or other.
- **Each cell reports:**
  - n, distinct dates, per year;
  - quantiles 10/25/50/75/90 of `low_σ`, `t_low`, `retrace_close` and the S+2 retrace;
  - P(retrace ≥ 0.25 / 0.5 / 1.0) by the close and by S+2;
  - P(fade | retrace ≥ 0.25);
  - mean `cont_h` with CR1 SE;
  - P(trigger), P(entry | trigger), exit-reason shares, mean r by exit reason with CR1 SE.
- **Rules:**
  - cells with n < 30 or fewer than 20 dates show counts only;
  - no p-values, no ranking of cells, no sweep of H1 constants;
  - any idea taken from the atlas becomes a new registered trial, fitted on train and tested once on validation.

---

## G. H1 pre-registration (`src/halal_trader/events/h1.py`)

*Docstring: "Pre-registered (written before any result was seen)."*

### G.1 Hypothesis

After a large drop on non-structural negative news (an analyst downgrade or an earnings miss, NSN_CORE), a long position bought when the selling is exhausted earns a positive mean return net of costs, abnormal to SPY at the same bars. Exhaustion means no new low for 20 minutes, a reclaim of the anchored VWAP, and at most a quarter of the drop retraced. The position exits at a 50% retrace target, a close below the capitulation low, a structural abort, a compliance exit or a time stop.

- **Test:** one-sided. H0: E[r] ≤ 0.
- **Cells (literal):**

  ```python
  CELLS = (Cell("NSN_CORE", "ID", 1), Cell("NSN_CORE", "MD3", 3))
  ```

- **Secondary (Technology) subgroup:** reported only, never gating.

### G.2 Universe (per reaction session S)

PRIMARY means all of:

- `verdict == 'halal'` in `halal_screen_current` at the newest `as_of < S`, with `cik IS NOT NULL`;
- 0-based index < 1000 in `universe_at(engine, S, top_n=3000)`;
- `prev_close_s ≥ 5.0`;
- one share class per CIK (the lower rank);
- σ available (≥ 40 of 60 sessions).

Screens start at 2016-09-30, so the first S is 2016-10-03.

### G.3 Events

- **Stories:** `stories-v1` stories with `nsn_at(entry_cutoff(S))` not None.
- **Times:** `at_news` is the `at` of the item that made the story NSN. `start_case` is "in" iff `at_news ∈ [open(S), close(S))`.
- **Reference prices:**
  - **"in":**
    - P0 = the close of the last bar of S with `ts + 60 s ≤ at_news`, or `prev_close_s` if there is none;
    - SPY0 = SPY's close at the same rule;
    - anchor = the first bar with `ts ≥ floor_minute(at_news)`.
  - **"out":**
    - P0 = `prev_close_s`;
    - SPY0 = `spy_prev_close_s`;
    - anchor = S's first bar.
- **Volatility and threshold:** σ from §C.1 (sessions closed strictly before `at_news`); `thr = max(2.0·σ, 0.03)`.

### G.4 Rule (`src/halabot/playbooks/bounce.py`; constants frozen)

```python
@dataclass(frozen=True, slots=True)
class BounceParams:
    k_sigma: float = 2.0; floor: float = 0.03
    quiet: timedelta = timedelta(minutes=20)
    entry_start_after_open: timedelta = timedelta(minutes=20)
    entry_cutoff_before_close: timedelta = timedelta(minutes=60)
    max_retrace_at_entry: float = 0.25
    target_retrace: float = 0.50
    market_break: float = -0.02
    flatten_before_close: timedelta = timedelta(minutes=5)
    hold_sessions: int = 1                     # 1 = ID, 3 = MD3
class BounceState(StrEnum): DETECTED, WATCHING, ARMED, ENTERING, ENTERED, EXITING, EXITED, EXPIRED, DISMISSED
class OverreactionBounce(Playbook):
    def __init__(self, story: Story, pre: PreEvent, elig: Eligibility, params: BounceParams) -> None
```

All quantities are in S units, evaluated on visible bars from the anchor onward.

- `D_t = (c_t/P0 − 1) − (spy_c_t/SPY0 − 1)`, where `spy_c_t` is SPY's bar with the same start, or SPY's latest bar with start ≤ ts_t.
- `L_t = min l_b` over [anchor, t]; `t_L` is the latest bar attaining it.
- `AVWAP_t = Σ w_b v_b / Σ v_b` over [anchor, t], with `w_b = clamp(vw_b, l_b, h_b)`, or `(h+l+c)/3` when vw is null.

**Transitions:**

1. **DETECTED → WATCHING** at `max(nsn_at, open(S))`, if the symbol is eligible and no playbook on it is live. Otherwise DISMISSED, with the reason (`blocked_open` or an eligibility reason).
2. **Triggered** once some bar in [anchor, t] has `D_b ≤ −thr`.
3. **WATCHING → ARMED** when triggered and `ts_t − ts_{t_L} ≥ 20 min`. **ARMED → WATCHING** when a new low prints.
4. **ARMED → ENTERING** at the visible time T of bar t, when all of these hold:
   - E1: T ∈ [open + 20 min, close − 60 min] (09:50-15:00, or 09:50-12:00 on early closes);
   - E2: `c_t > AVWAP_t`;
   - E3: `(c_t − L_t) ≤ 0.25·(P0 − L_t)`;
   - E4: no SPY bar close of S so far is ≤ 0.98·`spy_prev_close_s`;
   - E5: `card_at(T).family == "NSN_CORE"`.

   Then submit a market buy. Freeze `L* = L_t` and `TGT = L* + 0.5·(P0 − L*)`. E3 implies reward ≥ risk.
5. **ENTERING → ENTERED** on the fill. If the order is unfilled by close → EXPIRED (`entry_unfilled`).
6. **ENTERED → EXITING** on the first of these. When several fire on one bar, priority is X3 > X1 > X2.
   - X3 abort: an item of a **structural** type with `available_at ≤ now`;
   - X1 stop: a bar close below L* (`c_u < L*`, in S units);
   - X2 target: `c_u ≥ TGT`;
   - X5 compliance: at `pre_open` of S+1 or S+2, the screen is not halal;
   - X4 time stop: close − 5 min of session S (ID) or S+2 (MD3).

   Every exit is a market sell under D.5.
7. **WATCHING / ARMED → EXPIRED** on any of:
   - the entry cutoff (`cutoff`);
   - a SPY market break (`market_break`; it expires every WATCHING and ARMED story that session);
   - a card that is no longer NSN_CORE (`veto`).
8. One entry per story. No re-entry.

Nothing else is used: no volume rule, no trailing stop, no LLM. Those are Phase 2 trials.

### G.5 Windows

- **Train:** S ∈ [2016-10-03, 2021-12-31] for ID and [2016-10-03, 2021-12-29] for MD3; 5.25 years; 1,322 sessions.
- **Validation:** S ∈ [2022-01-03, 2024-12-31] for ID and [.., 2024-12-27] for MD3; 3.0 years; 753 sessions.
- **Holdout:** 2025-01-02 onward, untouched.
  - 2025-12-01..2026-10-30 is declared **non-confirmatory**, because the reactor and LLM studies observed outcomes there.
  - The clean holdout is 2025-01-02..2025-11-28 plus forward shadow.
  - No minute bar ≥ 2025-01-01 is fetched for H1 before a pass.

### G.6 Metric (per trade)

```
r_net_abn = (P_x·A(d_x))/(P_e·A(S)) − 1 − [(Q_x·B(d_x))/(Q_e·B(S)) − 1] − 2·c/10⁴
```

- P and Q are the stock and SPY fill prices by D.5 at the same bar starts; A and B are their A-factors; c is from §D.6.
- Equal weight. No winsorising. No plausibility drop.
- An `unresolved` trade is marked at its last trade and kept.

**Robustness quantities:**

- `r_beta_adj = (P_x A_x)/(P_e A_S) − 1 − β·[SPY gross] − 2c/10⁴`, with β from §C;
- `r_cost15 = r_net_abn − c/10⁴`.

### G.7 Overlap

- One live playbook per symbol (§D.10). A blocked story is `blocked_open`; its items can only abort the live playbook.
- One trade per story.
- Several names entering on the same day stay in; date clustering handles the dependence, and MD3 also uses calendar-time Newey-West.

### G.8 Statistics (`src/halal_trader/events/stats.py`, pure)

```python
@dataclass(frozen=True, slots=True)
class ClusteredMean: n: int; clusters: int; mean: float; se: float; t: float; df: int; p_one_sided: float; deff: float
@dataclass(frozen=True, slots=True)
class NWMean: n: int; mean: float; se: float; t: float; df: int; p_one_sided: float
def clustered_mean(values: Sequence[float], clusters: Sequence[Hashable]) -> ClusteredMean | None
def newey_west_mean(series: Sequence[float], *, lags: int = 4) -> NWMean | None
def calendar_series(legs: Mapping[str, Sequence[Leg]], sessions: Sequence[date]) -> list[tuple[date, float]]
def holm(p: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, bool]
def cluster_bootstrap_ci(stat: Callable[[Sequence[int]], float], clusters: Sequence[Hashable], *,
                         b: int = 2000, seed: int = 20261010, level: float = 0.95) -> tuple[float, float]
def tost(values: Sequence[float], clusters: Sequence[Hashable], *, margin: float = 0.001, alpha: float = 0.05) -> bool
```

- **CR1, clustered by entry session:**
  - `r̄ = Σr_i/N`; `E_g = Σ_{i∈g}(r_i − r̄)`;
  - `V = [G/(G−1)]·Σ_g E_g² / N²`; `t = r̄/√V`; df = G − 1;
  - `p = sf/2` for t ≥ 0 and `1 − sf/2` otherwise, with `sf = halabot.analysis.significance.student_t_sf_two_sided(t, df)`;
  - `DEFF = V/(s²/N)`.
- **Newey-West (MD3 only):**
  - `x_d` = the equal-weighted mean of the open trades' daily legs, over sessions with at least one leg (T of them);
  - `γ_k = Σ_{d>k} e_d e_{d−k} / T`; `V = [γ_0 + 2Σ_{k=1..4}(1 − k/5)γ_k]/T`; df = T − 1.
- **Holm:** sort p ascending; reject while `p_(k) ≤ α/(m−k+1)`; stop at the first non-rejection. For MD3, p = max(p_CR1, p_NW).

### G.9 Stage A (counts only; no exit or return is computed)

- **Run:** both windows with `stop_at="entry"` and `assume_full_hold=True`.
- **Record** `kind="stage-a"` with, per cell and window:
  - stories, eligible, `nsn_at`-NSN, triggered, armed, entries N, entry dates G, `blocked_open`;
  - expiry reasons;
  - every skip by reason.
- **Data budget.** Data skips (`units_missing`, `spy_missing`, `adjust_defect`, `bad_bars`) must be ≤ 2% of eligible NSN stories in each window. Otherwise the data is fixed and Stage A rerun (logged, not a trial). Nothing after Stage A runs until this holds. Data is frozen here.
- **Eligibility of a cell:**
  - validation N/3.0 ≥ 200 entries a year and G ≥ 100;
  - train N ≥ 500 and G ≥ 100.
- **If no cell is eligible:** H1 = `fail: insufficient events`, no returns are computed in either window, and the atlas runs.

### G.10 Pass rule

- **Train,** eligible cells only. A cell passes iff:
  - **T1:** r̄ > 0 and t_CR1 ≥ 2.0; for MD3, also t_NW ≥ 2.0 with a positive calendar mean;
  - **T2:** rejected by Holm at one-sided α = 0.05 across the eligible cells;
  - **T3:** mean `r_cost15` > 0;
  - **T4:** mean `r_beta_adj` > 0;
  - **T5:** mean > 0 after excluding entries 2020-02-20..2020-06-30;
  - **T6:** unresolved ≤ 0.5% of entries; otherwise the window is `inconclusive`.
- **Validation,** train passers only; the other cells never compute 2022-24 returns. A cell passes iff T1, T3, T4 and T6 hold, and it is rejected by Holm across the train passers.
- **Verdict:**
  - **pass** iff at least one cell passes both windows;
  - **inconclusive** iff T6 fails in a window that would otherwise decide. A data fix after returns exist is a logged amendment, with both results reported;
  - **fail** otherwise.

### G.11 Regimes

- **Gating:** T5 only.
- **Reported, not gated:**
  - per-year means with CR1 SE;
  - before vs after the screen break (2020-10-01);
  - before vs after the analyst-template change (2018-01-01);
  - downgrade vs miss;
  - "in" vs "out" starts.

### G.12 Sensitivities

Recorded with `record_trial(kind="sensitivity", name="research.news.h1.sens.<name>", config={"prereg": hash, "cell": …, "sensitivity": name})`. They carry no Sharpe, so they never count as trials. They run on train for eligible cells and on validation for train passers only.

- Tech subgroup. It must agree in sign before Tech-only live trading is considered.
- BROAD universe.
- News lag 60 s and 1,200 s.
- Costs × 2, and the surcharge model.
- Rank < 300 vs 300-999.
- Unresolved trades at −100%.
- Trades with participation > 0.10 dropped.

### G.13 Implementability trial

For each cell that passes H1, run `feed=SIP_DELAYED` on both windows. It is a **counted** trial (`record_backtest`) under the same pass rule.

- Phase 4 may use the free delayed feed only if this trial passes.
- Otherwise a real-time SIP subscription, or a separately validated IEX profile, is a precondition for Phase 4.
- IEX bars are never fed to H1.

### G.14 Ledger recording

```python
PREREG: Final[dict[str, object]] = {
  "hypothesis": "news.h1.overreaction_bounce", "version": 1,
  "builder": {"version": "stories-v1", "roundup_max_symbols": 3, "news_lag_s": 600, "react_cutoff_min": 90,
              "filing": "EDGAR business day [06:00,17:30) else next 06:00", "dup_jaccard": 0.60,
              "repost_jaccard": 0.50, "follow_sessions": 3, "alias_sha": ALIAS_SHA},
  "taxonomy": {"version": "rules-v1", "sha": TAXONOMY_SHA, "extractor": EXTRACTOR,  # f"benzinga-earnings-v4+{PARSER_SHA}"
               "parser_sha": PARSER_SHA, "dead_band": 0.005, "guide_band": 0.01},
  "family": {"NSN_CORE": ["analyst_downgrade", "earnings_miss"]},
  "universe": {"screen": "newest as_of < S, verdict halal, cik not null", "rank_lt": 1000, "rank_top_n": 3000,
               "min_prev_close": 5.0, "one_class_per_cik": "lower rank", "sigma_sessions": 60, "sigma_min_obs": 40},
  "playbook": {"k_sigma": 2.0, "floor": 0.03, "quiet_min": 20, "entry_start_min": 20, "entry_cutoff_min": 60,
               "max_retrace_at_entry": 0.25, "target_retrace": 0.5, "market_break": -0.02, "flatten_min": 5,
               "reclaim": "close > AVWAP from anchor", "stop": "bar close < L*", "abort": "structural item",
               "priority": ["abort", "stop", "target"], "compliance_exit": True},
  "cells": [["NSN_CORE", "ID", 1], ["NSN_CORE", "MD3", 3]],
  "fills": {"feed": "sip-rt", "bar_visible_s": 65, "order_lag_s": 3, "orders": ["market/day"],
            "price": "clamped VWAP of first bar ts >= active_at; open after a >=5 min gap",
            "partial": "none", "fallback": ["next session open (MD3)", "official close", "next session", "unresolved"]},
  "costs": "study.COST_BPS one-way, two sides",
  "metric": "net abnormal vs SPY at the same bars, A-ratio adjusted, no drop filter",
  "stats": {"se": "CR1 by entry date, df G-1", "md3_extra": "calendar-time Newey-West 4 lags", "alpha": 0.05,
            "multiple": "Holm one-sided over eligible cells", "t_min": 2.0},
  "count_rule": {"validation_per_year": 200, "validation_dates": 100, "train_n": 500, "train_dates": 100},
  "robustness": {"cost_x": 1.5, "beta_adjusted": True, "train_exclude": ["2020-02-20", "2020-06-30"]},
  "data": {"skip_max": 0.02, "unresolved_max": 0.005},
  "windows": {"train": ["2016-10-03", "2021-12-31"], "validation": ["2022-01-03", "2024-12-31"],
              "holdout": "2025-01-02.. (2025-12-01.. non-confirmatory)"},
  "book": {"slots": 8, "daily_loss_limit": 0.02, "benchmark": "SPY exposure-matched"},
}
CRITERION: Final = (
  "H1 passes iff some cell (NSN_CORE, ID|MD3) that Stage A finds eligible (validation >= 200 entries/yr and >= 100 "
  "entry dates; train >= 500 entries and >= 100 dates) has, on train 2016-10-03..2021-12-31: mean net abnormal return "
  "> 0 with date-clustered CR1 t >= 2.0 (MD3 also calendar-time Newey-West(4) t >= 2.0), Holm rejection at one-sided "
  "alpha 0.05 across eligible cells, mean > 0 at 1.5x costs, beta-adjusted mean > 0, and mean > 0 excluding entries "
  "2020-02-20..2020-06-30; and on validation 2022-01-03..2024-12-31 (train passers only) the same except the 2020 "
  "exclusion, Holm across train passers. Stage-A data skips <= 2% and unresolved exits <= 0.5% per window, else "
  "inconclusive. Nothing is changed after Stage A counts or any result is seen.")

def trial_config(cell: Cell, *, feed: str = "sip-rt") -> dict[str, object]:
    return {"prereg": config_hash(PREREG), "cell": [cell.family, cell.variant, cell.hold], "feed": feed}
async def register(engine: AsyncEngine) -> int
async def stage_a(engine: AsyncEngine, *, workers: int = 6) -> StageA
async def run_window(engine: AsyncEngine, window: Literal["train", "validation"], cells: Sequence[Cell], *,
                     cfg: SimConfig = SimConfig(), workers: int = 6) -> dict[Cell, WindowStats]
async def verdict(engine: AsyncEngine) -> Literal["pass", "fail", "inconclusive"]
async def sensitivities(engine: AsyncEngine) -> None
async def implementability(engine: AsyncEngine) -> None
```

**Rows written:**

| Row | Call | Notes |
|---|---|---|
| Registration | `record_trial(name="research.news.h1", kind="preregistration", config=PREREG, window=<windows>, criterion=CRITERION, verdict=None, metrics={code shas, git tag, D2/D3 residuals, D9 counts})` | The runner refuses to run unless it exists |
| Stage A, verdict | `kind="stage-a"` and `kind="verdict"`, both under `config=PREREG` | That hash is not a trial and has no Sharpe, so `_trial_sharpes` ignores it |
| Train and validation, per cell | `record_backtest(strategy=f"news.h1.nsn_core.{variant}", config=trial_config(cell), days, returns, benchmark, benchmark_label="SPY (exposure-matched)", extra={"window_role", "trades", "dates", "t_cr1", "t_nw", "mean_trade", "mean_cost15", "mean_beta", "mean_ex_covid", "pass", "skipped_full", "skipped_loss_limit", "exposure"})` | Both windows share one hash, so each cell is one trial |

- **DSR** is recorded for every trial. It does not gate Phase 1 (it gates Phase 2 at ≥ 0.95). It is also reported at `n_trials + 6` for the six roadmap §1 tests.
- **Code pins.** The SHAs of pinned files go in the registration's `metrics`, not in its hash.
  - A post-registration change to `taxonomy.py`, `stories.py`, `aliases.py` or `earnings_parse.py` that changes their pattern SHAs means a new registration (a new trial).
  - A simulator or loader bug fix is a logged amendment (`kind="amendment"`), with old and new results both reported.

### G.15 Order of work and locks

1. Code frozen and tagged `news-h1-v1`.
2. Data preconditions D1-D9.
3. Phase 0 gates G1-G3.
4. `register`.
5. Stage A.
6. Train, eligible cells only.
7. Validation, train passers only.
8. Verdict.
9. Implementability and sensitivities.
10. Atlas.

The loader (§D.9) enforces 4 before 5-7 and 8 before any holdout read. Between registration and the verdict, nobody reads 2016-24 minute bars of PRIMARY names outside the runner.

### G.16 Prior and planning (stated, not measured)

- **Prior.** Chan (2003) and Savor (2012) find continuation, not reversal, after news-driven drops. The repo's lexicon test (worst decile at 5 days, −0.29%) agrees. The prior runs against H1, especially MD3.
- **Count risk.** The prototype counted about 625 NSN_CORE stories a year in train and about 564 in validation, before any price condition. Reaching 200 entries a year therefore needs about 35% of validation stories to trigger and enter. Stage A may well fail, and that is an accepted answer. No threshold is relaxed after Stage A.
- **Power** (assumed s and DEFF): t = 2 needs `r̄ ≈ 2·s·√(DEFF/N)`. For ID with s = 3%, DEFF 1.2 and N = 600: about 0.27%.

---

## H. Minute backfill unit selection (`src/halal_trader/events/units.py`, built on `data/minutes.backfill`)

```python
@dataclass(frozen=True, slots=True)
class UnitPlan:
    parts: Mapping[str, frozenset[tuple[str, date]]]   # spy, train, validation, gate_g1, gate_sue, gate_calib
    def all(self) -> frozenset[tuple[str, date]]
    def sha(self, part: str) -> str                     # pinned for the WindowUnlock gate sets
async def h1_plan(engine: AsyncEngine, *, seed: int = 20261010) -> UnitPlan
def path(session: date, n: int, cap: date) -> list[date]   # n sessions from S via market_hours, truncated at cap
```

**Selection inputs.** Only stories-v1 news and 8-K items, screens, and `monthly_bars` liquidity: data available at or before S. No price on or after S selects anything. The set is a superset of every path H1, Stage A, its sensitivities, the implementability trial, the atlas and the gates can read, so selection cannot depend on outcomes. A unit that fails to fetch is counted as a Stage A data skip (≤ 2%).

| Part | Rule | Estimated units |
|---|---|---|
| `spy` | ("SPY", d) for every session 2016-01-04..2024-12-31 | 2,264 |
| `train` | every story with S ∈ [2016-10-03, 2021-12-31] whose symbol is in **BROAD** at S (§C.1; PRIMARY ⊂ BROAD) and that has an item type outside {noise, law_firm, mover, other, analyst_other}: `path(S, 3, 2021-12-31)` | about 60-75k |
| `validation` | every story with S ∈ [2022-01-03, 2024-12-31], symbol in BROAD at S, with `card_at(t).family == "NSN_CORE"` at some item time t ≤ entry cutoff of S: `path(S, 3, 2024-12-31)` | about 5-8k |
| `gate_g1` | 500 seeded stories S ∈ [2016-01-04, 2016-09-30], rank < 1000 at S, NSN_CORE-prefix first: `path(S, 3, 2016-09-30)` | about 1.5k |
| `gate_sue` | Σ_s, the 4,000-event seeded sample of Σ_c: entry session, and the exit sessions for h = 5 and h = 20 (`exit_i = i + h − 1` for open entries, `i + h` for close entries) | about 12k |
| `gate_calib` | 2,000 seeded (symbol, session) pairs, rank < 1000, sessions 2016-01-04..2016-08-31: the session and session + 4 | about 4k |

- **Total:** roughly 85-105k unique units, about 4-5 GB at 45.8 KB per unit. Grouped by session, with about 25 symbol-sessions per 10,000-bar page, that is about 5-6k requests, roughly 1 hour at 100 requests a minute. `halal-trader data minutes --plan h1 --dry-run` prints the exact counts first.
- **Fetch order:** spy, gate_g1, gate_calib, gate_sue, train, validation. Each unit is marked done only once settled, zero bars included.
- **When:** outside market hours and outside the 20:30-23:30 ET research job, at `--rate 100`.
- **Excluded:**
  - every session ≥ 2025-01-01 (the reactor gate uses the already-stored 2025-12-04..2026-10-08 bars);
  - pre-market and after-hours bars;
  - no price-based prefilter is used: the atlas needs non-triggering stories, and the volume is small.

---

## Appendix: changes to existing code and the CLI

- `events/intraday.first_in_session(engine, scorer_prefix="llm-batch:", *, scored_before: datetime | None = None)`
- `research/ledger.record_backtest(..., benchmark_label: str = BENCHMARK)`: the window string uses the label.
- `data/minutes.read_windows(...)` (§D.9).
- `events/earnings_parse.py` v4 patterns, `EXTRACTOR = f"benzinga-earnings-v4+{PARSER_SHA}"`; `drop_superseded` deletes the other v4 labels' facts (the evening refresh; `events extract --drop-superseded`; `books run --drop-superseded`).
- `sentiment/stocks_events.py:548`: the live payload stores `symbols`.
- New modules:
  - `events/{stories,aliases,renames,taxonomy,context,stats,units,h1,atlas,sim_gate}.py`
  - `halabot/playbooks/*`
- New tables:
  - `news_stories` and `story_aliases`: Alembic migration plus models, compared by `test_alembic_migrations.py`;
  - `hb_playbook_*`: `bootstrap_schema`.
- **Backups:**
  - `news_stories`: excluded;
  - `story_aliases`: included;
  - `hb_playbook_*`: only rows with `mode != 'sim'`.
- **CLI** (`halal-trader`, lazy imports, `run_db`):
  - `data minutes --plan h1 [--part …] [--dry-run] --rate 100`
  - `events renames {seed,backfill}`
  - `events stories {build,counts}`
  - `events sim-gate {lookahead,reactor,sue}`
  - `events h1 {register,stage-a,train,validation,verdict,sensitivities,implementability}`
  - `events atlas`
- **Commits:** code and its tests in separate commits, test commit after the code. Migrations, regex tables and docs each get their own commit.

**Sources read:** docs/NEWS_ENGINE_ROADMAP.md, the design panel's reports on the code, and the modules the spec names.
