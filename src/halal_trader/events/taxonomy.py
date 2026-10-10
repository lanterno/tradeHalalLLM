"""Rules taxonomy v1: what one news item or 8-K is, and what a story is at a moment.

Two levels, both pure:

* **Items.** :func:`classify_item` types one Benzinga headline or one 8-K
  filing by fixed rules, first match wins: an 8-K by its item codes;
  law-firm alerts and auto-generated filler (``law_firm``, ``noise``);
  headlines the earnings parser read (``earnings_fact``) or could not
  (``guidance_unparsed``); the structural negatives, checked **before**
  analyst actions so "Downgrades On Fraud Concerns" is fraud; analyst
  actions; price-mover pieces (``mover``, written because the price moved,
  so they never name a cause); then the unclear negatives, the positives and
  the neutrals.
* **Stories.** :func:`resolve` types a story (one symbol's items of one
  reaction session) from the items **available by** ``t`` only, so a card
  read at 10:05 never knows the 11:00 wire. A structural negative wins,
  then the earnings verdict (:func:`earnings_verdict`), then the analyst,
  unclear-negative, positive and neutral types in a fixed order.

The H1 family ``NSN_CORE`` (non-structural negative, core) is an analyst
downgrade or an earnings miss with nothing structural or unclear-negative,
nothing positive, not a follower of an earlier story, and not led by a
mover piece.

Every pattern, order and table is pinned by :data:`TAXONOMY_SHA` in the news
engine's pre-registration: a change here is a new trial.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, Protocol

from halal_trader.core.num import to_float
from halal_trader.events import headline_patterns
from halal_trader.events.earnings_parse import GUIDE_UNPARSED, EarningsFacts
from halal_trader.events.headline_patterns import ANALYST_ACTION

RULES_VERSION: Final = "rules-v1"

# Per-metric dead band on a result's surprise, and the band past which
# guidance counts as above or below consensus (relative to the estimate).
DEAD_BAND: Final = 0.005
GUIDE_BAND: Final = 0.01
# Surprises are compared after rounding to this many decimals, so a stated
# "$2.01 vs $2.00" (0.005 exactly, 0.00499... in binary) is on the band.
_SURPRISE_DIGITS: Final = 12

Direction = Literal["neg", "pos", "neutral", "none"]
Permanence = Literal["structural", "unclear", "transient", "none"]
Guide = Literal["down", "up", "inline", "unknown"]


@dataclass(frozen=True, slots=True)
class TypeMeta:
    direction: Direction
    permanence: Permanence


def _meta(direction: Direction, permanence: Permanence, *names: str) -> dict[str, TypeMeta]:
    return dict.fromkeys(names, TypeMeta(direction, permanence))


# Every item and story type (spec §B.1).
TYPES: Final[dict[str, TypeMeta]] = (
    # Structural negatives: veto before entry, abort after.
    _meta(
        "neg",
        "structural",
        "restatement",
        "insolvency",
        "delisting",
        "short_report",
        "fraud_probe",
        "dilution",
        "guidance_cut",
        "regulatory_block",
        "ma_break",
        "customer_loss",
        "impairment",
    )
    # Unclear negatives: veto before entry only.
    | _meta(
        "neg",
        "unclear",
        "antitrust_regulatory",
        "management_exit",
        "legal_adverse",
        "operational_incident",
        "auditor_change",
        "agreement_terminated",
        "sympathy",
    )
    # Transient negatives: the H1 candidates.
    | _meta("neg", "transient", "analyst_downgrade", "analyst_pt_cut", "analyst_init_neg")
    | _meta("neg", "transient", "earnings_miss")
    # Earnings, story level.
    | _meta("pos", "transient", "earnings_beat")
    | _meta("neutral", "transient", "earnings_inline")
    | _meta("neutral", "unclear", "earnings_miss_guide_up")
    | _meta("neg", "unclear", "earnings_miss_guide_unk")
    | _meta("pos", "unclear", "guidance_raise")
    | _meta("neutral", "unclear", "guidance_inline")
    | _meta("none", "none", "earnings_unparsed")
    # Positives.
    | _meta("pos", "structural", "ma_target")
    | _meta("pos", "transient", "analyst_upgrade", "analyst_pt_raise", "buyback")
    | _meta("pos", "unclear", "regulatory_win", "contract_win", "product")
    # Neutrals.
    | _meta(
        "neutral",
        "unclear",
        "ma_acquirer",
        "restructuring",
        "agreement",
        "ma_closed",
        "management_change",
        "debt_financing",
        "other",
    )
    | _meta("neutral", "transient", "dividend", "analyst_other")
    | _meta("none", "none", "filing_other")
    # Item-only, and the stories made of nothing else.
    | _meta(
        "none",
        "none",
        "noise",
        "law_firm",
        "mover",
        "earnings_fact",
        "earnings_8k",
        "guidance_unparsed",
        "mover_only",
        "noise_only",
    )
)

STRUCTURAL: Final = frozenset(
    {
        "restatement",
        "insolvency",
        "delisting",
        "short_report",
        "fraud_probe",
        "dilution",
        "guidance_cut",
        "regulatory_block",
        "ma_break",
        "customer_loss",
        "impairment",
    }
)
UNCLEAR_NEG: Final = frozenset(
    {
        "antitrust_regulatory",
        "management_exit",
        "legal_adverse",
        "operational_incident",
        "auditor_change",
        "agreement_terminated",
        "sympathy",
    }
)
# Items written in reaction to something else (spec §A.5(b)): a story made of
# nothing else follows an earlier one.
REACTIVE: Final = frozenset(
    {
        "analyst_pt_cut",
        "analyst_init_neg",
        "analyst_pt_raise",
        "analyst_other",
        "mover",
        "guidance_unparsed",
    }
)
# Shown with a story, never its type or its trigger.
NOISE_TYPES: Final[frozenset[str]] = frozenset({"noise", "law_firm", "mover"})

# 8-K item code -> type (spec §B.3). Codes not listed carry no type.
ITEM_TYPES: Final[dict[str, str]] = {
    "4.02": "restatement",
    "3.01": "delisting",
    "2.04": "insolvency",
    "1.03": "insolvency",
    "2.06": "impairment",
    "3.02": "dilution",
    "4.01": "auditor_change",
    "1.02": "agreement_terminated",
    "2.05": "restructuring",
    "2.01": "ma_closed",
    "1.01": "agreement",
    "5.02": "management_change",
    "2.03": "debt_financing",
    "2.02": "earnings_8k",
    "5.03": "filing_other",
    "5.07": "filing_other",
    "7.01": "filing_other",
    "8.01": "filing_other",
    "9.01": "filing_other",
}
FILING_KINDS: Final = frozenset({"8-k", "8-k/a"})

# ── Headline patterns (spec §B.4) ───────────────────────────────────────────

LAW_FIRM: Final = re.compile(
    r"Law Firm|Rosen Law|Pomerantz|Levi & Korsinsky|Bragar|Faruqi|Kessler Topaz|Glancy|"
    r"Bronstein|Schall Law|Robbins (?:Geller|LLP)|Gross Law|Kirby McInerney|Holzer|Portnoy|"
    r"Block & Leviton|Hagens Berman|Bernstein Liebhard|Johnson Fistel|Halper Sadeh|"
    r"Gainey McKenna|Kahn Swick|Monteverde|Ademi|Brodsky & Smith|Rigrodsky|Wohl & Fruchter|"
    r"Frank R\.? Cruz|Howard G\.? Smith|Edelson Lechtzin|Investor Alert|Shareholder Alert|"
    r"Investors? Who Lost|Lead Plaintiff|Class Action (?:Filed|Lawsuit Filed|Reminder)|"
    r"Securities Class Action|Deadline Alert",
    re.I,
)
NOISE: Final = re.compile(
    r"\$\s?\d[\d,]*\s+Invested\b|Would Be Worth|Here'?s How Much|If You (?:Had )?Invested|"
    r"How Much You Would Have|"
    r"Options? Activity|Unusual Options|Option Alert|Options Frenzy|Options Trading|\bWhales?\b|"
    r"Short Interest|Short Volume|"
    r"\bP/E\b|Price Over Earnings|Earnings Preview|^Preview:|Ahead Of (?:Its |Q\d )?Earnings|"
    r"Earnings Scheduled|To Report Q\d Earnings|"
    r"What To Expect|Expected To Report|Likely To Report|Most Accurate Analysts?|"
    r"Forecast Changes From|Analyst Ratings?|"
    r"Here'?s Every Rating|Price Target Changes|Analysts? Assess|"
    r"Insights From \d+ (?:Financial )?Analyst|Stands With Analysts|"
    r"Analyst Verdict|(?:In|Through) The Eyes Of|Analyst Insights|"
    r"(?:Lower|Raise|Slash|Boost|Cut|Trim) Their Forecasts|Peering Into|"
    r"Peeling Back|Demystifying|Expert Outlook|A Look At|Insights Into|Deep Dive|"
    r"In-Depth Analysis|Comparative Study|"
    r"Competitor Dynamics|Compared To Competitors|In Comparison To Competitors|"
    r"Return On (?:Capital Employed|Invested Capital|Equity)|"
    r"Gaining or Losing Market Support|How Is The Market Feeling|"
    r"Market Cap(?:italization)? (?:Of|Rises|Falls)|Stock Whisper|"
    r"Bulls And Bears|52-Week (?:High|Low)s?|Stocks? Moving|Movers|Mid-?Day|"
    r"(?:Pre-?Market|After-?Hours) (?:Gainers|Losers|Movers)|"
    r"Top (?:Gainers|Losers)|Biggest (?:Gainers|Losers)|Stocks To Watch|Final Trades|Weekly:|"
    r"Transcript|Recap|Earnings Summary|"
    r"Earnings (?:Review|Report): Q\d|Earnings (?:Outlook|Perspective|Insights)|"
    r"Analyst Expectations|Key Takeaways|Here'?s Why|"
    r"Should You Buy|Is It (?:Time|A Buy)|Buy The Dip|"
    r"What (?:Investors|You) (?:Need|Should) (?:To )?Know|Technical Analysis|"
    r"\bETFs?\b|Dividend (?:Stocks|Yield)|Benzinga Pro|Benzinga Live|SwingTrader|"
    r"Shares Indicated|IPO (?:Priced|Opens)|\b13F\b|"
    r"Share Stake|Has (?:Sold|Bought) Up To|Worth Of .{1,30} Stock|"
    r"Insider (?:Sells|Sale|Buys|Trades|Selling|Buying)|"
    r"Executive Sells|Director (?:Sells|Buys)",
    re.I,
)
MOVER: Final = re.compile(
    r"\bShares? (?:Are |Is )?(?:Trading|Moving|Moves?|Trade) (?:Lower|Higher|Down|Up)\b|"
    r"\bWhy\b.{1,80}\b(?:Is|Are|Was|Were)\b.{0,30}"
    r"\b(?:Falling|Down|Lower|Plunging|Sinking|Sliding|Tumbling|Dropping|Tanking|"
    r"Crashing|Rising|Up|Higher|Soaring|Jumping|Surging|Rallying|Moving|Climbing|Spiking|Trading)\b|"
    r"\bShares?\s+(?:Down|Up|Falls?|Drops?|Slides?|Sinks?|Rises?|Gains?|Jumps?|Spikes?|Tumbles?|"
    r"Plunges?|Climbs?|Halted)\b|"
    r"\bStock (?:Is )?(?:Falling|Rising|Sliding|Soaring|Tumbling|Plunging)\b|What'?s Going On|"
    r"Halted On Circuit Breaker|"
    r"Session (?:High|Low)|\bTrading (?:Lower|Higher)\b|"
    r"(?:Spikes|Rises|Jumps|Surges) To (?:A )?High Of|"
    r"(?:Sells Off|Falls|Drops|Slides|Dips) To (?:A )?Low Of|Moving (?:Lower|Higher)",
    re.I,
)

# Analyst actions, applied to the clause group of the symbol's company.
DOWNGRADE: Final = re.compile(
    r"\bDowngrade[sd]?\b|\bCut To (?:Neutral|Hold|Sell|Underperform|Underweight|Market Perform|"
    r"Equal-?Weight|Sector Perform|Reduce)\b",
    re.I,
)
UPGRADE: Final = re.compile(r"\bUpgrade[sd]?\b|\bRaised To (?:Buy|Outperform|Overweight)\b", re.I)
PT_CUT: Final = re.compile(
    r"\b(?:Lowers?|Lowered|Cuts?|Slashes|Trims|Reduces|Decreases)\b.{0,40}"
    r"\b(?:Price Target|PT|Target Price)\b|"
    r"\b(?:Price Target|PT)\b.{0,10}\b(?:Cut|Lowered|Reduced|Slashed|Trimmed)\b",
    re.I,
)
PT_RAISE: Final = re.compile(
    r"\b(?:Raises?|Raised|Boosts?|Lifts?|Increases?|Hikes?|Bumps?)\b.{0,40}"
    r"\b(?:Price Target|PT|Target Price)\b",
    re.I,
)
# Case-sensitive, like Benzinga's rating wires.
INIT_NEG: Final = re.compile(
    r"\bInitiates Coverage On .{1,60} [Ww]ith (?:Sell|Underperform|Underweight|Reduce)\b"
)

# Structural negatives, and antitrust (unclear) checked among them.
RESTATEMENT: Final = re.compile(
    r"\bRestat(?:e|es|ed|ement)\b|Non-?Reliance|Material Weakness|"
    r"Accounting (?:Irregularit|Review|Errors?|Issues?|Probe)|"
    r"Delay(?:s|ed)? (?:Its |Filing|Of )?(?:Annual|Quarterly)? ?(?:Report|10-[KQ]|Filing)|"
    r"Late Filing|NT 10-[KQ]|Unable To Timely File",
    re.I,
)
INSOLVENCY: Final = re.compile(
    r"\bBankrupt|Chapter (?:11|7)\b|Going Concern|\bDefault(?:s|ed)? On\b|"
    r"Missed (?:Interest|Coupon) Payment|Forbearance|Restructuring Support Agreement|Insolven|"
    r"Covenant (?:Breach|Waiver)|Debt Restructuring",
    re.I,
)
DELISTING: Final = re.compile(
    r"\bDelist(?:ing|ed)?\b|(?:Nasdaq|NYSE) (?:Notice|Deficiency)|Deficiency (?:Letter|Notice)|"
    r"Minimum Bid Price|Non-?Compliance With (?:Nasdaq|NYSE)",
    re.I,
)
SHORT_REPORT: Final = re.compile(
    r"\b(?:Hindenburg|Muddy Waters|Citron|Spruce Point|Grizzly Research|Culper|Kerrisdale|"
    r"Bonitas|Wolfpack|Fuzzy Panda|Iceberg Research|Blue Orca|Gotham City|Viceroy|J Capital|"
    r"Scorpion Capital|Morpheus Research|Hunterbrook|Andrew Left)\b|"
    r"Short[- ]Sell(?:er|ing) (?:Report|Attack|Alleg|Target)|Short Report|Bear Raid",
    re.I,
)
ANTITRUST: Final = re.compile(
    r"\bAntitrust|Anti-?Competitive|Competition (?:Watchdog|Authority|Regulator|Probe|Commission)|"
    r"European Commission|\bEU (?:Fine|Probe|Investigat|Regulators?|Antitrust|Commission)|"
    r"Digital Markets Act|\bDMA\b|\bCMA\b|\bFTC\b|Monopol|\bBreak(?:ing)?[- ]?[Uu]p\b(?! Fee)",
    re.I,
)
FRAUD_PROBE: Final = re.compile(
    r"\bSEC (?:Probe|Investigat|Subpoena|Charges|Sues|Inquiry|Enforcement)|Wells Notice|"
    r"\b(?:DOJ|Department Of Justice) (?:Probe|Investigat|Charges|Sues|Subpoena|Indict)|"
    r"\bSubpoena|Criminal (?:Probe|Investigation|Charges|Complaint)|"
    r"\bFBI (?:Probe|Investigat|Raid)|\b(?:FBI|Police|Agents|Authorities|Regulators) Raid|"
    r"\bRaided\b|\bIndict|"
    r"\bFraud\b(?! (?:Prevention|Detection|Protection|Solutions?|Platform|Management))|"
    r"Whistleblower|(?:Opens?|Launch(?:es)?|Faces?|Under) (?:An? )?(?:Probe|Investigation)|"
    r"Probe(?:s|d)? Into|Investigat(?:es|ing|ion) Into",
    re.I,
)
DILUTION: Final = re.compile(
    r"(?:Public|Secondary|Proposed|Underwritten|Follow-On|Common Stock|Equity|Share|Stock|Unit|"
    r"Registered Direct|Upsized|Overnight|Convertible(?: Senior)?(?: Notes)?|"
    r"Exchangeable(?: Senior)?(?: Notes)?) Offering|"
    r"Prices? (?:Upsized )?(?:\$?[\d.,]+[MBK]? )?(?:Shares|Offering|Public Offering)|"
    r"Private Placement|At-The-Market|\bATM (?:Program|Offering|Facility)|"
    r"Mixed (?:Securities )?Shelf|Shelf Registration|Files? For .{0,20}Shelf|Block Trade|"
    r"Bought Deal|(?:Sells?|Selling) [\d.,]+[MK]? (?:Shares|Of Its Shares)|Stock Sale By|"
    r"Secondary Sale|May (?:Offer|Sell|Issue)(?: And Sell)?\b.{0,40}\b(?:Shares|Common Stock)|"
    r"Equity Distribution Agreement|Sales Agreement For .{0,30}Shares",
    re.I,
)
GUIDANCE_CUT: Final = re.compile(
    r"\b(?:Cuts?|Lowers?|Lowered|Slashes|Reduces|Trims|Withdraws|Withdrew|Suspends|Pulls|Pulled)\b"
    r"(?:(?!Price Target|\bPT\b).){0,50}"
    r"\b(?:Guidance|Outlook|Forecast|Guide|"
    r"(?:Financial|Long-Term|Margin|Revenue|Sales|Earnings|Growth) Targets?)\b|"
    r"Profit Warning|"
    r"\bWarns (?:Of |On |About )?(?:Weak|Lower|Soft|Slow|Q[1-4]|FY|Revenue|Sales|Profit|"
    r"Earnings|Margin|Demand|Results)|"
    r"Below (?:Guidance|Expectations|Prior Guidance)|Pre-?[Aa]nnounc|"
    r"(?:Weak(?:er)?|Disappointing|Downbeat|Soft|(?:Worse|Lower|Weaker)-Than-Expected) "
    r"(?:Guidance|Outlook|Forecast)",
    re.I,
)
REG_BLOCK: Final = re.compile(
    r"Complete Response Letter|\bCRL\b|FDA (?:Rejects|Declines|Refuses|Delays)|Clinical Hold|"
    r"Fail(?:s|ed)? (?:To Meet )?(?:Its )?Primary|"
    r"Did(?:n't| Not) Meet (?:Its )?(?:Primary|Endpoint)|"
    r"Trial (?:Fail|Halt|Stopped)|Discontinu(?:e|es|ed) (?:Trial|Study|Development|Program)|"
    r"Export (?:Ban|Restriction|Curb|Control|License)|Blocks? (?:Merger|Deal|Acquisition)|"
    r"\b(?:Import|Export|Sales) Ban\b|"
    r"\bBan(?:s|ned)? (?:On|Of) (?:Sales|Imports?|Exports?|Shipments?)\b|\bBanned From\b",
    re.I,
)
MA_BREAK: Final = re.compile(
    r"Terminat(?:e|es|ed|ion) (?:Of )?(?:the )?(?:Merger|Acquisition|Deal|Agreement To)|"
    r"Deal (?:Collapses|Falls Through|Called Off|Blocked|Terminated)|"
    r"Walks Away|Abandons? (?:Deal|Bid|Merger|Acquisition)",
    re.I,
)
CUSTOMER_LOSS: Final = re.compile(
    r"Los(?:es|ing|t) (?:A |Its |Key |Major )?(?:Contract|Customer|Client|Account)|"
    r"Contract (?:Cancel|Terminat|Loss)|Customer (?:Loss|Exit)",
    re.I,
)
IMPAIRMENT: Final = re.compile(r"\bImpairment|Write-?(?:Down|Off)s?\b", re.I)

# Unclear negatives checked after analyst actions and movers.
MGMT_EXIT: Final = re.compile(
    r"\b(?:CEO|CFO|Chief Executive|Chief Financial|President|Chairman|Founder)\b.{0,60}"
    r"\b(?:Steps? Down|Stepping Down|Resign|Depart|Ousted|Fired|Terminated|Exits?|Leaves|"
    r"Leaving|To Leave|Abrupt|Unexpected|Sudden)",
    re.I,
)
LEGAL_ADVERSE: Final = re.compile(
    r"\bLawsuit (?:Filed )?Against\b|\bSued\b|\bFaces? (?:A )?(?:Lawsuit|Suit|Class Action)\b|"
    r"\bLoses? (?:Lawsuit|Case|Appeal|Patent (?:Case|Suit|Fight|Trial))|"
    r"\bJury (?:Verdict|Orders|Finds|Awards)|\b(?:Judge|Court) Rules? Against|"
    r"\bInfring(?:es|ed|ement) (?:Verdict|Ruling)|\bInjunction Against|\bFined\b|\bFine Of\b|"
    r"\bPenalty Of\b",
    re.I,
)
OP_INCIDENT: Final = re.compile(
    r"\bRecall(?:s|ed|ing)?\b|Outage|(?:Data|Security) Breach|Cyber-?attack|Hack(?:ed|ers)?\b|"
    r"Ransomware|Fire At|Explosion|"
    r"\b(?:Workers?|Union|Labor|Employees) (?:Go On |Begin |Launch )?Strike|\bOn Strike\b|"
    r"\bStrike At\b|Walkout|Production (?:Halt|Issue|Problem|Delay)|"
    r"Supply (?:Shortage|Disruption)|Grounded",
    re.I,
)
SYMPATHY: Final = re.compile(r"\bSympathy\b", re.I)

# Positives and neutrals: the prototype's table, verbatim.
MA_TARGET: Final = re.compile(
    r"To Be Acquired|Agrees? To Be Acquired|"
    r"(?:Receives?|Rejects?) (?:A |An )?(?:Takeover|Buyout|Acquisition) (?:Offer|Bid|Proposal)|"
    r"Takeover (?:Bid|Offer|Interest|Talks)|Buyout|Explor(?:es|ing) (?:A )?Sale|"
    r"Go(?:ing)? Private|Tender Offer For",
    re.I,
)
GUIDANCE_RAISE: Final = re.compile(
    r"\bRaises?\b(?:(?!Price Target|\bPT\b).){0,50}\b(?:Guidance|Outlook|Forecast)\b|"
    r"Above (?:Guidance|Expectations)|"
    r"(?:Better|Stronger)-Than-Expected (?:Guidance|Outlook)|Strong (?:Guidance|Outlook)|"
    r"Upbeat (?:Guidance|Outlook)",
    re.I,
)
BUYBACK: Final = re.compile(r"Repurchase|Buyback|Buy-?Back", re.I)
REGULATORY_WIN: Final = re.compile(
    r"FDA (?:Approv|Clear|Grants?)|Approval (?:For|Of)|Receives? (?:FDA|CE Mark|EUA)|Clearance",
    re.I,
)
CONTRACT_WIN: Final = re.compile(
    r"\bAwarded\b|\bWins?\b.{0,40}\b(?:Contract|Deal|Order|Award)|"
    r"\bContract (?:Worth|Valued|For)|Selected By|Partnership With|Partners With|"
    r"Collaborat|Agreement With",
    re.I,
)
PRODUCT: Final = re.compile(
    r"\bLaunch(?:es|ed)?\b|\bUnveil|\bIntroduc|\bAnnounces? (?:New|Next)|"
    r"\bRelease[sd]? (?:New|Its)",
    re.I,
)
MA_ACQUIRER: Final = re.compile(
    r"\bTo Acquire\b|\bAcquires?\b|\bAcquisition Of\b|\bMerger (?:With|Agreement)\b|"
    r"\bTo Buy\b.{0,40}\bFor \$",
    re.I,
)
RESTRUCTURING: Final = re.compile(
    r"\bLayoffs?\b|Lay(?:s|ing)? Off|Job Cuts|"
    r"Cut(?:s|ting)? [\d,]+ (?:Jobs|Employees|Workers|Positions)|Workforce Reduction|"
    r"Restructuring (?:Plan|Program|Charge)|Reduce (?:Its )?Workforce",
    re.I,
)
DIVIDEND: Final = re.compile(r"\bDividend\b", re.I)

# ── Orders (first match wins) ───────────────────────────────────────────────

# classify_item step 6: structural negatives (antitrust among them, ahead of
# fraud, so "EU Antitrust Probe" is antitrust), before any analyst action.
NEGATIVE_ORDER: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("restatement", RESTATEMENT),
    ("insolvency", INSOLVENCY),
    ("delisting", DELISTING),
    ("short_report", SHORT_REPORT),
    ("antitrust_regulatory", ANTITRUST),
    ("fraud_probe", FRAUD_PROBE),
    ("dilution", DILUTION),
    ("guidance_cut", GUIDANCE_CUT),
    ("regulatory_block", REG_BLOCK),
    ("ma_break", MA_BREAK),
    ("customer_loss", CUSTOMER_LOSS),
    ("impairment", IMPAIRMENT),
)
# Step 9: the unclear negatives a headline can state.
LATE_NEGATIVE_ORDER: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("management_exit", MGMT_EXIT),
    ("legal_adverse", LEGAL_ADVERSE),
    ("operational_incident", OP_INCIDENT),
    ("sympathy", SYMPATHY),
)
# Step 10: positives and neutrals.
OTHER_ORDER: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("ma_target", MA_TARGET),
    ("guidance_raise", GUIDANCE_RAISE),
    ("buyback", BUYBACK),
    ("regulatory_win", REGULATORY_WIN),
    ("contract_win", CONTRACT_WIN),
    ("product", PRODUCT),
    ("ma_acquirer", MA_ACQUIRER),
    ("restructuring", RESTRUCTURING),
    ("dividend", DIVIDEND),
)
# An 8-K's items: structural beats unclear-negative beats the rest; among the
# rest an earnings release (2.02) first, then the table's order.
FILING_ORDER: Final = (
    "restatement",
    "insolvency",
    "delisting",
    "dilution",
    "impairment",
    "auditor_change",
    "agreement_terminated",
    "earnings_8k",
    "restructuring",
    "ma_closed",
    "agreement",
    "management_change",
    "debt_financing",
    "filing_other",
)

# resolve() steps 5, 7 and 8.
UNCLEAR_ORDER: Final = (
    "antitrust_regulatory",
    "management_exit",
    "legal_adverse",
    "operational_incident",
    "auditor_change",
    "agreement_terminated",
    "sympathy",
)
POSITIVE_ORDER: Final = (
    "ma_target",
    "guidance_raise",
    "analyst_upgrade",
    "buyback",
    "regulatory_win",
    "contract_win",
    "product",
    "analyst_pt_raise",
)
NEUTRAL_TYPES: Final = (
    "ma_acquirer",
    "restructuring",
    "dividend",
    "agreement",
    "ma_closed",
    "management_change",
    "debt_financing",
    "filing_other",
    "other",
)
# Items a story reads nothing from (a mover still counts, as a story's lead).
_IGNORED: Final = frozenset({"noise", "law_firm"})
# Items an earnings story has whose numbers no template read.
_UNREAD_EARNINGS: Final = frozenset({"earnings_8k", "guidance_unparsed", "earnings_fact"})
# Positive items that keep a story out of the H1 family.
POSITIVE_VETOES: Final = frozenset({"analyst_upgrade", "guidance_raise", "ma_target"})

FAMILY: Final = "NSN_CORE"
FAMILY_TYPES: Final = frozenset({"analyst_downgrade", "earnings_miss"})

_DOWN_ACTIONS: Final = frozenset({"lowers", "cuts", "withdraws", "suspends"})
_VERDICT_WORDS: Final[dict[str, int]] = {
    "beat": 1,
    "beats": 1,
    "miss": -1,
    "misses": -1,
    "meet": 0,
    "meets": 0,
    "inline": 0,
    "in-line": 0,
    "inline with": 0,
    "in-line with": 0,
}


# ── Items ───────────────────────────────────────────────────────────────────


def _filing_type(items_8k: Sequence[str]) -> str:
    types = {ITEM_TYPES[code] for code in items_8k if code in ITEM_TYPES}
    return next((t for t in FILING_ORDER if t in types), "filing_other")


def _analyst_type(text: str) -> str:
    if DOWNGRADE.search(text) and not UPGRADE.search(text):
        return "analyst_downgrade"
    if UPGRADE.search(text):  # "Upgrades ..., Lowers Price Target" is an upgrade
        return "analyst_upgrade"
    if INIT_NEG.search(text):
        return "analyst_init_neg"
    if PT_CUT.search(text):
        return "analyst_pt_cut"
    if PT_RAISE.search(text):
        return "analyst_pt_raise"
    return "analyst_other"


def classify_item(
    kind: str,
    headline: str,
    items_8k: Sequence[str],
    facts: Sequence[EarningsFacts],
    clause: str | None = None,
) -> str:
    """The type of one item (spec §B.2, first match wins).

    ``kind`` is ``news``, ``8-k`` or ``8-k/a``; ``facts`` are the item's
    ``benzinga-earnings-v4`` facts; ``clause`` is the analyst clause group of
    the story's company (the story builder computes it), or None to read the
    whole headline. Only the analyst step reads the clause: every other rule
    reads the headline.
    """
    if kind in FILING_KINDS:
        return _filing_type(items_8k)
    if LAW_FIRM.search(headline):
        return "law_firm"
    if NOISE.search(headline):
        return "noise"
    if any(f.kind in ("result", "guidance") for f in facts):
        return "earnings_fact"
    if GUIDE_UNPARSED.search(headline):
        return "guidance_unparsed"
    for name, rx in NEGATIVE_ORDER:
        if rx.search(headline):
            return name
    if ANALYST_ACTION.search(headline):
        return _analyst_type(clause if clause is not None else headline)
    if MOVER.search(headline):
        return "mover"
    for name, rx in (*LATE_NEGATIVE_ORDER, *OTHER_ORDER):
        if rx.search(headline):
            return name
    return "other"


# ── Stories ─────────────────────────────────────────────────────────────────


class ItemLike(Protocol):
    """What :func:`resolve` reads of a story item (``stories.StoryItem`` satisfies it)."""

    @property
    def event_id(self) -> int: ...
    @property
    def available_at(self) -> datetime: ...
    @property
    def itype(self) -> str: ...
    @property
    def facts(self) -> Sequence[EarningsFacts]: ...
    @property
    def supersedes(self) -> Sequence[int]: ...  # event_ids whose facts this item replaces


@dataclass(frozen=True, slots=True)
class EarningsVerdict:
    eps: int | None  # +1 beat, 0 inline, -1 miss, None unknown
    sales: int | None
    guide: Guide | None  # None: no guidance stated at all
    type: str


@dataclass(frozen=True, slots=True)
class StoryCard:
    type: str
    direction: str
    permanence: str
    family: Literal["NSN_CORE"] | None
    vetoes: tuple[str, ...]  # structural and unclear-negative types present
    positives: tuple[str, ...]
    earnings: EarningsVerdict | None
    follower: bool
    structural: bool  # anything structural known (the playbook's abort)


def _known[T: ItemLike](items: Sequence[T], t: datetime) -> list[T]:
    """Items available by ``t``, in time order (event id breaks ties)."""
    return sorted(
        (i for i in items if i.available_at <= t), key=lambda i: (i.available_at, i.event_id)
    )


def _rounded(value: object) -> float | None:
    number = to_float(value)
    return None if number is None else round(number, _SURPRISE_DIGITS)


def metric_verdict(fields: Mapping[str, object], metric: Literal["eps", "sales"]) -> int | None:
    """+1 beat, 0 inline, -1 miss, None unknown, for one result fact's metric.

    A stated surprise decides, with a dead band of :data:`DEAD_BAND` (so a
    pre-2018 "EPS $1.27 vs $1.12 Est." is a beat); without one, Benzinga's
    verdict word does; "Up From ... YoY" and a bare "vs" decide nothing.
    """
    surprise = _rounded(fields.get(f"{metric}_surprise"))
    if surprise is not None:
        if surprise >= DEAD_BAND:
            return 1
        if surprise <= -DEAD_BAND:
            return -1
        return 0
    word = fields.get(f"{metric}_verdict")
    return _VERDICT_WORDS.get(str(word).lower().rstrip(".")) if word is not None else None


def _latest_verdict(
    results: Sequence[tuple[ItemLike, int, EarningsFacts]], metric: Literal["eps", "sales"]
) -> int | None:
    """The metric's verdict from the latest fact stating it, adjusted basis preferred."""
    stating = [r for r in results if r[2].fields.get(f"{metric}_verdict") is not None]
    adjusted = [r for r in stating if r[2].fields.get("basis", "gaap") != "gaap"]
    pool = adjusted or stating
    if not pool:
        return None
    # Latest item first; within one headline, its first statement.
    _, _, fact = max(pool, key=lambda r: (r[0].available_at, r[0].event_id, -r[1]))
    return metric_verdict(fact.fields, metric)


def _guidance_moves(guidance: Sequence[EarningsFacts]) -> tuple[bool, bool]:
    """(any guidance below consensus or cut, any raised or above consensus)."""
    down = up = False
    for g in guidance:
        action = str(g.fields.get("action") or "").lower()
        surprise = _rounded(g.fields.get("surprise"))
        down |= action in _DOWN_ACTIONS or (surprise is not None and surprise <= -GUIDE_BAND)
        up |= action == "raises" or (surprise is not None and surprise >= GUIDE_BAND)
    return down, up


def earnings_verdict(items: Sequence[ItemLike], t: datetime) -> EarningsVerdict | None:
    """The story's earnings reading at ``t`` (spec §B.5); None when no fact is known.

    Facts come from the items available by ``t`` that no correction available
    by ``t`` supersedes; noise and law-firm items are ignored, facts and all.
    Per metric the latest stated fact decides, adjusted basis preferred over
    GAAP; a print that beats on one metric and misses on the other is a miss.
    """
    every = _known(items, t)
    superseded = {e for i in every for e in i.supersedes}
    known = [i for i in every if i.itype not in _IGNORED]
    facts = [
        (i, k, f) for i in known if i.event_id not in superseded for k, f in enumerate(i.facts)
    ]
    results = [x for x in facts if x[2].kind == "result"]
    guidance = [x[2] for x in facts if x[2].kind == "guidance"]
    if not results and not guidance:
        return None
    present = {i.itype for i in known}
    eps, sales = _latest_verdict(results, "eps"), _latest_verdict(results, "sales")
    down, up = _guidance_moves(guidance)
    down |= "guidance_cut" in present
    up &= not down
    unknown = "guidance_unparsed" in present and not guidance
    guide: Guide | None = (
        "down" if down else "up" if up else "unknown" if unknown else "inline" if guidance else None
    )
    signs = {v for v in (eps, sales) if v is not None}
    if down:
        kind = "guidance_cut"
    elif -1 in signs:
        kind = (
            "earnings_miss_guide_up"
            if up
            else "earnings_miss_guide_unk"
            if unknown
            else "earnings_miss"
        )
    elif 1 in signs:
        kind = "earnings_beat"
    elif results:
        kind = "earnings_inline"
    else:
        kind = "guidance_raise" if up else "guidance_inline"
    return EarningsVerdict(eps, sales, guide, kind)


def _unique(types: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(types))


def _story_type(real: Sequence[ItemLike], earnings: EarningsVerdict | None) -> str:
    """Spec §B.6 over the non-noise items known at the moment, first match wins."""
    present = {i.itype for i in real}
    structural = [i.itype for i in real if i.itype in STRUCTURAL]
    if structural:
        return structural[0]
    if earnings is not None:
        return earnings.type
    if present & _UNREAD_EARNINGS:
        return "earnings_unparsed"
    if "analyst_downgrade" in present:
        return "analyst_downgrade"
    for name in UNCLEAR_ORDER:
        if name in present:
            return name
    if present & {"analyst_pt_cut", "analyst_init_neg"}:
        return "analyst_pt_cut"
    for name in POSITIVE_ORDER:
        if name in present:
            return name
    neutral = Counter(i.itype for i in real if i.itype in NEUTRAL_TYPES)
    if neutral:
        top = max(neutral.values())
        return next(i.itype for i in real if neutral.get(i.itype) == top)  # ties: earliest
    if "mover" in present:
        return "mover_only"
    if "analyst_other" in present:
        return "analyst_other"
    return "noise_only"


def resolve(items: Sequence[ItemLike], t: datetime, *, follower: bool) -> StoryCard:
    """The story's card at ``t``: only items with ``available_at <= t`` are read.

    ``follower`` is the story builder's ``follower_at(t)``. Noise and law-firm
    items are ignored; a mover leading the story keeps it out of the family.
    """
    known = _known(items, t)
    real = [i for i in known if i.itype not in _IGNORED]
    earnings = earnings_verdict(known, t)
    kind = _story_type(real, earnings)
    vetoes = [i.itype for i in real if i.itype in STRUCTURAL or i.itype in UNCLEAR_NEG]
    if earnings is not None and earnings.guide == "down":
        vetoes.append("guidance_cut")  # guidance below consensus, stated as a fact
    veto = _unique(vetoes)
    positives = _unique([i.itype for i in real if i.itype in POSITIVE_VETOES])
    led_by_mover = bool(real) and real[0].itype == "mover"
    family: Literal["NSN_CORE"] | None = (
        "NSN_CORE"
        if kind in FAMILY_TYPES and not veto and not positives and not follower and not led_by_mover
        else None
    )
    meta = TYPES[kind]
    return StoryCard(
        type=kind,
        direction=meta.direction,
        permanence=meta.permanence,
        family=family,
        vetoes=veto,
        positives=positives,
        earnings=earnings,
        follower=follower,
        structural=any(v in STRUCTURAL for v in veto),
    )


# ── Pins ────────────────────────────────────────────────────────────────────


def _patterns() -> dict[str, re.Pattern[str]]:
    return {
        "LAW_FIRM": LAW_FIRM,
        "NOISE": NOISE,
        "MOVER": MOVER,
        "DOWNGRADE": DOWNGRADE,
        "UPGRADE": UPGRADE,
        "PT_CUT": PT_CUT,
        "PT_RAISE": PT_RAISE,
        "INIT_NEG": INIT_NEG,
        "GUIDE_UNPARSED": GUIDE_UNPARSED,
        **{name.upper(): rx for name, rx in (*NEGATIVE_ORDER, *LATE_NEGATIVE_ORDER, *OTHER_ORDER)},
    }


def sources() -> dict[str, str]:
    """Every regex source, table and order the rules use, for the pre-registration's pins."""
    tables: dict[str, object] = {
        "RULES_VERSION": RULES_VERSION,
        "ITEM_TYPES": ITEM_TYPES,
        "TYPES": {k: [m.direction, m.permanence] for k, m in TYPES.items()},
        "STRUCTURAL": sorted(STRUCTURAL),
        "UNCLEAR_NEG": sorted(UNCLEAR_NEG),
        "REACTIVE": sorted(REACTIVE),
        "NOISE_TYPES": sorted(NOISE_TYPES),
        "NEGATIVE_ORDER": [n for n, _ in NEGATIVE_ORDER],
        "LATE_NEGATIVE_ORDER": [n for n, _ in LATE_NEGATIVE_ORDER],
        "OTHER_ORDER": [n for n, _ in OTHER_ORDER],
        "FILING_ORDER": list(FILING_ORDER),
        "UNCLEAR_ORDER": list(UNCLEAR_ORDER),
        "POSITIVE_ORDER": list(POSITIVE_ORDER),
        "NEUTRAL_TYPES": list(NEUTRAL_TYPES),
        "POSITIVE_VETOES": sorted(POSITIVE_VETOES),
        "IGNORED": sorted(_IGNORED),
        "UNREAD_EARNINGS": sorted(_UNREAD_EARNINGS),
        "FAMILY": {FAMILY: sorted(FAMILY_TYPES)},
        "DEAD_BAND": DEAD_BAND,
        "GUIDE_BAND": GUIDE_BAND,
        "SURPRISE_DIGITS": _SURPRISE_DIGITS,
        "DOWN_ACTIONS": sorted(_DOWN_ACTIONS),
        "VERDICT_WORDS": _VERDICT_WORDS,
    }
    return (
        {name: rx.pattern for name, rx in _patterns().items()}
        | headline_patterns.sources()
        | {name: json.dumps(value, sort_keys=True) for name, value in tables.items()}
    )


# The rules' pin in the news engine's pre-registration.
TAXONOMY_SHA: Final = hashlib.sha256(json.dumps(sources(), sort_keys=True).encode()).hexdigest()[
    :12
]
