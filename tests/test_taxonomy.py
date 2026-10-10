"""Rules taxonomy v1 (events/taxonomy.py): item types, precedence, story cards, the H1 family.

Every headline below is a real Benzinga headline from the news history unless
marked constructed. No price is read anywhere.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.events import headline_patterns
from halal_trader.events import taxonomy as tx
from halal_trader.events.earnings_parse import EarningsFacts, parse_headline
from halal_trader.events.taxonomy import classify_item, earnings_verdict, metric_verdict, resolve

T0 = datetime(2021, 5, 10, 12, 0, tzinfo=UTC)


def news_type(headline: str, clause: str | None = None) -> str:
    return classify_item("news", headline, (), parse_headline(headline), clause)


@dataclass(frozen=True, slots=True)
class Item:
    """A story item as resolve() reads it (stories.StoryItem's shape)."""

    event_id: int
    available_at: datetime
    itype: str
    facts: tuple[EarningsFacts, ...] = ()
    supersedes: tuple[int, ...] = ()


def news(event_id: int, minutes: float, headline: str, *, supersedes: Sequence[int] = ()) -> Item:
    """A news item available ``minutes`` after T0, typed and parsed as the builder would."""
    facts = tuple(parse_headline(headline))
    return Item(
        event_id,
        T0 + timedelta(minutes=minutes),
        classify_item("news", headline, (), facts),
        facts,
        tuple(supersedes),
    )


def filing(event_id: int, minutes: float, *items: str) -> Item:
    return Item(event_id, T0 + timedelta(minutes=minutes), classify_item("8-k", "", items, ()))


def typed(event_id: int, minutes: float, itype: str) -> Item:
    return Item(event_id, T0 + timedelta(minutes=minutes), itype)


LATER = T0 + timedelta(days=1)


# ── Tables ──────────────────────────────────────────────────────────────────


def test_every_type_the_rules_can_return_has_a_meta() -> None:
    returned = (
        {n for n, _ in tx.NEGATIVE_ORDER}
        | {n for n, _ in tx.LATE_NEGATIVE_ORDER}
        | {n for n, _ in tx.OTHER_ORDER}
        | set(tx.FILING_ORDER)
        | set(tx.ITEM_TYPES.values())
        | set(tx.UNCLEAR_ORDER)
        | set(tx.POSITIVE_ORDER)
        | set(tx.NEUTRAL_TYPES)
        | tx.REACTIVE
        | tx.NOISE_TYPES
        | tx.STRUCTURAL
        | tx.UNCLEAR_NEG
        | {
            "law_firm",
            "noise",
            "earnings_fact",
            "guidance_unparsed",
            "analyst_downgrade",
            "analyst_upgrade",
            "analyst_init_neg",
            "analyst_pt_cut",
            "analyst_pt_raise",
            "analyst_other",
            "mover",
            "other",
            "earnings_beat",
            "earnings_inline",
            "earnings_miss",
            "earnings_miss_guide_up",
            "earnings_miss_guide_unk",
            "guidance_cut",
            "guidance_raise",
            "guidance_inline",
            "earnings_unparsed",
            "mover_only",
            "noise_only",
        }
    )
    assert returned <= set(tx.TYPES)


def test_the_classes_of_spec_b1() -> None:
    assert tx.STRUCTURAL == {
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
    assert tx.UNCLEAR_NEG == {
        "antitrust_regulatory",
        "management_exit",
        "legal_adverse",
        "operational_incident",
        "auditor_change",
        "agreement_terminated",
        "sympathy",
    }
    for name in tx.STRUCTURAL:
        assert tx.TYPES[name] == tx.TypeMeta("neg", "structural")
    for name in tx.UNCLEAR_NEG:
        assert tx.TYPES[name] == tx.TypeMeta("neg", "unclear")
    for name in ("analyst_downgrade", "analyst_pt_cut", "analyst_init_neg", "earnings_miss"):
        assert tx.TYPES[name] == tx.TypeMeta("neg", "transient")
    assert tx.TYPES["ma_target"] == tx.TypeMeta("pos", "structural")
    assert tx.TYPES["earnings_beat"] == tx.TypeMeta("pos", "transient")
    assert tx.TYPES["earnings_miss_guide_up"] == tx.TypeMeta("neutral", "unclear")
    assert tx.TYPES["earnings_miss_guide_unk"] == tx.TypeMeta("neg", "unclear")
    assert tx.TYPES["guidance_raise"] == tx.TypeMeta("pos", "unclear")
    assert tx.TYPES["earnings_unparsed"] == tx.TypeMeta("none", "none")
    for name in ("noise", "law_firm", "mover", "earnings_fact", "earnings_8k", "guidance_unparsed"):
        assert tx.TYPES[name].direction == "none"


def test_reactive_and_noise_sets() -> None:
    assert tx.REACTIVE == {
        "analyst_pt_cut",
        "analyst_init_neg",
        "analyst_pt_raise",
        "analyst_other",
        "mover",
        "guidance_unparsed",
    }
    assert tx.NOISE_TYPES == {"noise", "law_firm", "mover"}


def test_8k_item_table() -> None:
    assert tx.ITEM_TYPES["4.02"] == "restatement"
    assert tx.ITEM_TYPES["3.01"] == "delisting"
    assert tx.ITEM_TYPES["2.04"] == tx.ITEM_TYPES["1.03"] == "insolvency"
    assert tx.ITEM_TYPES["2.06"] == "impairment"
    assert tx.ITEM_TYPES["3.02"] == "dilution"
    assert tx.ITEM_TYPES["4.01"] == "auditor_change"
    assert tx.ITEM_TYPES["1.02"] == "agreement_terminated"
    assert tx.ITEM_TYPES["2.02"] == "earnings_8k"
    assert tx.ITEM_TYPES["5.02"] == "management_change"
    for code in ("5.03", "5.07", "7.01", "8.01", "9.01"):
        assert tx.ITEM_TYPES[code] == "filing_other"


# ── Items: one real headline per type ───────────────────────────────────────


@pytest.mark.parametrize(
    ("itype", "headline"),
    [
        (
            "restatement",
            "Terraform Says They Have Identified Material Weakness In Internal Controls Over "
            "Financial Reporting, Will Be Necssary To Implement More Controls And Take Other, "
            "Remedial Actions",
        ),
        (
            "insolvency",
            "O-I Says Paddock Chapter 11 Filing Proceeding As Expected, Unit Is Seeking Final "
            "Resolution Of Its Asbestos-Related Liabilities",
        ),
        (
            "delisting",
            "NYSE To Suspend Trading In Sanchez Energy Corporation And Commence Delisting "
            "Proceedings",
        ),
        (
            "short_report",
            "Short Seller Andrew Left Goes Sour On Lemonade, Says Company Lies To Shareholders",
        ),
        (
            "antitrust_regulatory",
            "Google Fined €220M By French Antitrust Authority for Abuse of Market Power in Ad "
            "Business",
        ),
        (
            "fraud_probe",
            "Tesla Investigated For Fraud, HSBC For Money Laundering And Auditors Receive New "
            "Rules",
        ),
        (
            "dilution",
            "Beam Therapeutics Announces Proposed Public Offering Of Common Stock Of 4.5M Shares",
        ),
        ("guidance_cut", "McDonald's Withdraws Guidance After Significant Sales Decline In March"),
        ("guidance_cut", "Deere Cuts FY19 Sales Growth Guidance From 7% To 5%"),
        (
            "regulatory_block",
            "Biogen Announces Topline Results From Phase 2/3 Gene Therapy Study For XLRP; Study "
            "Did Not Meet Primary Endpoint",
        ),
        ("ma_break", "Albertsons Terminates Merger Agreement With Kroger"),
        (
            "customer_loss",
            "Boeing Expert Says 737 MAX Freeze & Certification Challenges May Lead To Customer "
            "Loss: BofA Analyst",
        ),
        (
            "impairment",
            "UPDATE: DowDuPont Impairment Charge Related To Agriculture Business Expected To Be "
            "About $4.6B, Says Has No Impact on Previously Announced Financial Guidance",
        ),
        (
            "management_exit",
            "Hertz Reports Pres, CEO Kathryn Marinello Resigns; Names Paul Stone Pres, CEO",
        ),
        ("legal_adverse", "Apple Sued Over Not Taking Down Telegram After Capitol Hill Riot"),
        (
            "operational_incident",
            "Polaris Recalls Sportsman 850 and 1000 ATVs Due To Burn, Fire Hazards",
        ),
        ("sympathy", "Steel Stocks Move In Sympathy After Cliffs Q4 Beat"),
        ("ma_target", "Forescout To Be Acquired By Advent International For $33/Share In Cash"),
        ("guidance_raise", "Northrop Grumman Raises FY19 Guidance"),
        ("buyback", "AutoZone Adds $750M to Buyback Plan"),
        (
            "regulatory_win",
            "Zoetis Receives FDA Approval For Simparica Trio, A New Combination Parasite "
            "Preventative For Dogs",
        ),
        ("contract_win", "Primoris Services Wins $155M+ Solar Award"),
        ("product", "Intel Unveils the Intel Neural Compute Stick 2"),
        ("ma_acquirer", "Arthur J Gallagher Acquires Charles Allen Agency"),
        ("restructuring", "Goldman Plans Second Round Of Job Cuts In 3 Months:Report"),
        ("dividend", "Ally Financial Raises Qtr. Dividend From $0.13 to $0.15/Share"),
        ("noise", "Understanding Vistra's Unusual Options Activity"),
        ("noise", "63 Stocks Moving In Monday's Mid-Day Session"),
        (
            "law_firm",
            "Peter Schiff: MSTR Is In A 'Death Spiral' As Rosen Law Firm Announces Investigation "
            "Into Strategy",
        ),
        ("mover", "Rhythm Pharma Shares Up 26% Upon Resumption"),
        (
            "earnings_fact",
            "Micron Technology Q1 EPS $(0.95) Misses $(0.91) Estimate, Sales $4.73B Beat $4.27B "
            "Estimate",
        ),
        ("earnings_fact", "Bloom Energy Q4 Sales $213.8M Miss $270.07M Estimate"),
        (
            "guidance_unparsed",
            "Ulta Sees Q1 Rev. $1.016B-$1.033B vs. Est. $1.01B, EPS $1.25-$1.30 vs. Est. $1.22",
        ),
        ("analyst_downgrade", "Raymond James Downgrades Comcast to Market Perform"),
        ("analyst_upgrade", "JP Morgan Upgrades Rambus To Overweight"),
        (
            "analyst_init_neg",
            "Goldman Sachs Initiates Coverage On Advance Auto Parts with Sell Rating, Announces "
            "$115 Price Target",
        ),
        (
            "analyst_pt_cut",
            "Barclays Maintains Equal-Weight on Workday, Lowers Price Target to $128",
        ),
        (
            "analyst_pt_raise",
            "Stifel Nicolaus Maintains Buy on Stitch Fix, Raises Price Target to $32",
        ),
        (
            "analyst_other",
            "Citigroup Initiates Coverage On VICI Properties with Neutral Rating, Announces $22 "
            "Price Target",
        ),
        ("other", "Momenta Pharma Names Young Kwon CFO"),
        ("other", "Ford Reports 6.2% Gain In Oct. Auto Sales"),
    ],
)
def test_one_real_headline_per_type(itype: str, headline: str) -> None:
    assert news_type(headline) == itype


# ── Items: precedence ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("itype", "headline"),
    [
        (
            "restatement",
            "UPDATE: Bank Of America Downgrades Hain Celestial, Remains Positive on Co Growth "
            "Story, Says Until Accounting Issues Are Resolved Upside Potential Remains Elusive",
        ),
        (
            "antitrust_regulatory",
            "Evercore Downgrades Henry Schein To In-Line Given 'Heightened' Medium Term "
            "Uncertainty, Firm Says They Lack Conviction In Ascertaining And Quantifying The "
            "Potential Threat From The Recent FTC Investigation",
        ),
        (
            "guidance_cut",
            "Starbucks Shares Down 3.7% Premarket On Multiple Downgrades Following Announcement "
            "That It Would Close 150 Stores In 2019 And Cut FY18 EPS Guidance From $2.48-$2.53 To "
            "$2.39-$2.43",
        ),
        # Constructed, the spec's own case: the cause beats the rating action.
        ("fraud_probe", "Firm Downgrades Acme On Fraud Concerns"),
    ],
)
def test_a_structural_or_regulatory_cause_beats_the_analyst_action(
    itype: str, headline: str
) -> None:
    assert news_type(headline) == itype


@pytest.mark.parametrize(
    ("itype", "headline"),
    [
        ("analyst_upgrade", "UBS Upgrades Moelis to Neutral, Lowers Price Target to $59"),
        (
            "analyst_upgrade",
            "Citigroup Upgrades JetBlue Airways to Neutral, Lowers Price Target to $4.5",
        ),
        (
            "analyst_downgrade",
            "Goldman Sachs Downgrades LPL Financial Holdings to Sell, Raises PT to $21.00, "
            "Recommends Buying May $24 Puts",
        ),
        (
            "analyst_downgrade",
            "Morgan Stanley Downgrades Cemex, S.A.B. de C.V. Sponsored ADR to Equal-Weight, Lowers "
            "Price Target to $5.5",
        ),
        # Analyst actions come before movers: the cause, not the price move.
        ("analyst_upgrade", "FLIR Systems Shares Up 4.1% After Upgrade From Raymond James"),
    ],
)
def test_the_rating_decides_an_analyst_wire(itype: str, headline: str) -> None:
    assert news_type(headline) == itype


def test_the_clause_group_decides_for_its_company() -> None:
    h = "Mo-Mo Pair Trade? FBR Downgrades Netflix, Upgrades Pandora"
    assert news_type(h, clause="Mo-Mo Pair Trade? FBR Downgrades Netflix") == "analyst_downgrade"
    assert news_type(h, clause="Upgrades Pandora") == "analyst_upgrade"
    # The whole headline holds an upgrade, so it is not a downgrade.
    assert news_type(h) == "analyst_upgrade"


def test_init_neg_is_case_sensitive_like_the_wire() -> None:
    assert (
        news_type("UBS Initiates Coverage On International Paper with Sell Rating")
        == "analyst_init_neg"
    )
    # Lowercase "initiates coverage on" is not Benzinga's template.
    assert news_type("UBS initiates coverage on International Paper with Sell Rating") == "other"


def test_law_firm_comes_before_noise_and_the_probe_it_announces() -> None:
    h = (
        "Peter Schiff: MSTR Is In A 'Death Spiral' As Rosen Law Firm Announces Investigation "
        "Into Strategy"
    )
    assert tx.FRAUD_PROBE.search(h) is not None
    assert news_type(h) == "law_firm"


def test_noise_comes_before_earnings_facts() -> None:
    h = (
        "FedEx Maintains Fiscal Year Adjusted Earnings Outlook; Sees FY24 EPS $17.00-$18.50 Vs "
        "$18.25 Est.; Capital Spending Of $5.7B"
    )
    assert parse_headline(h) != []
    assert news_type(h) == "noise"


def test_earnings_facts_come_before_the_guidance_cut_words() -> None:
    h = (
        "Itron Cuts FY18 Guidance: Adj. EPS From $2.75-$2.90 To $2.40-$2.50 vs $2.81 Estimate, "
        "Sales From $2.425B-$2.475B To $2.37B-$2.39B vs $2.44B Est."
    )
    assert tx.GUIDANCE_CUT.search(h) is not None
    assert news_type(h) == "earnings_fact"


def test_unread_guidance_comes_before_the_guidance_cut_words() -> None:
    # Spec order: GUIDE_UNPARSED (5) before the negative regexes (6).
    h = (
        "Core & Main Lowered 2024 Outlook: Now Expects Net Sales Of $7.3B-$7.4B (Prior "
        "$7.5B-$7.6B) Vs. $7.53B Consensus; Adjusted EBITDA Of $900M-$930M (Prior $935M-$975M)"
    )
    assert parse_headline(h) == [] and tx.GUIDANCE_CUT.search(h) is not None
    assert news_type(h) == "guidance_unparsed"


def test_the_facts_passed_decide_not_the_headline() -> None:
    fact = EarningsFacts("result", {"eps_verdict": "beats"})
    assert classify_item("news", "Some Headline", (), [fact]) == "earnings_fact"
    assert classify_item("news", "Some Headline", (), [EarningsFacts("none", {})]) == "other"


@pytest.mark.parametrize(
    ("items", "itype"),
    [
        (["2.02", "9.01"], "earnings_8k"),
        (["5.02", "7.01", "9.01"], "management_change"),
        (["1.01", "1.02", "2.03", "9.01"], "agreement_terminated"),  # unclear beats neutral
        (["2.02", "4.02"], "restatement"),  # structural beats earnings
        (["4.01", "3.02"], "dilution"),  # structural beats unclear
        (["2.02", "5.02"], "earnings_8k"),  # an earnings release beats the neutrals
        (["2.01", "1.01"], "ma_closed"),
        (["8.01", "9.01"], "filing_other"),
        (["5.07"], "filing_other"),
        (["3.03"], "filing_other"),  # an item code with no type
        ([], "filing_other"),
    ],
)
def test_an_8k_is_typed_by_its_items(items: list[str], itype: str) -> None:
    assert classify_item("8-k", "", items, ()) == itype
    assert classify_item("8-k/a", "", items, ()) == itype


def test_an_8k_ignores_any_headline() -> None:
    assert classify_item("8-k", "Albertsons Terminates Merger Agreement", ["2.02"], ()) == (
        "earnings_8k"
    )


# ── Earnings verdicts ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("fields", "metric", "verdict"),
    [
        ({"eps_surprise": 0.10, "eps_verdict": "beats"}, "eps", 1),
        ({"eps_surprise": -0.10, "eps_verdict": "misses"}, "eps", -1),
        # The dead band overrides the word: a 0.2% "beat" is in line.
        ({"eps_surprise": 0.002, "eps_verdict": "beats"}, "eps", 0),
        ({"sales_surprise": -0.004, "sales_verdict": "vs"}, "sales", 0),
        # On the band (a stated $2.01 vs $2.00 is 0.00499... in binary).
        ({"eps_surprise": (2.01 - 2.00) / 2.00, "eps_verdict": "vs"}, "eps", 1),
        ({"eps_surprise": (1.99 - 2.00) / 2.00, "eps_verdict": "vs"}, "eps", -1),
        ({"eps_surprise": 0.0049, "eps_verdict": "vs"}, "eps", 0),
        # No surprise: the verdict word decides.
        ({"eps_surprise": None, "eps_verdict": "beats"}, "eps", 1),
        ({"sales_surprise": None, "sales_verdict": "miss"}, "sales", -1),
        ({"eps_surprise": None, "eps_verdict": "in-line with"}, "eps", 0),
        ({"eps_verdict": "inline"}, "eps", 0),
        # Year on year, or a bare "vs" without an estimate, decides nothing.
        ({"eps_surprise": None, "eps_verdict": "up from"}, "eps", None),
        ({"eps_surprise": None, "eps_verdict": "vs."}, "eps", None),
        ({}, "sales", None),
    ],
)
def test_metric_verdict(fields: dict[str, object], metric: str, verdict: int | None) -> None:
    assert metric_verdict(fields, metric) == verdict  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("headline", "want"),
    [
        # Pre-2018 "vs" wires: the numbers decide, with the 0.5% dead band
        # (Norfolk's sales -0.4% and Target's -0.06% are in line).
        (
            "Norfolk Southern Reports Q4 EPS $1.42 vs $1.36 Est., Sales $2.49B vs $2.5B Est.",
            (1, 0, "earnings_beat"),
        ),
        (
            "Las Vegas Sands Reports Q4 Adj. EPS $0.62 vs $0.66 Est., Sales $3.08B vs $3.11B Est.",
            (-1, -1, "earnings_miss"),
        ),
        (
            "General Mills Reports Q2 EPS $0.82 vs $0.82 Est., Sales $4.2B vs $4.09B Est.",
            (0, 1, "earnings_beat"),
        ),
        (
            "Target Q2 EPS $1.23 vs $1.12 est, Revenue $16.17B vs $16.18B est",
            (1, 0, "earnings_beat"),
        ),
        ("Atmos Energy Reports Q4 Adj. EPS $0.34 vs. $0.34 Est.", (0, None, "earnings_inline")),
        (
            "TripAdvisor Reports Q4 EPS $0.16 vs. Est. $0.31, Rev. $316M vs. Est. $326M",
            (-1, -1, "earnings_miss"),
        ),
        # Estimate first for both: the EPS beats, the sales miss (-0.5%).
        (
            "21st Century Fox Reports Q2 EPS $0.53 vs. Est. $0.49, Rev. $7.68B vs. Est. $7.72B",
            (1, -1, "earnings_miss"),
        ),
        # A mixed print is a miss.
        (
            "Trade Desk Q1 EPS $0.45 Misses $0.77 Estimate, Sales $219.80M Beat $216.90M Estimate",
            (-1, 1, "earnings_miss"),
        ),
        (
            "Helen of Troy Reports Q1 Adj. EPS $1.27 vs $1.12 Est., Sales $347.938M vs $356M Est.",
            (1, -1, "earnings_miss"),
        ),
        ("Bloom Energy Q4 Sales $213.8M Miss $270.07M Estimate", (None, -1, "earnings_miss")),
        # Year on year only: result facts with no verdict.
        (
            "Kandi Technologies Gr H1 EPS $0.10 Up From $0.02 YoY, Sales $57.117M Up From "
            "$36.291M YoY",
            (None, None, "earnings_inline"),
        ),
    ],
)
def test_one_wire_verdicts(headline: str, want: tuple[int | None, int | None, str]) -> None:
    v = earnings_verdict([news(1, 0, headline)], LATER)
    assert v is not None and v.guide is None
    assert (v.eps, v.sales, v.type) == want


def test_no_facts_no_verdict() -> None:
    assert earnings_verdict([], LATER) is None
    assert earnings_verdict([filing(1, 0, "2.02", "9.01")], LATER) is None
    assert (
        earnings_verdict([news(1, 0, "Raymond James Downgrades Comcast to Market Perform")], LATER)
        is None
    )


def _guidance(**fields: object) -> Item:
    return Item(9, T0, "earnings_fact", (EarningsFacts("guidance", dict(fields)),))


@pytest.mark.parametrize(
    ("fields", "guide", "story"),
    [
        ({"action": "lowers", "surprise": 0.05}, "down", "guidance_cut"),
        ({"action": "cuts", "surprise": None}, "down", "guidance_cut"),
        ({"action": "withdraws"}, "down", "guidance_cut"),
        ({"action": "suspends"}, "down", "guidance_cut"),
        # Raised, yet still below consensus: a cut.
        ({"action": "raises", "surprise": -0.02}, "down", "guidance_cut"),
        ({"action": "sees", "surprise": -0.01}, "down", "guidance_cut"),
        ({"action": "raises", "surprise": -0.005}, "up", "guidance_raise"),
        ({"action": "sees", "surprise": 0.01}, "up", "guidance_raise"),
        ({"action": "expects", "surprise": 0.009}, "inline", "guidance_inline"),
        ({"action": "affirms", "surprise": None}, "inline", "guidance_inline"),
    ],
)
def test_guidance_only(fields: dict[str, object], guide: str, story: str) -> None:
    v = earnings_verdict([_guidance(**fields)], LATER)
    assert v is not None and (v.eps, v.sales, v.guide, v.type) == (None, None, guide, story)


def test_real_guidance_wires() -> None:
    def verdict(h: str) -> tuple[str | None, str]:
        v = earnings_verdict([news(1, 0, h)], LATER)
        assert v is not None
        return v.guide, v.type

    assert verdict("Brunswick Raises FY21 Adj. EPS Guidance To $7.30-$7.60 vs $6.44 Estimate") == (
        "up",
        "guidance_raise",
    )
    assert verdict(
        "General Motors Narrows FY22 EPS Guidance From $6.50-$7.50 To $6.75-$7.25 Vs. $7.19 Est."
    ) == ("down", "guidance_cut")
    assert verdict(
        "Abbott Expects Q2 2025 Adjusted EPS Of $1.23 to $1.27 Versus Consensus Of $1.25"
    ) == ("inline", "guidance_inline")


MISS = "Trade Desk Q1 EPS $0.45 Misses $0.77 Estimate, Sales $219.80M Beat $216.90M Estimate"


def test_a_miss_with_guidance() -> None:
    up = earnings_verdict([news(1, 0, MISS), _guidance(action="raises", surprise=0.03)], LATER)
    assert up is not None and (up.guide, up.type) == ("up", "earnings_miss_guide_up")
    unk = earnings_verdict(
        [
            news(1, 0, MISS),
            news(
                2,
                1,
                "Ulta Sees Q1 Rev. $1.016B-$1.033B vs. Est. $1.01B, EPS $1.25-$1.30 vs. Est. $1.22",
            ),
        ],
        LATER,
    )
    assert unk is not None and (unk.guide, unk.type) == ("unknown", "earnings_miss_guide_unk")
    down = earnings_verdict([news(1, 0, MISS), _guidance(action="sees", surprise=-0.05)], LATER)
    assert down is not None and (down.guide, down.type) == ("down", "guidance_cut")
    # A guidance-cut headline counts as guidance down.
    cut = earnings_verdict(
        [
            news(1, 0, MISS),
            news(2, 1, "McDonald's Withdraws Guidance After Significant Sales Decline In March"),
        ],
        LATER,
    )
    assert cut is not None and (cut.guide, cut.type) == ("down", "guidance_cut")
    # Unparsed guidance next to a parsed guidance fact is not "unknown".
    both = earnings_verdict(
        [
            news(1, 0, MISS),
            news(
                2,
                1,
                "Ulta Sees Q1 Rev. $1.016B-$1.033B vs. Est. $1.01B, EPS $1.25-$1.30 vs. Est. $1.22",
            ),
            _guidance(action="affirms", surprise=0.0),
        ],
        LATER,
    )
    assert both is not None and (both.guide, both.type) == ("inline", "earnings_miss")


def test_a_beat_with_unparsed_guidance_stays_a_beat() -> None:
    v = earnings_verdict(
        [
            news(1, 0, "Micron Technology Q4 Adj. EPS $33.42 Beats $31.45 Estimate"),
            news(
                2,
                1,
                "Ulta Sees Q1 Rev. $1.016B-$1.033B vs. Est. $1.01B, EPS $1.25-$1.30 vs. Est. $1.22",
            ),
        ],
        LATER,
    )
    assert v is not None and (v.eps, v.guide, v.type) == (1, "unknown", "earnings_beat")


def test_adjusted_basis_is_preferred_whatever_the_order() -> None:
    gaap = (
        "Diamondback Energy Q2 EPS $1.71 Misses $2.12 Estimate, Sales $1.68B Beat $1.32B Estimate"
    )
    adj = "CORRECTION: Diamondback Energy Q2 Adj. EPS $2.40 Beats $2.12 Estimate"
    for first, second in ((gaap, adj), (adj, gaap)):
        v = earnings_verdict([news(1, 0, first), news(2, 5, second)], LATER)
        assert v is not None
        # EPS from the adjusted fact; sales only the GAAP wire states.
        assert (v.eps, v.sales, v.type) == (1, 1, "earnings_beat")


def test_the_latest_fact_per_metric_decides() -> None:
    early = "Las Vegas Sands Reports Q4 Adj. EPS $0.62 vs $0.66 Est., Sales $3.08B vs $3.11B Est."
    late = "Las Vegas Sands Q4 Adj. EPS $0.67 Beats $0.66 Estimate"  # constructed re-wire
    v = earnings_verdict([news(1, 0, early), news(2, 5, late)], LATER)
    assert v is not None and (v.eps, v.sales) == (1, -1)
    # Before the second wire is available, the first decides alone.
    v0 = earnings_verdict([news(1, 0, early), news(2, 5, late)], T0 + timedelta(minutes=1))
    assert v0 is not None and (v0.eps, v0.sales) == (-1, -1)


def test_noise_items_lend_no_facts() -> None:
    noise = news(
        1,
        0,
        "FedEx Maintains Fiscal Year Adjusted Earnings Outlook; Sees FY24 EPS $17.00-$18.50 Vs "
        "$18.25 Est.; Capital Spending Of $5.7B",
    )
    assert noise.itype == "noise" and noise.facts
    assert earnings_verdict([noise], LATER) is None


# ── Corrections ─────────────────────────────────────────────────────────────

TTD = "Trade Desk Q1 EPS $0.45 Misses $0.77 Estimate, Sales $219.80M Beat $216.90M Estimate"
TTD_FIX = (
    "CORRECTION: Trade Desk Q1 EPS $1.41 Beats $0.77 Estimate, Sales $219.81M Beat $216.90M "
    "Estimate"
)


def test_a_correction_supersedes_from_its_own_availability() -> None:
    # 2021-05-10: the wire at 12:32 read a miss; the correction at 13:32 a beat.
    items = [news(1, 0, TTD), news(2, 60, TTD_FIX, supersedes=[1])]
    before = resolve(items, T0 + timedelta(minutes=59), follower=False)
    assert before.type == "earnings_miss" and before.family == "NSN_CORE"
    assert before.earnings is not None and before.earnings.eps == -1
    after = resolve(items, T0 + timedelta(minutes=60), follower=False)
    assert after.type == "earnings_beat" and after.family is None
    assert after.earnings is not None and (after.earnings.eps, after.earnings.sales) == (1, 1)


def test_without_supersession_the_latest_wire_still_decides() -> None:
    # The builder marks supersession; without it the later EPS still wins per metric.
    items = [news(1, 0, TTD), news(2, 60, TTD_FIX)]
    assert resolve(items, LATER, follower=False).type == "earnings_beat"


def test_a_correction_without_numbers_leaves_the_earnings_unread() -> None:
    items = [
        news(1, 0, "Progressive Q4 2023 GAAP EPS $3.37 Beats $2.56 Estimate"),
        news(
            2, 5, "CORRECTION: Progressive Q4 2023 GAAP EPS $3.37, Dec. EPS $1.53", supersedes=[1]
        ),
    ]
    assert resolve(items, T0 + timedelta(minutes=1), follower=False).type == "earnings_beat"
    card = resolve(items, LATER, follower=False)
    assert card.earnings is None and card.type == "earnings_unparsed"


def test_a_correction_flipping_a_beat_into_a_miss() -> None:
    items = [
        news(1, 0, "Alibaba Group Holding Q4 Adj. EPS $1.88 Beats $1.78 Estimate"),
        news(2, 11, "CORRECTION: Alibaba Q4 Adj. EPS $1.58 Misses $1.78 Estimate", supersedes=[1]),
    ]
    assert resolve(items, T0 + timedelta(minutes=5), follower=False).family is None
    late = resolve(items, LATER, follower=False)
    assert late.type == "earnings_miss" and late.family == "NSN_CORE"


# ── Story resolution ────────────────────────────────────────────────────────

DOWNGRADE = "Raymond James Downgrades Comcast to Market Perform"
MOVER = "Rhythm Pharma Shares Up 26% Upon Resumption"
OFFERING = "Beam Therapeutics Announces Proposed Public Offering Of Common Stock Of 4.5M Shares"


def test_a_downgrade_alone_is_the_h1_family() -> None:
    card = resolve([news(1, 0, DOWNGRADE)], LATER, follower=False)
    assert card == tx.StoryCard(
        type="analyst_downgrade",
        direction="neg",
        permanence="transient",
        family="NSN_CORE",
        vetoes=(),
        positives=(),
        earnings=None,
        follower=False,
        structural=False,
    )


def test_an_earnings_miss_is_the_h1_family() -> None:
    card = resolve([news(1, 0, MISS)], LATER, follower=False)
    assert card.type == "earnings_miss" and card.family == "NSN_CORE"
    assert card.earnings == tx.EarningsVerdict(-1, 1, None, "earnings_miss")


def test_a_follower_is_never_the_family() -> None:
    card = resolve([news(1, 0, DOWNGRADE)], LATER, follower=True)
    assert card.type == "analyst_downgrade" and card.follower and card.family is None


def test_a_story_led_by_a_mover_is_not_the_family() -> None:
    led = resolve([news(1, 0, MOVER), news(2, 5, DOWNGRADE)], LATER, follower=False)
    assert led.type == "analyst_downgrade" and led.family is None
    # A mover after the cause does not matter; noise before it is ignored.
    after = resolve(
        [
            news(0, -5, "Understanding Vistra's Unusual Options Activity"),
            news(1, 0, DOWNGRADE),
            news(2, 5, MOVER),
        ],
        LATER,
        follower=False,
    )
    assert after.family == "NSN_CORE"


def test_a_positive_item_keeps_the_story_out() -> None:
    card = resolve(
        [news(1, 0, DOWNGRADE), news(2, 5, "JP Morgan Upgrades Rambus To Overweight")],
        LATER,
        follower=False,
    )
    assert card.type == "analyst_downgrade" and card.positives == ("analyst_upgrade",)
    assert card.family is None


def test_an_unclear_negative_vetoes_but_does_not_type_a_downgrade() -> None:
    card = resolve(
        [
            news(1, 0, DOWNGRADE),
            news(2, 5, "Apple Sued Over Not Taking Down Telegram After Capitol Hill Riot"),
        ],
        LATER,
        follower=False,
    )
    assert card.type == "analyst_downgrade" and card.vetoes == ("legal_adverse",)
    assert card.family is None and not card.structural


def test_structural_news_types_the_story_from_when_it_is_known() -> None:
    items = [news(1, 0, DOWNGRADE), news(2, 60, OFFERING)]
    early = resolve(items, T0 + timedelta(minutes=30), follower=False)
    assert early.family == "NSN_CORE" and not early.structural
    late = resolve(items, T0 + timedelta(minutes=60), follower=False)
    assert late.type == "dilution" and late.permanence == "structural"
    assert late.vetoes == ("dilution",) and late.structural and late.family is None


def test_the_earliest_structural_item_types_the_story() -> None:
    items = [
        news(1, 0, OFFERING),
        news(2, 5, "Albertsons Terminates Merger Agreement With Kroger"),
        filing(3, 1, "4.02", "9.01"),
    ]
    card = resolve(items, LATER, follower=False)
    assert card.type == "dilution" and card.vetoes == ("dilution", "restatement", "ma_break")


def test_structural_beats_earnings_and_earnings_beat_the_analysts() -> None:
    assert resolve([news(1, 0, MISS), news(2, 5, OFFERING)], LATER, follower=False).type == (
        "dilution"
    )
    card = resolve([news(1, 0, DOWNGRADE), news(2, 5, MISS)], LATER, follower=False)
    assert card.type == "earnings_miss" and card.family == "NSN_CORE"


def test_guidance_below_consensus_is_a_structural_veto() -> None:
    cut = news(
        2,
        5,
        "General Motors Narrows FY22 EPS Guidance From $6.50-$7.50 To $6.75-$7.25 Vs. $7.19 Est.",
    )
    card = resolve([news(1, 0, MISS), cut], LATER, follower=False)
    assert card.type == "guidance_cut" and card.permanence == "structural"
    assert card.vetoes == ("guidance_cut",) and card.structural and card.family is None


def test_a_miss_with_raised_guidance_is_not_the_family() -> None:
    raise_ = news(2, 5, "Brunswick Raises FY21 Adj. EPS Guidance To $7.30-$7.60 vs $6.44 Estimate")
    card = resolve([news(1, 0, MISS), raise_], LATER, follower=False)
    assert card.type == "earnings_miss_guide_up" and card.family is None
    assert card.direction == "neutral" and card.vetoes == ()


def test_unread_earnings() -> None:
    assert (
        resolve([filing(1, 0, "2.02", "9.01")], LATER, follower=False).type == "earnings_unparsed"
    )
    unparsed = news(
        1, 0, "Ulta Sees Q1 Rev. $1.016B-$1.033B vs. Est. $1.01B, EPS $1.25-$1.30 vs. Est. $1.22"
    )
    assert resolve([unparsed], LATER, follower=False).type == "earnings_unparsed"
    # An earnings release outranks the analysts that follow it.
    card = resolve([filing(1, 0, "2.02"), news(2, 5, DOWNGRADE)], LATER, follower=False)
    assert card.type == "earnings_unparsed" and card.family is None


def test_unclear_negatives_in_their_order() -> None:
    items = [
        typed(1, 0, "legal_adverse"),
        typed(2, 1, "management_exit"),
        typed(3, 2, "antitrust_regulatory"),
    ]
    card = resolve(items, LATER, follower=False)
    assert card.type == "antitrust_regulatory"
    assert card.vetoes == ("legal_adverse", "management_exit", "antitrust_regulatory")


def test_price_target_cuts_and_negative_initiations() -> None:
    pt = "Barclays Maintains Equal-Weight on Workday, Lowers Price Target to $128"
    card = resolve([news(1, 0, pt)], LATER, follower=False)
    assert card.type == "analyst_pt_cut" and card.family is None  # not in H1
    init = (
        "Goldman Sachs Initiates Coverage On Advance Auto Parts with Sell Rating, Announces "
        "$115 Price Target"
    )
    assert resolve([news(1, 0, init)], LATER, follower=False).type == "analyst_pt_cut"
    # A downgrade outranks a price-target cut.
    both = resolve([news(1, 0, pt), news(2, 5, DOWNGRADE)], LATER, follower=False)
    assert both.type == "analyst_downgrade"


def test_positives_in_their_order() -> None:
    items = [
        typed(1, 0, "analyst_pt_raise"),
        typed(2, 1, "buyback"),
        typed(3, 2, "analyst_upgrade"),
    ]
    card = resolve(items, LATER, follower=False)
    assert card.type == "analyst_upgrade" and card.direction == "pos"
    assert card.positives == ("analyst_upgrade",)


def test_the_most_frequent_neutral_type_earliest_on_ties() -> None:
    items = [typed(1, 0, "dividend"), typed(2, 1, "other"), typed(3, 2, "other")]
    assert resolve(items, LATER, follower=False).type == "other"
    tie = [
        typed(1, 0, "dividend"),
        typed(2, 1, "ma_acquirer"),
        typed(3, 2, "ma_acquirer"),
        typed(4, 3, "dividend"),
    ]
    assert resolve(tie, LATER, follower=False).type == "dividend"
    filings = [filing(1, 0, "1.01", "9.01"), filing(2, 1, "8.01")]
    assert resolve(filings, LATER, follower=False).type == "agreement"


def test_stories_of_reactive_and_noise_items() -> None:
    assert resolve([], LATER, follower=False).type == "noise_only"
    noise_only = [
        news(1, 0, "Understanding Vistra's Unusual Options Activity"),
        typed(2, 1, "law_firm"),
    ]
    assert resolve(noise_only, LATER, follower=False).type == "noise_only"
    other = (
        "Citigroup Initiates Coverage On VICI Properties with Neutral Rating, Announces $22 "
        "Price Target"
    )
    assert (
        resolve([news(1, 0, MOVER), news(2, 1, other)], LATER, follower=False).type == "mover_only"
    )
    assert resolve([news(1, 0, other)], LATER, follower=False).type == "analyst_other"
    assert resolve([news(1, 0, MOVER)], LATER, follower=False).family is None


def test_nothing_after_t_is_read() -> None:
    items = [news(1, 0, DOWNGRADE), news(2, 60, OFFERING)]
    assert resolve(items, T0 - timedelta(seconds=1), follower=False).type == "noise_only"
    assert resolve(items, T0, follower=False).type == "analyst_downgrade"  # available_at <= t


# ── Pins ────────────────────────────────────────────────────────────────────


def test_taxonomy_sha_pins_every_pattern_and_table() -> None:
    src = tx.sources()
    for name in (
        "LAW_FIRM",
        "NOISE",
        "MOVER",
        "DOWNGRADE",
        "UPGRADE",
        "PT_CUT",
        "PT_RAISE",
        "INIT_NEG",
        "GUIDE_UNPARSED",
        "RESTATEMENT",
        "ANTITRUST_REGULATORY",
        "FRAUD_PROBE",
        "GUIDANCE_CUT",
        "MA_TARGET",
        "DIVIDEND",
    ):
        assert name in src
    assert src["FRAUD_PROBE"] == tx.FRAUD_PROBE.pattern
    for name, pattern in headline_patterns.sources().items():
        assert src[name] == pattern
    assert json.loads(src["ITEM_TYPES"]) == tx.ITEM_TYPES
    assert json.loads(src["REACTIVE"]) == sorted(tx.REACTIVE)
    assert json.loads(src["TYPES"])["guidance_cut"] == ["neg", "structural"]
    assert json.loads(src["RULES_VERSION"]) == tx.RULES_VERSION == "rules-v1"
    want = hashlib.sha256(json.dumps(src, sort_keys=True).encode()).hexdigest()[:12]
    assert tx.TAXONOMY_SHA == want and len(want) == 12
