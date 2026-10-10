"""Benzinga earnings headlines -> results and guidance vs consensus (real headlines)."""

from __future__ import annotations

import pytest

from halal_trader.events.earnings_parse import money, parse_headline


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("$1.244B", 1.244e9),
        ("$(1.26)", -1.26),
        ("$756.000M", 7.56e8),
        ("$0.48", 0.48),
        ("$33.42", 33.42),
    ],
)
def test_money(text: str, value: float) -> None:
    assert money(text) == pytest.approx(value)


def test_a_double_beat() -> None:
    (r,) = parse_headline(
        "Micron Technology Q4 Adj. EPS $33.42 Beats $31.45 Estimate, "
        "Sales $54.229B Beat $50.751B Estimate"
    )
    f = r.fields
    assert r.kind == "result" and f["period"] == "Q4" and f["basis"] == "adj"
    assert f["eps"] == 33.42 and f["eps_estimate"] == 31.45
    assert f["eps_surprise"] == pytest.approx((33.42 - 31.45) / 31.45)
    assert f["sales"] == pytest.approx(54.229e9) and f["sales_verdict"] == "beat"


def test_a_loss_that_misses() -> None:
    (r,) = parse_headline(
        "Cal-Maine Foods Q1 EPS $(1.26) Misses $(0.64) Estimate, "
        "Sales $539.607M Miss $574.815M Estimate"
    )
    assert r.fields["eps"] == -1.26 and r.fields["eps_estimate"] == -0.64
    assert r.fields["eps_surprise"] == pytest.approx((-1.26 + 0.64) / 0.64)
    assert r.fields["basis"] == "gaap"


def test_year_over_year_is_not_a_surprise() -> None:
    (r,) = parse_headline(
        "Kandi Technologies Gr H1 EPS $0.10 Up From $0.02 YoY, Sales $57.117M Up From $36.291M YoY"
    )
    assert r.fields["eps_surprise"] is None and r.fields["sales_surprise"] is None


def test_guidance_ranges_against_consensus_and_a_period_carried_to_the_next_segment() -> None:
    eps, sales = parse_headline(
        "Micron Technology Sees Q1 Adj EPS $37.15-$39.15 vs $35.07 Est; "
        "Sees Sales $60.000B-$63.000B vs $56.553B Est"
    )
    assert eps.kind == "guidance" and eps.fields["metric"] == "eps" and eps.fields["period"] == "Q1"
    assert eps.fields["mid"] == pytest.approx(38.15)
    assert eps.fields["surprise"] == pytest.approx((38.15 - 35.07) / 35.07)
    assert sales.fields["metric"] == "sales" and sales.fields["period"] == "Q1"


@pytest.mark.parametrize(
    ("headline", "action", "mid", "estimate"),
    [
        (
            "Carnival Raises FY2026 Adj EPS Guidance from $2.22 to $2.24 vs $2.22 Est",
            "raises",
            2.24,
            2.22,
        ),
        (
            "Cardinal Health Affirms FY2027 Adj EPS Of $12.40-$12.60 vs $12.04 Est",
            "affirms",
            12.50,
            12.04,
        ),
        ("Carnival Sees Q4 Adj EPS $0.20 vs $0.26 Est", "sees", 0.20, 0.26),
        ("Nike Sees FY2027 Adj EPS $1.15-$1.35 vs $1.69 Est", "sees", 1.25, 1.69),
        (
            "Conduent Affirms FY2026 Sales Guidance of $2.150B-$2.250B vs $2.315B Est",
            "affirms",
            2.2e9,
            2.315e9,
        ),
    ],
)
def test_guidance_forms(headline: str, action: str, mid: float, estimate: float) -> None:
    (g,) = parse_headline(headline)
    assert g.fields["action"] == action
    assert g.fields["mid"] == pytest.approx(mid) and g.fields["estimate"] == pytest.approx(estimate)


@pytest.mark.parametrize(
    "headline",
    [
        "Micron Delivers Q4 Double Beat, Sees 'Even Stronger' Year Ahead",
        "Barclays Maintains Equal-Weight on PepsiCo, Lowers Price Target to $133",
        "Conduent Sees FY2028 Sales $2.250B-$2.350B",  # no consensus stated
        "ACI Worldwide And Cognizant Say BASE24-eps Card Software Ran Over Double Target Load",
    ],
)
def test_headlines_without_a_stated_comparison_yield_nothing(headline: str) -> None:
    assert parse_headline(headline) == []


def test_qualified_guidance() -> None:
    eps, revenue = parse_headline(
        "Cigna Group Affirms FY2026 Adj EPS Of At Least $30.45 Vs $30.41 Est; "
        "Affirms FY2026 Revenue Of ~$280.000B Vs $285.608B Est"
    )
    assert eps.fields["mid"] == 30.45 and eps.fields["qualifier"] == "at least"
    assert revenue.fields["metric"] == "sales" and revenue.fields["mid"] == pytest.approx(2.8e11)


async def test_extraction_stores_facts_once_and_marks_the_rest(engine) -> None:
    from datetime import UTC, datetime

    from sqlalchemy import text

    from halal_trader.events.earnings_parse import extract_all
    from halal_trader.events.store import EventRecord, EventRecorder

    t = datetime(2026, 9, 25, 20, tzinfo=UTC)
    await EventRecorder(engine).record(
        [
            EventRecord(
                "alpaca",
                "1",
                "news",
                "MU",
                t,
                t,
                {"headline": "Micron Technology Q4 Adj. EPS $33.42 Beats $31.45 Estimate"},
            ),
            EventRecord(
                "alpaca", "2", "news", "MU", t, t, {"headline": "Micron Heads Into Earnings"}
            ),
        ]
    )
    assert await extract_all(engine) == 1
    assert await extract_all(engine) == 0
    async with engine.connect() as conn:
        kinds = sorted(r.kind for r in await conn.execute(text("SELECT kind FROM event_facts")))
    assert kinds == ["none", "result"]


@pytest.mark.parametrize(
    ("headline", "eps", "estimate", "sales_estimate"),
    [
        (
            "Helen of Troy Reports Q1 Adj. EPS $1.27 vs $1.12 Est., Sales $347.938M vs $356M Est.",
            1.27,
            1.12,
            356e6,
        ),
        ("Target Q2 EPS $1.23 vs $1.12 est, Revenue $16.17B vs $16.18B est", 1.23, 1.12, 16.18e9),
        (
            "American Water Works Reports Q4 Ajd. EPS $0.69 vs $0.66 Est., "
            "Sales $821M vs $843.36M Est.",
            0.69,
            0.66,
            843.36e6,
        ),
        (
            "Dolby Laboratories Q2 GAAP EPS $0.70 vs $0.52 Estimate, Adj. EPS $1.04",
            0.70,
            0.52,
            None,
        ),
    ],
)
def test_older_vs_estimate_formats(headline, eps, estimate, sales_estimate) -> None:
    (r,) = parse_headline(headline)
    assert r.fields["eps"] == eps and r.fields["eps_estimate"] == estimate
    if sales_estimate is not None:
        assert r.fields["sales_estimate"] == pytest.approx(sales_estimate)


def test_inline_means_a_zero_surprise_and_non_comparable_is_skipped() -> None:
    (r,) = parse_headline(
        "Williams Companies Q1 EPS $0.22, Inline, Sales $2.054B Miss $2.28B Estimate"
    )
    assert r.fields["eps_surprise"] == 0.0
    assert parse_headline("ManpowerGroup Q3 EPS $2.42 May Not Compare To $1.93 Estimate") == []


@pytest.mark.parametrize(
    ("headline", "period", "eps", "estimate"),
    [
        (
            "Oceaneering International Q4 2023 Adj EPS $0.19 Misses $0.25 Estimate, "
            "Sales $654.63M Beat $627.44M Estimate",
            "Q42023",
            0.19,
            0.25,
        ),
        (
            "Dolby Laboratories Q4 Adj $0.65 Beats $0.53 Estimate, "
            "Sales $290.56M Beat $290.20M Estimate",
            "Q4",
            0.65,
            0.53,
        ),
        (
            "Wolfspeed Q1 2024 Adj. EPS $(0.53) Beats $(0.67) Estimate, Revenue $197.4M Misses "
            "$207.64M Estimate",
            "Q12024",
            -0.53,
            -0.67,
        ),
    ],
)
def test_newer_formats(headline: str, period: str, eps: float, estimate: float) -> None:
    (r,) = parse_headline(headline)
    assert r.fields["period"] == period
    assert r.fields["eps"] == eps and r.fields["eps_estimate"] == estimate
    assert r.fields.get("sales_estimate") is not None


# ── v4 ──────────────────────────────────────────────────────────────────────


def test_v4_is_the_current_extractor_and_v3_stays_named() -> None:
    from halal_trader.events.earnings_parse import (
        EXTRACTOR,
        EXTRACTOR_V3,
        EXTRACTOR_V4,
        PARSER_SHA,
    )

    # The stored label names the parser that read the facts.
    assert EXTRACTOR == f"benzinga-earnings-v4+{PARSER_SHA}"
    assert EXTRACTOR_V4 == "benzinga-earnings-v4"
    assert EXTRACTOR_V3 == "benzinga-earnings-v3"


@pytest.mark.parametrize(
    ("headline", "period", "sales", "estimate", "verdict"),
    [
        # v3 read nothing here: a sales figure alone is now a sales result.
        ("Bullish Q2 Sales $92.600M Beat $87.810M Estimate", "Q2", 92.6e6, 87.81e6, "beat"),
        ("Bloom Energy Q4 Sales $213.8M Miss $270.07M Estimate", "Q4", 213.8e6, 270.07e6, "miss"),
        ("Lockheed Martin Q4 Sales $18.60B Miss $18.91B Estimate", "Q4", 18.6e9, 18.91e9, "miss"),
    ],
)
def test_a_sales_only_headline_is_a_sales_result_without_eps(
    headline: str, period: str, sales: float, estimate: float, verdict: str
) -> None:
    (r,) = parse_headline(headline)
    f = r.fields
    assert r.kind == "result" and f["period"] == period
    assert "eps" not in f and "eps_verdict" not in f
    assert f["sales"] == pytest.approx(sales) and f["sales_estimate"] == pytest.approx(estimate)
    assert f["sales_surprise"] == pytest.approx((sales - estimate) / estimate)
    assert f["sales_verdict"] == verdict and f["basis"] == "gaap"


def test_the_estimate_stated_before_its_number() -> None:
    (r,) = parse_headline(
        "TripAdvisor Reports Q4 EPS $0.16 vs. Est. $0.31, Rev. $316M vs. Est. $326M"
    )
    f = r.fields
    assert r.kind == "result" and f["period"] == "Q4" and f["basis"] == "gaap"
    assert f["eps"] == 0.16 and f["eps_estimate"] == 0.31 and f["eps_verdict"] == "vs"
    assert f["eps_surprise"] == pytest.approx((0.16 - 0.31) / 0.31)
    # The sales, estimate first too (SALES_EST_FIRST).
    assert f["sales"] == pytest.approx(316e6) and f["sales_estimate"] == pytest.approx(326e6)
    assert f["sales_surprise"] == pytest.approx((316 - 326) / 326)
    assert f["sales_verdict"] == "vs"


def test_the_estimate_first_with_a_capital_vs() -> None:
    (r,) = parse_headline(
        "Crown Holdings Reports Q1 EPS $0.57 Vs Est $0.63, Sales $1.89B Vs Est $2.03B"
    )
    assert r.fields["eps"] == 0.57 and r.fields["eps_estimate"] == 0.63
    assert r.fields["sales"] == pytest.approx(1.89e9)
    assert r.fields["sales_estimate"] == pytest.approx(2.03e9)


@pytest.mark.parametrize(
    ("headline", "sales", "estimate"),
    [
        # A mixed 2016-17 print: the EPS beats, the sales miss.
        ("21st Century Fox Reports Q2 EPS $0.53 vs. Est. $0.49, Rev. $7.68B vs. Est. $7.72B",
         7.68e9, 7.72e9),
        ("Amazon Reports Q2 EPS $1.78 Vs Est $1.11, Revs $30.4B Vs Est $29.54B", 30.4e9, 29.54e9),
        ("Arrowhead Pharmaceuticals Reports Q3 EPS $(0.32) vs. Est. $(0.37), "
         "Rev. $39.58k  vs. Est. $60K", 39.58e3, 60e3),
    ],
)  # fmt: skip
def test_the_sales_after_an_estimate_first_eps(
    headline: str, sales: float, estimate: float
) -> None:
    (r,) = parse_headline(headline)
    assert r.fields["sales"] == pytest.approx(sales)
    assert r.fields["sales_estimate"] == pytest.approx(estimate)
    assert r.fields["sales_surprise"] == pytest.approx((sales - estimate) / estimate)


@pytest.mark.parametrize(
    ("headline", "action", "metric", "period", "low", "high", "estimate"),
    [
        (
            "Kennametal Expects Q1 Sales Of $465M-$485M Vs $485.8M Est, "
            "Adjusted EPS Of $0.20-$0.30 Vs $0.29 Est",
            "expects",
            "sales",
            "Q1",
            465e6,
            485e6,
            485.8e6,
        ),
        (
            "Abbott Expects Q2 2025 Adjusted EPS Of $1.23 to $1.27 Versus Consensus Of $1.25",
            "expects",
            "eps",
            "Q22025",
            1.23,
            1.27,
            1.25,
        ),
        (
            "Amgen Narrows FY2026 GAAP EPS Guidance from $15.62-$17.10 to $15.80-$17.08 "
            "vs $15.44 Est",
            "narrows",
            "eps",
            "FY2026",
            15.80,
            17.08,
            15.44,
        ),
        (  # the new range, not the one it replaces
            "General Motors Narrows FY22 EPS Guidance From $6.50-$7.50 To $6.75-$7.25 "
            "Vs. $7.19 Est.",
            "narrows",
            "eps",
            "FY22",
            6.75,
            7.25,
            7.19,
        ),
        (
            "Freshworks Expects Q3 Revenue Of $180M - $183M (Est $178.7M), "
            "Adj EPS Of $0.07 - $0.08 (Est $0.08)",
            "expects",
            "sales",
            "Q3",
            180e6,
            183e6,
            178.7e6,
        ),
        (
            "Cognex Corp Expects Q1 Revenue of $190M-$205M Vs $211.56M Est",
            "expects",
            "sales",
            "Q1",
            190e6,
            205e6,
            211.56e6,
        ),
        (
            "Brunswick Raises FY21 Adj. EPS Guidance To $7.30-$7.60 vs $6.44 Estimate",
            "raises",
            "eps",
            "FY21",
            7.30,
            7.60,
            6.44,
        ),
    ],
)
def test_guidance_in_other_words(
    headline: str,
    action: str,
    metric: str,
    period: str,
    low: float,
    high: float,
    estimate: float,
) -> None:
    g = parse_headline(headline)[0]
    f = g.fields
    assert g.kind == "guidance" and f["action"] == action and f["metric"] == metric
    assert f["period"] == period
    assert f["low"] == pytest.approx(low) and f["high"] == pytest.approx(high)
    assert f["estimate"] == pytest.approx(estimate)
    mid = (low + high) / 2
    assert f["mid"] == pytest.approx(mid) and f["surprise"] == pytest.approx(
        (mid - estimate) / estimate
    )


def test_a_guidance_basis_is_read_before_its_metric() -> None:
    (g,) = parse_headline(
        "Amgen Narrows FY2026 GAAP EPS Guidance from $15.62-$17.10 to $15.80-$17.08 vs $15.44 Est"
    )
    assert g.fields["basis"] == "gaap"
    (g,) = parse_headline(
        "Abbott Expects Q2 2025 Adjusted EPS Of $1.23 to $1.27 Versus Consensus Of $1.25"
    )
    assert g.fields["basis"] == "adjusted"


@pytest.mark.parametrize(
    "headline",
    [
        "AMD Sees Q1 2025 Revenue $7.1B +/- $300M Vs $6.995B Est.",
        "Applied Materials Sees Q2 EPS $2.30+/- $0.18 Vs $2.30 Est.",
        "Sees Q1 EPS $(1.07) ± $0.07 Vs $(0.99) Est.",
        # Glued: GUIDE_V4's figure starts at the tolerance's own "-" (-$300M).
        "Lam Research Expects Q4 Revenue Of $3.8B +/-$300M (Est $3.772B); Adj EPS Of $7.50 "
        "+/- $0.75 (Est $7.33)",
    ],
)
def test_a_tolerance_is_never_read_as_the_guided_level(headline: str) -> None:
    # GUIDE_V4 alone reads the "+/-" width ($300M vs $6.995B: -96%), which would
    # type an in-line guide a guidance cut. The read is dropped: guidance unknown.
    from halal_trader.events.earnings_parse import GUIDE_UNPARSED, GUIDE_V4

    assert GUIDE_V4.search(headline) is not None
    assert parse_headline(headline) == []
    assert GUIDE_UNPARSED.search(headline) is not None


@pytest.mark.parametrize(
    ("headline", "action", "low", "high", "estimate"),
    [
        # GUIDE_V4 alone reads the "-$1.85" glued to the old range: -98%.
        ("ESCO Technologies Raises Preliminary Q2 Adj EPS Guidance from $1.75-$1.85 to $1.91 "
         "vs $1.77 Est", "raises", 1.91, 1.91, 1.77),
        ("Whirlpool Narrows FY2025 GAAP EPS Guidance from $5.00-$7.00 to $6.00 vs $5.32 Est; "
         "Affirms FY2025 Sales Guidance of $15.800B vs $15.491B Est", "narrows", 6.00, 6.00, 5.32),
        # ... the "1" of "Q1": +18%, where the guide is 14% below consensus.
        ("HP Expects Adj EPS Of $0.70 - $0.76 For Q1 (Est $0.85); $3.45 - $3.75 For FY25 "
         "(Est $3.60)", "expects", 0.70, 0.76, 0.85),
        # ... the range being replaced, which sits right before "vs".
        ("3M Raises FY24 Adj EPS Guidance To $7.00-$7.30 From $6.80-$7.30 vs. $7.17 Est",
         "raises", 7.00, 7.30, 7.17),
        ("ATI Lowers 2024 Guidance: Now Sees Adj. EPS Of $2.24-$2.30 (Prior $2.40-$2.60) Vs. "
         "$2.45 Est.", "lowers", 2.24, 2.30, 2.45),
        # "From X To Y" is the old guidance and the new, not a range.
        ("Axcelis Cuts Q2 Prelim. Sales Guidance From $80M To $75M vs $82.93M",
         "cuts", 75e6, 75e6, 82.93e6),
        ("Amneal Pharmaceuticals Raises FY19 Adj. EPS Guidance From ~$0.31 To $0.52-$0.62 vs "
         "$0.53 Est.", "raises", 0.52, 0.62, 0.53),
        # "By $0.05" is the change, not the guidance.
        ("Centene Raises FY22 Adj. EPS Guidance By $0.05 To $5.60-$5.75 vs $5.59 Est.",
         "raises", 5.60, 5.75, 5.59),
        # "Will Range From" introduces the guidance itself.
        ("Knight-Swift Transportation Expects Q4 Adjusted EPS Will Range From $0.32-$0.36 Vs "
         "$0.34 Est.", "expects", 0.32, 0.36, 0.34),
    ],
)  # fmt: skip
def test_a_guided_figure_is_never_a_fragment_nor_the_guidance_it_replaces(
    headline: str, action: str, low: float, high: float, estimate: float
) -> None:
    g = parse_headline(headline)[0]
    f = g.fields
    assert g.kind == "guidance" and f["action"] == action
    assert f["low"] == pytest.approx(low) and f["high"] == pytest.approx(high)
    mid = (low + high) / 2
    assert f["estimate"] == pytest.approx(estimate)
    assert f["surprise"] == pytest.approx((mid - estimate) / estimate)


@pytest.mark.parametrize(
    ("headline", "action", "low", "high", "estimate"),
    [
        # "$($0.57)" is a typo for $(0.57): read whole, never the "$0.57)" inside it (+0.57).
        ("Varonis Systems Sees EPS $($0.57)-($0.55) Vs. $(0.30) Est., Sales $59M-$60M Vs. "
         "$58.69M Est.", "sees", -0.57, -0.55, -0.30),
        # "Group $3.2B" ends in "up ", not the word "Up": the figure stands.
        ("Visteon Sees FY16 Sales for Electronics Products Group $3.2B vs $3.25B Est., Adj. "
         "EBITDA $305M-$335M, Adj. Free Cash Flow $110M-$150M", "sees", 3.2e9, 3.2e9, 3.25e9),
        # Prior or Previous names the guidance replaced in a few words first.
        ("Lowe's Companies Sees FY23 Adjusted EPS Of $13.00 Versus Prior Guidance Of "
         "$13.20-$13.60 Versus Consensus Of $13.33", "sees", 13.00, 13.00, 13.33),
        ("Super Micro Computer Sees Q2 2016 Sales $637M-$639M vs Previous Guidance $580M-$630M "
         "vs $600.1M Est", "sees", 637e6, 639e6, 600.1e6),
        ("Axon Sees FY23 Revenue ~$1.55B, Up From Prior Range Of $1.51B-$1.53B vs $1.53B Est.",
         "sees", 1.55e9, 1.55e9, 1.53e9),
        # No figure in a parenthetical stating the guidance replaced.
        ("Sees Adj. EPS Of $4.15-$4.20 (Prior Adj. EPS $4.25-$4.35) Vs. $4.30 Consensus",
         "sees", 4.15, 4.20, 4.30),
        ("Trane Technologies Raises FY23 Outlook, Sees Adj. EPS To $9.00 (From $8.80 To $8.90) "
         "Vs $8.87 Est.", "raises", 9.00, 9.00, 8.87),
        # Constructed: before any figure, the parenthetical states the old and the new.
        ("Acme Updates FY EPS Guidance (From $1.00 To $0.90) Vs $1.05 Est", "updates",
         0.90, 0.90, 1.05),
        ("Acme Raises FY EPS Guidance (From $1.00 To $1.10) Vs $1.05 Est", "raises",
         1.10, 1.10, 1.05),
        ("Acme Updates FY EPS Guidance (From $1.00 To ~$0.90) Vs $1.05 Est", "updates",
         0.90, 0.90, 1.05),
        # ... the "To" after the old figure, "Down" or "Up" between them.
        ("Acme Lowers FY EPS Guidance (From $1.10 Down To $0.95) Vs $1.05 Est", "lowers",
         0.95, 0.95, 1.05),
        ("Acme Updates FY EPS Guidance (From $1.00-$1.10 To $0.90-$1.00) Vs $1.05 Est",
         "updates", 0.90, 1.00, 1.05),
        ("Acme Updates FY EPS Guidance (From $1.20 Up To $1.30) Vs $1.25 Est", "updates",
         1.30, 1.30, 1.25),
        # Guidance kept as it was: "from" introduces its range when no "to" follows.
        ("PPL Reaffirms FY2017 EPS Guidance from $1.92-2.12 vs $2.16 Est", "reaffirms",
         1.92, 2.12, 2.16),
        ("Knight-Swift Maintains Q4 EPS Guidance from $0.71-0.75 vs $0.73 Est", "maintains",
         0.71, 0.75, 0.73),
        # ... but "Raised From $303M To $306M" is still the old figure and the new,
        ("Sees FY17 Adj. EBITDA $90M, Sales Raised From $303M To $306M vs $304.7M Est.",
         "sees", 306e6, 306e6, 304.7e6),
        # ... and a "from" after the guided figure introduces the range replaced.
        ("Atkore Sees FY Adj. EPS $1.37-$1.45 from $1.55-$1.65 vs $1.57 Est.", "sees",
         1.37, 1.45, 1.57),
        # A range's low without its "$" after "To".
        ("Updates FY24 Financial Guidance EPS From $3.28-$3.35 To 3.22-$3.31 Vs $3.33 Est.",
         "updates", 3.22, 3.31, 3.33),
    ],
)  # fmt: skip
def test_the_guided_figure_is_the_new_guidance_as_stated(
    headline: str, action: str, low: float, high: float, estimate: float
) -> None:
    g = parse_headline(headline)[0]
    f = g.fields
    assert g.kind == "guidance" and f["action"] == action
    assert f["low"] == pytest.approx(low) and f["high"] == pytest.approx(high)
    assert f["estimate"] == pytest.approx(estimate) and "not_comparable" not in f
    assert f["surprise"] == pytest.approx(((low + high) / 2 - estimate) / abs(estimate))


@pytest.mark.parametrize(
    ("before", "new"),
    [
        ("(From $1.00 To ", True),
        ("(From $1.00 To ~", True),
        ("(From $1.20B To ", True),
        ("(From $(0.10) To ", True),
        ("(From $(1.2)B To ", True),
        ("(From $1.10 Down To ", True),
        ("(Prior $1.00 Up To ", True),
        ("(Prior Up To ", False),
        ("(Previously Raised To ", False),
        ("(Previously Lowered To ~", False),
        ("(Prior Outlook Had Called For ", False),
    ],
)
def test_only_a_to_after_the_old_figure_introduces_the_new_guidance(before: str, new: bool) -> None:
    from halal_trader.events.earnings_parse import _TO_END

    assert (_TO_END.search(before) is not None) is new


def test_a_glued_parenthesis_never_flips_a_figure_s_sign() -> None:
    # Constructed: "x($0.57)" is glued to the "x", and "$0.57)" inside it is no figure.
    (g,) = parse_headline("Acme Sees Q1 EPS Of x($0.57) Vs. $(0.30) Est.")
    assert g.fields["low"] is None and g.fields["surprise"] is None


@pytest.mark.parametrize(
    ("headline", "action"),
    [
        # Only the guidance being replaced is stated: the action stands alone.
        ("10x Genomics Lowers FY24 Sales Guidance From $640M-$660M Vs. $663.06M Estimate",
         "lowers"),
        ("Ulta Beauty Sees Q3 EPS At High End Of Previously-Issued Range $2.11-$2.16 vs $2.16 "
         "Estimate", "sees"),
        ("UPDATE: Stryker Sees FY16 Adj. EPS at High End of Previously Stated $5.75-$5.80 vs "
         "$5.78 Est.", "sees"),
        # "Raises ... From" with no new figure: the From range is the old one.
        ("FMC Raises FY19 Adj. EPS Guidance From $5.68-$5.88 vs $5.76 Estimate", "raises"),
        # The "26" of "FY26" is all GUIDE_V4 found.
        ("Autoliv Expects 0% Organic Sales Growth For FY26 Vs $11.18B Estimate. FY25 Sales Was "
         "$10.82B", "expects"),
        # Constructed: a "from" not after "Guidance" states the base of a change,
        # never kept guidance's range (a false raise, a false cut).
        ("Acme Sees FY Sales Down 5% From $1.2B vs $1.1B Est", "sees"),
        ("Acme Sees FY Revenue Up From $1.2B vs $1.3B Est", "sees"),
        ("Acme Reaffirms FY Revenue Growth Of 10% From $1.00B vs $1.15B Est", "reaffirms"),
        # Constructed: before any figure, a figure in a parenthetical stating the
        # guidance replaced is the old one unless a "To" introduces it (a false
        # +23.8% raise, whatever words open it).
        ("Acme Updates FY EPS Guidance (Prior Outlook Had Called For $1.30) Vs $1.05 Est",
         "updates"),
        ("Acme Lowers FY EPS Guidance (From Its Earlier $1.10) Vs $1.05 Est", "lowers"),
        ("Acme Lowers FY EPS Guidance (Previously Expected EPS In The Range $1.10-$1.20) Vs "
         "$1.05 Est", "lowers"),
        # Constructed: a "To" after a word, not after the old figure, introduces
        # the old figure (a false +23.8% raise, on a cut too).
        ("Acme Updates FY EPS Guidance (Prior Up To $1.30) Vs $1.05 Est", "updates"),
        ("Acme Updates FY EPS Guidance (Previously Lowered To $1.30) Vs $1.05 Est", "updates"),
        ("Acme Lowers FY EPS Guidance (Previously Raised To $1.30) Vs $1.05 Est", "lowers"),
        ("Acme Updates FY EPS Guidance (Prior Up To ~$1.30) Vs $1.05 Est", "updates"),
    ],
)  # fmt: skip
def test_guidance_without_a_reliable_figure_keeps_its_action(headline: str, action: str) -> None:
    (g,) = parse_headline(headline)
    assert g.kind == "guidance" and g.fields["action"] == action
    assert g.fields["low"] is None and g.fields["mid"] is None
    assert g.fields["surprise"] is None and g.fields["estimate"] is not None


@pytest.mark.parametrize(
    ("headline", "low", "high"),
    [
        ("Plug Power Sees Q2 Sales $37-$41M vs $34.7M Est.", 37e6, 41e6),
        ("BioMarin Sees FY18 Sales $1.47-$1.53B vs $1.48B Est.", 1.47e9, 1.53e9),
    ],
)
def test_a_range_states_its_unit_once(headline: str, low: float, high: float) -> None:
    g = parse_headline(headline)[0]
    assert g.fields["low"] == pytest.approx(low) and g.fields["high"] == pytest.approx(high)
    assert g.fields["surprise"] is not None and "not_comparable" not in g.fields


@pytest.mark.parametrize(
    ("headline", "kind", "metric"),
    [
        # A suffix on one side only.
        ("Agilent Technologies Sees FY25 Adj. EPS $5.54-$5.61 Vs $5.66B Est.", "guidance", None),
        ("Merit Medical Reports Q2 EPS $0.26 vs. Est. $151.1M vs. Est. $147.76M", "result", "eps"),
        # The year read as the EPS: 2024 against $149.45M.
        ("Appian Preliminary Revenue Of $149.8M For Q1 2024 Vs $149.45M Est.; Cloud Subscription "
         "Revenue Expected To Be $86.6M", "result", "eps"),
        # Both suffixed, a thousand times apart.
        ("Takeda Pharmaceutical Earlier Reported Q1 Sales $8.60B Beat $7.49M Estimate",
         "result", "sales"),
        # The word is computed from the same broken figures ("Miss" whenever the actual
        # drops its unit), so it decides nothing either.
        ("Air Products & Chemicals Q2 2024 Adj EPS $2.85 Beats $2.69 Estimate, Sales $2.930 Miss "
         "$3.047B Estimate", "result", "sales"),
    ],
)  # fmt: skip
def test_a_figure_in_other_units_than_its_estimate_has_no_surprise(
    headline: str, kind: str, metric: str | None
) -> None:
    (fact,) = parse_headline(headline)
    f = fact.fields
    assert fact.kind == kind and f["not_comparable"] is True
    if metric is None:
        assert f["surprise"] is None
    else:
        assert f[f"{metric}_surprise"] is None and f[f"{metric}_not_comparable"] is True


def test_units_across_a_suffix_boundary_still_compare() -> None:
    (r,) = parse_headline(
        "Abercrombie & Fitch Q1 Adj EPS $2.14 Beats $1.73 Estimate, "
        "Sales $1.02B Beat $963.26M Estimate"
    )
    assert r.fields["sales_surprise"] == pytest.approx((1.02e9 - 963.26e6) / 963.26e6)
    assert "not_comparable" not in r.fields
    # Only the statement in other units loses its surprise.
    (r,) = parse_headline(
        "Air Products & Chemicals Q2 2024 Adj EPS $2.85 Beats $2.69 Estimate, "
        "Sales $2.930 Miss $3.047B Estimate"
    )
    assert r.fields["eps_surprise"] == pytest.approx((2.85 - 2.69) / 2.69)
    assert "eps_not_comparable" not in r.fields


@pytest.mark.parametrize(
    ("headline", "eps", "estimate"),
    [
        ("Ralph Lauren Q4 Adj. EPS $(0.68) Misses $0.01 Estimate, Sales $1.30B Beat $1.29B "
         "Estimate", -0.68, 0.01),
        ("Phillips 66 Q3 Adj. EPS $(0.01) Beats $(0.79) Estimate", -0.01, -0.79),
        ("Coinbase Glb Q4 EPS $1.04 Beats $(0.01) Estimate, Sales $953.79M Beat $822.36M "
         "Estimate", 1.04, -0.01),
        # A large EPS under the cap still compares.
        ("MicroStrategy Q2 Adj. EPS $32.52 Beats $(0.07) Estimate", 32.52, -0.07),
    ],
)  # fmt: skip
def test_an_eps_near_zero_compares_whatever_the_ratio(
    headline: str, eps: float, estimate: float
) -> None:
    # Over 50 times apart, but the smaller side is under $0.10: no dropped suffix.
    (r,) = parse_headline(headline)
    f = r.fields
    assert f["eps"] == pytest.approx(eps) and f["eps_estimate"] == pytest.approx(estimate)
    assert f["eps_surprise"] == pytest.approx((eps - estimate) / abs(estimate))
    assert "eps_not_comparable" not in f and "not_comparable" not in f


@pytest.mark.parametrize(
    "headline",
    [
        # Garbled figures a dollar or more apart from a real estimate still do not compare.
        "Akamai Reports Q3 EPS $68 vs $0.61 Est., Sales $584M vs $571.9M Est.",
        "Bidu Q1 EPS $1,89 Beats $1.66 Est; Revenue $4.29B Beats $4.22B Est",
        "ILG Reports Q3 EPS 40.25 vs $0.26 Est, Rev $418M vs $387M Est",
        "Bloom Energy Q4 EPS $(12) Misses $(0.18) Estimate, Sales $213.6M Beat $206.95M Estimate",
        # Constructed: the year read as the EPS, against an estimate near zero.
        "Acme Q1 2024 Vs $(0.03) Est.",
        "Acme Reports Q1 EPS $2,024 vs $0.05 Est.",
    ],
)
def test_a_garbled_eps_still_does_not_compare(headline: str) -> None:
    r = parse_headline(headline)[0]
    assert r.fields["eps_not_comparable"] is True and r.fields["eps_surprise"] is None


@pytest.mark.parametrize(
    "headline",
    [
        "Celgene Sees Q4 Adj EPS $1.18 Vs Est $1.30, Sees FY 2016 Adj EPS $5.50-$5.70 VS Est "
        "$5.68 & Sales $10.5-$11B Vs Est $11.13B",
        "Humana Sees FY16 EPS $8.85 vs. Est. $8.73",
        "Globus Medical Sees FY17 EPS $1.27, Inline",
    ],
)
def test_a_forecast_is_never_read_as_a_result(headline: str) -> None:
    assert parse_headline(headline) == []


def test_a_preannouncement_is_guidance() -> None:
    (g,) = parse_headline("Ashland Sees Prelim. Q1 Adj. EPS $0.97 vs $1.10 Est.")
    assert g.kind == "guidance" and g.fields["action"] == "sees"
    assert g.fields["surprise"] == pytest.approx((0.97 - 1.10) / 1.10)


def test_a_sales_figure_after_a_forecast_verb_is_not_the_results() -> None:
    # Constructed: the forecast verb is too far from its metric for GUIDE_V4.
    (r,) = parse_headline(
        "Acme Q3 EPS $1.00 Beats $0.90 Estimate, Also Sees The Fourth Quarter Benefiting From "
        "Strong Demand, Sales $5B vs $5.2B Est"
    )
    assert r.kind == "result" and "sales" not in r.fields
    (r,) = parse_headline(
        "Acme Q3 EPS $1.00 Beats $0.90 Estimate, Says The Fourth Quarter Will Be Seasonal As "
        "Usual, Sales $5B vs $5.2B Est"
    )
    assert r.fields["sales_estimate"] == pytest.approx(5.2e9)


@pytest.mark.parametrize(
    ("headline", "eps", "sales"),
    [
        # "May Not Compare" without "To", closing the segment: every statement.
        ("CMS Energy Reports Q4 GAAP EPS $(0.01) vs $0.51 Est., Sales $1.78B vs $1.77B Est., "
         "May Not Compare", True, True),
        ("Sanchez Energy Q2 EPS $0.31 vs $(0.14) Est., Sales $175.70M vs $182.24M Est., "
         "May Not Compare", True, True),
        # Within the EPS statement's clause (before the next metric): the EPS only.
        ("Tableau Reports Q3 non-GAAP EPS $0.16 vs $0.07 Est, May Not Compare, Revenue $206.1M "
         "vs $213.78M Est", True, False),
        ("AerCap Holdings Q1 EPS $1.72 Beats $1.57 Estimate, May Not Compare, Sales $1.22B Miss "
         "$1.23B Estimate", True, False),
        # The flag's other forms.
        ("Westlake Chemical Q3 EPS $0.51 vs $0.89 Est, Revenue $1.28B vs $1.52B Est, Does Not "
         "Compare", True, True),
        ("Northern Oil and Gas Q1 EPS $0.27 vs $(0.02) Est, Sales $48.85M vs $47.75M Est, Does "
         "Not Compare", True, True),
        ("Unity Software Q2 Adj. EPS $0.18 Beats $(0.27) Estimate (may not be comparable), Sales "
         "$440.944M Beat $425.459M Estimate.", True, False),
        ("Brookfield Asset Mgmt Q4 EPS $0.34 Misses $0.60 Estimate, May Not Be Comparable. Sales "
         "$1.394B Miss $1.609B Estimate.", True, False),
        # A segment that only flags covers every statement in the headline.
        ("Assurant Reports Q4 EPS $0.97 vs. Est. $1.51, Rev. $2.547B vs. Est. $2B; Estimates May "
         "Not Compare", True, True),
        ("Sanchez Energy Reports Q1 EPS $(1.20) vs. Est. $(0.12), Rev. $79.816M vs. Est. $127.2M; "
         "Estimates Do Not Compare", True, True),
        # Constructed: a segment naming a metric flags only its own statements.
        ("Acme Q1 EPS $1.00 Beats $0.90 Estimate, Sales $5.0B Beat $4.9B Estimate; Sales May Not "
         "Compare", False, False),
        # Constructed: a segment flagging the headline names the estimates, consensus,
        # a forecast, guidance or the outlook, or is a BZ NOTE ...
        ("Acme Q1 EPS $1.00 Beats $0.90 Estimate, Sales $5.0B Beat $4.9B Estimate; Consensus "
         "May Not Compare", True, True),
        ("Acme Q1 EPS $1.00 Beats $0.90 Estimate, Sales $5.0B Beat $4.9B Estimate; Outlook "
         "May Not Compare", True, True),
        ("Acme Q1 EPS $1.00 Beats $0.90 Estimate, Sales $5.0B Beat $4.9B Estimate; BZ NOTE: "
         "Figures Not Comparable", True, True),
        # ... a note about last year leaves the beats theirs.
        ("Acme Q1 EPS $1.00 Beats $0.90 Estimate, Sales $5.0B Beat $4.9B Estimate; Results Not "
         "Comparable To Prior Year", False, False),
        ("Acme Q1 EPS $1.00 Beats $0.90 Estimate, Sales $5.0B Beat $4.9B Estimate; Prior Year "
         "Not Comparable Due To Spin-Off", False, False),
    ],
)  # fmt: skip
def test_may_not_compare_with_or_without_to(headline: str, eps: bool, sales: bool) -> None:
    (r,) = parse_headline(headline)
    f = r.fields
    assert f.get("not_comparable", False) is (eps or sales)
    for metric, flagged in (("eps", eps), ("sales", sales)):
        assert f.get(f"{metric}_not_comparable", False) is flagged
        assert (f[f"{metric}_surprise"] is None) is flagged


def test_a_guide_closed_by_may_not_compare_has_no_surprise() -> None:
    (g,) = parse_headline(
        "Becton Dickinson Sees FY EPS $6.23 to $6.30 vs $8.41 est, May Not Compare"
    )
    assert g.fields["not_comparable"] is True and g.fields["surprise"] is None
    assert g.fields["mid"] == pytest.approx(6.265)


@pytest.mark.parametrize(
    "flag",
    [
        "May Not Compare",
        "May Not Compare To",
        "Does Not Compare",
        "Do Not Compare To",
        "May Not Be Comparable To",
        "(may not be comparable)",
        "May Not Be Compared To",
        "Not Comparable To",
        "Which May Not Compare To",
    ],
)
def test_every_form_of_the_not_comparable_flag(flag: str) -> None:
    from halal_trader.events.earnings_parse import NOT_COMPARABLE

    assert NOT_COMPARABLE.search(f"Acme Q1 EPS $0.34 {flag} $0.60 Estimate") is not None
    (r,) = parse_headline(f"Acme Q1 EPS $0.34 Misses $0.60 Estimate, {flag} Estimates")
    assert r.fields["eps_not_comparable"] is True and r.fields["eps_surprise"] is None


def test_a_flag_closing_the_segment_or_alone_in_one_covers_the_statement() -> None:
    # Carlyle 2017-02-08: a false miss (NSN_CORE) while only "May Not Compare" was read.
    (r,) = parse_headline("Carlyle Group Q4 EPS $(0.16) vs $0.41 Est, Does Not Compare")
    assert r.fields["eps_not_comparable"] is True and r.fields["eps_surprise"] is None
    # Rambus 2018-01-29: the note in its own segment, a false structural cut before.
    (g,) = parse_headline(
        "Rambus Sees Q1 Adj. EPS $(0.19)-$(0.12) vs $0.18 Est., Sales $41M-$47M vs $100.7M Est.; "
        "BZ NOTE: Forecast Likely Does Not Compare As Co.'s Results Account For ASC 606"
    )
    assert g.kind == "guidance" and g.fields["not_comparable"] is True
    assert g.fields["surprise"] is None and g.fields["mid"] == pytest.approx(-0.155)
    # Constructed: "Outlook May Not Compare" in its own segment, a false +11.1% raise
    # while _FLAG_SUBJECT left the outlook out.
    (g,) = parse_headline("Acme Sees FY EPS $1.00 Vs $0.90 Est; Outlook May Not Compare")
    assert g.kind == "guidance" and g.fields["not_comparable"] is True
    assert g.fields["surprise"] is None and g.fields["mid"] == pytest.approx(1.00)
    # Constructed: a segment stating a figure is no headline-wide flag.
    r, *_ = parse_headline(
        "Acme Q1 EPS $1.00 Beats $0.90 Estimate; Sees Q2 EPS $1.20 May Not Compare To $1.30 Est"
    )
    assert r.fields["eps_surprise"] == pytest.approx(0.1 / 0.9)
    assert "not_comparable" not in r.fields


@pytest.mark.parametrize(
    "headline",
    [
        "Ulta Sees Q1 Rev. $1.016B-$1.033B vs. Est. $1.01B, EPS $1.25-$1.30 vs. Est. $1.22",
        "Synopsys Expects Q1 Revenue Of $1.63B-$1.66B (Estimate $1.6B), "
        "Non-GAAP EPS Of $3.40-$3.45 (Estimate $3.05)",
        "Sees FY18 Adj. EPS $(0.07)-$(0.04) May Not Compare To $0.40 Est.",
        "Teledyne Technologies Sees Q1 EPS $1.15-$1.17 vs. Est. $1.21, "
        "FY17 EPS $5.40-$5.50 vs. Est. $5.42",
    ],
)
def test_guidance_against_consensus_no_template_reads_is_unparsed(headline: str) -> None:
    from halal_trader.events.earnings_parse import GUIDE_UNPARSED

    assert parse_headline(headline) == []
    assert GUIDE_UNPARSED.search(headline) is not None


def test_guidance_unparsed_needs_a_consensus() -> None:
    from halal_trader.events.earnings_parse import GUIDE_UNPARSED

    assert GUIDE_UNPARSED.search("Conduent Sees FY2028 Sales $2.250B-$2.350B") is None
    assert GUIDE_UNPARSED.search("Northrop Grumman Raises FY19 Guidance") is None


def test_an_inline_eps_keeps_the_sales_stated_after_it() -> None:
    # v3 dropped the sales miss after an inline EPS; v4 keeps it.
    (r,) = parse_headline(
        "Williams Companies Q1 EPS $0.22, Inline, Sales $2.054B Miss $2.28B Estimate"
    )
    assert r.fields["eps_verdict"] == "inline" and r.fields["eps_surprise"] == 0.0
    assert r.fields["sales_verdict"] == "miss"
    assert r.fields["sales_surprise"] == pytest.approx((2.054 - 2.28) / 2.28)


def test_not_comparable_belongs_to_the_statement_that_says_it() -> None:
    # The sales "May Not Compare To" is not read; the EPS beat keeps its surprise.
    (r,) = parse_headline(
        "CRISPR Therapeutics Q2 2024 Adj EPS $(1.49) Beats $(1.52) Estimate, "
        "Sales $517.00K May Not Compare To $3.54M Estimate"
    )
    assert r.fields["eps_surprise"] == pytest.approx((-1.49 + 1.52) / 1.52)
    assert "not_comparable" not in r.fields and "sales" not in r.fields
    # Guidance below consensus stays so though the sales clause does not compare.
    (g,) = parse_headline(
        "Leidos Sees FY21 Adj. EPS $6.15-$6.45 vs $6.47 Est., "
        "Sales $13.7B-$14.1B May Not Compare To $24.39B Est."
    )
    assert g.fields["surprise"] == pytest.approx((6.30 - 6.47) / 6.47)
    assert "not_comparable" not in g.fields


def test_a_read_spanning_may_not_compare_carries_no_surprise() -> None:
    # Constructed: GUIDE_V4's gap steps over the flag, so the figure it reads is
    # one Benzinga says does not compare.
    (g,) = parse_headline(
        "Acme Sees FY Sales May Not Compare To Prior Year, $1.00B-$1.10B vs $1.20B Est"
    )
    assert g.fields["not_comparable"] is True and g.fields["surprise"] is None
    assert g.fields["mid"] == pytest.approx(1.05e9) and g.fields["estimate"] == pytest.approx(1.2e9)


def test_parse_order_guidance_before_results() -> None:
    from halal_trader.events.earnings_parse import PARSE_ORDER

    assert PARSE_ORDER == (
        "_GUIDE_RANGE",
        "GUIDE_V4",
        "_RESULT",
        "RESULT_EST_FIRST",
        "_INLINE",
        "SALES_ONLY",
    )
    # One segment, one fact: the result read by _RESULT, not a SALES_ONLY second fact.
    (r,) = parse_headline(
        "Trade Desk Q1 EPS $0.45 Misses $0.77 Estimate, Sales $219.80M Beat $216.90M Estimate"
    )
    assert r.fields["eps_verdict"] == "misses" and r.fields["sales_verdict"] == "beat"


def test_a_correction_parses_like_the_wire_it_corrects() -> None:
    (r,) = parse_headline(
        "CORRECTION: Trade Desk Q1 EPS $1.41 Beats $0.77 Estimate, "
        "Sales $219.81M Beat $216.90M Estimate"
    )
    assert r.fields["eps"] == 1.41 and r.fields["eps_verdict"] == "beats"
    assert parse_headline("CORRECTION: Progressive Q4 2023 GAAP EPS $3.37, Dec. EPS $1.53") == []


def test_parser_sha_pins_every_pattern() -> None:
    import hashlib
    import json

    from halal_trader.events import earnings_parse as ep

    src = ep.sources()
    for name in (
        "_RESULT",
        "_INLINE",
        "_SALES",
        "_GUIDE_RANGE",
        "SALES_ONLY",
        "RESULT_EST_FIRST",
        "SALES_EST_FIRST",
        "GUIDE_V4",
        "NOT_COMPARABLE",
        "GUIDE_UNPARSED",
        "_BASIS_WORD",
        "_TOLERANCE_GAP",
        "_FIGURE",
        "_OLD",
        "_FROM",
        "_GUIDANCE_FROM",
        "_FORWARD",
        "_SUFFIXED",
        "_METRIC_WORD",
        "_CLOSING_FLAG",
        "_FLAG_SUBJECT",
        "_OLD_PAREN",
        "_TO",
        "_TO_END",
    ):
        assert src[name] == getattr(ep, name).pattern
    # The family label, not EXTRACTOR, which derives from the pin.
    assert src["EXTRACTOR_V4"] == ep.EXTRACTOR_V4 and "EXTRACTOR" not in src
    assert src["PARSE_ORDER"] == ",".join(ep.PARSE_ORDER)
    # The constants that shape a read, beside the patterns.
    assert src["UNIT_RATIO"] == repr(ep.UNIT_RATIO) == "50.0"
    assert src["EPS_RATIO_FLOOR"] == repr(ep.EPS_RATIO_FLOOR) == "0.1"
    assert src["EPS_RATIO_CAP"] == repr(ep.EPS_RATIO_CAP) == "100.0"
    assert src["OLD_WINDOW"] == repr(ep.OLD_WINDOW) == "40"
    assert src["KEEP_ACTIONS"] == "affirms,maintains,reaffirms,reiterates,sees"
    assert json.loads(src["SCALE"]) == ep.SCALE == {"K": 1e3, "M": 1e6, "B": 1e9}
    with pytest.raises(TypeError):  # pinned at import: no write can change it after
        ep.SCALE["T"] = 1e12  # type: ignore[index]
    assert src["SURPRISE_FLOOR"] == repr(ep.SURPRISE_FLOOR) == "0.01"
    assert src["SEGMENT_SEP"] == ep.SEGMENT_SEP == ";"
    want = hashlib.sha256(json.dumps(src, sort_keys=True).encode()).hexdigest()[:12]
    assert ep.PARSER_SHA == want and len(want) == 12


async def test_v4_extraction_reads_events_v3_already_read(engine) -> None:
    from datetime import UTC, datetime

    from sqlalchemy import text

    from halal_trader.events.earnings_parse import EXTRACTOR, EXTRACTOR_V3, extract_all
    from halal_trader.events.store import EventRecord, EventRecorder

    t = datetime(2021, 1, 7, 21, tzinfo=UTC)
    await EventRecorder(engine).record(
        [
            EventRecord(
                "alpaca",
                "1",
                "news",
                "MU",
                t,
                t,
                {"headline": "Micron Technology Q1 Sales $5.77B Beat $5.73B Estimate"},
            )
        ]
    )
    async with engine.begin() as conn:  # what v3 stored for it: nothing read
        await conn.execute(
            text(
                "INSERT INTO event_facts (event_id, extractor, kind, fields) "
                "SELECT id, :x, 'none', '{}'::jsonb FROM events"
            ),
            {"x": EXTRACTOR_V3},
        )
    assert await extract_all(engine) == 1
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text("SELECT extractor, kind, fields FROM event_facts ORDER BY extractor")
            )
        ).all()
    assert [(r.extractor, r.kind) for r in rows] == [(EXTRACTOR_V3, "none"), (EXTRACTOR, "result")]
    assert rows[1].fields["sales_verdict"] == "beat"


async def test_a_parser_change_re_extracts_and_only_drop_superseded_deletes(
    engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime

    from sqlalchemy import text

    from halal_trader.events import earnings_parse as ep
    from halal_trader.events.store import EventRecord, EventRecorder

    t = datetime(2021, 1, 7, 21, tzinfo=UTC)
    heads = (
        "Micron Technology Q1 Sales $5.77B Beat $5.73B Estimate",
        "Micron Technology Announces New Memory",
        "Micron Technology Sees Q2 Sales $5.7B-$6.1B vs $5.8B Est",
    )
    await EventRecorder(engine).record(
        [
            EventRecord("alpaca", str(i), "news", "MU", t, t, {"headline": h})
            for i, h in enumerate(heads)
        ]
    )
    async with engine.begin() as conn:  # what v3 and a pin-less v4 parser stored
        for label in (ep.EXTRACTOR_V3, ep.EXTRACTOR_V4):
            await conn.execute(
                text(
                    "INSERT INTO event_facts (event_id, extractor, kind, fields) "
                    "SELECT id, :x, 'none', '{}'::jsonb FROM events"
                ),
                {"x": label},
            )

    async def labels() -> dict[str, int]:
        async with engine.connect() as conn:
            rows = await conn.execute(
                text("SELECT extractor, count(*) AS n FROM event_facts GROUP BY 1")
            )
            return {r.extractor: r.n for r in rows}

    # One event a batch: the pages and the deletes run several rounds.
    assert await ep.extract_all(engine, batch=1) == 2
    current = ep.EXTRACTOR
    # Extraction only adds: the pin-less label stays until drop_superseded.
    assert await labels() == {ep.EXTRACTOR_V3: 3, ep.EXTRACTOR_V4: 3, current: 3}
    assert await ep.drop_superseded(engine, batch=1) == 3
    assert await labels() == {ep.EXTRACTOR_V3: 3, current: 3}
    assert await ep.extract_all(engine, batch=1) == 0  # every event is read

    # Another parser (a dev checkout, a rollback) is a new pin, so a new label:
    # it reads every event again beside the deployed label, whose rows stay.
    changed = f"{ep.EXTRACTOR_V4}+{'0' * 12}"
    monkeypatch.setattr(ep, "EXTRACTOR", changed)
    assert await ep.extract_all(engine, batch=1) == 2
    assert await labels() == {ep.EXTRACTOR_V3: 3, current: 3, changed: 3}
    # Only drop_superseded deletes them; v3's stay.
    assert await ep.drop_superseded(engine, batch=1) == 3
    assert await labels() == {ep.EXTRACTOR_V3: 3, changed: 3}
    assert await ep.drop_superseded(engine) == 0
