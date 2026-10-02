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
