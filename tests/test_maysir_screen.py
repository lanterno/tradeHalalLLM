"""Tests for the maysir (gambling) screen."""

from __future__ import annotations

import pytest

from halal_trader.halal.maysir_screen import (
    MaysirAssessment,
    MaysirInputs,
    MaysirPolicy,
    MaysirRisk,
    MaysirSignal,
    filter_blocked,
    is_tradable,
    render_assessment,
    screen_batch,
    screen_for_maysir,
)


def _healthy_inputs(**overrides) -> MaysirInputs:
    """Construct a deliberately-healthy ticker (passes everything)."""
    base = {
        "ticker": "AAPL",
        "stock_price": 200.0,
        "short_interest_pct": 1.0,
        "retail_flow_pct": 20.0,
        "analyst_coverage_count": 30,
        "realized_volatility_annualized": 0.25,
        "price_change_30d_pct": 5.0,
        "revenue_ttm_usd": 400_000_000_000.0,
    }
    base.update(overrides)
    return MaysirInputs(**base)


# --- Enum string-value pins ---------------------------------------------------


def test_risk_string_values():
    assert MaysirRisk.NONE.value == "none"
    assert MaysirRisk.LOW.value == "low"
    assert MaysirRisk.MODERATE.value == "moderate"
    assert MaysirRisk.HIGH.value == "high"
    assert MaysirRisk.EXTREME.value == "extreme"


def test_signal_string_values():
    assert MaysirSignal.PENNY_PRICE.value == "penny_price"
    assert MaysirSignal.EXTREME_SHORT_INTEREST.value == "extreme_short_interest"
    assert MaysirSignal.RETAIL_FLOW_DOMINANT.value == "retail_flow_dominant"
    assert MaysirSignal.NO_ANALYST_COVERAGE.value == "no_analyst_coverage"
    assert MaysirSignal.EXTREME_VOLATILITY.value == "extreme_volatility"
    assert MaysirSignal.MEME_PUMP_PATTERN.value == "meme_pump_pattern"
    assert MaysirSignal.ZERO_REVENUE.value == "zero_revenue"


# --- Policy validation --------------------------------------------------------


def test_default_policy_pins():
    p = MaysirPolicy()
    assert p.penny_price_threshold == 5.0
    assert p.extreme_short_interest_pct == 50.0
    assert p.retail_flow_dominant_pct == 70.0
    assert p.no_analyst_coverage_count == 0
    assert p.extreme_volatility_annualized == 1.0
    assert p.meme_pump_30d_pct == 100.0
    assert p.moderate_score_threshold == 2
    assert p.high_score_threshold == 4
    assert p.extreme_score_threshold == 6


def test_zero_penny_threshold_rejected():
    with pytest.raises(ValueError, match="penny_price_threshold"):
        MaysirPolicy(penny_price_threshold=0)


def test_short_interest_pct_outside_range_rejected():
    with pytest.raises(ValueError, match="extreme_short_interest_pct"):
        MaysirPolicy(extreme_short_interest_pct=0)
    with pytest.raises(ValueError, match="extreme_short_interest_pct"):
        MaysirPolicy(extreme_short_interest_pct=101)


def test_retail_flow_pct_outside_range_rejected():
    with pytest.raises(ValueError, match="retail_flow_dominant_pct"):
        MaysirPolicy(retail_flow_dominant_pct=0)
    with pytest.raises(ValueError, match="retail_flow_dominant_pct"):
        MaysirPolicy(retail_flow_dominant_pct=200)


def test_negative_analyst_count_rejected():
    with pytest.raises(ValueError, match="no_analyst_coverage_count"):
        MaysirPolicy(no_analyst_coverage_count=-1)


def test_zero_volatility_threshold_rejected():
    with pytest.raises(ValueError, match="extreme_volatility_annualized"):
        MaysirPolicy(extreme_volatility_annualized=0)


def test_score_threshold_ordering_pin():
    """Pin: moderate < high < extreme score thresholds."""
    with pytest.raises(ValueError, match="moderate < high < extreme"):
        MaysirPolicy(
            moderate_score_threshold=5,
            high_score_threshold=4,
            extreme_score_threshold=6,
        )
    with pytest.raises(ValueError, match="moderate < high < extreme"):
        MaysirPolicy(
            moderate_score_threshold=2,
            high_score_threshold=6,
            extreme_score_threshold=4,
        )


def test_zero_moderate_threshold_rejected():
    with pytest.raises(ValueError, match="moderate_score_threshold"):
        MaysirPolicy(moderate_score_threshold=0, high_score_threshold=4, extreme_score_threshold=6)


def test_policy_immutable():
    p = MaysirPolicy()
    with pytest.raises(Exception):
        p.penny_price_threshold = 99  # type: ignore[misc]


# --- MaysirInputs validation --------------------------------------------------


def test_empty_ticker_rejected():
    with pytest.raises(ValueError, match="ticker"):
        _healthy_inputs(ticker="")


def test_zero_price_rejected():
    with pytest.raises(ValueError, match="stock_price"):
        _healthy_inputs(stock_price=0)


def test_negative_price_rejected():
    with pytest.raises(ValueError, match="stock_price"):
        _healthy_inputs(stock_price=-1)


def test_short_interest_above_1000_rejected():
    """Sanity cap at 1000% even though synthetic shorts can exceed 100%."""
    with pytest.raises(ValueError, match="short_interest_pct"):
        _healthy_inputs(short_interest_pct=1001)


def test_short_interest_above_100_allowed():
    """Pin: synthetic-short can exceed 100% of float — allowed."""
    inputs = _healthy_inputs(short_interest_pct=200)
    assert inputs.short_interest_pct == 200


def test_retail_flow_above_100_rejected():
    with pytest.raises(ValueError, match="retail_flow_pct"):
        _healthy_inputs(retail_flow_pct=101)


def test_negative_revenue_rejected():
    with pytest.raises(ValueError, match="revenue_ttm_usd"):
        _healthy_inputs(revenue_ttm_usd=-1)


def test_inputs_immutable():
    i = _healthy_inputs()
    with pytest.raises(Exception):
        i.stock_price = 99  # type: ignore[misc]


# --- Signal detection: each signal in isolation -------------------------------


def test_healthy_ticker_no_signals():
    a = screen_for_maysir(_healthy_inputs())
    assert a.signals == frozenset()
    assert a.risk is MaysirRisk.NONE
    assert a.score == 0


def test_penny_price_fires_below_threshold():
    a = screen_for_maysir(_healthy_inputs(stock_price=4.99))
    assert MaysirSignal.PENNY_PRICE in a.signals


def test_penny_price_does_not_fire_at_threshold():
    """Pin: penny threshold is strict less-than (price=5.0 doesn't fire)."""
    a = screen_for_maysir(_healthy_inputs(stock_price=5.0))
    assert MaysirSignal.PENNY_PRICE not in a.signals


def test_extreme_short_interest_fires_above_threshold():
    a = screen_for_maysir(_healthy_inputs(short_interest_pct=51))
    assert MaysirSignal.EXTREME_SHORT_INTEREST in a.signals


def test_extreme_short_interest_does_not_fire_at_threshold():
    """Pin: short-interest threshold is strict greater-than."""
    a = screen_for_maysir(_healthy_inputs(short_interest_pct=50))
    assert MaysirSignal.EXTREME_SHORT_INTEREST not in a.signals


def test_retail_flow_dominant_fires_above_threshold():
    a = screen_for_maysir(_healthy_inputs(retail_flow_pct=71))
    assert MaysirSignal.RETAIL_FLOW_DOMINANT in a.signals


def test_no_analyst_coverage_fires_at_zero():
    a = screen_for_maysir(_healthy_inputs(analyst_coverage_count=0))
    assert MaysirSignal.NO_ANALYST_COVERAGE in a.signals


def test_no_analyst_coverage_does_not_fire_at_one():
    a = screen_for_maysir(_healthy_inputs(analyst_coverage_count=1))
    assert MaysirSignal.NO_ANALYST_COVERAGE not in a.signals


def test_extreme_volatility_fires_above_threshold():
    a = screen_for_maysir(_healthy_inputs(realized_volatility_annualized=1.01))
    assert MaysirSignal.EXTREME_VOLATILITY in a.signals


def test_meme_pump_pattern_fires_above_threshold():
    a = screen_for_maysir(_healthy_inputs(price_change_30d_pct=101))
    assert MaysirSignal.MEME_PUMP_PATTERN in a.signals


def test_meme_pump_does_not_fire_on_negative_move():
    """Pin: meme-pump signal is one-sided (only spikes count)."""
    a = screen_for_maysir(_healthy_inputs(price_change_30d_pct=-50))
    assert MaysirSignal.MEME_PUMP_PATTERN not in a.signals


def test_zero_revenue_fires_at_zero():
    a = screen_for_maysir(_healthy_inputs(revenue_ttm_usd=0))
    assert MaysirSignal.ZERO_REVENUE in a.signals


def test_zero_revenue_does_not_fire_with_revenue():
    a = screen_for_maysir(_healthy_inputs(revenue_ttm_usd=1))
    assert MaysirSignal.ZERO_REVENUE not in a.signals


# --- Score → Risk mapping -----------------------------------------------------


def test_low_risk_at_score_one():
    """One signal of weight 1 → score=1 → LOW (below moderate threshold of 2)."""
    a = screen_for_maysir(
        _healthy_inputs(analyst_coverage_count=0)  # weight 1
    )
    assert a.score == 1
    assert a.risk is MaysirRisk.LOW


def test_moderate_risk_at_score_two():
    """Pin: score >= moderate_score_threshold (default 2) → MODERATE."""
    a = screen_for_maysir(
        _healthy_inputs(stock_price=4.99)  # PENNY_PRICE weight 2
    )
    assert a.score == 2
    assert a.risk is MaysirRisk.MODERATE


def test_high_risk_at_score_four():
    """Score >= high_score_threshold (default 4) → HIGH."""
    a = screen_for_maysir(
        _healthy_inputs(
            stock_price=4.99,  # weight 2
            price_change_30d_pct=101,  # weight 2
        )
    )
    assert a.score == 4
    assert a.risk is MaysirRisk.HIGH


def test_extreme_risk_at_score_six():
    """Pin: score >= extreme_score_threshold (default 6) → EXTREME."""
    a = screen_for_maysir(
        _healthy_inputs(
            stock_price=4.99,  # 2
            price_change_30d_pct=101,  # 2
            revenue_ttm_usd=0,  # 3 → total 7
        )
    )
    assert a.score == 7
    assert a.risk is MaysirRisk.EXTREME


def test_extreme_at_exactly_threshold_inclusive():
    """Pin: score == extreme_score_threshold is inclusive."""
    a = screen_for_maysir(
        _healthy_inputs(
            stock_price=4.99,  # 2
            short_interest_pct=51,  # 2
            revenue_ttm_usd=0,  # 3 → total 7
        )
    )
    assert a.risk is MaysirRisk.EXTREME


# --- MaysirAssessment validation ---------------------------------------------


def test_assessment_immutable():
    a = screen_for_maysir(_healthy_inputs())
    with pytest.raises(Exception):
        a.score = 99  # type: ignore[misc]


def test_none_risk_with_signals_rejected():
    """Pin: NONE risk → empty signals required."""
    with pytest.raises(ValueError, match="NONE risk must have empty signals"):
        MaysirAssessment(
            ticker="X",
            signals=frozenset({MaysirSignal.PENNY_PRICE}),
            risk=MaysirRisk.NONE,
            score=0,
        )


def test_non_none_risk_without_signals_rejected():
    """Pin: non-NONE risk → at least one signal required."""
    with pytest.raises(ValueError, match="non-NONE risk requires at least one signal"):
        MaysirAssessment(
            ticker="X",
            signals=frozenset(),
            risk=MaysirRisk.LOW,
            score=1,
        )


def test_negative_score_rejected():
    with pytest.raises(ValueError, match="score"):
        MaysirAssessment(
            ticker="X",
            signals=frozenset({MaysirSignal.NO_ANALYST_COVERAGE}),
            risk=MaysirRisk.LOW,
            score=-1,
        )


def test_assessment_empty_ticker_rejected():
    with pytest.raises(ValueError, match="ticker"):
        MaysirAssessment(ticker="", signals=frozenset(), risk=MaysirRisk.NONE, score=0)


# --- is_tradable + filter_blocked --------------------------------------------


def test_tradable_for_none_low_moderate():
    """Pin: NONE + LOW + MODERATE are tradable."""
    for risk_score_signal in [
        (MaysirRisk.NONE, 0, frozenset()),
        (MaysirRisk.LOW, 1, frozenset({MaysirSignal.NO_ANALYST_COVERAGE})),
        (MaysirRisk.MODERATE, 2, frozenset({MaysirSignal.PENNY_PRICE})),
    ]:
        risk, score, signals = risk_score_signal
        a = MaysirAssessment(ticker="X", signals=signals, risk=risk, score=score)
        assert is_tradable(a) is True


def test_not_tradable_for_high_extreme():
    """Pin: HIGH + EXTREME are non-tradable."""
    for risk in [MaysirRisk.HIGH, MaysirRisk.EXTREME]:
        a = MaysirAssessment(
            ticker="X",
            signals=frozenset({MaysirSignal.PENNY_PRICE}),
            risk=risk,
            score=10,
        )
        assert is_tradable(a) is False


def test_filter_blocked_returns_only_blocked():
    healthy = screen_for_maysir(_healthy_inputs())
    extreme = screen_for_maysir(
        _healthy_inputs(stock_price=4.99, price_change_30d_pct=101, revenue_ttm_usd=0)
    )
    high = screen_for_maysir(_healthy_inputs(stock_price=4.99, price_change_30d_pct=101))
    blocked = filter_blocked([healthy, extreme, high])
    assert len(blocked) == 2
    for a in blocked:
        assert is_tradable(a) is False


# --- screen_batch -------------------------------------------------------------


def test_screen_batch_returns_sorted_by_ticker():
    """Pin: deterministic ticker-ascending order."""
    inputs = [
        _healthy_inputs(ticker="ZZZ"),
        _healthy_inputs(ticker="AAA"),
        _healthy_inputs(ticker="MMM"),
    ]
    results = screen_batch(inputs)
    assert [a.ticker for a in results] == ["AAA", "MMM", "ZZZ"]


def test_screen_batch_empty():
    assert screen_batch([]) == ()


# --- Render -------------------------------------------------------------------


def test_render_healthy_shows_clean():
    a = screen_for_maysir(_healthy_inputs())
    out = render_assessment(a)
    assert "✅" in out
    assert "AAPL" in out
    assert "none" in out


def test_render_extreme_shows_red():
    a = screen_for_maysir(
        _healthy_inputs(stock_price=4.99, price_change_30d_pct=101, revenue_ttm_usd=0)
    )
    out = render_assessment(a)
    assert "🔴" in out
    assert "extreme" in out


def test_render_includes_signal_labels():
    a = screen_for_maysir(_healthy_inputs(stock_price=4.99))
    out = render_assessment(a)
    assert "penny price" in out


def test_render_signals_sorted():
    """Pin: signal labels rendered in sorted order (deterministic)."""
    a = screen_for_maysir(_healthy_inputs(stock_price=4.99, price_change_30d_pct=101))
    out = render_assessment(a)
    # "meme-pump pattern" should come BEFORE "penny price" alphabetically
    assert out.index("meme-pump") < out.index("penny price")


def test_render_no_secret_leak():
    """Pin: render output never includes alt-data feed details."""
    a = screen_for_maysir(_healthy_inputs())
    out = render_assessment(a)
    # The render shouldn't expose URLs, IDs, etc. that don't belong.
    forbidden = [
        "reddit.com",
        "robinhood",
        "Authorization",
        "Bearer",
        "sk_live",
        "/api/",
    ]
    for word in forbidden:
        assert word not in out


# --- E2E flows ----------------------------------------------------------------


def test_e2e_meme_stock_pattern_flagged_extreme():
    """The classic meme-stock profile: penny + 200% short + retail-dominant
    + zero coverage + 200% vol + 500% 30d move + minor revenue → EXTREME."""
    inputs = MaysirInputs(
        ticker="MEME",
        stock_price=2.50,
        short_interest_pct=200,
        retail_flow_pct=85,
        analyst_coverage_count=0,
        realized_volatility_annualized=2.0,
        price_change_30d_pct=500,
        revenue_ttm_usd=1_000_000,
    )
    a = screen_for_maysir(inputs)
    assert a.risk is MaysirRisk.EXTREME
    assert is_tradable(a) is False
    # Every signal except ZERO_REVENUE should fire
    expected = {
        MaysirSignal.PENNY_PRICE,
        MaysirSignal.EXTREME_SHORT_INTEREST,
        MaysirSignal.RETAIL_FLOW_DOMINANT,
        MaysirSignal.NO_ANALYST_COVERAGE,
        MaysirSignal.EXTREME_VOLATILITY,
        MaysirSignal.MEME_PUMP_PATTERN,
    }
    assert a.signals == expected


def test_e2e_blue_chip_passes_clean():
    """Mega-cap with strong fundamentals: NONE risk, tradable."""
    inputs = MaysirInputs(
        ticker="MSFT",
        stock_price=420.0,
        short_interest_pct=0.5,
        retail_flow_pct=15.0,
        analyst_coverage_count=45,
        realized_volatility_annualized=0.22,
        price_change_30d_pct=3.0,
        revenue_ttm_usd=240_000_000_000.0,
    )
    a = screen_for_maysir(inputs)
    assert a.risk is MaysirRisk.NONE
    assert a.signals == frozenset()
    assert is_tradable(a) is True


def test_e2e_pre_revenue_biotech_flagged_high():
    """Pre-revenue biotech (zero revenue + low coverage + high vol) — even
    if no other signals fire, the weight-3 ZERO_REVENUE plus ancillary
    pushes it past HIGH."""
    inputs = MaysirInputs(
        ticker="BIOX",
        stock_price=15.0,
        short_interest_pct=20.0,
        retail_flow_pct=40.0,
        analyst_coverage_count=2,  # some coverage
        realized_volatility_annualized=1.5,  # extreme vol
        price_change_30d_pct=20.0,
        revenue_ttm_usd=0,  # pre-revenue
    )
    a = screen_for_maysir(inputs)
    # ZERO_REVENUE (3) + EXTREME_VOLATILITY (1) = score 4 → HIGH
    assert a.score == 4
    assert a.risk is MaysirRisk.HIGH
    assert is_tradable(a) is False


def test_e2e_replay_consistency():
    """Pin: same inputs → equal assessment."""
    inputs = _healthy_inputs(stock_price=4.99, short_interest_pct=51)
    a1 = screen_for_maysir(inputs)
    a2 = screen_for_maysir(inputs)
    assert a1 == a2


def test_e2e_custom_policy_loosens_to_low():
    """Operator that wants a more permissive screen sets higher cutoffs."""
    permissive = MaysirPolicy(
        moderate_score_threshold=4, high_score_threshold=8, extreme_score_threshold=12
    )
    a = screen_for_maysir(_healthy_inputs(stock_price=4.99), policy=permissive)
    # PENNY_PRICE = score 2; with cutoff 4, falls into LOW
    assert a.score == 2
    assert a.risk is MaysirRisk.LOW
    assert is_tradable(a) is True
