"""Tests for marketplace/etf_basket.py — Round-5 Wave 21.D."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from halal_trader.marketplace.etf_basket import (
    BasketDefinition,
    Constituent,
    RebalanceCadence,
    ScreenAction,
    ScreenIssue,
    any_drift_exceeds,
    assert_basket_halal,
    compute_drift,
    divest_failing,
    next_rebalance,
    render_basket,
    render_drift,
    render_screen,
    screen_basket,
)


def _basket(
    basket_id: str = "B1",
    name: str = "Halal Tech",
    author: str = "alice",
    cadence: RebalanceCadence = RebalanceCadence.MONTHLY,
    constituents: tuple[Constituent, ...] | None = None,
    drift_threshold: float = 0.05,
) -> BasketDefinition:
    if constituents is None:
        constituents = (
            Constituent(ticker="AAPL", target_weight=0.40, sector="technology"),
            Constituent(ticker="MSFT", target_weight=0.40, sector="technology"),
            Constituent(ticker="GOOG", target_weight=0.20, sector="communications"),
        )
    return BasketDefinition(
        basket_id=basket_id,
        name=name,
        author_id=author,
        constituents=constituents,
        cadence=cadence,
        drift_threshold=drift_threshold,
    )


def _all_halal(_t: str) -> bool:
    return True


# --- Constituent validation -----------------------------------


def test_constituent_valid():
    c = Constituent(ticker="AAPL", target_weight=0.50, sector="technology")
    assert c.halal_compliant


def test_constituent_zero_weight_rejected():
    with pytest.raises(ValueError):
        Constituent(ticker="AAPL", target_weight=0)


def test_constituent_weight_above_one_rejected():
    with pytest.raises(ValueError):
        Constituent(ticker="AAPL", target_weight=1.5)


def test_constituent_empty_ticker_rejected():
    with pytest.raises(ValueError):
        Constituent(ticker=" ", target_weight=0.5)


# --- BasketDefinition validation ------------------------------


def test_basket_valid():
    b = _basket()
    assert len(b.constituents) == 3


def test_basket_empty_constituents_rejected():
    with pytest.raises(ValueError):
        BasketDefinition(
            basket_id="B1",
            name="X",
            author_id="alice",
            constituents=(),
            cadence=RebalanceCadence.MONTHLY,
        )


def test_basket_duplicate_ticker_rejected():
    bad = (
        Constituent(ticker="AAPL", target_weight=0.5),
        Constituent(ticker="AAPL", target_weight=0.5),
    )
    with pytest.raises(ValueError):
        _basket(constituents=bad)


def test_basket_weights_not_summing_to_one_rejected():
    bad = (
        Constituent(ticker="AAPL", target_weight=0.30),
        Constituent(ticker="MSFT", target_weight=0.30),
    )
    with pytest.raises(ValueError):
        _basket(constituents=bad)


def test_basket_weight_below_min_rejected():
    bad = (
        Constituent(ticker="A", target_weight=0.005),
        Constituent(ticker="B", target_weight=0.995),
    )
    with pytest.raises(ValueError):
        _basket(constituents=bad)


def test_basket_invalid_drift_threshold_rejected():
    with pytest.raises(ValueError):
        _basket(drift_threshold=0.0)


def test_basket_long_name_rejected():
    with pytest.raises(ValueError):
        _basket(name="x" * 200)


def test_basket_immutable():
    b = _basket()
    with pytest.raises(AttributeError):
        b.name = "X"  # type: ignore[misc]


# --- assert_basket_halal --------------------------------------


def test_assert_halal_passes_when_clean():
    b = _basket()
    assert_basket_halal(b)


def test_assert_halal_rejects_non_compliant():
    bad = (
        Constituent(ticker="AAPL", target_weight=0.50),
        Constituent(ticker="MO", target_weight=0.50, halal_compliant=False),
    )
    b = _basket(constituents=bad)
    with pytest.raises(ValueError):
        assert_basket_halal(b)


# --- compute_drift --------------------------------------------


def test_compute_drift_no_drift_when_actual_matches_target():
    b = _basket()
    drift = compute_drift(b, {"AAPL": 0.40, "MSFT": 0.40, "GOOG": 0.20})
    for d in drift:
        assert d.drift == 0.0
        assert not d.needs_rebalance


def test_compute_drift_flags_when_above_threshold():
    b = _basket(drift_threshold=0.03)
    drift = compute_drift(b, {"AAPL": 0.50, "MSFT": 0.40, "GOOG": 0.10})
    by_t = {d.ticker: d for d in drift}
    assert by_t["AAPL"].needs_rebalance
    assert by_t["GOOG"].needs_rebalance
    assert not by_t["MSFT"].needs_rebalance


def test_compute_drift_missing_ticker_treated_as_zero():
    b = _basket()
    drift = compute_drift(b, {"AAPL": 0.50})
    by_t = {d.ticker: d for d in drift}
    assert by_t["MSFT"].actual_weight == 0.0


def test_compute_drift_extra_ticker_ignored():
    """Pin: extra tickers in actual_weights are silently ignored."""
    b = _basket()
    drift = compute_drift(b, {"AAPL": 0.40, "MSFT": 0.40, "GOOG": 0.20, "RESIDUAL": 0.1})
    # Only 3 reports.
    assert len(drift) == 3


def test_compute_drift_negative_actual_rejected():
    b = _basket()
    with pytest.raises(ValueError):
        compute_drift(b, {"AAPL": -0.1, "MSFT": 0.5, "GOOG": 0.6})


# --- any_drift_exceeds ---------------------------------------


def test_any_drift_exceeds_true():
    b = _basket(drift_threshold=0.03)
    assert any_drift_exceeds(b, {"AAPL": 0.50, "MSFT": 0.40, "GOOG": 0.10})


def test_any_drift_exceeds_false():
    b = _basket(drift_threshold=0.10)
    assert not any_drift_exceeds(b, {"AAPL": 0.40, "MSFT": 0.40, "GOOG": 0.20})


# --- screen_basket -------------------------------------------


def test_screen_clean_emits_nothing():
    b = _basket()
    issues = screen_basket(b, is_ticker_halal=_all_halal)
    assert issues == ()


def test_screen_failing_constituent_divests():
    b = _basket()
    issues = screen_basket(b, is_ticker_halal=lambda t: t != "AAPL")
    by_t = {i.ticker: i for i in issues}
    assert "AAPL" in by_t
    assert by_t["AAPL"].action is ScreenAction.DIVEST


def test_screen_passing_but_marked_noncompliant_reweights():
    """Pin: a constituent currently halal but historically marked
    non-compliant → REWEIGHT (operator decides to re-include)."""
    constituents = (
        Constituent(ticker="AAPL", target_weight=0.50),
        Constituent(
            ticker="MSFT",
            target_weight=0.50,
            halal_compliant=False,
        ),
    )
    b = _basket(constituents=constituents)
    issues = screen_basket(b, is_ticker_halal=_all_halal)
    by_t = {i.ticker: i for i in issues}
    assert "MSFT" in by_t
    assert by_t["MSFT"].action is ScreenAction.REWEIGHT


# --- divest_failing -----------------------------------------


def test_divest_failing_drops_and_renormalises():
    b = _basket()
    nb = divest_failing(b, is_ticker_halal=lambda t: t != "GOOG")
    assert len(nb.constituents) == 2
    s = sum(c.target_weight for c in nb.constituents)
    assert s == pytest.approx(1.0)


def test_divest_failing_all_failures_rejected():
    b = _basket()
    with pytest.raises(ValueError):
        divest_failing(b, is_ticker_halal=lambda t: False)


def test_divest_failing_preserves_proportions():
    b = _basket()
    nb = divest_failing(b, is_ticker_halal=lambda t: t != "GOOG")
    # Original AAPL:MSFT was 0.40:0.40 (1:1); should remain 1:1.
    by_t = {c.ticker: c for c in nb.constituents}
    assert by_t["AAPL"].target_weight == pytest.approx(by_t["MSFT"].target_weight)


# --- next_rebalance ------------------------------------------


def test_next_rebalance_standard_cadence():
    b = _basket(cadence=RebalanceCadence.MONTHLY)
    nr = next_rebalance(b, last_rebalance=date(2026, 4, 11))
    assert nr == date(2026, 4, 11) + timedelta(days=30)


def test_next_rebalance_weekly():
    b = _basket(cadence=RebalanceCadence.WEEKLY)
    nr = next_rebalance(b, last_rebalance=date(2026, 4, 11))
    assert nr == date(2026, 4, 18)


def test_next_rebalance_pulled_forward_on_drift():
    b = _basket(drift_threshold=0.03)
    nr = next_rebalance(
        b,
        last_rebalance=date(2026, 4, 11),
        actual_weights={"AAPL": 0.50, "MSFT": 0.40, "GOOG": 0.10},
    )
    assert nr == date(2026, 4, 12)


def test_next_rebalance_not_pulled_forward_without_drift():
    b = _basket(cadence=RebalanceCadence.WEEKLY, drift_threshold=0.10)
    nr = next_rebalance(
        b,
        last_rebalance=date(2026, 4, 11),
        actual_weights={"AAPL": 0.42, "MSFT": 0.39, "GOOG": 0.19},
    )
    assert nr == date(2026, 4, 18)


# --- Render ------------------------------------------------


def test_render_basket_no_secret_leak():
    b = _basket(author="alice@example.com")
    out = render_basket(b)
    assert "alice@example.com" not in out


def test_render_basket_marks_non_halal():
    constituents = (
        Constituent(ticker="AAPL", target_weight=0.5),
        Constituent(ticker="MO", target_weight=0.5, halal_compliant=False),
    )
    b = _basket(constituents=constituents)
    out = render_basket(b)
    assert "non-halal" in out


def test_render_drift_empty():
    out = render_drift([])
    assert "No drift" in out


def test_render_drift_marks_warnings():
    b = _basket(drift_threshold=0.03)
    reports = compute_drift(b, {"AAPL": 0.50, "MSFT": 0.40, "GOOG": 0.10})
    out = render_drift(reports)
    assert "⚠️" in out


def test_render_screen_empty():
    out = render_screen([])
    assert "clean" in out


def test_render_screen_action_emoji():
    issues = (ScreenIssue(ticker="MO", action=ScreenAction.DIVEST, reason="..."),)
    out = render_screen(issues)
    assert "⛔" in out
