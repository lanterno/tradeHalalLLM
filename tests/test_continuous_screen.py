"""Tests for halal/continuous_screen.py — Round-5 Wave 1.H."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.continuous_screen import (
    CorporateAction,
    CorporateEvent,
    ScreenPolicy,
    ScreenSnapshot,
    ScreenStatus,
    filter_regressions,
    reassess,
    reassess_batch,
    render_result,
)


def _snap(**overrides) -> ScreenSnapshot:
    base = {
        "ticker": "AAPL",
        "debt_to_market_cap": 0.10,
        "interest_income_to_revenue": 0.01,
        "liquid_assets_to_market_cap": 0.20,
        "non_halal_revenue_pct": 0.01,
        "sector_is_halal": True,
        "snapshot_date": date(2026, 5, 1),
    }
    base.update(overrides)
    return ScreenSnapshot(**base)


def _ev(action: CorporateAction = CorporateAction.DEBT_ISSUANCE) -> CorporateEvent:
    return CorporateEvent(
        ticker="AAPL",
        action=action,
        event_date=date(2026, 5, 1),
        summary="filed 8-K",
    )


# --- Enum string-value pins --------------------------------------------------


def test_corporate_action_string_values():
    assert CorporateAction.DEBT_ISSUANCE.value == "debt_issuance"
    assert CorporateAction.DEBT_RETIREMENT.value == "debt_retirement"
    assert CorporateAction.ACQUISITION.value == "acquisition"
    assert CorporateAction.DIVESTITURE.value == "divestiture"
    assert CorporateAction.DIVIDEND_DECLARATION.value == "dividend_declaration"
    assert CorporateAction.EARNINGS_RELEASE.value == "earnings_release"
    assert CorporateAction.SECTOR_RECLASSIFICATION.value == "sector_reclassification"
    assert CorporateAction.BUSINESS_MIX_CHANGE.value == "business_mix_change"
    assert CorporateAction.BANKRUPTCY_FILING.value == "bankruptcy_filing"
    assert CorporateAction.GOING_PRIVATE.value == "going_private"


def test_screen_status_string_values():
    assert ScreenStatus.PASSED.value == "passed"
    assert ScreenStatus.FLAGGED.value == "flagged"
    assert ScreenStatus.FAILED.value == "failed"


# --- Policy validation -------------------------------------------------------


def test_default_policy_loads():
    p = ScreenPolicy()
    assert p.debt_ratio_cap == 0.30


def test_policy_zero_cap_rejected():
    with pytest.raises(ValueError):
        ScreenPolicy(debt_ratio_cap=0.0)


def test_policy_above_one_cap_rejected():
    with pytest.raises(ValueError):
        ScreenPolicy(debt_ratio_cap=1.5)


def test_policy_negative_buffer_rejected():
    with pytest.raises(ValueError):
        ScreenPolicy(flag_buffer=-0.01)


def test_policy_buffer_too_large_rejected():
    with pytest.raises(ValueError):
        ScreenPolicy(flag_buffer=0.30)  # equals smallest cap (5% interest income)


def test_policy_zero_stale_window_rejected():
    with pytest.raises(ValueError):
        ScreenPolicy(stale_after_days=0)


def test_policy_immutable():
    p = ScreenPolicy()
    with pytest.raises(AttributeError):
        p.debt_ratio_cap = 0.5  # type: ignore[misc]


# --- Snapshot validation -----------------------------------------------------


def test_snapshot_empty_ticker_rejected():
    with pytest.raises(ValueError):
        _snap(ticker="")


def test_snapshot_negative_debt_rejected():
    with pytest.raises(ValueError):
        _snap(debt_to_market_cap=-0.1)


def test_snapshot_immutable():
    s = _snap()
    with pytest.raises(AttributeError):
        s.debt_to_market_cap = 0.99  # type: ignore[misc]


# --- Event validation --------------------------------------------------------


def test_event_empty_ticker_rejected():
    with pytest.raises(ValueError):
        CorporateEvent(ticker="", action=CorporateAction.DEBT_ISSUANCE, event_date=date.today())


# --- Reassess core ----------------------------------------------------------


def test_reassess_clean_snapshot_passes():
    r = reassess(_snap(), _snap(), _ev(), today=date(2026, 5, 1))
    assert r.new_status is ScreenStatus.PASSED
    assert r.previous_status is ScreenStatus.PASSED
    assert not r.regression()


def test_reassess_passed_to_failed_on_debt_issuance():
    prev = _snap()
    curr = _snap(debt_to_market_cap=0.40, snapshot_date=date(2026, 5, 1))
    r = reassess(prev, curr, _ev(CorporateAction.DEBT_ISSUANCE), today=date(2026, 5, 1))
    assert r.previous_status is ScreenStatus.PASSED
    assert r.new_status is ScreenStatus.FAILED
    assert r.regression()


def test_reassess_passed_to_flagged_within_buffer():
    prev = _snap()
    curr = _snap(debt_to_market_cap=0.29)  # within 0.02 buffer of 0.30
    r = reassess(prev, curr, _ev(), today=date(2026, 5, 1))
    assert r.new_status is ScreenStatus.FLAGGED
    assert r.regression()


def test_reassess_failed_to_passed_after_debt_retirement():
    prev = _snap(debt_to_market_cap=0.40)
    curr = _snap(debt_to_market_cap=0.10)
    r = reassess(prev, curr, _ev(CorporateAction.DEBT_RETIREMENT), today=date(2026, 5, 1))
    assert r.previous_status is ScreenStatus.FAILED
    assert r.new_status is ScreenStatus.PASSED
    assert not r.regression()
    assert r.status_flipped()


def test_reassess_sector_change_fails_immediately():
    prev = _snap()
    curr = _snap(sector_is_halal=False)
    r = reassess(prev, curr, _ev(CorporateAction.BUSINESS_MIX_CHANGE), today=date(2026, 5, 1))
    assert r.new_status is ScreenStatus.FAILED


def test_reassess_changes_only_for_changed_metrics():
    """Unchanged metrics should not appear in the changes tuple."""
    prev = _snap()
    curr = _snap(debt_to_market_cap=0.20)  # only this changed
    r = reassess(prev, curr, _ev(), today=date(2026, 5, 1))
    metrics = {ch.metric for ch in r.changes}
    assert "debt_to_market_cap" in metrics
    assert "interest_income_to_revenue" not in metrics


def test_reassess_crossed_cap_marker():
    prev = _snap()
    curr = _snap(debt_to_market_cap=0.40)
    r = reassess(prev, curr, _ev(), today=date(2026, 5, 1))
    crossed = [ch for ch in r.changes if ch.metric == "debt_to_market_cap"]
    assert crossed and crossed[0].crossed_cap


def test_reassess_no_crossing_when_both_below_cap():
    prev = _snap(debt_to_market_cap=0.10)
    curr = _snap(debt_to_market_cap=0.20)  # both below 0.30
    r = reassess(prev, curr, _ev(), today=date(2026, 5, 1))
    crossed = [ch for ch in r.changes if ch.metric == "debt_to_market_cap"]
    assert crossed and crossed[0].crossed_cap is False


def test_reassess_ticker_mismatch_rejected():
    prev = _snap()
    curr = _snap(ticker="MSFT")
    with pytest.raises(ValueError):
        reassess(prev, curr, _ev(), today=date(2026, 5, 1))


def test_reassess_uses_explicit_previous_status():
    """When previous_status is supplied, classifier isn't re-run on prev."""
    prev = _snap(debt_to_market_cap=0.20)
    curr = _snap(debt_to_market_cap=0.40)
    r = reassess(
        prev,
        curr,
        _ev(),
        today=date(2026, 5, 1),
        previous_status=ScreenStatus.FLAGGED,
    )
    assert r.previous_status is ScreenStatus.FLAGGED


def test_reassess_stale_data_flagged():
    prev = _snap()
    curr = _snap(snapshot_date=date(2026, 4, 1))  # 30 days old
    r = reassess(prev, curr, _ev(), today=date(2026, 5, 1))
    assert r.is_stale


def test_reassess_fresh_data_not_stale():
    prev = _snap()
    curr = _snap(snapshot_date=date(2026, 4, 30))  # 1 day
    r = reassess(prev, curr, _ev(), today=date(2026, 5, 1))
    assert r.is_stale is False


def test_reassess_at_stale_boundary_not_stale():
    """Exactly 7 days old → still fresh."""
    prev = _snap()
    curr = _snap(snapshot_date=date(2026, 4, 24))
    r = reassess(prev, curr, _ev(), today=date(2026, 5, 1))
    assert r.is_stale is False


def test_reassess_just_past_boundary_stale():
    prev = _snap()
    curr = _snap(snapshot_date=date(2026, 4, 23))
    r = reassess(prev, curr, _ev(), today=date(2026, 5, 1))
    assert r.is_stale


# --- Custom policy -----------------------------------------------------------


def test_custom_policy_strict_debt_cap():
    """Operator with strict 0.20 cap fails ratios that default policy would pass."""
    prev = _snap()
    curr = _snap(debt_to_market_cap=0.25)
    r = reassess(prev, curr, _ev(), today=date(2026, 5, 1), policy=ScreenPolicy(debt_ratio_cap=0.20, flag_buffer=0.01))
    assert r.new_status is ScreenStatus.FAILED


# --- Batch + filters ---------------------------------------------------------


def test_reassess_batch_returns_one_per_pair():
    pairs = [
        (_snap(), _snap(), _ev()),
        (_snap(ticker="MSFT"), _snap(ticker="MSFT", debt_to_market_cap=0.40), CorporateEvent(ticker="MSFT", action=CorporateAction.DEBT_ISSUANCE, event_date=date(2026, 5, 1))),
    ]
    results = reassess_batch(pairs, today=date(2026, 5, 1))
    assert len(results) == 2
    assert results[0].new_status is ScreenStatus.PASSED
    assert results[1].new_status is ScreenStatus.FAILED


def test_filter_regressions_only_returns_worse():
    a = reassess(_snap(), _snap(), _ev(), today=date(2026, 5, 1))  # no change
    b = reassess(
        _snap(),
        _snap(debt_to_market_cap=0.40),
        _ev(),
        today=date(2026, 5, 1),
    )  # passed → failed
    c = reassess(
        _snap(debt_to_market_cap=0.40),
        _snap(),
        _ev(CorporateAction.DEBT_RETIREMENT),
        today=date(2026, 5, 1),
    )  # failed → passed (improvement)
    out = filter_regressions([a, b, c])
    assert b in out
    assert a not in out
    assert c not in out


# --- Render ------------------------------------------------------------------


def test_render_clean_uses_check():
    r = reassess(_snap(), _snap(), _ev(), today=date(2026, 5, 1))
    out = render_result(r)
    assert "✅" in out
    assert "AAPL" in out


def test_render_regression_uses_warning():
    r = reassess(_snap(), _snap(debt_to_market_cap=0.40), _ev(), today=date(2026, 5, 1))
    out = render_result(r)
    assert "⚠️" in out
    assert "passed→failed" in out


def test_render_includes_stale_marker():
    r = reassess(_snap(), _snap(snapshot_date=date(2026, 4, 1)), _ev(), today=date(2026, 5, 1))
    assert "[STALE DATA]" in render_result(r)


def test_render_no_secret_leak():
    r = reassess(_snap(), _snap(debt_to_market_cap=0.40), _ev(), today=date(2026, 5, 1))
    out = render_result(r)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ----------------------------------------------------------------------


def test_e2e_acquisition_flips_to_flagged_when_target_carries_debt():
    prev = _snap()
    curr = _snap(debt_to_market_cap=0.29, liquid_assets_to_market_cap=0.28)
    r = reassess(prev, curr, _ev(CorporateAction.ACQUISITION), today=date(2026, 5, 1))
    assert r.new_status is ScreenStatus.FLAGGED
    assert r.regression()


def test_replay_consistency():
    a = reassess(_snap(), _snap(debt_to_market_cap=0.40), _ev(), today=date(2026, 5, 1))
    b = reassess(_snap(), _snap(debt_to_market_cap=0.40), _ev(), today=date(2026, 5, 1))
    assert a == b
