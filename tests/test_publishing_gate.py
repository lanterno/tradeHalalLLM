"""Tests for marketplace/publishing_gate.py — Round-5 Wave 21.A."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.marketplace.publishing_gate import (
    BacktestRecord,
    GateFailure,
    GatePolicy,
    GateVerdict,
    PaperRecord,
    StrategyApplication,
    evaluate,
    render_result,
)


def _bt(
    n_bars: int = 300,
    sharpe: float = 1.0,
    dd: float = 0.15,
) -> BacktestRecord:
    return BacktestRecord(
        n_bars=n_bars,
        sharpe=sharpe,
        max_drawdown_pct=dd,
        start_date=date(2025, 5, 1),
        end_date=date(2026, 5, 1),
    )


def _pp(
    started: date = date(2026, 2, 1),
    last_active: date = date(2026, 5, 10),
    trades: int = 50,
    sharpe: float = 0.6,
    dd: float = 0.10,
) -> PaperRecord:
    return PaperRecord(
        started_on=started,
        last_active_on=last_active,
        n_trades=trades,
        sharpe=sharpe,
        max_drawdown_pct=dd,
    )


def _app(
    application_id: str = "A1",
    strategy_id: str = "S1",
    author_id: str = "alice",
    universe: tuple[str, ...] = ("AAPL", "MSFT"),
    backtest: BacktestRecord | None = None,
    paper: PaperRecord | None = None,
    submitted_at: date = date(2026, 5, 11),
) -> StrategyApplication:
    return StrategyApplication(
        application_id=application_id,
        strategy_id=strategy_id,
        author_id=author_id,
        universe_tickers=universe,
        backtest=backtest or _bt(),
        paper=paper or _pp(),
        submitted_at=submitted_at,
    )


def _all_halal(_t: str) -> bool:
    return True


def _registered_alice(uid: str) -> bool:
    return uid == "alice"


# --- BacktestRecord validation -------------------------------------


def test_bt_valid():
    bt = _bt()
    assert bt.n_bars == 300


def test_bt_negative_bars_rejected():
    with pytest.raises(ValueError):
        BacktestRecord(
            n_bars=-1,
            sharpe=1.0,
            max_drawdown_pct=0.1,
            start_date=date(2025, 5, 1),
            end_date=date(2026, 5, 1),
        )


def test_bt_dd_outside_range_rejected():
    with pytest.raises(ValueError):
        BacktestRecord(
            n_bars=300,
            sharpe=1.0,
            max_drawdown_pct=1.5,
            start_date=date(2025, 5, 1),
            end_date=date(2026, 5, 1),
        )


def test_bt_unreasonable_sharpe_rejected():
    with pytest.raises(ValueError):
        _bt(sharpe=100.0)


def test_bt_dates_inverted_rejected():
    with pytest.raises(ValueError):
        BacktestRecord(
            n_bars=300,
            sharpe=1.0,
            max_drawdown_pct=0.1,
            start_date=date(2026, 5, 1),
            end_date=date(2025, 5, 1),
        )


# --- PaperRecord validation ----------------------------------------


def test_pp_valid():
    pp = _pp()
    assert pp.days_live() > 0


def test_pp_negative_trades_rejected():
    with pytest.raises(ValueError):
        PaperRecord(
            started_on=date(2026, 2, 1),
            last_active_on=date(2026, 5, 10),
            n_trades=-1,
            sharpe=0.5,
            max_drawdown_pct=0.1,
        )


def test_pp_last_active_before_started_rejected():
    with pytest.raises(ValueError):
        PaperRecord(
            started_on=date(2026, 5, 10),
            last_active_on=date(2026, 2, 1),
            n_trades=10,
            sharpe=0.5,
            max_drawdown_pct=0.1,
        )


def test_pp_days_live_pinned():
    pp = _pp(started=date(2026, 2, 1), last_active=date(2026, 5, 2))
    assert pp.days_live() == 90


# --- GatePolicy validation -----------------------------------------


def test_policy_default_valid():
    p = GatePolicy()
    assert p.min_paper_days == 90


def test_policy_invalid_bars_rejected():
    with pytest.raises(ValueError):
        GatePolicy(min_backtest_bars=0)


def test_policy_invalid_band_rejected():
    with pytest.raises(ValueError):
        GatePolicy(provisional_band=0.0)


# --- StrategyApplication validation ---------------------------------


def test_app_empty_universe_rejected():
    with pytest.raises(ValueError):
        _app(universe=())


def test_app_empty_id_rejected():
    with pytest.raises(ValueError):
        _app(application_id="")


def test_app_empty_ticker_rejected():
    with pytest.raises(ValueError):
        _app(universe=("AAPL", " "))


# --- evaluate — happy path ---------------------------------------


def test_evaluate_clean_approved():
    app = _app()
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    assert result.verdict is GateVerdict.APPROVED
    assert not result.failures


# --- evaluate — author + halal -----------------------------------


def test_evaluate_author_not_registered_rejected():
    app = _app(author_id="mallory")
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    assert result.verdict is GateVerdict.REJECTED
    assert GateFailure.AUTHOR_NOT_REGISTERED in result.failures


def test_evaluate_haram_universe_rejected():
    app = _app(universe=("AAPL", "MO"))
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=lambda t: t != "MO",
    )
    assert result.verdict is GateVerdict.REJECTED
    assert GateFailure.UNIVERSE_HARAM in result.failures


# --- evaluate — backtest -----------------------------------------


def test_evaluate_insufficient_bars_rejected():
    app = _app(backtest=_bt(n_bars=100))
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    assert GateFailure.INSUFFICIENT_BACKTEST in result.failures


def test_evaluate_low_sharpe_below_band_rejected():
    app = _app(backtest=_bt(sharpe=0.0))
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    assert GateFailure.LOW_SHARPE in result.failures
    assert result.verdict is GateVerdict.REJECTED


def test_evaluate_sharpe_in_provisional_band():
    """Default min_sharpe=0.50, provisional_band=0.10 → band [0.45, 0.50)."""
    app = _app(backtest=_bt(sharpe=0.47))
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    assert result.verdict is GateVerdict.PROVISIONAL
    assert GateFailure.LOW_SHARPE in result.provisional_reasons


def test_evaluate_high_drawdown_rejected():
    """Default max_dd=0.30, band 1.10 → reject above 0.33."""
    app = _app(backtest=_bt(dd=0.50))
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    assert GateFailure.HIGH_DRAWDOWN in result.failures


def test_evaluate_dd_in_provisional_band():
    app = _app(backtest=_bt(dd=0.32))
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    assert result.verdict is GateVerdict.PROVISIONAL
    assert GateFailure.HIGH_DRAWDOWN in result.provisional_reasons


# --- evaluate — paper --------------------------------------------


def test_evaluate_insufficient_paper_days():
    short_paper = _pp(started=date(2026, 4, 1), last_active=date(2026, 5, 10), trades=50)
    app = _app(paper=short_paper)
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    assert GateFailure.INSUFFICIENT_PAPER_DAYS in result.failures


def test_evaluate_insufficient_paper_trades():
    few_paper = _pp(trades=5)
    app = _app(paper=few_paper)
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    assert GateFailure.INSUFFICIENT_PAPER_TRADES in result.failures


def test_evaluate_paper_sharpe_low_rejected():
    bad_paper = _pp(sharpe=-1.0)
    app = _app(paper=bad_paper)
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    assert GateFailure.PAPER_SHARPE_LOW in result.failures


def test_evaluate_paper_sharpe_in_band_provisional():
    """Default min_paper_sharpe=0.30, band=0.10 → band [0.27, 0.30)."""
    band_paper = _pp(sharpe=0.28)
    app = _app(paper=band_paper)
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    assert result.verdict is GateVerdict.PROVISIONAL


def test_evaluate_paper_drawdown_high():
    bad_paper = _pp(dd=0.50)
    app = _app(paper=bad_paper)
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    assert GateFailure.PAPER_DRAWDOWN_HIGH in result.failures


# --- evaluate — combined --------------------------------------


def test_evaluate_combined_failures():
    app = _app(
        backtest=_bt(n_bars=10, sharpe=-1.0, dd=0.80),
        paper=_pp(trades=1, sharpe=-1.0),
    )
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    assert result.verdict is GateVerdict.REJECTED
    # Multiple failures captured.
    assert len(result.failures) >= 3


def test_evaluate_provisional_only_when_no_failures():
    """Pin: a single fail flips to REJECTED even with provisionals."""
    app = _app(
        backtest=_bt(sharpe=0.47),  # provisional
        paper=_pp(trades=1),  # failure
    )
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    assert result.verdict is GateVerdict.REJECTED


# --- Render --------------------------------------------------


def test_render_approved():
    app = _app()
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    out = render_result(result)
    assert "✅" in out
    assert "APPROVED" in out


def test_render_rejected_includes_notes():
    app = _app(author_id="mallory")
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    out = render_result(result)
    assert "❌" in out
    assert "registry" in out.lower()


def test_render_provisional_emoji():
    app = _app(backtest=_bt(sharpe=0.47))
    result = evaluate(
        app,
        is_author_registered=_registered_alice,
        is_ticker_halal=_all_halal,
    )
    out = render_result(result)
    assert "🟡" in out
