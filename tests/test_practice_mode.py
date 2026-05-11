"""Tests for education/practice_mode.py — Round-5 Wave 20.F."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from halal_trader.education.academy import Tier
from halal_trader.education.practice_mode import (
    PracticeFeature,
    PracticeSession,
    PracticeTrade,
    SessionStatus,
    conclude_session,
    is_feature_unlocked,
    min_tier_for,
    place_trade,
    render_milestones,
    render_session,
    resume_session,
    session_milestones,
    start_session,
    suspend_session,
    update_pnl,
)


def _session(
    session_id: str = "S1",
    user_id: str = "alice",
    seed_cash: float = 100_000.0,
    qualified_tier: Tier = Tier.BEGINNER,
    started_on: date = date(2026, 5, 1),
) -> PracticeSession:
    return start_session(
        session_id=session_id,
        user_id=user_id,
        seed_cash_usd=seed_cash,
        qualified_tier=qualified_tier,
        started_on=started_on,
    )


def _trade(
    trade_id: str = "T1",
    ticker: str = "AAPL",
    side: str = "buy",
    quantity: float = 10.0,
    fill_price: float = 200.0,
    placed_at: datetime = datetime(2026, 5, 2, 10, 0),
    feature: PracticeFeature = PracticeFeature.BASIC_TRADING,
) -> PracticeTrade:
    return PracticeTrade(
        trade_id=trade_id,
        ticker=ticker,
        side=side,
        quantity=quantity,
        fill_price=fill_price,
        placed_at=placed_at,
        feature=feature,
    )


# --- min_tier_for / is_feature_unlocked ----------


def test_min_tier_for_basic_is_beginner():
    assert min_tier_for(PracticeFeature.BASIC_TRADING) is Tier.BEGINNER


def test_min_tier_for_live_promotion_is_expert():
    assert min_tier_for(PracticeFeature.LIVE_PROMOTION) is Tier.EXPERT


def test_unlocked_when_tier_meets_minimum():
    assert is_feature_unlocked(PracticeFeature.OPTIONS_PRACTICE, Tier.INTERMEDIATE)


def test_unlocked_when_tier_above_minimum():
    assert is_feature_unlocked(PracticeFeature.OPTIONS_PRACTICE, Tier.EXPERT)


def test_not_unlocked_when_below_minimum():
    assert not is_feature_unlocked(PracticeFeature.MARGIN_PRACTICE, Tier.BEGINNER)


def test_basic_trading_unlocked_at_beginner():
    assert is_feature_unlocked(PracticeFeature.BASIC_TRADING, Tier.BEGINNER)


# --- PracticeTrade validation -------------------


def test_trade_valid():
    t = _trade()
    assert t.notional() == 2000.0


def test_trade_empty_id_rejected():
    with pytest.raises(ValueError):
        _trade(trade_id="")


def test_trade_invalid_side_rejected():
    with pytest.raises(ValueError):
        _trade(side="hold")


def test_trade_zero_quantity_rejected():
    with pytest.raises(ValueError):
        _trade(quantity=0)


def test_trade_immutable():
    t = _trade()
    with pytest.raises(AttributeError):
        t.fill_price = 0  # type: ignore[misc]


# --- start_session + PracticeSession ---------


def test_session_valid():
    s = _session()
    assert s.status is SessionStatus.ACTIVE
    assert s.balance_usd() == 100_000.0


def test_session_zero_seed_rejected():
    with pytest.raises(ValueError):
        _session(seed_cash=0)


def test_session_excessive_seed_rejected():
    with pytest.raises(ValueError):
        _session(seed_cash=50_000_000.0)


def test_session_empty_user_rejected():
    with pytest.raises(ValueError):
        _session(user_id="")


def test_session_immutable():
    s = _session()
    with pytest.raises(AttributeError):
        s.seed_cash_usd = 0  # type: ignore[misc]


# --- place_trade ----------------------------


def test_place_basic_trade():
    s = _session()
    s2 = place_trade(s, _trade())
    assert len(s2.trades) == 1


def test_place_trade_unlocked_feature():
    s = _session(qualified_tier=Tier.INTERMEDIATE)
    s2 = place_trade(s, _trade(feature=PracticeFeature.OPTIONS_PRACTICE))
    assert s2.trades[0].feature is PracticeFeature.OPTIONS_PRACTICE


def test_place_trade_locked_feature_rejected():
    s = _session(qualified_tier=Tier.BEGINNER)
    with pytest.raises(ValueError):
        place_trade(s, _trade(feature=PracticeFeature.MARGIN_PRACTICE))


def test_place_trade_on_suspended_rejected():
    s = suspend_session(_session())
    with pytest.raises(ValueError):
        place_trade(s, _trade())


def test_place_trade_on_concluded_rejected():
    s = conclude_session(_session(), on=date(2026, 5, 10))
    with pytest.raises(ValueError):
        place_trade(s, _trade())


def test_place_trade_duplicate_id_rejected():
    s = place_trade(_session(), _trade(trade_id="T1"))
    with pytest.raises(ValueError):
        place_trade(s, _trade(trade_id="T1"))


# --- update_pnl ----------------------------


def test_update_pnl_realised_only():
    s = _session()
    s2 = update_pnl(s, realised=500.0)
    assert s2.realised_pnl_usd == 500.0
    assert s2.unrealised_pnl_usd == 0.0


def test_update_pnl_unrealised_only():
    s = _session()
    s2 = update_pnl(s, unrealised=200.0)
    assert s2.unrealised_pnl_usd == 200.0


def test_update_pnl_both():
    s = _session()
    s2 = update_pnl(s, realised=500.0, unrealised=200.0)
    assert s2.balance_usd() == 100_700.0


def test_update_pnl_on_concluded_rejected():
    s = conclude_session(_session(), on=date(2026, 5, 10))
    with pytest.raises(ValueError):
        update_pnl(s, realised=500.0)


def test_update_pnl_out_of_bounds_rejected():
    s = _session()
    with pytest.raises(ValueError):
        update_pnl(s, realised=2e9)


# --- FSM transitions -----------------------


def test_suspend_active_to_suspended():
    s = _session()
    s2 = suspend_session(s)
    assert s2.status is SessionStatus.SUSPENDED


def test_suspend_already_suspended_rejected():
    s = suspend_session(_session())
    with pytest.raises(ValueError):
        suspend_session(s)


def test_resume_suspended_to_active():
    s = resume_session(suspend_session(_session()))
    assert s.status is SessionStatus.ACTIVE


def test_resume_active_rejected():
    s = _session()
    with pytest.raises(ValueError):
        resume_session(s)


def test_conclude_active():
    s = conclude_session(_session(), on=date(2026, 5, 10))
    assert s.status is SessionStatus.CONCLUDED


def test_conclude_suspended():
    s = conclude_session(suspend_session(_session()), on=date(2026, 5, 10))
    assert s.status is SessionStatus.CONCLUDED


def test_conclude_concluded_rejected():
    s = conclude_session(_session(), on=date(2026, 5, 10))
    with pytest.raises(ValueError):
        conclude_session(s, on=date(2026, 5, 11))


def test_conclude_before_started_rejected():
    s = _session(started_on=date(2026, 5, 1))
    with pytest.raises(ValueError):
        conclude_session(s, on=date(2026, 4, 1))


def test_conclude_with_promotion_requires_expert():
    s = _session(qualified_tier=Tier.INTERMEDIATE)
    with pytest.raises(ValueError):
        conclude_session(s, on=date(2026, 5, 10), promote_to_live=True)


def test_conclude_with_promotion_expert_allowed():
    s = _session(qualified_tier=Tier.EXPERT)
    s2 = conclude_session(s, on=date(2026, 5, 10), promote_to_live=True)
    assert s2.promoted_to_live is True


# --- session_milestones ----------------------


def test_milestones_no_trades():
    s = _session()
    m = session_milestones(s)
    assert m.n_trades == 0
    assert m.return_pct == 0.0


def test_milestones_with_trades_and_pnl():
    s = _session(seed_cash=100_000.0)
    s = place_trade(s, _trade(trade_id="T1", ticker="AAPL"))
    s = place_trade(s, _trade(trade_id="T2", ticker="MSFT"))
    s = update_pnl(s, realised=5_000.0, unrealised=2_000.0)
    m = session_milestones(s)
    assert m.n_trades == 2
    assert m.unique_tickers == 2
    assert m.realised_pnl_usd == 5_000.0
    assert m.final_balance_usd == pytest.approx(107_000.0)
    assert m.return_pct == pytest.approx(0.07)


# --- Render --------------------------------


def test_render_session_no_secret_leak():
    s = _session(user_id="alice@example.com")
    out = render_session(s)
    assert "alice@example.com" not in out


def test_render_session_status_emoji():
    s = _session()
    out = render_session(s)
    assert "🟢" in out


def test_render_session_marks_promotion():
    s = _session(qualified_tier=Tier.EXPERT)
    s = conclude_session(s, on=date(2026, 5, 10), promote_to_live=True)
    out = render_session(s)
    assert "🚀" in out


def test_render_milestones_format():
    s = _session(seed_cash=100_000.0)
    s = place_trade(s, _trade())
    s = update_pnl(s, realised=1000.0)
    m = session_milestones(s)
    out = render_milestones(m)
    assert "🏁" in out
    assert "return" in out
