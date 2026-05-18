"""Tests for `halal_trader.ml.risk_aggregator`.

Auxiliary primitive complementing Wave 4.G + crypto monitor SL/TP.
Covers: per-position risk math, account-level aggregation, pre-trade
gate (3 caps), no-secret render contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from halal_trader.ml.risk_aggregator import (
    DEFAULT_POLICY,
    Position,
    RiskGateOutcome,
    RiskPolicy,
    RiskSnapshot,
    aggregate_risk,
    at_risk_usd,
    evaluate_pre_trade_gate,
    render_decision,
    render_snapshot,
)

# --------------------------- Enum string pins --------------------------------


def test_risk_gate_outcome_string_values_pinned() -> None:
    assert RiskGateOutcome.APPROVED.value == "approved"
    assert RiskGateOutcome.REJECTED_TOTAL_RISK.value == "rejected_total_risk"
    assert RiskGateOutcome.REJECTED_SINGLE_POSITION.value == "rejected_single_position"
    assert RiskGateOutcome.REJECTED_POSITION_COUNT.value == "rejected_position_count"


# --------------------------- RiskPolicy --------------------------------------


def test_default_policy_pins() -> None:
    """Pin: conventional 6%/2%/20-position defaults."""

    assert DEFAULT_POLICY.max_total_at_risk_pct == 0.06
    assert DEFAULT_POLICY.max_single_position_risk_pct == 0.02
    assert DEFAULT_POLICY.max_position_count == 20


def test_policy_rejects_total_at_zero() -> None:
    with pytest.raises(ValueError, match="max_total_at_risk_pct"):
        RiskPolicy(max_total_at_risk_pct=0.0)


def test_policy_rejects_total_above_one() -> None:
    with pytest.raises(ValueError, match="max_total_at_risk_pct"):
        RiskPolicy(max_total_at_risk_pct=1.5)


def test_policy_rejects_single_above_total() -> None:
    """Pin: single-position cap can't exceed total cap."""

    with pytest.raises(ValueError, match="single_position"):
        RiskPolicy(
            max_total_at_risk_pct=0.05,
            max_single_position_risk_pct=0.10,  # higher than total
        )


def test_policy_rejects_zero_position_count() -> None:
    with pytest.raises(ValueError, match="max_position_count"):
        RiskPolicy(max_position_count=0)


def test_policy_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.max_total_at_risk_pct = 0.10  # type: ignore[misc]


# --------------------------- Position validation -----------------------------


def _position(**overrides: object) -> Position:
    base: dict[str, object] = {
        "symbol": "BTCUSDT",
        "notional_usd": 1000.0,
        "entry_price": 100.0,
        "stop_loss_price": 95.0,
    }
    base.update(overrides)
    return Position(**base)  # type: ignore[arg-type]


def test_position_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        _position(symbol="")


def test_position_rejects_zero_notional() -> None:
    with pytest.raises(ValueError, match="notional_usd"):
        _position(notional_usd=0.0)


def test_position_rejects_negative_notional() -> None:
    with pytest.raises(ValueError, match="notional_usd"):
        _position(notional_usd=-100.0)


def test_position_rejects_zero_entry() -> None:
    with pytest.raises(ValueError, match="entry_price"):
        _position(entry_price=0.0)


def test_position_rejects_zero_stop_loss() -> None:
    with pytest.raises(ValueError, match="stop_loss_price"):
        _position(stop_loss_price=0.0)


def test_position_rejects_sl_at_or_above_entry() -> None:
    """Pin: long positions only — SL must be strictly below entry."""

    with pytest.raises(ValueError, match="stop_loss_price"):
        _position(entry_price=100.0, stop_loss_price=100.0)
    with pytest.raises(ValueError, match="stop_loss_price"):
        _position(entry_price=100.0, stop_loss_price=105.0)


def test_position_is_frozen() -> None:
    p = _position()
    with pytest.raises(FrozenInstanceError):
        p.notional_usd = 99.99  # type: ignore[misc]


# --------------------------- at_risk_usd -------------------------------------


def test_at_risk_basic() -> None:
    """Pin: $1000 notional, $100 entry, $95 SL → $50 at risk.

    at_risk = notional × (entry - SL) / entry = 1000 × 5/100 = 50.
    """

    pos = _position(
        notional_usd=1000.0,
        entry_price=100.0,
        stop_loss_price=95.0,
    )
    assert at_risk_usd(pos) == pytest.approx(50.0)


def test_at_risk_tight_stop() -> None:
    """1% stop-loss → 1% of notional at risk."""

    pos = _position(
        notional_usd=1000.0,
        entry_price=100.0,
        stop_loss_price=99.0,  # 1% below entry
    )
    assert at_risk_usd(pos) == pytest.approx(10.0)


def test_at_risk_wide_stop() -> None:
    """10% stop-loss → 10% of notional at risk."""

    pos = _position(
        notional_usd=2000.0,
        entry_price=100.0,
        stop_loss_price=90.0,
    )
    assert at_risk_usd(pos) == pytest.approx(200.0)


def test_at_risk_proportional_to_notional() -> None:
    """Pin: at_risk scales linearly with notional."""

    pos_small = _position(
        notional_usd=100.0,
        entry_price=50.0,
        stop_loss_price=45.0,
    )
    pos_large = _position(
        notional_usd=1000.0,
        entry_price=50.0,
        stop_loss_price=45.0,
    )
    assert at_risk_usd(pos_large) == pytest.approx(at_risk_usd(pos_small) * 10)


# --------------------------- aggregate_risk ----------------------------------


def test_aggregate_empty_positions() -> None:
    snapshot = aggregate_risk([], account_value_usd=10_000)
    assert snapshot.position_count == 0
    assert snapshot.total_notional_usd == 0
    assert snapshot.total_at_risk_usd == 0
    assert snapshot.total_at_risk_pct == 0
    assert snapshot.largest_position_symbol == ""


def test_aggregate_single_position() -> None:
    pos = _position()  # $50 at risk
    snapshot = aggregate_risk([pos], account_value_usd=10_000)
    assert snapshot.position_count == 1
    assert snapshot.total_notional_usd == 1000.0
    assert snapshot.total_at_risk_usd == pytest.approx(50.0)
    assert snapshot.total_at_risk_pct == pytest.approx(0.005)  # 0.5%
    assert snapshot.largest_position_symbol == "BTCUSDT"


def test_aggregate_multiple_positions() -> None:
    """Pin: at-risk sums across positions."""

    pos1 = _position(symbol="BTCUSDT")  # $50 at risk
    pos2 = _position(
        symbol="ETHUSDT",
        notional_usd=2000.0,
        entry_price=200.0,
        stop_loss_price=190.0,  # $100 at risk
    )
    snapshot = aggregate_risk([pos1, pos2], account_value_usd=10_000)
    assert snapshot.position_count == 2
    assert snapshot.total_at_risk_usd == pytest.approx(150.0)
    assert snapshot.total_at_risk_pct == pytest.approx(0.015)  # 1.5%


def test_aggregate_largest_position_identified() -> None:
    pos1 = _position(symbol="BTCUSDT")  # $50 at risk
    pos2 = _position(
        symbol="ETHUSDT",
        notional_usd=2000.0,
        entry_price=200.0,
        stop_loss_price=190.0,  # $100 at risk
    )
    pos3 = _position(
        symbol="SOLUSDT",
        notional_usd=500.0,
        entry_price=20.0,
        stop_loss_price=19.0,  # $25 at risk
    )
    snapshot = aggregate_risk([pos1, pos2, pos3], account_value_usd=10_000)
    assert snapshot.largest_position_symbol == "ETHUSDT"
    assert snapshot.largest_position_risk_usd == pytest.approx(100.0)


def test_aggregate_rejects_zero_account() -> None:
    with pytest.raises(ValueError, match="account_value_usd"):
        aggregate_risk([], account_value_usd=0.0)


def test_aggregate_rejects_negative_account() -> None:
    with pytest.raises(ValueError, match="account_value_usd"):
        aggregate_risk([], account_value_usd=-100.0)


def test_aggregate_is_deterministic() -> None:
    """Pin: same input → same output."""

    positions = [_position()]
    a = aggregate_risk(positions, account_value_usd=10_000)
    b = aggregate_risk(positions, account_value_usd=10_000)
    assert a == b


def test_snapshot_rejects_negative_position_count() -> None:
    with pytest.raises(ValueError, match="position_count"):
        RiskSnapshot(
            position_count=-1,
            total_notional_usd=0,
            total_at_risk_usd=0,
            account_value_usd=10_000,
            total_at_risk_pct=0,
            max_single_position_risk_pct=0,
            largest_position_symbol="",
            largest_position_risk_usd=0,
        )


def test_snapshot_is_frozen() -> None:
    snapshot = aggregate_risk([], account_value_usd=10_000)
    with pytest.raises(FrozenInstanceError):
        snapshot.position_count = 99  # type: ignore[misc]


# --------------------------- evaluate_pre_trade_gate -------------------------


def test_gate_approves_clean_new_position() -> None:
    """Pin: small new position with empty book → APPROVED."""

    new_pos = _position(
        notional_usd=500.0,  # $25 at risk = 0.25% of $10k
    )
    decision = evaluate_pre_trade_gate(
        open_positions=[],
        new_position=new_pos,
        account_value_usd=10_000,
    )
    assert decision.outcome is RiskGateOutcome.APPROVED


def test_gate_rejects_single_position_too_large() -> None:
    """Pin: single position above 2% cap → REJECTED_SINGLE_POSITION.

    $1000 notional, 5% stop = $50 at risk. On $1000 account = 5%.
    """

    big_pos = _position(notional_usd=1000.0)  # $50 at risk
    decision = evaluate_pre_trade_gate(
        open_positions=[],
        new_position=big_pos,
        account_value_usd=1000.0,  # $50 / $1000 = 5% > 2% cap
    )
    assert decision.outcome is RiskGateOutcome.REJECTED_SINGLE_POSITION


def test_gate_approves_new_position_at_2pct_boundary() -> None:
    """Pin: exactly 2% on a new position is approved (inclusive)."""

    pos = _position(notional_usd=1000.0)  # $50 at risk
    decision = evaluate_pre_trade_gate(
        open_positions=[],
        new_position=pos,
        account_value_usd=2500.0,  # $50/$2500 = 2.0%
    )
    # At cap (=, not >), accepted
    assert decision.outcome is RiskGateOutcome.APPROVED


def test_gate_rejects_total_risk_breach() -> None:
    """Pin: existing 5% + new 2% = 7% total > 6% cap → REJECTED_TOTAL_RISK.

    Each existing position at 1% of $10k = $100 at risk;
    5 positions × $100 = $500 = 5% total existing.
    New position at 2% = $200, projected total = 7%.
    """

    existing = []
    for i in range(5):
        existing.append(
            Position(
                symbol=f"X{i}USDT",
                notional_usd=2000.0,
                entry_price=100.0,
                stop_loss_price=95.0,  # $100 at risk each
            )
        )
    # New $200 risk = 2%
    new_pos = Position(
        symbol="NEWUSDT",
        notional_usd=4000.0,
        entry_price=100.0,
        stop_loss_price=95.0,
    )
    decision = evaluate_pre_trade_gate(
        open_positions=existing,
        new_position=new_pos,
        account_value_usd=10_000,
    )
    assert decision.outcome is RiskGateOutcome.REJECTED_TOTAL_RISK
    assert decision.projected_total_at_risk_pct > 0.06


def test_gate_approves_at_total_boundary() -> None:
    """Pin: projected total at exactly 6% is APPROVED (inclusive)."""

    existing = []
    for i in range(4):
        existing.append(
            Position(
                symbol=f"X{i}USDT",
                notional_usd=2000.0,
                entry_price=100.0,
                stop_loss_price=95.0,  # $100 at risk = 1% each
            )
        )
    # New $200 risk = 2% → total = 4% + 2% = 6%
    new_pos = Position(
        symbol="NEWUSDT",
        notional_usd=4000.0,
        entry_price=100.0,
        stop_loss_price=95.0,
    )
    decision = evaluate_pre_trade_gate(
        open_positions=existing,
        new_position=new_pos,
        account_value_usd=10_000,
    )
    assert decision.outcome is RiskGateOutcome.APPROVED


def test_gate_rejects_position_count() -> None:
    """Pin: 20 existing + 1 new = 21 > 20 cap → REJECTED_POSITION_COUNT."""

    existing = [
        Position(
            symbol=f"X{i}USDT",
            notional_usd=100.0,  # tiny
            entry_price=100.0,
            stop_loss_price=99.5,
        )
        for i in range(20)
    ]
    new_pos = Position(
        symbol="NEWUSDT",
        notional_usd=100.0,
        entry_price=100.0,
        stop_loss_price=99.5,
    )
    decision = evaluate_pre_trade_gate(
        open_positions=existing,
        new_position=new_pos,
        account_value_usd=100_000,
    )
    assert decision.outcome is RiskGateOutcome.REJECTED_POSITION_COUNT


def test_gate_approves_at_position_count_boundary() -> None:
    """Pin: 19 existing + 1 new = 20 <= 20 cap → APPROVED."""

    existing = [
        Position(
            symbol=f"X{i}USDT",
            notional_usd=100.0,
            entry_price=100.0,
            stop_loss_price=99.5,
        )
        for i in range(19)
    ]
    new_pos = Position(
        symbol="NEWUSDT",
        notional_usd=100.0,
        entry_price=100.0,
        stop_loss_price=99.5,
    )
    decision = evaluate_pre_trade_gate(
        open_positions=existing,
        new_position=new_pos,
        account_value_usd=100_000,
    )
    assert decision.outcome is RiskGateOutcome.APPROVED


def test_gate_priority_position_count_first() -> None:
    """Pin: position count is checked first.

    A position-count breach with also a single-position breach should
    surface the position-count outcome.
    """

    existing = [
        Position(
            symbol=f"X{i}USDT",
            notional_usd=100.0,
            entry_price=100.0,
            stop_loss_price=99.5,
        )
        for i in range(20)
    ]
    # New position is also a single-position breach
    new_pos = Position(
        symbol="NEWUSDT",
        notional_usd=10_000.0,
        entry_price=100.0,
        stop_loss_price=90.0,  # $1000 at risk on $10k = 10% > 2%
    )
    decision = evaluate_pre_trade_gate(
        open_positions=existing,
        new_position=new_pos,
        account_value_usd=10_000,
    )
    assert decision.outcome is RiskGateOutcome.REJECTED_POSITION_COUNT


def test_gate_custom_strict_policy() -> None:
    """Strict 1% total cap rejects what default 6% would approve."""

    existing = [
        Position(
            symbol="X1USDT",
            notional_usd=2000.0,
            entry_price=100.0,
            stop_loss_price=95.0,  # $100 at risk = 1%
        ),
    ]
    new_pos = Position(
        symbol="NEWUSDT",
        notional_usd=200.0,
        entry_price=100.0,
        stop_loss_price=95.0,  # $10 at risk = 0.1%
    )
    strict = RiskPolicy(
        max_total_at_risk_pct=0.01,  # 1% total
        max_single_position_risk_pct=0.005,  # 0.5%
        max_position_count=10,
    )
    decision = evaluate_pre_trade_gate(
        open_positions=existing,
        new_position=new_pos,
        account_value_usd=10_000,
        policy=strict,
    )
    # Existing 1% + new 0.1% = 1.1% > 1% cap → REJECTED
    assert decision.outcome is RiskGateOutcome.REJECTED_TOTAL_RISK


def test_gate_decision_carries_message() -> None:
    """Pin: decision message is non-empty + actionable."""

    new_pos = _position(notional_usd=10_000.0)
    decision = evaluate_pre_trade_gate(
        open_positions=[],
        new_position=new_pos,
        account_value_usd=1000.0,
    )
    assert (
        "approved" in decision.message.lower()
        or "rejected" in decision.message.lower()
        or len(decision.message) > 0
    )


def test_gate_rejects_zero_account() -> None:
    new_pos = _position()
    with pytest.raises(ValueError, match="account_value_usd"):
        evaluate_pre_trade_gate(
            open_positions=[],
            new_position=new_pos,
            account_value_usd=0.0,
        )


# --------------------------- render ------------------------------------------


def test_render_snapshot_empty() -> None:
    snapshot = aggregate_risk([], account_value_usd=10_000)
    out = render_snapshot(snapshot)
    assert "0" in out  # 0 positions
    assert "—" in out  # no largest position


def test_render_snapshot_with_positions() -> None:
    pos = _position()  # BTCUSDT
    snapshot = aggregate_risk([pos], account_value_usd=10_000)
    out = render_snapshot(snapshot)
    assert "BTCUSDT" in out
    assert "$50" in out  # at-risk
    assert "0.50%" in out  # 0.5% of account


def test_render_snapshot_no_secret_leak() -> None:
    """Pin: render never includes broker account numbers / position
    IDs / operator-side fields."""

    pos = _position()
    snapshot = aggregate_risk([pos], account_value_usd=10_000)
    out = render_snapshot(snapshot)
    assert "account_id" not in out.lower()
    assert "broker_id" not in out.lower()
    assert "position_id" not in out.lower()
    assert "@" not in out  # no email-shape


def test_render_decision_approved() -> None:
    new_pos = _position()
    decision = evaluate_pre_trade_gate(
        open_positions=[],
        new_position=new_pos,
        account_value_usd=100_000,
    )
    out = render_decision(decision)
    assert "✅" in out
    assert "approved" in out


def test_render_decision_rejected_emoji_per_outcome() -> None:
    """Pin: each rejection outcome has a distinct emoji."""

    big_pos = _position(notional_usd=10_000.0)
    decision = evaluate_pre_trade_gate(
        open_positions=[],
        new_position=big_pos,
        account_value_usd=1000.0,
    )
    out = render_decision(decision)
    # REJECTED_SINGLE_POSITION has ⚠️
    assert "⚠️" in out


# --------------------------- e2e flows ---------------------------------------


def test_e2e_realistic_account_within_budget() -> None:
    """Real-world: $100k account with 5 positions at 1% risk each =
    5% total. Within 6% cap. New 0.5% position keeps us at 5.5%."""

    existing = []
    for i in range(5):
        existing.append(
            Position(
                symbol=f"SYM{i}USDT",
                notional_usd=20_000.0,  # 20% of account
                entry_price=100.0,
                stop_loss_price=95.0,  # 5% stop = $1000 at risk
            )
        )
    snapshot = aggregate_risk(existing, account_value_usd=100_000)
    assert snapshot.total_at_risk_pct == pytest.approx(0.05)
    assert snapshot.position_count == 5

    # Attempt new 0.5% risk position
    new_pos = Position(
        symbol="NEWUSDT",
        notional_usd=10_000.0,
        entry_price=100.0,
        stop_loss_price=95.0,  # $500 at risk = 0.5%
    )
    decision = evaluate_pre_trade_gate(
        open_positions=existing,
        new_position=new_pos,
        account_value_usd=100_000,
    )
    assert decision.outcome is RiskGateOutcome.APPROVED
    assert decision.projected_total_at_risk_pct == pytest.approx(0.055)


def test_e2e_overconcentrated_blocked() -> None:
    """Real-world: operator tries to add a too-large position;
    gate blocks before submission."""

    new_pos = Position(
        symbol="WHALEUSDT",
        notional_usd=20_000.0,
        entry_price=100.0,
        stop_loss_price=80.0,  # 20% stop = $4000 at risk = 4% > 2% cap
    )
    decision = evaluate_pre_trade_gate(
        open_positions=[],
        new_position=new_pos,
        account_value_usd=100_000,
    )
    assert decision.outcome is RiskGateOutcome.REJECTED_SINGLE_POSITION


def test_e2e_replay_consistency() -> None:
    positions = [_position()]
    new_pos = _position(symbol="NEWUSDT")
    a = evaluate_pre_trade_gate(
        open_positions=positions,
        new_position=new_pos,
        account_value_usd=10_000,
    )
    b = evaluate_pre_trade_gate(
        open_positions=positions,
        new_position=new_pos,
        account_value_usd=10_000,
    )
    assert a == b
