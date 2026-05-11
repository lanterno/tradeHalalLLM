"""Tests for community/group_portfolio.py — Round-5 Wave 17.E."""

from __future__ import annotations

from datetime import datetime

import pytest

from halal_trader.community.group_portfolio import (
    EventKind,
    GroupStatus,
    PortfolioEvent,
    apply_event,
    dissolve,
    fold_events,
    initial_state,
    per_member_attribution,
    render_attribution,
    render_state,
)


def _event(
    event_id: str = "E1",
    kind: EventKind = EventKind.CONTRIBUTION,
    occurred_at: datetime = datetime(2026, 5, 1, 10, 0),
    member_id: str = "alice",
    amount_usd: float = 1000.0,
    new_total_assets_usd: float | None = None,
) -> PortfolioEvent:
    if kind is EventKind.MTM:
        member_id = ""
        if new_total_assets_usd is None:
            new_total_assets_usd = 0.0
    if kind is EventKind.DISTRIBUTION:
        member_id = ""
    return PortfolioEvent(
        event_id=event_id,
        kind=kind,
        occurred_at=occurred_at,
        member_id=member_id,
        amount_usd=amount_usd,
        new_total_assets_usd=new_total_assets_usd,
    )


# --- PortfolioEvent validation ---------------------------


def test_event_valid_contribution():
    e = _event()
    assert e.kind is EventKind.CONTRIBUTION


def test_event_empty_id_rejected():
    with pytest.raises(ValueError):
        _event(event_id="")


def test_event_contribution_without_member_rejected():
    with pytest.raises(ValueError):
        PortfolioEvent(
            event_id="E1",
            kind=EventKind.CONTRIBUTION,
            occurred_at=datetime(2026, 5, 1),
            member_id="",
            amount_usd=1000.0,
        )


def test_event_contribution_zero_amount_rejected():
    with pytest.raises(ValueError):
        _event(amount_usd=0)


def test_event_mtm_without_new_assets_rejected():
    with pytest.raises(ValueError):
        PortfolioEvent(
            event_id="E1",
            kind=EventKind.MTM,
            occurred_at=datetime(2026, 5, 1),
            new_total_assets_usd=None,
        )


def test_event_mtm_with_member_rejected():
    with pytest.raises(ValueError):
        PortfolioEvent(
            event_id="E1",
            kind=EventKind.MTM,
            occurred_at=datetime(2026, 5, 1),
            member_id="alice",
            new_total_assets_usd=1000.0,
        )


def test_event_distribution_with_member_rejected():
    with pytest.raises(ValueError):
        PortfolioEvent(
            event_id="E1",
            kind=EventKind.DISTRIBUTION,
            occurred_at=datetime(2026, 5, 1),
            member_id="alice",
            amount_usd=100.0,
        )


def test_event_immutable():
    e = _event()
    with pytest.raises(AttributeError):
        e.amount_usd = 0  # type: ignore[misc]


# --- initial_state -------------------------------------


def test_initial_state_valid():
    s = initial_state("G1")
    assert s.status is GroupStatus.FORMING
    assert s.total_assets_usd == 0
    assert s.total_units == 0


def test_initial_state_empty_id_rejected():
    with pytest.raises(ValueError):
        initial_state("")


def test_unit_nav_starts_at_one():
    s = initial_state("G1")
    assert s.unit_nav() == 1.0


# --- apply_event — CONTRIBUTION -----------------------


def test_first_contribution_mints_units_at_dollar_per_unit():
    s = initial_state("G1")
    e = _event(amount_usd=1000.0)
    s2 = apply_event(s, e)
    assert s2.total_assets_usd == 1000.0
    assert s2.total_units == 1000.0
    assert s2.unit_nav() == 1.0
    assert s2.status is GroupStatus.ACTIVE


def test_second_contribution_at_current_nav():
    s = apply_event(initial_state("G1"), _event(amount_usd=1000.0))
    # MTM up: assets to $2000 with 1000 units → NAV = $2.
    s = apply_event(
        s,
        _event(
            event_id="E2",
            kind=EventKind.MTM,
            occurred_at=datetime(2026, 5, 2),
            new_total_assets_usd=2000.0,
        ),
    )
    assert s.unit_nav() == 2.0
    # Bob contributes $1000 at NAV $2 → mints 500 units.
    s = apply_event(
        s,
        _event(
            event_id="E3",
            member_id="bob",
            amount_usd=1000.0,
            occurred_at=datetime(2026, 5, 3),
        ),
    )
    bob = next(e for e in s.cap_table if e.member_id == "bob")
    assert bob.units == pytest.approx(500.0)
    assert s.total_assets_usd == 3000.0
    assert s.total_units == 1500.0


# --- apply_event — WITHDRAWAL ----------------------


def test_withdrawal_burns_units():
    s = apply_event(initial_state("G1"), _event(amount_usd=1000.0))
    s = apply_event(
        s,
        _event(
            event_id="E2",
            kind=EventKind.WITHDRAWAL,
            occurred_at=datetime(2026, 5, 2),
            member_id="alice",
            amount_usd=400.0,
        ),
    )
    assert s.total_assets_usd == 600.0
    assert s.total_units == 600.0
    alice = next(e for e in s.cap_table if e.member_id == "alice")
    assert alice.units == 600.0


def test_withdrawal_full_member_exits():
    s = apply_event(initial_state("G1"), _event(amount_usd=1000.0))
    s = apply_event(
        s,
        _event(
            event_id="E2",
            kind=EventKind.WITHDRAWAL,
            occurred_at=datetime(2026, 5, 2),
            member_id="alice",
            amount_usd=1000.0,
        ),
    )
    assert not any(e.member_id == "alice" for e in s.cap_table)


def test_withdrawal_above_member_holding_rejected():
    s = apply_event(initial_state("G1"), _event(amount_usd=1000.0))
    with pytest.raises(ValueError):
        apply_event(
            s,
            _event(
                event_id="E2",
                kind=EventKind.WITHDRAWAL,
                occurred_at=datetime(2026, 5, 2),
                member_id="alice",
                amount_usd=2000.0,
            ),
        )


def test_withdrawal_unknown_member_rejected():
    s = apply_event(initial_state("G1"), _event(amount_usd=1000.0))
    with pytest.raises(ValueError):
        apply_event(
            s,
            _event(
                event_id="E2",
                kind=EventKind.WITHDRAWAL,
                occurred_at=datetime(2026, 5, 2),
                member_id="charlie",
                amount_usd=100.0,
            ),
        )


# --- apply_event — MTM -------------------------------


def test_mtm_adjusts_total_only():
    s = apply_event(initial_state("G1"), _event(amount_usd=1000.0))
    s = apply_event(
        s,
        _event(
            event_id="E2",
            kind=EventKind.MTM,
            occurred_at=datetime(2026, 5, 2),
            new_total_assets_usd=1500.0,
        ),
    )
    assert s.total_assets_usd == 1500.0
    # Units unchanged.
    assert s.total_units == 1000.0
    assert s.unit_nav() == 1.5


def test_mtm_can_decrease():
    s = apply_event(initial_state("G1"), _event(amount_usd=1000.0))
    s = apply_event(
        s,
        _event(
            event_id="E2",
            kind=EventKind.MTM,
            occurred_at=datetime(2026, 5, 2),
            new_total_assets_usd=800.0,
        ),
    )
    assert s.unit_nav() == 0.8


# --- apply_event — DISTRIBUTION --------------------


def test_distribution_reduces_assets_keeps_units():
    s = apply_event(initial_state("G1"), _event(amount_usd=1000.0))
    s = apply_event(
        s,
        _event(
            event_id="E2",
            kind=EventKind.DISTRIBUTION,
            occurred_at=datetime(2026, 5, 2),
            amount_usd=100.0,
        ),
    )
    assert s.total_assets_usd == 900.0
    assert s.total_units == 1000.0


def test_distribution_above_assets_rejected():
    s = apply_event(initial_state("G1"), _event(amount_usd=1000.0))
    with pytest.raises(ValueError):
        apply_event(
            s,
            _event(
                event_id="E2",
                kind=EventKind.DISTRIBUTION,
                occurred_at=datetime(2026, 5, 2),
                amount_usd=2000.0,
            ),
        )


def test_distribution_with_zero_units_rejected():
    s = initial_state("G1")
    with pytest.raises(ValueError):
        apply_event(
            s,
            _event(
                event_id="E1",
                kind=EventKind.DISTRIBUTION,
                occurred_at=datetime(2026, 5, 1),
                amount_usd=100.0,
            ),
        )


# --- Chronological order pin ----------------------


def test_out_of_order_events_rejected():
    s = apply_event(initial_state("G1"), _event(amount_usd=1000.0))
    with pytest.raises(ValueError):
        apply_event(
            s,
            _event(
                event_id="E2",
                kind=EventKind.MTM,
                occurred_at=datetime(2026, 4, 1),  # earlier than first
                new_total_assets_usd=2000.0,
            ),
        )


# --- fold_events -------------------------------


def test_fold_events_full_lifecycle():
    events = [
        _event(
            event_id="E1",
            kind=EventKind.CONTRIBUTION,
            occurred_at=datetime(2026, 5, 1),
            member_id="alice",
            amount_usd=1000.0,
        ),
        _event(
            event_id="E2",
            kind=EventKind.CONTRIBUTION,
            occurred_at=datetime(2026, 5, 2),
            member_id="bob",
            amount_usd=1000.0,
        ),
        _event(
            event_id="E3",
            kind=EventKind.MTM,
            occurred_at=datetime(2026, 5, 10),
            new_total_assets_usd=2400.0,
        ),
    ]
    state = fold_events("G1", events)
    assert state.total_assets_usd == 2400.0
    # Two members, equal contributions → equal units → equal shares.
    by_id = {e.member_id: e for e in state.cap_table}
    assert by_id["alice"].units == by_id["bob"].units


# --- dissolve ----------------------------------


def test_dissolve_active_to_dissolved():
    s = apply_event(initial_state("G1"), _event(amount_usd=1000.0))
    s2 = dissolve(s)
    assert s2.status is GroupStatus.DISSOLVED


def test_dissolve_forming_rejected():
    s = initial_state("G1")
    with pytest.raises(ValueError):
        dissolve(s)


def test_dissolve_dissolved_rejected():
    s = apply_event(initial_state("G1"), _event(amount_usd=1000.0))
    s = dissolve(s)
    with pytest.raises(ValueError):
        dissolve(s)


def test_dissolved_blocks_events():
    s = apply_event(initial_state("G1"), _event(amount_usd=1000.0))
    s = dissolve(s)
    with pytest.raises(ValueError):
        apply_event(
            s,
            _event(
                event_id="E2",
                kind=EventKind.MTM,
                occurred_at=datetime(2026, 5, 10),
                new_total_assets_usd=2000.0,
            ),
        )


# --- per_member_attribution -----------------


def test_attribution_proportional_to_units():
    events = [
        _event(
            event_id="E1",
            kind=EventKind.CONTRIBUTION,
            occurred_at=datetime(2026, 5, 1),
            member_id="alice",
            amount_usd=1000.0,
        ),
        _event(
            event_id="E2",
            kind=EventKind.CONTRIBUTION,
            occurred_at=datetime(2026, 5, 2),
            member_id="bob",
            amount_usd=3000.0,
        ),
    ]
    state = fold_events("G1", events)
    attr = per_member_attribution(state)
    by_id = {a.member_id: a for a in attr}
    assert by_id["alice"].share_pct == pytest.approx(0.25)
    assert by_id["bob"].share_pct == pytest.approx(0.75)


# --- GroupState helpers ----------------------


def test_member_value():
    s = apply_event(initial_state("G1"), _event(amount_usd=1000.0))
    assert s.member_value("alice") == 1000.0
    assert s.member_value("unknown") == 0.0


def test_member_share():
    events = [
        _event(
            event_id="E1",
            kind=EventKind.CONTRIBUTION,
            occurred_at=datetime(2026, 5, 1),
            member_id="alice",
            amount_usd=600.0,
        ),
        _event(
            event_id="E2",
            kind=EventKind.CONTRIBUTION,
            occurred_at=datetime(2026, 5, 2),
            member_id="bob",
            amount_usd=400.0,
        ),
    ]
    s = fold_events("G1", events)
    assert s.member_share("alice") == pytest.approx(0.60)
    assert s.member_share("bob") == pytest.approx(0.40)


# --- Render -------------------------------


def test_render_state_status_emoji():
    s = initial_state("G1")
    out = render_state(s)
    assert "🌱" in out


def test_render_attribution_no_secret_leak():
    s = apply_event(
        initial_state("G1"),
        _event(member_id="alice@example.com", amount_usd=1000.0),
    )
    attr = per_member_attribution(s)
    out = render_attribution(attr)
    assert "alice@example.com" not in out


def test_render_attribution_empty():
    out = render_attribution([])
    assert "No members" in out


def test_render_attribution_sorted_by_share():
    events = [
        _event(
            event_id="E1",
            kind=EventKind.CONTRIBUTION,
            occurred_at=datetime(2026, 5, 1),
            member_id="alice",
            amount_usd=100.0,
        ),
        _event(
            event_id="E2",
            kind=EventKind.CONTRIBUTION,
            occurred_at=datetime(2026, 5, 2),
            member_id="bob",
            amount_usd=900.0,
        ),
    ]
    s = fold_events("G1", events)
    attr = per_member_attribution(s)
    out = render_attribution(attr)
    # Bob (90%) should appear before Alice (10%).
    assert out.find("bo") < out.find("al")
