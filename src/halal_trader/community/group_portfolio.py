"""Group portfolio (Musharakah pool) — Round-5 Wave 17.E.

Groups of users co-own an ongoing portfolio. Cap-table is updated by
contribution / withdrawal events; per-member equity share is computed
on demand. Performance accrues per share; AAOIFI Standard 12 applies
(profit + loss in proportion to capital).

This module differs from `halal/musharakah_coinvest.py`:
- the co-invest pool is *event-deal-bounded* (one deal liquidates)
- this group-portfolio is *ongoing* (continuous contributions,
  withdrawals, mark-to-market valuations)

Pinned semantics:

- **Closed-set EventKind** — CONTRIBUTION / WITHDRAWAL / MTM /
  DISTRIBUTION.
- **Closed-set GroupStatus FSM** — FORMING / ACTIVE / DISSOLVED.
- **Unit-based cap table.** Each member holds N units; unit NAV is
  `total_assets / total_units`. CONTRIBUTION mints new units at current
  NAV; WITHDRAWAL burns units at current NAV.
- **First contribution sets the inception NAV to $1/unit.**
- **No-secret-leak pin** — member IDs masked.
- **Pure-Python deterministic.**
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum


class EventKind(str, Enum):
    """Closed-set portfolio event ladder."""

    CONTRIBUTION = "contribution"
    WITHDRAWAL = "withdrawal"
    MTM = "mtm"
    """Mark-to-market: adjusts total_assets without minting/burning units."""
    DISTRIBUTION = "distribution"
    """Distribution to all members pro-rata (e.g. dividends paid out)."""


class GroupStatus(str, Enum):
    """Closed-set group lifecycle ladder."""

    FORMING = "forming"
    ACTIVE = "active"
    DISSOLVED = "dissolved"


@dataclass(frozen=True)
class PortfolioEvent:
    """One event in the portfolio's history."""

    event_id: str
    kind: EventKind
    occurred_at: datetime
    member_id: str = ""
    """Required for CONTRIBUTION / WITHDRAWAL; empty for MTM / DISTRIBUTION."""
    amount_usd: float = 0.0
    """For CONTRIBUTION / WITHDRAWAL / DISTRIBUTION."""
    new_total_assets_usd: float | None = None
    """For MTM only: the new total asset valuation."""

    def __post_init__(self) -> None:
        if not self.event_id or not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        if self.kind in (EventKind.CONTRIBUTION, EventKind.WITHDRAWAL):
            if not self.member_id.strip():
                raise ValueError(f"{self.kind.value} requires member_id")
            if self.amount_usd <= 0:
                raise ValueError(f"{self.kind.value} requires positive amount_usd")
        if self.kind is EventKind.MTM:
            if self.new_total_assets_usd is None or self.new_total_assets_usd < 0:
                raise ValueError("MTM requires non-negative new_total_assets_usd")
            if self.member_id:
                raise ValueError("MTM must not specify member_id")
        if self.kind is EventKind.DISTRIBUTION:
            if self.amount_usd <= 0:
                raise ValueError("DISTRIBUTION requires positive amount_usd")
            if self.member_id:
                raise ValueError("DISTRIBUTION must not specify a single member_id")


@dataclass(frozen=True)
class CapTableEntry:
    """One member's stake."""

    member_id: str
    units: float

    def __post_init__(self) -> None:
        if not self.member_id or not self.member_id.strip():
            raise ValueError("member_id must be non-empty")
        if self.units < 0:
            raise ValueError("units must be non-negative")


@dataclass(frozen=True)
class GroupState:
    """Result of folding events into a state."""

    group_id: str
    status: GroupStatus
    total_assets_usd: float
    total_units: float
    cap_table: tuple[CapTableEntry, ...]
    last_event_at: datetime | None

    def unit_nav(self) -> float:
        """Per-unit NAV. Returns 1.0 when no units have been issued."""
        if self.total_units <= 0:
            return 1.0
        return self.total_assets_usd / self.total_units

    def member_value(self, member_id: str) -> float:
        for e in self.cap_table:
            if e.member_id == member_id:
                return e.units * self.unit_nav()
        return 0.0

    def member_share(self, member_id: str) -> float:
        """Member's share of total_units in [0, 1]."""
        if self.total_units <= 0:
            return 0.0
        for e in self.cap_table:
            if e.member_id == member_id:
                return e.units / self.total_units
        return 0.0


def initial_state(group_id: str) -> GroupState:
    if not group_id or not group_id.strip():
        raise ValueError("group_id must be non-empty")
    return GroupState(
        group_id=group_id,
        status=GroupStatus.FORMING,
        total_assets_usd=0.0,
        total_units=0.0,
        cap_table=(),
        last_event_at=None,
    )


def _bump_units(
    cap_table: tuple[CapTableEntry, ...],
    member_id: str,
    delta_units: float,
) -> tuple[CapTableEntry, ...]:
    """Return new cap table with `delta_units` added to `member_id`'s
    stake (negative delta = burn). Drops zero-unit members."""
    found = False
    out: list[CapTableEntry] = []
    for e in cap_table:
        if e.member_id == member_id:
            new_units = e.units + delta_units
            if new_units < -1e-12:
                raise ValueError(f"member {member_id} cannot go negative units")
            if new_units > 1e-12:
                out.append(replace(e, units=new_units))
            found = True
        else:
            out.append(e)
    if not found:
        if delta_units < 0:
            raise ValueError(f"member {member_id} not on cap table")
        out.append(CapTableEntry(member_id=member_id, units=delta_units))
    out.sort(key=lambda e: e.member_id)
    return tuple(out)


def apply_event(state: GroupState, event: PortfolioEvent) -> GroupState:
    """Fold a single event into the state."""
    if state.status is GroupStatus.DISSOLVED:
        raise ValueError("cannot apply events to a DISSOLVED group")
    if state.last_event_at is not None and event.occurred_at < state.last_event_at:
        raise ValueError("events must arrive in non-decreasing chronological order")
    nav = state.unit_nav()
    new_assets = state.total_assets_usd
    new_units = state.total_units
    new_cap = state.cap_table
    new_status = state.status
    if event.kind is EventKind.CONTRIBUTION:
        # Inception: NAV pinned to 1.0 for the first contribution.
        if state.total_units <= 0:
            nav = 1.0
        minted_units = event.amount_usd / nav
        new_assets += event.amount_usd
        new_units += minted_units
        new_cap = _bump_units(new_cap, event.member_id, minted_units)
        if state.status is GroupStatus.FORMING:
            new_status = GroupStatus.ACTIVE
    elif event.kind is EventKind.WITHDRAWAL:
        if state.total_assets_usd <= 0:
            raise ValueError("cannot withdraw from empty group")
        member_entry = next(
            (e for e in state.cap_table if e.member_id == event.member_id),
            None,
        )
        if member_entry is None:
            raise ValueError(f"member {event.member_id} not on cap table")
        max_withdrawal = member_entry.units * nav
        if event.amount_usd > max_withdrawal + 1e-9:
            raise ValueError(
                f"withdrawal {event.amount_usd:.2f} exceeds member's holding {max_withdrawal:.2f}"
            )
        burned_units = event.amount_usd / nav
        new_assets -= event.amount_usd
        new_units -= burned_units
        new_cap = _bump_units(new_cap, event.member_id, -burned_units)
    elif event.kind is EventKind.MTM:
        assert event.new_total_assets_usd is not None
        new_assets = event.new_total_assets_usd
    elif event.kind is EventKind.DISTRIBUTION:
        if state.total_units <= 0:
            raise ValueError("cannot distribute with zero units")
        if event.amount_usd > state.total_assets_usd + 1e-9:
            raise ValueError("distribution exceeds total_assets")
        new_assets -= event.amount_usd
    return replace(
        state,
        total_assets_usd=new_assets,
        total_units=new_units,
        cap_table=new_cap,
        last_event_at=event.occurred_at,
        status=new_status,
    )


def fold_events(group_id: str, events: Iterable[PortfolioEvent]) -> GroupState:
    """Fold a chronological event stream into a final state."""
    state = initial_state(group_id)
    for e in events:
        state = apply_event(state, e)
    return state


def dissolve(state: GroupState) -> GroupState:
    """Transition the group to DISSOLVED. Pinned: only valid from ACTIVE."""
    if state.status is not GroupStatus.ACTIVE:
        raise ValueError(
            f"dissolve illegal from {state.status.value}; only ACTIVE groups can dissolve"
        )
    return replace(state, status=GroupStatus.DISSOLVED)


@dataclass(frozen=True)
class MemberAttribution:
    """Per-member attribution snapshot."""

    member_id: str
    units: float
    value_usd: float
    share_pct: float


def per_member_attribution(
    state: GroupState,
) -> tuple[MemberAttribution, ...]:
    nav = state.unit_nav()
    out: list[MemberAttribution] = []
    for e in state.cap_table:
        share = e.units / state.total_units if state.total_units > 0 else 0.0
        out.append(
            MemberAttribution(
                member_id=e.member_id,
                units=e.units,
                value_usd=e.units * nav,
                share_pct=share,
            )
        )
    return tuple(out)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_STATUS_EMOJI: dict[GroupStatus, str] = {
    GroupStatus.FORMING: "🌱",
    GroupStatus.ACTIVE: "🤝",
    GroupStatus.DISSOLVED: "🗂️",
}


def render_state(state: GroupState) -> str:
    head = (
        f"{_STATUS_EMOJI[state.status]} {state.group_id} "
        f"[{state.status.value}]: "
        f"NAV ${state.unit_nav():,.4f}/unit, "
        f"total assets ${state.total_assets_usd:,.2f}, "
        f"{state.total_units:,.4f} units, "
        f"{len(state.cap_table)} members"
    )
    return head


def render_attribution(
    rows: Sequence[MemberAttribution],
) -> str:
    if not rows:
        return "👥 No members."
    lines = [f"👥 Cap table ({len(rows)} members):"]
    for r in sorted(rows, key=lambda x: -x.share_pct):
        lines.append(
            f"  • {_mask(r.member_id)}: {r.units:,.4f} units "
            f"({r.share_pct * 100:.2f}%) = ${r.value_usd:,.2f}"
        )
    return "\n".join(lines)
