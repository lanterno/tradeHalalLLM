"""Halal LP/GP fund structure — Round-5 Wave 6.G.

For operators running their own halal funds, the standard
**LP (Limited Partner) / GP (General Partner)** structure has to be
adapted: conventional "2 and 20" with a hurdle clauses are riba-laden
because the hurdle effectively guarantees a return to the LP if the GP
misses (i.e. catch-up clawback). The halal equivalents:

1. **Mudarabah fund** — LPs are rabb-al-mal (capital), GP is mudarib
   (manager). Profit shared per pre-agreed ratio; loss borne by LPs
   only (no negligence). No hurdle, no catch-up.

2. **Musharakah fund** — LPs and GP both contribute capital; profit
   AND loss shared in proportion to capital + agreed profit-share.

Both reject:
- Hurdle / catch-up clauses (guaranteed-return language).
- Performance fee structured as % of excess returns (mathematical
  guarantee on LP return).
- Preferred return on capital (riba).

This module is the **fund-spec validator + per-LP attribution + per-
period distribution computer**.

Pinned semantics:

- **Closed-set FundKind** — MUDARABAH / MUSHARAKAH.
- **Closed-set FundStatus FSM** — FORMING → ACTIVE → DISSOLVING →
  WOUND_DOWN.
- **Closed-set ProhibitedClause ladder** — HURDLE / CATCH_UP /
  PREFERRED_RETURN / GUARANTEED_RETURN.
- **Profit-share ratio in (0, 1)** — both parties get a non-zero
  slice (a 0% GP carry is a gift, not Mudarabah).
- **Mudarabah loss-rule pin**: loss borne by LPs in proportion to
  contributed capital; GP loses time/effort only.
- **Musharakah loss-rule pin**: loss borne pro-rata by all capital
  contributors (including GP if they contributed capital).
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum


class FundKind(str, Enum):
    """Closed-set fund-kind ladder."""

    MUDARABAH = "mudarabah"
    MUSHARAKAH = "musharakah"


class FundStatus(str, Enum):
    """Closed-set fund FSM ladder."""

    FORMING = "forming"
    ACTIVE = "active"
    DISSOLVING = "dissolving"
    WOUND_DOWN = "wound_down"


class ProhibitedClause(str, Enum):
    """Closed-set ladder of clauses incompatible with halal fund structures."""

    HURDLE = "hurdle"
    CATCH_UP = "catch_up"
    PREFERRED_RETURN = "preferred_return"
    GUARANTEED_RETURN = "guaranteed_return"


@dataclass(frozen=True)
class LPCommitment:
    """One LP's capital commitment."""

    lp_id: str
    committed_capital_usd: float
    funded_capital_usd: float

    def __post_init__(self) -> None:
        if not self.lp_id or not self.lp_id.strip():
            raise ValueError("lp_id must be non-empty")
        if self.committed_capital_usd <= 0:
            raise ValueError("committed_capital_usd must be positive")
        if self.funded_capital_usd < 0:
            raise ValueError("funded_capital_usd must be non-negative")
        if self.funded_capital_usd > self.committed_capital_usd + 1e-9:
            raise ValueError("funded cannot exceed committed")


@dataclass(frozen=True)
class FundTerms:
    """Halal fund terms."""

    fund_id: str
    kind: FundKind
    gp_id: str
    gp_capital_usd: float
    """GP's own capital. Must be > 0 for MUSHARAKAH; can be 0 for
    pure-Mudarabah."""
    gp_profit_share_pct: float
    """GP's slice of profit. (1 - this) goes to LPs pro-rata."""
    base_management_fee_annual_pct: float
    """Flat Wakalah-style fee on AUM. Capped at 3%/yr (≥3% reads as carry)."""
    has_hurdle: bool = False
    has_catch_up: bool = False
    has_preferred_return: bool = False
    has_guaranteed_return: bool = False
    inception_date: date | None = None

    def __post_init__(self) -> None:
        if not self.fund_id or not self.fund_id.strip():
            raise ValueError("fund_id must be non-empty")
        if not self.gp_id or not self.gp_id.strip():
            raise ValueError("gp_id must be non-empty")
        if self.gp_capital_usd < 0:
            raise ValueError("gp_capital_usd must be non-negative")
        if not 0.0 < self.gp_profit_share_pct < 1.0:
            raise ValueError("gp_profit_share_pct must be in (0, 1)")
        if not 0.0 <= self.base_management_fee_annual_pct < 0.03:
            raise ValueError(
                "base_management_fee_annual_pct must be in [0, 0.03) — "
                "≥3% reads as performance carry, not Wakalah"
            )
        if self.kind is FundKind.MUSHARAKAH and self.gp_capital_usd <= 0:
            raise ValueError("MUSHARAKAH fund requires non-zero gp_capital_usd")


def validate_clauses(terms: FundTerms) -> tuple[ProhibitedClause, ...]:
    """Surface every prohibited clause set on the terms.

    Pinned: returns the *complete* list of violations in one pass.
    """
    out: list[ProhibitedClause] = []
    if terms.has_hurdle:
        out.append(ProhibitedClause.HURDLE)
    if terms.has_catch_up:
        out.append(ProhibitedClause.CATCH_UP)
    if terms.has_preferred_return:
        out.append(ProhibitedClause.PREFERRED_RETURN)
    if terms.has_guaranteed_return:
        out.append(ProhibitedClause.GUARANTEED_RETURN)
    return tuple(out)


def is_halal(terms: FundTerms) -> bool:
    return not validate_clauses(terms)


@dataclass(frozen=True)
class Fund:
    """A halal LP/GP fund instance."""

    terms: FundTerms
    lps: tuple[LPCommitment, ...]
    status: FundStatus = FundStatus.FORMING
    """Lifecycle status — operator transitions via `transition_status`."""

    def __post_init__(self) -> None:
        if not self.lps:
            raise ValueError("fund must have at least one LP")
        # Unique LP IDs; LP ID must differ from GP.
        ids = [c.lp_id for c in self.lps]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate lp_id")
        for c in self.lps:
            if c.lp_id == self.terms.gp_id:
                raise ValueError("GP cannot also be an LP")
        if not is_halal(self.terms):
            raise ValueError(
                f"fund terms contain prohibited clauses: "
                f"{[c.value for c in validate_clauses(self.terms)]}"
            )

    def total_lp_funded_usd(self) -> float:
        return sum(c.funded_capital_usd for c in self.lps)

    def total_capital_usd(self) -> float:
        return self.total_lp_funded_usd() + self.terms.gp_capital_usd


_LEGAL_TRANSITIONS: dict[FundStatus, set[FundStatus]] = {
    FundStatus.FORMING: {FundStatus.ACTIVE, FundStatus.WOUND_DOWN},
    FundStatus.ACTIVE: {FundStatus.DISSOLVING},
    FundStatus.DISSOLVING: {FundStatus.WOUND_DOWN},
    FundStatus.WOUND_DOWN: set(),
}


def transition_status(fund: Fund, *, new_status: FundStatus) -> Fund:
    """Move the fund through the FSM."""
    if new_status not in _LEGAL_TRANSITIONS[fund.status]:
        raise ValueError(f"illegal transition {fund.status.value} → {new_status.value}")
    return replace(fund, status=new_status)


@dataclass(frozen=True)
class DistributionRecord:
    """One distribution slice."""

    party_id: str
    role: str
    """'lp' or 'gp'."""
    proceeds: float


def distribute(fund: Fund, *, period_pnl: float) -> tuple[DistributionRecord, ...]:
    """Distribute period P&L per the fund's structure.

    Pinned:
    - Profit branch: GP gets `gp_profit_share_pct × pnl`; LPs split
      the rest pro-rata by funded capital.
    - Loss branch (MUDARABAH): GP gets 0; LPs absorb pro-rata.
    - Loss branch (MUSHARAKAH): all capital (LPs + GP) absorbs pro-
      rata by capital share.
    """
    total_funded_lp = fund.total_lp_funded_usd()
    if total_funded_lp <= 0:
        raise ValueError("no LP funded capital — cannot distribute")
    records: list[DistributionRecord] = []
    if period_pnl >= 0:
        gp_share = period_pnl * fund.terms.gp_profit_share_pct
        lp_total = period_pnl - gp_share
        records.append(
            DistributionRecord(
                party_id=fund.terms.gp_id,
                role="gp",
                proceeds=gp_share,
            )
        )
        for lp in fund.lps:
            frac = lp.funded_capital_usd / total_funded_lp
            records.append(
                DistributionRecord(
                    party_id=lp.lp_id,
                    role="lp",
                    proceeds=lp_total * frac,
                )
            )
    else:
        loss = -period_pnl
        if fund.terms.kind is FundKind.MUDARABAH:
            # GP loses time only; LPs absorb pro-rata.
            records.append(
                DistributionRecord(
                    party_id=fund.terms.gp_id,
                    role="gp",
                    proceeds=0.0,
                )
            )
            for lp in fund.lps:
                frac = lp.funded_capital_usd / total_funded_lp
                records.append(
                    DistributionRecord(
                        party_id=lp.lp_id,
                        role="lp",
                        proceeds=-loss * frac,
                    )
                )
        else:  # MUSHARAKAH — pro-rata across all capital.
            total_cap = fund.total_capital_usd()
            gp_frac = fund.terms.gp_capital_usd / total_cap
            records.append(
                DistributionRecord(
                    party_id=fund.terms.gp_id,
                    role="gp",
                    proceeds=-loss * gp_frac,
                )
            )
            for lp in fund.lps:
                frac = lp.funded_capital_usd / total_cap
                records.append(
                    DistributionRecord(
                        party_id=lp.lp_id,
                        role="lp",
                        proceeds=-loss * frac,
                    )
                )
    return tuple(records)


def annual_management_fee(fund: Fund, *, aum_usd: float, days: int = 365) -> float:
    """Pro-rata Wakalah management fee.

    Pinned: simple-interest math (`pct × aum × days/365`), NOT compound."""
    if aum_usd < 0:
        raise ValueError("aum_usd must be non-negative")
    if days <= 0:
        raise ValueError("days must be positive")
    return aum_usd * fund.terms.base_management_fee_annual_pct * (days / 365.0)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_STATUS_EMOJI: dict[FundStatus, str] = {
    FundStatus.FORMING: "🌱",
    FundStatus.ACTIVE: "🟢",
    FundStatus.DISSOLVING: "🟡",
    FundStatus.WOUND_DOWN: "⚫",
}


def render_fund(fund: Fund) -> str:
    return (
        f"{_STATUS_EMOJI[fund.status]} {fund.terms.fund_id} "
        f"[{fund.terms.kind.value}/{fund.status.value}]: "
        f"GP {_mask(fund.terms.gp_id)} "
        f"(carry {fund.terms.gp_profit_share_pct * 100:.0f}%, "
        f"capital ${fund.terms.gp_capital_usd:,.0f}); "
        f"{len(fund.lps)} LPs, "
        f"funded ${fund.total_lp_funded_usd():,.0f}"
    )


def render_distribution(records: Iterable[DistributionRecord]) -> str:
    rs = tuple(records)
    if not rs:
        return "💸 No distribution."
    lines = [f"💸 Distribution ({len(rs)} parties):"]
    for r in rs:
        emoji = "🎯" if r.role == "gp" else "👥"
        lines.append(f"  {emoji} {_mask(r.party_id)} ({r.role}): {r.proceeds:+.2f}")
    return "\n".join(lines)
