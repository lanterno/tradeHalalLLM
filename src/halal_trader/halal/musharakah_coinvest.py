"""Musharakah co-investment rails — Round-5 Wave 6.C.

For deals larger than any single investor will fund, multiple
investors co-own one Musharakah pool. Each investor commits a capital
amount; the pool draws on commitments as the deal closes; profit is
split per capital share; loss is also borne per capital share (this
distinguishes Musharakah from Mudarabah — both parties have skin in
the game per AAOIFI Standard 12).

This module is the **commitment ladder + cap-table + drawdown
allocator + distribution computer**. Use it to:

1. Open a deal with a target raise + soft/hard caps.
2. Accept commitments from N investors (with FIFO ordering).
3. Close the round when target hit; reject over-cap commitments.
4. Track drawdowns + remaining un-called capital.
5. Distribute proceeds (profit OR loss) per capital share.

Pinned semantics:

- **Commitments are FIFO**. Earliest commitment wins on tie.
- **Hard cap is sticky**. Once raised >= hard_cap, no more commitments.
- **Capital share is computed from `funded_amount`** — committed but
  un-called capital does not earn or lose proceeds.
- **Loss-share = capital-share** per AAOIFI Standard 12.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — investor IDs masked.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum


class DealStatus(str, Enum):
    """Closed-set deal lifecycle ladder."""

    OPEN = "open"
    SOFT_CIRCLED = "soft_circled"
    """≥ soft_cap raised — operator can choose to close."""
    HARD_CLOSED = "hard_closed"
    LIQUIDATED = "liquidated"


@dataclass(frozen=True)
class Commitment:
    """An investor's commitment to a Musharakah pool."""

    commitment_id: str
    investor_id: str
    amount_usd: float
    committed_at: date
    funded_usd: float = 0.0

    def __post_init__(self) -> None:
        if not self.commitment_id or not self.commitment_id.strip():
            raise ValueError("commitment_id must be non-empty")
        if not self.investor_id or not self.investor_id.strip():
            raise ValueError("investor_id must be non-empty")
        if self.amount_usd <= 0:
            raise ValueError("amount_usd must be positive")
        if self.funded_usd < 0:
            raise ValueError("funded_usd must be non-negative")
        if self.funded_usd > self.amount_usd:
            raise ValueError("funded_usd cannot exceed committed amount")


@dataclass(frozen=True)
class CoInvestmentDeal:
    """A Musharakah co-investment pool."""

    deal_id: str
    sponsor_id: str
    soft_cap_usd: float
    hard_cap_usd: float
    target_raise_usd: float
    open_date: date
    close_date: date | None = None
    commitments: tuple[Commitment, ...] = ()
    status: DealStatus = DealStatus.OPEN
    realised_pnl: float = 0.0

    def __post_init__(self) -> None:
        if not self.deal_id or not self.deal_id.strip():
            raise ValueError("deal_id must be non-empty")
        if not self.sponsor_id or not self.sponsor_id.strip():
            raise ValueError("sponsor_id must be non-empty")
        if self.soft_cap_usd <= 0:
            raise ValueError("soft_cap_usd must be positive")
        if self.hard_cap_usd < self.soft_cap_usd:
            raise ValueError("hard_cap_usd must be ≥ soft_cap_usd")
        if not self.soft_cap_usd <= self.target_raise_usd <= self.hard_cap_usd:
            raise ValueError("target_raise_usd must be in [soft_cap, hard_cap]")
        if self.close_date is not None and self.close_date <= self.open_date:
            raise ValueError("close_date must be after open_date")
        # Commitment IDs unique.
        ids: set[str] = set()
        for c in self.commitments:
            if c.commitment_id in ids:
                raise ValueError(f"duplicate commitment_id {c.commitment_id}")
            ids.add(c.commitment_id)

    def total_committed(self) -> float:
        return sum(c.amount_usd for c in self.commitments)

    def total_funded(self) -> float:
        return sum(c.funded_usd for c in self.commitments)

    def uncalled_capital(self) -> float:
        return self.total_committed() - self.total_funded()

    def computed_status(self) -> DealStatus:
        """Recompute status from current commitments (excludes LIQUIDATED).

        Soft-circled iff total ≥ soft_cap; hard-closed iff total ≥ hard_cap.
        """
        if self.status is DealStatus.LIQUIDATED:
            return DealStatus.LIQUIDATED
        total = self.total_committed()
        if total >= self.hard_cap_usd:
            return DealStatus.HARD_CLOSED
        if total >= self.soft_cap_usd:
            return DealStatus.SOFT_CIRCLED
        return DealStatus.OPEN


def add_commitment(
    deal: CoInvestmentDeal,
    commitment: Commitment,
) -> CoInvestmentDeal:
    """Append a commitment to a deal — FIFO, with hard-cap rejection.

    Returns a NEW CoInvestmentDeal (the dataclass is frozen). Raises
    if the deal is HARD_CLOSED, LIQUIDATED, or the commitment would
    push past the hard cap.
    """
    if deal.status is DealStatus.HARD_CLOSED:
        raise ValueError("deal is HARD_CLOSED — no new commitments accepted")
    if deal.status is DealStatus.LIQUIDATED:
        raise ValueError("deal is LIQUIDATED — no new commitments accepted")
    if any(c.commitment_id == commitment.commitment_id for c in deal.commitments):
        raise ValueError(f"commitment_id {commitment.commitment_id} duplicated")
    new_total = deal.total_committed() + commitment.amount_usd
    if new_total > deal.hard_cap_usd + 1e-9:
        raise ValueError(
            f"commitment would push raise to {new_total:.2f} > hard_cap {deal.hard_cap_usd:.2f}"
        )
    new_commitments = (*deal.commitments, commitment)
    new_deal = replace(deal, commitments=new_commitments)
    return replace(new_deal, status=new_deal.computed_status())


def call_capital(
    deal: CoInvestmentDeal,
    *,
    amount_usd: float,
) -> CoInvestmentDeal:
    """Draw down `amount_usd` from un-called commitments, FIFO.

    Returns a new deal with updated `funded_usd` per commitment. Raises
    if the call exceeds total un-called capital.
    """
    if amount_usd <= 0:
        raise ValueError("amount_usd must be positive")
    if amount_usd > deal.uncalled_capital() + 1e-9:
        raise ValueError(f"call {amount_usd:.2f} exceeds uncalled {deal.uncalled_capital():.2f}")
    remaining = amount_usd
    new_commitments: list[Commitment] = []
    # Sort by committed_at ascending; FIFO call.
    ordered = sorted(deal.commitments, key=lambda c: c.committed_at)
    drawn: dict[str, float] = {}
    for c in ordered:
        avail = c.amount_usd - c.funded_usd
        if remaining <= 0 or avail <= 0:
            drawn[c.commitment_id] = c.funded_usd
            continue
        draw = min(avail, remaining)
        drawn[c.commitment_id] = c.funded_usd + draw
        remaining -= draw
    # Reconstruct in original order.
    for c in deal.commitments:
        new_funded = drawn.get(c.commitment_id, c.funded_usd)
        new_commitments.append(replace(c, funded_usd=new_funded))
    return replace(deal, commitments=tuple(new_commitments))


@dataclass(frozen=True)
class DistributionRecord:
    """Per-investor distribution slice."""

    investor_id: str
    capital_share: float
    """Investor's share of total funded capital. In [0, 1]."""
    proceeds: float
    """Signed proceeds (negative = loss)."""


def distribute(
    deal: CoInvestmentDeal,
    *,
    proceeds: float,
) -> tuple[DistributionRecord, ...]:
    """Distribute proceeds (profit or loss) per capital-share.

    Pin: AAOIFI Standard 12 — both profit and loss are shared in
    proportion to funded capital. No preferred returns; no asymmetric
    loss-share.
    """
    funded = deal.total_funded()
    if funded < 1e-9:
        raise ValueError("no funded capital — cannot distribute")
    by_investor: dict[str, float] = {}
    for c in deal.commitments:
        by_investor[c.investor_id] = by_investor.get(c.investor_id, 0.0) + c.funded_usd
    out: list[DistributionRecord] = []
    for investor_id, capital in by_investor.items():
        share = capital / funded
        out.append(
            DistributionRecord(
                investor_id=investor_id,
                capital_share=share,
                proceeds=proceeds * share,
            )
        )
    out.sort(key=lambda r: r.investor_id)
    return tuple(out)


def liquidate(
    deal: CoInvestmentDeal,
    *,
    realised_pnl: float,
    close_date: date,
) -> tuple[CoInvestmentDeal, tuple[DistributionRecord, ...]]:
    """Liquidate the pool: distribute realised_pnl + funded capital
    back to investors per their share.

    Returns the updated deal (status=LIQUIDATED) and the distribution
    records. The records' `proceeds` field is the *total* return
    (capital + P&L share), not just the P&L share.
    """
    funded = deal.total_funded()
    if funded < 1e-9:
        raise ValueError("no funded capital — cannot liquidate")
    by_investor: dict[str, float] = {}
    for c in deal.commitments:
        by_investor[c.investor_id] = by_investor.get(c.investor_id, 0.0) + c.funded_usd
    records: list[DistributionRecord] = []
    for investor_id, capital in by_investor.items():
        share = capital / funded
        total_back = capital + realised_pnl * share
        records.append(
            DistributionRecord(
                investor_id=investor_id,
                capital_share=share,
                proceeds=total_back,
            )
        )
    records.sort(key=lambda r: r.investor_id)
    new_deal = replace(
        deal,
        status=DealStatus.LIQUIDATED,
        close_date=close_date,
        realised_pnl=realised_pnl,
    )
    return new_deal, tuple(records)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


def render_deal(deal: CoInvestmentDeal) -> str:
    """Operator-readable summary."""
    head = (
        f"🏗️ Deal {deal.deal_id} ({deal.computed_status().value}): "
        f"committed ${deal.total_committed():,.0f} / funded "
        f"${deal.total_funded():,.0f} / target "
        f"${deal.target_raise_usd:,.0f}\n"
        f"  • Soft ${deal.soft_cap_usd:,.0f} | Hard ${deal.hard_cap_usd:,.0f} | "
        f"sponsor {_mask(deal.sponsor_id)}"
    )
    lines = [head]
    if deal.commitments:
        lines.append(f"  • {len(deal.commitments)} commitment(s):")
        for c in deal.commitments:
            lines.append(
                f"    - [{c.commitment_id}] {_mask(c.investor_id)}: "
                f"${c.amount_usd:,.0f} (funded ${c.funded_usd:,.0f})"
            )
    return "\n".join(lines)


def render_distribution(records: Iterable[DistributionRecord]) -> str:
    rs = tuple(records)
    if not rs:
        return "💸 No distribution records."
    lines = [f"💸 Distribution: {len(rs)} investors"]
    for r in rs:
        lines.append(
            f"  • {_mask(r.investor_id)}: share={r.capital_share * 100:.2f}%, "
            f"proceeds=${r.proceeds:+,.2f}"
        )
    return "\n".join(lines)
