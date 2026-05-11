"""Profit-Loss-Sharing equity strategy class — Round-5 Wave 7.D.

Conventional hedge-fund "2 + 20" performance fees are not halal — the
20% performance carry shares the upside but not the downside, so it
behaves like an asymmetric option (gharar). The halal analogue is a
**profit-loss-sharing** structure where:

1. Both manager and investor contribute capital to a common pool
   (Mudarabah/Musharakah-flavoured).
2. Profit above a hurdle rate is split per a pre-agreed ratio.
3. Losses are *also* shared in proportion to capital — the manager has
   skin in the game (this is the structural property that distinguishes
   PLS from a one-sided performance fee).
4. Hurdle rate is a high-water mark: prior peak NAV must be exceeded
   before any new performance fee accrues.

This module ships the **fee + P&L accounting primitives**. The upstream
strategy (signal generation, execution) is unchanged; this module sits
between portfolio P&L and the operator's NAV reporting.

Pinned semantics:

- **Closed-set FeeStructure**: HURDLE_ONLY, HURDLE_HWM (default),
  HURDLE_HWM_LOSS_SHARE.
- **High-water mark is monotone non-decreasing** — once a peak NAV is
  recorded, the HWM never falls. Underwater positions accrue zero
  performance fee until back above HWM.
- **Loss share is symmetric per capital ratio** — for HURDLE_HWM_LOSS_SHARE,
  if the manager's capital share is 10%, they absorb 10% of any
  drawdown below the prior peak. This is the AAOIFI Standard 13
  (Mudarabah) requirement: profit ratio is contractual, *loss ratio
  follows capital exposure*.
- **Fee accrual is per-period** — a quarterly NAV cycle records a
  closing fee; the fee is paid out at year-end after liquidity events.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — investor IDs masked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum


class FeeStructure(str, Enum):
    """Closed-set PLS fee structure."""

    HURDLE_ONLY = "hurdle_only"
    """Profit above hurdle is split; no HWM (each period is independent)."""
    HURDLE_HWM = "hurdle_hwm"
    """Profit above hurdle AND above HWM is split. (Default — most halal.)"""
    HURDLE_HWM_LOSS_SHARE = "hurdle_hwm_loss_share"
    """HWM + manager absorbs proportional loss below HWM."""


@dataclass(frozen=True)
class PLSAgreement:
    """The negotiated PLS terms."""

    agreement_id: str
    investor_id: str
    manager_id: str
    starting_capital: float
    """Total pool capital at inception."""
    manager_capital_pct: float
    """Manager's share of the starting capital. 0.0 = pure
    Mudarabah (manager contributes only labour); >0 = Musharakah-flavour."""
    hurdle_rate_annual: float
    """Annualised hurdle rate (e.g. 0.04 = 4% = "Murabaha bills"). Below
    this, the manager earns no performance fee."""
    profit_share_pct: float
    """Manager's share of profit *above* the hurdle. Common: 0.20 (20%)."""
    fee_structure: FeeStructure = FeeStructure.HURDLE_HWM
    base_management_fee_annual: float = 0.0
    """Optional Wakalah-style fixed fee paid quarterly. Halal because
    it's a fixed fee for service, not a performance carry. Default 0."""

    def __post_init__(self) -> None:
        if not self.agreement_id or not self.agreement_id.strip():
            raise ValueError("agreement_id must be non-empty")
        if not self.investor_id or not self.investor_id.strip():
            raise ValueError("investor_id must be non-empty")
        if not self.manager_id or not self.manager_id.strip():
            raise ValueError("manager_id must be non-empty")
        if self.investor_id == self.manager_id:
            raise ValueError("investor and manager must be distinct parties")
        if self.starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        if not 0.0 <= self.manager_capital_pct < 1.0:
            raise ValueError("manager_capital_pct must be in [0, 1)")
        if not -0.05 < self.hurdle_rate_annual < 0.30:
            raise ValueError("hurdle_rate_annual outside reasonable bounds")
        if not 0.0 <= self.profit_share_pct < 1.0:
            raise ValueError("profit_share_pct must be in [0, 1)")
        if not 0.0 <= self.base_management_fee_annual < 0.10:
            raise ValueError("base_management_fee_annual must be in [0, 0.10)")


@dataclass(frozen=True)
class PeriodReport:
    """Output of `compute_period_fee` for one accounting period."""

    period_end: date
    starting_nav: float
    ending_nav: float
    period_return_pct: float
    hurdle_pct_for_period: float
    hwm_at_start: float
    hwm_at_end: float
    base_fee: float
    performance_fee: float
    manager_loss_share: float
    """Loss absorbed by the manager (HURDLE_HWM_LOSS_SHARE only). 0
    otherwise. Positive number means the manager pays into the pool."""
    investor_net_return: float
    """Investor's net P&L after fees (positive = gain)."""

    def total_fee(self) -> float:
        return self.base_fee + self.performance_fee


def _annualise_to_period(annual_rate: float, days_in_period: int) -> float:
    """Convert an annual rate to the period rate using simple linear
    scaling (252 trading days). Pinned simple-interest semantics —
    for halal performance accounting, a clean operator-readable
    formula matters more than compounding accuracy."""
    return annual_rate * (days_in_period / 365.0)


def compute_period_fee(
    agreement: PLSAgreement,
    *,
    period_start: date,
    period_end: date,
    starting_nav: float,
    ending_nav: float,
    hwm_at_start: float,
) -> PeriodReport:
    """Compute base + performance fee for one period.

    Inputs are operator-provided NAVs at period bookends; this
    function does not maintain state.

    Logic:
    1. Period return = (ending - starting) / starting.
    2. Period hurdle = annual hurdle × (days / 365).
    3. Performance fee accrues only on the *excess* above hurdle.
    4. If fee_structure includes HWM, performance only accrues if
       ending > HWM.
    5. If fee_structure includes loss share, the manager pays into
       the pool when ending < HWM, in proportion to manager_capital_pct.
    """
    if starting_nav <= 0:
        raise ValueError("starting_nav must be positive")
    if ending_nav < 0:
        raise ValueError("ending_nav cannot be negative")
    if hwm_at_start <= 0:
        raise ValueError("hwm_at_start must be positive")
    if period_end <= period_start:
        raise ValueError("period_end must be after period_start")

    days = (period_end - period_start).days
    period_return = (ending_nav - starting_nav) / starting_nav
    hurdle = _annualise_to_period(agreement.hurdle_rate_annual, days)
    base = starting_nav * _annualise_to_period(agreement.base_management_fee_annual, days)
    perf = 0.0
    loss_share = 0.0
    new_hwm = hwm_at_start
    if agreement.fee_structure is FeeStructure.HURDLE_ONLY:
        if period_return > hurdle:
            excess = period_return - hurdle
            perf = excess * starting_nav * agreement.profit_share_pct
        if ending_nav > hwm_at_start:
            new_hwm = ending_nav
    elif agreement.fee_structure is FeeStructure.HURDLE_HWM:
        if ending_nav > hwm_at_start and period_return > hurdle:
            excess_dollar = ending_nav - hwm_at_start
            perf = excess_dollar * agreement.profit_share_pct
            new_hwm = ending_nav
    elif agreement.fee_structure is FeeStructure.HURDLE_HWM_LOSS_SHARE:
        if ending_nav > hwm_at_start and period_return > hurdle:
            excess_dollar = ending_nav - hwm_at_start
            perf = excess_dollar * agreement.profit_share_pct
            new_hwm = ending_nav
        elif ending_nav < hwm_at_start:
            drawdown_dollar = hwm_at_start - ending_nav
            loss_share = drawdown_dollar * agreement.manager_capital_pct
    investor_net = ending_nav - starting_nav - base - perf + loss_share
    return PeriodReport(
        period_end=period_end,
        starting_nav=starting_nav,
        ending_nav=ending_nav,
        period_return_pct=period_return,
        hurdle_pct_for_period=hurdle,
        hwm_at_start=hwm_at_start,
        hwm_at_end=new_hwm,
        base_fee=base,
        performance_fee=perf,
        manager_loss_share=loss_share,
        investor_net_return=investor_net,
    )


@dataclass(frozen=True)
class CumulativeReport:
    """Output of `run_full_history` — N periods rolled up."""

    period_reports: tuple[PeriodReport, ...]
    final_hwm: float
    total_base_fees: float
    total_performance_fees: float
    total_manager_loss_share: float
    total_investor_return: float


def run_full_history(
    agreement: PLSAgreement,
    bookends: Sequence[tuple[date, float]],
) -> CumulativeReport:
    """Walk a sequence of (period_end, NAV) bookends + roll up totals.

    `bookends[0]` is the inception bookend (its NAV must equal
    `agreement.starting_capital`). Each subsequent bookend is a period
    close. The first bookend's NAV seeds the HWM.
    """
    if len(bookends) < 2:
        raise ValueError("at least 2 bookends required (inception + 1 period)")
    inception_date, inception_nav = bookends[0]
    if abs(inception_nav - agreement.starting_capital) > 1e-6:
        raise ValueError("first bookend NAV must equal starting_capital")
    reports: list[PeriodReport] = []
    hwm = inception_nav
    last_date = inception_date
    last_nav = inception_nav
    total_base = 0.0
    total_perf = 0.0
    total_loss = 0.0
    total_inv = 0.0
    for end_date, end_nav in bookends[1:]:
        if end_date <= last_date:
            raise ValueError("bookend dates must be strictly increasing")
        rep = compute_period_fee(
            agreement,
            period_start=last_date,
            period_end=end_date,
            starting_nav=last_nav,
            ending_nav=end_nav,
            hwm_at_start=hwm,
        )
        reports.append(rep)
        hwm = rep.hwm_at_end
        last_date = end_date
        last_nav = end_nav
        total_base += rep.base_fee
        total_perf += rep.performance_fee
        total_loss += rep.manager_loss_share
        total_inv += rep.investor_net_return
    return CumulativeReport(
        period_reports=tuple(reports),
        final_hwm=hwm,
        total_base_fees=total_base,
        total_performance_fees=total_perf,
        total_manager_loss_share=total_loss,
        total_investor_return=total_inv,
    )


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


def render_period(report: PeriodReport) -> str:
    """Operator-readable summary of a single period."""
    return (
        f"📈 Period {report.period_end.isoformat()}: "
        f"NAV {report.starting_nav:.2f} → {report.ending_nav:.2f} "
        f"({report.period_return_pct * 100:+.2f}%)\n"
        f"  • Hurdle: {report.hurdle_pct_for_period * 100:.2f}%, "
        f"HWM: {report.hwm_at_start:.2f} → {report.hwm_at_end:.2f}\n"
        f"  • Base fee: {report.base_fee:.2f}, "
        f"perf fee: {report.performance_fee:.2f}, "
        f"manager loss-share: {report.manager_loss_share:.2f}\n"
        f"  • Investor net: {report.investor_net_return:+.2f}"
    )


def render_cumulative(report: CumulativeReport) -> str:
    """Operator-readable summary of a cumulative roll-up."""
    head = (
        f"📊 Cumulative ({len(report.period_reports)} periods): "
        f"final HWM {report.final_hwm:.2f}\n"
        f"  • Total base fees: {report.total_base_fees:.2f}\n"
        f"  • Total perf fees: {report.total_performance_fees:.2f}\n"
        f"  • Total manager loss-share: {report.total_manager_loss_share:.2f}\n"
        f"  • Total investor net: {report.total_investor_return:+.2f}"
    )
    return head
